"""SB3 callbacks of D2 training (spec §4 step 5, §6, §7).

Order in `learn(callback=[...])`: PhaseCallback, CollapseGuard, TrainStatsCallback,
DrivingEvalCallback, PlateauStop (R1 §2.6), CheckpointCallback, WallClockLimit. SB3 calls `on_step` after every vector
step (num_timesteps already advanced by n_envs) and before the transition is stored; returning
False ends `learn()` (SB3 still calls `on_training_end`).
`on_event(kind, **data)` hooks let train.py record events in status.json.
"""
import copy
import math
import os
import time
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from ..runs import append_jsonl
from .env import FAILURES
from .vec import base_venv


def _noop(*_, **__):
    pass


def save_model_atomic(model, path):
    """model.save to `<path>.tmp`, then rename over `path`: a hard kill mid-write leaves the
    previous file intact (SB3 keeps the '.tmp' suffix, it only appends '.zip' to suffix-less paths)."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    model.save(tmp)
    os.replace(tmp, path)


def save_replay_buffer_atomic(model, path):
    """model.save_replay_buffer via `<path>.tmp` + rename (see save_model_atomic)."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    model.save_replay_buffer(tmp)
    os.replace(tmp, path)


class ReplayBufferSaver:
    """`saver(model)`: save_replay_buffer_atomic to `path` unless already saved at this
    num_timesteps (the eval and checkpoint callbacks can both fire at one step), then
    `on_saved(step)`. Returns True if it saved. `last_step`: step of an earlier save (e.g. the
    prefill save at step 0)."""

    def __init__(self, path, on_saved=_noop, last_step=None):
        self.path, self.on_saved, self.last_step = Path(path), on_saved, last_step

    def __call__(self, model):
        n = int(model.num_timesteps)
        if n == self.last_step:
            return False
        save_replay_buffer_atomic(model, self.path)
        self.last_step = n
        self.on_saved(n)
        return True


def cut_nstep_chains(rb):
    """Resume cut (QA D2 m1): make the last stored row (pos - 1) % size of every env column an
    episode end, so SB3's n-step sums never chain from the loaded transitions into the
    episodes the resumed envs start. A row that is not done becomes done = timeout = 1 (time
    limit: the target bootstraps from that row's own next_obs, as SB3 does temporarily for
    pos - 1 at sampling); a true termination (done = 1, timeout = 0) stays terminal. No-op on an
    empty buffer. Returns the row index or None."""
    if rb.size() == 0:
        return None
    i = (rb.pos - 1) % rb.buffer_size
    open_ = rb.dones[i] == 0
    rb.timeouts[i] = np.logical_or(rb.timeouts[i] != 0, open_)
    rb.dones[i] = 1.
    return i


def load_replay_buffer_for_resume(model, path):
    """model.load_replay_buffer(path) + cut_nstep_chains. Returns the cut row (or None)."""
    model.load_replay_buffer(path)
    return cut_nstep_chains(model.replay_buffer)


class PhaseCallback(BaseCallback):
    """Step 5: at num_timesteps >= unfreeze_at, unfreeze actor and alpha (actor Adam is fresh:
    it never stepped while frozen; alpha continues from its initial value). The 'unfreeze' event
    carries sha256 digests of the actor and log alpha as they leave the frozen phase (R1 A4)."""

    def __init__(self, unfreeze_at, on_event=_noop):
        super().__init__()
        self.unfreeze_at = int(unfreeze_at)
        self.on_event = on_event
        self.unfrozen_at = None

    def _check(self):
        m = self.model
        if (m.freeze_actor or m.freeze_ent_coef) and m.num_timesteps >= self.unfreeze_at:
            from .agent import param_digest
            digests = {'actor_sha256': param_digest(m.actor),
                       'log_ent_coef_sha256': None if m.log_ent_coef is None else param_digest(m.log_ent_coef)}
            m.set_frozen(False, False)
            self.unfrozen_at = m.num_timesteps
            self.logger.record('phase/unfrozen_at', m.num_timesteps)
            self.on_event('unfreeze', step=m.num_timesteps, **digests)

    def _on_training_start(self):
        self._check()

    def _on_step(self):
        self._check()
        return True


def snapshot_state(model):
    """Deep copies of everything a rollback restores (never the replay buffer)."""
    return {'policy': copy.deepcopy(model.policy.state_dict()),
            'actor_opt': copy.deepcopy(model.actor.optimizer.state_dict()),
            'critic_opt': copy.deepcopy(model.critic.optimizer.state_dict()),
            'log_ent_coef': None if model.log_ent_coef is None else model.log_ent_coef.detach().clone(),
            'ent_opt': None if model.ent_coef_optimizer is None
            else copy.deepcopy(model.ent_coef_optimizer.state_dict())}


def restore_state(model, snap):
    model.policy.load_state_dict(snap['policy'])
    model.actor.optimizer.load_state_dict(copy.deepcopy(snap['actor_opt']))
    model.critic.optimizer.load_state_dict(copy.deepcopy(snap['critic_opt']))
    if snap['log_ent_coef'] is not None:
        with th.no_grad():
            model.log_ent_coef.copy_(snap['log_ent_coef'])
    if snap['ent_opt'] is not None:
        model.ent_coef_optimizer.load_state_dict(copy.deepcopy(snap['ent_opt']))


class CollapseGuard(BaseCallback):
    """From the unfreeze step to unfreeze + window: every `interval` steps run `probe_fn(model)`
    (mean probe progress [m]). >= threshold x baseline: keep an in-memory "last good" snapshot.
    Below: restore the snapshot and set model.actor_lr = rollback_actor_lr (rollback); after
    max_rollbacks rollbacks the next collapse stops training (event 'collapsed'). The snapshot
    taken at the unfreeze step (before any actor update) is the first "last good" state.
    After the window: nothing (the eval callback logs clean -> failed regressions).
    `last_path`: after a rollback the restored model (with the rollback actor_lr) is saved there
    atomically BEFORE the 'rollback' event is recorded (QA D2 m3), so a crash right after a
    rollback cannot resume the collapsed weights with the rollback already counted (the
    remaining window: a crash between the save and the event allows one extra rollback)."""

    def __init__(self, probe_fn, baseline, cfg, *, start_step=None, rollbacks=0, on_event=_noop, log_path=None,
                 last_path=None):
        super().__init__()
        self.probe_fn, self.baseline, self.cfg = probe_fn, float(baseline), cfg
        self.last_path = None if last_path is None else Path(last_path)
        self.start_step = start_step
        self.rollbacks = int(rollbacks)
        self.on_event = on_event
        self.log_path = log_path
        self.snapshot = None
        self.next_probe = None
        self.collapsed = False
        self.history = []

    def _arm(self, step):
        if self.start_step is None:
            self.start_step = step
        self.snapshot = snapshot_state(self.model)
        k = max(step - self.start_step, 0) // self.cfg.interval + 1
        self.next_probe = self.start_step + k * self.cfg.interval

    def _on_training_start(self):
        if not self.model.freeze_actor:
            self._arm(self.model.num_timesteps)

    def _on_step(self):
        n = self.model.num_timesteps
        if self.snapshot is None:
            if not self.model.freeze_actor:
                self._arm(n)
            return True
        if n >= self.start_step + self.cfg.window or n < self.next_probe:
            return True
        while self.next_probe <= n:
            self.next_probe += self.cfg.interval
        progress = float(self.probe_fn(self.model))
        ok = progress >= self.cfg.threshold * self.baseline
        rec = {'step': n, 'mean_progress_m': progress, 'baseline_m': self.baseline, 'ok': ok}
        self.history.append(rec)
        self.logger.record('guard/probe_progress_m', progress)
        if self.log_path is not None:
            append_jsonl(self.log_path, rec)
        if ok:
            self.snapshot = snapshot_state(self.model)
            return True
        if self.rollbacks >= self.cfg.max_rollbacks:
            self.collapsed = True
            self.on_event('collapsed', step=n, progress_m=progress, baseline_m=self.baseline)
            return False
        restore_state(self.model, self.snapshot)
        self.model.actor_lr = self.cfg.rollback_actor_lr
        self.rollbacks += 1
        if self.last_path is not None:
            save_model_atomic(self.model, self.last_path)
        self.on_event('rollback', step=n, progress_m=progress, baseline_m=self.baseline,
                      actor_lr=self.cfg.rollback_actor_lr, rollbacks=self.rollbacks)
        return True


class TrainStatsCallback(BaseCallback):
    """Per `window` steps one line in `path` (train_stats.jsonl): termination / truncation counts;
    mean speed and |e| at failure and the share of failures with v < slow_mps [m/s] or
    |e| > wide_m [m]; stalls among episodes that started (v0) below low_v0_mps [m/s]; episode
    return and length (VecMonitor) and the mean episode shaping (info 'episode_shaping'; in
    time_pbrs the VecMonitor return includes it, R1 §6 #5). Thresholds: EvalConfig.stats_*
    (train.py passes them). `totals` keeps cumulative cause counts for the eval records (cause =
    termination, else truncation: a time_pbrs stall counts as 'stall' in both modes)."""

    def __init__(self, path, window, *, slow_mps=15., wide_m=3., low_v0_mps=5.):
        super().__init__()
        self.path, self.window = path, int(window)
        self.slow_mps, self.wide_m, self.low_v0_mps = float(slow_mps), float(wide_m), float(low_v0_mps)
        self.totals = {}
        self.v0 = None
        self._reset_window()

    def _reset_window(self):
        self.w = {'episodes': 0, 'causes': {}, 'fail_speed': [], 'fail_e': [], 'low_v0': 0,
                  'low_v0_stall': 0, 'returns': [], 'lengths': [], 'shaping': []}

    def _v0(self, i):
        info = base_venv(self.training_env).reset_infos[i] or {}
        return info.get('start', {}).get('v0')

    def _on_training_start(self):
        n = self.training_env.num_envs
        self.v0 = [self._v0(i) for i in range(n)]
        self.next_dump = (self.model.num_timesteps // self.window + 1) * self.window

    def _on_step(self):
        infos, dones = self.locals['infos'], self.locals['dones']
        w = self.w
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i]
            cause = info.get('termination') or info.get('truncation') or 'unknown'
            w['episodes'] += 1
            w['causes'][cause] = w['causes'].get(cause, 0) + 1
            self.totals[cause] = self.totals.get(cause, 0) + 1
            if cause in FAILURES:
                w['fail_speed'].append(info.get('speed', math.nan))
                w['fail_e'].append(abs(info.get('e_ref', math.nan)))
            v0 = self.v0[i]
            if v0 is not None and v0 < self.low_v0_mps:
                w['low_v0'] += 1
                w['low_v0_stall'] += cause == 'stall'
            if 'episode_shaping' in info:
                w['shaping'].append(info['episode_shaping'])
            if 'episode' in info:
                w['returns'].append(info['episode']['r'])
                w['lengths'].append(info['episode']['l'])
            self.v0[i] = self._v0(i)
        if self.model.num_timesteps >= self.next_dump:
            self.dump()
            while self.next_dump <= self.model.num_timesteps:
                self.next_dump += self.window
        return True

    def dump(self):
        w = self.w
        sp, e = np.asarray(w['fail_speed'], float), np.asarray(w['fail_e'], float)
        nf = len(sp)
        rec = {'step': self.model.num_timesteps, 'episodes': w['episodes'], 'causes': w['causes'],
               'failures': nf,
               'fail_speed_mean': float(np.nanmean(sp)) if nf else None,
               'fail_abs_e_mean': float(np.nanmean(e)) if nf else None,
               'fail_share_slow_or_wide': float(np.mean((sp < self.slow_mps) | (e > self.wide_m))) if nf else None,
               'low_v0_episodes': w['low_v0'], 'low_v0_stalls': w['low_v0_stall'],
               'ep_return_mean': float(np.mean(w['returns'])) if w['returns'] else None,
               'ep_length_mean': float(np.mean(w['lengths'])) if w['lengths'] else None,
               'ep_shaping_mean': float(np.mean(w['shaping'])) if w['shaping'] else None}
        append_jsonl(self.path, rec)
        self._reset_window()
        return rec

    def _on_training_end(self):
        if self.w['episodes']:
            self.dump()


class DrivingEvalCallback(BaseCallback):
    """Evaluation at num_timesteps = 0 (baseline, at the start of learn()), every `interval`
    steps and at the end. `eval_fn(model, step) -> dict` (evaluate.evaluate_model); the callback
    adds throughput (env steps/s and grad steps/s over training wall time, excluding evaluation),
    cumulative train / eval wall time, the training obs-clip fraction and termination counts,
    appends one JSON line to `evals_path`, saves the replay buffer (`buffer_saver`, if given;
    QA D2 m4: last.zip and the buffer stay paired after a hard kill), then last.zip, and best.zip
    on a new best CLEAN deterministic lap 2. A clean -> failed deterministic regression is an event.
    Critic check (R1 §2.6): at the first evaluation with step >= `critic_check_at` (None: never)
    the record gets critic_check = |Q_min - MC^n| / |MC^n| (q_vs_mc, n-step-consistent soft MC;
    per-step MC if the n-step values are absent) and critic_check_tol; above the tol (or not
    computable) -> event 'critic_not_converged' (training continues). `records` keeps every record
    of this process (PlateauStop reads it)."""

    def __init__(self, eval_fn, interval, evals_path, ckpt_dir, *, stats=None, best_lap2=math.inf,
                 last_det_clean=None, on_event=_noop, train_wall0=0., eval_wall0=0., buffer_saver=None,
                 critic_check_at=None, critic_check_tol=None, critic_check_done=False):
        super().__init__()
        self.critic_check_at, self.critic_check_tol = critic_check_at, critic_check_tol
        self.critic_check_done = bool(critic_check_done)
        self.records = []
        self.buffer_saver = buffer_saver
        self.eval_fn, self.interval = eval_fn, int(interval)
        self.evals_path, self.ckpt_dir = Path(evals_path), Path(ckpt_dir)
        self.stats = stats
        self.best_lap2 = best_lap2
        self.last_det_clean = last_det_clean
        self.on_event = on_event
        self.train_wall, self.eval_wall = float(train_wall0), float(eval_wall0)   # carried over on resume
        self.last_step = None

    def _on_training_start(self):
        self.t_mark = time.perf_counter()
        self.mark_steps, self.mark_updates = self.model.num_timesteps, self.model._n_updates
        n = self.model.num_timesteps
        self.next_eval = (n // self.interval + 1) * self.interval
        if n == 0:
            self.evaluate()

    def _on_step(self):
        if self.model.num_timesteps >= self.next_eval:
            self.evaluate()
            while self.next_eval <= self.model.num_timesteps:
                self.next_eval += self.interval
        return True

    def _on_training_end(self):
        if self.last_step != self.model.num_timesteps:
            self.evaluate()

    def evaluate(self):
        m = self.model
        n = m.num_timesteps
        t0 = time.perf_counter()
        seg = t0 - self.t_mark                      # training wall time since the last eval ended
        self.train_wall += seg
        steps, updates = n - self.mark_steps, m._n_updates - self.mark_updates
        rec = {'step': n, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        rec.update(self.eval_fn(m, n))
        self.eval_wall += time.perf_counter() - t0
        rec.update(env_steps_per_s=steps / seg if seg > 0 and steps else None,
                   grad_steps_per_s=updates / seg if seg > 0 and updates else None,
                   wall_train_s=self.train_wall, wall_eval_s=self.eval_wall,
                   freeze_actor=m.freeze_actor, actor_lr=m.actor_lr)
        try:
            clipped = sum(self.training_env.get_attr('obs_clipped_total'))
            values = sum(self.training_env.get_attr('obs_values_total'))
            rec['train_obs_clip_fraction'] = clipped / values if values else None
        except AttributeError:
            rec['train_obs_clip_fraction'] = None
        if self.stats is not None:
            rec['train_causes_total'] = dict(self.stats.totals)
        if self.critic_check_at is not None and not self.critic_check_done and n >= self.critic_check_at:
            self.critic_check_done = True
            value = critic_check_value(rec.get('q_vs_mc'))
            rec.update(critic_check=value, critic_check_tol=self.critic_check_tol)
            if value is None or value > self.critic_check_tol:
                self.on_event('critic_not_converged', step=n, critic_check=value, tol=self.critic_check_tol)
        clean, lap2 = rec.get('det_clean'), rec.get('lap2_s')
        if self.last_det_clean and clean is False:
            self.on_event('det_regression', step=n, termination=rec.get('det_termination'))
        self.last_det_clean = clean
        if self.buffer_saver is not None:
            self.buffer_saver(m)
        save_model_atomic(m, self.ckpt_dir / 'last.zip')
        if clean and lap2 is not None and lap2 < self.best_lap2:
            self.best_lap2 = lap2
            save_model_atomic(m, self.ckpt_dir / 'best.zip')
            rec['new_best'] = True
        append_jsonl(self.evals_path, rec)
        self.records.append(rec)
        for k in ('lap2_s', 'sto_clean_rate', 'alpha', 'policy_entropy', 'critic_check'):
            if isinstance(rec.get(k), (int, float)):
                self.logger.record(f'eval/{k}', rec[k])
        self.last_step = n
        self.t_mark, self.mark_steps, self.mark_updates = time.perf_counter(), n, m._n_updates
        return rec


def critic_check_value(q):
    """|q_min_minus_mc_nstep| / |mc_nstep_mean| of a summarize_q_mc dict (per-step keys if the
    n-step ones are absent); None if not computable (no pairs or MC mean 0)."""
    q = q or {}
    num, den = q.get('q_min_minus_mc_nstep'), q.get('mc_nstep_mean')
    if num is None or den is None:
        num, den = q.get('q_min_minus_mc'), q.get('mc_mean')
    if num is None or not den:
        return None
    return abs(float(num)) / abs(float(den))


def plateau_entry(rec):
    """(step, clean deterministic lap 2 or None, stochastic mean lap or None) of an eval record."""
    return (int(rec['step']), rec.get('lap2_s') if rec.get('det_clean') else None, rec.get('sto_lap_mean_s'))


def plateau_reached(history, step, cfg):
    """R1 §5 stop rule on `history` = [plateau_entry(...)]: True iff step >= cfg.min_steps and
    neither best (clean det lap 2, stochastic mean lap) improved by >= cfg.min_improvement_s over
    the last cfg.window steps, i.e. for both: best over evals with step' <= step - window minus
    best over evals with step' <= step < min_improvement_s (1e-9 s slack for float sums). A best
    with no value before the window but one inside it counts as improved; a best with no value
    at all counts as not improved."""
    if step < cfg.min_steps:
        return False
    cut = step - cfg.window

    def best(i, upto):
        vals = [h[i] for h in history if h[0] <= upto and h[i] is not None]
        return min(vals) if vals else math.inf
    for i in (1, 2):
        before, now = best(i, cut), best(i, step)
        if now < math.inf and before - now >= cfg.min_improvement_s - 1e-9:
            return False
    return True


class PlateauStop(BaseCallback):
    """Stops training (returns False, event 'plateau') after an evaluation at which
    `plateau_reached` holds (StopConfig; R1 §2.6, §5); nothing if cfg.plateau is False. Reads new
    records of the DrivingEvalCallback `evalcb` (placed before it in the callback list);
    `history`: plateau entries of earlier processes (resume: from evals.jsonl)."""

    def __init__(self, cfg, evalcb, *, history=(), on_event=_noop):
        super().__init__()
        self.cfg, self.evalcb, self.on_event = cfg, evalcb, on_event
        self.history = list(history)
        self._seen = 0
        self.triggered = False

    def _on_step(self):
        new = self.evalcb.records[self._seen:]
        self._seen = len(self.evalcb.records)
        if not new:
            return True
        self.history += [plateau_entry(r) for r in new]
        if not self.cfg.plateau:
            return True
        n = self.history[-1][0]
        if plateau_reached(self.history, n, self.cfg):
            self.triggered = True
            self.on_event('plateau', step=n, window=self.cfg.window, min_improvement_s=self.cfg.min_improvement_s)
            return False
        return True


class CheckpointCallback(BaseCallback):
    """checkpoints/step_<n>.zip every `interval` steps (and last.zip alongside, atomically), and
    the replay buffer to `buffer_path` every `buffer_every` steps (None / 0: never here),
    atomically, so a hard-killed run can resume (TrainConfig.replay_buffer_save_every).
    The buffer saved at step n lacks that vector step's n_envs transitions (SB3 stores them after
    the callbacks). `on_buffer_saved(step)` lets train.py record the step in status.json.
    `buffer_saver` (a ReplayBufferSaver shared with the eval callback) replaces buffer_path /
    on_buffer_saved, so a step with both an eval and a checkpoint saves the buffer once."""

    def __init__(self, interval, ckpt_dir, *, buffer_path=None, buffer_every=None, on_buffer_saved=_noop,
                 buffer_saver=None):
        super().__init__()
        self.interval, self.ckpt_dir = int(interval), Path(ckpt_dir)
        if buffer_saver is None and buffer_path is not None:
            buffer_saver = ReplayBufferSaver(buffer_path, on_buffer_saved)
        self.buffer_saver = buffer_saver
        self.buffer_every = int(buffer_every or 0) if self.buffer_saver is not None else 0

    @staticmethod
    def _next(n, every):
        return (n // every + 1) * every

    def _on_training_start(self):
        n = self.model.num_timesteps
        self.next = self._next(n, self.interval)
        self.next_buffer = self._next(n, self.buffer_every) if self.buffer_every > 0 else math.inf

    def _on_step(self):
        n = self.model.num_timesteps
        if n >= self.next:
            self.model.save(self.ckpt_dir / f'step_{n}.zip')
            save_model_atomic(self.model, self.ckpt_dir / 'last.zip')
            self.next = self._next(n, self.interval)
        if n >= self.next_buffer:
            self.buffer_saver(self.model)
            self.next_buffer = self._next(n, self.buffer_every)
        return True


class WallClockLimit(BaseCallback):
    """Returns False once `max_wall_h` hours have passed since `t0` (time.time(); default: the
    start of learn()). The graceful end (final eval, saves) follows in on_training_end / train.py."""

    def __init__(self, max_wall_h, t0=None, on_event=_noop):
        super().__init__()
        self.max_s = None if max_wall_h is None else float(max_wall_h) * 3600.
        self.t0 = t0
        self.triggered = False
        self.on_event = on_event

    def _on_training_start(self):
        if self.t0 is None:
            self.t0 = time.time()

    def _on_step(self):
        if self.max_s is not None and time.time() - self.t0 > self.max_s:
            self.triggered = True
            self.on_event('wall_clock_limit', step=self.model.num_timesteps)
            return False
        return True
