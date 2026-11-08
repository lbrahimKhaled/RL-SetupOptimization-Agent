"""SB3 callbacks of D2 training (spec §4 step 5, §6, §7).

Order in `learn(callback=[...])`: [BonusScheduleCallback (R2, first)], PhaseCallback,
CollapseGuard, TrainStatsCallback, [FocusedStartCallback (R2)], DrivingEvalCallback,
PlateauStop (R1 §2.6), CheckpointCallback, WallClockLimit. SB3 calls `on_step` after every vector
step (num_timesteps already advanced by n_envs) and before the transition is stored and before
train(); returning False ends `learn()` (SB3 still calls every callback of that step and then
`on_training_end`).
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
from ..runs import append_jsonl, write_json
from .env import FAILURES, TERMINAL_FAILURES
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
    termination, else truncation: a time_pbrs stall counts as 'stall' in both modes).
    R2-8.4 (`groups=True`): the record adds 'groups' {normal, low_noise (env index in
    low_noise_envs), focused, uniform (focused flag of the episode's reset)}: episodes, failures,
    failure_rate, crash_bins {bin start m: terminal failures} (bins of `bin_m` on Track.path) and
    bonus_per_episode (info 'episode_corner_bonus'); `eval_groups()` returns the same since its
    previous call (eval records)."""

    GROUPS = ('normal', 'low_noise', 'focused', 'uniform')

    def __init__(self, path, window, *, slow_mps=15., wide_m=3., low_v0_mps=5., groups=False, low_noise_envs=(),
                 bin_m=50.):
        super().__init__()
        self.path, self.window = path, int(window)
        self.slow_mps, self.wide_m, self.low_v0_mps = float(slow_mps), float(wide_m), float(low_v0_mps)
        self.groups, self.low_noise_envs, self.bin_m = bool(groups), tuple(low_noise_envs), float(bin_m)
        self.totals = {}
        self.v0 = None
        self.focused = None
        self._eval_groups = self._empty_groups()
        self._reset_window()

    @classmethod
    def _empty_groups(cls):
        return {g: {'episodes': 0, 'failures': 0, 'crash_bins': {}, 'bonus': []} for g in cls.GROUPS}

    @staticmethod
    def _summarize_groups(groups):
        out = {}
        for g, v in groups.items():
            out[g] = {'episodes': v['episodes'], 'failures': v['failures'],
                      'failure_rate': v['failures'] / v['episodes'] if v['episodes'] else None,
                      'crash_bins': dict(sorted(v['crash_bins'].items())),
                      'bonus_per_episode': float(np.mean(v['bonus'])) if v['bonus'] else None}
        return out

    def eval_groups(self):
        """Group statistics since the previous call (None without groups)."""
        if not self.groups:
            return None
        out, self._eval_groups = self._summarize_groups(self._eval_groups), self._empty_groups()
        return out

    def _focused(self, i):
        info = base_venv(self.training_env).reset_infos[i] or {}
        return bool(info.get('start', {}).get('focused', False))

    def _add_group(self, i, info, cause):
        names = ('low_noise' if i in self.low_noise_envs else 'normal', 'focused' if self.focused[i] else 'uniform')
        crash = cause in TERMINAL_FAILURES
        for target in (self.w['groups'], self._eval_groups):
            for g in names:
                v = target[g]
                v['episodes'] += 1
                v['failures'] += cause in FAILURES
                if crash and info.get('s_track') is not None:
                    b = f"{int(float(info['s_track']) // self.bin_m * self.bin_m)}"
                    v['crash_bins'][b] = v['crash_bins'].get(b, 0) + 1
                if 'episode_corner_bonus' in info:
                    v['bonus'].append(float(info['episode_corner_bonus']))

    def _reset_window(self):
        self.w = {'episodes': 0, 'causes': {}, 'fail_speed': [], 'fail_e': [], 'low_v0': 0,
                  'low_v0_stall': 0, 'returns': [], 'lengths': [], 'shaping': []}
        if self.groups:
            self.w['groups'] = self._empty_groups()

    def _v0(self, i):
        info = base_venv(self.training_env).reset_infos[i] or {}
        return info.get('start', {}).get('v0')

    def _on_training_start(self):
        n = self.training_env.num_envs
        self.v0 = [self._v0(i) for i in range(n)]
        if self.groups:
            self.focused = [self._focused(i) for i in range(n)]
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
            if self.groups:
                self._add_group(i, info, cause)
                self.focused[i] = self._focused(i)
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
        if self.groups:
            rec['groups'] = self._summarize_groups(w['groups'])
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
    Critic check (R1 §2.6; QA M1): an extra evaluation is scheduled at `critic_check_at` (None:
    never) if it is not on the interval grid, so the check sees the critic at the unfreeze (the
    PhaseCallback runs first in the same step, the actor has not moved yet); the first evaluation
    with step >= critic_check_at gets critic_check = |Q_min - MC^n| / |MC^n| (q_vs_mc, n-step-consistent soft MC;
    per-step MC if the n-step values are absent) and critic_check_tol; above the tol (or not
    computable / non-finite) -> event 'critic_not_converged' (training continues). `records` keeps every record
    of this process (PlateauStop reads it).
    R2: `extras` = callables (model, step) -> dict merged into the record (focus / low-noise /
    buffer statistics; they may persist their own state). `schedule` (BonusScheduleCallback):
    the record gets bonus_w, report_from, reportable (and the Amendment-1 trigger is checked);
    best.zip is then written only from best-eligible evaluations (`best_eligible`: reportable, or
    after an Amendment-1 trigger an evaluation of the bonus-free result stretch, Amendment 2) and a
    better clean lap 2 of any other one goes to best_aided.zip (never reported; also written at the
    trigger). Without a schedule every evaluation is reportable (pre-R2 behaviour)."""

    def __init__(self, eval_fn, interval, evals_path, ckpt_dir, *, stats=None, best_lap2=math.inf,
                 last_det_clean=None, on_event=_noop, train_wall0=0., eval_wall0=0., buffer_saver=None,
                 critic_check_at=None, critic_check_tol=None, critic_check_done=False, extras=(), schedule=None,
                 best_aided_lap2=math.inf):
        super().__init__()
        self.extras, self.schedule, self.best_aided_lap2 = tuple(extras), schedule, best_aided_lap2
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
        self._schedule(n)
        if n == 0:
            self.evaluate()

    def _schedule(self, n):
        """next_eval = the next interval multiple after step n, or critic_check_at if earlier and
        still pending (QA M1)."""
        self.next_eval = (n // self.interval + 1) * self.interval
        c = self.critic_check_at
        if c is not None and not self.critic_check_done and c > n:
            self.next_eval = min(self.next_eval, c)

    def _on_step(self):
        if self.model.num_timesteps >= self.next_eval:
            self.evaluate()
            self._schedule(self.model.num_timesteps)
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
        for extra in self.extras:
            rec.update(extra(m, n))
        if self.schedule is not None:
            self.schedule.on_eval(rec)
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
        eligible = best_eligible(rec)
        if clean and lap2 is not None and eligible and lap2 < self.best_lap2:
            self.best_lap2 = lap2
            save_model_atomic(m, self.ckpt_dir / 'best.zip')
            rec['new_best'] = True
        if clean and lap2 is not None and ((not eligible and lap2 < self.best_aided_lap2) or rec.get('aided_target_lap')):
            self.best_aided_lap2 = min(lap2, self.best_aided_lap2)
            save_model_atomic(m, self.ckpt_dir / 'best_aided.zip')
            rec['new_best_aided'] = True
        append_jsonl(self.evals_path, rec)
        self.records.append(rec)
        for k in ('lap2_s', 'sto_clean_rate', 'alpha', 'policy_entropy', 'critic_check', 'bonus_w'):
            if isinstance(rec.get(k), (int, float)):
                self.logger.record(f'eval/{k}', rec[k])
        self.last_step = n
        self.t_mark, self.mark_steps, self.mark_updates = time.perf_counter(), n, m._n_updates
        return rec


def critic_check_value(q):
    """|q_min_minus_mc_nstep| / |mc_nstep_mean| of a summarize_q_mc dict (per-step keys if the
    n-step ones are absent); None if not computable (no pairs, MC mean 0, non-finite: QA m1), which
    the eval callback treats as not converged."""
    q = q or {}
    num, den = q.get('q_min_minus_mc_nstep'), q.get('mc_nstep_mean')
    if num is None or den is None:
        num, den = q.get('q_min_minus_mc'), q.get('mc_mean')
    if num is None or not den or not (math.isfinite(num) and math.isfinite(den)):
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


# --- R2 training aids (docs/specs/R2_review.md §R2-5, R2-7, Amendment 1) -------------------------
def bonus_weight(cfg, n):
    """Scheduled corner-bonus weight w(n) of BonusScheduleConfig `cfg` at this run's step n:
    1 before hold_until, linear 1 -> 0 on [hold_until, fade_until], 0 from fade_until on;
    1 at every step for a never-fade schedule (hold_until None, R3)."""
    if cfg.hold_until is None or n < cfg.hold_until:
        return 1.
    if n >= cfg.fade_until:
        return 0.
    return 1. - (n - cfg.hold_until) / (cfg.fade_until - cfg.hold_until)


class BonusScheduleCallback(BaseCallback):
    """First callback of an R2 run. Sets `model.replay_buffer.bonus_weight` (and the audit mirror
    model.bonus_weight) to w(num_timesteps) at the start of learn() and at every on_step (SB3
    calls on_step before train()), so every gradient step uses the current weight.
    Amendment 1: `on_eval(rec)` (called by DrivingEvalCallback with each record) adds bonus_w
    (weight in effect up to this evaluation), report_from and reportable (step >= report_from);
    the first evaluation with a clean deterministic lap 2 <= cfg.trigger_lap2_s while w > 0 or
    step < report_from is the trigger: w = 0 for the rest of the run (no fade), report_from =
    trigger step + cfg.bonus_free_before_report, record flag aided_target_lap, event
    'aided_target_lap'. After a trigger the run stops itself (returns False, event
    'report_complete') at the first step >= report_from. State {trigger_step, report_from} goes to
    `state_path` (bonus_state.json, atomic) at the trigger and every evaluation, and to the model
    (bonus_trigger_step / bonus_report_from, pickled in the checkpoints); a resume passes it back.
    Amendment 2: after a trigger, the evaluations of the bonus-free stretch (`result_eligible`:
    trigger_step <= step <= report_from and bonus_w == 0) get result_eval = True (best.zip comes
    from them) and the run's `result()` = best clean deterministic lap 2 + every clean
    deterministic lap 2 (step, lap) of the stretch goes to bonus_state.json 'result' and to
    `on_result(result)` (train.py: status.json 'result') at every evaluation. `history`: evaluation
    records of earlier processes (resume), from which the stretch's clean laps are rebuilt.
    R3 never-fade schedule (cfg.hold_until None, no trigger_step): w = 1 always, report_from None,
    no evaluation reportable, the trigger is never checked (early trigger off), the run never
    stops itself."""

    def __init__(self, cfg, state_path=None, *, trigger_step=None, report_from=None, on_event=_noop,
                 on_result=_noop, history=()):
        super().__init__()
        self.cfg, self.on_event, self.on_result = cfg, on_event, on_result
        self.state_path = None if state_path is None else Path(state_path)
        self.trigger_step = None if trigger_step is None else int(trigger_step)
        rf = (report_from if report_from is not None else
              self.trigger_step + cfg.bonus_free_before_report if self.trigger_step is not None else cfg.report_from)
        self.report_from = None if rf is None else int(rf)      # None: never-fade schedule (R3)
        self.stopped = False
        self.result_evals = 0
        self.result_laps = []                  # [(step, clean deterministic lap 2 [s])] of the result stretch
        for r in history:
            self._add_result(r)

    def _add_result(self, rec):
        if result_eligible(rec, self.trigger_step, self.report_from):
            self.result_evals += 1
            if rec.get('det_clean') and rec.get('lap2_s') is not None:
                self.result_laps.append((int(rec['step']), float(rec['lap2_s'])))

    def result(self):
        """Amendment 2 result block (None before a trigger): trigger_step, report_from, n_evals of
        the stretch, best clean deterministic lap 2 (best_lap2_s, best_step; None if none) and all
        clean deterministic laps 2 [{step, lap2_s}] of [trigger_step, report_from] with w = 0."""
        if self.trigger_step is None:
            return None
        best = min(self.result_laps, key=lambda x: (x[1], x[0])) if self.result_laps else (None, None)
        return {'trigger_step': self.trigger_step, 'report_from': self.report_from, 'n_evals': self.result_evals,
                'best_lap2_s': best[1], 'best_step': best[0],
                'clean_laps': [{'step': n, 'lap2_s': lap} for n, lap in self.result_laps]}

    def weight(self, n):
        return 0. if self.trigger_step is not None else bonus_weight(self.cfg, n)

    def reportable(self, n):
        return self.report_from is not None and int(n) >= self.report_from

    def state(self, step=None):
        st = {'step': None if step is None else int(step), 'trigger_step': self.trigger_step,
              'report_from': self.report_from, 'bonus_w': self.weight(step or 0) if step is not None else None,
              'schedule': {'hold_until': self.cfg.hold_until, 'fade_until': self.cfg.fade_until,
                           'bonus_free_before_report': self.cfg.bonus_free_before_report}}
        if self.trigger_step is not None:
            st['result'] = self.result()
        return st

    def save_state(self, step=None):
        if self.state_path is not None:
            write_json(self.state_path, self.state(step))

    def apply(self):
        m = self.model
        w = self.weight(m.num_timesteps)
        if hasattr(m.replay_buffer, 'bonus_weight'):
            m.replay_buffer.bonus_weight = w
        m.bonus_weight = w
        m.bonus_trigger_step, m.bonus_report_from = self.trigger_step, self.report_from
        return w

    def _on_training_start(self):
        self.apply()

    def _on_step(self):
        w = self.apply()
        self.logger.record('bonus/w', w)
        if self.trigger_step is not None and self.model.num_timesteps >= self.report_from and not self.stopped:
            self.stopped = True
            self.on_event('report_complete', step=int(self.model.num_timesteps), trigger_step=self.trigger_step,
                          report_from=self.report_from)
        return not self.stopped

    def trigger(self, step, lap2_s=None):
        self.trigger_step = int(step)
        self.report_from = self.trigger_step + int(self.cfg.bonus_free_before_report)
        if getattr(self, 'model', None) is not None:
            self.apply()
        self.save_state(step)
        self.on_event('aided_target_lap', step=int(step), lap2_s=lap2_s, report_from=self.report_from)

    def on_eval(self, rec):
        n = int(rec['step'])
        w = self.weight(n)
        rec['bonus_w'] = w
        lap2 = rec.get('lap2_s')
        if (self.trigger_step is None and self.report_from is not None and rec.get('det_clean') and lap2 is not None
                and lap2 <= self.cfg.trigger_lap2_s and (w > 0 or n < self.report_from)):
            self.trigger(n, lap2)
            rec['aided_target_lap'] = True
        rec.update(report_from=self.report_from, reportable=self.reportable(n))
        if self.trigger_step is not None:      # Amendment 2
            rec['result_eval'] = result_eligible(rec, self.trigger_step, self.report_from)
            self._add_result(rec)
        self.save_state(n)
        if self.trigger_step is not None:
            self.on_result(self.result())
        return rec


def result_eligible(rec, trigger_step, report_from):
    """Amendment 2: evaluation record `rec` lies in the bonus-free result stretch after a trigger,
    trigger_step <= step <= report_from with bonus_w == 0 (the trigger evaluation itself only if
    the weight in effect up to it was already 0). False without a trigger."""
    return (trigger_step is not None and int(trigger_step) <= int(rec['step']) <= int(report_from)
            and rec.get('bonus_w') == 0.)


def best_eligible(rec, trigger_step=None, report_from=None):
    """best.zip may come from evaluation record `rec`: reportable (records without the key, i.e.
    pre-R2 runs, are), or in the Amendment-2 result stretch (`result_eligible`)."""
    return bool(rec.get('reportable', True) or rec.get('result_eval')
                or result_eligible(rec, trigger_step, report_from))


def load_bonus_state(run_dir, model=None):
    """(trigger_step, report_from) from <run_dir>/bonus_state.json, else from the model's pickled
    attributes, else (None, None)."""
    from ..runs import read_json
    st = read_json(Path(run_dir) / 'bonus_state.json')
    if st:
        return st.get('trigger_step'), st.get('report_from')
    if model is not None and getattr(model, 'bonus_report_from', None) is not None:
        return model.bonus_trigger_step, model.bonus_report_from
    return None, None


def focus_bin(s, bin_m, n_bins):
    return min(int(float(s) // bin_m), n_bins - 1)


def focus_seed(seed_dir, ucfg, bin_m, length_m):
    """R2-7.4: crash histogram of `seed_dir`'s deterministic evaluations (laps/step_<n>_det.npz with
    n >= ucfg.seed_min_step): N_evals = number of those files, crashes_b = failed laps (termination
    off_track / wrong_way / nonfinite) whose final s lies in bin b. Returns a JSON-ready dict:
    n_evals, crashes (list per bin), hot (bins with >= ucfg.hot_min_crashes), total_crashes."""
    import re
    n_bins = int(math.ceil(length_m / bin_m))
    crashes = np.zeros(n_bins, dtype=int)
    n_evals = 0
    for f in sorted(Path(seed_dir, 'laps').glob('step_*_det.npz')):
        m = re.fullmatch(r'step_(\d+)_det\.npz', f.name)
        if m is None or int(m.group(1)) < ucfg.seed_min_step:
            continue
        n_evals += 1
        with np.load(f) as z:
            if bool(z['clean']) or str(z['termination']) not in TERMINAL_FAILURES or len(z['s']) == 0:
                continue
            crashes[focus_bin(z['s'][-1], bin_m, n_bins)] += 1
    hot = [int(b) for b in np.flatnonzero(crashes >= ucfg.hot_min_crashes)]
    return {'seed_dir': str(seed_dir), 'n_evals': n_evals, 'crashes': crashes.tolist(), 'hot': hot,
            'total_crashes': int(crashes.sum()), 'min_step': int(ucfg.seed_min_step)}


class FocusedStartCallback(BaseCallback):
    """R2-7.3 (main process): per focus bin of `bin_m` on Track.path, decayed passes n_b and
    crashes c_b of the training envs. A pass is counted when the bin of info['s_track'] changes
    (the first bin of an episode comes from the reset info); a crash at the bin of the final
    s_track of an episode that terminated off_track / wrong_way / nonfinite. Every
    ucfg.update_every steps both decay by 0.5^(update_every / half_life), then
    P = (1 - hot_floor_share) r / sum r + hot_floor_share Uniform(H0), r_b = c_b / n_b (uniform
    instead of r / sum r while sum r = 0, and instead of Uniform(H0) while H0 is empty) is pushed
    to every env (`push(P)`, default env_method('set_focus_weights', P)); also at the start of learn().
    Initial counters: `state` (resume, focus_state.json) or `seed` (focus_seed: n = prior_passes,
    c = prior_passes crashes_b / N_evals, H0 = seed hot bins) or n = prior_passes, c = 0.
    `eval_extra(model, step)` (a DrivingEvalCallback extra) writes focus_state.json atomically and
    returns {'focus': stats since the previous evaluation}: share_resets, share_transitions,
    fallback_rate (fallbacks / resets drawn as focused), handover mean / median [m] (signed, hand-over minus target), top-5 bins with P,
    r_b of the H0 bins."""

    def __init__(self, ucfg, length_m, bin_m, state_path=None, *, seed=None, state=None, push=None):
        super().__init__()
        self.ucfg, self.bin_m, self.length = ucfg, float(bin_m), float(length_m)
        self.n_bins = int(math.ceil(self.length / self.bin_m))
        self.state_path = None if state_path is None else Path(state_path)
        self.push_fn = push
        self.seed = seed
        if state is not None:
            self.n, self.c = np.asarray(state['n'], float), np.asarray(state['c'], float)
            self.hot = [int(b) for b in state['hot']]
            self.updates = int(state.get('updates', 0))
        else:
            self.n = np.full(self.n_bins, float(ucfg.prior_passes))
            if seed is not None and seed['n_evals']:
                self.c = ucfg.prior_passes * np.asarray(seed['crashes'], float) / seed['n_evals']
                self.hot = list(seed['hot'])
            else:
                self.c = np.zeros(self.n_bins)
                self.hot = list(seed['hot']) if seed else []
            self.updates = 0
        if len(self.n) != self.n_bins:
            raise ValueError(f'focus state has {len(self.n)} bins, track needs {self.n_bins}')
        self.decay = .5 ** (ucfg.update_every / ucfg.half_life)
        self._reset_counts()

    def _reset_counts(self):
        self.k = {'resets': 0, 'focused': 0, 'fallback': 0, 'transitions': 0, 'focused_transitions': 0,
                  'handover_m': []}

    def bin(self, s):
        return focus_bin(s, self.bin_m, self.n_bins)

    def weights(self):
        r = self.c / self.n
        base = r / r.sum() if r.sum() > 0 else np.full(self.n_bins, 1. / self.n_bins)
        floor = np.zeros(self.n_bins)
        if self.hot:
            floor[self.hot] = 1. / len(self.hot)
        else:
            floor[:] = 1. / self.n_bins
        h = self.ucfg.hot_floor_share
        p = (1. - h) * base + h * floor
        return p / p.sum()

    def push(self):
        p = self.weights()
        if self.push_fn is not None:
            self.push_fn(p)
        else:
            self.training_env.env_method('set_focus_weights', p)
        return p

    def _reset_info(self, i):
        return base_venv(self.training_env).reset_infos[i] or {}

    def _new_episode(self, i):
        info = self._reset_info(i)
        st = info.get('start', {}) or {}
        self.last[i] = None if info.get('s_track') is None else self.bin(info['s_track'])
        self.focused[i] = bool(st.get('focused', False))
        self.k['resets'] += 1
        self.k['focused'] += self.focused[i]
        self.k['fallback'] += bool(st.get('focus_fallback', False))
        if st.get('handover_to_target_m') is not None:
            self.k['handover_m'].append(float(st['handover_to_target_m']))

    def _on_training_start(self):
        n = self.training_env.num_envs
        self.last, self.focused = [None] * n, [False] * n
        for i in range(n):
            self._new_episode(i)
        self.next_update = (self.model.num_timesteps // self.ucfg.update_every + 1) * self.ucfg.update_every
        self.push()

    def _on_step(self):
        infos, dones = self.locals['infos'], self.locals['dones']
        for i, info in enumerate(infos):
            self.k['transitions'] += 1
            self.k['focused_transitions'] += self.focused[i]
            s = info.get('s_track')
            if s is not None and math.isfinite(s):
                b = self.bin(s)
                if b != self.last[i]:
                    self.n[b] += 1.
                    self.last[i] = b
            if dones[i]:
                if info.get('termination') in TERMINAL_FAILURES and s is not None and math.isfinite(s):
                    self.c[self.bin(s)] += 1.
                self._new_episode(i)
        n = self.model.num_timesteps
        if n >= self.next_update:
            while n >= self.next_update:
                self.n *= self.decay
                self.c *= self.decay
                self.updates += 1
                self.next_update += self.ucfg.update_every
            self.push()
        return True

    def state(self, step=None):
        return {'step': None if step is None else int(step), 'n': self.n.tolist(), 'c': self.c.tolist(),
                'hot': list(self.hot), 'updates': self.updates, 'bin_m': self.bin_m, 'n_bins': self.n_bins,
                'seed': None if self.seed is None else {k: self.seed[k] for k in ('seed_dir', 'n_evals', 'total_crashes', 'hot')}}

    def save_state(self, step=None):
        if self.state_path is not None:
            write_json(self.state_path, self.state(step))

    def eval_extra(self, model, step):
        self.save_state(step)
        k, p = self.k, self.weights()
        r = self.c / self.n
        top = np.argsort(-p)[:5]
        ho = np.asarray(k['handover_m'], float)
        out = {'resets': k['resets'], 'share_resets': k['focused'] / k['resets'] if k['resets'] else None,
               'share_transitions': k['focused_transitions'] / k['transitions'] if k['transitions'] else None,
               'fallback_rate': k['fallback'] / (k['focused'] + k['fallback']) if k['focused'] + k['fallback'] else None,
               'handover_mean_m': float(ho.mean()) if ho.size else None,
               'handover_median_m': float(np.median(ho)) if ho.size else None,
               'top5': [[int(b * self.bin_m), float(p[b])] for b in top],
               'r_hot': {str(int(b * self.bin_m)): float(r[b]) for b in self.hot}}
        self._reset_counts()
        return {'focus': out}


def load_focus_state(run_dir):
    from ..runs import read_json
    return read_json(Path(run_dir) / 'focus_state.json')
