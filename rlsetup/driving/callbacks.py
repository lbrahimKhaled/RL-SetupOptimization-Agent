"""SB3 callbacks of D2 training (spec §4 step 5, §6, §7).

Order in `learn(callback=[...])`: PhaseCallback, CollapseGuard, TrainStatsCallback,
DrivingEvalCallback, CheckpointCallback, WallClockLimit. SB3 calls `on_step` after every vector
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


class PhaseCallback(BaseCallback):
    """Step 5: at num_timesteps >= unfreeze_at, unfreeze actor and alpha (actor Adam is fresh:
    it never stepped while frozen; alpha continues from its initial value)."""

    def __init__(self, unfreeze_at, on_event=_noop):
        super().__init__()
        self.unfreeze_at = int(unfreeze_at)
        self.on_event = on_event
        self.unfrozen_at = None

    def _check(self):
        m = self.model
        if (m.freeze_actor or m.freeze_ent_coef) and m.num_timesteps >= self.unfreeze_at:
            m.set_frozen(False, False)
            self.unfrozen_at = m.num_timesteps
            self.logger.record('phase/unfrozen_at', m.num_timesteps)
            self.on_event('unfreeze', step=m.num_timesteps)

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
    After the window: nothing (the eval callback logs clean -> failed regressions)."""

    def __init__(self, probe_fn, baseline, cfg, *, start_step=None, rollbacks=0, on_event=_noop, log_path=None):
        super().__init__()
        self.probe_fn, self.baseline, self.cfg = probe_fn, float(baseline), cfg
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
        self.on_event('rollback', step=n, progress_m=progress, baseline_m=self.baseline,
                      actor_lr=self.cfg.rollback_actor_lr, rollbacks=self.rollbacks)
        return True


class TrainStatsCallback(BaseCallback):
    """Per `window` steps one line in `path` (train_stats.jsonl): termination / truncation counts;
    mean speed and |e| at failure and the share of failures with v < 15 m/s or |e| > 3 m; stalls
    among episodes that started (v0) below 5 m/s; episode return and length (VecMonitor).
    `totals` keeps cumulative cause counts for the eval records."""
    SLOW_MPS, WIDE_M, LOW_V0_MPS = 15., 3., 5.

    def __init__(self, path, window):
        super().__init__()
        self.path, self.window = path, int(window)
        self.totals = {}
        self.v0 = None
        self._reset_window()

    def _reset_window(self):
        self.w = {'episodes': 0, 'causes': {}, 'fail_speed': [], 'fail_e': [], 'low_v0': 0,
                  'low_v0_stall': 0, 'returns': [], 'lengths': []}

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
            if v0 is not None and v0 < self.LOW_V0_MPS:
                w['low_v0'] += 1
                w['low_v0_stall'] += cause == 'stall'
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
               'fail_share_slow_or_wide': float(np.mean((sp < self.SLOW_MPS) | (e > self.WIDE_M))) if nf else None,
               'low_v0_episodes': w['low_v0'], 'low_v0_stalls': w['low_v0_stall'],
               'ep_return_mean': float(np.mean(w['returns'])) if w['returns'] else None,
               'ep_length_mean': float(np.mean(w['lengths'])) if w['lengths'] else None}
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
    appends one JSON line to `evals_path`, saves last.zip, and best.zip on a new best CLEAN
    deterministic lap 2. A clean -> failed deterministic regression is an event."""

    def __init__(self, eval_fn, interval, evals_path, ckpt_dir, *, stats=None, best_lap2=math.inf,
                 last_det_clean=None, on_event=_noop, train_wall0=0., eval_wall0=0.):
        super().__init__()
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
        clean, lap2 = rec.get('det_clean'), rec.get('lap2_s')
        if self.last_det_clean and clean is False:
            self.on_event('det_regression', step=n, termination=rec.get('det_termination'))
        self.last_det_clean = clean
        save_model_atomic(m, self.ckpt_dir / 'last.zip')
        if clean and lap2 is not None and lap2 < self.best_lap2:
            self.best_lap2 = lap2
            save_model_atomic(m, self.ckpt_dir / 'best.zip')
            rec['new_best'] = True
        append_jsonl(self.evals_path, rec)
        for k in ('lap2_s', 'sto_clean_rate', 'alpha', 'policy_entropy'):
            if isinstance(rec.get(k), (int, float)):
                self.logger.record(f'eval/{k}', rec[k])
        self.last_step = n
        self.t_mark, self.mark_steps, self.mark_updates = time.perf_counter(), n, m._n_updates
        return rec


class CheckpointCallback(BaseCallback):
    """checkpoints/step_<n>.zip every `interval` steps (and last.zip alongside, atomically), and
    the replay buffer to `buffer_path` every `buffer_every` steps (None / 0: never here),
    atomically, so a hard-killed run can resume (TrainConfig.replay_buffer_save_every).
    The buffer saved at step n lacks that vector step's n_envs transitions (SB3 stores them after
    the callbacks). `on_buffer_saved(step)` lets train.py record the step in status.json."""

    def __init__(self, interval, ckpt_dir, *, buffer_path=None, buffer_every=None, on_buffer_saved=_noop):
        super().__init__()
        self.interval, self.ckpt_dir = int(interval), Path(ckpt_dir)
        self.buffer_path = None if buffer_path is None else Path(buffer_path)
        self.buffer_every = int(buffer_every or 0) if self.buffer_path is not None else 0
        self.on_buffer_saved = on_buffer_saved

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
            save_replay_buffer_atomic(self.model, self.buffer_path)
            self.on_buffer_saved(n)
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
