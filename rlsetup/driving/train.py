"""D2 training pipeline and CLI (spec §2, §4, §7, §9).

    .venv/bin/python -m rlsetup.driving.train --preset full --seed 0 --run-name NAME [--set k=v ...]
    .venv/bin/python -m rlsetup.driving.train --resume --run-name NAME --seed 0 [--set total_timesteps=3000000]
    .venv/bin/python -m rlsetup.driving.train --preset r1_warm --seed 0 --run-name r1_warm \
        --init-from runs/D2/full1/seed0/checkpoints/best.zip

Pipeline (AgentConfig.init_mode = 'bc'):
  1 build the vec env + StagedSAC, init_output_bias(actor)
  2 demos (cached per dataset key; gate with steering-noise fallback)
  3 BC (own Adam) -> bc/actor_bc.pt, bc/report.json; gate = deterministic BC actor completes a
    clean 2-lap evaluation run (+ informative stochastic laps); failure -> status
    'bc_gate_failed', stop (BCConfig.gate_required)
    BC baseline for the CollapseGuard = mean probe progress of the BC actor
  4a prefill the replay buffer with the demos; actor + alpha frozen; offline critic warm-up
     (blocks of model.train) with critic loss and mean Q on a fixed demo batch per block
     (warmup.jsonl), then Q vs soft MC on demo episodes
  4b learn(): actor + alpha still frozen until online_warmup_steps (stochastic BC actor acts)
  5  PhaseCallback unfreezes; CollapseGuard watches the next GuardConfig.window steps
'bias_only': steps 1 + 5 only (no demos, no BC, nothing frozen; guard baseline = probe of the
bias-initialized actor).
'warm' (R1 spec §2.5, §2.6, Amendment 1 §B; requires --init-from): build the model,
`agent.warm_start` (actor, critics, targets, log alpha and Adam states from init_from; step
counter 0; actor + alpha frozen), `agent.preload_buffer` (init_from's replay_buffer.pkl or
--init-buffer, relabelled to the run's reward, status 'init_buffer'), no demos / BC; guard
baseline = probe of the loaded actor; offline critic phase (warm.offline_critic_grad_steps in
blocks, warmup.jsonl); learn(): online critic phase until warm.critic_warmup_steps, then unfreeze
(actor lr warm.actor_lr if set); critic check at the first evaluation at or after the unfreeze;
init_from / init_buffer (path, sha256, source steps, source rlsetup commit) in provenance.json
and status.json. A source whose spaces / gamma / architecture / env config (without reward) or
buffer layout differ, or a missing buffer -> status 'init_mismatch', exit 6.
StopConfig.plateau adds PlateauStop (stop reason 'plateau').
Run folder runs/D2/<run_name>/seed<k>/ (spec §7); checkpoints/last.zip and best.zip live in
checkpoints/. A resumed run logs SB3 output to sb3/resume<k>/ (SB3's CSV logger would truncate
progress.csv). last.zip / best.zip / replay_buffer.pkl are written atomically (temp file + rename);
replay_buffer.pkl is saved every TrainConfig.buffer_save_interval steps, at every evaluation
(with last.zip) and at the end, so a hard-killed seed resumes from its last checkpoint. A seed
folder with last.zip but no buffer resumes with an empty buffer, a 'warnings' entry in
status.json and no gradient updates until the buffer holds agent.batch_size x
resume_empty_buffer_min_batches transitions. Resume is not bit-identical to an uninterrupted run
(SB3 re-seeds the envs and the global RNGs with the same seed on load, and the env episodes
restart; the last loaded row is marked as an episode end so n-step targets do not chain across
the cut).
Resume checks the code against provenance.json (rlsetup commit + diff sha, ocd_lap code_version)
and refuses on a mismatch (exit 5) unless --allow-code-change, and refuses a run with
TrainConfig.strict_provenance (configs written before R1: preset 'full') and uncommitted rlsetup
changes (exit 4) unless --allow-dirty; both flags are recorded in the 'resume' event together
with the code state. A new run with strict_provenance refuses a dirty tree the same way. Resume
never re-applies init_from; a run resumed in its frozen phase ('bc' or 'warm') stays frozen until
its unfreeze step.
R2 training aids (docs/specs/R2_review.md + Amendment 1; presets f1_r2, f1_r2_smoke, f1_r3): the corner
table (corners.py) is built once here and handed to the vector-env workers and the eval envs;
corner_bonus -> BonusNStepReplayBuffer (a pre-R2 init buffer is adopted in place, bonus 0),
BonusScheduleCallback first in the callback list (w(0) is set before the offline critic phase;
state in bonus_state.json; stop reason 'report_complete' after an Amendment-1 trigger), best.zip
only from reportable evaluations or, after a trigger, from the bonus-free result stretch
[trigger, report_from] (Amendment 2; 'result' block in status.json and bonus_state.json),
best_aided.zip otherwise; focus -> FocusedStartCallback seeded
from focus_update.seed_run (state in focus_state.json); low_noise_envs -> StagedSAC._sample_action.
provenance.json 'r2' records the table, the configs, the focus seed and the buffer's labelled
fraction at the start. A resume restores the schedule state (incl. the result stretch), the focus
counters (re-seeded with a warning if focus_state.json is missing) and best / best_aided from the
evaluation records.
R3 (docs/specs/R3_review.md; preset f1_r3): a never-fade schedule (bonus_schedule.hold_until None:
w = 1, nothing reportable, no trigger, only best_aided.zip); a preloaded bonus buffer written with a
different (B, floor) is relabelled exactly (agent.preload_buffer, stats in status.json init_buffer
and provenance.json init_buffer.bonus_relabel); lifetime_steps_offset = the source run's own
offset + its num_timesteps (chained warm starts, both parts in provenance.json init_from); the
offline critic phase also logs mean Q_min on a fixed batch of pre-exit rows (warning if it still
moves); focus_update.init_state continues the focus counters of an earlier run (provenance 'r2'
focus_init_state).
R4 (docs/specs/R4_review.md; preset f1_r4): the corner table is also built for the speed potential
(reward.speed_k > 0); warm.speed_relabel relabels the adopted init buffer (agent.preload_buffer ->
relabel.relabel_speed_profile, stats in status.json / provenance.json init_buffer.speed_relabel) and
the offline critic phase probes mean Q_min on rows <= warm.probe_before_m before an entry / exit
(recovered s; 'mean_q_pre_entry' / 'mean_q_pre_exit', warning if still moving). More than
warm.speed_relabel_max_unresolved unresolved -> fallback (event 'relabel_fallback'): the model keeps
its fresh buffer, no gradient step until warm.fallback_steps transitions from the frozen actor, then
the offline critic phase (callbacks.DeferredOfflineCritic) and the usual online critic phase; the
unfreeze moves to fallback_steps + critic_warmup_steps (status 'planned_unfreeze_at'; a resume
continues the fallback). provenance.json 'r4' records the terms, the entry / exit points and the
fallback flag.
R4 from scratch (preset f1_r4_scratch, init_mode 'bc'): the demo rewards hold the speed potential, the
demo dataset stores the entry / exit terms (demos.py) and the prefill writes them to the bonus buffer;
w(0) is set before the offline warm-up and its Q vs soft MC uses r + w b. Nothing is relabelled.
init_mode 'bc' honours warm.actor_lr from the unfreeze on (PhaseCallback; warm.actor_lr None: unchanged).
"""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import sys
import time
import traceback
import numpy as np

EXIT_OK, EXIT_GATE, EXIT_COLLAPSED, EXIT_DIRTY, EXIT_CODE_CHANGED, EXIT_INIT_MISMATCH = 0, 2, 3, 4, 5, 6
RESUME_OVERRIDABLE = {'total_timesteps', 'max_wall_h', 'save_replay_buffer', 'checkpoint_interval',
                      'replay_buffer_save_every'}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def eval_envs(cfg, track, qss, corners=None):
    """Evaluation / probe envs in the main process, all from the run's EnvConfig (spec §2, §6);
    `corners`: the run's corner table (R2), shared with the training envs."""
    from .env import DrivingEnv
    mk = lambda c: DrivingEnv(c, track=track, qss=qss, corners=corners)
    e = cfg.env
    return {'eval_det': mk(replace(e, evaluation=True, record_telemetry=True)),
            'eval_sto': mk(replace(e, evaluation=True, eval_laps=1)),
            'probe': mk(replace(e, max_episode_s=cfg.guard.probe_s))}


def reference_metadata(env_cfg, env, track, qss):
    """provenance.json 'reference' for a 'csv:' xi_ref (None otherwise, so the provenance of
    'path' / 'raceline' runs is unchanged): spec, name, digest, length, the CSV's path and sha256,
    and the QSS lap on the line with the run's setup (the start-speed profile `qss` stays on
    Track.path)."""
    from .reference import csv_reference_path, reference_source
    if csv_reference_path(env_cfg.reference) is None:
        return None
    from . import qss_profile
    ref = env.ref
    return {**reference_source(env_cfg.reference), 'name': ref.name, 'digest': ref.digest,
            'length_m': ref.length, 'drive_line_is_reference': env.drive_line is ref,
            'qss_lap_on_line_s': qss_profile.line_lap_time(qss, track, ref.path, env_cfg),
            'qss_lap_on_track_path_s': float(qss.lap_time_s)}


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corner_table(cfg, track, qss):
    """The run's corner table (R2-2; cached in data/), None unless corner_bonus, focus or the R4
    speed potential is on."""
    if not (cfg.env.corner_bonus.enabled or cfg.env.focus.enabled or cfg.env.speed_on):
        return None
    from . import corners
    return corners.for_env(track, cfg.env, qss)


def _repo_path(p):
    from ..provenance import REPO_ROOT
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def focus_init_state_for(cfg):
    """R3 req 7: (state dict, provenance dict) of cfg.focus_update.init_state (a focus_state.json,
    relative to the repo root unless absolute), or (None, None). ValueError if its bin_m differs
    from cfg.env.focus.bin_m; FocusedStartCallback refuses a different bin count."""
    u = cfg.focus_update
    if not cfg.env.focus.enabled or u.init_state is None:
        return None, None
    import json
    path = _repo_path(u.init_state)
    st = json.loads(path.read_text())
    if st.get('bin_m') is not None and float(st['bin_m']) != float(cfg.env.focus.bin_m):
        raise ValueError(f"focus init_state {path} has bin_m {st['bin_m']}, the run uses {cfg.env.focus.bin_m}")
    return st, {'path': str(path), 'sha256': _sha_file(path), 'step': st.get('step'), 'updates': st.get('updates'),
                'n_bins': len(st['n']), 'hot': list(st['hot'])}


def focus_seed_for(cfg, track):
    """FocusUpdateConfig seed histogram (callbacks.focus_seed) of cfg.focus_update.seed_run, or None
    (also when focus_update.init_state is set: the continued state takes precedence)."""
    u = cfg.focus_update
    if not cfg.env.focus.enabled or u.seed_run is None or u.init_state is not None:
        return None
    length_m = float(track.length)
    from ..provenance import REPO_ROOT
    from .callbacks import focus_seed
    p = Path(u.seed_run)
    return focus_seed(p if p.is_absolute() else REPO_ROOT / p, u, cfg.env.focus.bin_m, length_m)


def r2_provenance(cfg, table, seed, labelled_fraction, focus_init=None):
    """provenance.json 'r2' (R2-9.7; R3: focus_init_state), None for runs without R2 aids."""
    if not cfg.r2_aids:
        return None
    from dataclasses import asdict
    return {'corner_table': None if table is None else table.summary(),
            'corner_bonus': asdict(cfg.env.corner_bonus), 'focus': asdict(cfg.env.focus),
            'bonus_schedule': dict(asdict(cfg.bonus_schedule), report_from=cfg.bonus_schedule.report_from),
            'focus_update': asdict(cfg.focus_update), 'focus_seed': seed,
            'low_noise': {'envs': list(cfg.agent.low_noise_envs), 'std_scale': cfg.agent.low_noise_std_scale},
            'labelled_fraction_at_start': labelled_fraction, 'focus_init_state': focus_init}


def r4_provenance(cfg, table, fallback=False):
    """provenance.json 'r4' (R4_review req 5-6), None without an R4 term."""
    if not cfg.env.r4:
        return None
    from .speed_profile import entry_points
    rw, cb = cfg.env.reward, cfg.env.corner_bonus
    pts = [{'name': name, 'entry_m': s_in, 'v_qss_entry': v_in, 'exit_m': float(u['exit_m']),
            'v_qss_exit': float(u['v_qss_exit'])} for (name, s_in, v_in), u in zip(entry_points(table), table.units)]
    return {'speed_potential': {'speed_k': rw.speed_k, 'speed_sigma': rw.speed_sigma},
            'entry_exit': {'on': cfg.env.entry_exit_on, 'bonus_in': cb.bonus_in, 'bonus_out': cb.bonus_out,
                           'sigma': cb.sigma}, 'points': pts, 'corner_table_key': table.meta.get('key'),
            'relabel_fallback': bool(fallback)}


class Run:
    """State shared by the pipeline steps of one seed."""

    def __init__(self, cfg, run_dir, log):
        self.cfg, self.dir, self.log = cfg, Path(run_dir), log
        self.t0 = time.time()

    def status(self, **kw):
        from ..runs import update_status
        return update_status(self.dir, **kw)

    def event(self, kind, **data):
        from ..runs import add_event, record_phase, read_json
        add_event(self.dir, kind, **data)
        self.log(f'event {kind}: {data}')
        if kind == 'unfreeze':
            now = time.time()
            record_phase(self.dir, 'online_warmup', wall_s=now - self.t_learn, env_steps=data['step'] - self.step0)
            self.t_unfreeze, self.step_unfreeze = now, data['step']
            self.status(phase='rl', unfrozen_at=data['step'])
        elif kind == 'rollback':
            self.status(rollbacks=data['rollbacks'])
        elif kind == 'collapsed':
            self.status(state='collapsed', stop_reason='collapsed')
        elif kind == 'wall_clock_limit':
            self.status(stop_reason='wall_clock_limit')
        elif kind == 'plateau':
            self.status(stop_reason='plateau')
        elif kind == 'aided_target_lap':
            self.status(aided_target_step=data['step'], report_from=data['report_from'])
        elif kind == 'report_complete':
            self.status(stop_reason='report_complete')
        elif kind == 'critic_not_converged':
            _warn(self, data['step'], f"critic check {data['critic_check']} above tol {data['tol']} at the first "
                                      f"evaluation after the critic-only phase (critic_not_converged)")


def _callbacks(run, model, envs, starts, baseline, *, unfreeze_at, guard_start=None, rollbacks=0,
               best_lap2=float('inf'), last_det_clean=None, wall0=(0., 0.), t0=None, buffer_saved_step=None,
               critic_check_at=None, critic_check_done=False, plateau_history=(), best_aided_lap2=float('inf'),
               bonus_state=(None, None), focus_seed=None, focus_state=None, eval_history=(), pre=()):
    """`buffer_saved_step`: step of a replay-buffer save made before learn() (the prefill save at
    step 0), so the step-0 evaluation does not save the same buffer again. `critic_check_at`:
    step of the R1 critic check (None: none); `plateau_history`: PlateauStop entries of earlier
    processes. R2: `bonus_state` = (trigger_step, report_from) of a resumed run, `eval_history` its
    evaluation records (Amendment-2 result stretch), `focus_seed` / `focus_state` initialise the
    FocusedStartCallback, `best_aided_lap2` the best_aided.zip threshold. `pre`: extra callbacks
    placed right after the schedule (R4 fallback: DeferredOfflineCritic). Returns (callbacks, guard,
    wall, plateau)."""
    from .callbacks import (PhaseCallback, CollapseGuard, TrainStatsCallback, DrivingEvalCallback,
                            CheckpointCallback, WallClockLimit, ReplayBufferSaver, PlateauStop,
                            BonusScheduleCallback, FocusedStartCallback)
    from .evaluate import probe, evaluate_model
    cfg, d = run.cfg, run.dir
    saver = ReplayBufferSaver(d / 'replay_buffer.pkl', lambda n: run.status(replay_buffer_step=int(n)),
                              last_step=buffer_saved_step) if cfg.buffer_save_interval else None
    e = cfg.eval
    stats = TrainStatsCallback(d / 'train_stats.jsonl', e.stats_window, slow_mps=e.stats_slow_mps,
                               wide_m=e.stats_wide_m, low_v0_mps=e.stats_low_v0_mps, groups=cfg.r2_aids,
                               low_noise_envs=cfg.agent.low_noise_envs, bin_m=cfg.env.focus.bin_m)
    schedule = focus = None
    extras = []
    if cfg.env.corner_bonus.enabled:
        schedule = BonusScheduleCallback(cfg.bonus_schedule, d / 'bonus_state.json', trigger_step=bonus_state[0],
                                         report_from=bonus_state[1], on_event=run.event,
                                         on_result=lambda res: run.status(result=res), history=eval_history)
        extras.append(lambda m, n: {'buffer_labelled_fraction': m.replay_buffer.labelled_fraction()})
    if cfg.env.focus.enabled:
        focus = FocusedStartCallback(cfg.focus_update, float(envs['probe'].unwrapped.track.length), cfg.env.focus.bin_m,
                                     d / 'focus_state.json', seed=focus_seed, state=focus_state)
        extras.append(focus.eval_extra)
    if cfg.r2_aids:
        extras.append(lambda m, n: {'train_groups': stats.eval_groups()})
    if cfg.env.r4:
        extras.append(lambda m, n: {'train_r4': stats.eval_r4()})
    guard = CollapseGuard(lambda m: probe(m, envs['probe'], starts).mean_progress_m, baseline, cfg.guard,
                          start_step=guard_start, rollbacks=rollbacks, on_event=run.event,
                          log_path=d / 'probes.jsonl', last_path=d / 'checkpoints' / 'last.zip')
    evalcb = DrivingEvalCallback(
        lambda m, n: evaluate_model(m, envs, cfg.eval, laps_path=d / 'laps' / f'step_{n}_det.npz'),
        cfg.eval.interval, d / 'evals.jsonl', d / 'checkpoints', stats=stats, best_lap2=best_lap2,
        last_det_clean=last_det_clean, on_event=run.event, train_wall0=wall0[0], eval_wall0=wall0[1],
        buffer_saver=saver, critic_check_at=critic_check_at, critic_check_tol=cfg.warm.critic_check_tol,
        critic_check_done=critic_check_done, extras=extras, schedule=schedule, best_aided_lap2=best_aided_lap2)
    plateau = PlateauStop(cfg.stop, evalcb, history=plateau_history, on_event=run.event)
    wall = WallClockLimit(cfg.max_wall_h, t0=t0, on_event=run.event)
    ckpt = CheckpointCallback(cfg.checkpoint_interval, d / 'checkpoints', buffer_every=cfg.buffer_save_interval,
                              buffer_saver=saver)
    phase = PhaseCallback(unfreeze_at, on_event=run.event, actor_lr=cfg.warm.actor_lr if cfg.agent.init_mode == 'bc' else None)
    cbs = ([schedule] if schedule else []) + list(pre) + [phase, guard, stats] \
        + ([focus] if focus else []) + [evalcb, plateau, ckpt, wall]
    return cbs, guard, wall, plateau


def _learn_and_finish(run, model, cbs, guard, wall, plateau, total):
    from ..runs import record_phase
    from .callbacks import save_model_atomic, save_replay_buffer_atomic, BonusScheduleCallback
    cfg, d = run.cfg, run.dir
    schedule = next((c for c in cbs if isinstance(c, BonusScheduleCallback)), None)
    run.t_learn, run.step0 = time.time(), model.num_timesteps
    run.t_unfreeze = run.step_unfreeze = None
    run.status(phase='rl' if not model.freeze_actor else 'online_warmup', state='running')
    model.learn(total, callback=cbs, reset_num_timesteps=model.num_timesteps == 0, log_interval=4)
    now = time.time()
    if run.t_unfreeze is not None:
        record_phase(d, 'rl', wall_s=now - run.t_unfreeze, env_steps=model.num_timesteps - run.step_unfreeze,
                     n_updates_total=int(model._n_updates))
    elif not model.freeze_actor:
        prev = (run.status().get('phases') or {}).get('rl', {})
        record_phase(d, 'rl', wall_s=prev.get('wall_s', 0.) + now - run.t_learn,
                     env_steps=prev.get('env_steps', 0) + model.num_timesteps - run.step0,
                     n_updates_total=int(model._n_updates))
    else:
        record_phase(d, 'online_warmup', wall_s=now - run.t_learn, env_steps=model.num_timesteps - run.step0)
    save_model_atomic(model, d / 'checkpoints' / 'last.zip')
    reason = ('collapsed' if guard.collapsed else 'plateau' if plateau.triggered
              else 'report_complete' if schedule is not None and schedule.stopped
              else 'wall_clock_limit' if wall.triggered else 'completed')
    if cfg.save_replay_buffer:
        save_replay_buffer_atomic(model, d / 'replay_buffer.pkl')
        run.status(replay_buffer_step=int(model.num_timesteps))
    extra = {}
    offset = run.status().get('lifetime_steps_offset')
    if offset is not None:   # warm start: steps of the source model + this run's steps
        extra['lifetime_timesteps'] = int(offset) + int(model.num_timesteps)
    run.status(phase='done', state=reason, stop_reason=reason, num_timesteps=int(model.num_timesteps),
               n_updates=int(model._n_updates), wall_total_s=time.time() - run.t0, **extra)
    run.log(f'done: {reason} at {model.num_timesteps} steps')
    return EXIT_COLLAPSED if reason == 'collapsed' else EXIT_OK


def _set_bonus_w0(cfg, model):
    """R2-5.2: w(0) in the replay buffer (and its audit mirror) before an offline critic phase, so its
    gradient steps use the weight BonusScheduleCallback applies from learn() on. No corner bonus: nothing."""
    if cfg.env.corner_bonus.enabled:
        from .callbacks import bonus_weight
        model.replay_buffer.bonus_weight = model.bonus_weight = bonus_weight(cfg.bonus_schedule, 0)


def _offline_warmup(run, model, demos, val_episodes):
    """Step 4a after the prefill: frozen actor + alpha, blocks of model.train(). The Q vs soft MC
    check uses r + w b per decision when the dataset has corner terms, as the critic's targets do."""
    import torch as th
    from ..runs import append_jsonl
    from .env import FAILURES
    from .evaluate import q_values, q_vs_mc, summarize_q_mc
    cfg, w = run.cfg, run.cfg.warmup
    obs_tr = demos['obs'][demos['obs_index']]
    rng = np.random.default_rng(cfg.seed)
    fixed = rng.choice(len(obs_tr), size=min(w.q_probe_batch, len(obs_tr)), replace=False)
    rows = []
    for b in range(w.offline_blocks):
        model.train(w.offline_steps_per_block, cfg.agent.batch_size)
        q_min, _ = q_values(model.critic, obs_tr[fixed], demos['actions'][fixed])
        rec = {'block': b, 'grad_steps': (b + 1) * w.offline_steps_per_block,
               'critic_loss': float(model.logger.name_to_value.get('train/critic_loss', np.nan)),
               'mean_q_demo_batch': float(q_min.mean())}
        rows.append(rec)
        append_jsonl(run.dir / 'warmup.jsonl', rec)
        model.logger.record('warmup/critic_loss', rec['critic_loss'])
        model.logger.record('warmup/mean_q_demo_batch', rec['mean_q_demo_batch'])
        model.logger.dump(step=b)
        run.log(f'offline block {b}: critic loss {rec["critic_loss"]:.3f}, mean Q {rec["mean_q_demo_batch"]:.1f}')
    # Q vs soft MC (per-step and n-step-consistent) on demo episodes (validation episodes first)
    E = len(demos['ep_start_tr'])
    eps = list(val_episodes) + [e for e in range(E) if e not in set(val_episodes)]
    eps = eps[:w.q_mc_max_episodes]
    actor = model.actor
    w_bonus = getattr(model.replay_buffer, 'bonus_weight', 0.)
    pairs = []
    for e in eps:
        t0, T = demos['ep_start_tr'][e], demos['ep_len'][e]
        sl = slice(t0, t0 + T)
        o, a = obs_tr[sl], demos['actions'][sl]
        with th.no_grad():
            mean, log_std, _ = actor.get_action_dist_params(th.as_tensor(o, device=actor.device))
            logp = actor.action_dist.proba_distribution(mean, log_std).log_prob(
                th.as_tensor(a, device=actor.device).clamp(-cfg.bc.action_clip, cfg.bc.action_clip)).cpu().numpy()
        rewards = demos['rewards'][sl]
        if w_bonus and 'corner_bonus' in demos:
            rewards = rewards + w_bonus * demos['corner_bonus'][sl]
        pairs.append(q_vs_mc(model.critic, o, a, rewards, logp, gamma=model.gamma,
                             alpha=model.alpha, failed=str(demos['ep_cause'][e]) in FAILURES,
                             stride=cfg.eval.q_mc_stride, min_left=cfg.eval.q_mc_min_left,
                             n_steps=model.n_steps))
    summary = summarize_q_mc(pairs)
    append_jsonl(run.dir / 'warmup.jsonl', {'q_vs_mc_demo': summary, 'episodes': [int(e) for e in eps]})
    run.log(f'offline warm-up Q vs soft MC on demos: {summary}')
    return rows, summary


def _offline_critic(run, model, s_rec=None, table=None):
    """Amendment 1 §B step 4: actor + alpha frozen, cfg.warm.offline_critic_grad_steps critic
    grad steps on the preloaded buffer in blocks of cfg.warm.offline_block; per block the critic
    loss and mean Q_min on a fixed batch of buffer rows (warmup.q_probe_batch) -> warmup.jsonl.
    R3 req 6 (bonus buffer with nonzero bonuses): also mean Q_min on a fixed batch of pre-exit rows
    (buffers.pre_exit_cells, horizon warm.pre_exit_horizon) per block ('mean_q_pre_exit'); if its
    range over the last warm.pre_exit_warn_blocks blocks exceeds warm.pre_exit_warn_q, a status
    warning (the global critic_check cannot see a critic still moving just before the exits).
    R4 (`s_rec` = recovered Track.path s of the stored rows, `table` = corner table): instead, fixed
    batches of rows <= warm.probe_before_m before an entry ('mean_q_pre_entry') and before an exit
    ('mean_q_pre_exit') (buffers.pre_point_cells), same warning rule for each.
    Returns the warmup.jsonl block records."""
    from ..runs import append_jsonl
    from .buffers import BonusNStepReplayBuffer, pre_exit_cells, pre_point_cells
    from .evaluate import q_values
    cfg, w = run.cfg, run.cfg.warm
    rb = model.replay_buffer
    n = rb.size() * rb.n_envs
    idx = np.random.default_rng(cfg.seed).choice(n, size=min(cfg.warmup.q_probe_batch, n), replace=False)
    rows, cols = idx // rb.n_envs, idx % rb.n_envs
    obs, act = rb.observations[rows, cols], rb.actions[rows, cols]
    probes = {}            # record key -> (obs, actions) of a fixed batch

    def add_probe(key, pr, pc, seed, what):
        if len(pr):
            k = np.random.default_rng(seed).choice(len(pr), size=min(cfg.warmup.q_probe_batch, len(pr)), replace=False)
            probes[key] = (rb.observations[pr[k], pc[k]], rb.actions[pr[k], pc[k]])
            run.log(f'{key} probe: {len(k)} of {len(pr)} rows {what}')
    if s_rec is not None and table is not None:     # R4: by recovered s
        L, before = float(table.length), w.probe_before_m
        add_probe('mean_q_pre_exit', *pre_point_cells(s_rec, [u['exit_m'] for u in table.units], L, before),
                  cfg.seed + 1, f'<= {before:g} m before an exit')
        add_probe('mean_q_pre_entry', *pre_point_cells(s_rec, [u['start_m'] for u in table.units], L, before),
                  cfg.seed + 2, f'<= {before:g} m before an entry')
    elif isinstance(rb, BonusNStepReplayBuffer):
        add_probe('mean_q_pre_exit', *pre_exit_cells(rb, w.pre_exit_horizon), cfg.seed + 1,
                  f'with a bonus within {w.pre_exit_horizon} decisions')
    out = []
    done, b = 0, 0
    while done < w.offline_critic_grad_steps:
        k = min(w.offline_block, w.offline_critic_grad_steps - done)
        model.train(k, cfg.agent.batch_size)
        done += k
        q_min, _ = q_values(model.critic, obs, act)
        rec = {'block': b, 'grad_steps': done,
               'critic_loss': float(model.logger.name_to_value.get('train/critic_loss', np.nan)),
               'mean_q_fixed_batch': float(q_min.mean())}
        for key, batch in probes.items():
            rec[key] = float(q_values(model.critic, *batch)[0].mean())
            model.logger.record(f'warmup/{key}', rec[key])
        append_jsonl(run.dir / 'warmup.jsonl', rec)
        out.append(rec)
        model.logger.record('warmup/critic_loss', rec['critic_loss'])
        model.logger.record('warmup/mean_q_fixed_batch', rec['mean_q_fixed_batch'])
        model.logger.dump(step=b)
        run.log(f"offline critic block {b}: {done} grad steps, critic loss {rec['critic_loss']:.3f}, "
                f"mean Q {rec['mean_q_fixed_batch']:.1f}"
                + ''.join(f", {key[7:].replace('_', '-')} mean Q {rec[key]:.2f}" for key in probes))
        b += 1
    for key in probes:
        if len(out) < w.pre_exit_warn_blocks:
            break
        last = [r[key] for r in out[-w.pre_exit_warn_blocks:]]
        moved = max(last) - min(last)
        if moved > w.pre_exit_warn_q:
            where = 'exits' if key == 'mean_q_pre_exit' else 'entries'
            _warn(run, 0, f"{key[7:].replace('_', '-')} mean Q_min moved {moved:.2f} (> {w.pre_exit_warn_q:g}) over "
                          f'the last {w.pre_exit_warn_blocks} offline critic blocks: critic not settled near the {where}')
    return out


def run_pipeline(cfg, run_name, *, allow_dirty=False, argv=None):
    """One seed of D2 training. Returns an exit code."""
    import torch as th
    from stable_baselines3.common.logger import configure
    from ocd_lap.track import Track
    from ..provenance import REPO_ROOT, git_state, training_provenance
    from ..runs import make_run_dir, write_json, record_phase
    from . import qss_profile
    from .agent import build_model, init_output_bias
    from .bc import train_bc, plot_throttle_vs_slip
    from .demos import ensure_demos, load_demos, prefill_replay_buffer, DemoGateFailed
    from .evaluate import flying_lap, probe, make_probe_starts, lap_metrics
    from .train_config import config_json
    from .vec import build_vec_env, track_sha256

    if (cfg.agent.init_mode == 'warm') != (cfg.init_from is not None):
        raise ValueError("init_from is required for init_mode 'warm' and only allowed with it "
                         f'(init_mode {cfg.agent.init_mode!r}, init_from {cfg.init_from!r})')
    if cfg.init_from is not None and cfg.warm.init_buffer is None:   # recorded in config.json
        from .agent import default_init_buffer
        cfg = replace(cfg, warm=replace(cfg.warm, init_buffer=default_init_buffer(cfg.init_from)))
    if cfg.init_from is not None:   # QA m2: refuse before the run folder exists (it would block a relaunch)
        need = [Path(cfg.init_from), Path(cfg.init_from).resolve().parents[1] / 'config.json', Path(cfg.warm.init_buffer)]
        missing = [str(p) for p in need if not p.is_file()]
        if missing:
            print(f'refusing to warm start: missing {missing}')
            return EXIT_INIT_MISMATCH
    if cfg.env.focus.enabled and cfg.focus_update.init_state is not None \
            and not _repo_path(cfg.focus_update.init_state).is_file():      # R3: same rule as QA m2
        print(f'refusing to run: focus_update.init_state {_repo_path(cfg.focus_update.init_state)} missing')
        return EXIT_INIT_MISMATCH
    rl_git = git_state(REPO_ROOT)
    if cfg.strict_provenance and rl_git['dirty'] and not allow_dirty:
        print(f'refusing to run preset {cfg.preset} (strict_provenance): rlsetup has uncommitted changes '
              '(commit, or pass --allow-dirty)')
        return EXIT_DIRTY
    run_dir = make_run_dir(cfg.runs_root, run_name, cfg.seed)
    logf = open(run_dir / 'stdout.log', 'a')
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, logf), _Tee(old_err, logf)
    log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
    run = Run(cfg, run_dir, log)
    vec = None
    try:
        th.set_num_threads(cfg.torch_threads)
        track = Track(cfg.env.track)
        qss = qss_profile.for_env(track, cfg.env)
        tsha = track_sha256(track)
        cj = config_json(cfg)
        (run_dir / 'config.json').write_text(cj)
        table = corner_table(cfg, track, qss)
        envs = eval_envs(cfg, track, qss, table)
        prov = training_provenance(track, n_envs=cfg.n_envs, qss=qss, reference_digest=envs['probe'].ref.digest,
                                   track_sha256=tsha, config_json=cj, argv=argv, allow_dirty=allow_dirty,
                                   reference=reference_metadata(cfg.env, envs['probe'], track, qss))
        write_json(run_dir / 'provenance.json', prov)
        run.status(phase='setup', state='running', run_name=run_name, seed=cfg.seed, preset=cfg.preset,
                   rollbacks=0, stop_reason=None)
        log(f'D2 {run_name} seed {cfg.seed} ({cfg.preset}): {cfg.n_envs} envs, UTD {cfg.utd:g}, '
            f'{cfg.total_timesteps} steps, wall limit {cfg.max_wall_h} h -> {run_dir}')
        if prov.get('reference'):
            r = prov['reference']
            log(f"xi_ref {r['spec']} (digest {r['digest']}, {r['length_m']:.1f} m): QSS {r['qss_lap_on_line_s']:.3f} s "
                f"on the line, {r['qss_lap_on_track_path_s']:.3f} s on Track.path (start speeds)")
        vec = build_vec_env(cfg.env, cfg.n_envs, track, qss, start_method=cfg.vec_start_method, corners=table)
        model = build_model(cfg, vec)
        a = cfg.agent
        seed = focus_seed_for(cfg, track)
        focus_init, focus_init_prov = focus_init_state_for(cfg)
        if table is not None:
            log('corner table ' + ', '.join(f"{u['name']} [{u['start_m']:.0f}, {u['exit_m']:.0f}] m v_exit "
                                            f"{u['v_qss_exit']:.1f}" for u in table.units))
        if a.init_mode != 'warm':   # warm: the loaded actor, no output-bias init (R1 §2.5 step 6)
            init_output_bias(model.actor, pedal_deadzone=cfg.env.pedal_deadzone, action=a.init_action,
                             log_std=a.init_log_std, weight_scale=a.init_weight_scale)
        model.set_logger(configure(str(run_dir / 'sb3'), ['csv', 'tensorboard']))
        starts = make_probe_starts(envs['probe'], cfg.guard)
        write_json(run_dir / 'probe_starts.json', starts)
        unfreeze_at, buffer_saved_step, critic_check_at = 0, None, None
        fallback, pre_cbs = None, []
        if a.init_mode == 'bc':
            # --- step 2: demos ---
            run.status(phase='demos')
            t = time.time()
            try:
                path, meta, used = ensure_demos(cfg.env, cfg.demos, qss, tsha, cfg.demo_dir,
                                                start_method=cfg.vec_start_method, log=log)
            except DemoGateFailed as e:
                run.status(state='demo_gate_failed', stop_reason=str(e))
                log(str(e))
                return EXIT_GATE
            write_json(run_dir / 'demos.json', {'path': str(path), 'sha256': meta['npz_sha256'], 'key': meta['key'],
                                               'noise_std': list(used.noise_std), 'summary': meta['summary'],
                                               'fallback_from': meta.get('fallback_from')})
            cached = bool(meta.get('cached'))
            record_phase(run_dir, 'demos', wall_s=time.time() - t, cached=cached,
                         env_steps=0 if cached else meta['summary']['transitions'],
                         dataset_transitions=meta['summary']['transitions'])
            demos = load_demos(path)
            # --- step 3: BC + gate ---
            run.status(phase='bc')
            t = time.time()
            model.actor.cpu()   # BC and its report are CPU code; the actor goes back to the GPU below
            report = train_bc(model.actor, demos, cfg.bc, seed=cfg.seed, slip_gain=cfg.env.obs.slip_gain,
                              pedal_deadzone=cfg.env.pedal_deadzone, log=log)
            model.actor.to(model.device)
            th.save(model.actor.state_dict(), run_dir / 'bc' / 'actor_bc.pt')
            det = flying_lap(model, envs['eval_det'], deterministic=True, seed=cfg.eval.seed)
            sto = [flying_lap(model, envs['eval_sto'], deterministic=False, seed=cfg.eval.seed + k)
                   for k in range(cfg.bc.gate_stochastic_laps)]
            report['gate'] = {'det_clean': det.clean, 'det_termination': det.termination, 'lap1_s': det.lap1_s,
                              'lap2_s': det.lap2_s, 'lap2': lap_metrics(det, cfg.eval),
                              'stochastic': [{'clean': s.clean, 'lap_s': s.lap1_s, 'termination': s.termination}
                                             for s in sto]}
            write_json(run_dir / 'bc' / 'report.json', report)
            plot_throttle_vs_slip(report, run_dir / 'bc' / 'throttle_vs_rear_slip.png')
            prov.update(demo_dataset_sha256=meta['npz_sha256'], bc_actor_sha256=_sha_file(run_dir / 'bc' / 'actor_bc.pt'))
            write_json(run_dir / 'provenance.json', prov)
            record_phase(run_dir, 'bc', wall_s=time.time() - t, env_steps=det.decisions + sum(s.decisions for s in sto))
            log(f'BC gate: clean {det.clean} ({det.termination}), lap 2 {det.lap2_s}, stochastic '
                f'{sum(s.clean for s in sto)}/{len(sto)} clean')
            if not det.clean:
                run.status(state='bc_gate_failed', stop_reason=f'BC deterministic lap: {det.termination}')
                if cfg.bc.gate_required:
                    return EXIT_GATE
            pr = probe(model, envs['probe'], starts)
            baseline = pr.mean_progress_m
            run.status(bc_baseline_m=baseline, bc_probe=pr.progress_m, bc_lap2_s=det.lap2_s)
            # --- step 4a: prefill + offline critic warm-up ---
            run.status(phase='offline_warmup')
            t = time.time()
            fill = prefill_replay_buffer(model.replay_buffer, demos)
            log(f'prefill: {fill}')
            model.set_frozen(actor=True, ent_coef=True)
            _set_bonus_w0(cfg, model)
            _offline_warmup(run, model, demos, report['val_episodes'])
            record_phase(run_dir, 'offline_warmup', wall_s=time.time() - t, env_steps=0,
                         grad_steps=int(model._n_updates), prefill=fill)
            if cfg.buffer_save_interval:
                # the step-0 evaluation writes last.zip; without this a kill before the first
                # periodic buffer save would resume without the demo prefill
                from .callbacks import save_replay_buffer_atomic
                save_replay_buffer_atomic(model, run_dir / 'replay_buffer.pkl')
                run.status(replay_buffer_step=0)
            unfreeze_at = cfg.warmup.online_warmup_steps
            buffer_saved_step = 0 if cfg.buffer_save_interval else None
            del demos
        elif a.init_mode == 'bias_only':
            pr = probe(model, envs['probe'], starts)
            baseline = pr.mean_progress_m
            run.status(bc_baseline_m=baseline, bc_probe=pr.progress_m, baseline_source='bias_only')
        elif a.init_mode == 'warm':
            from .agent import warm_start, preload_buffer, InitMismatch, RelabelUnresolved
            aux = {}
            try:
                init = warm_start(model, cfg.init_from, cfg)
                t = time.time()
                try:
                    buf = preload_buffer(model, cfg.warm.init_buffer, cfg.init_from, cfg, table=table,
                                         grid=envs['probe'].grid if cfg.warm.speed_relabel else None, aux=aux)
                except RelabelUnresolved as e:     # R4 fallback: fresh buffer from the frozen actor
                    fallback = e.stats
                    buf = {'path': str(Path(cfg.warm.init_buffer).resolve()), 'sha256': None, 'rows': 0,
                           'transitions': 0, 'relabel': None, 'bonus_relabel': None, 'speed_relabel': e.stats,
                           'fallback': True}
            except InitMismatch as e:
                run.status(state='init_mismatch', stop_reason=str(e))
                log(str(e))
                return EXIT_INIT_MISMATCH
            prov['init_from'] = {k: init[k] for k in ('path', 'sha256', 'src_num_timesteps', 'src_rlsetup_commit',
                                                      'src_config_sha256', 'src_alpha', 'src_lifetime_steps_offset',
                                                      'lifetime_steps_offset')}
            prov['init_buffer'] = {k: buf.get(k) for k in ('path', 'sha256', 'transitions', 'bonus_relabel',
                                                           'speed_relabel')}
            write_json(run_dir / 'provenance.json', prov)
            log(f"warm start from {init['path']} (step {init['src_num_timesteps']}, alpha {init['src_alpha']:.4f}); "
                f"actor {init['actor_sha256'][:12]}, log alpha {init['log_ent_coef_sha256'][:12]}; optimizers "
                f"{'loaded' if cfg.warm.load_optimizers else 'fresh'}; buffer {buf['transitions']} transitions "
                f"({time.time() - t:.1f} s), relabel {buf['relabel']}, bonus relabel {buf['bonus_relabel']}; "
                f"lifetime offset {init['src_lifetime_steps_offset']} + {init['src_num_timesteps']} = "
                f"{init['lifetime_steps_offset']}")
            pr = probe(model, envs['probe'], starts)
            baseline = pr.mean_progress_m
            run.status(bc_baseline_m=baseline, bc_probe=pr.progress_m, baseline_source='init_from', init_from=init,
                       init_buffer=buf, lifetime_steps_offset=init['lifetime_steps_offset'])
            if fallback is not None:
                pre_cbs = [_fallback_offline(run, model, cfg)]
                model.min_train_transitions = cfg.warm.fallback_steps
                unfreeze_at = critic_check_at = cfg.warm.fallback_steps + cfg.warm.critic_warmup_steps
                run.status(relabel_fallback=True, planned_unfreeze_at=unfreeze_at)
                run.event('relabel_fallback', step=0, unresolved_fraction=fallback['unresolved_fraction'],
                          fallback_steps=cfg.warm.fallback_steps, unfreeze_at=unfreeze_at)
                _warn(run, 0, f"speed relabel left {fallback['unresolved_fraction']:.4%} of the init buffer unresolved "
                              f'(> {cfg.warm.speed_relabel_max_unresolved:.4%}): fresh buffer, {cfg.warm.fallback_steps} '
                              f'frozen-actor transitions before the offline critic phase')
            else:
                run.status(phase='offline_warmup')
                t = time.time()
                _set_bonus_w0(cfg, model)
                _offline_critic(run, model, aux.get('s'), table)
                record_phase(run_dir, 'offline_warmup', wall_s=time.time() - t, env_steps=0,
                             grad_steps=int(model._n_updates))
                unfreeze_at = critic_check_at = cfg.warm.critic_warmup_steps
            aux.clear()
        else:
            raise ValueError(f'unknown init_mode {a.init_mode!r}')
        r2 = r2_provenance(cfg, table, seed, model.replay_buffer.labelled_fraction()
                           if hasattr(model.replay_buffer, 'labelled_fraction') else None, focus_init_prov)
        if r2 is not None:
            prov['r2'] = r2
            write_json(run_dir / 'provenance.json', prov)
            run.status(report_from=cfg.bonus_schedule.report_from if cfg.env.corner_bonus.enabled else None,
                       labelled_fraction_at_start=r2['labelled_fraction_at_start'])
        r4 = r4_provenance(cfg, table, fallback is not None)
        if r4 is not None:
            prov['r4'] = r4
            write_json(run_dir / 'provenance.json', prov)
        # --- steps 4b + 5 ---
        cbs, guard, wall, plateau = _callbacks(run, model, envs, starts, baseline, unfreeze_at=unfreeze_at,
                                               t0=run.t0, buffer_saved_step=buffer_saved_step,
                                               critic_check_at=critic_check_at, focus_seed=seed,
                                               focus_state=focus_init, pre=pre_cbs)
        return _learn_and_finish(run, model, cbs, guard, wall, plateau, cfg.total_timesteps)
    except BaseException as e:
        run.status(state='error', stop_reason=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        raise
    finally:
        if vec is not None:
            vec.close()
        sys.stdout, sys.stderr = old_out, old_err
        logf.close()


def _fallback_offline(run, model, cfg):
    """R4 fallback: DeferredOfflineCritic running the offline critic phase once the fresh buffer
    holds warm.fallback_steps transitions (status 'offline_critic_done_at')."""
    from .callbacks import DeferredOfflineCritic
    from ..runs import record_phase

    def phase():
        t = time.time()
        run.status(phase='offline_warmup')
        _offline_critic(run, model)
        record_phase(run.dir, 'offline_warmup', wall_s=time.time() - t, env_steps=0, grad_steps=int(model._n_updates))
        run.status(phase='online_warmup', offline_critic_done_at=int(model.num_timesteps))
    return DeferredOfflineCritic(cfg.warm.fallback_steps, phase,
                                 on_done=lambda n: run.event('fallback_offline_critic', step=int(n)))


def _warn(run, step, msg):
    """Append a {'utc', 'step', 'message'} entry to status.json 'warnings' and log it."""
    from ..runs import read_json, utc_now
    run.status(warnings=(read_json(run.dir / 'status.json').get('warnings') or []) +
               [{'utc': utc_now(), 'step': int(step), 'message': msg}])
    run.log(f'WARNING: {msg}')


def resume_pipeline(run_dir, overrides=(), *, argv=None, allow_dirty=False, allow_code_change=False, init_from=None):
    """Continue a run from checkpoints/last.zip + replay_buffer.pkl. Only RESUME_OVERRIDABLE
    fields may be overridden (the report needs identical configs across seeds).
    Code check (QA D2 M2): the current rlsetup git state + ocd_lap code_version are compared with
    provenance.json; a mismatch refuses (EXIT_CODE_CHANGED) unless allow_code_change, uncommitted
    rlsetup changes with strict_provenance (missing in the config: preset 'full') refuse
    (EXIT_DIRTY) unless allow_dirty. A refusal adds a
    'resume_refused' event and changes nothing else; the 'resume' event records the code state,
    mismatches and both flags.
    Replay buffer: loaded and the last row marked as an episode end (cut_nstep_chains). A buffer
    older than last.zip (replay_buffer_step < num_timesteps - n_envs, or unknown) adds a warning.
    Without replay_buffer.pkl (hard kill before the first buffer save) the run continues with an
    empty buffer (no demo transitions), a warning ('warnings', event 'resume_without_replay_buffer')
    and no gradient updates until the buffer holds agent.batch_size x
    resume_empty_buffer_min_batches transitions. Without last.zip: FileNotFoundError.
    `init_from` (CLI --init-from) is never re-applied; a path different from the run's config
    raises ValueError. A run in its frozen phase ('bc' / 'warm', no unfreeze event yet) unfreezes
    at its configured step; PlateauStop and the critic check continue from evals.jsonl."""
    import torch as th
    from stable_baselines3.common.logger import configure
    from ocd_lap.track import Track
    from ..provenance import code_state, code_mismatches
    from ..runs import read_json, read_jsonl, add_event
    from . import qss_profile
    from .agent import StagedSAC, check_sb3_version
    from .callbacks import load_replay_buffer_for_resume
    from .train_config import train_config_from_dict, apply_overrides, config_json
    from .vec import build_vec_env
    run_dir = Path(run_dir)
    raw = read_json(run_dir / 'config.json')
    cfg = train_config_from_dict(raw)
    strict = bool(raw.get('strict_provenance', cfg.preset == 'full'))   # pre-R1 configs: preset rule
    if init_from is not None and str(Path(init_from).resolve()) != cfg.init_from:
        raise ValueError(f'--init-from {init_from} differs from the run\'s init_from {cfg.init_from} '
                         '(resume never re-applies init_from)')
    bad = {o.split('=', 1)[0].strip() for o in overrides} - RESUME_OVERRIDABLE
    if bad:
        raise ValueError(f'cannot override {sorted(bad)} on resume (allowed: {sorted(RESUME_OVERRIDABLE)})')
    cfg = apply_overrides(cfg, overrides)
    status = read_json(run_dir / 'status.json')
    last_zip = run_dir / 'checkpoints' / 'last.zip'
    if not last_zip.exists():
        raise FileNotFoundError(f'{last_zip} missing: nothing to resume from')
    buf = run_dir / 'replay_buffer.pkl'
    logf = open(run_dir / 'stdout.log', 'a')
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, logf), _Tee(old_err, logf)
    log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
    run = Run(cfg, run_dir, log)
    vec = None
    argv = list(argv or sys.argv)
    try:
        check_sb3_version()
        th.set_num_threads(cfg.torch_threads)
        track = Track(cfg.env.track)
        # --- code check (before anything in the run folder changes) ---
        code = code_state(track)
        mismatches = code_mismatches(read_json(run_dir / 'provenance.json', {}) or {}, code)
        dirty = strict and bool(code['rlsetup']['dirty'])
        refused = []
        if mismatches and not allow_code_change:
            refused.append((EXIT_CODE_CHANGED, 'code differs from provenance.json: ' + '; '.join(mismatches)
                            + ' (pass --allow-code-change to resume anyway)'))
        if dirty and not allow_dirty:
            refused.append((EXIT_DIRTY, 'rlsetup has uncommitted changes and the run has strict_provenance '
                                        '(commit, or pass --allow-dirty)'))
        if refused:
            add_event(run_dir, 'resume_refused', reasons=[r for _, r in refused], code=code, argv=argv)
            for _, r in refused:
                log(f'refusing to resume: {r}')
            return refused[0][0]
        if overrides:
            (run_dir / 'config.json').write_text(config_json(cfg))
        qss = qss_profile.for_env(track, cfg.env)
        table = corner_table(cfg, track, qss)
        envs = eval_envs(cfg, track, qss, table)
        vec = build_vec_env(cfg.env, cfg.n_envs, track, qss, start_method=cfg.vec_start_method, corners=table)
        model = StagedSAC.load(last_zip, env=vec, device=cfg.device)
        n = int(model.num_timesteps)
        buffer_step = status.get('replay_buffer_step')
        cut_row = None
        if buf.exists():
            cut_row = load_replay_buffer_for_resume(model, buf)
            model.min_train_transitions = 0
            buffer_lag = buffer_step is None or buffer_step < n - cfg.n_envs
            if buffer_lag:
                _warn(run, n, f'replay_buffer.pkl is from step {buffer_step} but last.zip from step {n}: '
                              f'the transitions in between are lost (hard kill between the two saves)')
        else:
            buffer_lag = None
            model.min_train_transitions = cfg.agent.batch_size * cfg.resume_empty_buffer_min_batches
            add_event(run_dir, 'resume_without_replay_buffer', step=n)
            _warn(run, n, f'resume at step {n} without replay_buffer.pkl: continuing with an empty replay '
                          f'buffer (demo prefill and earlier transitions lost); no gradient updates until it '
                          f'holds {model.min_train_transitions} transitions')
            buffer_step = None
        if mismatches:
            _warn(run, n, 'resumed with different code (--allow-code-change): ' + '; '.join(mismatches))
        if dirty:
            _warn(run, n, 'resumed a strict_provenance run with uncommitted rlsetup changes (--allow-dirty)')
        k = 1 + sum(1 for p in (run_dir / 'sb3').glob('resume*'))
        model.set_logger(configure(str(run_dir / 'sb3' / f'resume{k}'), ['csv', 'tensorboard']))
        add_event(run_dir, 'resume', step=n, overrides=list(overrides), argv=argv,
                  replay_buffer=buf.exists(), replay_buffer_step=buffer_step,
                  replay_buffer_size=int(model.replay_buffer.size()), replay_buffer_cut_row=cut_row,
                  replay_buffer_lag=buffer_lag, min_train_transitions=int(model.min_train_transitions),
                  code=code, code_mismatches=mismatches, allow_code_change=bool(allow_code_change),
                  allow_dirty=bool(allow_dirty))
        log(f'resume {run_dir} at {n} steps -> {cfg.total_timesteps}')
        starts = read_json(run_dir / 'probe_starts.json')
        evals = read_jsonl(run_dir / 'evals.jsonl')
        from .callbacks import load_bonus_state, load_focus_state, best_eligible
        bonus_state = load_bonus_state(run_dir, model) if cfg.env.corner_bonus.enabled else (None, None)
        # R2-9.3 / Amendment 2: best.zip from reportable evals, or after a trigger from the bonus-free
        # result stretch [trigger_step, report_from]; best_aided.zip from the others and the trigger
        done = [r for r in evals if r.get('det_clean') and r.get('lap2_s') is not None]
        clean = [r['lap2_s'] for r in done if best_eligible(r, *bonus_state)]
        aided = [r['lap2_s'] for r in done if not best_eligible(r, *bonus_state) or r.get('aided_target_lap')]
        last = evals[-1] if evals else {}
        if bonus_state[0] is not None and n >= bonus_state[1]:   # Amendment 1: ended itself after a trigger
            log(f'nothing to do: the run stopped itself at report_from {bonus_state[1]} after the trigger at '
                f'{bonus_state[0]}')
            return EXIT_OK
        focus_state = load_focus_state(run_dir) if cfg.env.focus.enabled else None
        seed = None
        if cfg.env.focus.enabled and focus_state is None:
            if cfg.focus_update.init_state is not None:      # R3: back to the run's initial state
                focus_state = focus_init_state_for(cfg)[0]
                src = cfg.focus_update.init_state
            else:
                seed = focus_seed_for(cfg, track)
                src = cfg.focus_update.seed_run
            _warn(run, n, f'focus_state.json missing: focus counters re-seeded from {src} (training-time counts lost)')
        from .callbacks import plateau_entry
        warm = cfg.agent.init_mode == 'warm'
        frozen_phase = cfg.agent.init_mode in ('bc', 'warm') and status.get('unfrozen_at') is None
        unfreeze_step = cfg.warm.critic_warmup_steps if warm else cfg.warmup.online_warmup_steps
        pre_cbs = []
        if status.get('planned_unfreeze_at') is not None:      # R4 fallback run
            unfreeze_step = int(status['planned_unfreeze_at'])
            if status.get('offline_critic_done_at') is None:
                pre_cbs = [_fallback_offline(run, model, cfg)]
                model.min_train_transitions = cfg.warm.fallback_steps
                log(f'R4 fallback: offline critic phase still due at {cfg.warm.fallback_steps} steps')
        cbs, guard, wall, plateau = _callbacks(
            run, model, envs, starts, status['bc_baseline_m'],
            unfreeze_at=unfreeze_step if frozen_phase else 0,
            guard_start=status.get('unfrozen_at'), rollbacks=status.get('rollbacks', 0),
            best_lap2=min(clean) if clean else float('inf'), last_det_clean=last.get('det_clean'),
            wall0=(last.get('wall_train_s') or 0., last.get('wall_eval_s') or 0.),
            buffer_saved_step=buffer_step if buffer_step == n else None,
            critic_check_at=unfreeze_step if warm else None,
            critic_check_done=any('critic_check' in r for r in evals),
            plateau_history=[plateau_entry(r) for r in evals],
            best_aided_lap2=min(aided) if aided else float('inf'), bonus_state=bonus_state,
            focus_seed=seed, focus_state=focus_state, eval_history=evals, pre=pre_cbs)
        remaining = cfg.total_timesteps - n
        if remaining <= 0:
            log('nothing to do: total_timesteps already reached')
            return EXIT_OK
        return _learn_and_finish(run, model, cbs, guard, wall, plateau, remaining)
    except BaseException as e:
        run.status(state='error', stop_reason=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        raise
    finally:
        if vec is not None:
            vec.close()
        sys.stdout, sys.stderr = old_out, old_err
        logf.close()


def parse_args(argv=None):
    from .train_config import PRESETS
    p = argparse.ArgumentParser(description='D2 SAC driving-agent training (docs/specs/D2_agent.md)')
    p.add_argument('--preset', default='full', choices=sorted(PRESETS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-name', required=True)
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help="config override, e.g. --set agent.batch_size=256 (repeatable)")
    p.add_argument('--resume', action='store_true',
                   help='continue from checkpoints/last.zip + replay_buffer.pkl (empty buffer + warning if missing)')
    p.add_argument('--init-from', default=None, metavar='PATH',
                   help="model .zip to warm start from (required iff agent.init_mode == 'warm'; its run folder "
                        "must hold config.json); stored as an absolute path")
    p.add_argument('--init-buffer', default=None, metavar='PATH',
                   help='replay_buffer.pkl to preload with --init-from (default: the run folder of --init-from)')
    p.add_argument('--allow-dirty', action='store_true',
                   help='strict_provenance preset with uncommitted rlsetup changes (recorded)')
    p.add_argument('--allow-code-change', action='store_true',
                   help='resume although the code differs from provenance.json (recorded)')
    return p.parse_args(argv)


def main(argv=None):
    from .train_config import PRESETS, apply_overrides
    from ..runs import resolve
    args = parse_args(argv)
    full_argv = sys.argv if argv is None else ['rlsetup.driving.train', *argv]
    if args.resume:
        root = apply_overrides(PRESETS[args.preset](), [o for o in args.overrides if o.startswith('runs_root=')]).runs_root
        run_dir = resolve(root) / args.run_name / f'seed{args.seed}'
        return resume_pipeline(run_dir, [o for o in args.overrides if not o.startswith('runs_root=')], argv=full_argv,
                               allow_dirty=args.allow_dirty, allow_code_change=args.allow_code_change,
                               init_from=args.init_from)
    cfg = replace(apply_overrides(PRESETS[args.preset](), args.overrides), seed=args.seed)
    if args.init_from is not None:
        cfg = replace(cfg, init_from=str(Path(args.init_from).resolve()))
    if args.init_buffer is not None:
        if args.init_from is None:
            sys.exit('error: --init-buffer needs --init-from')
        cfg = replace(cfg, warm=replace(cfg.warm, init_buffer=str(Path(args.init_buffer).resolve())))
    if (cfg.agent.init_mode == 'warm') != (cfg.init_from is not None):
        sys.exit(f"error: --init-from is required iff agent.init_mode == 'warm' (init_mode {cfg.agent.init_mode!r})")
    return run_pipeline(cfg, args.run_name, allow_dirty=args.allow_dirty, argv=full_argv)


if __name__ == '__main__':
    sys.exit(main())
