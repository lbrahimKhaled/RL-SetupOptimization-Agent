"""D2 training pipeline and CLI (spec §2, §4, §7, §9).

    .venv/bin/python -m rlsetup.driving.train --preset full --seed 0 --run-name NAME [--set k=v ...]
    .venv/bin/python -m rlsetup.driving.train --resume --run-name NAME --seed 0 [--set total_timesteps=3000000]

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
Run folder runs/D2/<run_name>/seed<k>/ (spec §7); checkpoints/last.zip and best.zip live in
checkpoints/. A resumed run logs SB3 output to sb3/resume<k>/ (SB3's CSV logger would truncate
progress.csv). Resume is not bit-identical to an uninterrupted run (SB3 re-seeds the envs and the
global RNGs with the same seed on load, and the env episodes restart).
"""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import sys
import time
import traceback
import numpy as np

EXIT_OK, EXIT_GATE, EXIT_COLLAPSED, EXIT_DIRTY = 0, 2, 3, 4
RESUME_OVERRIDABLE = {'total_timesteps', 'max_wall_h', 'save_replay_buffer', 'checkpoint_interval'}


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


def eval_envs(cfg, track, qss):
    """Evaluation / probe envs in the main process, all from the run's EnvConfig (spec §2, §6)."""
    from .env import DrivingEnv
    mk = lambda c: DrivingEnv(c, track=track, qss=qss)
    e = cfg.env
    return {'eval_det': mk(replace(e, evaluation=True, record_telemetry=True)),
            'eval_sto': mk(replace(e, evaluation=True, eval_laps=1)),
            'probe': mk(replace(e, max_episode_s=cfg.guard.probe_s))}


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _callbacks(run, model, envs, starts, baseline, *, unfreeze_at, guard_start=None, rollbacks=0,
               best_lap2=float('inf'), last_det_clean=None, wall0=(0., 0.), t0=None):
    from .callbacks import (PhaseCallback, CollapseGuard, TrainStatsCallback, DrivingEvalCallback,
                            CheckpointCallback, WallClockLimit)
    from .evaluate import probe, evaluate_model
    cfg, d = run.cfg, run.dir
    stats = TrainStatsCallback(d / 'train_stats.jsonl', cfg.eval.stats_window)
    guard = CollapseGuard(lambda m: probe(m, envs['probe'], starts).mean_progress_m, baseline, cfg.guard,
                          start_step=guard_start, rollbacks=rollbacks, on_event=run.event,
                          log_path=d / 'probes.jsonl')
    evalcb = DrivingEvalCallback(
        lambda m, n: evaluate_model(m, envs, cfg.eval, laps_path=d / 'laps' / f'step_{n}_det.npz'),
        cfg.eval.interval, d / 'evals.jsonl', d / 'checkpoints', stats=stats, best_lap2=best_lap2,
        last_det_clean=last_det_clean, on_event=run.event, train_wall0=wall0[0], eval_wall0=wall0[1])
    wall = WallClockLimit(cfg.max_wall_h, t0=t0, on_event=run.event)
    cbs = [PhaseCallback(unfreeze_at, on_event=run.event), guard, stats, evalcb,
           CheckpointCallback(cfg.checkpoint_interval, d / 'checkpoints'), wall]
    return cbs, guard, wall


def _learn_and_finish(run, model, cbs, guard, wall, total):
    from ..runs import record_phase
    cfg, d = run.cfg, run.dir
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
    model.save(d / 'checkpoints' / 'last.zip')
    reason = 'collapsed' if guard.collapsed else 'wall_clock_limit' if wall.triggered else 'completed'
    if cfg.save_replay_buffer:
        model.save_replay_buffer(d / 'replay_buffer.pkl')
    run.status(phase='done', state=reason, stop_reason=reason, num_timesteps=int(model.num_timesteps),
               n_updates=int(model._n_updates), wall_total_s=time.time() - run.t0)
    run.log(f'done: {reason} at {model.num_timesteps} steps')
    return EXIT_COLLAPSED if reason == 'collapsed' else EXIT_OK


def _offline_warmup(run, model, demos, val_episodes):
    """Step 4a after the prefill: frozen actor + alpha, blocks of model.train()."""
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
    # Q vs soft MC on demo episodes (validation episodes first)
    E = len(demos['ep_start_tr'])
    eps = list(val_episodes) + [e for e in range(E) if e not in set(val_episodes)]
    eps = eps[:w.q_mc_max_episodes]
    actor = model.actor
    pairs = []
    for e in eps:
        t0, T = demos['ep_start_tr'][e], demos['ep_len'][e]
        sl = slice(t0, t0 + T)
        o, a = obs_tr[sl], demos['actions'][sl]
        with th.no_grad():
            mean, log_std, _ = actor.get_action_dist_params(th.as_tensor(o))
            logp = actor.action_dist.proba_distribution(mean, log_std).log_prob(
                th.as_tensor(a).clamp(-cfg.bc.action_clip, cfg.bc.action_clip)).numpy()
        pairs.append(q_vs_mc(model.critic, o, a, demos['rewards'][sl], logp, gamma=model.gamma,
                             alpha=model.alpha, failed=str(demos['ep_cause'][e]) in FAILURES,
                             stride=cfg.eval.q_mc_stride, min_left=cfg.eval.q_mc_min_left))
    summary = summarize_q_mc(pairs)
    append_jsonl(run.dir / 'warmup.jsonl', {'q_vs_mc_demo': summary, 'episodes': [int(e) for e in eps]})
    run.log(f'offline warm-up Q vs soft MC on demos: {summary}')
    return rows, summary


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

    rl_git = git_state(REPO_ROOT)
    if cfg.preset == 'full' and rl_git['dirty'] and not allow_dirty:
        print('refusing to run the full preset: rlsetup has uncommitted changes (commit, or pass --allow-dirty)')
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
        qss = qss_profile.load_or_build(track)
        tsha = track_sha256(track)
        cj = config_json(cfg)
        (run_dir / 'config.json').write_text(cj)
        envs = eval_envs(cfg, track, qss)
        prov = training_provenance(track, n_envs=cfg.n_envs, qss=qss, reference_digest=envs['probe'].ref.digest,
                                   track_sha256=tsha, config_json=cj, argv=argv, allow_dirty=allow_dirty)
        write_json(run_dir / 'provenance.json', prov)
        run.status(phase='setup', state='running', run_name=run_name, seed=cfg.seed, preset=cfg.preset,
                   rollbacks=0, stop_reason=None)
        log(f'D2 {run_name} seed {cfg.seed} ({cfg.preset}): {cfg.n_envs} envs, UTD {cfg.utd:g}, '
            f'{cfg.total_timesteps} steps, wall limit {cfg.max_wall_h} h -> {run_dir}')
        vec = build_vec_env(cfg.env, cfg.n_envs, track, qss, start_method=cfg.vec_start_method)
        model = build_model(cfg, vec)
        a = cfg.agent
        init_output_bias(model.actor, pedal_deadzone=cfg.env.pedal_deadzone, action=a.init_action,
                         log_std=a.init_log_std, weight_scale=a.init_weight_scale)
        model.set_logger(configure(str(run_dir / 'sb3'), ['csv', 'tensorboard']))
        starts = make_probe_starts(envs['probe'], cfg.guard)
        write_json(run_dir / 'probe_starts.json', starts)
        unfreeze_at = 0
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
            report = train_bc(model.actor, demos, cfg.bc, seed=cfg.seed, slip_gain=cfg.env.obs.slip_gain,
                              pedal_deadzone=cfg.env.pedal_deadzone, log=log)
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
            _offline_warmup(run, model, demos, report['val_episodes'])
            record_phase(run_dir, 'offline_warmup', wall_s=time.time() - t, env_steps=0,
                         grad_steps=int(model._n_updates), prefill=fill)
            unfreeze_at = cfg.warmup.online_warmup_steps
            del demos
        elif a.init_mode == 'bias_only':
            pr = probe(model, envs['probe'], starts)
            baseline = pr.mean_progress_m
            run.status(bc_baseline_m=baseline, bc_probe=pr.progress_m, baseline_source='bias_only')
        else:
            raise ValueError(f'unknown init_mode {a.init_mode!r}')
        # --- steps 4b + 5 ---
        cbs, guard, wall = _callbacks(run, model, envs, starts, baseline, unfreeze_at=unfreeze_at, t0=run.t0)
        return _learn_and_finish(run, model, cbs, guard, wall, cfg.total_timesteps)
    except BaseException as e:
        run.status(state='error', stop_reason=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        raise
    finally:
        if vec is not None:
            vec.close()
        sys.stdout, sys.stderr = old_out, old_err
        logf.close()


def resume_pipeline(run_dir, overrides=(), *, argv=None):
    """Continue a run from checkpoints/last.zip + replay_buffer.pkl. Only RESUME_OVERRIDABLE
    fields may be overridden (the report needs identical configs across seeds)."""
    import torch as th
    from stable_baselines3.common.logger import configure
    from ocd_lap.track import Track
    from ..runs import read_json, read_jsonl, add_event
    from . import qss_profile
    from .agent import StagedSAC, check_sb3_version
    from .train_config import train_config_from_dict, apply_overrides, config_json
    from .vec import build_vec_env
    run_dir = Path(run_dir)
    cfg = train_config_from_dict(read_json(run_dir / 'config.json'))
    bad = {o.split('=', 1)[0].strip() for o in overrides} - RESUME_OVERRIDABLE
    if bad:
        raise ValueError(f'cannot override {sorted(bad)} on resume (allowed: {sorted(RESUME_OVERRIDABLE)})')
    cfg = apply_overrides(cfg, overrides)
    if overrides:
        (run_dir / 'config.json').write_text(config_json(cfg))
    status = read_json(run_dir / 'status.json')
    buf = run_dir / 'replay_buffer.pkl'
    if not buf.exists():
        raise FileNotFoundError(f'{buf} missing: resume needs the replay buffer (spec §7)')
    logf = open(run_dir / 'stdout.log', 'a')
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, logf), _Tee(old_err, logf)
    log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
    run = Run(cfg, run_dir, log)
    vec = None
    try:
        check_sb3_version()
        th.set_num_threads(cfg.torch_threads)
        track = Track(cfg.env.track)
        qss = qss_profile.load_or_build(track)
        envs = eval_envs(cfg, track, qss)
        vec = build_vec_env(cfg.env, cfg.n_envs, track, qss, start_method=cfg.vec_start_method)
        model = StagedSAC.load(run_dir / 'checkpoints' / 'last.zip', env=vec, device='cpu')
        model.load_replay_buffer(buf)
        k = 1 + sum(1 for p in (run_dir / 'sb3').glob('resume*'))
        model.set_logger(configure(str(run_dir / 'sb3' / f'resume{k}'), ['csv', 'tensorboard']))
        add_event(run_dir, 'resume', step=int(model.num_timesteps), overrides=list(overrides), argv=argv or sys.argv)
        log(f'resume {run_dir} at {model.num_timesteps} steps -> {cfg.total_timesteps}')
        starts = read_json(run_dir / 'probe_starts.json')
        evals = read_jsonl(run_dir / 'evals.jsonl')
        clean = [r['lap2_s'] for r in evals if r.get('det_clean') and r.get('lap2_s') is not None]
        last = evals[-1] if evals else {}
        frozen_phase = cfg.agent.init_mode == 'bc' and status.get('unfrozen_at') is None
        cbs, guard, wall = _callbacks(
            run, model, envs, starts, status['bc_baseline_m'],
            unfreeze_at=cfg.warmup.online_warmup_steps if frozen_phase else 0,
            guard_start=status.get('unfrozen_at'), rollbacks=status.get('rollbacks', 0),
            best_lap2=min(clean) if clean else float('inf'), last_det_clean=last.get('det_clean'),
            wall0=(last.get('wall_train_s') or 0., last.get('wall_eval_s') or 0.))
        remaining = cfg.total_timesteps - model.num_timesteps
        if remaining <= 0:
            log('nothing to do: total_timesteps already reached')
            return EXIT_OK
        return _learn_and_finish(run, model, cbs, guard, wall, remaining)
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
    p.add_argument('--resume', action='store_true', help='continue from checkpoints/last.zip + replay_buffer.pkl')
    p.add_argument('--allow-dirty', action='store_true', help='full preset with uncommitted rlsetup changes (recorded)')
    return p.parse_args(argv)


def main(argv=None):
    from .train_config import PRESETS, apply_overrides
    from ..runs import resolve
    args = parse_args(argv)
    full_argv = sys.argv if argv is None else ['rlsetup.driving.train', *argv]
    if args.resume:
        root = apply_overrides(PRESETS[args.preset](), [o for o in args.overrides if o.startswith('runs_root=')]).runs_root
        run_dir = resolve(root) / args.run_name / f'seed{args.seed}'
        return resume_pipeline(run_dir, [o for o in args.overrides if not o.startswith('runs_root=')], argv=full_argv)
    cfg = replace(apply_overrides(PRESETS[args.preset](), args.overrides), seed=args.seed)
    return run_pipeline(cfg, args.run_name, allow_dirty=args.allow_dirty, argv=full_argv)


if __name__ == '__main__':
    sys.exit(main())
