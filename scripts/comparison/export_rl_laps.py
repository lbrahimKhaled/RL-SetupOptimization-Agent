#!/usr/bin/env python3
"""Export the RL driver's laps (deterministic flying lap + 10 stochastic laps) as CSV.

    .venv/bin/python scripts/comparison/export_rl_laps.py \
        [--run runs/mac/D2/sf70h_mac_r1/seed0] [--checkpoint best.zip]

Uses the repo's own evaluation path: the env config is the run's config.json (`env`), the eval
envs are built exactly as `rlsetup.driving.train.eval_envs` (deterministic: evaluation=True,
eval_laps=2; stochastic: evaluation=True, eval_laps=1) except record_telemetry=True for both, the
model is `StagedSAC.load(checkpoint)` on CPU, and every episode is `evaluate.flying_lap` with the
training evaluation's seeds (deterministic seed = eval.seed = 0; stochastic k = 0..9 with env
reset seed and torch seed eval.seed + k).

Recording: `RecordingEnv` = DrivingEnv with a read-only hook after each 50 Hz base tick
(between DrivingEnv and OCDRacingEnv in the MRO, so no rlsetup or ocd_lap file changes). The hook
reads the backend (signals, telemetry, state) and never steps or syncs it; the reproduced lap
times prove the trajectory is unchanged.
  rl_det_seed0.csv       every 50 Hz base tick (2 per 25 Hz decision); decision columns
                         (reward terms, termination) on the tick that ends each decision
  rl_sto_seed<k>.csv     decision-end ticks only (25 Hz; keeps the data small)
Failed episodes are kept (trace until termination; 'termination' on the last row). The env
stops at the first off-track 2 ms step, so on_track is 0 only on the last row of an off-track
failure. Time t_s starts at the hand-over (after the 2 s settle that is not part of the
episode); the settle starts 200 m before the line on Track.path, abeam on xi_ref = Track.path.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import ocd_trace  # noqa: E402
from ocd_lap.env import OCDRacingEnv  # noqa: E402
from ocd_lap.track import Track  # noqa: E402
from rlsetup.driving import qss_profile  # noqa: E402
from rlsetup.driving.config import EnvConfig  # noqa: E402
from rlsetup.driving.env import DrivingEnv  # noqa: E402
from rlsetup.driving.evaluate import flying_lap  # noqa: E402
from rlsetup.driving.train_config import train_config_from_dict  # noqa: E402
from rlsetup.provenance import run_provenance, sha256_file  # noqa: E402

DEFAULT_RUN = common.REPO_ROOT / 'runs/mac/D2/sf70h_mac_r1/seed0'
EXPECTED = {'step': 6950000, 'lap2_s': 103.23790594919477}   # evals.jsonl of sf70h_mac_r1/seed0


class _TickHook(OCDRacingEnv):
    """Calls `self._on_tick` after every OCDRacingEnv.step (one 50 Hz base tick)."""

    def step(self, action):
        out = super().step(action)
        self._on_tick(action, out[4])
        return out


class RecordingEnv(DrivingEnv, _TickHook):
    """DrivingEnv + per-tick rows. MRO: RecordingEnv, DrivingEnv, _TickHook, OCDRacingEnv."""

    def __init__(self, config, **kw):
        super().__init__(config, **kw)
        self.geo = ocd_trace.Geometry(self.track)

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        self.rows, self.decision = [], 0
        self.geo.reset(s_path=self.s, s_raceline=None, s_path_ref=self._s_ref)
        self.clock = ocd_trace.LapClock(self.track, self.s, len(self._lines))
        self._rec_dist, self._rec_s, self._t_prev = 0., self.s, 0.
        return obs, info

    def _on_tick(self, action, base):
        a = np.asarray(action, dtype=float)
        row = dict(t_s=self.time_s, decision=self.decision, decision_end=0, a_steer=a[0], a_throttle=a[1], a_brake=a[2])
        if not base['nonfinite']:
            st, L = self.state, float(self.track.length)
            g = self.geo.locate(st['x'], st['y'], st['yaw'], s_path=self.s)
            d0 = self._rec_dist
            self._rec_dist += (self.s - self._rec_s + L / 2) % L - L / 2
            self._rec_s = self.s
            self.clock.update(d0, self._rec_dist, self._t_prev, self.time_s)
            row.update(lap=self.clock.lap, t_lap_s=self.clock.t_lap(self.time_s), progress_m=self._rec_dist, **g,
                       **ocd_trace.backend_sample(self.backend, st, self.to_inputs(a)),
                       on_track=int(not base['off_track']))
        self.rows.append(row)
        self._t_prev = self.time_s

    def step(self, action):
        self._t_prev = self.time_s
        obs, r, term, trunc, info = super().step(action)
        c = info['step_cost']
        self.rows[-1].update(decision_end=1, reward=r, r_progress=c['progress'], r_time=c['time'],
                             r_smoothness=c['smoothness'], r_tracking=c['tracking'], r_penalty=c['penalty'],
                             r_shaping=info['shaping'], e_ref_env_m=info['e_ref'],
                             termination=info['termination'] or '', truncation=info['truncation'] or '')
        self.decision += 1
        return obs, r, term, trunc, info


def check(env, rep):
    """The recorder's own line crossings / safe-line offsets must equal the env's."""
    lap_ok = (not rep.lap_times_s) or np.allclose(env.clock.lap_times[:len(rep.lap_times_s)], rep.lap_times_s,
                                                    rtol=0, atol=1e-9)
    dec = [r for r in env.rows if r['decision_end'] and 'e_path_m' in r]
    e_dev = max((abs(r['e_path_m'] - r['e_ref_env_m']) for r in dec), default=0.)
    return {'lap_times_match_env': bool(lap_ok), 'max_abs_e_path_minus_env_e_ref_m': float(e_dev),
            'recorder_crossings_s': list(env.clock.crossings)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default=str(DEFAULT_RUN))
    ap.add_argument('--checkpoint', default='best.zip')
    ap.add_argument('--out', default=str(common.OUT_ROOT / 'rl'))
    args = ap.parse_args()
    import torch as th
    from rlsetup.driving.agent import StagedSAC
    th.manual_seed(0)
    run_dir, out = Path(args.run).resolve(), Path(args.out)
    cfg_text = (run_dir / 'config.json').read_text()
    cfg = train_config_from_dict(json.loads(cfg_text))
    ckpt = run_dir / 'checkpoints' / args.checkpoint
    model = StagedSAC.load(ckpt, device='cpu')
    track = Track(cfg.env.track)
    qss = qss_profile.load_or_build(track, car=cfg.env.car, ers=cfg.env.ers)
    det_env = RecordingEnv(replace(cfg.env, evaluation=True, record_telemetry=True), track=track, qss=qss)
    sto_env = RecordingEnv(replace(cfg.env, evaluation=True, eval_laps=1, record_telemetry=True), track=track, qss=qss)
    evals = {}
    for line in (run_dir / 'evals.jsonl').read_text().splitlines():
        rec = json.loads(line)
        evals[rec['step']] = rec
    ref_eval = evals.get(EXPECTED['step'], {})
    prov = run_provenance(track.name)
    base_meta = {
        'source': 'rl', 'run': str(run_dir.relative_to(common.REPO_ROOT)) if run_dir.is_relative_to(common.REPO_ROOT) else str(run_dir),
        'checkpoint': {'file': args.checkpoint, 'sha256': sha256_file(ckpt), 'num_timesteps': int(model.num_timesteps),
                       'training_eval_step_of_best': EXPECTED['step']},
        'run_config_sha256': sha256_file(run_dir / 'config.json'),
        'run_provenance': json.loads((run_dir / 'provenance.json').read_text()),
        'conditions': {'car': cfg.env.car, 'ers': cfg.env.ers, 'drs': False, 'recuperation': False,
                       'max_brake_torque_Nm': cfg.env.max_brake_torque_Nm, 'setup': det_env.setup.to_dict(),
                       'line': 'own line; reward shaping toward xi_ref = Track.path (safe line), time_pbrs',
                       'reward_mode': cfg.env.reward.mode, 'physics_dt_s': 0.002, 'tick_dt_s': det_env.dt_tick,
                       'decision_dt_s': det_env.decision_dt, 'run_in_m': cfg.env.eval_run_in_m,
                       'settle_s': cfg.env.settle_s, 'eval_speed_fraction': cfg.env.eval_speed_fraction,
                       'footprint_m': det_env.footprint},
        'env_metadata': det_env.env_metadata(),
        'code': {**prov, 'script_sha256': sha256_file(__file__), 'common_sha256': sha256_file(common.__file__),
                 'ocd_trace_sha256': sha256_file(ocd_trace.__file__)},
        'torch': th.__version__, 'torch_threads_inference': 1, 'device': 'cpu', 'python': sys.version.split()[0],
        'command': ' '.join(sys.argv)}

    # deterministic flying lap (training eval: eval_det, seed eval.seed)
    rep = flying_lap(model, det_env, deterministic=True, seed=cfg.eval.seed)
    path = common.write_csv(out / f'rl_det_seed{cfg.eval.seed}.csv', 'rl', det_env.rows)
    det_check = check(det_env, rep)
    common.write_meta(path, {**base_meta, 'episode': {
        'kind': 'deterministic', 'seed': cfg.eval.seed, 'eval_laps': det_env.cfg.eval_laps, 'sample': '50 Hz base ticks',
        'clean': rep.clean, 'termination': rep.termination, 'truncation': rep.truncation,
        'lap_times_s': rep.lap_times_s, 'laps': rep.laps, 'decisions': rep.decisions, 'episode_return': rep.episode_return,
        'training_eval_lap2_s': ref_eval.get('lap2_s'), 'training_eval_lap1_s': ref_eval.get('lap1_s'),
        'reproduces_training_eval': rep.lap2_s is not None and abs(rep.lap2_s - EXPECTED['lap2_s']) < 1e-6,
        'recorder_check': det_check}, 'lap_time_s': rep.lap2_s})
    print(f'det: clean {rep.clean} laps {rep.lap_times_s} (training eval lap2 {ref_eval.get("lap2_s")}) {det_check}', flush=True)

    # stochastic laps (training eval: eval_sto, 1 lap, seeds eval.seed + k)
    ref_sto = ref_eval.get('sto_laps', [])
    sto = []
    for k in range(cfg.eval.stochastic_laps):
        seed = cfg.eval.seed + k
        rep = flying_lap(model, sto_env, deterministic=False, seed=seed)
        rows = [r for r in sto_env.rows if r['decision_end']]
        path = common.write_csv(out / f'rl_sto_seed{seed}.csv', 'rl', rows)
        ref = ref_sto[k] if k < len(ref_sto) else {}
        same = (ref.get('termination') == rep.termination and ref.get('truncation') == rep.truncation
                and (ref.get('lap_s') is None if rep.lap1_s is None else abs(ref['lap_s'] - rep.lap1_s) < 1e-6))
        chk = check(sto_env, rep)
        common.write_meta(path, {**base_meta, 'episode': {
            'kind': 'stochastic', 'seed': seed, 'torch_seed': seed, 'eval_laps': sto_env.cfg.eval_laps,
            'sample': '25 Hz decision ends', 'clean': rep.clean, 'termination': rep.termination,
            'truncation': rep.truncation, 'lap_times_s': rep.lap_times_s, 'laps': rep.laps, 'decisions': rep.decisions,
            'episode_return': rep.episode_return, 'training_eval': ref, 'reproduces_training_eval': bool(same),
            'recorder_check': chk}, 'lap_time_s': rep.lap1_s})
        sto.append({'seed': seed, 'clean': rep.clean, 'lap_s': rep.lap1_s, 'termination': rep.termination,
                    'truncation': rep.truncation, 'decisions': rep.decisions, 'training_eval': ref, 'reproduces': bool(same)})
        print(f'sto seed {seed}: {rep.termination} lap {rep.lap1_s} (training {ref.get("lap_s")}, {ref.get("termination")}) '
              f'same={same} {chk["lap_times_match_env"]} {chk["max_abs_e_path_minus_env_e_ref_m"]:.2e}', flush=True)
    clean = [s['lap_s'] for s in sto if s['clean']]
    (out / 'summary.json').write_text(json.dumps(common._clean({
        'checkpoint': base_meta['checkpoint'], 'deterministic_lap_times_s': json.loads(
            (out / f'rl_det_seed{cfg.eval.seed}.meta.json').read_text())['episode']['lap_times_s'],
        'stochastic': sto, 'stochastic_clean': len(clean), 'stochastic_lap_mean_s': float(np.mean(clean)) if clean else None,
        'training_eval_sto_lap_mean_s': ref_eval.get('sto_lap_mean_s')}), indent=1) + '\n')


if __name__ == '__main__':
    main()
