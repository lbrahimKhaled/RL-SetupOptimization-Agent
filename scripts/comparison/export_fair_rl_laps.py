#!/usr/bin/env python3
"""Replay an immutable, run-specific RL checkpoint and export its actual lap(s).

  .venv/bin/python scripts/comparison/export_fair_rl_laps.py \
    --run runs/mac/D2/p1_TUM/seed0 --out /new/output/rl --stochastic-count 0

Reads the requested run's config and provenance, freezes best.zip before model loading,
then selects the clean logged evaluation at that snapshot's num_timesteps. Uses the same
DrivingEnv evaluation reset, settle, action rate, seeds, limits and flying-lap code as
training. CPU inference with one torch thread is the default; --device source reproduces
the training backend (including MPS). Every DNF is preserved before a replay-mismatch
error. The explicit tolerance is a replay check, not permission to treat a DNF as clean.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import comparison_state
from comparison_recording import RecordingEnv, check
from ocd_lap.track import Track
from rlsetup.driving import qss_profile
from rlsetup.driving.train_config import train_config_from_dict
from rlsetup.provenance import run_provenance, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', default='best.zip')
    parser.add_argument('--out', required=True, help='NEW directory; existing paths rejected')
    parser.add_argument('--stochastic-count', type=int, choices=(0, 10), default=0)
    parser.add_argument('--lap-tolerance-s', type=float, default=1e-6)
    parser.add_argument('--device', choices=('cpu', 'mps', 'source'), default='cpu')
    args = parser.parse_args()
    if not np.isfinite(args.lap_tolerance_s) or args.lap_tolerance_s < 0.:
        parser.error('--lap-tolerance-s must be finite and nonnegative')
    out = Path(args.out).resolve()
    frozen = comparison_state.prepare_snapshot(args.run, out, args.checkpoint)
    cfg = train_config_from_dict(json.loads((out / 'config.json').read_text()))
    reference_eval = frozen['selected_eval']
    track = Track(cfg.env.track, cfg.env.track_vmax_mps)
    provenance = run_provenance(track.name)
    source_differences = comparison_state.source_differences(
        json.loads((out / 'provenance.json').read_text()), provenance)
    comparison_state.write_json(out / 'source_check.json', {'differences': source_differences, 'current': provenance})
    if source_differences:
        raise RuntimeError('saved-run simulator/driver source differs: ' + '; '.join(source_differences))

    import torch as th
    from rlsetup.driving.agent import StagedSAC
    from rlsetup.driving.evaluate import flying_lap
    th.set_num_threads(1)
    th.manual_seed(cfg.eval.seed)
    device = cfg.device if args.device == 'source' else args.device
    if device not in ('cpu', 'mps'):
        raise ValueError(f'unsupported comparison inference device {device!r}; use cpu or mps')
    if device == 'mps' and not th.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable; no implicit CPU fallback')
    model = StagedSAC.load(out / 'checkpoint.zip', device=device)
    if int(model.num_timesteps) != frozen['checkpoint']['num_timesteps']:
        raise RuntimeError('loaded model timestep count differs from frozen checkpoint metadata')
    qss = qss_profile.for_env(track, cfg.env)
    det_env = RecordingEnv(replace(cfg.env, evaluation=True, eval_laps=2, record_telemetry=True), track=track, qss=qss)
    base_meta = {
        'source': 'rl', 'run': frozen['run'], 'snapshot': frozen,
        'checkpoint': {**frozen['checkpoint'], 'training_eval_step_of_best': reference_eval['step']},
        'run_config_sha256': frozen['config_sha256'],
        'run_provenance': json.loads((out / 'provenance.json').read_text()),
        'saved_run_config': cfg.to_dict(), 'evaluation_env_config': det_env.cfg.to_dict(),
        'conditions': {'car': cfg.env.car, 'ers': cfg.env.ers, 'drs': False, 'recuperation': False,
                       'max_brake_torque_Nm': cfg.env.max_brake_torque_Nm, 'setup': det_env.setup.to_dict(),
                       'line': f'own line; configured reference={cfg.env.reference}', 'reference': cfg.env.reference,
                       'reward_mode': cfg.env.reward.mode, 'physics_dt_s': .002, 'tick_dt_s': det_env.dt_tick,
                       'decision_dt_s': det_env.decision_dt, 'run_in_m': cfg.env.eval_run_in_m,
                       'settle_s': cfg.env.settle_s, 'eval_speed_fraction': cfg.env.eval_speed_fraction,
                       'footprint_m': det_env.footprint},
        'column_semantics_overrides': {
            'e_ref_env_m': 'offset to the configured saved-run reference, positive left',
            'r_tracking': 'diagnostic tracking term to the configured saved-run reference',
            'r_shaping': 'potential shaping using the configured saved-run reference',
            'a_throttle/a_brake': f'raw action; pedal_deadzone={cfg.env.pedal_deadzone}'},
        'env_metadata': det_env.env_metadata(),
        'code': {**provenance, 'comparison_sources_sha256': {p.name: sha256_file(p) for p in
                  [Path(__file__), Path(common.__file__), Path(comparison_state.__file__),
                   Path(sys.modules['comparison_recording'].__file__)]}},
        'torch': th.__version__, 'torch_threads_inference': 1, 'device': device,
        'training_device': cfg.device, 'python': sys.version.split()[0], 'command': sys.argv}
    stochastic = []
    try:
        report = flying_lap(model, det_env, deterministic=True, seed=cfg.eval.seed)
        reproduction = comparison_state.reproduction(report.lap2_s, reference_eval['lap2_s'], report.clean,
                                                      args.lap_tolerance_s, cfg.device, device)
        recorder_check = check(det_env, report)
        path = common.write_csv(out / f'rl_det_seed{cfg.eval.seed}.csv', 'rl', det_env.rows)
        common.write_meta(path, {**base_meta, 'episode': {
            'kind': 'deterministic', 'seed': cfg.eval.seed, 'eval_laps': det_env.cfg.eval_laps,
            'sample': '50 Hz base ticks', 'clean': report.clean, 'termination': report.termination,
            'truncation': report.truncation, 'lap_times_s': report.lap_times_s, 'laps': report.laps,
            'decisions': report.decisions, 'episode_return': report.episode_return,
            'training_eval': reference_eval, 'reproduction': reproduction,
            'reproduces_training_eval': reproduction['within_tolerance'], 'recorder_check': recorder_check,
            'handover': det_env.handover}, 'lap_time_s': report.lap2_s})
        summary = {'checkpoint': base_meta['checkpoint'], 'deterministic_clean': report.clean,
                   'deterministic_lap_times_s': report.lap_times_s, 'reproduction': reproduction,
                   'recorder_check': recorder_check, 'stochastic_requested': args.stochastic_count,
                   'stochastic': stochastic, 'complete': False}
        comparison_state.write_json(out / 'summary.json', common._clean(summary))
        print(f'det: clean={report.clean} laps={report.lap_times_s} reproduction={reproduction}', flush=True)
        if not recorder_check['lap_times_match_env']:
            raise RuntimeError('recorder clock differs from DrivingEnv; trace preserved')
        if not reproduction['within_tolerance']:
            raise RuntimeError('deterministic snapshot replay differs from training evaluation; trace and metadata preserved. '
                               'For an MPS-trained source, rerun --device source into a NEW output directory.')
        if args.stochastic_count:
            sto_env = RecordingEnv(replace(cfg.env, evaluation=True, eval_laps=1, record_telemetry=True),
                                   track=track, qss=qss)
            try:
                references = reference_eval.get('sto_laps', [])
                for index in range(args.stochastic_count):
                    seed = cfg.eval.seed + index
                    report = flying_lap(model, sto_env, deterministic=False, seed=seed)
                    rows = [row for row in sto_env.rows if row['decision_end']]
                    path = common.write_csv(out / f'rl_sto_seed{seed}.csv', 'rl', rows)
                    reference = references[index] if index < len(references) else None
                    episode = {'kind': 'stochastic', 'seed': seed, 'torch_seed': seed, 'eval_laps': 1,
                               'sample': '25 Hz decision ends', 'clean': report.clean,
                               'termination': report.termination, 'truncation': report.truncation,
                               'lap_times_s': report.lap_times_s, 'laps': report.laps, 'decisions': report.decisions,
                               'episode_return': report.episode_return, 'training_eval': reference,
                               'recorder_check': check(sto_env, report), 'handover': sto_env.handover}
                    common.write_meta(path, {**base_meta, 'episode': episode, 'lap_time_s': report.lap1_s})
                    stochastic.append({'seed': seed, 'clean': report.clean, 'lap_s': report.lap1_s,
                                       'termination': report.termination, 'truncation': report.truncation})
                    print(f'sto seed={seed}: clean={report.clean} lap={report.lap1_s}', flush=True)
            finally:
                sto_env.close()
        clean_times = [s['lap_s'] for s in stochastic if s['clean']]
        summary.update(complete=True, stochastic_clean=len(clean_times),
                       stochastic_lap_mean_s=float(np.mean(clean_times)) if clean_times else None)
        comparison_state.write_json(out / 'summary.complete.json', common._clean(summary))
        (out / 'summary.complete.json').replace(out / 'summary.json')
    finally:
        det_env.close()


if __name__ == '__main__':
    main()
