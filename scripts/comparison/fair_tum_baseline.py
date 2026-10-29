#!/usr/bin/env python3
"""Strict, run-matched TUM-speed-guided pure pursuit in the current OCD simulator.

Example (serial, no torch, does not touch live training):
  .venv/bin/python scripts/comparison/fair_tum_baseline.py \
    --run runs/mac/D2/p1_TUM/seed0 --out /new/output/path

A: original TUM raceline and unscaled TUM speed, diagnostic only, strict footprint limits.
B: TUM-speed-guided pure pursuit on OCD safe line. Spatially project safe-line preview
points onto TUM geometry using the existing sequentially matched TrackGrid; interpolate
TumProfile there. This is not an actual TUM lap or an OCD-QSS-speed baseline.

Saved run config fixes car/setup/brakes/ERS, reference, 25 Hz decisions, 2 s settle, start,
seed and every termination rule. Only the lapping controller's brake cap is changed to 1.0
to match the RL driver's full brake authority; the saved settle controller is untouched.
All attempts, including DNFs, are exported. Select the minimum clean lap 2 across the
predeclared factors, never the highest factor. Lap timing is DrivingEnv's own clock.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import comparison_state
from comparison_recording import RecordingEnv, check
from tum_pure_pursuit import TumProfile
from rlsetup.driving import qss_profile
from rlsetup.driving.demo_driver import DemoDriver
from rlsetup.driving.reference import ReferenceLine, track_grid
from rlsetup.driving.train_config import train_config_from_dict
from rlsetup.provenance import run_provenance, sha256_file
from ocd_lap.track import Track
import ocd_native

DEFAULT_FACTORS = tuple(round(1. - .02 * k, 2) for k in range(26))
LABELS = {'original': 'Original TUM raceline and TUM speed in OCD (strict diagnostic)',
          'safe': 'TUM-speed-guided pure pursuit on OCD safe line'}
EXTRA_COLUMNS = [('target_speed_mps', 'm/s', 'factor times TUM speed at the spatial preview point'),
                 ('s_controller_m', 'm', 'current arc length on the line followed by pure pursuit'),
                 ('s_preview_controller_m', 'm', 'speed preview arc length on the controller line'),
                 ('s_preview_tum_m', 'm', 'spatially matched preview arc length on TUM raceline'),
                 ('speed_scale_f', '-', 'predeclared speed factor for this attempt')]
SEMANTICS = {'e_ref_env_m': 'offset to the SAVED RUN reference (path or raceline), positive left',
             'r_tracking': 'diagnostic tracking term to the SAVED RUN reference',
             'r_shaping': 'potential shaping using the SAVED RUN reference'}
COLUMNS = [(n, u, SEMANTICS.get(n, description)) for n, u, description in common.RL_COLUMNS] + EXTRA_COLUMNS


def evaluation_config(config):
    """Keep the run's start, settle (including demo.brake_cap), setup and stop rules fixed."""
    return replace(config, evaluation=True, eval_laps=2, record_telemetry=True)


def choose_best(attempts):
    """Select fastest valid flying lap, excluding every incomplete/nonfinite/DNF attempt."""
    clean = [a for a in attempts if a.get('clean') is True
             and a.get('termination') == 'lap_complete' and a.get('truncation') is None
             and len(a.get('lap_times_s', [])) == 2 and a.get('recorder_check', {}).get('lap_times_match_env')
             and a.get('lap2_s') is not None and math.isfinite(a['lap2_s']) and a['lap2_s'] > 0.]
    return min(clean, key=lambda a: a['lap2_s']) if clean else None


class SpatialTumTarget:
    """TUM v(s) matched by physical XY, never by fractional arc length between lines."""

    def __init__(self, track, raceline, profile, grid_m):
        self.profile = profile
        # TrackGrid matches every safe-line XY sequentially onto the TUM polyline, then
        # interpolates this spatial map at the requested safe-line preview station.
        self.grid = track_grid(track, raceline, grid_m)

    def preview(self, s_safe, vx, preview_s):
        station = s_safe + max(vx, 0.) * preview_s
        tum_station = float(self.grid.s_ref_at(station))
        return station, tum_station, self.profile.speed(tum_station)


def run_attempt(env, branch, factor, profile, spatial, seed):
    """One genuine DrivingEnv evaluation. The controller reads state; it never steps backend."""
    start_wall = time.perf_counter()
    _, _ = env.reset(seed=int(seed), options={'evaluation': True})
    raceline = env.geo.raceline
    line = env.geo.path_ref if branch == 'safe' else raceline
    controller_config = replace(env.cfg.demo, brake_cap=1.)
    driver = DemoDriver(line, controller_config, pedal_deadzone=env.cfg.pedal_deadzone,
                        control_dt_s=env.decision_dt, car=env.cfg.car)
    driver.reset(env.settle_driver.integral)
    initial_integral = driver.integral
    hint, total = None, 0.
    while True:
        state = env.state
        if branch == 'safe':
            s_control = env.s
            preview, s_tum, speed = spatial.preview(s_control, state['vx'], controller_config.speed_preview_s)
        else:
            hint = raceline.match((state['x'], state['y']), hint)
            s_control = hint
            preview = s_tum = hint + max(state['vx'], 0.) * controller_config.speed_preview_s
            speed = profile.speed(s_tum)
        target = factor * speed
        action = driver.act(state, s_control, target)
        env.command_context = {'target_speed_mps': target, 's_controller_m': s_control,
                               's_preview_controller_m': preview, 's_preview_tum_m': s_tum,
                               'speed_scale_f': factor}
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    times = [lap['lap_time_s'] for lap in info['laps']]
    report = SimpleNamespace(lap_times_s=times)
    clean = info['termination'] == 'lap_complete' and len(times) == 2 and info['truncation'] is None
    recorder_check = check(env, report)
    if not recorder_check['lap_times_match_env']:
        raise RuntimeError('read-only recorder clock differs from DrivingEnv')
    summary = {'branch': branch, 'label': LABELS[branch], 'factor': factor, 'clean': clean,
               'termination': info['termination'], 'truncation': info['truncation'],
               'lap_times_s': times, 'lap2_s': times[1] if len(times) == 2 else None,
               'laps': info['laps'], 'decisions': env.decision, 'sim_time_s': info['time_s'],
               'progress_m': info['progress_m'], 'episode_return': total,
               'wall_s': time.perf_counter() - start_wall, 'recorder_check': recorder_check,
               'handover': env.handover, 'controller_initial_integral_m': initial_integral}
    return env.rows, summary


def write_trace(path, rows):
    columns = [c[0] for c in COLUMNS]
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run', required=True, help='saved source training run; primary p1_TUM')
    parser.add_argument('--out', required=True, help='NEW output directory; existing paths are rejected')
    parser.add_argument('--tum-csv', default=str(common.OUT_ROOT / 'tum/tum_none.csv'))
    parser.add_argument('--branch', choices=('both', 'original', 'safe'), default='both')
    parser.add_argument('--factors', type=float, nargs='+', default=list(DEFAULT_FACTORS),
                        help='predeclare safe-line sweep; a smaller explicit list is allowed for smoke checks')
    args = parser.parse_args()
    if not args.factors or len(set(args.factors)) != len(args.factors) or any(
            not math.isfinite(f) or f < .5 or f > 1. for f in args.factors):
        parser.error('factors must be unique finite values in [0.50, 1.00]')
    out = Path(args.out).resolve()
    frozen = comparison_state.prepare_snapshot(args.run, out)
    cfg = train_config_from_dict(json.loads((out / 'config.json').read_text()))
    env_cfg = evaluation_config(cfg.env)
    track = Track(env_cfg.track, env_cfg.track_vmax_mps)
    current_code = run_provenance(track.name)
    differences = comparison_state.source_differences(json.loads((out / 'provenance.json').read_text()), current_code)
    comparison_state.write_json(out / 'source_check.json', {'differences': differences, 'current': current_code})
    if differences:
        raise RuntimeError('saved-run simulator/driver source differs: ' + '; '.join(differences))
    tum_path = Path(args.tum_csv).resolve()
    tum_meta = json.loads(tum_path.with_suffix('.meta.json').read_text())
    tum_bytes = tum_path.read_bytes()
    tum_sha256 = hashlib.sha256(tum_bytes).hexdigest()
    if cfg.env.ers or tum_meta.get('variant') != 'none':
        raise ValueError('this protocol requires the current no-ERS source runs and TUM variant none')
    if tum_meta.get('sha256') != tum_sha256:
        raise ValueError('TUM CSV hash differs from its metadata')
    frozen_tum = out / 'tum_input.csv'
    frozen_tum.write_bytes(tum_bytes)
    comparison_state.write_json(out / 'tum_input.meta.json', tum_meta)
    qss = qss_profile.for_env(track, env_cfg)  # reset/settle ONLY; never used as the lapping speed target
    env = RecordingEnv(env_cfg, track=track, qss=qss)
    profile = TumProfile(frozen_tum, env.geo.raceline)
    spatial = SpatialTumTarget(track, env.geo.raceline, profile, env_cfg.obs.grid_m)
    plan = ([('original', 1.)] if args.branch in ('both', 'original') else []) + (
        [('safe', factor) for factor in args.factors] if args.branch in ('both', 'safe') else [])
    meta = {'source': 'fair_tum_baseline', 'snapshot': frozen, 'saved_run_config': cfg.to_dict(),
            'evaluation_env_config': env_cfg.to_dict(), 'env_metadata': env.env_metadata(),
            'conditions': {'car': env_cfg.car, 'setup_name': env_cfg.setup, 'setup': env.setup.to_dict(),
                           'max_brake_torque_Nm': env_cfg.max_brake_torque_Nm, 'ers': env_cfg.ers,
                           'drs': False, 'recuperation': False, 'reference': env_cfg.reference,
                           'footprint_m': env.footprint, 'decision_dt_s': env.decision_dt,
                           'tick_dt_s': env.dt_tick, 'seed': cfg.eval.seed,
                           'run_in_m': env_cfg.eval_run_in_m, 'settle_s': env_cfg.settle_s,
                           'eval_laps': env_cfg.eval_laps},
            'controller': {'class': 'rlsetup.driving.demo_driver.DemoDriver',
                           'settle_demo_config': asdict(env_cfg.demo),
                           'lapping_demo_config': asdict(replace(env_cfg.demo, brake_cap=1.)),
                           'integral_initialization': 'saved settle_driver.integral',
                           'mapping': 'sequential XY matches safe-line grid onto TUM raceline, then periodic interpolation',
                           'mapping_grid_m': env_cfg.obs.grid_m,
                           'target': 'f * TUM_speed(spatial_match(s_safe + max(vx,0) * speed_preview_s))'},
            'tum_profile': {'source': str(tum_path), 'snapshot': frozen_tum.name,
                            'sha256': tum_sha256, 'metadata': tum_meta,
                            'max_tum_point_offset_m': profile.max_match_offset_m},
            'track': {'name': track.name, 'safe_line_digest': env.geo.path_ref.digest,
                      'raceline_digest': env.geo.raceline.digest, 'safe_length_m': track.length,
                      'raceline_length_m': env.geo.raceline.length},
            'code': {**current_code, 'ocd_native_sha256': sha256_file(ocd_native.__file__),
                     'comparison_sources_sha256': {p.name: sha256_file(p) for p in
                         [Path(__file__), Path(common.__file__), Path(sys.modules['comparison_recording'].__file__),
                          Path(comparison_state.__file__), Path(sys.modules['tum_pure_pursuit'].__file__)]}},
            'schema': COLUMNS, 'column_semantics_overrides': SEMANTICS,
            'predeclared_plan': [{'branch': b, 'factor': f} for b, f in plan],
            'selection_rule': 'minimum clean lap2_s across all predeclared safe-line factors tried',
            'python': sys.version, 'command': sys.argv}
    comparison_state.write_json(out / 'protocol.json', common._clean(meta))
    attempts = []
    try:
        for index, (branch, factor) in enumerate(plan):
            rows, summary = run_attempt(env, branch, factor, profile, spatial, cfg.eval.seed)
            name = f'{index:02d}_{branch}_f{factor:.2f}'
            path = out / f'{name}.csv'
            write_trace(path, rows)
            common.write_meta(path, {**meta, 'summary': summary, 'lap_time_s': summary['lap2_s']})
            summary['trace'] = path.name
            attempts.append(summary)
            comparison_state.write_json(out / f'{name}.summary.json', common._clean(summary))
            # Keep a progress artifact even if an interrupted sweep is incomplete.
            common_summary = {'complete': len(attempts) == len(plan), 'predeclared_plan': meta['predeclared_plan'],
                              'selection_rule': meta['selection_rule'], 'attempts': attempts,
                              'best_safe': choose_best([a for a in attempts if a['branch'] == 'safe'])}
            temporary = out / 'summary.next.json'
            comparison_state.write_json(temporary, common._clean(common_summary))
            temporary.replace(out / 'summary.json')
            print(f"{branch} f={factor:.2f}: clean={summary['clean']} {summary['termination'] or summary['truncation']} "
                  f"laps={summary['lap_times_s']} wall={summary['wall_s']:.1f}s", flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
