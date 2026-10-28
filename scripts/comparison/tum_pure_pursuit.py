#!/usr/bin/env python3
"""Drive TUM's Shanghai lap in OCD: pure pursuit on TUM's raceline at TUM's `ers` speed profile.

    .venv/bin/python scripts/comparison/tum_pure_pursuit.py            # needs comparison/tum/tum_ers.csv

Car and simulator: ocd_first_lap OCDBackend(powertrain='engine', car='sf70h_tum', ers=True,
max_brake_torque_Nm=12000) with the p0 setup `Setup()` (as the RL driver's env, but ERS on). DRS
does not exist in the time-domain backend, so TUM variant `ers` (ERS 4 MJ, no recuperation, no
DRS) is the closest TUM lap. The ERS store is refilled to 4 MJ at every line crossing (as
ocd_lap.simulation.drive).

Controller (reused, no new law): rlsetup.driving.demo_driver.DemoDriver on a ReferenceLine of
Track.reference (TUM's raceline), at 50 Hz (one command per 10 physics steps of 2 ms):
  steering  pure pursuit from the rear axle, goal 3 m + 0.35 s x vx ahead on the raceline,
            delta = atan(2 L sin(alpha) / d), |delta| <= 0.6 rad (ScriptedDriver geometry)
  pedals    speed PI on vx: u = 0.4 (v_t - vx) + 0.2 int(v_t - vx) dt, integral clipped to +-5 m;
            throttle = clip(u, 0, 1) scaled by the rear-slip traction control
            (1 - (slip - 0.08) / 0.12 above slip 0.08); brake = clip(-u, 0, BRAKE_CAP)
  target    v_t = f x v_TUM(s_raceline + 0.5 s x vx) (DemoPolicy's preview rule, DemoConfig
            speed_preview_s), v_TUM = TUM `ers` speed mapped onto Track.reference by position.
Only change to DemoConfig: brake_cap 0.3 -> 1.0 (DemoConfig's 0.3 x 12 kN m = 3.6 kN m is about
1.5 g, far below TUM's 4-5 g braking). Everything else is DemoConfig's default.

Flying lap: the car is placed on the raceline abeam Track.path s = -200 m, heading along the
raceline, at f x v_TUM there, then drives 2 laps; lap 2 is the reference. Line crossings and lap
times on Track.path progress, interpolated between ticks (as DrivingEnv; the raceline's s = 0 is
abeam Track.path s = -0.06 m). Track limits are NOT enforced: `on_track` records the env's own
footprint check at every 2 ms step and the car keeps driving. The run stops only on
  spin       |heading error to the raceline| > pi/2 (EnvConfig.wrong_way_rad)
  nonfinite  non-finite state
  stall      speed < 3 m/s for > 2 s (EnvConfig.stall_*)
  time_limit 400 s
If f = 1.00 does not complete lap 2, f = 0.98, 0.96, ... is searched and the first f that
completes lap 2 without spin is also exported. After f* is found the sweep continues down to
f = 0.80 as a diagnostic (summaries in search.json only, no CSV). Writes
comparison/tum_pure_pursuit_ocd/pp_f<f>.csv + .meta.json (columns: common.PP_COLUMNS) and
search.json. Deterministic (no random numbers anywhere).
"""
import argparse
from dataclasses import replace, asdict
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import ocd_trace  # noqa: E402
from ocd_lap.backend import OCDBackend, StartPose, Inputs, DT  # noqa: E402
from ocd_lap.car_setup import Setup  # noqa: E402
from ocd_lap.simulation import CONTROL_STEPS  # noqa: E402
from ocd_lap.track import Track  # noqa: E402
from rlsetup.driving.config import EnvConfig, DemoConfig  # noqa: E402
from rlsetup.driving.demo_driver import DemoDriver  # noqa: E402
from rlsetup.provenance import run_provenance  # noqa: E402

CAR = 'sf70h_tum'
BRAKE_CAP = 1.                      # DemoConfig.brake_cap override (see docstring)
MAX_BRAKE_TORQUE_NM = EnvConfig().max_brake_torque_Nm     # 12 kN m, DECISIONS #5
RUN_IN_M = EnvConfig().eval_run_in_m                      # 200 m
LAPS = EnvConfig().eval_laps                              # 2
SPIN_RAD = EnvConfig().wrong_way_rad                      # pi / 2
STALL_MPS, STALL_S = EnvConfig().stall_speed_mps, EnvConfig().stall_time_s
MAX_TIME_S = 400.
F_STEP, F_MIN = .02, .50
SWEEP_TO = .80                      # after f* is found, keep sweeping (summaries only) down to this f


class TumProfile:
    """TUM speed v(s) on Track.reference: every TUM raceline point is matched onto the polyline
    (sequential hints), so the target follows position, not TUM's spline arc length."""

    def __init__(self, csv_path, raceline):
        d = np.genfromtxt(csv_path, delimiter=',', names=True)
        d = d[d['closing_point'] == 0]
        s, hint = [], None
        for x, y in zip(d['x_m'], d['y_m']):
            hint = raceline.match((x, y), hint)
            s.append(hint)
        s = np.unwrap(np.asarray(s), period=raceline.length)
        s -= np.floor(s[0] / raceline.length) * raceline.length
        self.L = raceline.length
        order = np.argsort(s % self.L)
        self.s = (s % self.L)[order]
        self.v = d['speed_mps'][order]
        self.max_match_offset_m = float(max(abs(raceline.errors_at(si, x, y, 0.)[0])
                                            for si, x, y in zip(s, d['x_m'], d['y_m'])))

    def speed(self, s):
        return float(np.interp(s % self.L, self.s, self.v, period=self.L))


def run(f, track, geo, profile, setup):
    """One flying-lap run at speed scale f. Returns (rows, summary)."""
    cfg = replace(DemoConfig(), brake_cap=BRAKE_CAP)
    backend = OCDBackend(powertrain='engine', car=CAR, ers=True, max_brake_torque_Nm=MAX_BRAKE_TORQUE_NM)
    footprint = backend.spec.footprint
    raceline = geo.raceline
    driver = DemoDriver(raceline, cfg, pedal_deadzone=True, control_dt_s=CONTROL_STEPS * DT, car=CAR)
    s_path0 = float((-RUN_IN_M) % track.length)
    s_ref = raceline.match(track.point(s_path0))
    x0, y0 = raceline.point(s_ref)
    v0 = f * profile.speed(s_ref)
    st = backend.reset(setup, StartPose(float(x0), float(y0), raceline.heading(s_ref), v0))
    geo.reset(s_path=s_path0, s_raceline=s_ref, s_path_ref=geo.path_ref.match((x0, y0)))
    clock = ocd_trace.LapClock(track, s_path0, LAPS + 1)
    s_path, dist, t, steps, stall_s = s_path0, 0., 0., 0, 0.
    rows, cause = [], None
    while cause is None:
        target = f * profile.speed(s_ref + max(st['vx'], 0.) * cfg.speed_preview_s)
        steer = driver.steering(st, s_ref)
        throttle, brake = driver.pedals(target, st['vx'], max(abs(st['slip_rl']), abs(st['slip_rr'])))
        inputs = Inputs(steer, throttle, brake)
        inside = True
        for _ in range(CONTROL_STEPS):
            st = backend.step(inputs)
            steps += 1
            if not all(map(math.isfinite, st.values())):
                cause = 'nonfinite'
                break
            inside &= bool(track.contains_car(st['x'], st['y'], st['yaw'], footprint))
        t0, t = t, steps * DT
        if cause:
            break
        g = geo.locate(st['x'], st['y'], st['yaw'])
        d0 = dist
        dist += (g['s_path_m'] - s_path + track.length / 2) % track.length - track.length / 2
        s_path, s_ref = g['s_path_m'], g['s_raceline_m']
        if clock.update(d0, dist, t0, t):
            backend.refill_ers()                 # per-lap 4 MJ, as ocd_lap.simulation.drive
        row = dict(t_s=t, lap=clock.lap, t_lap_s=clock.t_lap(t), progress_m=dist, **g,
                   **ocd_trace.backend_sample(backend, st, inputs), on_track=int(inside), target_speed_mps=target)
        rows.append(row)
        speed = row['speed_mps']
        stall_s = stall_s + (t - t0) if speed < STALL_MPS else 0.
        if abs(g['heading_err_raceline_rad']) > SPIN_RAD:
            cause = 'spin'
        elif stall_s > STALL_S:
            cause = 'stall'
        elif clock.lap == LAPS + 1:
            cause = 'lap2_complete'
        elif t >= MAX_TIME_S:
            cause = 'time_limit'
    laps = clock.lap_times
    lap2 = [r for r in rows if r['lap'] == 2]
    summary = dict(f=f, termination=cause, completed_lap2=cause == 'lap2_complete', lap_times_s=laps,
                   lap2_s=laps[1] if len(laps) >= 2 else None, sim_time_s=t, progress_m=dist,
                   start={'s_path_m': s_path0, 's_raceline_m': s_ref if not rows else rows[0]['s_raceline_m'],
                          'speed_mps': v0},
                   on_track_fraction_all=float(np.mean([r['on_track'] for r in rows])) if rows else None,
                   on_track_fraction_lap2=float(np.mean([r['on_track'] for r in lap2])) if lap2 else None,
                   max_abs_e_raceline_lap2_m=float(max(abs(r['e_raceline_m']) for r in lap2)) if lap2 else None,
                   mean_abs_e_raceline_lap2_m=float(np.mean([abs(r['e_raceline_m']) for r in lap2])) if lap2 else None,
                   max_abs_rear_slip_lap2=float(max(max(abs(r['slip_rl']), abs(r['slip_rr'])) for r in lap2)) if lap2 else None,
                   max_abs_sideslip_lap2_rad=float(max(abs(math.atan2(r['vy_mps'], r['vx_mps'])) for r in lap2)) if lap2 else None,
                   v_max_lap2_kmh=float(3.6 * max(r['speed_mps'] for r in lap2)) if lap2 else None)
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tum-csv', default=str(common.OUT_ROOT / 'tum' / 'tum_ers.csv'))
    ap.add_argument('--out', default=str(common.OUT_ROOT / 'tum_pure_pursuit_ocd'))
    args = ap.parse_args()
    out = Path(args.out)
    track = Track('Shanghai')
    geo = ocd_trace.Geometry(track)
    profile = TumProfile(args.tum_csv, geo.raceline)
    setup = Setup()
    tum_meta = json.loads(Path(args.tum_csv).with_suffix('.meta.json').read_text())
    prov = run_provenance(track.name)
    ceiling = {'raceline': line_inside_fraction(track, geo.raceline), 'safe_line': line_inside_fraction(track, geo.path_ref)}
    print(f'footprint inside the track, car exactly on the line: {ceiling}', flush=True)
    search, written, found = [], [], None
    k = 0
    while True:
        f = round(1. - F_STEP * k, 2)
        k += 1
        if f < F_MIN - 1e-9 or (found is not None and f < SWEEP_TO - 1e-9):
            break
        w0 = time.perf_counter()
        rows, summary = run(f, track, geo, profile, setup)
        summary['wall_s'] = time.perf_counter() - w0
        summary['exported'] = f == 1. or (found is None and summary['completed_lap2'])
        search.append(summary)
        print(f"f={summary['f']:.2f}: {summary['termination']}, laps {['%.3f' % x for x in summary['lap_times_s']]}, "
              f"on_track lap2 {summary['on_track_fraction_lap2']}, max|e| lap2 {summary['max_abs_e_raceline_lap2_m']}, "
              f"{summary['wall_s']:.0f} s wall", flush=True)
        if summary['completed_lap2'] and found is None:
            found = f
        if summary['exported']:
            path = common.write_csv(out / f"pp_f{summary['f']:.2f}.csv", 'tum_pure_pursuit_ocd', rows)
            common.write_meta(path, {
                'source': 'tum_pure_pursuit_ocd', 'summary': summary,
                'lap_time_s': summary['lap2_s'], 'speed_scale_f': summary['f'],
                'conditions': {'car': CAR, 'setup': asdict(setup), 'ers': True, 'ers_refill_at_line': True,
                               'drs': False, 'recuperation': False,
                               'max_brake_torque_Nm': MAX_BRAKE_TORQUE_NM, 'brake_bias_front': setup.brake_bias_front,
                               'line': 'Track.reference (TUM raceline)', 'speed_profile': 'TUM variant ers',
                               'physics_dt_s': DT, 'control_dt_s': CONTROL_STEPS * DT, 'sample_rate_hz': 1 / (CONTROL_STEPS * DT),
                               'run_in_m': RUN_IN_M, 'laps': LAPS, 'footprint_m': backend_footprint()},
                'controller': {'class': 'rlsetup.driving.demo_driver.DemoDriver',
                               'demo_config': asdict(replace(DemoConfig(), brake_cap=BRAKE_CAP)),
                               'changed_from_default': {'brake_cap': [DemoConfig().brake_cap, BRAKE_CAP]},
                               'target': 'f * v_TUM(s_raceline + speed_preview_s * max(vx, 0))',
                               'stop_rules': {'spin_heading_err_rad': SPIN_RAD, 'stall_speed_mps': STALL_MPS,
                                              'stall_s': STALL_S, 'max_time_s': MAX_TIME_S}},
                'tum_profile': {'csv': Path(args.tum_csv).name, 'csv_sha256': common.sha256_file(args.tum_csv),
                                'tum_lap_time_s': tum_meta['lap_time_s'],
                                'max_tum_point_offset_from_Track_reference_m': profile.max_match_offset_m},
                'footprint_inside_fraction_on_line': ceiling,
                'track': {'name': track.name, 'path_length_m': float(track.length),
                          'raceline_length_m': geo.raceline.length, 'raceline_digest': geo.raceline.digest,
                          'path_digest': geo.path_ref.digest},
                'search': [{k: s[k] for k in ('f', 'termination', 'lap_times_s', 'progress_m')} for s in search],
                'code': {**prov, 'script_sha256': common.sha256_file(__file__),
                         'common_sha256': common.sha256_file(common.__file__),
                         'ocd_trace_sha256': common.sha256_file(ocd_trace.__file__)},
                'python': sys.version.split()[0], 'command': ' '.join(sys.argv)})
            written.append(path.name)
    (out / 'search.json').write_text(json.dumps(common._clean(
        {'rule': 'f* = highest f in 1.00, 0.98, ... that completes lap 2 without spin/nonfinite/stall; '
                 f'runs below f* down to {SWEEP_TO:.2f} are diagnostics only (no CSV)',
         'f_star': found, 'written': written, 'footprint_inside_fraction_on_line': ceiling, 'runs': search}), indent=1) + '\n')


def backend_footprint():
    from ocd_lap import cars
    return cars.get(CAR).footprint


def line_inside_fraction(track, line, step_m=1.):
    """Share of `line` (per metre) where the car footprint, placed exactly on the line with the
    line's heading, is inside the track (Track.contains_car): the on_track ceiling of a perfect
    follower of that line."""
    fp = backend_footprint()
    s = np.arange(0., line.length, step_m)
    return float(np.mean([track.contains_car(*line.point(si), line.heading(si), fp) for si in s]))


if __name__ == '__main__':
    main()
