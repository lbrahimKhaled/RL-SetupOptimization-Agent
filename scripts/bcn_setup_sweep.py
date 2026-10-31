#!/usr/bin/env python3
"""QSS setup selection for Barcelona (racetrack-database `Catalunya`), car sf70h_tum, ERS/DRS off.

Every evaluation = one g-g-v measurement on OCD (`ocd_lap.qss_lap.measure_ggv`, brake ramp up to
the given total brake torque) + `qss_lap.lap_time` on Track.path (the safe line, what the
driving env's QSS profile uses). The same g-g-v also gives the lap on Track.reference (TUM's
raceline; no new measurement). Steps:
  1. named setups p0, p1_TUM at 12 and 15 kN m total brake torque          (4 evaluations)
  2. one knob at a time around the fastest named (setup, brake)            (SWEEP)
  3. extend a knob past the sweep edge (same step, up to its bound) while it keeps improving
  4. greedy combination: add the improving single-knob changes in order of gain, keep a change
     only if the combined lap improves; ride-height / rake changes that LOWER the car are not
     admitted (see SENSIBLE below)
  5. the result at the other brake torque                                   (1 evaluation)
Physical checks per row: translator warnings (e.g. a clamped anti-roll bar), lift-clamp
warnings, spin / understeer gradient in the ramp steer, knobs at a bound, and the static ride
heights at the QSS top speed (OCD has no floor contact: a negative value = below the ground).

    .venv/bin/python scripts/bcn_setup_sweep.py [--out docs/results/bcn/qss_sweep.json]

Deterministic (OCD and the QSS are); ~1 min on the M4.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from ocd_lap import qss_lap
from ocd_lap.aero_map import lift_warnings
from ocd_lap.backend import OCDBackend, StartPose, Inputs
from ocd_lap.car_setup import KNOB
from ocd_lap.manoeuvres import DT_CONTROL, STEPS, _speed_pedals
from ocd_lap.track import Track
from rlsetup.driving import setups
from rlsetup.provenance import run_provenance

TRACK, CAR = 'Catalunya', 'sf70h_tum'
BRAKES_NM = (12000., 15000.)
# one-at-a-time values around the base (PLACEHOLDER grid: ~1/6 .. 1/3 of each knob's range)
SWEEP = {
    'aero_level': (.3, .7, .9),
    'aero_front_share': (.42, .48),
    'ride_height_front_m': (.025, .045),
    'rake_m': (.03, .07),
    'brake_bias_front': (.46, .52, .55),
    'final_drive_ratio': (.95, 1.05, 1.1),
    'roll_stiffness_front_share': (.40, .50, .55),
    'roll_stiffness_Nm_per_rad': (2.0e5, 4.0e5),
    'lsd_lock_drive': (.5,),
    'lsd_preload_Nm': (100.,),
}
MAX_EXTEND = 2            # extra steps past the sweep edge per knob
MIN_GAIN_S = .005         # a change counts as an improvement above this [s]. PLACEHOLDER
HOLD_S = 3.               # ride-height check: constant-speed hold [s]


def lowers_car(setup, base):
    """SENSIBLE rule: True if the static front or rear ride height is below the base's. With
    p0 / p1_TUM the car already sits below the ground (negative ride height) at the QSS top
    speed, and the QSS gain of a lower car comes only from the PLACEHOLDER 2 %/cm aero
    sensitivity and a lower CoG in a region OCD does not model (no floor contact)."""
    rear = lambda s: s.ride_height_front_m + s.rake_m
    return setup.ride_height_front_m < base.ride_height_front_m - 1e-12 or rear(setup) < rear(base) - 1e-12


def ride_heights_at(setup, speed, brake):
    """Static ride heights [m] (front, rear) after HOLD_S s at constant `speed` [m/s] on a straight."""
    b = OCDBackend(powertrain='direct', max_drive_torque_Nm=8000., max_brake_torque_Nm=brake, car=CAR)
    st, integral = b.reset(setup, StartPose(0., 0., 0., speed)), 0.
    for _ in range(int(HOLD_S / DT_CONTROL)):
        err = speed - st['vx']
        integral = float(np.clip(integral + err * DT_CONTROL, -5., 5.))
        st = b.step(Inputs(0., *_speed_pedals(err, integral)), STEPS)
    t = b.get_telemetry()
    return float(t['ride_height_front_m']), float(t['ride_height_rear_m'])


def changes(setup, ref):
    a, b = setup.to_dict(), ref.to_dict()
    return {k: v for k, v in a.items() if v != b[k]}


def at_bound(setup):
    return [k for k, v in setup.to_dict().items()
            if v in (KNOB[k].low, KNOB[k].high) and KNOB[k].low != KNOB[k].default != KNOB[k].high]


class Evaluator:
    def __init__(self, track):
        self.track, self.rows, self.n_qss = track, [], 0

    def __call__(self, label, setup, brake, phase):
        t0 = time.perf_counter()
        ggv = qss_lap.measure_ggv(setup, car=CAR, brake_torque_Nm=brake)
        self.n_qss += 1
        kw = dict(ggv_data=ggv, car=CAR, ers=False, drs=False)
        safe = qss_lap.lap_time(setup, self.track, **kw)
        race = qss_lap.lap_time(setup, self.track, path=self.track.reference, **kw)
        b = OCDBackend(car=CAR, max_brake_torque_Nm=brake)
        b.reset(setup, StartPose(speed=0.))
        rh = ride_heights_at(setup, min(safe['v_max_mps'], 85.), brake)
        row = dict(label=label, phase=phase, brake_torque_Nm=brake, setup=setup.to_dict(),
                   changes_vs_p0=changes(setup, setups.get('p0')),
                   lap_safe_s=safe['lap_time_s'], lap_raceline_s=race['lap_time_s'],
                   v_max_mps=safe['v_max_mps'], v_min_mps=safe['v_min_mps'],
                   ggv=np.asarray(ggv[0]).tolist(), traction=np.asarray(ggv[1]).tolist(),
                   understeer_deg_per_g=[d['understeer_gradient_deg_per_g'] for d in ggv[2]],
                   spun=any(d['spun'] for d in ggv[2]),
                   setup_warnings=b.run_metadata().get('warnings', []), lift_warnings=lift_warnings(setup),
                   at_bound=at_bound(setup), ride_height_at_vmax_m=rh,
                   wall_s=time.perf_counter() - t0)
        self.rows.append(row)
        print(f'{phase:8s} {label:40s} {brake / 1e3:4.0f} kNm  safe {row["lap_safe_s"]:7.3f} s  '
              f'raceline {row["lap_raceline_s"]:7.3f} s  rh@vmax {rh[0] * 100:+.1f}/{rh[1] * 100:+.1f} cm'
              + ('  SPUN' if row['spun'] else '') + (f'  bound {row["at_bound"]}' if row['at_bound'] else ''),
              flush=True)
        return row


def main(out):
    track = Track(TRACK)
    ev = Evaluator(track)
    # 1. named setups x brake torque
    named = [ev(f'{n}', setups.get(n), b, 'named') for n in ('p0', 'p1_TUM') for b in BRAKES_NM]
    best_named = min(named, key=lambda r: r['lap_safe_s'])
    base_name, brake = best_named['label'], best_named['brake_torque_Nm']
    base = setups.get(base_name)
    t_base = best_named['lap_safe_s']
    # 2. one knob at a time
    single = {}
    for knob, values in SWEEP.items():
        for v in values:
            if v == getattr(base, knob):
                continue
            r = ev(f'{base_name} {knob}={v:g}', base.replace(**{knob: v}), brake, 'sweep')
            single.setdefault(knob, []).append((v, r['lap_safe_s']))
    # 3. extend past the edge of the swept values
    for knob, res in single.items():
        v_best, t_best = min(res, key=lambda x: x[1])
        grid = sorted({v for v, _ in res} | {getattr(base, knob)})   # swept values + the base value
        if t_best > t_base - MIN_GAIN_S or v_best not in (grid[0], grid[-1]):
            continue
        lo, hi = KNOB[knob].low, KNOB[knob].high
        step = v_best - (grid[-2] if v_best == grid[-1] else grid[1])   # outward, size of the last gap
        v, t_prev = v_best, t_best
        for _ in range(MAX_EXTEND):
            v = float(np.clip(v + step, lo, hi))
            if any(abs(v - x) < 1e-12 for x, _ in res):
                break
            r = ev(f'{base_name} {knob}={v:g}', base.replace(**{knob: v}), brake, 'extend')
            res.append((v, r['lap_safe_s']))
            if r['lap_safe_s'] > t_prev - MIN_GAIN_S or v in (lo, hi):
                break
            t_prev = r['lap_safe_s']
    # 4. greedy combination of the improving, sensible single-knob changes
    gains = []
    for knob, res in single.items():
        v_best, t_best = min(res, key=lambda x: x[1])
        cand = base.replace(**{knob: v_best})
        if t_base - t_best > MIN_GAIN_S and not lowers_car(cand, base):
            gains.append((t_base - t_best, knob, v_best))
    gains.sort(reverse=True)
    excluded = [dict(knob=k, value=min(r, key=lambda x: x[1])[0], gain_s=t_base - min(r, key=lambda x: x[1])[1])
                for k, r in single.items()
                if t_base - min(r, key=lambda x: x[1])[1] > MIN_GAIN_S
                and lowers_car(base.replace(**{k: min(r, key=lambda x: x[1])[0]}), base)]
    current, t_cur, kept = base, t_base, []
    for i, (gain, knob, v) in enumerate(gains):
        cand = current.replace(**{knob: v})
        if i == 0:   # the best single change is already evaluated
            r = next(x for x in ev.rows if x['setup'] == cand.to_dict() and x['brake_torque_Nm'] == brake)
        else:
            r = ev(f'combo +{knob}={v:g}', cand, brake, 'combo')
        if r['lap_safe_s'] < t_cur - MIN_GAIN_S:
            current, t_cur = cand, r['lap_safe_s']
            kept.append(dict(knob=knob, value=v, single_gain_s=gain, lap_after_s=t_cur))
    # 5. the other brake torque
    other = [b for b in BRAKES_NM if b != brake][0]
    r_other = ev('best (other brake)', current, other, 'brake')
    best_brake = other if r_other['lap_safe_s'] < t_cur - MIN_GAIN_S else brake
    result = dict(
        track=TRACK, car=CAR, ers=False, drs=False, track_length_m=float(track.length),
        raceline_length_m=float(track.reference[-1, 0]), n_qss_evaluations=ev.n_qss,
        base=dict(setup=base_name, brake_torque_Nm=brake, lap_safe_s=t_base), sweep=SWEEP,
        min_gain_s=MIN_GAIN_S, excluded_lowers_car=excluded, kept=kept,
        best=dict(setup=current.to_dict(), changes_vs_p0=changes(current, setups.get('p0')),
                  changes_vs_base=changes(current, base), brake_torque_Nm=best_brake,
                  lap_safe_s=t_cur if best_brake == brake else r_other['lap_safe_s']),
        rows=ev.rows, provenance=run_provenance(TRACK))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(f'\n{ev.n_qss} QSS evaluations; best: {result["best"]["changes_vs_p0"]} at '
          f'{best_brake / 1e3:g} kN m, safe line {result["best"]["lap_safe_s"]:.3f} s -> {out}')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'docs/results/bcn/qss_sweep.json'))
    main(p.parse_args().out)
