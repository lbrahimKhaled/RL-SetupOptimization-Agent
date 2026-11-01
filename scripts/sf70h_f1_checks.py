#!/usr/bin/env python3
"""Checks of the 'sf70h_f1' cars (ocd_first_lap branch sf70h-f1) against 'sf70h_tum'.

    .venv/bin/python scripts/sf70h_f1_checks.py [--ocd-root ../ocd_first_lap-f1] [PART ...]

PART (default all): params tyres qss_sha qss_bcn closed_loop. Results are merged into
docs/results/sf70h_f1/results.json under the part's key (other keys are kept), with provenance.

    params   old -> new value of every changed car parameter, read from the code (cars.py)
    tyres    MF5.2 pure-slip sweep (ocd_first_lap scripts/sf70h_checks.tyre_sweep: peak mu vs
             TUM's law, slip at the Fx / Fy peaks, Fx at the QSS 12 % slip limit / peak)
    qss_sha  QSS lap (qss_lap.measure_ggv with a 15 kN m brake ramp, ellipse exponent 2, ERS and DRS
             off) on Shanghai's safe line (Track.path) and TUM's raceline (Track.reference)
    qss_bcn  the same on Catalunya with bcn_best
    handling ramp steer at p1_TUM: peak a_y, understeer gradient fitted before the a_y peak
    closed_loop  summaries of the closed-loop TUM-lap runs in runs/sf70h_f1/closed_loop/<label>/
             (scripts/comparison/tum_closed_loop.py --out ...; run separately, commands in each runs.json)

Setups: 'sf70h_tum' runs the named setup itself (p1_TUM / bcn_best). The F1 cars run the named
setup moved onto their own defaults (rlsetup.driving.setups.for_car): heave on (1.0 / 0.25 MN/m
per wheel) and brake bias 0.58 ('car bias'); the variant 'named bias' uses the named setup's own
brake bias (p1_TUM 0.49, bcn_best 0.55) to separate the bias from the rest; 'car bias + heave
damping 0.3' adds heave_damping_ratio 0.3 (PLACEHOLDER diagnostic: the approved car keeps the knob
default 0, which leaves its stiff heave mode almost undamped and porpoising at high speed).
"""
import argparse
import datetime
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCD_ROOT = Path('/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1')
OUT = ROOT / 'docs/results/sf70h_f1/results.json'
BRAKE_NM = 15000.
F1_CARS = ('sf70h_f1', 'sf70h_f1_iz560')
PARTS = ('params', 'tyres', 'qss_sha', 'qss_bcn', 'handling', 'closed_loop')
CL_DIR = ROOT / 'runs/sf70h_f1/closed_loop'   # tum_closed_loop.py --out folders (one per label)
# Earlier result for the debug car (comparison/tum_ocd_debug_20261031_144747Z): constant aero, TUM r(v)
CL_DEBUG_CAR = {'1.00': 103.680, '0.96': 102.985}
HEAVE_DAMPING_DIAG = .3       # PLACEHOLDER heave_damping_ratio of the diagnostic setups (no F1 source)


def setup_paths(ocd_root):
    """Import ocd_lap from `ocd_root` (purging a copy imported from elsewhere)."""
    ocd_root = Path(ocd_root).resolve()
    sys.path.insert(0, str(ocd_root))
    sys.path.insert(1, str(ROOT))
    for name in [m for m in sys.modules if m == 'ocd_lap' or m.startswith('ocd_lap.')]:
        del sys.modules[name]
    import ocd_lap
    if Path(ocd_lap.__file__).resolve().parents[1] != ocd_root:
        raise RuntimeError(f'ocd_lap imported from {ocd_lap.__file__}, expected {ocd_root}')
    return ocd_root


def update(key, value, path=OUT, sub=None):
    """results.json[key] = value, or results.json[key][sub] = value (file-locked read-modify-write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = path.with_name('.results.lock')
    lock.touch(exist_ok=True)
    with open(lock) as fh:   # two parts may run in parallel
        fcntl.flock(fh, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        if sub is None:
            data[key] = value
        else:
            data.setdefault(key, {})[sub] = value
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(clean(data), indent=1) + '\n')
        tmp.replace(path)
        fcntl.flock(fh, fcntl.LOCK_UN)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def load_checks(ocd_root):
    import importlib.util
    spec = importlib.util.spec_from_file_location('sf70h_checks', Path(ocd_root) / 'scripts' / 'sf70h_checks.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def car_setups(name, car):
    """(label, Setup) pairs for one car and one named setup (module docstring)."""
    from rlsetup.driving import setups
    if car not in F1_CARS:
        return [('named', setups.get(name))]
    own = setups.for_car(name, car, keep=('brake_bias_front',))
    return [('car bias', own), ('named bias', setups.for_car(name, car)),
            (f'car bias + heave damping {HEAVE_DAMPING_DIAG}', own.replace(heave_damping_ratio=HEAVE_DAMPING_DIAG))]


def part_params():
    from ocd_lap import cars
    from ocd_lap.translator import P
    from ocd_lap.backend import OCDBackend, StartPose
    old, new = cars.get('sf70h_tum'), cars.get('sf70h_f1')
    tm = P + 'z.tire_model.'
    b_old, b_new = OCDBackend(car='sf70h_tum'), OCDBackend(car='sf70h_f1')
    tyre = {}
    for axle in ('front', 'rear'):
        for k in ('PKX1', 'PKX2', 'PKX3', 'PCX1', 'PCY1', 'PEX1', 'PEX2', 'PEX3', 'PEX4',
                  'PEY1', 'PEY2', 'PEY3', 'PEY4', 'PEY5', 'PKY1', 'PKY2'):
            key = tm + f'{axle}_left.{k}'
            tyre[f'{axle}.{k}'] = dict(old=b_old.base_params[key], new=b_new.base_params[key])
    s_old, s_new = cars.default_setup('sf70h_tum'), cars.default_setup('sf70h_f1')
    rates = {}
    for car, s in (('sf70h_tum', s_old), ('sf70h_f1', s_new)):
        b = OCDBackend(car=car)
        b.reset(s, StartPose(speed=0.))
        sp = P + 'suspension.heave.spring_stiffness_Npm.'
        rates[car] = dict(heave_per_wheel_front_Npm=b.params[sp + 'front'] / 2,
                          heave_per_wheel_rear_Npm=b.params[sp + 'rear'] / 2,
                          heave_enabled=b.params[P + 'suspension.heave.enabled'],
                          corner_spring_front_Npm=b.params[P + 'suspension.vehicle_spring_stiffness_Npm.front'],
                          corner_spring_rear_Npm=b.params[P + 'suspension.vehicle_spring_stiffness_Npm.rear'])
    knobs = {k.name: dict(old=dict(low=k0.low, high=k0.high, default=k0.default),
                          new=dict(low=k.low, high=k.high, default=k.default), note=k.note)
             for k0, k in zip(__import__('ocd_lap.car_setup', fromlist=['KNOBS']).KNOBS, cars.F1_KNOBS)
             if k0 != k}
    return dict(
        tyre_ocd=tyre, knobs=knobs, rates_at_default_setup=rates,
        steer_max_rad=dict(old=old.steer_max_rad, new=new.steer_max_rad),
        steer_rate_max_radps=dict(old=old.steer_rate_max_radps, new=new.steer_rate_max_radps),
        brake_torque_Nm=dict(old='12 kN m in the RL repo (backend default 360 N m)', new=new.brake_torque_Nm),
        footprint=dict(old=old.footprint, new=new.footprint),
        Iz_kgm2=dict(sf70h_tum=old.ocd[P + 'moment_of_inertia_kgpm2.z'],
                     sf70h_f1=new.ocd[P + 'moment_of_inertia_kgpm2.z'],
                     sf70h_f1_iz560=cars.get('sf70h_f1_iz560').ocd[P + 'moment_of_inertia_kgpm2.z']),
        f1_constants=cars.F1, pl14_tyre=cars.PL14_TYRE, pl14_QS=cars.pl14_stiffness_factor())


def part_tyres(ocd_root):
    from ocd_lap.backend import OCDBackend, StartPose
    checks = load_checks(ocd_root)
    out = {}
    for car in ('sf70h_tum', 'sf70h_f1'):
        b = OCDBackend(car=car)
        b.reset(None, StartPose(speed=0.))
        out[car] = checks.tyre_sweep(b.params)
    return out


def ggv_row_valid(d, g=9.81):
    """False when the settle at constant speed before the pedal ramps collapsed (porpoising, wheel
    lift, wheelspin; seen for the F1 car at 90 m/s with high downforce): traction peak below 0.9 v
    or zero, brake peak below 0.5 v or above 8 g. (Traction peaks above the 3.4 g that the direct
    8 kN m drive allows also occur for sf70h_tum at >= 75 m/s: dv/dt noise from body oscillation.
    They do not move the lap, where the engine limit is far lower; not flagged.)"""
    v = d['speed_mps']
    return bool(d['v_traction'] >= .9 * v and d['ax_traction'] > 0.
                and d['v_brake'] >= .5 * v and d['ax_brake'] < 8. * g)


def qss(track, setup, car, lines):
    """QSS laps with the car's g-g-v speeds (15-90 m/s) and, for a like-for-like comparison where
    a 90 m/s row is invalid, with 15-75 m/s (g-g-v held flat above 75 m/s)."""
    from ocd_lap import qss_lap
    t0 = time.time()
    row = dict(setup=asdict(setup), setup_class=type(setup).__name__)
    for tag, speeds in (('', None), ('_ggv75', (15., 30., 45., 60., 75.))):
        ggv = qss_lap.measure_ggv(setup, speeds, car=car, brake_torque_Nm=BRAKE_NM)
        row['ggv' + tag] = [dict(speed_mps=d['speed_mps'], ay_max_g=d['ay_max'] / 9.81, ax_brake_g=d['ax_brake'] / 9.81,
                                 v_brake_mps=d['v_brake'], ax_traction_g=d['ax_traction'] / 9.81,
                                 v_traction_mps=d['v_traction'], understeer_deg_per_g=d['understeer_gradient_deg_per_g'],
                                 spun=d['spun'], valid=ggv_row_valid(d)) for d in ggv[2]]
        for label, path in lines.items():
            r = qss_lap.lap_time(setup, track, car=car, ggv_data=ggv, ers=False, drs=False, path=path)
            row[label + tag] = dict(lap_time_s=r['lap_time_s'], v_max_kmh=3.6 * r['v_max_mps'],
                                    v_min_kmh=3.6 * r['v_min_mps'])
    row['all_ggv_rows_valid'] = all(d['valid'] for d in row['ggv'])
    row['wall_s'] = time.time() - t0
    return row


def part_qss(track_name, setup_name, raceline):
    from ocd_lap.track import Track
    track = Track(track_name)
    lines = {'safe_line': None}
    if raceline:
        lines['raceline'] = track.reference
    out = {}
    for car in ('sf70h_tum',) + F1_CARS:
        for label, setup in car_setups(setup_name, car):
            key = f'{car} | {setup_name} {label}'
            out[key] = qss(track, setup, car, lines)
            print(key, {k: round(v['lap_time_s'], 3) for k, v in out[key].items() if isinstance(v, dict) and 'lap_time_s' in v},
                  'ggv rows valid' if out[key]['all_ggv_rows_valid'] else
                  'INVALID ggv rows at ' + str([d['speed_mps'] for d in out[key]['ggv'] if not d['valid']]),
                  f"{out[key]['wall_s']:.0f} s", flush=True)
    return dict(track=track_name, setup=setup_name, brake_torque_Nm=BRAKE_NM, ers=False, drs=False,
                exponent='car default (2)', results=out)


def prepeak_understeer(ay, steer, vx, wheelbase):
    """Understeer gradient [deg/g] from a ramp steer, fitted like handling.ramp_steer (0.3 m/s^2 <
    a_y < 40 % of the peak, delta - L a_y / v^2 against a_y) but on samples BEFORE the a_y peak only.
    ramp_steer's own window also takes post-peak samples, which inflates the gradient for a tyre
    whose force falls steeply after its peak (sf70h_f1: 1-3.4 instead of 0.1-0.3 deg/g)."""
    import math
    ay = np.convolve(ay, np.ones(25) / 25, mode='same')
    i = int(np.argmax(ay[:-13]))
    lin = (ay > .3) & (ay < .4 * ay[i])
    lin[:5] = False
    lin[i:] = False
    x = ay[lin]
    return math.degrees(np.polyfit(x, steer[lin] - wheelbase * x / vx[lin] ** 2, 1)[0]) * 9.81


def part_handling(speeds=(30., 45., 60., 75.)):
    """Ramp steer (0.03 rad/s, as the QSS) at p1_TUM: peak a_y and the pre-peak understeer gradient,
    for sf70h_tum, sf70h_tum + only the F1 tyre, sf70h_f1 with heave off, sf70h_f1, sf70h_f1_iz560."""
    from ocd_lap import cars, handling
    from ocd_lap.translator import P
    from rlsetup.driving import setups
    p1 = setups.get('p1_TUM')
    f1 = setups.for_car('p1_TUM', 'sf70h_f1', keep=('brake_bias_front',))
    cases = {'sf70h_tum': ('sf70h_tum', p1, None),
             'sf70h_tum + F1 tyre only': ('sf70h_tum', p1, cars.f1_tyre_overrides()),
             'sf70h_f1 heave off': ('sf70h_f1', f1.replace(heave_freq_front_hz=0., heave_freq_rear_hz=0.), None),
             'sf70h_f1': ('sf70h_f1', f1, None), 'sf70h_f1_iz560': ('sf70h_f1_iz560', f1, None)}
    out = {}
    for label, (car, s, ov) in cases.items():
        rows = []
        for v in speeds:
            r = handling._drive(s, v, lambda t: .03 * max(0., t - 1.5), 1.5 + .4 / .03, overrides=ov,
                                sample_every=5, car=car)
            ref = handling.ramp_steer(s, v, rate_radps=.03, car=car, overrides=ov)
            rows.append(dict(speed_mps=v, max_ay_g=ref['max_ay_mps2'] / 9.81, steer_at_max_deg=ref['steer_at_max_deg'],
                             understeer_prepeak_deg_per_g=prepeak_understeer(
                                 r['ay'], r['steer'], r['vx'], float(r['params'][P + 'wheelbase_m'])),
                             understeer_ramp_steer_fit_deg_per_g=ref['understeer_gradient_deg_per_g'],
                             spun=ref['spun']))
        out[label] = rows
        print(label, [(r['speed_mps'], round(r['understeer_prepeak_deg_per_g'], 2), round(r['max_ay_g'], 2)) for r in rows],
              flush=True)
    return out


def part_closed_loop(cl_dir=CL_DIR):
    """Summaries of the closed-loop TUM-lap runs (scripts/comparison/tum_closed_loop.py) in cl_dir."""
    out = dict(reference_debug_car_lap2_s=CL_DEBUG_CAR, runs={})
    for d in sorted(p for p in Path(cl_dir).iterdir() if (p / 'runs.json').exists()):
        r = json.loads((d / 'runs.json').read_text())
        an = json.loads((d / 'analysis.json').read_text()) if (d / 'analysis.json').exists() else {}
        meta = json.loads(next(d.glob('cl_f*.meta.json')).read_text())
        row = dict(car=meta['conditions']['car'], setup_class=meta['conditions'].get('setup_class'),
                   setup=r['setup'], gains=r['gains'], command=r['command'], per_f={})
        for x in r['runs']:
            a = an.get(f"{x['f']:.2f}") or {}
            row['per_f'][f"{x['f']:.2f}"] = dict(
                termination=x['termination'], lap_times_s=x['lap_times_s'], lap2_s=x['lap2_s'],
                spin_at_s_raceline_m=x['spin_at_s_raceline_m'], spin_lap=x['spin_lap'],
                cog_on_track_lap2=x['on_track_cog_fraction_lap2'],
                footprint_on_track_lap2=x['on_track_footprint_fraction_lap2'],
                max_abs_e_lap2_m=x['max_abs_e_raceline_lap2_m'], rms_e_lap2_m=x['rms_e_raceline_lap2_m'],
                v_max_lap2_kmh=x['v_max_lap2_kmh'], loss_by_limit_s=a.get('loss_by_limit_s'))
        for k in ('tyre_check', 'lateral_check', 'qss_debug_car'):
            if (d / f'{k}.json').exists():
                row[k] = json.loads((d / f'{k}.json').read_text())
        out['runs'][d.name] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ocd-root', default=str(DEFAULT_OCD_ROOT))
    ap.add_argument('parts', nargs='*', default=list(PARTS))
    args = ap.parse_args()
    ocd_root = setup_paths(args.ocd_root)
    from rlsetup.provenance import git_state
    prov = dict(ocd_first_lap=git_state(ocd_root), rl_repo=git_state(ROOT),
                script=str(Path(__file__).relative_to(ROOT)), command=' '.join(sys.argv),
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for part in args.parts:
        t0 = time.time()
        if part == 'params':
            res = part_params()
        elif part == 'tyres':
            res = part_tyres(ocd_root)
        elif part == 'qss_sha':
            res = part_qss('Shanghai', 'p1_TUM', raceline=True)
        elif part == 'qss_bcn':
            res = part_qss('Catalunya', 'bcn_best', raceline=True)
        elif part == 'handling':
            res = part_handling()
        elif part == 'closed_loop':
            res = part_closed_loop()
        else:
            raise SystemExit(f'unknown part {part!r} (choose from {PARTS})')
        res['provenance'] = {**prov, 'wall_s': time.time() - t0}
        update(part, res)
        print('wrote', part, f'({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
