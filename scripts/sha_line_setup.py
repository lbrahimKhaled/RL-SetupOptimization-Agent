#!/usr/bin/env python3
"""Shanghai, car sf70h_f1: how much of the safe-line QSS gap (102.5 s) to TUM's raceline (94.7 s)
is the line and how much is the setup. Needs the f1 simulator worktree:

    PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 OMP_NUM_THREADS=1 \
        .venv/bin/python scripts/sha_line_setup.py lines    # part 1, ~1 min
    ... scripts/sha_line_setup.py sweep [--line w2.6]       # part 2, <= 40 g-g-v measurements

lines  TUM's minimum-curvature optimiser, called exactly as ocd_lap/track.py does
       (tph.opt_min_curv on the racetrack-database centre line, kappa_bound 0.2, create_raceline
       at 2 m), for several vehicle-width allowances w_veh [m], once as a single QP (track.py) and
       once iterated (tph.iqp_handler, 'iqp_w*'). opt_min_curv keeps the line
       w_veh / 2 from each edge (point constraint at the spline knots only). Per line: footprint
       fit (Track.contains_car with the car's footprint, body yaw = line tangent, i.e. zero body
       slip; every 0.5 m), signed min edge clearance of the footprint [m] (negative = outside),
       length, curvature, QSS lap (p1_TUM_f1, 15 kN m, ERS/DRS off) from one shared g-g-v.
       Also TUM's raceline (Track.reference) and the PL14-style OCP line of sf70h_f1
       (docs/results/sf70h_f1/ocp_solution_sha_f1_0.5m.csv, x_car / y_car). Lines are written to
       docs/results/shanghai_line_setup/line_<tag>.csv (x_m, y_m; ReferenceLine.from_csv reads it).
sweep  scripts/bcn_setup_sweep.py's procedure (one knob at a time, extend at the edge, greedy
       combination, SENSIBLE rule: no lowering of the car) for sf70h_f1 around p1_TUM_f1 on
       the chosen line, with the car's own knob ranges. Each evaluation = one g-g-v; the laps on
       every line come from the same g-g-v. Collapse check: g-g-v rows whose settle collapsed
       (sf70h_f1_checks.ggv_row_valid) are dropped (g-g-v held flat above the last valid speed)
       and flagged; the lap with rows to 75 m/s only is recorded for every setup as well.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import trajectory_planning_helpers as tph
from ocd_lap import cars, qss_lap
from ocd_lap.backend import OCDBackend, StartPose, Inputs
from ocd_lap.manoeuvres import DT_CONTROL, STEPS, _speed_pedals
from ocd_lap.track import Track, PLANNING, closed_path, ROOT as OCD_ROOT
from rlsetup.driving import setups
from rlsetup.provenance import run_provenance
from bcn_setup_sweep import changes, lowers_car
from sf70h_f1_checks import ggv_row_valid

TRACK, CAR, SETUP, BRAKE_NM = 'Shanghai', 'sf70h_f1', 'p1_TUM_f1', 15000.
OUT = ROOT / 'docs/results/shanghai_line_setup'
WIDTHS_M = (4.5, 3.5, 3.0, 2.6)          # w_veh allowances, single QP (2.6 = 2.07 m car + ~0.25 m/side)
WIDTHS_IQP_M = (4.5, 4.0, 3.5, 3.0, 2.6)  # the same, iterated QP
OCP_LINE = ROOT / 'docs/results/sf70h_f1/ocp_solution_sha_f1_0.5m.csv'
CHECK_STEP_M = .5
FIT_MIN = .995                                       # "fits the footprint" threshold (task spec)


# ------------------------------------------------------------------------------------- lines
def min_curv_line(w_veh, track, iterate=False):
    """track.py's optimiser call with width allowance `w_veh` [m]; closed (s, x, y). iterate:
    the same QP repeated on the updated line until the linearised curvature is right
    (tph.iqp_handler, as TUM's global_racetrajectory_optimization 'mincurv_iqp': iters_min 3,
    curvature error <= 0.01 rad/m, re-interpolation every 3 m); the single QP linearises the
    curvature around the centre line and stays far from the minimum on Shanghai."""
    center = np.loadtxt(OCD_ROOT / 'vendor/racetrack-database/tracks' / f'{TRACK}.csv', delimiter=',')
    cx, cy, A, normals = tph.calc_splines.calc_splines(track.center[:, 1:])
    ref = center
    if iterate:
        n = len(center)
        lengths = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=cx, coeffs_y=cy)
        psi, kappa, dkappa = tph.calc_head_curv_an.calc_head_curv_an(
            coeffs_x=cx, coeffs_y=cy, ind_spls=np.arange(n), t_spls=np.zeros(n), calc_dcurv=True)
        alpha, ref, normals = tph.iqp_handler.iqp_handler(
            reftrack=center.copy(), normvectors=normals, A=A, spline_len=lengths, psi=psi, kappa=kappa,
            dkappa=dkappa, kappa_bound=PLANNING['kappa_bound'], w_veh=w_veh, print_debug=False,
            plot_debug=False, stepsize_interp=3., iters_min=3, curv_error_allowed=.01)[:3]
    else:
        alpha, _ = tph.opt_min_curv.opt_min_curv(center, normals, A, kappa_bound=PLANNING['kappa_bound'],
                                                w_veh=w_veh)
    xy = tph.create_raceline.create_raceline(ref[:, :2], normals, alpha,
                                             stepsize_interp=PLANNING['spacing_m'])[0]
    return closed_path(xy)


def footprint_check(path, track, footprint, step=CHECK_STEP_M):
    """Footprint along `path` every `step` m, yaw = tangent. Returns fraction inside, signed
    min clearance [m] (footprint to the nearest edge; negative = depth of the worst corner
    outside), CoG-point min clearance [m], and the outside stretches [(s_start, s_end, depth)]."""
    s = np.arange(0., path[-1, 0], step)
    xy = np.column_stack([np.interp(s, path[:, 0], path[:, i]) for i in (1, 2)])
    d = np.column_stack([np.interp((s + .5) % path[-1, 0], path[:, 0], path[:, i])
                         - np.interp((s - .5) % path[-1, 0], path[:, 0], path[:, i]) for i in (1, 2)])
    yaw = np.arctan2(d[:, 1], d[:, 0])
    edge, corners = track.polygon.boundary, np.asarray(footprint)
    inside, clear, clear_cog = np.zeros(len(s), bool), np.zeros(len(s)), np.zeros(len(s))
    for i, ((x, y), psi) in enumerate(zip(xy, yaw)):
        c, sn = np.cos(psi), np.sin(psi)
        pts = corners @ np.array([[c, sn], [-sn, c]]) + [x, y]
        poly = Polygon(pts)
        inside[i] = track.safe.covers(poly)
        clear[i] = edge.distance(poly) if inside[i] else -max(
            track.polygon.distance(Point(p)) for p in pts)
        clear_cog[i] = edge.distance(Point(x, y)) * (1 if track.safe.covers(Point(x, y)) else -1)
    out, stretches, start = ~inside, [], None
    for i in range(len(s) + 1):
        o = out[i] if i < len(s) else False
        if o and start is None:
            start = i
        elif not o and start is not None:
            stretches.append((float(s[start]), float(s[i - 1]), float(clear[start:i].min())))
            start = None
    return dict(fit_fraction=float(inside.mean()), min_clearance_m=float(clear.min()),
                min_cog_clearance_m=float(clear_cog.min()), n_outside_stretches=len(stretches),
                outside_length_m=float(out.sum() * step), outside_stretches=stretches)


def curvature_stats(path):
    kappa, el = qss_lap.path_curvature(path)
    return dict(length_m=float(path[-1, 0]), kappa_abs_max=float(np.abs(kappa).max()),
                kappa_abs_mean=float(np.average(np.abs(kappa), weights=el)),
                kappa_sq_integral=float((kappa ** 2 * el).sum()))


def qss(setup, track, lines, ggv):
    return {k: qss_lap.lap_time(setup, track, car=CAR, ggv_data=ggv, ers=False, drs=False,
                                path=p)['lap_time_s'] for k, p in lines.items()}


def all_lines(track):
    """{tag: closed (s, x, y)}: safe line, min-curvature lines per width (single / iterated QP),
    raceline, OCP line (every 4th point = 2 m)."""
    lines = {'safe (Track.path)': track.path}
    for w in WIDTHS_M:
        lines[f'w{w:g}'] = min_curv_line(w, track)
    for w in WIDTHS_IQP_M:
        lines[f'iqp_w{w:g}'] = min_curv_line(w, track, iterate=True)
    lines['raceline (Track.reference)'] = track.reference
    ocp = np.genfromtxt(OCP_LINE, delimiter=',', names=True)
    lines['ocp_sf70h_f1'] = closed_path(np.column_stack([ocp['x_car'], ocp['y_car']])[::4])  # 2 m (0.5 m: noisy curvature)
    return lines


def part_lines():
    track, spec = Track(TRACK), cars.get(CAR)
    setup = setups.get(SETUP)
    t0 = time.perf_counter()
    ggv = qss_lap.measure_ggv(setup, car=CAR, brake_torque_Nm=BRAKE_NM)
    t_ggv = time.perf_counter() - t0
    lines = all_lines(track)
    rebuilt = lines['w4.5']
    safe_diff = (float(np.abs(rebuilt - track.path).max()) if rebuilt.shape == track.path.shape
                 else float('nan'))
    laps = qss(setup, track, lines, ggv)
    rows = {}
    for tag, path in lines.items():
        fp = footprint_check(path, track, spec.footprint)
        rows[tag] = dict(qss_lap_s=laps[tag], **curvature_stats(path), **fp)
        print(f'{tag:28s} QSS {laps[tag]:7.3f} s  fit {fp["fit_fraction"] * 100:6.2f} %  '
              f'min clear {fp["min_clearance_m"]:+.2f} m (CoG {fp["min_cog_clearance_m"]:+.2f})  '
              f'L {path[-1, 0]:7.1f} m  |k|max {rows[tag]["kappa_abs_max"]:.4f}  '
              f'int k^2 {rows[tag]["kappa_sq_integral"]:.4f}  outside {fp["n_outside_stretches"]} '
              f'stretches {fp["outside_length_m"]:.0f} m', flush=True)
        if tag.startswith(('w', 'iqp')):
            np.savetxt(OUT / f'line_{tag}.csv', path[:-1, 1:], delimiter=',', header='x_m,y_m',
                       fmt='%.6f')
    res = dict(track=TRACK, car=CAR, setup=SETUP, setup_values=setup.to_dict(), brake_torque_Nm=BRAKE_NM,
               ers=False, drs=False, footprint=[list(p) for p in spec.footprint], check_step_m=CHECK_STEP_M,
               planning=PLANNING, rebuilt_w4_5_vs_track_path_max_abs_diff=safe_diff,
               ggv_rows_valid=[ggv_row_valid(d) for d in ggv[2]], ggv_wall_s=t_ggv,
               lines=rows, provenance=run_provenance(TRACK))
    (OUT / 'lines.json').write_text(json.dumps(res, indent=1, default=float))
    print(f'rebuilt w4.5 vs Track.path max |diff| {safe_diff:.3g}; g-g-v {t_ggv:.1f} s -> {OUT / "lines.json"}')
    return res


# ------------------------------------------------------------------------------------- sweep
# one-at-a-time values around p1_TUM_f1 (PLACEHOLDER grid, as bcn_setup_sweep.SWEEP; heave / ride
# frequencies added because the F1 car has heave springs)
SWEEP = {
    'aero_level': (.7, .9, 1.),
    'aero_front_share': (.42, .48),
    'ride_height_front_m': (.025, .045),
    'rake_m': (.03, .07),
    'brake_bias_front': (.54, .62),
    'final_drive_ratio': (.95, 1.05),
    'roll_stiffness_front_share': (.40, .50),
    'roll_stiffness_Nm_per_rad': (2.0e5, 4.0e5),
    'heave_freq_front_hz': (10., 15.),
    'heave_freq_rear_hz': (4.5, 7.),
    'heave_damping_ratio': (.5,),
    'ride_freq_front_hz': (5.,),
    'ride_freq_rear_hz': (4.,),
    'lsd_lock_drive': (.5,),
    'lsd_preload_Nm': (100.,),
}
MAX_EXTEND = 2            # extra steps past the sweep edge per knob
MIN_GAIN_S = .005         # improvement threshold [s]. PLACEHOLDER (bcn_setup_sweep)
MAX_EVALS = 40            # g-g-v measurements (task budget)
HOLD_S, HOLD_SPEED = 3., 90.   # ride-height / porpoising check: constant-speed hold [s] at [m/s]
                               # (90 = top g-g-v speed ~ Shanghai v_max; bcn_setup_sweep used 85)


def ride_check(setup):
    """Constant HOLD_SPEED m/s for HOLD_S s on a straight (bcn_setup_sweep.ride_heights_at with
    the samples kept): mean front / rear ride height [m] and front std [m] over the last second
    (porpoising: sf70h_f1 with heave damping 0 had ~10 mm std at 90 m/s, SF70H_F1_UPGRADE.md §3)."""
    b = OCDBackend(powertrain='direct', max_drive_torque_Nm=8000., max_brake_torque_Nm=BRAKE_NM, car=CAR)
    st, integral, front, rear = b.reset(setup, StartPose(0., 0., 0., HOLD_SPEED)), 0., [], []
    n = int(HOLD_S / DT_CONTROL)
    for k in range(n):
        err = HOLD_SPEED - st['vx']
        integral = float(np.clip(integral + err * DT_CONTROL, -5., 5.))
        st = b.step(Inputs(0., *_speed_pedals(err, integral)), STEPS)
        if k >= n - int(1. / DT_CONTROL):
            t = b.get_telemetry()
            front.append(t['ride_height_front_m']); rear.append(t['ride_height_rear_m'])
    return dict(front_m=float(np.mean(front)), rear_m=float(np.mean(rear)), front_std_m=float(np.std(front)),
                front_min_m=float(np.min(front)))


def at_bound(setup):
    """Knobs at a bound of the car's own range (that are not pinned there by default)."""
    return [k.name for k in type(setup).KNOBS
            if getattr(setup, k.name) in (k.low, k.high) and k.low != k.default != k.high]


def load_line(tag, track):
    if tag in ('safe (Track.path)', 'safe'):
        return track.path
    if tag.startswith('raceline'):
        return track.reference
    return closed_path(np.loadtxt(OUT / f'line_{tag}.csv', delimiter=','))


class Evaluator:
    """One g-g-v per setup; QSS laps on every line of `lines` from it. Rows whose settle collapsed
    (ggv_row_valid False) are dropped from the g-g-v (held flat above the last valid speed)."""

    def __init__(self, track, lines, target):
        self.track, self.lines, self.target, self.rows = track, lines, target, []

    def __call__(self, label, setup, phase):
        if len(self.rows) >= MAX_EVALS:
            raise RuntimeError('evaluation budget used up')
        t0 = time.perf_counter()
        ggv, trac, det = qss_lap.measure_ggv(setup, car=CAR, brake_torque_Nm=BRAKE_NM)
        valid = np.array([ggv_row_valid(d) for d in det])
        used = (ggv[valid], trac[valid], [d for d, v in zip(det, valid) if v])
        laps = qss(setup, self.track, self.lines, used)
        n75 = int(np.sum(ggv[:, 0] <= 75.))
        lap75 = qss(setup, self.track, {self.target: self.lines[self.target]},
                    (ggv[:n75], trac[:n75], det[:n75]))[self.target]
        rh = ride_check(setup)
        row = dict(label=label, phase=phase, setup=setup.to_dict(), changes_vs_base=changes(setup, self.base),
                   laps_s=laps, lap_s=laps[self.target], lap_ggv75_s=lap75,
                   ggv_rows_valid=valid.tolist(), porpoise_or_collapse=not valid.all(),
                   ggv=ggv.tolist(), traction=trac.tolist(),
                   understeer_deg_per_g=[d['understeer_gradient_deg_per_g'] for d in det],
                   spun=any(d['spun'] for d in det), at_bound=at_bound(setup), ride_hold=rh,
                   below_ground=rh['front_min_m'] < 0. or rh['rear_m'] < 0.,
                   wall_s=time.perf_counter() - t0)
        self.rows.append(row)
        print(f'{phase:7s} {label:42s} {self.target} {row["lap_s"]:7.3f} s (ggv75 {lap75:7.3f})  '
              + '  '.join(f'{k.split()[0]} {v:7.3f}' for k, v in laps.items() if k != self.target)
              + f'  rh@{HOLD_SPEED:g} {rh["front_m"] * 100:+.1f}/{rh["rear_m"] * 100:+.1f} cm '
              f'(std {rh["front_std_m"] * 1e3:.1f} mm)'
              + ('  INVALID-GGV' if row['porpoise_or_collapse'] else '') + ('  SPUN' if row['spun'] else '')
              + (f'  bound {row["at_bound"]}' if row['at_bound'] else ''), flush=True)
        return row


def admissible(row, cand, base):
    """SENSIBLE rule of bcn_setup_sweep (no lowering of the car) + every g-g-v row valid (no settle
    collapse up to 90 m/s) + car above the ground in the 90 m/s hold (OCD has no floor contact).
    A ramp-steer spin (|beta| > 15 deg at the end of the ramp = limit oversteer) is flagged, not
    excluded (bcn_best has it too); the understeer gradients are recorded."""
    return not lowers_car(cand, base) and not row['porpoise_or_collapse'] and not row['below_ground']


def part_sweep(line):
    track = Track(TRACK)
    res_lines = json.loads((OUT / 'lines.json').read_text())['lines']
    if line is None:   # fastest line that fits
        line = min((t for t, r in res_lines.items() if r['fit_fraction'] >= FIT_MIN),
                   key=lambda t: res_lines[t]['qss_lap_s'])
    tags = [line] + [t for t in ('iqp_w3.5', 'safe (Track.path)', 'raceline (Track.reference)') if t != line]
    lines = {t: load_line(t, track) for t in tags}
    base = setups.get(SETUP)
    ev = Evaluator(track, lines, line)
    ev.base = base
    t_base = ev(SETUP, base, 'base')['lap_s']
    single = {}
    for knob, values in SWEEP.items():                                   # 1. one knob at a time
        for v in values:
            if v != getattr(base, knob):
                r = ev(f'{knob}={v:g}', base.replace(**{knob: v}), 'sweep')
                single.setdefault(knob, []).append((v, r['lap_s'], r))
    for knob, res in single.items():                                     # 2. extend at the edge
        v_best, t_best, _ = min(res, key=lambda x: x[1])
        grid = sorted({x[0] for x in res} | {getattr(base, knob)})
        if t_best > t_base - MIN_GAIN_S or v_best not in (grid[0], grid[-1]):
            continue
        k = type(base).KNOBS[[q.name for q in type(base).KNOBS].index(knob)]
        step = v_best - (grid[-2] if v_best == grid[-1] else grid[1])
        v, t_prev = v_best, t_best
        for _ in range(MAX_EXTEND):
            v = float(np.clip(v + step, k.low, k.high))
            if any(abs(v - x[0]) < 1e-12 for x in res) or len(ev.rows) >= MAX_EVALS - 8:
                break
            r = ev(f'{knob}={v:g}', base.replace(**{knob: v}), 'extend')
            res.append((v, r['lap_s'], r))
            if r['lap_s'] > t_prev - MIN_GAIN_S or v in (k.low, k.high):
                break
            t_prev = r['lap_s']
    gains, excluded = [], []                                             # 3. greedy combination
    for knob, res in single.items():
        ok = [x for x in res if admissible(x[2], base.replace(**{knob: x[0]}), base)]
        best_all = min(res, key=lambda x: x[1])
        if t_base - best_all[1] > MIN_GAIN_S and (not ok or min(ok, key=lambda x: x[1]) is not best_all):
            excluded.append(dict(knob=knob, value=best_all[0], gain_s=t_base - best_all[1],
                                 why='lowers the car' if lowers_car(base.replace(**{knob: best_all[0]}), base)
                                 else 'invalid g-g-v row (settle collapse) or below ground at 90 m/s'))
        if ok:
            v, t, _ = min(ok, key=lambda x: x[1])
            if t_base - t > MIN_GAIN_S:
                gains.append((t_base - t, knob, v))
    gains.sort(reverse=True)
    current, t_cur, kept, cur_row = base, t_base, [], ev.rows[0]
    for i, (gain, knob, v) in enumerate(gains):
        cand = current.replace(**{knob: v})
        if i == 0:
            r = next(x for x in ev.rows if x['setup'] == cand.to_dict())
        elif len(ev.rows) >= MAX_EVALS:
            break
        else:
            r = ev(f'combo +{knob}={v:g}', cand, 'combo')
        if r['lap_s'] < t_cur - MIN_GAIN_S and admissible(r, cand, base):
            current, t_cur, cur_row = cand, r['lap_s'], r
            kept.append(dict(knob=knob, value=v, single_gain_s=gain, lap_after_s=t_cur))
    for x in excluded:   # 4. probe: halfway to an excluded value (invalid g-g-v) on top of the combination
        if x['why'] == 'lowers the car' or len(ev.rows) >= MAX_EVALS:
            continue
        v = .5 * (getattr(current, x['knob']) + x['value'])
        cand = current.replace(**{x['knob']: v})
        r = ev(f'probe +{x["knob"]}={v:g}', cand, 'probe')
        if r['lap_s'] < t_cur - MIN_GAIN_S and admissible(r, cand, base):
            current, t_cur, cur_row = cand, r['lap_s'], r
            kept.append(dict(knob=x['knob'], value=v, single_gain_s=None, lap_after_s=t_cur))
    result = dict(track=TRACK, car=CAR, base=SETUP, brake_torque_Nm=BRAKE_NM, ers=False, drs=False,
                  target_line=line, lines=tags, sweep=SWEEP, min_gain_s=MIN_GAIN_S,
                  n_ggv_evaluations=len(ev.rows), base_laps_s=ev.rows[0]['laps_s'], kept=kept,
                  excluded=excluded,
                  best=dict(setup=current.to_dict(), changes_vs_base=changes(current, base),
                            laps_s=cur_row['laps_s'], lap_ggv75_s=cur_row['lap_ggv75_s'],
                            at_bound=cur_row['at_bound'], ride_hold=cur_row['ride_hold'],
                            understeer_deg_per_g=cur_row['understeer_deg_per_g']),
                  rows=ev.rows, provenance=run_provenance(TRACK))
    (OUT / 'sweep.json').write_text(json.dumps(result, indent=1, default=float))
    print(f'\n{len(ev.rows)} g-g-v evaluations; best {result["best"]["changes_vs_base"]}: '
          + ', '.join(f'{k} {v:.3f}' for k, v in cur_row['laps_s'].items()) + f' -> {OUT / "sweep.json"}')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('part', choices=('lines', 'sweep'))
    p.add_argument('--line', default=None, help='sweep: line tag from lines.json (default: fastest that fits)')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    part_lines() if a.part == 'lines' else part_sweep(a.line)
