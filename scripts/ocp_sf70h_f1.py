"""PL14-style minimum-lap-time OCP with the sf70h_f1 mapping (rlsetup.ocp.pl14_model.sf70h_f1_car).

Usage (2 threads; at most 2 of these processes at a time):
  OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python scripts/ocp_sf70h_f1.py CASE [CASE ...]
CASE: sha_f1 sha_f1_iz560 sha_f1_d06 sha_f1_hw09 sha_f1_alp bcn_tum bcn_f1 bcn_f1_iz560
  sha_*  Shanghai; bcn_* Catalunya. f1 = sf70h_f1_car (|delta| <= 0.30 rad), iz560 = the same with
  Iz 560.3 kg m^2, d06 = sf70h_f1_car with the old |delta| <= 0.6 bound (diagnostic: does the lock
  bind?), hw09 = sf70h_f1_car with the old 0.9 m car half-width (corridor), alp = with PL14's lateral
  peak-slip coefficients 9 / 8 deg (diagnostics: which mapped value moves the lap), bcn_tum = the old sf70h_tum mapping on Catalunya (beta 0.60, |delta| <= 0.6), the baseline
  for bcn_f1 (Shanghai's baseline is limebeer_ocp sha_sf70h_homotopy: 94.2458 s 1 m, 94.2491 s 0.5 m).
Route per case, 1 m nominal, speed-adapted mesh (h = ds clip((u/30)^2, 0.1, 1), u_min 3 m/s): if the warm
start's car differs from the target only in Iz or the steer bound (*_iz560, sha_f1_d06), a direct solve;
otherwise (and if the direct solve fails) an adaptive parameter homotopy from the warm start's car (lambda
step 0.25, halved after a failure, x1.5 after a success), the steer bound blended with it. (Direct solves of sha_f1 / bcn_tum from the old cars ran
> 15 CPU min without converging, as the earlier SF70H OCP.) Then 0.5 m from the 1 m solution. Warm starts:
  sha_f1   limebeer_ocp/_warm_sha_sf70h_homotopy.npz (sf70h_tum mapping, 1 m adaptive)
  bcn_tum  limebeer_ocp/_warm_bcn_beta060_adapt.npz (PL14 car, Catalunya, adaptive)
  *_iz560, sha_f1_d06 <- sha_f1 / bcn_f1 1 m; bcn_f1 <- bcn_tum 1 m.
Results: docs/results/sf70h_f1/results.json key 'ocp' (per case), 0.5 m solution CSVs in
docs/results/sf70h_f1/; warm starts in runs/sf70h_f1/ocp/ (not tracked).
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from rlsetup.ocp.pl14_model import (F1_IZ_PL14_KGM2, F1_STEER_MAX_RAD, pl14_car, sf70h_f1_car,  # noqa: E402
                                    sf70h_tum_car)
from rlsetup.ocp.solve import SolveSettings, solve  # noqa: E402
import ocp_pl14  # noqa: E402  (summary, write_csv, track)
from sf70h_f1_checks import update  # noqa: E402

LIME = ROOT / 'docs/results/limebeer_ocp'
OUT = ROOT / 'docs/results/sf70h_f1'
WARM = ROOT / 'runs/sf70h_f1/ocp'
ADAPT = dict(mesh_u_ref=30.0, u_min=3.0)
OLD_DELTA = 0.6


def blend(a, b, lam):
    """Linear blend of two CarParams (floats and nested tyre dataclasses); strings from b."""
    kw = {}
    for f_ in fields(a):
        va, vb = getattr(a, f_.name), getattr(b, f_.name)
        if isinstance(va, float):
            kw[f_.name] = (1 - lam) * va + lam * vb
        elif hasattr(va, '__dataclass_fields__'):
            kw[f_.name] = blend(va, vb, lam)
    return replace(b, **kw)


def stats(sol):
    d = sol.state('delta')
    st = sol.settings
    return dict(ocp_pl14.summary(sol), delta_max_abs_rad=float(np.abs(d).max()),
                delta_bound_rad=st['delta_max'],
                nodes_at_delta_bound=int(np.sum(np.abs(d) > st['delta_max'] - 1e-3)),
                beta_sideslip_max_deg=float(np.degrees(np.abs(np.arctan2(sol.state('v'), sol.state('u'))).max())))


def case_def(case):
    """(track, target car, delta bound, warm npz, source car for the homotopy, source delta bound,
    direct first?)."""
    f1, f1_iz = sf70h_f1_car(), sf70h_f1_car(Iz=F1_IZ_PL14_KGM2)
    tum = sf70h_tum_car(beta=0.60)
    return {
        'sha_f1': ('Shanghai', f1, F1_STEER_MAX_RAD, LIME / '_warm_sha_sf70h_homotopy.npz', tum, OLD_DELTA, False),
        'sha_f1_iz560': ('Shanghai', f1_iz, F1_STEER_MAX_RAD, WARM / 'sha_f1_1m.npz', f1, F1_STEER_MAX_RAD, True),
        'sha_f1_d06': ('Shanghai', f1, OLD_DELTA, WARM / 'sha_f1_1m.npz', f1, F1_STEER_MAX_RAD, True),
        'sha_f1_hw09': ('Shanghai', replace(f1, half_width=0.9), F1_STEER_MAX_RAD, WARM / 'sha_f1_1m.npz', f1,
                        F1_STEER_MAX_RAD, True),
        'sha_f1_alp': ('Shanghai', replace(f1, tyre_f=replace(f1.tyre_f, alp1=tum.tyre_f.alp1, alp2=tum.tyre_f.alp2),
                                           tyre_r=replace(f1.tyre_r, alp1=tum.tyre_r.alp1, alp2=tum.tyre_r.alp2)),
                       F1_STEER_MAX_RAD, WARM / 'sha_f1_1m.npz', f1, F1_STEER_MAX_RAD, True),
        'bcn_tum': ('Catalunya', tum, OLD_DELTA, LIME / '_warm_bcn_beta060_adapt.npz', pl14_car(beta=0.60), OLD_DELTA,
                    False),
        'bcn_f1': ('Catalunya', f1, F1_STEER_MAX_RAD, WARM / 'bcn_tum_1m.npz', tum, OLD_DELTA, False),
        'bcn_f1_iz560': ('Catalunya', f1_iz, F1_STEER_MAX_RAD, WARM / 'bcn_f1_1m.npz', f1, F1_STEER_MAX_RAD, True),
    }[case]


def run(case):
    t0 = time.time()
    tname, car, dmax, warm, src, src_dmax, direct = case_def(case)
    tr = ocp_pl14.track(tname)
    w = np.load(warm)
    W0, s0 = w['W'], w['s']
    st1 = SolveSettings(ds=1.0, delta_max=dmax, **ADAPT)
    entry = dict(track=tname, car=asdict(car), settings_1m=asdict(st1), warm_start=str(warm.relative_to(ROOT)),
                 homotopy_source=src.name, route=[])

    def log(step, sol):
        entry['route'].append(dict(step=step, **stats(sol)))
        update('ocp', entry, sub=case)
        print(case, step, sol.status, sol.iters, round(sol.t_solve, 1), round(sol.laptime, 4), flush=True)

    sol = None
    if direct:
        sol = solve(tr, car, st1, W0=W0, s0=s0)
        log('direct 1m', sol)
    if sol is None or not sol.success:
        # adaptive parameter homotopy: step 0.25, halved after a failure (down to 1/64), x1.5 after
        # a success; every converged step is the next warm start (and saved)
        W, s, lam, step = W0, s0, 0.0, 0.25
        while lam < 1.0 and step >= 1 / 64:
            nxt = min(1.0, lam + step)
            st = replace(st1, delta_max=(1 - nxt) * src_dmax + nxt * dmax)
            sol = solve(tr, blend(src, car, nxt), st, W0=W, s0=s)
            log(f'homotopy lam {nxt:.4g} 1m', sol)
            if sol.success:
                W, s, lam, step = sol.W, sol.s, nxt, step * 1.5
                WARM.mkdir(parents=True, exist_ok=True)
                np.savez(WARM / f'{case}_lam{lam:.4f}_1m.npz', W=W, s=s)
            else:
                step /= 2
        entry['homotopy_reached_lam'] = lam
    if not sol.success:
        entry['result'] = 'FAILED at 1 m'
        update('ocp', entry, sub=case)
        return
    WARM.mkdir(parents=True, exist_ok=True)
    np.savez(WARM / f'{case}_1m.npz', W=sol.W, s=sol.s)
    entry['final_1m'] = stats(sol)
    sol2 = solve(tr, car, replace(st1, ds=0.5), W0=sol.W, s0=sol.s)
    log('0.5m from 1m', sol2)
    if sol2.success:
        entry['final_0.5m'] = stats(sol2)
        entry['mesh_delta_s'] = sol2.laptime - sol.laptime
        ocp_pl14.write_csv(sol2, tr, OUT / f'ocp_solution_{case}_0.5m.csv')
    entry['result'] = 'OK' if sol2.success else 'OK at 1 m, 0.5 m failed'
    entry['wall_s'] = time.time() - t0
    update('ocp', entry, sub=case)
    print(case, 'done in', round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    for c in sys.argv[1:]:
        run(c)
