"""PL14 minimum-lap-time OCP runs: D-check (Barcelona) and Shanghai. Results -> docs/results/limebeer_ocp/.

Usage (2 threads, keeps a parallel training run usable):
  OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python scripts/ocp_pl14.py CASE [CASE ...]
CASE: bcn_beta055 bcn_beta060 bcn_beta065 bcn_half bcn_r1e6 bcn_pm4 bcn_guesses bcn_reint sha_pl14 sha_sf70h
Each case appends to results.json (keyed by case) and writes 1 m solution CSVs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlsetup.ocp.pl14_model import pl14_car, sf70h_tum_car  # noqa: E402
from rlsetup.ocp.solve import (ALG_NAMES, CTRL_NAMES, STATE_NAMES, SolveSettings, cumulative_time,  # noqa: E402
                               initial_guess, qss_guess, solve)
from rlsetup.ocp.track import Track  # noqa: E402

OUT = ROOT / "docs/results/limebeer_ocp"
RES = OUT / "results.json"
MESHES = (4.0, 2.0, 1.0)
PL14_REF = 82.57
_TRACKS: dict = {}


def track(name, r=None):
    key = (name, r)
    if key not in _TRACKS:
        _TRACKS[key] = Track.from_csv(name) if r is None else Track.from_csv(name, r=r)
    return _TRACKS[key]


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save(case, entry):
    data = json.loads(RES.read_text()) if RES.exists() else {}
    data[case] = entry
    import casadi
    from rlsetup.ocp.track import R_DEFAULT, TRACKS_DIR
    data.setdefault("_meta", {}).update(
        git=git_hash(), pl14_mesh_converged_ref_s=PL14_REF, casadi=casadi.__version__, tracks_dir=str(TRACKS_DIR),
        curvature_r_default_m6=R_DEFAULT, solver="IPOPT+MUMPS, trapezoidal, uniform mesh, periodic",
        threads="OMP_NUM_THREADS / VECLIB_MAXIMUM_THREADS <= 2; IPOPT run is deterministic for fixed inputs")
    RES.write_text(json.dumps(data, indent=1, default=float))


def summary(sol, ref=None):
    d = dict(ds=sol.extras["ds_mean"], ds_min=sol.extras["ds_min"], nodes=sol.extras["N"], laptime_s=sol.laptime, penalty=sol.penalty,
             status=sol.status, success=sol.success, iters=sol.iters, t_solve_s=round(sol.t_solve, 1),
             max_eq_residual_scaled=sol.extras["max_eq_residual_scaled"],
             u_max_kmh=float(sol.state("u").max() * 3.6), u_min_kmh=float(sol.state("u").min() * 3.6),
             power_max_kW=float(sol.extras["power"].max() / 1e3),
             N_min_N=float(sol.W[11:15].min()))
    if ref:
        d["pct_vs_pl14"] = 100 * (sol.laptime - ref) / ref
    return d


def write_csv(sol, tr, path):
    s = sol.s
    th = np.interp(s, tr.s, np.unwrap(tr.theta_map))
    xc, yc = tr.at(s, "x"), tr.at(s, "y")
    n = sol.state("n")
    cols = dict(s=s, t=cumulative_time(sol)[: len(s)], C=tr.at(s, "C"))
    for i, nm in enumerate(STATE_NAMES + CTRL_NAMES + ALG_NAMES):
        cols[nm] = sol.W[i]
    cols["power_W"] = sol.extras["power"]
    cols["x_car"], cols["y_car"] = xc + n * np.sin(th), yc - n * np.cos(th)   # n > 0 right of travel
    lo, hi = sol.extras["corridor"]
    cols["n_min"], cols["n_max"] = lo, hi
    np.savetxt(path, np.column_stack(list(cols.values())), delimiter=",", header=",".join(cols), comments="",
               fmt="%.6g")


def chain(case, tname, car, st_kw, ref=None, W0=None, s0=None, csv=True, r=None, meshes=MESHES):
    tr = track(tname, r)
    entry = dict(track=tname, track_L_m=tr.L, track_info=tr.info, car=asdict(car), meshes=[])
    sol = None
    for ds in meshes:
        st = SolveSettings(ds=ds, **st_kw)
        sol = solve(tr, car, st, W0=W0, s0=s0)
        W0, s0 = sol.W, sol.s
        entry["meshes"].append(summary(sol, ref))
        entry["settings"] = asdict(st)
        save(case, entry)
        print(case, ds, sol.status, sol.iters, round(sol.t_solve, 1), round(sol.laptime, 4), flush=True)
        if not sol.success:
            break
    m = entry["meshes"]
    if len(m) >= 2:
        entry["mesh_delta_s"] = {f"{a['ds']:.2f}->{b['ds']:.2f}": b["laptime_s"] - a["laptime_s"] for a, b in zip(m, m[1:])}
    save(case, entry)
    if csv and sol is not None and sol.success:
        write_csv(sol, tr, OUT / f"solution_{case}_{st.ds:g}m.csv")
        np.savez(OUT / f"_warm_{case}.npz", W=sol.W, s=sol.s)
    return sol


def run(case):
    t0 = time.time()
    if case.startswith("bcn_beta"):
        beta = int(case[-3:]) / 100
        chain(case, "Catalunya", pl14_car(beta=beta), {}, ref=PL14_REF)
    elif case.endswith("_adapt"):   # speed-adapted mesh, u_min 3 m/s, meshes 4/2/1/0.5 m nominal
        kw = {"mesh_u_ref": 30.0, "u_min": 3.0}
        base = case[:-6]
        if base == "bcn_beta060":
            chain(case, "Catalunya", pl14_car(beta=0.60), kw, ref=PL14_REF, meshes=MESHES + (0.5,))
        else:
            car = pl14_car(beta=0.60) if base == "sha_pl14" else sf70h_tum_car(beta=0.60)
            chain(case, "Shanghai", car, kw, meshes=MESHES + (0.5,))
    elif case == "bcn_r1e6":   # curvature-smoothing sensitivity: PL14's r = 1e6 read as SI m^6
        chain(case, "Catalunya", pl14_car(beta=0.60), {}, ref=PL14_REF, r=1e6)
    elif case == "bcn_pm4":
        chain(case, "Catalunya", pl14_car(beta=0.60), {"corridor_fixed": 4.0}, ref=PL14_REF)
    elif case == "bcn_half":   # extra 0.5 m mesh from the 1 m beta 0.60 solution (convergence check)
        tr, car = track("Catalunya"), pl14_car(beta=0.60)
        w = np.load(OUT / "_warm_bcn_beta060.npz")
        sol = solve(tr, car, SolveSettings(ds=0.5), W0=w["W"], s0=w["s"])
        save(case, summary(sol, PL14_REF))
        print(case, sol.status, sol.iters, round(sol.t_solve, 1), round(sol.laptime, 4), flush=True)
    elif case in ("sha_pl14_half", "sha_sf70h_half"):   # 0.5 m from the 1 m solution
        base = case[:-5]
        tr = track("Shanghai")
        car = pl14_car(beta=0.60) if base == "sha_pl14" else sf70h_tum_car(beta=0.60)
        w = np.load(OUT / f"_warm_{base}.npz")
        sol = solve(tr, car, SolveSettings(ds=0.5), W0=w["W"], s0=w["s"])
        save(case, summary(sol))
        print(case, sol.status, sol.iters, round(sol.t_solve, 1), round(sol.laptime, 4), flush=True)
    elif case == "bcn_guesses":
        tr, car = track("Catalunya"), pl14_car(beta=0.60)
        out = {}
        N = int(round(tr.L / 4.0))
        s = np.arange(N) * tr.L / N
        guesses = {"qss_mu1.5": qss_guess(tr, car, s), "qss_mu1.0": qss_guess(tr, car, s, mu=1.0),
                   "const_40mps": np.full(N, 40.0)}
        for k, u in guesses.items():
            sol = solve(tr, car, SolveSettings(ds=4.0), W0=initial_guess(tr, car, s, u))
            out[k] = summary(sol, PL14_REF) | {"guess_laptime_s": float(np.sum((s[1] - s[0]) / u))}
            print(case, k, sol.status, round(sol.laptime, 4), flush=True)
            save(case, out)
    elif case == "bcn_reint":
        from rlsetup.ocp.verify import reintegrate
        tr, car = track("Catalunya"), pl14_car(beta=0.60)
        res = {}
        for ds in (4.0, 1.0):   # re-solve quickly from the 1 m warm start
            w = np.load(OUT / "_warm_bcn_beta060.npz")
            sol = solve(tr, car, SolveSettings(ds=ds), W0=w["W"], s0=w["s"])
            res[f"ds_{ds:g}"] = summary(sol, PL14_REF) | reintegrate(sol, tr, car)
            print(case, ds, json.dumps(res[f"ds_{ds:g}"], default=float)[:400], flush=True)
            save(case, res)
    elif case == "sha_pl14":
        chain(case, "Shanghai", pl14_car(beta=0.60), {})
    elif case == "sha_sf70h":
        chain(case, "Shanghai", sf70h_tum_car(beta=0.60), {})
    elif case.startswith("sha_pl14_beta") or case.startswith("sha_sf70h_beta"):
        beta = int(case[-3:]) / 100
        car = pl14_car(beta=beta) if "pl14" in case else sf70h_tum_car(beta=beta)
        chain(case, "Shanghai", car, {})
    else:
        raise SystemExit(f"unknown case {case}")
    print(case, "done in", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
