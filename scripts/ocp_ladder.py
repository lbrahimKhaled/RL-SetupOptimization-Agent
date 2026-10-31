"""OCP ladder: the PL14-style OCP made closer to OCD's car one feature per step (rlsetup.ocp.ladder_model).

Usage (2 threads; at most 2 of these processes at a time):
  OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python scripts/ocp_ladder.py chain  [--track T] [--from STEP] [--to STEP]
  OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python scripts/ocp_ladder.py refine [--track T] [STEP ...]   (0.5 m from 1 m)
  .venv/bin/python scripts/ocp_ladder.py report
T: Shanghai (setup p1_TUM, default) or Catalunya (setup bcn_best). `refine --watch` waits for each 1 m
solution of the chain and refines it (the second process). Results: docs/results/limebeer_ocp/
results.json key `ladder`, LADDER.md (rewritten by `report`, called after every solve), _warm_ladder_*.npz,
solution_ladder_*_1m.csv.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlsetup.ocp.ladder_model import FEATURES, LadderCar  # noqa: E402
from rlsetup.ocp.ladder_solve import (NAMES, LadderSettings, convert_l0, cumulative_time, diagnostics,  # noqa: E402
                                      solve_ladder)
from rlsetup.ocp.ocd_car import from_backend  # noqa: E402
from rlsetup.ocp.pl14_model import pl14_car, sf70h_tum_car  # noqa: E402
from rlsetup.ocp.track import Track  # noqa: E402

OUT = ROOT / "docs/results/limebeer_ocp"
RES = OUT / "results.json"
LOCK = OUT / ".results.lock"
TRACKS = {"Shanghai": dict(setup="p1_TUM", tag="sha"), "Catalunya": dict(setup="bcn_best", tag="bcn")}
REF = {"Shanghai": {"OCD RL driver (p1_TUM)": 101.27, "OCD QSS on TUM raceline (debug car)": 95.73,
                    "TUM laptime-sim none": 94.886},
       "Catalunya": {"OCD RL run bcn best clean lap": 85.21, "OCD QSS (bcn_best, safe line)": 82.89}}
# (step, feature switched on, short description). Order = ladder order; each step keeps all earlier ones.
STEPS = [
    ("L0", None, "PL14 model, SF70H-TUM mapping (beta 0.60, 575 kW, kd diff, PL14 loads, CoG corridor w - 0.9 m)"),
    ("L1", "tyre", "OCD MF5.2 tyre (car's coefficients, combined slip, |kappa| <= 1), R = 0.3299 m"),
    ("L2", "brakes", "OCD brakes: 15 kN m total, bias 0.49 (setup), per-wheel torques, wheel-spin equilibrium"),
    ("L2b", "rollres", "OCD rolling resistance 0.03 N in the wheel torque balances"),
    ("L3", "engine", "OCD engine map x gear ratios x 0.96 (static upshift map) instead of 575 kW; ERS off"),
    ("L3b", "inertia", "wheel + engine rotating inertia (quasi-steady wheel spin, engine inertia per gear)"),
    ("L4", "lsd", "OCD LSD: (L_rr - L_rl) = 2 tanh(dOmega/2) * 1.0 * T_drive (coast lock 0) instead of kd"),
    ("L5", "loads", "OCD steady suspension: springs/ARBs from the translator, roll/pitch centres 0.2 m, "
                    "tyre springs, aero at static ride heights"),
    ("L6", "steer", "OCD steering rate |d delta/dt| <= 1.5 rad/s (|delta| <= 0.6 already in L0)"),
    ("L7", "track", "OCD track rule: 4.6 x 2.0 m footprint corners inside the edges"),
    ("L8", "rhaero", "OCD ride-height aero (CzA +2 %/cm below reference) at the steady heave/pitch"),
]
STEP_IDX = {s[0]: i for i, s in enumerate(STEPS)}


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@contextlib.contextmanager
def locked():
    LOCK.touch(exist_ok=True)
    with open(LOCK) as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def update(track_name, step, key, value):
    """results.json['ladder'][track][step][key] = value (file-locked read-modify-write)."""
    with locked():
        data = json.loads(RES.read_text()) if RES.exists() else {}
        lad = data.setdefault("ladder", {})
        lad.setdefault("_meta", {}).update(git=git_hash(), casadi="3.8.2", steps=[list(s) for s in STEPS],
                                           threads="OMP/VECLIB <= 2", references=REF,
                                           note="lap times exclude the control-rate regularisation (penalty)")
        lad.setdefault(track_name, {}).setdefault(step, {})[key] = value
        RES.write_text(json.dumps(data, indent=1, default=float))


def lam_for(step):
    i = STEP_IDX[step]
    return {f: (1.0 if any(s[1] == f for s in STEPS[1:i + 1]) else 0.0) for f in FEATURES}


_CACHE = {}


def get_track(name):
    if name not in _CACHE:
        _CACHE[name] = Track.from_csv(name)
    return _CACHE[name]


def get_cars(track_name):
    key = ("car", track_name)
    if key not in _CACHE:
        ocd = from_backend(TRACKS[track_name]["setup"])
        if track_name == "Shanghai":
            l0 = sf70h_tum_car(beta=0.60)                    # exactly the 94.25 s L0 car
        else:   # L0 mapping with the setup's aero at static ride heights (OCD values)
            czA, _, _ = ocd.aero_linear()
            l0 = sf70h_tum_car(beta=0.60, ClA=float(czA.sum()), CdA=ocd.cd_A, aA=ocd.l * float(czA[1] / czA.sum()),
                               rho=ocd.rho)
        _CACHE[key] = (l0, ocd)
    return _CACHE[key]


def warm_path(tag, step, mesh):
    return OUT / f"_warm_ladder_{tag}_{step}_{mesh}.npz"


def stats(sol, tr, lc):
    W = sol.W
    d = diagnostics(sol, tr, lc)
    u = sol.row("u")
    lam = lc.lam
    out = dict(laptime_s=sol.laptime, penalty_s=sol.penalty, status=sol.status, success=sol.success, iters=sol.iters,
               t_solve_s=round(sol.t_solve, 1), nodes=sol.extras["N"], ds_mean=sol.extras["ds_mean"],
               ds_min=sol.extras["ds_min"], max_eq_residual_scaled=sol.extras["max_eq_residual_scaled"],
               u_min_kmh=float(u.min() * 3.6), u_max_kmh=float(u.max() * 3.6), s_at_u_min=float(sol.s[np.argmin(u)]),
               N_min_N=float(W[12:16].min()), kappa_min=float(W[8:12].min()), kappa_max=float(W[8:12].max()),
               delta_max_abs=float(np.abs(sol.row("delta")).max()),
               frac_drive_limited=float(np.mean(d["g_drive"] > -1e-3)), frac_braking=float(np.mean(sol.row("b") > 0.01)),
               frac_brake_at_cap=float(np.mean(sol.row("b") > 0.999)),
               frac_drive_and_brake=float(np.mean((sol.row("b") > 0.01) & (sol.row("T_drive") > 10))),
               max_power_kW=float(d["power"].max() / 1e3), max_sideslip_deg=float(np.degrees(np.abs(np.arctan(sol.row("v") / u)).max())),
               ax_min_g=float(d["du"].min() / 9.81), ax_max_g=float(d["du"].max() / 9.81),
               ay_max_g=float(np.abs(d["Fy"]).max() / lc.l0.M / 9.81))
    if lam["steer"] > 0:
        Sf, h = sol.extras["Sf"], sol.extras["h"]
        rate = np.abs(np.roll(sol.row("delta"), -1) - sol.row("delta")) / (h * 0.5 * (Sf + np.roll(Sf, -1)))
        out.update(steer_rate_max=float(rate.max()), frac_steer_rate_limited=float(np.mean(rate > 0.999 * lc.ocd.steer_rate_max / lam["steer"])))
    if lam["track"] > 0:
        lo = np.array([-tr.at(sol.s + dx, "w_left") for dx, _ in lc.footprint()])
        hi = np.array([tr.at(sol.s + dx, "w_right") for dx, _ in lc.footprint()])
        c = d["corners"]
        out.update(frac_footprint_active=float(np.mean(np.any((c - lo < 1e-3) | (hi - c < 1e-3), axis=0))),
                   footprint_violation_m=float(max((lo - c).max(), (c - hi).max())))
    if lam["loads"] > 0:   # packer check: bump travel from the steady suspension
        out.update(stats_bump(sol, tr, lc, d))
    if lam["rhaero"] > 0:
        out.update(rh_front_min_m=float(d["rh_f"].min()), rh_rear_min_m=float(d["rh_r"].min()))
    if lam["engine"] > 0:
        om = d["om_ax"]
        wu = lc.ocd.upshift_omega()
        gear = 1 + np.searchsorted(wu, om)
        out["gear_hist"] = {int(g): int((gear == g).sum()) for g in np.unique(gear)}
    lsd = d["Fx_rr"] - d["Fx_rl"]
    out.update(rear_dFx_median_abs_N=float(np.median(np.abs(lsd))), rear_Fx_min_N=float(min(d["Fx_rr"].min(), d["Fx_rl"].min())))
    return out


def stats_bump(sol, tr, lc, d):
    """Suspension bump travel [m] (OCD's packer convention) at every node vs the packer gaps."""
    lm = lc.lmap
    cdl, sdl = np.cos(sol.row("delta")), np.sin(sol.row("delta"))
    vx = {"fr": cdl * d["Fx_fr"] - sdl * d["Fy_fr"], "fl": cdl * d["Fx_fl"] - sdl * d["Fy_fl"], "rr": d["Fx_rr"], "rl": d["Fx_rl"]}
    vy = {"fr": cdl * d["Fy_fr"] + sdl * d["Fx_fr"], "fl": cdl * d["Fy_fl"] + sdl * d["Fx_fl"], "rr": d["Fy_rr"], "rl": d["Fy_rl"]}
    idx = {"fl": 0, "fr": 1, "rl": 2, "rr": 3}
    f = np.zeros((10, sol.extras["N"]))
    for k, i in idx.items():
        f[i], f[4 + i] = vx[k], -vy[k]
    u = sol.row("u")
    q = 0.5 * lc.ocd.rho * u ** 2
    if lc.lam["rhaero"] > 0:
        # recompute the axle downforce split as the model does
        K = lc.lam["rhaero"] * lc.dczA @ lm["rh_rows"]
        Yf8, YL = lm["Yf"][:, :8], lm["Yf"][:, 8:]
        FL = np.zeros((2, f.shape[1]))
        for j in range(f.shape[1]):
            M2 = np.eye(2) - q[j] * K @ YL
            FL[:, j] = np.linalg.solve(M2, q[j] * (lc.czA_s + K @ (lm["Y0"] + Yf8 @ f[:8, j])))
    else:
        FL = np.outer(lc.czA_s, q)
    f[8:] = FL
    comp = lm["comp0"][:, None] + lm["Bcomp"] @ f
    bump = -(comp + lm["s0"][:, None])
    return dict(bump_max_m=float(bump.max()), packer_gap_m=float(lm["packer_gap"].min()),
                packer_touched=bool(lm["packer_on"] and (bump > lm["packer_gap"][:, None]).any()),
                aero_FL_max_N=[float(FL[0].max()), float(FL[1].max())])


def write_csv(sol, tr, path, extra=None):
    s = sol.s
    th = np.interp(s, tr.s, np.unwrap(tr.theta_map))
    xc, yc = tr.at(s, "x"), tr.at(s, "y")
    n = sol.row("n")
    cols = dict(s=s, t=cumulative_time(sol)[: len(s)], C=tr.at(s, "C"))
    for i, nm in enumerate(NAMES):
        cols[nm] = sol.W[i]
    cols["x_car"], cols["y_car"] = xc + n * np.sin(th), yc - n * np.cos(th)
    for k, v in (extra or {}).items():
        cols[k] = v
    np.savetxt(path, np.column_stack(list(cols.values())), delimiter=",", header=",".join(cols), comments="", fmt="%.6g")


def blend_l0(a, b, lam):
    """CarParams homotopy a -> b (floats and nested tyre params)."""
    from dataclasses import fields
    kw = {}
    for f_ in fields(a):
        va, vb = getattr(a, f_.name), getattr(b, f_.name)
        if isinstance(va, float):
            kw[f_.name] = (1 - lam) * va + lam * vb
        elif hasattr(va, "__dataclass_fields__"):
            kw[f_.name] = blend_l0(va, vb, lam)
    return replace(b, **kw)


def initial_warm(track_name, l0):
    """L0 warm start from the earlier PL14 runs (pl14_model layout, converted)."""
    if track_name == "Shanghai":
        w = np.load(OUT / "_warm_sha_sf70h_homotopy.npz")
    else:
        w = np.load(OUT / "_warm_bcn_beta060_adapt.npz")
    return convert_l0(w["W"], l0), w["s"]


def solve_step(track_name, step, st: LadderSettings, W0, s0, log):
    """Direct solve at full weight; on failure a homotopy on the new feature's weight (and for L0 on
    Catalunya, a CarParams homotopy from the PL14 car). Returns (solution, route)."""
    tr = get_track(track_name)
    l0, ocd = get_cars(track_name)
    lam = lam_for(step)
    feat = STEPS[STEP_IDX[step]][1]
    if step == "L0" and track_name == "Catalunya":
        route, sol = [], None
        for lm_ in (0.25, 0.5, 0.75, 1.0):
            lc = LadderCar(blend_l0(pl14_car(beta=0.60), l0, lm_), ocd, lam)
            sol = solve_ladder(tr, lc, st, W0, s0)
            route.append((lm_, sol.status, sol.iters, round(sol.laptime, 4)))
            log(f"{track_name} L0 car-homotopy {lm_}: {sol.status} {sol.iters} it {sol.t_solve:.0f} s {sol.laptime:.4f}")
            if not sol.success:
                return sol, {"homotopy_l0_car": route}
            W0, s0 = sol.W, sol.s
        return sol, {"homotopy_l0_car": route}
    lc = LadderCar(l0, ocd, lam)
    sol = solve_ladder(tr, lc, st, W0, s0)
    log(f"{track_name} {step} direct: {sol.status} {sol.iters} it {sol.t_solve:.0f} s {sol.laptime:.4f}")
    if sol.success or feat is None:
        return sol, {"direct": (sol.status, sol.iters)}
    route = {"direct": (sol.status, sol.iters), "homotopy": []}
    grids = [(0.25, 0.5, 0.75, 1.0), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    for grid in grids:
        Wh, sh = W0, s0
        ok = True
        for lm_ in grid:
            lam_h = dict(lam)
            lam_h[feat] = lm_
            sol = solve_ladder(tr, LadderCar(l0, ocd, lam_h), st, Wh, sh)
            route["homotopy"].append((lm_, sol.status, sol.iters, round(sol.laptime, 4)))
            log(f"{track_name} {step} homotopy {feat}={lm_}: {sol.status} {sol.iters} it {sol.t_solve:.0f} s {sol.laptime:.4f}")
            if not sol.success:
                ok = False
                break
            Wh, sh = sol.W, sol.s
        if ok:
            return sol, route
    return sol, route


def run_chain(track_name, first, last, log):
    tag = TRACKS[track_name]["tag"]
    tr = get_track(track_name)
    l0, ocd = get_cars(track_name)
    i0, i1 = STEP_IDX[first], STEP_IDX[last]
    for i in range(i0, i1 + 1):
        step, feat, desc = STEPS[i]
        if i == 0:
            W0, s0 = initial_warm(track_name, l0)
        else:
            w = np.load(warm_path(tag, STEPS[i - 1][0], "1m"))
            W0, s0 = w["W"], w["s"]
        update(track_name, step, "description", desc)
        update(track_name, step, "lam", lam_for(step))
        if i == 0:
            update(track_name, "_car", "ocd", ocd.summary())
            update(track_name, "_car", "l0", {k: v for k, v in l0.__dict__.items() if not hasattr(v, "__dataclass_fields__")})
        t0 = time.time()
        sol, route = solve_step(track_name, step, LadderSettings(ds=1.0), W0, s0, log)
        lc = LadderCar(l0, ocd, lam_for(step))
        st_ = stats(sol, tr, lc) if sol.success else dict(status=sol.status, success=False, iters=sol.iters)
        update(track_name, step, "mesh_1m", st_ | {"route": route, "wall_s": round(time.time() - t0, 1)})
        if not sol.success:
            log(f"{track_name} {step} FAILED: {route}")
            report()
            return False
        np.savez(warm_path(tag, step, "1m"), W=sol.W, s=sol.s)
        write_csv(sol, tr, OUT / f"solution_ladder_{tag}_{step}_1m.csv")
        report()
    return True


def run_refine(track_name, steps, log, watch=False):
    tag = TRACKS[track_name]["tag"]
    tr = get_track(track_name)
    l0, ocd = get_cars(track_name)
    for step in steps:
        src = warm_path(tag, step, "1m")
        while watch and not src.exists():
            time.sleep(30)
        if not src.exists():
            log(f"no 1 m solution for {step}")
            continue
        time.sleep(2)
        w = np.load(src)
        lc = LadderCar(l0, ocd, lam_for(step))
        sol = solve_ladder(tr, lc, LadderSettings(ds=0.5), w["W"], w["s"])
        log(f"{track_name} {step} 0.5 m: {sol.status} {sol.iters} it {sol.t_solve:.0f} s {sol.laptime:.4f}")
        st_ = stats(sol, tr, lc) if sol.success else dict(status=sol.status, success=False, iters=sol.iters)
        update(track_name, step, "mesh_0.5m", st_)
        if sol.success:
            np.savez(warm_path(tag, step, "0.5m"), W=sol.W, s=sol.s)
        report()


# ---------------------------------------------------------------------------------- report --
def corner_segments(tr: Track, thr=0.012, merge_m=120.0):
    """Corner apexes (local |C| maxima above thr [1/m], merged within merge_m) and segment
    boundaries halfway between consecutive apexes. Returns (apex_s, bounds)."""
    s = np.arange(0.0, tr.L, 1.0)
    C = np.abs(tr.at(s, "C"))
    k = 15
    Cs = np.convolve(np.r_[C[-k:], C, C[:k]], np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    pk = [i for i in range(len(s)) if Cs[i] > thr and Cs[i] >= Cs[i - 1] and Cs[i] >= Cs[(i + 1) % len(s)]]
    apex = []
    for i in pk:
        if apex and s[i] - apex[-1][0] < merge_m:
            if Cs[i] > apex[-1][1]:
                apex[-1] = (s[i], Cs[i])
        else:
            apex.append((s[i], Cs[i]))
    if len(apex) > 1 and apex[0][0] + tr.L - apex[-1][0] < merge_m:
        apex.pop(0 if apex[0][1] < apex[-1][1] else -1)
    a = np.array([p[0] for p in apex])
    mids = (a + np.roll(a, -1) + np.r_[np.zeros(len(a) - 1), tr.L]) / 2 % tr.L
    return a, np.sort(mids)


def seg_times(track_name, step, mesh="1m"):
    """Time to each segment boundary from the saved solution (trapezoid on 1/s_dot)."""
    tag = TRACKS[track_name]["tag"]
    p = warm_path(tag, step, mesh)
    if not p.exists():
        return None
    w = np.load(p)
    W, s = w["W"], w["s"]
    tr = get_track(track_name)
    C = tr.at(s, "C")
    n, xi, u, v = W[0], W[1], W[2], W[3]
    Sf = (1 - n * C) / (u * np.cos(xi) - v * np.sin(xi))
    h = np.diff(np.r_[s, tr.L])
    t = np.r_[0.0, np.cumsum(h * 0.5 * (Sf + np.roll(Sf, -1)))]
    return np.r_[s, tr.L], t, u


def report():
    """Rewrite LADDER.md from results.json (cheap, no solves)."""
    with locked():
        data = json.loads(RES.read_text()) if RES.exists() else {}
    lad = data.get("ladder", {})
    lines = ["# OCP ladder: PL14 OCP made closer to OCD's car, one feature per step",
             "", f"Updated {time.strftime('%Y-%m-%d %H:%M')}. Code: rlsetup/ocp/{{ocd_car,ladder_model,ladder_solve}}.py, "
             "scripts/ocp_ladder.py. Every step keeps all earlier ones. Lap times exclude the control-rate "
             "regularisation; only Solve_Succeeded laps are listed. Mesh = speed-adapted, nominal 1 m / 0.5 m.", ""]
    for tname in ("Shanghai", "Catalunya"):
        T = lad.get(tname)
        if not T:
            continue
        lines += [f"## {tname} ({TRACKS[tname]['setup']})", "",
                  "References: " + "; ".join(f"{k} {v} s" for k, v in REF[tname].items()), "",
                  "| Step | Feature added | 1 m [s] | 0.5 m [s] | mesh Δ [s] | Δ vs prev [s] | route | v min / max [km/h] | notes |",
                  "|---|---|---|---|---|---|---|---|---|"]
        prev = None
        segs = corner_segments(get_track(tname))
        seg_rows = []
        prev_step = None
        for step, feat, desc in STEPS:
            e = T.get(step)
            if not e:
                continue
            m1, m5 = e.get("mesh_1m", {}), e.get("mesh_0.5m", {})
            t1 = m1.get("laptime_s") if m1.get("success") else None
            t5 = m5.get("laptime_s") if m5.get("success") else None
            route = m1.get("route", {})
            rtxt = "direct" if "homotopy" not in route and "homotopy_l0_car" not in route else (
                "homotopy " + ",".join(str(h[0]) for h in route.get("homotopy", route.get("homotopy_l0_car", []))
                                      if h[1] == "Solve_Succeeded"))
            if m1 and not m1.get("success"):
                rtxt = f"FAILED ({m1.get('status')})"
            notes = []
            for k, fmt in (("frac_drive_limited", "drive-lim {:.0%}"), ("frac_brake_at_cap", "brake cap {:.0%}"),
                           ("frac_steer_rate_limited", "steer-rate-lim {:.0%}"), ("frac_footprint_active", "footprint act {:.0%}"),
                           ("N_min_N", "N_min {:.0f} N"), ("bump_max_m", "bump max {:.3f} m"), ("rh_front_min_m", "rh_f min {:.3f} m")):
                if k in m1:
                    notes.append(fmt.format(m1[k]))
            if m1.get("packer_touched"):
                notes.append("PACKER TOUCHED")
            d_prev = (t1 - prev) if (t1 is not None and prev is not None) else None
            f = lambda v: "–" if v is None else f"{v:.3f}"
            vr = f"{m1.get('u_min_kmh', float('nan')):.1f} / {m1.get('u_max_kmh', float('nan')):.1f}" if t1 else "–"
            lines.append(f"| {step} | {desc} | {f(t1)} | {f(t5)} | {f(t5 - t1 if t1 and t5 else None)} | "
                         f"{('%+.3f' % d_prev) if d_prev is not None else '–'} | {rtxt} | {vr} | {'; '.join(notes)} |")
            if t1 is not None:
                prev = t1
                # per-segment delta vs previous step
                cur = seg_times(tname, step)
                if prev_step is not None and cur is not None:
                    old = seg_times(tname, prev_step)
                    apex, bnd = segs
                    edges = np.r_[0.0, bnd, get_track(tname).L]
                    tc, to = np.interp(edges, cur[0], cur[1]), np.interp(edges, old[0], old[1])
                    dseg = np.diff(tc) - np.diff(to)
                    order = np.argsort(-np.abs(dseg))[:4]
                    vmin_c = [float(np.min(cur[2][(cur[0][:-1] >= edges[j]) & (cur[0][:-1] < edges[j + 1])])) * 3.6 for j in order]
                    vmin_o = [float(np.min(old[2][(old[0][:-1] >= edges[j]) & (old[0][:-1] < edges[j + 1])])) * 3.6 for j in order]
                    seg_rows.append(f"- **{step}** vs {prev_step}: " + "; ".join(
                        f"{edges[j]:.0f}–{edges[j + 1]:.0f} m {dseg[j]:+.3f} s (v_min {vo:.0f}→{vc:.0f} km/h)"
                        for j, vo, vc in zip(order, vmin_o, vmin_c)))
                prev_step = step
        lines += ["", "Largest per-segment changes vs the previous step (segments split halfway between corner apexes; "
                  f"apexes at {', '.join(f'{a:.0f}' for a in segs[0])} m):", ""] + seg_rows + [""]
    (OUT / "LADDER.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("chain", "refine", "report"))
    ap.add_argument("steps", nargs="*")
    ap.add_argument("--track", default="Shanghai")
    ap.add_argument("--from", dest="first", default="L0")
    ap.add_argument("--to", dest="last", default=STEPS[-1][0])
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    logf = OUT / f"ladder_{a.cmd}_{TRACKS[a.track]['tag']}.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(logf, "a") as fh:
            fh.write(line + "\n")

    if a.cmd == "chain":
        run_chain(a.track, a.first, a.last, log)
    elif a.cmd == "refine":
        steps = a.steps or [s[0] for s in STEPS]
        run_refine(a.track, steps, log, watch=a.watch)
    else:
        report()


if __name__ == "__main__":
    main()
