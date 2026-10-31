"""Direct-collocation minimum-lap-time OCP for the ladder model (rlsetup.ocp.ladder_model).

Same transcription as solve.py (periodic, trapezoidal in s, PL14 scaling, IPOPT + MUMPS, speed-adapted
mesh), with layout x = [n, xi, u, v, omega], c = [delta, T_drive, b], z = [kappa x 4, N x 4] and the
extra path constraints of the ladder:
  g_drive <= 0                      power limit (L0) or engine map (lam['engine'] = 1)
  |delta_{k+1} - delta_k| <= rate * dt_k,  dt_k = h_k (Sf_k + Sf_{k+1}) / 2   (lam['steer'] > 0;
                                    rate = 1.5 rad/s / lam, so lam < 1 is a looser continuation)
  -w_left(s + dx_c) <= n_corner <= w_right(s + dx_c) for the 4 footprint corners (lam['track'] > 0);
                                    the mass-centre bound then relaxes to the track edges.
Cost: lap time + control-rate regularisation as solve.py: w_ddelta (d delta/ds)^2 +
w_dT [(d(T_drive/(M g R))/ds)^2 + (d(b T_bmax/(M g R))/ds)^2] (ASSUMPTION, reported separately).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time

import casadi as ca
import numpy as np

from .ladder_model import ALG_NAMES, CTRL_NAMES, NC, NX, NZ, STATE_NAMES, LadderCar, ladder_model
from .solve import make_mesh
from .track import Track

NAMES = STATE_NAMES + CTRL_NAMES + ALG_NAMES
NW = NX + NC + NZ


@dataclass
class LadderSettings:
    ds: float = 1.0
    w_ddelta: float = 0.1
    w_dT: float = 1e-3
    N_min: float = 50.0
    kappa_max_l0: float = 0.6       # L0's slip bound; with the MF5.2 tyre OCD's clamp |kappa| <= 1 is used
    u_min: float = 3.0
    mesh_u_ref: float | None = 30.0
    mesh_pow: float = 2.0
    mesh_min_frac: float = 0.1
    max_iter: int = 3000
    tol: float = 1e-6
    print_level: int = 0


@dataclass
class LadderSolution:
    s: np.ndarray
    W: np.ndarray
    laptime: float
    penalty: float
    status: str
    success: bool
    iters: int
    t_solve: float
    settings: dict
    car: dict
    extras: dict = field(default_factory=dict)

    def row(self, name):
        return self.W[NAMES.index(name)]


def _scales(lc: LadderCar):
    car = lc.l0
    Lc, g, M = car.w, car.g, car.M
    Vc, Tc, Fc = math.sqrt(g * Lc), math.sqrt(Lc / g), M * g
    R = car.R
    return dict(x=np.array([Lc, 1.0, Vc, Vc, 1 / Tc]), c=np.array([0.1, Fc * R, 1.0]),
                z=np.array([0.1] * 4 + [Fc / 4] * 4), alg=np.array([Fc] * 4 + [Fc * R] * 4), Fc=Fc, R=R)


def convert_l0(W13: np.ndarray, car) -> np.ndarray:
    """Warm start from a pl14_model solution (rows x5, [delta, T], z8) to the ladder layout."""
    T = W13[6]
    out = np.zeros((NW, W13.shape[1]))
    out[:5] = W13[:5]
    out[5] = W13[5]
    out[6] = np.maximum(T, 0) * car.R
    out[7] = np.maximum(-T, 0) / (8 * car.M * car.g)
    out[8:] = W13[7:]
    return out


def node_function(lc: LadderCar, sc):
    xs, cs_, zs, Cs = ca.SX.sym("x", NX), ca.SX.sym("c", NC), ca.SX.sym("z", NZ), ca.SX.sym("C")
    out = ladder_model(xs * sc["x"], cs_ * sc["c"], zs * sc["z"], Cs, lc)
    return ca.Function("node", [xs, cs_, zs, Cs],
                       [out["dxds"] / sc["x"], out["Sf"], out["alg"] / sc["alg"], out["g_drive"], out["corners"]])


def solve_ladder(track: Track, lc: LadderCar, st: LadderSettings, W0: np.ndarray, s0: np.ndarray) -> LadderSolution:
    """Solve on the speed-adapted (or uniform) mesh of nominal step st.ds, warm-started from (W0, s0)."""
    from .solve import SolveSettings
    mesh_st = SolveSettings(ds=st.ds, mesh_u_ref=st.mesh_u_ref, mesh_pow=st.mesh_pow, mesh_min_frac=st.mesh_min_frac)
    s = make_mesh(track, lc.l0, mesh_st, W0, s0)
    N = len(s)
    hk = np.diff(np.r_[s, track.L])
    W0 = np.array([np.interp(s, np.r_[s0, track.L], np.r_[r, r[0]]) for r in W0])
    sc = _scales(lc)
    Sv = np.r_[sc["x"], sc["c"], sc["z"]]
    fmap = node_function(lc, sc).map(N, "serial")
    V = ca.MX.sym("V", NW * N)
    Wm = ca.reshape(V, NW, N)
    X, Cc, Z = Wm[:NX, :], Wm[NX:NX + NC, :], Wm[NX + NC:, :]
    Cv = track.at(s, "C")
    F, Sf, ALG, GD, CORN = fmap(X, Cc, Z, Cv.reshape(1, -1))
    nxt = lambda A: ca.horzcat(A[:, 1:], A[:, :1])
    H = ca.DM(hk.reshape(1, -1))
    defect = nxt(X) - X - ca.repmat(H / 2, NX, 1) * (F + nxt(F))
    lap = ca.sum2(H * (Sf + nxt(Sf)) / 2)
    dU = (nxt(Cc) - Cc) / ca.repmat(H, NC, 1)
    Tbs = lc.Tb_max / sc["c"][1]
    pen = (st.w_ddelta * sc["c"][0] ** 2 * ca.sum2(H * dU[0, :] ** 2)
           + st.w_dT * (ca.sum2(H * dU[1, :] ** 2) + Tbs ** 2 * ca.sum2(H * dU[2, :] ** 2)))
    g = [ca.vec(defect), ca.vec(ALG), ca.vec(GD)]
    lbg = [np.zeros(NX * N + NZ * N), -np.inf * np.ones(N)]
    ubg = [np.zeros(NX * N + NZ * N), np.zeros(N)]
    lam = lc.lam
    if lam["steer"] > 0:
        rate = lc.ocd.steer_rate_max / lam["steer"]
        ddel = (nxt(Cc[0, :]) - Cc[0, :]) * sc["c"][0]
        dt = H * (Sf + nxt(Sf)) / 2
        g += [ca.vec((ddel - rate * dt) / 0.01), ca.vec((-ddel - rate * dt) / 0.01)]
        lbg += [-np.inf * np.ones(2 * N)]
        ubg += [np.zeros(2 * N)]
    if lam["track"] > 0:
        g.append(ca.vec(CORN))
        lo_c, hi_c = [], []
        for dx, _ in lc.footprint():
            lo_c.append(-track.at(s + dx, "w_left"))
            hi_c.append(track.at(s + dx, "w_right"))
        lbg.append(np.array(lo_c).T.ravel())
        ubg.append(np.array(hi_c).T.ravel())
        lo, hi = -track.at(s, "w_left"), track.at(s, "w_right")
    else:
        lo, hi = track.corridor(s, lc.l0.half_width)
    g = ca.vertcat(*g)
    lbg, ubg = np.concatenate(lbg), np.concatenate(ubg)
    kmax = 1.0 if lam["tyre"] > 0 else st.kappa_max_l0
    rows = {0: (lo, hi), 1: (-1.0, 1.0), 2: (st.u_min, 120.0), 3: (-20.0, 20.0), 4: (-3.0, 3.0),
            5: (-lc.ocd.steer_max, lc.ocd.steer_max), 6: (0.0, 3 * sc["Fc"] * sc["R"]), 7: (0.0, 1.0)}
    for i in range(8, 12):
        rows[i] = (-kmax, kmax)
    for i in range(12, 16):
        rows[i] = (st.N_min, 6 * sc["Fc"])
    lbw, ubw = np.zeros((NW, N)), np.zeros((NW, N))
    for i, (l_, u_) in rows.items():
        lbw[i], ubw[i] = l_, u_
    lbw /= Sv[:, None]
    ubw /= Sv[:, None]
    x0 = np.clip(W0 / Sv[:, None], lbw, ubw)
    opts = {"ipopt.linear_solver": "mumps", "ipopt.max_iter": st.max_iter, "ipopt.tol": st.tol,
            "ipopt.print_level": st.print_level, "print_time": 0, "ipopt.mu_strategy": "adaptive"}
    solver = ca.nlpsol("ladder", "ipopt", {"x": V, "f": lap + pen, "g": g}, opts)
    t0 = time.time()
    sol = solver(x0=ca.vec(x0), lbx=ca.vec(lbw), ubx=ca.vec(ubw), lbg=lbg, ubg=ubg)
    t_solve = time.time() - t0
    stats = solver.stats()
    Wopt = np.array(ca.reshape(sol["x"], NW, N)) * Sv[:, None]
    lap_v, pen_v = (float(v) for v in ca.Function("lp", [V], [lap, pen])(sol["x"]))
    Sf_v = np.array(ca.Function("sf", [V], [Sf])(sol["x"])).ravel()
    gmax = float(np.max(np.abs(np.array(sol["g"]).ravel()[: (NX + NZ) * N])))
    return LadderSolution(s=s, W=Wopt, laptime=lap_v, penalty=pen_v, status=stats["return_status"],
                          success=stats["return_status"] == "Solve_Succeeded", iters=int(stats["iter_count"]),
                          t_solve=t_solve, settings=asdict(st), car=lc.summary(),
                          extras=dict(Sf=Sf_v, h=hk, N=N, ds_mean=float(hk.mean()), ds_min=float(hk.min()),
                                      max_eq_residual_scaled=gmax, corridor=(lo, hi)))


def cumulative_time(sol: LadderSolution) -> np.ndarray:
    Sf = sol.extras["Sf"]
    return np.r_[0.0, np.cumsum(sol.extras["h"] * 0.5 * (Sf + np.roll(Sf, -1)))]


def diagnostics(sol: LadderSolution, track: Track, lc: LadderCar) -> dict:
    """Per-node physical quantities of a solution (numpy arrays)."""
    x, c, z, C = ca.SX.sym("x", NX), ca.SX.sym("c", NC), ca.SX.sym("z", NZ), ca.SX.sym("C")
    o = ladder_model(x, c, z, C, lc)
    keys = ["Fx", "Fy", "du", "cap", "I_eng", "om_ax", "Faz", "Tb", "power", "g_drive", "res_lsd"]
    outs = [o[k] if not isinstance(o[k], (int, float)) else ca.SX(o[k]) for k in keys]
    outs += [o["F"][k][i] for k in ("fr", "fl", "rr", "rl") for i in (0, 1)]
    outs += [o["corners"]]
    if o["rh_f"] is not None:
        outs += [o["rh_f"], o["rh_r"]]
    f = ca.Function("d", [x, c, z, C], outs).map(sol.extras["N"])
    W = sol.W
    r = f(W[:NX], W[NX:NX + NC], W[NX + NC:], track.at(sol.s, "C").reshape(1, -1))
    r = [np.array(v) for v in r]
    d = {k: r[i].ravel() for i, k in enumerate(keys)}
    j = len(keys)
    for k in ("fr", "fl", "rr", "rl"):
        d[f"Fx_{k}"], d[f"Fy_{k}"] = r[j].ravel(), r[j + 1].ravel()
        j += 2
    d["corners"] = r[j]
    if o["rh_f"] is not None:
        d["rh_f"], d["rh_r"] = r[j + 1].ravel(), r[j + 2].ravel()
    return d
