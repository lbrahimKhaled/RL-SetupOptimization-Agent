"""Distance-domain, periodic, trapezoidal direct-collocation minimum-lap-time OCP for the PL14 car.

Transcription (PL14 §3.2): uniform s grid with N nodes, step h = L/N; states x = [n, xi, u, v, om],
controls c = [delta, T] and algebraic z = [kappa_fr, kappa_fl, kappa_rr, kappa_rl, N_fr, N_fl, N_rr, N_rl]
at every node. Trapezoidal defects x_{k+1} - x_k = h/2 (f_k + f_{k+1}) with x_N = x_0 (track closure),
algebraic equations (23)-(26), (34), brake split, (35) at every node, power <= Pmax at every node.
Cost: lap time sum_k h Sf_k (trapezoid on a periodic uniform grid) plus a small control-rate
regularisation (ASSUMPTION; ML16 eq 52 style, reported separately and kept < 0.1 % of the lap).

Scaling (PL14 §3.3.1): unit mass M, unit length = wheelbase, g = 1, i.e. velocity sqrt(g w),
time sqrt(w/g), force M g. Implemented as variable/constraint scale factors on the SI model.
Solver: casadi 3.8.2 nlpsol IPOPT + MUMPS, exact Hessian.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math
import time

import casadi as ca
import numpy as np

from .pl14_model import CarParams, car_model, static_loads
from .track import Track

NX, NC, NZ = 5, 2, 8
STATE_NAMES = ["n", "xi", "u", "v", "omega"]
CTRL_NAMES = ["delta", "T"]
ALG_NAMES = ["kappa_fr", "kappa_fl", "kappa_rr", "kappa_rl", "N_fr", "N_fl", "N_rr", "N_rl"]


@dataclass
class SolveSettings:
    ds: float = 4.0                 # mesh step [m]
    eps: float = 1e-3               # eq-48 smoothing (on T scaled by M g: eps * (M g)^2 in N^2)
    w_ddelta: float = 0.1           # ASSUMPTION regularisation weight on (d delta/ds)^2 [s m]
    w_dT: float = 1e-3              # ASSUMPTION regularisation weight on (d(T/Mg)/ds)^2 [s m]
    corridor_fixed: float | None = None   # +-fixed [m] (ML16: 4 m); None -> half-width minus car half-width
    N_min: float = 50.0             # wheel-load floor [N] (replaces PL14 light-wheel clipping)
    kappa_max: float = 0.6          # |longitudinal slip| bound [-] (kappa = -1 is a locked wheel)
    u_min: float = 10.0             # [m/s]
    mesh_u_ref: float | None = None  # speed-adapted mesh: h(s) = ds * clip((u/mesh_u_ref)^mesh_pow, mesh_min_frac, 1)
    mesh_pow: float = 2.0           # 2: keeps h * (s-domain yaw/lateral eigenvalue ~ 1/u^2) about constant
    mesh_min_frac: float = 0.1
    max_iter: int = 3000
    tol: float = 1e-6
    print_level: int = 0


def _scales(car: CarParams):
    Lc, g, M = car.w, car.g, car.M
    Vc, Tc, Fc = math.sqrt(g * Lc), math.sqrt(Lc / g), M * g
    xs = np.array([Lc, 1.0, Vc, Vc, 1 / Tc])
    cs = np.array([0.1, Fc])
    zs = np.array([0.1] * 4 + [Fc / 4] * 4)
    gs = np.array([Fc, Fc * Lc, Fc * Lc, Fc, Fc, Fc, Fc, Fc * Lc])   # algebraic residual scales
    return dict(x=xs, c=cs, z=zs, alg=gs, Fc=Fc, Lc=Lc, Tc=Tc, Vc=Vc)


def qss_guess(track: Track, car: CarParams, s: np.ndarray, mu: float = 1.5, umax: float = 100.0) -> np.ndarray:
    """Point-mass speed profile on the centre line (lateral limit + forward/backward passes, g-g ellipse)."""
    C = np.abs(track.at(s, "C"))
    hk = np.diff(np.r_[s, s[0] + track.L])   # interval k -> k+1
    kA = 0.5 * car.rho * car.ClA / car.M
    kD = 0.5 * car.rho * car.CdA / car.M
    den = np.maximum(C - mu * kA, 1e-9)
    ulat = np.minimum(np.sqrt(mu * car.g / den), umax)
    u = ulat.copy()
    N = len(s)
    rear = car.a / car.w
    for _ in range(2):   # periodic: two sweeps
        for k in range(2 * N):
            i, j = k % N, (k + 1) % N
            ay = u[i] ** 2 * C[i]
            aymax = mu * (car.g + kA * u[i] ** 2)
            r = math.sqrt(max(0.0, 1 - (ay / aymax) ** 2))
            ax = min(r * mu * rear * aymax, car.Pmax / (car.M * u[i])) - kD * u[i] ** 2
            u[j] = min(u[j], math.sqrt(max(u[i] ** 2 + 2 * ax * hk[i], 1.0)))
        for k in range(2 * N, 0, -1):
            i, j = k % N, (k - 1) % N
            ay = u[i] ** 2 * C[i]
            aymax = mu * (car.g + kA * u[i] ** 2)
            r = math.sqrt(max(0.0, 1 - (ay / aymax) ** 2))
            ax = r * aymax + kD * u[i] ** 2
            u[j] = min(u[j], math.sqrt(u[i] ** 2 + 2 * ax * hk[j]))
    return u


def initial_guess(track: Track, car: CarParams, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(NX+NC+NZ, N) guess on the centre line from a speed profile u(s)."""
    C = track.at(s, "C")
    hk = np.diff(np.r_[s, s[0] + track.L])
    du = (np.roll(u, -1) - np.roll(u, 1)) / (hk + np.roll(hk, 1))
    T = car.M * u * du + 0.5 * car.rho * car.CdA * u ** 2
    W = np.zeros((NX + NC + NZ, len(s)))
    W[2] = u
    W[4] = C * u
    W[5] = C * car.w
    W[6] = T
    W[11:15] = np.array([static_loads(car, ui) for ui in u]).T
    return W


@dataclass
class Solution:
    s: np.ndarray
    W: np.ndarray
    laptime: float
    penalty: float
    status: str
    success: bool
    iters: int
    t_solve: float
    settings: dict
    car: str
    extras: dict = field(default_factory=dict)

    def state(self, name):
        return self.W[(STATE_NAMES + CTRL_NAMES + ALG_NAMES).index(name)]


def solve(track: Track, car: CarParams, st: SolveSettings, W0: np.ndarray | None = None,
          s0: np.ndarray | None = None) -> Solution:
    """Solve the OCP on a mesh of nominal step st.ds. W0/s0: warm start (periodically interpolated)."""
    s = make_mesh(track, car, st, W0, s0)
    N = len(s)
    hk = np.diff(np.r_[s, track.L])          # interval k -> k+1 (periodic)
    if W0 is None:
        W0 = initial_guess(track, car, s, qss_guess(track, car, s))
    elif s0 is not None:
        W0 = np.array([np.interp(s, np.r_[s0, track.L], np.r_[r, r[0]]) for r in W0])
    sc = _scales(car)
    Sv = np.r_[sc["x"], sc["c"], sc["z"]]
    # one-node model function in scaled variables
    xs, cs_, zs, Cs = ca.SX.sym("x", NX), ca.SX.sym("c", NC), ca.SX.sym("z", NZ), ca.SX.sym("C")
    out = car_model(xs * sc["x"], cs_ * sc["c"], zs * sc["z"], Cs, car, eps=st.eps * sc["Fc"] ** 2)
    fnode = ca.Function("node", [xs, cs_, zs, Cs],
                        [out["dxds"] / sc["x"], out["Sf"], out["alg"] / sc["alg"], out["power"] / car.Pmax])
    fmap = fnode.map(N, "serial")
    V = ca.MX.sym("V", (NX + NC + NZ) * N)
    Wm = ca.reshape(V, NX + NC + NZ, N)
    X, Cc, Z = Wm[:NX, :], Wm[NX:NX + NC, :], Wm[NX + NC:, :]
    Cv = track.at(s, "C").reshape(1, -1)
    F, Sf, ALG, P = fmap(X, Cc, Z, Cv)
    nxt = lambda A: ca.horzcat(A[:, 1:], A[:, :1])
    H = ca.DM(hk.reshape(1, -1))
    defect = nxt(X) - X - ca.repmat(H / 2, NX, 1) * (F + nxt(F))
    lap = ca.sum2(H * (Sf + nxt(Sf)) / 2)
    dU = (nxt(Cc) - Cc) / ca.repmat(H, NC, 1)
    pen = (st.w_ddelta * (sc["c"][0] ** 2) * ca.sum2(H * dU[0, :] ** 2) + st.w_dT * ca.sum2(H * dU[1, :] ** 2))
    g = ca.vertcat(ca.vec(defect), ca.vec(ALG), ca.vec(P))
    lbg = np.r_[np.zeros(NX * N + NZ * N), -np.inf * np.ones(N)]
    ubg = np.r_[np.zeros(NX * N + NZ * N), np.ones(N)]
    lo, hi = track.corridor(s, car.half_width, st.corridor_fixed)
    lbw = np.zeros((NX + NC + NZ, N))
    ubw = np.zeros((NX + NC + NZ, N))
    rows = {  # physical bounds
        0: (lo, hi), 1: (-1.0, 1.0), 2: (st.u_min, 120.0), 3: (-20.0, 20.0), 4: (-3.0, 3.0),
        5: (-0.6, 0.6), 6: (-8 * sc["Fc"], 3 * sc["Fc"]),
    }
    for i in range(7, 11):
        rows[i] = (-st.kappa_max, st.kappa_max)
    for i in range(11, 15):
        rows[i] = (st.N_min, 6 * sc["Fc"])
    for i, (l, u_) in rows.items():
        lbw[i], ubw[i] = l, u_
    lbw /= Sv[:, None]
    ubw /= Sv[:, None]
    x0 = np.clip(W0 / Sv[:, None], lbw, ubw)
    nlp = {"x": V, "f": lap + pen, "g": g}
    opts = {"ipopt.linear_solver": "mumps", "ipopt.max_iter": st.max_iter, "ipopt.tol": st.tol,
            "ipopt.print_level": st.print_level, "print_time": 0, "ipopt.mu_strategy": "adaptive"}
    solver = ca.nlpsol("pl14", "ipopt", nlp, opts)
    t0 = time.time()
    sol = solver(x0=ca.vec(x0), lbx=ca.vec(lbw), ubx=ca.vec(ubw), lbg=lbg, ubg=ubg)
    t_solve = time.time() - t0
    stats = solver.stats()
    Wopt = np.array(ca.reshape(sol["x"], NX + NC + NZ, N)) * Sv[:, None]
    f_lap = ca.Function("lp", [V], [lap, pen])
    lap_v, pen_v = (float(v) for v in f_lap(sol["x"]))
    Sf_v = np.array(ca.Function("sf", [V], [Sf])(sol["x"])).ravel()
    P_v = np.array(ca.Function("p", [V], [P])(sol["x"])).ravel() * car.Pmax
    gmax = float(np.max(np.abs(np.array(sol["g"]).ravel()[: (NX + NZ) * N])))
    return Solution(s=s, W=Wopt, laptime=lap_v, penalty=pen_v, status=stats["return_status"],
                    success=stats["return_status"] == "Solve_Succeeded", iters=int(stats["iter_count"]),
                    t_solve=t_solve, settings=asdict(st), car=car.name,
                    extras=dict(Sf=Sf_v, power=P_v, max_eq_residual_scaled=gmax, corridor=(lo, hi), h=hk, N=N,
                                ds_mean=float(hk.mean()), ds_min=float(hk.min())))


def cumulative_time(sol: Solution) -> np.ndarray:
    """t at s_0..s_{N-1} and at s = L (N+1 values) by the trapezoid rule on Sf, t(0) = 0."""
    Sf = sol.extras["Sf"]
    return np.r_[0.0, np.cumsum(sol.extras["h"] * 0.5 * (Sf + np.roll(Sf, -1)))]


def make_mesh(track: Track, car: CarParams, st: SolveSettings, W0=None, s0=None) -> np.ndarray:
    """Uniform mesh (N = round(L/ds)) or, if st.mesh_u_ref is set, nodes with local step
    h(s) = ds * clip((u(s)/mesh_u_ref)^mesh_pow, mesh_min_frac, 1), u(s) from the warm start or the QSS guess."""
    if st.mesh_u_ref is None:
        N = int(round(track.L / st.ds))
        return np.arange(N) * (track.L / N)
    sf = np.linspace(0.0, track.L, int(track.L / 0.1), endpoint=False)
    if W0 is not None and s0 is not None:
        u = np.interp(sf, np.r_[s0, track.L], np.r_[W0[2], W0[2][0]])
    else:
        u = qss_guess(track, car, np.arange(int(track.L)) * track.L / int(track.L))
        u = np.interp(sf, np.arange(len(u)) * track.L / len(u), u)
    dens = 1.0 / (st.ds * np.clip((u / st.mesh_u_ref) ** st.mesh_pow, st.mesh_min_frac, 1.0))
    Q = np.r_[0.0, np.cumsum(dens * (track.L / len(sf)))]
    N = int(math.ceil(Q[-1]))
    return np.interp(np.arange(N) * Q[-1] / N, Q, np.r_[sf, track.L])
