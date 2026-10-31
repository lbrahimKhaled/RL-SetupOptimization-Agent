"""A-posteriori checks of an OCP solution: re-integration with a fine DAE integrator (IDAS).

Each mesh interval [s_k, s_k+h] is re-integrated from the collocated state x_k with the controls linearly
interpolated between nodes (the trapezoidal method's implied control) and the algebraic variables
solved exactly (IDAS, rtol = atol = 1e-10). Reported: per-interval state mismatch at s_k+h, chained
(open-loop) mismatch over windows, and the lap time from the re-integrated dt/ds quadrature.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .pl14_model import CarParams, car_model
from .solve import NC, NX, NZ, STATE_NAMES, Solution
from .track import Track


def _interval_integrator(car: CarParams, eps_N2: float):
    x = ca.SX.sym("x", NX)
    z = ca.SX.sym("z", NZ)
    tau = ca.SX.sym("tau")                        # normalised local s in [0, 1]
    p = ca.SX.sym("p", 2 * NC + 3)                # c_k, c_{k+1}, C_k, C_{k+1}, h_k
    h = p[2 * NC + 2]
    lam = tau
    c = (1 - lam) * p[0:NC] + lam * p[NC:2 * NC]
    Cs = (1 - lam) * p[2 * NC] + lam * p[2 * NC + 1]
    out = car_model(x, c, z, Cs, car, eps=eps_N2)
    sc = ca.DM([car.M * car.g] + [car.M * car.g * car.w] * 2 + [car.M * car.g] * 4 + [car.M * car.g * car.w])
    xa = ca.vertcat(x, ca.SX.sym("t"))
    dae = {"x": xa, "z": z, "p": p, "t": tau, "ode": h * ca.vertcat(out["dxds"], out["Sf"]), "alg": out["alg"] / sc}
    return ca.integrator("F", "idas", dae, 0.0, 1.0, {"abstol": 1e-10, "reltol": 1e-10, "max_num_steps": 20000})


def reintegrate(sol: Solution, track: Track, car: CarParams, window_m: float = 50.0) -> dict:
    h = sol.extras["h"]
    N = sol.extras["N"]
    W = sol.W
    X, Cc, Z = W[:NX], W[NX:NX + NC], W[NX + NC:]
    Cv = track.at(sol.s, "C")
    nx = lambda A: np.roll(A, -1, axis=-1)
    P = np.vstack([Cc, nx(Cc), Cv, nx(Cv), h])
    F = _interval_integrator(car, sol.settings["eps"] * (car.M * car.g) ** 2)
    xf = np.full((NX + 1, N), np.nan)
    fails = []
    for k in range(N):   # per interval, so one IDAS failure does not void the check
        try:
            rr = F(x0=np.r_[X[:, k], 0.0], z0=Z[:, k], p=P[:, k])
            xf[:, k] = np.array(rr["xf"]).ravel()
        except RuntimeError:
            fails.append(k)
    ok = ~np.isnan(xf[0])
    err = (xf[:NX] - nx(X))[:, ok]
    # lap time: re-integrated dt where IDAS succeeded, collocation trapezoid elsewhere
    Sf = sol.extras["Sf"]
    t_re = float(np.where(ok, xf[NX], 0.5 * h * (Sf + nx(Sf))).sum())
    # chained open-loop windows
    k_win = max(1, int(round(window_m / float(np.mean(h)))))
    werr = []
    for k0 in range(0, N - k_win, k_win):
        xk = X[:, k0].copy()
        try:
            for k in range(k0, k0 + k_win):
                rr = F(x0=np.r_[xk, 0.0], z0=Z[:, k], p=P[:, k])
                xk = np.array(rr["xf"]).ravel()[:NX]
        except RuntimeError:
            continue
        werr.append(np.abs(xk - X[:, k0 + k_win]))
    werr = np.array(werr)
    return dict(
        interval_max_abs_err={nm: float(np.abs(err[i]).max()) for i, nm in enumerate(STATE_NAMES)},
        interval_rms_err={nm: float(np.sqrt(np.mean(err[i] ** 2))) for i, nm in enumerate(STATE_NAMES)},
        window_m=k_win * float(np.mean(h)),
        window_max_abs_err={nm: float(werr[:, i].max()) for i, nm in enumerate(STATE_NAMES)},
        window_median_abs_err={nm: float(np.median(werr[:, i])) for i, nm in enumerate(STATE_NAMES)},
        laptime_reintegrated_from_nodes=t_re, laptime_collocation=sol.laptime,
        idas_failed_intervals=len(fails), idas_failed_s=[float(sol.s[k]) for k in fails[:50]],
        windows_ok=int(len(werr)), windows_total=int(len(range(0, N - k_win, k_win))),
    )
