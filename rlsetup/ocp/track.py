"""Curvilinear, periodic track for the PL14 OCP, from a racetrack-database centre line.

Input CSV (TUMFTM racetrack-database): x_m, y_m, w_tr_right_m, w_tr_left_m, closed loop, points in the
driving direction, map frame x east / y north.

Curvature smoothing (PL14 §2.1.1 eqs 9-13, auxiliary OCP): states theta, C, x, y on a uniform s grid,
control dC/ds; minimise sum ds*(r (dC/ds)^2 + (x-xc)^2 + (y-yc)^2) subject to trapezoidal kinematics
dx/ds = cos theta, dy/ds = sin theta, dtheta/ds = C, periodic closure. PL14 used r = 1e6 (units not
stated). Read as SI (m^6) it gives a 0.25 m RMS / 1.9 m max fit error on Catalunya; read in unit-wheelbase
units (1e6 * 3.4^6 m^6) it cuts corners by 17 m. ASSUMPTION (default R_DEFAULT = 1e4 m^6): fit within
0.04 m RMS / 0.5 m max on Catalunya and Shanghai (sweep in docs/results/limebeer_ocp/PROGRESS.md).

Sign conventions on output (PL14 SAE frame): C [1/m] > 0 for a RIGHT turn (C = -d(theta_map)/ds),
n > 0 to the right, so the mass-centre corridor is  -(w_left - margin) <= n <= (w_right - margin).
Widths are taken from the raw centre line and not re-projected onto the smoothed line (offsets are cm).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRACKS_DIR = Path("/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap/vendor/racetrack-database/tracks")
R_DEFAULT = 1e4   # m^6, ASSUMPTION (see module docstring)


def load_csv(name_or_path: str | Path) -> np.ndarray:
    p = Path(name_or_path)
    if not p.suffix:
        p = TRACKS_DIR / f"{name_or_path}.csv"
    return np.loadtxt(p, delimiter=",", comments="#")


def _periodic_interp(s_grid: np.ndarray, s_data: np.ndarray, vals: np.ndarray, L: float) -> np.ndarray:
    return np.interp(np.mod(s_grid, L), np.r_[s_data, L], np.r_[vals, vals[0]])


def smooth_centreline(xy: np.ndarray, ds: float = 3.0, r: float = R_DEFAULT) -> dict:
    """PL14 auxiliary OCP (casadi/IPOPT). Returns s, x, y, theta_map, kappa_map (left-positive) and fit RMS."""
    import casadi as ca
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    s_raw = np.r_[0.0, np.cumsum(seg)[:-1]]
    L = float(seg.sum())
    N = int(round(L / ds))
    h = L / N
    s = np.arange(N) * h
    xc = _periodic_interp(s, s_raw, xy[:, 0], L)
    yc = _periodic_interp(s, s_raw, xy[:, 1], L)
    th0 = np.unwrap(np.arctan2(np.gradient(np.r_[yc, yc[:1]])[:-1], np.gradient(np.r_[xc, xc[:1]])[:-1]))
    turns = round((th0[-1] - th0[0]) / (2 * np.pi))
    w = ca.SX.sym("w", 4 * N)
    X, Y, TH, C = (w[i * N:(i + 1) * N] for i in range(4))
    nxt = lambda v: ca.vertcat(v[1:], v[:1])
    wrap = np.zeros(N)
    wrap[-1] = 2 * np.pi * turns
    g = ca.vertcat(nxt(X) - X - h / 2 * (ca.cos(TH) + ca.cos(nxt(TH))),
                   nxt(Y) - Y - h / 2 * (ca.sin(TH) + ca.sin(nxt(TH))),
                   nxt(TH) + wrap - TH - h / 2 * (C + nxt(C)))
    dC = (nxt(C) - C) / h
    J = h * ca.sumsqr(X - xc) + h * ca.sumsqr(Y - yc) + h * r * ca.sumsqr(dC)
    dth = np.diff(np.r_[th0, th0[0] + 2 * np.pi * turns]) / h
    w0 = np.r_[xc, yc, th0, 0.5 * (dth + np.r_[dth[-1], dth[:-1]])]
    solver = ca.nlpsol("cl", "ipopt", {"x": w, "f": J, "g": g},
                       {"ipopt.print_level": 0, "print_time": 0, "ipopt.linear_solver": "mumps", "ipopt.tol": 1e-9})
    sol = solver(x0=w0, lbg=0, ubg=0)
    status = solver.stats()["return_status"]
    v = np.array(sol["x"]).ravel()
    x, y, th, k = (v[i * N:(i + 1) * N] for i in range(4))
    err = np.hypot(x - xc, y - yc)
    return dict(s=s, x=x, y=y, theta_map=th, kappa_map=k, L=L, ds=h, status=status,
                rms_fit=float(np.sqrt(np.mean(err ** 2))), max_fit=float(err.max()), s_raw=s_raw)


@dataclass
class Track:
    """Periodic curvilinear track. All arrays on the smoothing grid s [m] (uniform, length L)."""
    name: str
    s: np.ndarray
    C: np.ndarray            # PL14 sign: > 0 right turn [1/m]
    w_right: np.ndarray      # [m] centre line to right edge
    w_left: np.ndarray       # [m]
    x: np.ndarray
    y: np.ndarray
    theta_map: np.ndarray
    L: float
    info: dict

    @classmethod
    def from_csv(cls, name: str, ds: float = 3.0, r: float = R_DEFAULT) -> "Track":
        data = load_csv(name)
        sm = smooth_centreline(data[:, :2], ds=ds, r=r)
        wr = _periodic_interp(sm["s"], sm["s_raw"], data[:, 2], sm["L"])
        wl = _periodic_interp(sm["s"], sm["s_raw"], data[:, 3], sm["L"])
        info = {k: sm[k] for k in ("status", "rms_fit", "max_fit", "ds")} | {"r_m6": r, "source": str(name)}
        return cls(name, sm["s"], -sm["kappa_map"], wr, wl, sm["x"], sm["y"], sm["theta_map"], sm["L"], info)

    def at(self, s_query: np.ndarray, field: str) -> np.ndarray:
        """Periodic interpolation of a field (cubic periodic spline for C, linear for widths)."""
        v = getattr(self, field)
        if field == "C":
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(np.r_[self.s, self.L], np.r_[v, v[0]], bc_type="periodic")
            return cs(np.mod(s_query, self.L))
        return _periodic_interp(s_query, self.s, v, self.L)

    def corridor(self, s_query: np.ndarray, car_half_width: float | None = None, fixed: float | None = None):
        """Bounds on n [m] (n > 0 right). Either track half-width minus car half-width, or +-fixed (ML16 4 m)."""
        if fixed is not None:
            o = np.full_like(np.asarray(s_query, float), fixed)
            return -o, o
        return -(self.at(s_query, "w_left") - car_half_width), self.at(s_query, "w_right") - car_half_width
