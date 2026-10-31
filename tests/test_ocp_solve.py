"""Track smoothing and a small OCP solve on a circular track (fast)."""
import math

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from rlsetup.ocp.pl14_model import pl14_car
from rlsetup.ocp.solve import SolveSettings, solve
from rlsetup.ocp.track import Track

R = 60.0


@pytest.fixture(scope="module")
def circle(tmp_path_factory):
    p = tmp_path_factory.mktemp("trk") / "circle.csv"
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    # counter-clockwise in the map frame = LEFT turn -> PL14 curvature C = -1/R
    np.savetxt(p, np.c_[R * np.cos(th), R * np.sin(th), np.full(200, 6.0), np.full(200, 5.0)], delimiter=",")
    return Track.from_csv(p, ds=2.0)


def test_track_curvature_sign_and_length(circle):
    assert circle.info["status"] == "Solve_Succeeded"
    assert circle.L == pytest.approx(2 * np.pi * R, rel=1e-3)
    assert np.allclose(circle.C, -1 / R, rtol=1e-2)   # polyline input: +-0.5 % ripple
    lo, hi = circle.corridor(circle.s[:3], car_half_width=0.9)
    assert np.allclose(lo, -(5.0 - 0.9)) and np.allclose(hi, 6.0 - 0.9)   # left width 5, right width 6
    lo, hi = circle.corridor(circle.s[:3], fixed=4.0)
    assert np.allclose(lo, -4.0) and np.allclose(hi, 4.0)


def test_circle_ocp(circle):
    car = pl14_car()
    sol = solve(circle, car, SolveSettings(ds=6.0, corridor_fixed=0.5))
    assert sol.success, sol.status
    u = sol.state("u")
    # steady cornering: near-constant speed, inside the friction band of a point mass with mu in [1.4, 1.8]
    kA = 0.5 * car.rho * car.ClA / car.M
    R_eff = R - 0.5            # car hugs the inside (left) edge of the 0.5 m corridor
    lim = lambda mu: math.sqrt(mu * car.g / (1 / R_eff - mu * kA))
    assert lim(1.4) < u.mean() < lim(1.8)
    assert np.ptp(u) < 0.5
    assert np.all(sol.state("n") < -0.45)      # inside of a left turn is n < 0
    assert sol.laptime == pytest.approx(np.sum(sol.extras["h"] * sol.extras["Sf"]))


def test_reintegration_matches_collocation(circle):
    from rlsetup.ocp.verify import reintegrate
    car = pl14_car()
    sol = solve(circle, car, SolveSettings(ds=6.0, corridor_fixed=0.5))
    rep = reintegrate(sol, circle, car, window_m=30.0)
    assert rep["interval_max_abs_err"]["u"] < 0.05          # m/s, steady state: trapezoid nearly exact
    assert rep["interval_max_abs_err"]["n"] < 0.01
    assert rep["laptime_reintegrated_from_nodes"] == pytest.approx(sol.laptime, rel=1e-3)


def test_speed_adapted_mesh(circle):
    from rlsetup.ocp.solve import make_mesh
    car = pl14_car()
    st = SolveSettings(ds=6.0, corridor_fixed=0.5, mesh_u_ref=60.0)
    s = make_mesh(circle, car, st)
    assert s[0] == 0.0 and np.all(np.diff(s) > 0) and s[-1] < circle.L
    sol = solve(circle, car, st)
    assert sol.success
    uni = solve(circle, car, SolveSettings(ds=6.0, corridor_fixed=0.5))
    # small non-convex problem: IPOPT can stop in a slightly non-steady local optimum (seen: +0.3 %)
    assert sol.laptime == pytest.approx(uni.laptime, rel=5e-3)
