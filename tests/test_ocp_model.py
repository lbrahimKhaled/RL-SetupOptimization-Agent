"""Sign conventions and consistency of the PL14 model (rlsetup/ocp/pl14_model.py)."""
import math

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from rlsetup.ocp.pl14_model import (TyreParams, car_model, pl14_car, sf70h_tum_car, smax, smin,
                                    static_loads, tyre_forces)

TP = TyreParams()


def f(k, a, N, tp=TP):
    Fx, Fy = tyre_forces(k, a, N, tp)
    return float(Fx), float(Fy)


def test_free_rolling_zero_force():
    assert f(0.0, 0.0, 4000.0) == (0.0, 0.0)


def test_drive_brake_and_side_force_signs():
    assert f(0.05, 0.0, 4000.0)[0] > 0        # drive slip -> forward force
    assert f(-0.05, 0.0, 4000.0)[0] < 0       # brake slip -> rearward force
    assert f(0.0, 0.05, 4000.0)[1] < 0        # wheel sliding right -> force to the left
    assert f(0.0, -0.05, 4000.0)[1] > 0


def test_symmetry():
    for k, a in [(0.03, 0.07), (-0.08, 0.02), (0.1, -0.1)]:
        Fx1, Fy1 = f(k, a, 3000.0)
        Fx2, Fy2 = f(k, -a, 3000.0)
        assert Fx1 == pytest.approx(Fx2) and Fy1 == pytest.approx(-Fy2)
        Fx3, Fy3 = f(-k, a, 3000.0)
        assert Fx3 == pytest.approx(-Fx1) and Fy3 == pytest.approx(Fy1)


# As printed (A.13-A.14) S = pi/(2 atan Q) puts the force peak at rho* = tan(pi/(2Q))/S = 0.750 (Q = 1.9),
# i.e. at 0.75 * kappa_max / alpha_max, not at kappa_max. Peak mu is exact. Pinned here, documented in REPORT.
RHO_PK = math.tan(math.pi / (2 * 1.9)) / (math.pi / (2 * math.atan(1.9)))


@pytest.mark.parametrize("N, mux, kpk, muy, apk", [(2000.0, 1.75, 0.11, 1.80, 9.0), (6000.0, 1.40, 0.10, 1.45, 8.0)])
def test_peak_mu_and_slip_scale_with_load(N, mux, kpk, muy, apk):
    kpk, apk = kpk * RHO_PK, apk * RHO_PK
    ks = np.linspace(0.0, 0.4, 4001)
    Fx = np.array([f(k, 0.0, N)[0] for k in ks])
    i = Fx.argmax()
    assert Fx[i] / N == pytest.approx(mux, rel=1e-4)
    assert ks[i] == pytest.approx(kpk, abs=2e-4)
    al = np.radians(np.linspace(0.0, 30.0, 3001))
    Fy = -np.array([f(0.0, a, N)[1] for a in al])
    j = Fy.argmax()
    assert Fy[j] / N == pytest.approx(muy, rel=1e-4)
    assert math.degrees(al[j]) == pytest.approx(apk, abs=0.02)


def test_combined_slip_inside_friction_ellipse():
    N = 4000.0
    for k, a in [(0.11, 0.15), (-0.2, 0.3), (0.05, -0.05)]:
        Fx, Fy = f(k, a, N)
        assert (Fx / (TP.lerp(N, TP.mux1, TP.mux2) * N)) ** 2 + (Fy / (TP.lerp(N, TP.muy1, TP.muy2) * N)) ** 2 <= 1 + 1e-9


def test_smooth_max_min():
    assert float(smax(10.0, 1e-3)) == pytest.approx(10.0, abs=1e-4)
    assert float(smax(-10.0, 1e-3)) == pytest.approx(0.0, abs=1e-4)
    assert float(smin(-10.0, 1e-3)) == pytest.approx(-10.0, abs=1e-4)
    assert float(smin(10.0, 1e-3)) == pytest.approx(0.0, abs=1e-4)


def _eval(car, x, ctrl, z, C=0.0):
    out = car_model(ca.DM(x), ca.DM(ctrl), ca.DM(z), C, car)
    return {k: (np.array(ca.DM(v)).ravel() if not isinstance(v, dict) else v) for k, v in out.items()}


def test_static_straight_line_equilibrium():
    car = pl14_car()
    u = 50.0
    N = static_loads(car, u)
    out = _eval(car, [0, 0, u, 0, 0], [0, 0], [0, 0, 0, 0] + N)
    res = out["alg"].copy()
    assert res[2] == pytest.approx(-car.h * out["Fax"][0])   # drag (in Fx, eq 18) shifts load in eq 25
    res[2] = 0.0
    assert np.allclose(res, 0.0, atol=0.02)                   # 0.0095 N from the eq-48 smoothing at T=0
    assert out["Sf"][0] == pytest.approx(1 / u)
    dx = out["dxds"]
    assert dx[0] == pytest.approx(0) and dx[1] == pytest.approx(0) and dx[4] == pytest.approx(0)
    assert dx[2] == pytest.approx(-car.CdA * 0.5 * car.rho * u / car.M)   # drag deceleration / u
    assert sum(N) == pytest.approx(car.M * car.g + car.ClA * 0.5 * car.rho * u * u)
    assert N[2] > N[0]                                                         # rear-biased static load


def test_right_turn_conventions():
    """Steering right (delta>0) at speed gives a rightward force and clockwise (positive) yaw moment."""
    car = pl14_car()
    u = 40.0
    N = static_loads(car, u)
    out = _eval(car, [0, 0, u, 0, 0], [0.02, 0], [0, 0, 0, 0] + N)
    assert out["Fy"][0] > 0 and out["Mz"][0] > 0
    # curvilinear: car aligned with track in a right turn, positive yaw rate -> xi stays if om = C u
    C = 0.01
    out = _eval(car, [0, 0, u, 0, C * u], [0, 0], [0, 0, 0, 0] + N, C=C)
    assert out["dxds"][1] == pytest.approx(0, abs=1e-12)
    # mass centre right of centre line (inside of a right turn) -> shorter path, smaller dt/ds
    out2 = _eval(car, [1.0, 0, u, 0, C * u], [0, 0], [0, 0, 0, 0] + N, C=C)
    assert out2["Sf"][0] < out["Sf"][0]


def test_load_transfer_signs():
    """Rightward net force -> left wheels heavier; forward force -> rear heavier (eqs 24-25 residuals)."""
    car = pl14_car()
    N = static_loads(car, 0.0)
    # a load set shifted to the left satisfies (24) only with Fy > 0
    dN = 300.0
    z = [0, 0, 0, 0, N[0] - dN, N[1] + dN, N[2] - dN, N[3] + dN]
    out = _eval(car, [0, 0, 1e-3 + 10, 0, 0], [0, 0], z)
    # residual (24) = wr(Nrl-Nrr)+wf(Nfl-Nfr) - h Fy with Fy = 0 -> positive -> needs Fy>0 to vanish
    assert out["alg"][1] > 0


def test_diff_and_brake_split():
    car = pl14_car(beta=0.6)
    u = 50.0
    N = static_loads(car, u)
    T = -10000.0
    # front: equal forces carrying 60 %, rear 40 %: solve slips by bisection on straight line
    def kap_for(F, Nw):
        lo, hi = -0.09, 0.09
        for _ in range(80):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if f(m, 0.0, Nw)[0] < F else (lo, m)
        return 0.5 * (lo + hi)
    kf = kap_for(0.3 * T, N[0])
    kr = kap_for(0.2 * T, N[2])
    out = _eval(car, [0, 0, u, 0, 0], [0, T], [kf, kf, kr, kr] + N)
    assert np.allclose(out["alg"][4:], 0.0, atol=0.5)   # force residuals within the eps smoothing
    assert out["power"][0] == pytest.approx(0.0, abs=1.0)


def test_sf70h_mapping():
    car = sf70h_tum_car()
    assert car.M == 733.0 and car.Pmax == 575e3
    assert car.tyre_f.mux1 == pytest.approx(1.70) and car.tyre_r.muy2 == pytest.approx(2.0)
    assert car.aA == pytest.approx(1.977, abs=1e-3)
