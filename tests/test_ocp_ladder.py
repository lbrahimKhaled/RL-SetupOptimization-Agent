"""OCP ladder: OCD MF5.2 port, steady-state loads, LSD, engine map, L0 equivalence, small solve.

Native checks use OCD itself (ocd_native through ocd_lap's OCDBackend). OCD integrates with DoPri45
and logs the forces of the 6th stage but the state after the step, so a snapshot is made consistent
by one final step of 1e-12 s (stage state = logged state to ~1e-10).
"""
import math

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from ocd_lap.backend import Inputs, OCDBackend, StartPose  # noqa: E402
from rlsetup.driving.setups import get  # noqa: E402
from rlsetup.ocp.ladder_model import (FEATURES, LadderCar, engine_capacity, ladder_model,  # noqa: E402
                                      pl14_loads)
from rlsetup.ocp.ocd_car import OCD_WHEELS, casadi_ops, from_backend, mf52, ocd_tyre_forces  # noqa: E402
from rlsetup.ocp.pl14_model import car_model, sf70h_tum_car  # noqa: E402

SIG = dict(front_left="FL", front_right="FR", rear_left="RL", rear_right="RR")


@pytest.fixture(scope="module")
def ocd():
    return from_backend("p1_TUM")


def _backend(speed):
    b = OCDBackend(powertrain="engine", car="sf70h_tum", ers=False, max_brake_torque_Nm=15000)
    b.reset(get("p1_TUM"), StartPose(speed=speed))
    return b


def _freeze(b, inp):
    """One 1e-12 s step: logged state and logged intermediate results then belong together."""
    p = dict(b.params)
    p["integration_step_size_s"] = 1e-12
    b.car.set_parameters(p)
    b.step(inp, 1)


def _mf52_cpp(F_z, kappa, alpha, gamma, p):
    """Independent line-by-line transliteration of mf_52.cpp (scalar, `math`), for cross-checking."""
    def pure_x(F_z, kappa):
        F_z_o_a = p["LFZO"] * p["FNOMIN"]; dfz = (F_z - F_z_o_a) / F_z_o_a
        S_H_x = (p["PHX1"] + p["PHX2"] * dfz) * p["LHX"]
        S_V_x = F_z * (p["PVX1"] + p["PVX2"] * dfz) * p["LVX"] * p["LMUX"]
        kappa_x = kappa + S_H_x; C_x = p["PCX1"] * p["LCX"]
        D_x = (p["PDX1"] + p["PDX2"] * dfz) * p["LMUX"] * F_z
        E_x = (p["PEX1"] + p["PEX2"] * dfz + p["PEX3"] * dfz ** 2) * (1.0 - p["PEX4"] * math.copysign(1.0, kappa_x)) * p["LEX"]
        K_x = F_z * (p["PKX1"] + p["PKX2"] * dfz) * math.exp(p["PKX3"] * dfz) * p["LKX"]
        B_x = K_x / (C_x * D_x)
        return D_x * math.sin(C_x * math.atan(B_x * kappa_x - E_x * (B_x * kappa_x - math.atan(B_x * kappa_x)))) + S_V_x

    def pure_y(F_z, alpha, gamma):
        gamma_s = math.sin(gamma); F_z_o_a = p["LFZO"] * p["FNOMIN"]; dfz = (F_z - F_z_o_a) / F_z_o_a
        K_y_g_o = F_z * (p["PKY6"] + p["PKY7"] * dfz) * p["LKYG"]
        S_V_y_g = F_z * (p["PVY3"] + p["PVY4"] * dfz) * gamma_s * p["LKYG"] * p["LMUY"]
        S_V_y = F_z * (p["PVY1"] + p["PVY2"] * dfz) * p["LVY"] * p["LMUY"] + S_V_y_g
        K_y = (p["PKY1"] * F_z_o_a * (1.0 - p["PKY3"] * abs(gamma_s))
               * math.sin(p["PKY4"] * math.atan(F_z / ((p["PKY2"] + p["PKY5"] * gamma_s ** 2) * F_z_o_a))) * p["LKY"])
        S_H_y = (p["PHY1"] + p["PHY2"] * dfz) * p["LHY"] + (K_y_g_o * gamma_s - S_V_y_g) / K_y
        C_y = p["PCY1"] * p["LCY"]
        D_y = ((p["PDY1"] + p["PDY2"] * dfz) * (1.0 - p["PDY3"] * gamma_s ** 2)) * p["LMUY"] * F_z
        B_y = K_y / (C_y * D_y); alpha_y = alpha + S_H_y
        E_y = ((p["PEY1"] + p["PEY2"] * dfz) * (1.0 + p["PEY5"] * gamma_s ** 2 - (p["PEY3"] + p["PEY4"] * gamma_s)
                                                 * math.copysign(1.0, alpha_y)) * p["LEY"])
        return D_y * math.sin(C_y * math.atan(B_y * alpha_y - E_y * (B_y * alpha_y - math.atan(B_y * alpha_y)))) + S_V_y

    gamma_s = math.sin(gamma); F_z_o_a = p["LFZO"] * p["FNOMIN"]; dfz = (F_z - F_z_o_a) / F_z_o_a
    F_x_o, F_y_o = pure_x(F_z, kappa), pure_y(F_z, alpha, gamma)
    S_H_x_a = p["RHX1"]; E_x_a = p["REX1"] + p["REX2"] * dfz; C_x_a = p["RCX1"]
    B_x_a = (p["RBX1"] + p["RBX3"] * gamma_s ** 2) * math.cos(math.atan(p["RBX2"] * kappa)) * p["LXAL"]
    alpha_S = alpha + S_H_x_a
    G_x_a_o = math.cos(C_x_a * math.atan(B_x_a * S_H_x_a - E_x_a * (B_x_a * S_H_x_a - math.atan(B_x_a * S_H_x_a))))
    G_x_a = math.cos(C_x_a * math.atan(B_x_a * alpha_S - E_x_a * (B_x_a * alpha_S - math.atan(B_x_a * alpha_S)))) / G_x_a_o
    mu_y = ((p["PDY1"] + p["PDY2"] * dfz) / (1.0 + p["PDY3"] * gamma_s ** 2)) * p["LMUY"]
    D_V_y_k = mu_y * F_z * (p["RVY1"] + p["RVY2"] * dfz + p["RVY3"] * gamma_s) * math.cos(math.atan(p["RVY4"] * alpha))
    S_V_y_k = D_V_y_k * math.sin(p["RVY5"] * math.atan(p["RVY6"] * kappa)) * p["LVYKA"]
    S_H_y_k = p["RHY1"] + p["RHY2"] * dfz; E_y_k = p["REY1"] + p["REY2"] * dfz; C_y_k = p["RCY1"]
    B_y_k = (p["RBY1"] + p["RBY4"] * gamma_s ** 2) * math.cos(math.atan(p["RBY2"] * (alpha - p["RBY3"]))) * p["LYKA"]
    kappa_S = kappa + S_H_y_k
    G_y_k_o = math.cos(C_y_k * math.atan(B_y_k * S_H_y_k - E_y_k * (B_y_k * S_H_y_k - math.atan(B_y_k * S_H_y_k))))
    G_y_k = math.cos(C_y_k * math.atan(B_y_k * kappa_S - E_y_k * (B_y_k * kappa_S - math.atan(B_y_k * kappa_S)))) / G_y_k_o
    return G_x_a * F_x_o, G_y_k * F_y_o + S_V_y_k


def test_mf52_matches_native_ocd(ocd):
    """>= 20 (kappa, alpha, Fz) points from OCD itself, both sides, drive/brake/lock: rel err < 1e-6."""
    st = [f"vehicle_dynamics/x_vec/{q}_tire_{s}" for s in SIG.values() for q in ("kappa", "alpha")]
    im = [f"vehicle_dynamics/imr/{q}" for w in SIG for q in (f"vertical_tire_force_N/{w}",
          f"tire_forces_tire_frame_N/{w}/longitudinal", f"tire_forces_tire_frame_N/{w}/lateral")]
    pts = []
    for k, (steer, th, br) in enumerate([(0.05, .6, 0), (-.1, 1, 0), (.15, 0, .5), (-.05, 0, 1.), (.2, .3, 0), (-.25, .8, 0)]):
        b = _backend(25. + 10 * k)
        b.step(Inputs(steer, th, br), 60 + 20 * k)
        _freeze(b, Inputs(steer, th, br))
        S = np.array(b.car.signals(st)).reshape(4, 2)
        V = np.array(b.car.signals(im)).reshape(4, 3)
        for i, w in enumerate(OCD_WHEELS):
            Fx, Fy = ocd_tyre_forces(S[i, 0], S[i, 1], V[i, 0], ocd.tyre(w), mirror=w.endswith("right"))
            pts.append((S[i, 0], S[i, 1], V[i, 0], Fx, Fy, V[i, 1], V[i, 2]))
    P = np.array(pts)
    assert len(P) >= 20
    assert P[:, 0].min() < -0.5 and P[:, 0].max() > 0.05 and np.ptp(P[:, 1]) > 0.2 and np.ptp(P[:, 2]) > 3000
    for col_ours, col_ocd in ((3, 5), (4, 6)):
        err = np.abs(P[:, col_ours] - P[:, col_ocd]) / np.maximum(np.abs(P[:, col_ocd]), 1.0)
        assert err.max() < 1e-6, err.max()


def test_mf52_port_matches_cpp_transliteration_and_casadi(ocd):
    rng = np.random.default_rng(3)
    ops = casadi_ops()
    k, a, fz = ca.SX.sym("k"), ca.SX.sym("a"), ca.SX.sym("fz")
    for wheel in ("front_left", "rear_left"):
        p = ocd.tyre(wheel)
        for gamma in (0.0, 0.03):
            fun = ca.Function("f", [k, a, fz], list(mf52(k, a, fz, p, gamma=gamma, ops=ops)))
            for _ in range(15):
                kap, alp, Fz = rng.uniform(-1, 1), rng.uniform(-.4, .4), rng.uniform(200, 12000)
                ref = _mf52_cpp(Fz, kap, alp, gamma, p)
                got = mf52(kap, alp, Fz, p, gamma=gamma)
                sym = [float(v) for v in fun(kap, alp, Fz)]
                for r, g_, s_ in zip(ref, got, sym):
                    assert g_ == pytest.approx(r, rel=1e-12, abs=1e-9)
                    assert s_ == pytest.approx(r, rel=1e-12, abs=1e-9)


def _steady(steer, th, n=20000):
    b = _backend(40.)
    for _ in range(n // 5000):
        b.step(Inputs(steer, th, 0.), 5000)
    _freeze(b, Inputs(steer, th, 0.))
    return b


@pytest.mark.parametrize("steer,th", [(0.04, .45), (-0.08, .35)])
def test_load_map_matches_native_steady_state(ocd, steer, th):
    lm = ocd.load_map()
    W = OCD_WHEELS
    nm = ([f"vehicle_dynamics/imr/tire_forces_N/{w}/longitudinal" for w in W]
          + [f"vehicle_dynamics/imr/tire_forces_N/{w}/lateral" for w in W]
          + ["vehicle_dynamics/imr/aero_force_N/z", "vehicle_dynamics/imr/aero_torque_Nm/y"]
          + [f"vehicle_dynamics/imr/vertical_tire_force_N/{w}" for w in W]
          + ["drivetrain/M_LSD_Nm"] + [f"drivetrain/drivetrain_load_torque_per_wheel_Nm/{w}" for w in W[2:]]
          + ["drivetrain/x_vec/omega_diff_rear_radps", "drivetrain/input/transmission_output_torque_Nm"])
    b = _steady(steer, th)
    v = np.array(b.car.signals(nm))
    lfa = ocd.p("aerodynamics.l_f")
    lra = ocd.p("aerodynamics.l") - lfa
    Flf = (v[9] - lra * v[8]) / (lfa + lra)        # aero_z = -(Flf + Flr), torque_y = Flf lfa - Flr lra
    Flr = -v[8] - Flf
    pred = lm["N0"] + lm["A"] @ np.r_[v[:8], Flf, Flr]
    assert np.max(np.abs(pred - v[10:14]) / v[10:14]) < 1e-6
    # LSD steady state: L_RR - L_RL = 2 M_LSD with M_LSD = tanh(omega_diff) lock_drive T_in (no brakes)
    M, L_rl, L_rr, wd, T_in = v[14:19]
    assert M == pytest.approx(math.tanh(wd) * ocd.lsd["lock_drive"] * T_in, rel=1e-9)
    assert L_rr - L_rl == pytest.approx(2 * M, rel=1e-4, abs=0.5)


def test_engine_capacity_matches_powertrain(ocd):
    from ocd_lap.powertrain import Powertrain
    pt = Powertrain(vehicle_mass_kg=ocd.m, wheel_radius_m=ocd.R_r, wheel_inertia_kgm2=2 * (ocd.I_wf + ocd.I_wr))
    wu = ocd.upshift_omega()
    edges = np.r_[0.0, wu, 285.0]     # 285 rad/s = 94 m/s, above the car's top speed
    for g in range(8):
        for om in np.linspace(edges[g] + 4, edges[g + 1] - 4, 3):
            if om < 20:
                continue
            pt.gear = g
            rpm = pt.rpm(om)
            if abs(rpm - 7875) < 400:      # inside the smoothed rpm clip
                continue
            cap, I = (float(v) for v in engine_capacity(ca.DM(om), ocd))
            assert cap == pytest.approx(pt.axle_torque_capacity_Nm(om), rel=2e-3)
            assert I == pytest.approx(pt.engine_inertia_kgm2[g], rel=2e-3)


def test_l0_equivalence_and_pl14_loads(ocd):
    l0 = sf70h_tum_car(beta=0.60)
    lc = LadderCar(l0, ocd)
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = np.r_[rng.uniform(-3, 3), rng.uniform(-.2, .2), rng.uniform(20, 80), rng.uniform(-2, 2), rng.uniform(-.5, .5)]
        d, T = rng.uniform(-.1, .1), rng.uniform(-2e4, 1e4)
        z = np.r_[rng.uniform(-.1, .1, 4), rng.uniform(1000, 6000, 4)]
        C = rng.uniform(-.02, .02)
        a = car_model(x, [d, T], z, C, l0, eps=1e-14)
        b = ladder_model(x, [d, max(T, 0) * l0.R, max(-T, 0) / (8 * l0.M * l0.g)], z, C, lc)
        assert np.allclose(np.array(ca.DM(a["dxds"])), np.array(ca.DM(b["dxds"])), rtol=1e-12, atol=1e-12)
        ra, rb = np.array(ca.DM(a["alg"][4:])).ravel(), np.array(ca.DM(b["alg"][4:])).ravel()
        R = l0.R
        # ladder: per-wheel torque balances; PL14: equal fronts, front sum, rear sum, diff
        assert np.allclose([(rb[0] - rb[1]) / R, (rb[0] + rb[1]) / R, rb[2] / R, rb[3]], -ra, rtol=1e-6, atol=1e-3)
    fr, fl, rr, rl = pl14_loads(-12000., 25000., 15000., l0)
    c = l0
    assert fr + fl + rr + rl == pytest.approx(c.M * c.g + 15000.)
    assert c.wr * (rl - rr) + c.wf * (fl - fr) == pytest.approx(c.h * 25000.)
    assert c.b * (rr + rl) - c.a * (fr + fl) == pytest.approx(c.h * -12000. + (c.aA - c.a) * 15000.)
    assert (fl - fr) == pytest.approx(c.D_roll * ((fl - fr) + (rl - rr)))


def test_footprint_homotopy_and_lam_validation(ocd):
    l0 = sf70h_tum_car()
    assert LadderCar(l0, ocd).footprint() == [(0.0, -0.9), (0.0, 0.9), (0.0, 0.9), (0.0, -0.9)]
    fp = LadderCar(l0, ocd, {"track": 1.0}).footprint()
    assert sorted(fp) == sorted((x, -y) for x, y in ocd.footprint)
    assert set(LadderCar(l0, ocd).lam) == set(FEATURES)


def test_full_ladder_model_steady_circle_solve(ocd, tmp_path):
    """All OCD features on: a small circular-track OCP solves and corners at a sensible speed."""
    from rlsetup.ocp.ladder_solve import LadderSettings, convert_l0, solve_ladder
    from rlsetup.ocp.solve import SolveSettings, solve
    from rlsetup.ocp.track import Track
    p = tmp_path / "circle.csv"
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    np.savetxt(p, np.c_[60 * np.cos(th), 60 * np.sin(th), np.full(200, 6.0), np.full(200, 6.0)], delimiter=",")
    tr = Track.from_csv(p, ds=2.0)
    l0 = sf70h_tum_car(beta=0.60)
    s0 = solve(tr, l0, SolveSettings(ds=6.0))
    assert s0.success
    W, s = convert_l0(s0.W, l0), s0.s
    lam = {k: 0.0 for k in FEATURES}
    for f in FEATURES:          # walk the ladder on the circle
        lam[f] = 1.0
        sol = solve_ladder(tr, LadderCar(l0, ocd, dict(lam)), LadderSettings(ds=6.0, mesh_u_ref=None), W, s)
        assert sol.success, (f, sol.status)
        W, s = sol.W, sol.s
    u = sol.row("u")
    assert 20 < u.mean() < 45 and np.ptp(u) < 2.0
