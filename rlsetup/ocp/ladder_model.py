"""PL14 planar OCP model made progressively closer to OCD's car, one feature per step (the "ladder").

Each OCD feature has a homotopy weight lam[k] in [0, 1]: lam = 0 is the L0 model (PL14 equations with
the SF70H-TUM mapping, rlsetup.ocp.pl14_model.sf70h_tum_car), lam = 1 is OCD's steady-state behaviour
with the parameters read from OCD (rlsetup.ocp.ocd_car). A ladder step sets one more weight to 1;
intermediate values are only used as continuation (homotopy) paths when a direct solve fails.

Frame: PL14's (x forward, y RIGHT, yaw rate clockwise positive, n > 0 right, C > 0 right turn), as in
pl14_model. OCD quantities (ISO, y left) are converted at the interface: Fy_iso = -Fy_pl14.
Layout: x = [n, xi, u, v, omega]; c = [delta, T_drive, b]; z = [kappa_fr, kappa_fl, kappa_rr, kappa_rl,
N_fr, N_fl, N_rr, N_rl]. delta [rad] (> 0 steers right), T_drive [N m] rear-axle drive torque >= 0,
b [-] brake pedal in [0, 1], kappa = R Omega / u_w - 1, N [N] > 0.

Features (keys of lam):
  tyre     PL14 tyre with the SF70H mapping -> OCD MF5.2 (pure + combined slip, load sensitivity,
           FNOMIN, LMU, offsets; right wheels mirrored as OCD does); rolling radius 0.33 -> OCD's.
  brakes   brake torque b * T_bmax split bias/2 per front wheel, (1-bias)/2 per rear wheel:
           T_bmax 8 M g R (L0's force bound) -> 15 kN m, bias 0.60 -> setup's (0.49 at p1_TUM).
  rollres  OCD rolling resistance c_rr N in each wheel's torque balance (c_rr 0 -> 0.03).
  engine   drive torque <= OCD engine map x gear ratio x eta_g (static upshift map, smooth blend)
           instead of T_drive * omega_axle <= Pmax (575 kW). ERS off.
  inertia  quasi-steady wheel spin (kappa_dot = 0): I * u_dot / R in each wheel balance; rear axle
           carries 2 I_wr + engine inertia of the gear (ocd_lap Powertrain, TUM e_i).
  lsd      differential: PL14 viscous kd -> OCD LSD steady state
           (L_rr + Tb_rr) - (L_rl + Tb_rl) = 2 tanh((Omega_rl - Omega_rr)/2) lock_drive T_drive.
  loads    PL14 load equations (23)-(26) -> OCD's suspension in steady state (springs, ARBs,
           roll/pitch-centre jacking, tyre springs; ocd_car.load_map), aero at static ride heights,
           plus OCD's lateral aero force -0.5 rho CxA v|v|.
  steer    steering rate |d delta/dt| <= 1.5 rad/s (|delta| <= 0.6 rad is already L0's bound). In the
           solver (needs neighbouring nodes).
  track    track rule: mass centre within half-width - 0.9 m -> all four 4.6 x 2.0 m footprint
           corners inside the edges. In the solver.
  rhaero   ride-height aero: OCD's CzA(rh) law (2 %/cm) with the steady-state heave and pitch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import casadi as ca
import numpy as np

from .ocd_car import OCDCar, casadi_ops, mf52
from .pl14_model import CarParams, smax, tyre_forces, wheel_velocities

FEATURES = ("tyre", "brakes", "rollres", "engine", "inertia", "lsd", "loads", "steer", "track", "rhaero")
NX, NC, NZ = 5, 3, 8
STATE_NAMES = ["n", "xi", "u", "v", "omega"]
CTRL_NAMES = ["delta", "T_drive", "b"]
ALG_NAMES = ["kappa_fr", "kappa_fl", "kappa_rr", "kappa_rl", "N_fr", "N_fl", "N_rr", "N_rl"]
GEAR_BLEND_RADPS = 1.0      # PLACEHOLDER smoothing width of the static upshift map [rad/s at the axle]
RPM_CLIP_SMOOTH = 100.0     # PLACEHOLDER smoothing of OCD's rpm clip [rpm]
# PL14 wheel order fr, fl, rr, rl  ->  OCD index FL=0, FR=1, RL=2, RR=3
PL2OCD = {"fr": 1, "fl": 0, "rr": 3, "rl": 2}


@dataclass
class LadderCar:
    """The two end points of every homotopy: L0 (PL14 mapping) and OCD."""
    l0: CarParams
    ocd: OCDCar
    lam: dict = field(default_factory=lambda: {k: 0.0 for k in FEATURES})

    def __post_init__(self):
        self.lam = {k: float(self.lam.get(k, 0.0)) for k in FEATURES}
        unknown = set(self.lam) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features {unknown}")
        o, c = self.ocd, self.l0
        assert abs(o.m - c.M) < 1e-9 and abs(o.l - c.w) < 1e-9 and abs(o.l_f - c.a) < 1e-9, "geometry differs"
        assert abs(o.b_f / 2 - c.wf) < 1e-9 and abs(o.b_r / 2 - c.wr) < 1e-9 and abs(o.Iz - c.Iz) < 0.05   # L0 rounded Iz to 1159.6
        self.lmap = o.load_map()
        self.czA_s, self.dczA, self.rh_s = o.aero_linear()

    # blended scalars --------------------------------------------------------------------------
    def mix(self, k, a, b):
        lam = self.lam[k]
        return a if lam == 0.0 else (b if lam == 1.0 else (1 - lam) * a + lam * b)

    @property
    def R(self):
        return self.mix("tyre", self.l0.R, self.ocd.R_r)

    @property
    def Tb_max(self):
        return self.mix("brakes", 8 * self.l0.M * self.l0.g * self.l0.R, self.ocd.max_brake_torque_Nm)

    @property
    def bias(self):
        return self.mix("brakes", self.l0.beta, self.ocd.brake_bias_front)

    def footprint(self):
        """Corners (dx forward, dy right) [m] about the CoG, blended from (0, +-half_width)."""
        lam = self.lam["track"]
        out = []
        for x, y in self.ocd.footprint:          # (x fwd, y left)
            dy = -y                               # PL14: y right
            out.append((lam * x, math.copysign(self.l0.half_width * (1 - lam) + abs(dy) * lam, dy)))
        return out

    def summary(self):
        return dict(lam=dict(self.lam), R=self.R, Tb_max=self.Tb_max, bias=self.bias, footprint=self.footprint())


def engine_capacity(omega_axle, ocd: OCDCar):
    """Full-throttle rear-axle torque [N m] and reflected engine inertia [kg m^2] at axle speed
    omega_axle [rad/s]: OCD Powertrain (ERS off) with a static upshift map blended over
    GEAR_BLEND_RADPS. Same as Powertrain.axle_torque_capacity_Nm in the gear the upshift map gives."""
    d = ocd.pt_data
    n_lo, n_hi = .75 * d["n_begin"], 1.2 * d["n_end"]
    w_up = ocd.upshift_omega()
    S = [0.5 * (1 + ca.tanh((omega_axle - wu) / GEAR_BLEND_RADPS)) for wu in w_up]
    weights = [1 - S[0]] + [S[g - 1] - S[g] for g in range(1, len(S))] + [S[-1]]
    cap = 0
    inertia = 0
    for g, wg in enumerate(weights):
        rpm = omega_axle * ocd.ratios[g] * 60 / (2 * math.pi)
        n = n_lo + smax(rpm - n_lo, RPM_CLIP_SMOOTH ** 2)
        n = n_hi - smax(n_hi - n, RPM_CLIP_SMOOTH ** 2)
        Pw = smax(ca.polyval(ca.DM(ocd.power_coeffs), n), 1e2 ** 2)
        T = Pw / (2 * math.pi * n / 60)
        cap = cap + wg * d["eta_g"] * ocd.ratios[g] * T
        inertia = inertia + wg * ocd.engine_inertia[g]
    return cap, inertia


def pl14_loads(Fx, Fy, Faz, car: CarParams):
    """Explicit solution of PL14 (23)-(26) [N], order fr, fl, rr, rl (Fx, Fy body totals, PL14 frame)."""
    a, b, w = car.a, car.b, car.w
    W = car.M * car.g + Faz
    Nf = (b * W - car.h * Fx - (car.aA - a) * Faz) / w
    Nr = W - Nf
    D = car.D_roll
    df = D * car.h * Fy / (car.wr * (1 - D) + car.wf * D)     # Nfl - Nfr
    dr = df * (1 - D) / D                                      # Nrl - Nrr
    return [(Nf - df) / 2, (Nf + df) / 2, (Nr - dr) / 2, (Nr + dr) / 2]


def ladder_model(x, ctrl, z, C, lc: LadderCar):
    """One node. Returns dict: dxds (5), Sf, alg (8 residuals: 4 loads [N], 4 wheel torques [N m]),
    g_drive (drive-limit inequality <= 0, scaled), corners (4 corner lateral offsets [m]), diagnostics."""
    car, ocd, lam = lc.l0, lc.ocd, lc.lam
    n, xi, u, v, om = (x[i] for i in range(5))
    delta, Td, bp = ctrl[0], ctrl[1], ctrl[2]
    kap = {k: z[i] for i, k in enumerate(("fr", "fl", "rr", "rl"))}
    N = {k: z[4 + i] for i, k in enumerate(("fr", "fl", "rr", "rl"))}
    R = lc.R
    wv = wheel_velocities(u, v, om, delta, car)
    ops = casadi_ops() if lam["tyre"] > 0 else None
    F, Om = {}, {}
    for k in ("fr", "fl", "rr", "rl"):
        uw, vw = wv[k]
        alpha = ca.atan(vw / uw)                    # PL14 A.17 (= -OCD alpha on the left, OCD alpha on the right)
        Om[k] = (1 + kap[k]) * uw / R
        Fpl = tyre_forces(kap[k], alpha, N[k], car.tyre_f if k[0] == "f" else car.tyre_r) if lam["tyre"] < 1 else (0, 0)
        if lam["tyre"] > 0:
            wheel = ("front_" if k[0] == "f" else "rear_") + ("right" if k[1] == "r" else "left")
            right = k[1] == "r"
            Fx_mf, Fy_mf = mf52(kap[k], alpha if right else -alpha, N[k], ocd.tyre(wheel), ops=ops)
            Fmf = (Fx_mf, Fy_mf if right else -Fy_mf)
        else:
            Fmf = (0, 0)
        F[k] = tuple((1 - lam["tyre"]) * a + lam["tyre"] * b for a, b in zip(Fpl, Fmf))
    cd, sd = ca.cos(delta), ca.sin(delta)
    (Ffrx, Ffry), (Fflx, Ffly), (Frrx, Frry), (Frlx, Frly) = (F[k] for k in ("fr", "fl", "rr", "rl"))
    # aero: drag and downforce on body x speed (PL14 and OCD agree); OCD lateral aero -0.5 rho CxA v|v|
    q = 0.5 * car.rho * u * u
    Fax = -car.CdA * q
    Fay = -lam["loads"] * 0.5 * ocd.rho * ocd.cd_A * v * ca.fabs(v)
    a, b, wf, wr = car.a, car.b, car.wf, car.wr
    Fx_t = cd * (Ffrx + Fflx) - sd * (Ffry + Ffly) + Frrx + Frlx            # tyre forces, body frame
    Fy_t = cd * (Ffry + Ffly) + sd * (Ffrx + Fflx) + Frry + Frly
    Fx = Fx_t + Fax
    Fy = Fy_t + Fay
    Mz = (a * (cd * (Ffry + Ffly) + sd * (Ffrx + Fflx)) + wf * (sd * Ffry - cd * Ffrx) - wr * Frrx
          + wf * (cd * Fflx - sd * Ffly) + wr * Frlx - b * (Frry + Frly))
    du = om * v + Fx / car.M
    dv = -om * u + Fy / car.M
    dom = Mz / car.Iz
    Sf = (1 - n * C) / (u * ca.cos(xi) - v * ca.sin(xi))
    dxds = ca.vertcat(Sf * (u * ca.sin(xi) + v * ca.cos(xi)), Sf * om - C, Sf * du, Sf * dv, Sf * dom)

    # ---- wheel loads
    Faz_l0 = car.ClA * q
    N_l0 = pl14_loads(Fx, Fy_t, Faz_l0, car) if lam["loads"] < 1 else [0] * 4
    rh_f = rh_r = None
    if lam["loads"] > 0:
        # vehicle-frame tyre forces in OCD's ISO frame and wheel order FL, FR, RL, RR
        vx = {"fr": cd * Ffrx - sd * Ffry, "fl": cd * Fflx - sd * Ffly, "rr": Frrx, "rl": Frlx}
        vy = {"fr": cd * Ffry + sd * Ffrx, "fl": cd * Ffly + sd * Fflx, "rr": Frry, "rl": Frly}
        f8 = [0] * 8
        for k, i in PL2OCD.items():
            f8[i] = vx[k]
            f8[4 + i] = -vy[k]
        f8 = ca.vertcat(*f8)
        lm = lc.lmap
        A8, AL = ca.DM(lm["A"][:, :8]), ca.DM(lm["A"][:, 8:])
        qa = 0.5 * ocd.rho * u * u
        cS = ca.DM(lc.czA_s)
        if lam["rhaero"] > 0:   # downforce at the ride heights it produces (2x2 linear fixed point)
            K = lam["rhaero"] * ca.DM(lc.dczA) @ ca.DM(lm["rh_rows"])          # dCzA/dy  (2 x 7)
            Yf8, YL, Y0 = ca.DM(lm["Yf"][:, :8]), ca.DM(lm["Yf"][:, 8:]), ca.DM(lm["Y0"])
            M2 = ca.DM.eye(2) - qa * (K @ YL)
            rhs = qa * (cS + K @ (Y0 + Yf8 @ f8))
            det = M2[0, 0] * M2[1, 1] - M2[0, 1] * M2[1, 0]
            FL = ca.vertcat(M2[1, 1] * rhs[0] - M2[0, 1] * rhs[1], -M2[1, 0] * rhs[0] + M2[0, 0] * rhs[1]) / det
            y = Y0 + Yf8 @ f8 + YL @ FL
            rh = ca.DM(lc.rh_s) + ca.DM(lm["rh_rows"]) @ y
            rh_f, rh_r = rh[0], rh[1]
        else:
            FL = qa * cS
        Nocd = ca.DM(lm["N0"]) + A8 @ f8 + AL @ FL
        N_ocd = [Nocd[PL2OCD[k]] for k in ("fr", "fl", "rr", "rl")]
        Faz = FL[0] + FL[1]
    else:
        N_ocd = [0] * 4
        Faz = Faz_l0
    Nt = [(1 - lam["loads"]) * p0 + lam["loads"] * p1 for p0, p1 in zip(N_l0, N_ocd)]

    # ---- wheel spin equilibria (OCD drivetrain_rwd_lsd steady state)
    Tb = bp * lc.Tb_max
    Tbf, Tbr = lc.bias * Tb / 2, (1 - lc.bias) * Tb / 2
    crr = lam["rollres"] * ocd.c_rr
    L = {k: (F[k][0] + crr * N[k]) * R for k in F}                          # tyre load torque [N m]
    om_ax = 0.5 * (Om["rr"] + Om["rl"])
    cap, I_eng = engine_capacity(om_ax, ocd) if lam["engine"] > 0 or lam["inertia"] > 0 else (0, 0)
    li = lam["inertia"]
    w_dot = du / R                                                           # quasi-steady wheel spin-up
    res_fr = -Tbf - L["fr"] - li * ocd.I_wf * w_dot
    res_fl = -Tbf - L["fl"] - li * ocd.I_wf * w_dot
    res_ax = Td - 2 * Tbr - L["rr"] - L["rl"] - li * (2 * ocd.I_wr + I_eng) * w_dot
    dT = (L["rr"] + Tbr) - (L["rl"] + Tbr)
    dOm = Om["rl"] - Om["rr"]
    res_kd = dT - car.kd * dOm
    res_lsd = dT - 2 * ca.tanh(dOm / 2) * ocd.lsd["lock_drive"] * Td
    res_diff = (1 - lam["lsd"]) * res_kd + lam["lsd"] * res_lsd
    alg = ca.vertcat(*[N[k] - Nt[i] for i, k in enumerate(("fr", "fl", "rr", "rl"))],
                     res_fr, res_fl, res_ax, res_diff)

    # ---- drive limit (<= 0)
    Fc_R = car.M * car.g * R
    g_pow = Td * om_ax / car.Pmax - 1
    g_eng = (Td - cap) / Fc_R if lam["engine"] > 0 else 0
    g_drive = (1 - lam["engine"]) * g_pow + lam["engine"] * g_eng

    # ---- footprint corner lateral offsets (second order in curvature)
    corners = []
    for dx, dy in lc.footprint():
        dxp = dx * ca.cos(xi) - dy * ca.sin(xi)
        dyp = dx * ca.sin(xi) + dy * ca.cos(xi)
        corners.append(n + dyp - C * dxp ** 2 / (2 * (1 - C * (n + dyp))))
    return dict(dxds=dxds, Sf=Sf, alg=alg, g_drive=g_drive, corners=ca.vertcat(*corners),
                Fx=Fx, Fy=Fy, Mz=Mz, du=du, F=F, Om=Om, N_target=Nt, cap=cap, I_eng=I_eng, om_ax=om_ax,
                Faz=Faz, Tb=Tb, rh_f=rh_f, rh_r=rh_r, power=Td * om_ax, res_lsd=res_lsd)
