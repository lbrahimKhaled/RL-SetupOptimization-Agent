"""Perantoni & Limebeer 2014 (PL14) planar F1 car model for minimum-lap-time OCPs.

Source: docs/research/limebeer_ocp_model.md (PL14 eqs 6-35, App. A eqs A.1-A.18, App. B Tables 3-4).

Frame (PL14's own, SAE-like): body x forward, y RIGHT, z DOWN; yaw rate omega positive clockwise seen
from above (a right turn). Curvilinear track: n [m] > 0 to the RIGHT of the centre line, centre-line
curvature C [1/m] > 0 for a RIGHT turn, xi = psi - theta [rad] (car heading relative to the track).
Wheels: fr, fl, rr, rl; the right wheels sit at y = +wf/+wr.

Deviations from the printed PL14 equations (the printed signs are mutually inconsistent, see spec 1.8):
- Normal loads are positive numbers N_ij = -Fz_ij >= 0 [N] (PL14 has Fz <= 0, z down). Load-sensitive
  tyre data (A.3-A.6) are interpolated in N, as the spec's reading 2 says.
- Longitudinal slip kappa = R*Omega/u_w - 1 with Omega >= 0 the wheel spin speed [rad/s] for forward
  rolling (free rolling -> kappa = 0, drive -> kappa > 0). Longitudinal force Fx = +mu_x N kn/rho.
- Slip angle alpha = atan(v_w/u_w) as printed in A.17 (wheel side-slip, positive = wheel sliding to the
  right); the lateral force opposes it: Fy = -mu_y N an/rho.
- rho = sqrt(an^2 + kn^2 + eps_rho) (smoothing at zero slip, spec reading 3).
- Light-wheel clipping (eqs 28, 31) is not modelled: loads stay unclipped and the OCP keeps every load
  above a small positive floor instead (an inactive constraint for this car; checked a posteriori).
- Drive/brake: one longitudinal demand T [N]. Drive max(T,0) goes to the rear axle; brake min(T,0) is
  split beta to the front axle, 1-beta to the rear (ASSUMPTION: PL14 gives no brake ratio). Front wheels
  only brake and carry equal forces (eq 34 with unlocked wheels). Rear wheels are coupled by the
  limited-slip differential (eq 35). max/min use PL14's sqrt smoothing (eq 48).
- Power: max(T,0) * R * (Omega_rr + Omega_rl)/2 <= Pmax (ASSUMPTION: PL14 does not describe the engine;
  Pmax = 560 kW from LPR14 Table 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import casadi as ca

WHEELS = ("fr", "fl", "rr", "rl")


@dataclass(frozen=True)
class TyreParams:
    """PL14 Table 3 magic-formula-like tyre. Loads [N], slip angles [rad], kappa [-]."""
    Fz1: float = 2000.0
    Fz2: float = 6000.0
    mux1: float = 1.75
    mux2: float = 1.40
    kap1: float = 0.11
    kap2: float = 0.10
    muy1: float = 1.80
    muy2: float = 1.45
    alp1: float = math.radians(9.0)
    alp2: float = math.radians(8.0)
    Qx: float = 1.9
    Qy: float = 1.9

    def lerp(self, N, v1: float, v2: float):
        """Linear load interpolation A.3-A.6 in the positive normal load N [N] (extrapolates)."""
        return (N - self.Fz1) * (v2 - v1) / (self.Fz2 - self.Fz1) + v1


@dataclass(frozen=True)
class CarParams:
    """PL14 Table 4 (defaults) plus the items PL14 does not give (marked ASSUMPTION)."""
    name: str = "pl14"
    M: float = 660.0          # kg
    Iz: float = 450.0         # kg m^2
    w: float = 3.4            # wheelbase [m]
    a: float = 1.8            # mass centre to front axle [m]
    h: float = 0.3            # mass-centre height [m]
    D_roll: float = 0.5       # front share of lateral load transfer [-]
    wf: float = 0.73          # front wheel to car centre line [m]
    wr: float = 0.73          # rear wheel to car centre line [m]
    R: float = 0.33           # wheel radius [m]
    kd: float = 10.47         # differential friction coefficient [N m s/rad]
    ClA: float = 3.0 * 1.5    # Cl*A [m^2]
    CdA: float = 0.9 * 1.5    # Cd*A [m^2]
    rho: float = 1.2          # air density [kg/m^3]
    aA: float = 1.9           # centre of pressure to front axle [m]
    g: float = 9.81           # [m/s^2] (PL14 3.3.1: "approximately g=9.8")
    Pmax: float = 560e3       # ASSUMPTION: LPR14 Table 2, 2013 car [W]
    beta: float = 0.60        # ASSUMPTION: front share of brake force (not published) [-]
    half_width: float = 0.9   # ASSUMPTION: car half-width [m] (2013 F1 max width 1.8 m)
    tyre_f: TyreParams = field(default_factory=TyreParams)
    tyre_r: TyreParams = field(default_factory=TyreParams)

    @property
    def b(self) -> float:
        return self.w - self.a


def pl14_car(**kw) -> CarParams:
    """PL14 Table 3/4 car; keyword overrides (e.g. beta=0.55)."""
    return replace(CarParams(), **kw)


def sf70h_tum_car(**kw) -> CarParams:
    """PL14 model with the SF70H-TUM values where they map (ocd_first_lap docs/SF70H_TUM_CAR.md).

    MAPPING (each labelled): mass 733 kg; wheelbase 3.6 m, a = 1.968 m; h = 0.335 m; track 1.6/1.6 m ->
    wf = wr = 0.8 m; Iz = 1159.6 kg m^2 (PLACEHOLDER in the source: scaled from AV-21); rho 1.18;
    CzA_f/CzA_r = 2.20/2.68 -> ClA = 4.88 m^2 at aA = w*CzA_r/(CzA_f+CzA_r) = 1.977 m; CxA = 1.56 m^2;
    Pmax 575 kW (task value). Tyre peak mu: TUM linear law mu(N) = mu0 - 5e-5 (N - 3000) per axle,
    evaluated at PL14's reference loads 2000/6000 N (front x 1.65, y 1.85; rear x 1.95, y 2.15).
    NOT mapped (kept PL14): peak slips kappa/alpha, shape factors Q, kd, D_roll, wheel radius 0.33 m
    (= TUM tyre radius), half-width 0.9 m.
    """
    def mu(mu0, N):
        return mu0 - 5e-5 * (N - 3000.0)
    tf = replace(TyreParams(), mux1=mu(1.65, 2000), mux2=mu(1.65, 6000), muy1=mu(1.85, 2000), muy2=mu(1.85, 6000))
    tr = replace(TyreParams(), mux1=mu(1.95, 2000), mux2=mu(1.95, 6000), muy1=mu(2.15, 2000), muy2=mu(2.15, 6000))
    base = CarParams(name="sf70h_tum_pl14", M=733.0, Iz=1159.6, w=3.6, a=1.968, h=0.335, wf=0.8, wr=0.8,
                     ClA=4.88, CdA=1.56, rho=1.18, aA=3.6 * 2.68 / 4.88, Pmax=575e3, tyre_f=tf, tyre_r=tr)
    return replace(base, **kw)


def smax(x, eps: float):
    """PL14 eq 48: max(x,0) ~ (x + sqrt(x^2+eps))/2."""
    return 0.5 * (x + ca.sqrt(x * x + eps))


def smin(x, eps: float):
    """PL14 eq 48: min(x,0) ~ -(-x + sqrt(x^2+eps))/2."""
    return -0.5 * (-x + ca.sqrt(x * x + eps))


def tyre_forces(kappa, alpha, N, tp: TyreParams, eps_rho: float = 1e-6):
    """Combined-slip forces A.3-A.16 in the wheel frame [N]. kappa>0 drive; alpha>0 wheel sliding right.

    Returns (Fx, Fy): Fx > 0 forward (drive), Fy opposes the side-slip (alpha > 0 -> Fy < 0).
    """
    mux = tp.lerp(N, tp.mux1, tp.mux2)
    muy = tp.lerp(N, tp.muy1, tp.muy2)
    kmax = tp.lerp(N, tp.kap1, tp.kap2)
    amax = tp.lerp(N, tp.alp1, tp.alp2)
    kn = kappa / kmax
    an = alpha / amax
    r = ca.sqrt(an * an + kn * kn + eps_rho)
    Sx = math.pi / (2 * math.atan(tp.Qx))
    Sy = math.pi / (2 * math.atan(tp.Qy))
    fx = mux * ca.sin(tp.Qx * ca.atan(Sx * r))
    fy = muy * ca.sin(tp.Qy * ca.atan(Sy * r))
    return fx * N * kn / r, -fy * N * an / r


def wheel_velocities(u, v, om, delta, car: CarParams):
    """Wheel-centre velocities in each wheel's frame (A.17-A.18 numerators/denominators) [m/s]."""
    a, b, wf, wr = car.a, car.b, car.wf, car.wr
    cd, sd = ca.cos(delta), ca.sin(delta)
    out = {}
    for k, yk in (("fr", wf), ("fl", -wf)):
        ub, vb = u - om * yk, v + om * a
        out[k] = (cd * ub + sd * vb, -sd * ub + cd * vb)
    for k, yk in (("rr", wr), ("rl", -wr)):
        out[k] = (u - om * yk, v - om * b)
    return out


def car_model(x, ctrl, z, C, car: CarParams, eps: float = 1e-3):
    """Full PL14 model at one point.

    x = [n, xi, u, v, om] (m, rad, m/s, m/s, rad/s); ctrl = [delta, T] (rad, N);
    z = [k_fr, k_fl, k_rr, k_rl, N_fr, N_fl, N_rr, N_rl] (slips [-], positive normal loads [N]);
    C = centre-line curvature [1/m]. eps: max/min smoothing (eq 48) in units of N^2 for T.

    Returns dict: dxds (5, d/ds of x), Sf (dt/ds [s/m]), alg (8 residuals [N]/[N m], zero when
    consistent), power (W, drive power), plus diagnostics (forces, loads, aero, Omega).
    """
    n, xi, u, v, om = (x[i] for i in range(5))
    delta, T = ctrl[0], ctrl[1]
    kap = {k: z[i] for i, k in enumerate(WHEELS)}
    N = {k: z[4 + i] for i, k in enumerate(WHEELS)}
    wv = wheel_velocities(u, v, om, delta, car)
    F = {}
    Om = {}
    for k in WHEELS:
        uw, vw = wv[k]
        alpha = ca.atan(vw / uw)
        tp = car.tyre_f if k[0] == "f" else car.tyre_r
        F[k] = tyre_forces(kap[k], alpha, N[k], tp)
        Om[k] = (1 + kap[k]) * uw / car.R
    cd, sd = ca.cos(delta), ca.sin(delta)
    (Ffrx, Ffry), (Fflx, Ffly), (Frrx, Frry), (Frlx, Frly) = (F[k] for k in WHEELS)
    q = 0.5 * car.rho * u * u
    Faz = car.ClA * q                     # downforce, positive down (z down) [N]
    Fax = -car.CdA * q                    # drag [N]
    a, b, wf, wr = car.a, car.b, car.wf, car.wr
    Fx = cd * (Ffrx + Fflx) - sd * (Ffry + Ffly) + Frrx + Frlx + Fax                     # (18)
    Fy = cd * (Ffry + Ffly) + sd * (Ffrx + Fflx) + Frry + Frly                           # (19)
    Mz = (a * (cd * (Ffry + Ffly) + sd * (Ffrx + Fflx)) + wf * (sd * Ffry - cd * Ffrx) - wr * Frrx
          + wf * (cd * Fflx - sd * Ffly) + wr * Frlx - b * (Frry + Frly))              # (17)
    du = om * v + Fx / car.M
    dv = -om * u + Fy / car.M
    dom = Mz / car.Iz
    Sf = (1 - n * C) / (u * ca.cos(xi) - v * ca.sin(xi))                                # (14)
    dxds = ca.vertcat(Sf * (u * ca.sin(xi) + v * ca.cos(xi)), Sf * om - C, Sf * du, Sf * dv, Sf * dom)
    drive = smax(T, eps)
    brake = smin(T, eps)
    Nfr, Nfl, Nrr, Nrl = (N[k] for k in WHEELS)
    alg = ca.vertcat(
        Nfr + Nfl + Nrr + Nrl - car.M * car.g - Faz,                                     # (23)
        wr * (Nrl - Nrr) + wf * (Nfl - Nfr) - car.h * Fy,                                # (24)
        b * (Nrr + Nrl) - a * (Nfr + Nfl) - car.h * Fx - (car.aA - a) * Faz,             # (25)
        (Nfl - Nfr) - car.D_roll * ((Nfl - Nfr) + (Nrl - Nrr)),                          # (26)
        Ffrx - Fflx,                                                                     # (34)
        Ffrx + Fflx - car.beta * brake,                                                  # front brake share
        Frrx + Frlx - drive - (1 - car.beta) * brake,                                    # rear drive + brake
        car.R * (Frlx - Frrx) + car.kd * (Om["rl"] - Om["rr"]),                          # (35)
    )
    power = drive * car.R * 0.5 * (Om["rr"] + Om["rl"])
    return dict(dxds=dxds, Sf=Sf, alg=alg, power=power, Fx=Fx, Fy=Fy, Mz=Mz, Faz=Faz, Fax=Fax,
                F=F, N=N, Om=Om, ax=Fx / car.M, ay=Fy / car.M)


def static_loads(car: CarParams, u: float = 0.0):
    """Positive static + aero wheel loads [N] at speed u (no inertial transfer). Order fr, fl, rr, rl."""
    Faz = car.ClA * 0.5 * car.rho * u * u
    Nf = (car.M * car.g * car.b + Faz * (car.w - car.aA)) / car.w
    Nr = car.M * car.g + Faz - Nf
    return [Nf / 2, Nf / 2, Nr / 2, Nr / 2]
