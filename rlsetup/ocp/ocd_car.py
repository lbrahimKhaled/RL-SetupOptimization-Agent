"""OCD's car, read from the actual simulator parameters, as steady-state pieces for the OCP ladder.

Source of every number: the effective OCD parameter dict of
    OCDBackend(powertrain='engine', car='sf70h_tum', ers=False, max_brake_torque_Nm=15000).reset(setup)
(ocd_first_lap, not edited) plus its Powertrain object. Code ported (not changed) from
vendor/Open-Car-Dynamics:
    lib/ocd_tire_models_cpp/src/mf_52.cpp                               -> mf52()
    lib/ocd_vehicle_dynamics_double_track_cpp/.../vehicle_dynamics_equations_impl.hpp
        (suspension, anti-roll bars, roll/pitch-centre jacking, tyre springs, heave/roll/pitch
        balances; steady state = all derivatives zero)                  -> OCDCar.load_map()
    lib/ocd_aerodynamics_models_cpp/src/ride_height.cpp                 -> OCDCar.aero_*
    lib/ocd_drivetrain_rwd_lsd_cpp/src/drivetrain_rwd_lsd_eqns.cpp      -> (used in ladder_model)
    ocd_lap/powertrain.py (engine curve, gear ratios, upshift rpm, eta_g, inertia) -> engine_*

Frames. OCD is ISO: x forward, y LEFT, z up, yaw counter-clockwise positive. load_map() takes and
returns ISO quantities; ladder_model converts from the PL14 frame (y right) used by the OCP.
Wheel order here: FL, FR, RL, RR (OCD's).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

P = "vehicle_dynamics_double_track."
TM = P + "z.tire_model."
G = 9.81                                    # OCD G_CONST (vehicle_dynamics_equations.hpp)
OCD_WHEELS = ("front_left", "front_right", "rear_left", "rear_right")
MF_KEYS = ("FNOMIN LFZO LCX LMUX LEX LKX LHX LVX LCY LMUY LEY LKY LKYG LHY LVY LXAL LYKA LVYKA "
           "PCX1 PDX1 PDX2 PEX1 PEX2 PEX3 PEX4 PKX1 PKX2 PKX3 PHX1 PHX2 PVX1 PVX2 "
           "RBX1 RBX2 RBX3 RCX1 REX1 REX2 RHX1 "
           "PCY1 PDY1 PDY2 PDY3 PEY1 PEY2 PEY3 PEY4 PEY5 PKY1 PKY2 PKY3 PKY4 PKY5 PKY6 PKY7 "
           "PHY1 PHY2 PVY1 PVY2 PVY3 PVY4 RBY1 RBY2 RBY3 RBY4 RCY1 REY1 REY2 RHY1 RHY2 "
           "RVY1 RVY2 RVY3 RVY4 RVY5 RVY6").split()


# ---------------------------------------------------------------- MF 5.2 (mf_52.cpp, verbatim) --
class _NumpyOps:
    sin, cos, atan, exp = staticmethod(np.sin), staticmethod(np.cos), staticmethod(np.arctan), staticmethod(np.exp)

    @staticmethod
    def sign1(x):                      # std::copysign(1.0, x)
        return np.copysign(1.0, x)


class _CasadiOps:
    def __init__(self):
        import casadi as ca
        self.sin, self.cos, self.atan, self.exp = ca.sin, ca.cos, ca.atan, ca.exp
        self.sign1 = lambda x: ca.if_else(x < 0, -1.0, 1.0)


NP = _NumpyOps()


def casadi_ops():
    return _CasadiOps()


def mf52(kappa, alpha, Fz, p: dict, gamma=0.0, ops=NP):
    """MF5.2 combined-slip forces exactly as OCD's MF52::evaluate after its sign flip.

    Arguments are MF-internal: `alpha` = -input.slip_angle_rad of mf_52.cpp (OCD passes
    input.slip_angle = mirror * (-atan2(vy, max(1, vx))) in the tyre frame, mirror = -1 on the right
    wheels), `gamma` = input.gamma_rad (= -camber_out). kappa [-], alpha [rad], Fz [N] > 0.
    Returns (Fx, Fy) [N] in MF's own output convention (OCD multiplies Fy by mirror)."""
    o = ops
    Fz0 = p["LFZO"] * p["FNOMIN"]
    dfz = (Fz - Fz0) / Fz0
    gs = math.sin(float(gamma))        # camber is a setup constant here, never a decision variable
    # long_force_pure_slip
    SHx = (p["PHX1"] + p["PHX2"] * dfz) * p["LHX"]
    SVx = Fz * (p["PVX1"] + p["PVX2"] * dfz) * p["LVX"] * p["LMUX"]
    kx = kappa + SHx
    Cx = p["PCX1"] * p["LCX"]
    Dx = (p["PDX1"] + p["PDX2"] * dfz) * p["LMUX"] * Fz
    Ex = (p["PEX1"] + p["PEX2"] * dfz + p["PEX3"] * dfz ** 2) * (1.0 - p["PEX4"] * o.sign1(kx)) * p["LEX"]
    Kx = Fz * (p["PKX1"] + p["PKX2"] * dfz) * o.exp(p["PKX3"] * dfz) * p["LKX"]
    Bx = Kx / (Cx * Dx)
    Fx0 = Dx * o.sin(Cx * o.atan(Bx * kx - Ex * (Bx * kx - o.atan(Bx * kx)))) + SVx
    # lat_force_pure_slip
    Kyg0 = Fz * (p["PKY6"] + p["PKY7"] * dfz) * p["LKYG"]
    SVyg = Fz * (p["PVY3"] + p["PVY4"] * dfz) * gs * p["LKYG"] * p["LMUY"]
    SVy = Fz * (p["PVY1"] + p["PVY2"] * dfz) * p["LVY"] * p["LMUY"] + SVyg
    Ky = (p["PKY1"] * Fz0 * (1.0 - p["PKY3"] * abs(gs))
          * o.sin(p["PKY4"] * o.atan(Fz / ((p["PKY2"] + p["PKY5"] * gs ** 2) * Fz0))) * p["LKY"])
    SHy = (p["PHY1"] + p["PHY2"] * dfz) * p["LHY"] + (Kyg0 * gs - SVyg) / Ky
    Cy = p["PCY1"] * p["LCY"]
    Dy = ((p["PDY1"] + p["PDY2"] * dfz) * (1.0 - p["PDY3"] * gs ** 2)) * p["LMUY"] * Fz
    By = Ky / (Cy * Dy)
    ay = alpha + SHy
    Ey = (p["PEY1"] + p["PEY2"] * dfz) * (1.0 + p["PEY5"] * gs ** 2 - (p["PEY3"] + p["PEY4"] * gs) * o.sign1(ay)) * p["LEY"]
    Fy0 = Dy * o.sin(Cy * o.atan(By * ay - Ey * (By * ay - o.atan(By * ay)))) + SVy
    # long_force_combined_slip
    SHxa = p["RHX1"]
    Exa = p["REX1"] + p["REX2"] * dfz
    Cxa = p["RCX1"]
    Bxa = (p["RBX1"] + p["RBX3"] * gs ** 2) * o.cos(o.atan(p["RBX2"] * kappa)) * p["LXAL"]
    aS = alpha + SHxa
    Gxa0 = o.cos(Cxa * o.atan(Bxa * SHxa - Exa * (Bxa * SHxa - o.atan(Bxa * SHxa))))
    Gxa = o.cos(Cxa * o.atan(Bxa * aS - Exa * (Bxa * aS - o.atan(Bxa * aS)))) / Gxa0
    Fx = Gxa * Fx0
    # lat_force_combined_slip
    muy = ((p["PDY1"] + p["PDY2"] * dfz) / (1.0 + p["PDY3"] * gs ** 2)) * p["LMUY"]
    DVyk = muy * Fz * (p["RVY1"] + p["RVY2"] * dfz + p["RVY3"] * gs) * o.cos(o.atan(p["RVY4"] * alpha))
    SVyk = DVyk * o.sin(p["RVY5"] * o.atan(p["RVY6"] * kappa)) * p["LVYKA"]
    SHyk = p["RHY1"] + p["RHY2"] * dfz
    Eyk = p["REY1"] + p["REY2"] * dfz
    Cyk = p["RCY1"]
    Byk = (p["RBY1"] + p["RBY4"] * gs ** 2) * o.cos(o.atan(p["RBY2"] * (alpha - p["RBY3"]))) * p["LYKA"]
    kS = kappa + SHyk
    Gyk0 = o.cos(Cyk * o.atan(Byk * SHyk - Eyk * (Byk * SHyk - o.atan(Byk * SHyk))))
    Gyk = o.cos(Cyk * o.atan(Byk * kS - Eyk * (Byk * kS - o.atan(Byk * kS)))) / Gyk0
    Fy = Gyk * Fy0 + SVyk
    return Fx, Fy


def ocd_tyre_forces(kappa, slip_angle_ocd, Fz, p: dict, mirror: bool, camber_out=0.0, ops=NP):
    """OCD's tyre_model_eval: tyre-frame (Fx, Fy) [N], ISO frame (Fy > 0 to the left).

    slip_angle_ocd = OCD's alpha state, -atan2(vy_tyre, max(1, vx_tyre)); mirror = right wheel."""
    m = -1.0 if mirror else 1.0
    Fx, Fy = mf52(kappa, -(m * slip_angle_ocd), Fz, p, gamma=-camber_out, ops=ops)
    return Fx, m * Fy


# ----------------------------------------------------------------------- OCD car (steady parts) --
@dataclass
class OCDCar:
    """Steady-state OCD car. Lengths m, masses kg, forces N, torques N m, angles rad."""
    name: str
    setup_name: str
    prm: dict = field(repr=False)                  # full effective OCD parameter dict
    brake_bias_front: float
    max_brake_torque_Nm: float
    final_drive: float
    pt_data: dict = field(repr=False)              # ocd_lap.powertrain.F1_SHANGHAI
    ratios: np.ndarray = field(repr=False)         # overall engine/wheel ratio per gear
    power_coeffs: np.ndarray = field(repr=False)   # full-load power cubic in rpm [W]
    engine_inertia: np.ndarray = field(repr=False) # reflected to the rear axle per gear [kg m^2]
    footprint: tuple = ()                          # CarSpec.footprint, (x fwd, y left) [m] about the CoG

    # convenience accessors ------------------------------------------------------------------
    def p(self, key):
        return self.prm[P + key]

    @property
    def m(self): return self.p("mass_vehicle_kg")
    @property
    def l(self): return self.p("wheelbase_m")
    @property
    def l_f(self): return self.p("cog.distance_from_front_axle_m")
    @property
    def l_r(self): return self.l - self.l_f
    @property
    def h_cg(self): return self.p("cog.height_m")
    @property
    def b_f(self): return self.p("track_width_m.front")
    @property
    def b_r(self): return self.p("track_width_m.rear")
    @property
    def R_f(self): return self.p("tire.rolling_radius_m.front")
    @property
    def R_r(self): return self.p("tire.rolling_radius_m.rear")
    @property
    def c_rr(self): return self.p("tire.rolling_resistance_coefficient")
    @property
    def I_wf(self): return self.prm["drivetrain.I_wheel_front_kgm2"]
    @property
    def I_wr(self): return self.prm["drivetrain.I_wheel_rear_kgm2"]
    @property
    def Iz(self): return self.p("moment_of_inertia_kgpm2.z")
    @property
    def rho(self): return self.p("aerodynamics.air_density_kgpm3")
    @property
    def cd_A(self): return self.p("aerodynamics.c_d") * self.p("aerodynamics.A_m2")
    @property
    def lsd(self):
        d = self.prm
        return dict(lock_drive=d["drivetrain.LSD_lock_drive"], lock_coast=d["drivetrain.LSD_lock_coast"],
                    preload=d["drivetrain.LSD_preload"])
    @property
    def steer_max(self): return self.prm["steering_actuator.angle_max_rad"]
    @property
    def steer_rate_max(self): return self.prm["steering_actuator.angle_rate_max_radps"]

    def tyre(self, wheel: str) -> dict:
        """MF5.2 coefficients of one wheel ('front_left', ...)."""
        return {k: float(self.prm[TM + f"{wheel}.{k}"]) for k in MF_KEYS}

    # aero (ride_height.cpp) -----------------------------------------------------------------
    def aero_lift_coeffs(self, rh_f, rh_r):
        """(c_l front, c_l rear) at ride heights rh [m] (unclamped; OCD clamps to [0, 5])."""
        a = lambda k: self.p("aerodynamics." + k)
        cf = (a("c0") + a("c1_F") * rh_f + a("c2_F") * rh_f ** 2 + a("c3_F") * rh_f ** 3 + a("c1_R") * rh_r
              + a("c2_R") * rh_r ** 2 + a("c3_R") * rh_r ** 3 + a("c4") * rh_f * rh_r + a("c5") * rh_f ** 2 * rh_r
              + a("c6") * rh_f * rh_r ** 2)
        cr = (a("a0") + a("a1_F") * rh_f + a("a2_F") * rh_f ** 2 + a("a3_F") * rh_f ** 3 + a("a1_R") * rh_r
              + a("a2_R") * rh_r ** 2 + a("a3_R") * rh_r ** 3 + a("a4") * rh_f * rh_r + a("a5") * rh_f ** 2 * rh_r
              + a("a6") * rh_f * rh_r ** 2)
        return cf, cr

    @property
    def static_ride_heights(self):
        """Ride heights at rest = -static_ride_height_* (translator convention) [m]."""
        return -self.p("aerodynamics.static_ride_height_front_m"), -self.p("aerodynamics.static_ride_height_rear_m")

    def aero_linear(self):
        """Linear ride-height law around the static heights: c_l = c_s + K (rh - rh_s).

        Exact for the translator's map (only c0, c1_F, a0, a1_R are nonzero; checked)."""
        a = lambda k: self.p("aerodynamics." + k)
        for k in ("c2_F", "c3_F", "c2_R", "c3_R", "c4", "c5", "c6", "a2_F", "a3_F", "a2_R", "a3_R", "a4", "a5", "a6"):
            if a(k) != 0.0:
                raise NotImplementedError(f"nonlinear aero map term {k} = {a(k)}")
        rh = self.static_ride_heights
        c_s = np.array(self.aero_lift_coeffs(*rh))
        K = np.array([[a("c1_F"), a("c1_R")], [a("a1_F"), a("a1_R")]])
        A = a("A_m2")
        return c_s * A, K * A, np.array(rh)     # CzA at static heights [m^2], dCzA/drh [m], rh_s [m]

    # powertrain (ocd_lap/powertrain.py) ------------------------------------------------------
    def upshift_omega(self):
        """Rear-axle speeds [rad/s] at which the gearbox leaves gear g (g = 1..7)."""
        n = np.asarray(self.pt_data["n_shift"][:-1])
        return n * 2 * math.pi / 60 / self.ratios[:-1]

    def engine_torque_np(self, rpm):
        """Powertrain.engine_torque_Nm (full load, rpm clipped to [0.75 n_begin, 1.2 n_end])."""
        n = np.clip(rpm, .75 * self.pt_data["n_begin"], 1.2 * self.pt_data["n_end"])
        return np.maximum(0.0, np.polyval(self.power_coeffs, n)) / (2 * math.pi * n / 60)

    # steady-state vertical model -------------------------------------------------------------
    def load_map(self):
        """Steady-state wheel loads of OCD's double-track model as an exact linear map.

        Inputs f (ISO, vehicle frame, after steering rotation): [Fx_FL, Fx_FR, Fx_RL, Fx_RR,
        Fy_FL, Fy_FR, Fy_RL, Fy_RR, Fl_front, Fl_rear] with Fl the aero downforce at each axle
        (positive down, N). All suspension velocities zero; dampers then exert no force.
        Returns dict: N0 (4,), A (4, 10): Fz = N0 + A f; Y0 (7,), Yf (7, 10): body/wheel states
        y = [z, phi, theta, z_w FL, FR, RL, RR] = Y0 + Yf f; bump (4, ...) suspension bump travel.
        Assumes heave springs off, packers not touched (bump < gap; checked by caller), tyres in
        contact, equal pitch-centre heights for accel/decel (all true at p1_TUM / bcn_best)."""
        d = self
        if d.p("suspension.heave.enabled") > 0.5:
            raise NotImplementedError("heave springs on")
        if d.p("suspension.pitch_center_height_m.accel") != d.p("suspension.pitch_center_height_m.decel"):
            raise NotImplementedError("different accel/decel pitch centres (sign switch)")
        m, mtf, mtr, l, lf, h = d.m, d.p("mass_wheel_kg.front"), d.p("mass_wheel_kg.rear"), d.l, d.l_f, d.h_cg
        bf, br = d.b_f, d.b_r
        lr = l - lf
        ms = m - 2 * mtf - 2 * mtr
        lrs = (m * lr - 2 * mtf * l) / ms
        lfs = l - lrs
        hrc, hpc = d.p("suspension.roll_center_height_m"), d.p("suspension.pitch_center_height_m.accel")
        slf, slr = hrc / (0.5 * bf), hrc / (0.5 * br)
        sadf, sadr = hpc / lf, hpc / lr
        cf, cr = d.p("suspension.vehicle_spring_stiffness_Npm.front"), d.p("suspension.vehicle_spring_stiffness_Npm.rear")
        caf = d.p("suspension.antirollbar_virtual_spring_stiffness_Npm.front")
        car = d.p("suspension.antirollbar_virtual_spring_stiffness_Npm.rear")
        ctf, ctr = d.p("tire.spring_stiffness_Npm.front"), d.p("tire.spring_stiffness_Npm.rear")
        s0t = np.array([m * G * lr / (2 * l) / ctf] * 2 + [m * G * lf / (2 * l) / ctr] * 2)
        s0 = np.array([ms * G * lrs / (2 * cf * l)] * 2 + [ms * G * lfs / (2 * cr * l)] * 2)
        ct = np.array([ctf, ctf, ctr, ctr])
        mt = np.array([mtf, mtf, mtr, mtr])
        la = d.p("aerodynamics.l")
        lfa = d.p("aerodynamics.l_f")
        lra = la - lfa

        def residual(y, f):
            z, phi, th, zw = y[0], y[1], y[2], y[3:7]
            Fx, Fy, Flf, Flr = f[0:4], f[4:8], f[8], f[9]
            comp = np.array([-s0[0] + z + phi * bf / 2 - th * lfs - zw[0],
                             -s0[1] + z - phi * bf / 2 - th * lfs - zw[1],
                             -s0[2] + z + phi * br / 2 + th * lrs - zw[2],
                             -s0[3] + z - phi * br / 2 + th * lrs - zw[3]])
            spring = np.array([cf, cf, cr, cr]) * comp
            arb_f = 0.5 * caf * (comp[0] - comp[1])
            arb_r = 0.5 * car * (comp[2] - comp[3])
            arb = np.array([arb_f, -arb_f, arb_r, -arb_r])
            jack = np.array([slf * Fy[0] + sadf * Fx[0], -slf * Fy[1] + sadf * Fx[1],
                             slr * Fy[2] - sadr * Fx[2], -slr * Fy[3] - sadr * Fx[3]])
            susp = spring + arb + jack
            tspring = ct * (zw - s0t)
            wheel = susp - tspring - mt * G
            heave = -susp.sum() - ms * G - (Flf + Flr)
            roll = Fy.sum() * h - susp[0] * bf / 2 + susp[1] * bf / 2 - susp[2] * br / 2 + susp[3] * br / 2
            aero_ty = -Flr * lra + Flf * lfa
            pitch = (-Fx.sum() * h + (susp[0] + susp[1]) * lf - (susp[2] + susp[3]) * lr
                     + ms * G * (lf - lfs) + aero_ty)
            return np.r_[wheel, heave, roll, pitch], -tspring, comp

        y0, f0 = np.zeros(7), np.zeros(10)
        r0 = residual(y0, f0)[0]
        J = np.column_stack([residual(e, f0)[0] - r0 for e in np.eye(7)])
        Gm = np.column_stack([residual(y0, e)[0] - r0 for e in np.eye(10)])
        Y0 = -np.linalg.solve(J, r0)
        Yf = -np.linalg.solve(J, Gm)
        Fz = lambda y, f: residual(y, f)[1]
        N0 = Fz(Y0, f0)
        A = np.column_stack([Fz(Y0 + Yf[:, i], e) - N0 for i, e in enumerate(np.eye(10))])
        comp0 = residual(Y0, f0)[2]
        Bc = np.column_stack([residual(Y0 + Yf[:, i], e)[2] - comp0 for i, e in enumerate(np.eye(10))])
        rh_rows = np.array([[1.0, 0, -lfa, 0, 0, 0, 0], [1.0, 0, lra, 0, 0, 0, 0]])   # d rh / d y
        return dict(N0=N0, A=A, Y0=Y0, Yf=Yf, comp0=comp0, Bcomp=Bc, s0=s0, rh_rows=rh_rows,
                    packer_gap=np.array([d.p("suspension.packer.gap_m.front")] * 2 + [d.p("suspension.packer.gap_m.rear")] * 2),
                    packer_on=d.p("suspension.packer.enabled") > 0.5)

    def roll_stiffness_front_share(self):
        from ocd_lap.translator import roll_stiffness
        return roll_stiffness(self.prm)[1]

    def digest(self) -> str:
        blob = json.dumps({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in sorted(self.prm.items())
                           if not k.startswith("initial_state")}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def summary(self) -> dict:
        lm = self.load_map()
        return dict(name=self.name, setup=self.setup_name, params_sha16=self.digest(), m=self.m, Iz=self.Iz, l=self.l,
                    l_f=self.l_f, h_cg=self.h_cg, b_f=self.b_f, b_r=self.b_r, R=self.R_r, c_rr=self.c_rr,
                    I_wf=self.I_wf, I_wr=self.I_wr, engine_inertia=list(self.engine_inertia),
                    brake_bias_front=self.brake_bias_front, max_brake_torque_Nm=self.max_brake_torque_Nm,
                    final_drive=self.final_drive, ratios=list(self.ratios), lsd=self.lsd,
                    steer_max=self.steer_max, steer_rate_max=self.steer_rate_max, footprint=self.footprint,
                    CzA_static=list(self.aero_linear()[0]), CxA=self.cd_A, rho=self.rho,
                    roll_stiffness_front_share_translator=self.roll_stiffness_front_share(),
                    roll_centre_h=self.p("suspension.roll_center_height_m"),
                    pitch_centre_h=self.p("suspension.pitch_center_height_m.accel"),
                    static_loads_N=list(lm["N0"]))


def from_backend(setup_name: str, car: str = "sf70h_tum", max_brake_torque_Nm: float = 15000.0) -> OCDCar:
    """Effective OCD car at a named setup (rlsetup.driving.setups), ERS off, engine powertrain."""
    from ocd_lap import cars
    from ocd_lap.backend import OCDBackend, StartPose
    from rlsetup.driving.setups import get
    setup = get(setup_name)
    b = OCDBackend(powertrain="engine", car=car, ers=False, max_brake_torque_Nm=max_brake_torque_Nm)
    b.reset(setup, StartPose())
    pt = b.powertrain
    for w in ("front", "rear"):   # left == right everywhere (the OCP model relies on it for MF)
        for k in MF_KEYS:
            assert b.params[TM + f"{w}_left.{k}"] == b.params[TM + f"{w}_right.{k}"], k
    return OCDCar(name=car, setup_name=setup_name, prm=dict(b.params), brake_bias_front=b.brake_bias_front,
                  max_brake_torque_Nm=b.max_brake_torque_Nm, final_drive=setup.final_drive_ratio,
                  pt_data=dict(pt.d), ratios=np.array(pt.ratios, float), power_coeffs=np.array(pt.power_coeffs, float),
                  engine_inertia=np.array(pt.engine_inertia_kgm2, float), footprint=tuple(cars.get(car).footprint))
