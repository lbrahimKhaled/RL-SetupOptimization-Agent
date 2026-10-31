#!/usr/bin/env python3
"""Closed-loop replay of TUM's 'none' lap (no ERS / DRS / recuperation, 94.886 s) in the TEMPORARY
debug OCD car of ocd_first_lap branch `tum-debug` (worktree ../ocd_first_lap-tumdebug).

    .venv/bin/python scripts/comparison/tum_closed_loop.py [--ocd-root <worktree>] [--out <dir>]

Debug OCD (branch tum-debug, Python-only changes, native .so symlinked from the main checkout):
  car    'sf70h_tum_debug' = sf70h_tum (TUM mass, geometry, tyre peak-mu law, engine, gearbox)
         + ERS off + constant aero CzA 2.20 / 2.68, CxA 1.56 (no ride-height terms) + TUM r(v)
  setup  p1_TUM (rlsetup.driving.setups; roll split = static axle load, brake bias 0.49, LSD
         locked on drive), total brake torque 15 kN m: the settings of the 95.65 s OCD QSS
  track  Track('Shanghai', tum_mode=True): path = TUM raceline (Track.reference), off-track =
         CoG outside the racetrack-database edges; the 4.6 x 2.0 m footprint check is logged only

Controller, all references indexed by raceline arc length s (TUM points matched onto the
polyline by position, as tum_pure_pursuit.TumProfile), 50 Hz, one command per 10 x 2 ms steps:
  lateral       curvature command on the CoG path (s-domain 2nd-order error dynamics):
                kappa_cmd = kappa_TUM(s + v tau_lat) - w^2 e - 2 zeta w chi,
                e = CoG offset (left +), chi = course-angle error (yaw + beta - line heading),
                w = min(w_t / v, w_s_max) [1/m]
                delta = atan(L kappa_cmd) + K_us v^2 kappa_cmd   (understeer feed-forward)
  longitudinal  a_cmd = f^2 ax_TUM(s_p) + kp (f v_TUM(s_p) - v) + ki int(.) dt, s_p = s + v tau_lon
                axle torque = r(v) (m e_i a_cmd + drag + rolling) -> throttle / capacity (drive)
                or brake torque -> pedal / 15 kN m (brake)
  limiters      traction control: throttle factor x0.6 per tick while the mean rear slip ratio
                > 0.08, else +0.1; per-wheel ABS, same rule on each wheel's slip < -0.10, applied
                through the debug backend's `brake_scale` (TUM brakes every wheel at its limit);
                front slip-angle cap 11.5 -> 7.5 deg from 25 to 60 m/s (no steering past the MF
                peak); yaw-rate damping delta -= k_r (r - v kappa_cmd).
Speed scale f: 1.00, 0.98, 0.96, 0.94, 0.92; the stop rule takes the first f whose flying lap 2
completes without spin; --all also records the remaining f (diagnostic). Per 50 Hz tick:
common.PP_COLUMNS + debug columns (EXTRA); per run CSV + .meta.json; tyre_check.json,
lateral_check.json, qss_debug_car.json, analysis.json, runs.json. FINDINGS.md is written by hand.
Caveat: in this debug car OCD's logged x_dot (column ax_mps2, ocd_trace) differs from dv/dt by up
to ~7 m/s^2 on straights; analyse() uses dv/dt. Controller constants are PLACEHOLDERs.
"""
import argparse
from dataclasses import asdict, dataclass
import datetime
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

DEFAULT_OCD_ROOT = Path('/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-tumdebug')
TUM_CSV = common.OUT_ROOT / 'tum' / 'tum_none.csv'
CAR = 'sf70h_tum_debug'
SETUP = 'p1_TUM'
BRAKE_TORQUE_NM = 15000.
F_LIST = (1.00, .98, .96, .94, .92)
RUN_IN_M = 200.
LAPS = 2
CONTROL_STEPS = 10                    # 50 Hz commands on the 2 ms physics step
SPIN_RAD = math.pi / 2
STALL_MPS, STALL_S, MAX_TIME_S = 3., 2., 400.
G = 9.81
WHEELS = ('FL', 'FR', 'RL', 'RR')
_LONG = dict(FL='front_left', FR='front_right', RL='rear_left', RR='rear_right')


@dataclass(frozen=True)
class Gains:
    """Controller constants (PLACEHOLDER tuning values, hand-set in a few development runs)."""
    tau_lat_s: float = .10       # curvature preview: PT1 steering 0.06 s + half a 20 ms tick
    w_t_radps: float = 4.        # lateral error bandwidth in time
    w_s_max_pm: float = .20      # cap of the spatial bandwidth at low speed [1/m]
    zeta: float = .9
    k_us_rad_per_mps2: float = .0003   # 0.17 deg/g (docs/SF70H_TUM_CAR.md ramp steer)
    alpha_f_max_rad: tuple = (.20, .13)  # front slip-angle cap at <= 25 / >= 60 m/s: the MF peak
                                         # is 9-12 deg at 2-3 kN, 7 deg at 6 kN (tyre_check.json)
    tau_lon_s: float = .04       # speed / ax preview (20 ms ZOH + reaction)
    kp_per_s: float = 2.         # speed error -> acceleration [1/s]
    ki_per_s2: float = .5
    i_lim_mps: float = 3.        # |integral| limit [m/s]
    k_r_s: float = .04           # yaw-rate damping: delta -= k_r (r - v kappa_cmd)
    tc_slip: float = .08         # traction control target: mean rear slip ratio (DemoConfig start)
    abs_slip: float = .10        # ABS target: most negative wheel slip ratio
    cut: float = .6              # limiter factor x cut per 20 ms tick above the target
    rise: float = .1             # limiter factor + rise per tick below it (back to 1 in 0.2 s)


# ---------------------------------------------------------------- pure helpers (tested)
def load_tum(csv_path, raceline):
    """TUM rows (closing row dropped) with s on the raceline polyline (position matched, as
    tum_pure_pursuit.TumProfile). Returns a dict of arrays sorted by s in [0, L)."""
    d = np.genfromtxt(csv_path, delimiter=',', names=True)
    d = d[d['closing_point'] == 0]
    s, hint = [], None
    for x, y in zip(d['x_m'], d['y_m']):
        hint = raceline.match((x, y), hint)
        s.append(hint)
    L = raceline.length
    s = np.unwrap(np.asarray(s), period=L)
    s = (s - np.floor(s[0] / L) * L) % L
    order = np.argsort(s)
    out = {k: np.asarray(d[k], float)[order] for k in d.dtype.names}
    out['s'] = s[order]
    out['L'] = L
    return out


class Profile:
    """TUM reference along the raceline: v, kappa (linear, periodic), ax (TUM's step value, held
    over the 5 m step that starts at the point), t (time since TUM's line)."""

    def __init__(self, tum, lap_time_s):
        self.s, self.L = tum['s'], tum['L']
        self.v, self.kappa, self.ax = tum['speed_mps'], tum['kappa_radpm'], tum['ax_mps2']
        self.t, self.lap_time_s = tum['t_s'], lap_time_s
        self.tum = tum

    def v_at(self, s):
        return np.interp(np.asarray(s) % self.L, self.s, self.v, period=self.L)

    def kappa_at(self, s):
        return np.interp(np.asarray(s) % self.L, self.s, self.kappa, period=self.L)

    def ax_at(self, s):
        i = np.searchsorted(self.s, np.asarray(s) % self.L, side='right') - 1
        return self.ax[i % len(self.ax)]

    def t_at(self, s):
        """TUM time since the line at raceline s in [0, L] (closing point = lap time)."""
        return np.interp(np.asarray(s), np.r_[self.s, self.L], np.r_[self.t, self.lap_time_s])


def lateral_command(e, chi, v, kappa_ff, gains, wheelbase):
    """Steering [rad] from CoG offset e (left +), course error chi (left +), speed v. A car left
    of the line, or pointing left of it, gets a right (negative) correction."""
    w = min(gains.w_t_radps / max(v, 1.), gains.w_s_max_pm)
    kappa_cmd = kappa_ff - w * w * e - 2. * gains.zeta * w * chi
    return math.atan(wheelbase * kappa_cmd) + gains.k_us_rad_per_mps2 * v * v * kappa_cmd, kappa_cmd


def slip_limiter(factor, slip, target, cut, rise):
    """Multiplicative slip limiter (traction control / ABS) on a pedal factor in [0, 1]:
    x cut while |slip| > target, + rise (up to 1) otherwise."""
    return factor * cut if abs(slip) > target else min(1., factor + rise)


def envelope_exponent(r45):
    """Exponent n of |x|^n + |y|^n = 1 through the 45 deg point at radius r45 (n = 2: circle)."""
    return math.log(2.) / (math.log(math.sqrt(2.)) - math.log(r45))


def mf52_forces(p, Fz, kappa, alpha):
    """OCD MF5.2 (mf_52.cpp, zero camber) combined-slip (Fx, Fy) for one wheel's parameter
    getter `p(name)`; `alpha` is OCD's lateral argument (= -slip angle). Arrays broadcast."""
    fz0 = p('LFZO') * p('FNOMIN')
    dfz = (Fz - fz0) / fz0
    kx = kappa + (p('PHX1') + p('PHX2') * dfz) * p('LHX')
    C = p('PCX1') * p('LCX')
    D = (p('PDX1') + p('PDX2') * dfz) * p('LMUX') * Fz
    E = (p('PEX1') + p('PEX2') * dfz + p('PEX3') * dfz ** 2) * (1 - p('PEX4') * np.sign(kx)) * p('LEX')
    K = Fz * (p('PKX1') + p('PKX2') * dfz) * np.exp(p('PKX3') * dfz) * p('LKX')
    B = K / (C * D)
    fx0 = D * np.sin(C * np.arctan(B * kx - E * (B * kx - np.arctan(B * kx)))) \
        + Fz * (p('PVX1') + p('PVX2') * dfz) * p('LVX') * p('LMUX')
    Ky = p('PKY1') * fz0 * np.sin(p('PKY4') * np.arctan(Fz / (p('PKY2') * fz0))) * p('LKY')
    ay = alpha + (p('PHY1') + p('PHY2') * dfz) * p('LHY')
    Cy = p('PCY1') * p('LCY')
    muy = (p('PDY1') + p('PDY2') * dfz) * p('LMUY')
    Dy = muy * Fz
    By = Ky / (Cy * Dy)
    Ey = (p('PEY1') + p('PEY2') * dfz) * (1 - p('PEY3') * np.sign(ay)) * p('LEY')
    fy0 = Dy * np.sin(Cy * np.arctan(By * ay - Ey * (By * ay - np.arctan(By * ay)))) \
        + Fz * (p('PVY1') + p('PVY2') * dfz) * p('LVY') * p('LMUY')
    shxa = p('RHX1')
    bxa = p('RBX1') * np.cos(np.arctan(p('RBX2') * kappa)) * p('LXAL')
    cxa, exa = p('RCX1'), p('REX1') + p('REX2') * dfz
    g = lambda b, c, e_, x: np.cos(c * np.arctan(b * x - e_ * (b * x - np.arctan(b * x))))
    gxa = g(bxa, cxa, exa, alpha + shxa) / g(bxa, cxa, exa, shxa)
    shyk = p('RHY1') + p('RHY2') * dfz
    byk = p('RBY1') * np.cos(np.arctan(p('RBY2') * (alpha - p('RBY3')))) * p('LYKA')
    cyk, eyk = p('RCY1'), p('REY1') + p('REY2') * dfz
    gyk = g(byk, cyk, eyk, kappa + shyk) / g(byk, cyk, eyk, shyk)
    svyk = muy * Fz * (p('RVY1') + p('RVY2') * dfz) * np.cos(np.arctan(p('RVY4') * alpha)) \
        * np.sin(p('RVY5') * np.arctan(p('RVY6') * kappa)) * p('LVYKA')
    return gxa * fx0, gyk * fy0 + svyk


def tum_mu(axle, Fz, spec):
    t, fz0 = spec['tyres'][axle], spec['fz_0']
    return t['mux'] + t['dmux_dfz'] * (Fz - fz0), t['muy'] + t['dmuy_dfz'] * (Fz - fz0)


def tyre_check(params, spec, P, loads=(2000., 4000., 6000., 9000.)):
    """Peak mu_x / mu_y of OCD's MF5.2 (debug car parameters) vs TUM's linear law, and the
    combined-slip envelope (max |F| per direction in (Fx/Fx_peak, Fy/Fy_peak)) vs TUM's ellipse."""
    kap = np.linspace(-.6, .6, 1201)
    alp = np.linspace(-.5, .5, 1001)
    K, A = np.meshgrid(np.linspace(-.5, .5, 401), np.linspace(-.4, .4, 401))
    rows = []
    for axle, wheel in (('front', 'front_left'), ('rear', 'rear_left')):
        p = lambda k, w=wheel: float(params[P + f'z.tire_model.{w}.{k}'])
        for fz in loads:
            fx, _ = mf52_forces(p, fz, kap, 0. * kap)
            _, fy = mf52_forces(p, fz, 0. * alp, alp)
            mux_t, muy_t = tum_mu(axle, fz, spec)
            fxd, fxb = fx.max(), -fx.min()
            fyp, fyn = fy.max(), -fy.min()
            cx, cy = mf52_forces(p, fz, K, A)
            out = dict(axle=axle, Fz_N=fz, tum_mux=mux_t, mf_mux_drive=fxd / fz, mf_mux_brake=fxb / fz,
                       tum_muy=muy_t, mf_muy_pos=fyp / fz, mf_muy_neg=fyn / fz,
                       kappa_peak_drive=float(kap[fx.argmax()]), kappa_peak_brake=float(kap[fx.argmin()]),
                       alpha_peak_deg=float(np.degrees(abs(alp[fy.argmax()]))))
            for quad, sx in (('brake', -1.), ('drive', 1.)):
                m = (np.sign(cx) == sx) | (cx == 0.)
                xn = np.abs(cx[m]) / (fxb if sx < 0 else fxd)
                yn = np.abs(cy[m]) / max(fyp, fyn)
                ang = np.degrees(np.arctan2(yn, xn))
                rad = np.hypot(xn, yn)
                for a in (30, 45, 60):
                    sel = np.abs(ang - a) < 1.
                    out[f'{quad}_r{a}'] = float(rad[sel].max()) if sel.any() else None
                out[f'{quad}_exponent_45'] = envelope_exponent(out[f'{quad}_r45'])
            rows.append(out)
    return rows


def lateral_check(setup, speeds=(15., 25., 35., 50., 70.), overrides=None):
    """Steady-state lateral limit of the debug car (ocd_lap.handling.ramp_steer, 0.03 rad/s,
    LSD as in the setup and opened) against TUM's own axle-wise limit (sf70h_checks.tum_limits)."""
    from ocd_lap import handling
    import importlib.util
    import ocd_lap
    path = Path(ocd_lap.__file__).resolve().parents[1] / 'scripts' / 'sf70h_checks.py'
    spec = importlib.util.spec_from_file_location('sf70h_checks', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    out = []
    for v in speeds:
        a = handling.ramp_steer(setup, v, rate_radps=.03, car=CAR, overrides=overrides)
        b = handling.ramp_steer(setup.replace(lsd_lock_drive=0.), v, rate_radps=.03, car=CAR,
                                overrides=overrides)
        tum = checks.tum_limits(v)[0]
        out.append(dict(v_mps=v, tum_ay_max_mps2=tum, ocd_ay_max_mps2=a['max_ay_mps2'],
                        ocd_over_tum=a['max_ay_mps2'] / tum, steer_at_max_deg=a['steer_at_max_deg'],
                        spun=a['spun'], understeer_deg_per_g=a['understeer_gradient_deg_per_g'],
                        ocd_ay_max_lsd_open_mps2=b['max_ay_mps2']))
    return out


def corner_segments(prof, window_m=75., min_drop_mps=5.):
    """Split the lap at TUM's braking points: local maxima of v_TUM (within +-window_m) followed
    by a drop of at least min_drop_mps. Returns sorted s of the cuts on [0, L)."""
    s, v, L = prof.s, prof.v, prof.L
    cuts = []
    for i in range(len(s)):
        d = (s - s[i] + L / 2) % L - L / 2
        near = np.abs(d) <= window_m
        if v[i] < v[near].max():
            continue
        ahead = (d > 0) & (d <= 400.)
        if v[i] - v[ahead].min() >= min_drop_mps:
            cuts.append(float(s[i]))
    return sorted(cuts)


LIMITS = ('braking', 'traction', 'power', 'lateral_grip', 'controller')


def classify(row):
    """Which limit binds at one OCD sample: brake pedal on -> braking; throttle cut by the
    traction control -> traction; full throttle, no cut -> power; an axle at >= 0.9 of TUM's
    friction potential -> lateral grip; otherwise (part pedal, grip unused) -> controller."""
    if row['brake'] > .02:
        return 'braking'
    if row['throttle_ff'] > 0. and row['tc_factor'] < .95:
        return 'traction'
    if row['throttle'] >= .99:
        return 'power'
    if max(row['mu_use_front'], row['mu_use_rear']) >= .9:
        return 'lateral_grip'
    return 'controller'


def analyse(csv_path, prof, step_m=5.):
    """Lap 2 of a run against TUM on a common raceline-s grid: dt(s) = t_OCD(s) - t_TUM(s),
    per corner segment (cut at TUM's braking points) the time lost, speeds and the binding
    limit. A speed deficit made in a corner is carried down the next straight, so a segment's
    dt is split over the `classify` classes in proportion to where the deficit dv = v_TUM -
    v_OCD GROWS (sum of max(d dv / ds, 0) per class), not where the time is counted."""
    d = np.genfromtxt(csv_path, delimiter=',', names=True)
    d = d[d['lap'] == 2]
    if not len(d):
        return None
    s = np.maximum.accumulate(np.unwrap(d['s_raceline_m'], period=prof.L))
    s -= np.floor(s[0] / prof.L) * prof.L if s[0] >= prof.L else 0.
    t = d['t_lap_s']
    grid = np.arange(0., prof.L, step_m)
    sel = np.r_[True, np.diff(s) > 1e-9]
    t_ocd = np.interp(grid, np.r_[0., s[sel]], np.r_[0., t[sel]])
    t_tum = prof.t_at(grid)
    dt = t_ocd - t_tum
    rate = np.gradient(dt, grid)
    idx = np.clip(np.searchsorted(s, grid), 0, len(d) - 1)
    names = d.dtype.names
    cls = np.array([classify({k: d[k][i] for k in names}) for i in idx])
    v_ocd = step_m / np.maximum(np.gradient(t_ocd), 1e-6)
    growth = np.maximum(np.gradient(prof.v_at(grid) - v_ocd, grid), 0.)

    def split(m, total_dt):
        g = {c: float(np.sum(growth[m & (cls == c)])) for c in LIMITS}
        tot = sum(g.values())
        return {c: (total_dt * g[c] / tot if tot > 0. else 0.) for c in LIMITS}
    cuts = corner_segments(prof)
    edges = sorted(set([0.] + cuts + [prof.L]))
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (grid >= a) & (grid < b)
        if not m.any():
            continue
        seg_dt = float(dt[m][-1] - (dt[m][0] - rate[m][0] * step_m))
        loss = split(m, seg_dt)
        k = np.argmin(prof.v_at(grid[m]))
        segs.append(dict(s_from_m=a, s_to_m=b, dt_s=seg_dt,
                         v_tum_min_mps=float(prof.v_at(grid[m])[k]), v_ocd_min_mps=float(v_ocd[m].min()),
                         s_apex_tum_m=float(grid[m][k]),
                         v_tum_entry_mps=float(prof.v_at(a)), v_ocd_entry_mps=float(np.interp(a, grid, v_ocd)),
                         max_abs_e_m=float(np.abs(d['e_raceline_m'][idx[m]]).max()),
                         cog_off_track_share=float(1. - d['on_track_cog'][idx[m]].mean()),
                         loss_by_limit_s=loss, binding=max(loss, key=loss.get)))
    total = {c: sum(x['loss_by_limit_s'][c] for x in segs) for c in LIMITS}
    # straight-line capability at equal speed (|ay| < 5 m/s^2), 5 m/s bins: full throttle without
    # traction cut (OCD) vs TUM's accelerating points, and peak braking (brake pedal > 0.3)
    # ax from dv/dt: in this debug car the logged OCD x_dot (column ax_mps2) differs from dv/dt
    ax = np.gradient(d['vx_mps'], d['t_s'])
    bins = np.arange(10., 90., 5.)
    sl = np.abs(d['ay_mps2']) < 5.
    acc = sl & (d['throttle'] >= .99) & (d['tc_factor'] >= .95)
    brk = sl & (d['brake'] > .3)
    tsl = np.abs(prof.tum['ay_mps2']) < 5.
    cap = []
    for lo in bins:
        mo = (d['speed_mps'] >= lo) & (d['speed_mps'] < lo + 5.)
        mt = (prof.v >= lo) & (prof.v < lo + 5.)
        q = lambda x: float(np.median(x)) if len(x) else None
        cap.append(dict(v_from_mps=float(lo), ocd_ax_full_throttle=q(ax[mo & acc]),
                        tum_ax_accelerating=q(prof.ax[mt & tsl & (prof.ax > 0)]),
                        ocd_ax_brake_min=float(ax[mo & brk].min()) if (mo & brk).any() else None,
                        tum_ax_brake_min=float(prof.ax[mt & (prof.ax < 0)].min()) if (mt & (prof.ax < 0)).any() else None))
    return dict(lap2_s=float(t[-1]), dt_end_s=float(dt[-1]), loss_by_limit_s=total,
                segments=sorted(segs, key=lambda x: -x['dt_s']), cuts_m=cuts,
                share_of_lap_by_class={c: float(np.mean(cls == c)) for c in LIMITS}, straight_line=cap)


# ---------------------------------------------------------------- simulation
def setup_paths(ocd_root):
    ocd_root = Path(ocd_root).resolve()
    sys.path.insert(0, str(ocd_root))
    for name in [m for m in sys.modules if m == 'ocd_lap' or m.startswith('ocd_lap.')]:
        del sys.modules[name]
    import ocd_lap
    if Path(ocd_lap.__file__).resolve().parents[1] != ocd_root:
        raise RuntimeError(f'ocd_lap imported from {ocd_lap.__file__}, expected {ocd_root}')
    return ocd_root


def initial_gear(pt, omega):
    """TUM find_gear: the lowest gear whose rpm is below its shift rpm (initial condition only)."""
    for g in range(len(pt.ratios)):
        if pt.rpm(omega, g) < pt.d['n_shift'][g]:
            return g
    return len(pt.ratios) - 1


EXTRA = ['v_tum_mps', 'ax_tum_mps2', 'kappa_ref_radpm', 'kappa_cmd_radpm', 'a_cmd_mps2', 'speed_err_mps',
         'course_err_rad', 'steer_unlimited_rad', 'tc_factor', 'abs_factor', 'abs_FL', 'abs_FR', 'abs_RL', 'abs_RR', 'throttle_ff', 'on_track_cog',
         'Fx_FL_N', 'Fx_FR_N', 'Fx_RL_N', 'Fx_RR_N', 'Fy_FL_N', 'Fy_FR_N', 'Fy_RL_N', 'Fy_RR_N',
         'mu_use_front', 'mu_use_rear', 'aero_Fz_N', 'tyre_radius_m']
FORCE_SIGNALS = {**{f'Fx_{w}_N': f'vehicle_dynamics/imr/tire_forces_tire_frame_N/{_LONG[w]}/longitudinal' for w in WHEELS},
                 **{f'Fy_{w}_N': f'vehicle_dynamics/imr/tire_forces_tire_frame_N/{_LONG[w]}/lateral' for w in WHEELS},
                 'tyre_radius_m': 'vehicle_dynamics/imr/dynamic_tire_radius/rear_left'}


KAPPA_SIGNALS = [f'vehicle_dynamics/x_vec/kappa_tire_{w}' for w in WHEELS]


def mu_use(row, spec):
    """Per-axle friction use with TUM's axle ellipse (exponent 2) on TUM's potentials at the
    OCD wheel loads: sqrt((sum Fx / sum Fx_pot)^2 + (sum Fy / sum Fy_pot)^2)."""
    out = {}
    for axle, (a, b) in (('front', ('FL', 'FR')), ('rear', ('RL', 'RR'))):
        fxp = fyp = 0.
        for w in (a, b):
            mux, muy = tum_mu(axle, max(row[f'Fz_{w}_N'], 30.), spec)
            fxp += mux * max(row[f'Fz_{w}_N'], 30.)
            fyp += muy * max(row[f'Fz_{w}_N'], 30.)
        fx = row[f'Fx_{a}_N'] + row[f'Fx_{b}_N']
        fy = row[f'Fy_{a}_N'] + row[f'Fy_{b}_N']
        out[f'mu_use_{axle}'] = math.hypot(fx / fxp, fy / fyp)
    return out


def run(f, ctx, gains):
    from ocd_lap.backend import OCDBackend, StartPose, Inputs, DT
    import ocd_trace
    track, ttrack, geo, prof, setup, spec_tum = (ctx[k] for k in ('track', 'ttrack', 'geo', 'prof', 'setup', 'spec_tum'))
    backend = OCDBackend(powertrain='engine', car=CAR, max_brake_torque_Nm=BRAKE_TORQUE_NM,
                         overrides=ctx.get('overrides'))
    spec = backend.spec
    raceline = geo.raceline
    L = raceline.length
    s0 = (-RUN_IN_M) % L
    x0, y0 = raceline.point(s0)
    v0 = f * float(prof.v_at(s0))
    st = backend.reset(setup, StartPose(float(x0), float(y0), raceline.heading(s0), v0))
    pt = backend.powertrain
    pt.gear = initial_gear(pt, backend.car.rear_axle_omega())
    gear0 = pt.gear + 1
    m = backend.params['vehicle_dynamics_double_track.mass_vehicle_kg']
    rho, cxa = 1.18, 1.56
    cza = 2.20 + 2.68
    f_roll = spec_tum['f_roll']
    wheelbase = spec.wheelbase_m
    geo.reset(s_path=None, s_raceline=s0, s_path_ref=None)
    s_path0 = float(track.match([x0, y0])[0])
    geo.reset(s_path=s_path0, s_raceline=s0, s_path_ref=geo.path_ref.match((x0, y0)))
    clock = ocd_trace.LapClock(ttrack, s0, LAPS + 1)
    s_ref, dist, t, steps, stall_s, integ, tc, abw = s0, 0., 0., 0, 0., 0., 1., [1.] * 4
    rows, cause = [], None
    while cause is None:
        vx, vy = st['vx'], st['vy']
        v = math.hypot(vx, vy)
        e, h = raceline.errors_at(s_ref, st['x'], st['y'], st['yaw'])
        chi = (h + math.atan2(vy, max(vx, .1)) + math.pi) % (2 * math.pi) - math.pi
        kff = float(prof.kappa_at(s_ref + v * gains.tau_lat_s))
        steer, kcmd = lateral_command(e, chi, v, kff, gains, wheelbase)
        steer -= gains.k_r_s * (st['yaw_rate'] - v * kcmd)
        lf = ctx['lf']
        a_kin = math.atan2(vy + lf * st['yaw_rate'], max(vx, 1.))   # front-axle velocity angle
        steer_free = steer
        a_cap = float(np.interp(v, [25., 60.], gains.alpha_f_max_rad))
        steer = float(np.clip(steer, a_kin - a_cap, a_kin + a_cap))
        steer = float(np.clip(steer, -.6, .6))
        sp = s_ref + v * gains.tau_lon_s
        v_t = f * float(prof.v_at(sp))
        a_ff = f * f * float(prof.ax_at(sp))
        err = v_t - vx
        integ = float(np.clip(integ + err * CONTROL_STEPS * DT, -gains.i_lim_mps, gains.i_lim_mps))
        a_cmd = a_ff + gains.kp_per_s * err + gains.ki_per_s2 * integ
        r = float(np.interp(v, [0., 100.], [ctx['r0'] * x for x in ctx['r_scale']]))
        resist = .5 * rho * cxa * v * v + f_roll * (m * G + .5 * rho * cza * v * v)
        capacity = backend.drive_torque_capacity()
        e_i = pt.d['e_i'][pt.gear]
        if a_cmd >= 0.:
            torque = r * (m * e_i * a_cmd + resist)
            thr_ff = torque / max(capacity, 1.)
            brake = 0.
        else:
            torque = r * (m * a_cmd + resist)        # negative: brakes take the rest
            thr_ff = max(0., torque / max(capacity, 1.))
            brake = max(0., -torque) / BRAKE_TORQUE_NM
        rear_slip = .5 * (st['slip_rl'] + st['slip_rr'])   # axle slip (TUM: LSD sums both wheels)
        tc = slip_limiter(tc, rear_slip, gains.tc_slip, gains.cut, gains.rise)
        kap = backend.car.signals(KAPPA_SIGNALS)
        abw = [slip_limiter(a, min(k, 0.), gains.abs_slip, gains.cut, gains.rise) for a, k in zip(abw, kap)]
        backend.brake_scale = abw if brake > 0. else [1.] * 4   # per-wheel ABS (debug backend)
        ab = min(abw)
        inputs = Inputs(steer, float(np.clip(thr_ff * tc, 0., 1.)), float(np.clip(brake, 0., 1.)))
        inside_fp = inside_cog = True
        for _ in range(CONTROL_STEPS):
            st = backend.step(inputs)
            steps += 1
            if not all(map(math.isfinite, st.values())):
                cause = 'nonfinite'
                break
            inside_fp &= bool(track.contains_car(st['x'], st['y'], st['yaw'], spec.footprint))
            inside_cog &= not ttrack.off_track(st['x'], st['y'], st['yaw'])
        t0, t = t, steps * DT
        if cause:
            break
        g = geo.locate(st['x'], st['y'], st['yaw'])
        d0 = dist
        dist += (g['s_raceline_m'] - s_ref + L / 2) % L - L / 2
        s_ref = g['s_raceline_m']
        clock.update(d0, dist, t0, t)
        row = dict(t_s=t, lap=clock.lap, t_lap_s=clock.t_lap(t), progress_m=dist, **g,
                   **ocd_trace.backend_sample(backend, st, inputs), on_track=int(inside_fp), target_speed_mps=v_t,
                   v_tum_mps=float(prof.v_at(s_ref)), ax_tum_mps2=float(prof.ax_at(s_ref)),
                   kappa_ref_radpm=kff, kappa_cmd_radpm=kcmd, a_cmd_mps2=a_cmd, speed_err_mps=err,
                   course_err_rad=chi, steer_unlimited_rad=steer_free, tc_factor=tc, abs_factor=ab, **{f'abs_{w}': a for w, a in zip(WHEELS, backend.brake_scale)}, throttle_ff=thr_ff,
                   on_track_cog=int(inside_cog))
        row.update(zip(FORCE_SIGNALS, backend.car.signals(list(FORCE_SIGNALS.values()))))
        row['aero_Fz_N'] = backend.get_telemetry()['aero_Fz_N']
        row.update(mu_use(row, spec_tum))
        rows.append(row)
        stall_s = stall_s + (t - t0) if row['speed_mps'] < STALL_MPS else 0.
        if abs(g['heading_err_raceline_rad']) > SPIN_RAD:
            cause = 'spin'
        elif stall_s > STALL_S:
            cause = 'stall'
        elif clock.lap == LAPS + 1:
            cause = 'lap2_complete'
        elif t >= MAX_TIME_S:
            cause = 'time_limit'
    laps = clock.lap_times
    lap2 = [r for r in rows if r['lap'] == 2]
    mean = lambda k: float(np.mean([r[k] for r in lap2])) if lap2 else None
    summary = dict(f=f, termination=cause, completed_lap2=cause == 'lap2_complete', lap_times_s=laps,
                   lap2_s=laps[1] if len(laps) >= 2 else None, sim_time_s=t, progress_m=dist,
                   spin_at_s_raceline_m=rows[-1]['s_raceline_m'] if cause == 'spin' and rows else None,
                   spin_lap=rows[-1]['lap'] if cause == 'spin' and rows else None,
                   on_track_cog_fraction_lap2=mean('on_track_cog'), on_track_footprint_fraction_lap2=mean('on_track'),
                   max_abs_e_raceline_lap2_m=float(max(abs(r['e_raceline_m']) for r in lap2)) if lap2 else None,
                   rms_e_raceline_lap2_m=float(np.sqrt(np.mean([r['e_raceline_m'] ** 2 for r in lap2]))) if lap2 else None,
                   v_max_lap2_kmh=float(3.6 * max(r['speed_mps'] for r in lap2)) if lap2 else None,
                   start={'s_raceline_m': s0, 'speed_mps': v0, 'gear': gear0})
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ocd-root', default=str(DEFAULT_OCD_ROOT))
    ap.add_argument('--out', default=None)
    ap.add_argument('--f', type=float, nargs='+', default=list(F_LIST))
    ap.add_argument('--all', action='store_true', help='run every f (default: stop at the first clean lap 2)')
    ap.add_argument('--gain', nargs=2, action='append', default=[], metavar=('NAME', 'VALUE'),
                    help='diagnostic only: override one Gains field (float)')
    ap.add_argument('--lsd-lock', type=float, default=None,
                    help='diagnostic only: Setup.lsd_lock_drive instead of the p1_TUM value (1.0)')
    ap.add_argument('--tyre-scale', nargs=2, action='append', default=[], metavar=('NAME', 'VALUE'),
                    help='diagnostic only: MF5.2 scaling factor (e.g. LKX 1.6) on all four wheels')
    ap.add_argument('--no-checks', action='store_true', help='skip the tyre / lateral / QSS checks')
    args = ap.parse_args()
    ocd_root = setup_paths(args.ocd_root)
    from ocd_lap.backend import OCDBackend, StartPose
    from ocd_lap.track import Track
    from ocd_lap.translator import P
    from ocd_lap import cars
    import ocd_trace
    from rlsetup.driving import setups
    from rlsetup.provenance import run_provenance, git_state
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    out = Path(args.out) if args.out else common.OUT_ROOT / f'tum_ocd_debug_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    from dataclasses import replace as _replace
    gains = _replace(Gains(), **{k: float(v) for k, v in args.gain})
    track = Track('Shanghai')
    ttrack = Track('Shanghai', tum_mode=True)
    geo = ocd_trace.Geometry(track)
    tum = load_tum(TUM_CSV, geo.raceline)
    tum_meta = json.loads(TUM_CSV.with_suffix('.meta.json').read_text())
    prof = Profile(tum, tum_meta['lap_time_s'])
    setup = setups.get(SETUP)
    if args.lsd_lock is not None:
        setup = setup.replace(lsd_lock_drive=args.lsd_lock)
    spec_tum = cars.TUM_F1
    overrides = {P + f'z.tire_model.{w}.{k}': float(v) for k, v in args.tyre_scale
                 for w in ('front_left', 'front_right', 'rear_left', 'rear_right')} or None
    probe = OCDBackend(powertrain='engine', car=CAR, max_brake_torque_Nm=BRAKE_TORQUE_NM, overrides=overrides)
    probe.reset(setup, StartPose(speed=10.))
    params = probe.run_metadata()['parameters']
    r_scale = params[P + 'tire.rolling_radius_m.velocity_scaling.factors']
    ctx = dict(overrides=overrides, lf=params[P + 'cog.distance_from_front_axle_m'], track=track, ttrack=ttrack, geo=geo, prof=prof, setup=setup, spec_tum=spec_tum,
               r0=params[P + 'tire.rolling_radius_m.rear'], r_scale=r_scale)
    if not args.no_checks:
        tyres = tyre_check(params, spec_tum, P)
        (out / 'tyre_check.json').write_text(json.dumps(common._clean(tyres), indent=1) + '\n')
        lat = lateral_check(setup, overrides=overrides)
        (out / 'lateral_check.json').write_text(json.dumps(common._clean(lat), indent=1) + '\n')
        from ocd_lap import qss_lap
        w0 = time.perf_counter()
        ggv = qss_lap.measure_ggv(setup, car=CAR, brake_torque_Nm=BRAKE_TORQUE_NM, overrides=overrides)
        q = qss_lap.lap_time(setup, track, car=CAR, ers=False, drs=False, path=track.reference, ggv_data=ggv,
                             overrides=overrides)
        q = {k: v for k, v in q.items() if k not in ('v_profile', 's_m')}
        q.update(wall_s=time.perf_counter() - w0, note='OCD QSS (qss_lap) of the debug car on TUM raceline, '
                 'g-g-v measured on the debug car, brake ramp 15 kN m, exponent 2, ERS/DRS off')
        (out / 'qss_debug_car.json').write_text(json.dumps(common._clean(q), indent=1) + '\n')
        print('QSS debug car on TUM raceline:', q['lap_time_s'], flush=True)
    s_grid = np.arange(0., geo.raceline.length, 1.)
    fp = cars.get(CAR).footprint
    ceiling = dict(footprint=float(np.mean([track.contains_car(*geo.raceline.point(si), geo.raceline.heading(si), fp) for si in s_grid])),
                   cog=float(np.mean([ttrack.contains_cog(*geo.raceline.point(si)) for si in s_grid])))
    print('raceline inside the track (per metre):', ceiling, flush=True)
    prov = {**run_provenance(track.name), 'ocd_first_lap_debug_worktree': {'path': str(ocd_root), **git_state(ocd_root)}}
    runs = []
    cols = [c[0] for c in common.PP_COLUMNS] + EXTRA
    for f in args.f:
        w0 = time.perf_counter()
        rows, summary = run(f, ctx, gains)
        summary['wall_s'] = time.perf_counter() - w0
        runs.append(summary)
        print(json.dumps(common._clean(summary)), flush=True)
        path = out / f'cl_f{f:.2f}.csv'
        with path.open('w') as fh:
            fh.write(','.join(cols) + '\n')
            for r in rows:
                fh.write(','.join(common._fmt(r.get(c)) for c in cols) + '\n')
        common.write_meta(path, {
            'source': 'tum_closed_loop_debug_ocd', 'summary': summary, 'lap_time_s': summary['lap2_s'],
            'speed_scale_f': f, 'tum_reference': {'csv': str(TUM_CSV), 'sha256': common.sha256_file(TUM_CSV),
                                                  'lap_time_s': tum_meta['lap_time_s'], 'variant': tum_meta['variant']},
            'conditions': {'car': CAR, 'setup_name': SETUP if args.lsd_lock is None else f'{SETUP} with lsd_lock_drive={args.lsd_lock}', 'setup': asdict(setup), 'max_brake_torque_Nm': BRAKE_TORQUE_NM,
                           'ers': False, 'drs': False, 'recuperation': False, 'post_setup_params': cars.get(CAR).post_setup,
                           'diagnostic_ocd_overrides': overrides,
                           'track': "Track('Shanghai', tum_mode=True): raceline path, CoG off-track (logged)",
                           'on_track': 'footprint 4.6 x 2.0 m at every 2 ms step (logged only)',
                           'on_track_cog': 'CoG inside the track at every 2 ms step (TUM mode check, logged)',
                           'run_in_m': RUN_IN_M, 'laps': LAPS, 'control_dt_s': .02, 'physics_dt_s': .002,
                           'initial_gear': 'TUM find_gear at the start speed (initial condition only)'},
            'controller': {'gains_PLACEHOLDER': asdict(gains), 'law': __doc__.split('Controller,')[1].split('Speed scale')[0].strip()},
            'raceline_inside_fraction_per_metre': ceiling,
            'columns_extra': EXTRA, 'columns_base': 'common.PP_COLUMNS (target_speed_mps = f v_TUM(s + v tau_lon))',
            'code': {**prov, 'script_sha256': common.sha256_file(__file__)},
            'python': sys.version.split()[0], 'command': ' '.join(sys.argv)})
        if summary['completed_lap2'] and not args.all:
            break
    analysis = {f'{r["f"]:.2f}': analyse(out / f'cl_f{r["f"]:.2f}.csv', prof) for r in runs}
    (out / 'analysis.json').write_text(json.dumps(common._clean(analysis), indent=1) + '\n')
    (out / 'runs.json').write_text(json.dumps(common._clean({'runs': runs, 'raceline_inside_fraction_per_metre': ceiling,
                                                              'gains': asdict(gains), 'setup': asdict(setup),
                                                              'command': ' '.join(sys.argv)}), indent=1) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
