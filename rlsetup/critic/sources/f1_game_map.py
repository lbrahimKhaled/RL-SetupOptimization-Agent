"""F1 25 / F1 26 game setup -> the 15 OCD knobs (S4, review R1), and the box check of S2 step 4.

Every click constant is FITTED TO THE OCD BOX, NOT MEASURED: the games publish no physical values.
Selection rule (R1, to be stated in every result): one map per click type for both axles; the p1-p99
support covers sha_f1_best's chi 0.40 and K_phi 4e5 and the OCD default chi 0.613; bars carry about
half of K_phi (median 0.50); <= 5 % of rows dropped per game.
Toe: label-supported; per-wheel vs total unverified. The F1 25 menu labels are "Front Toe-Out" and
"Rear Toe-In" (docs/research/c2_game_setup_units.md section 4), which is the C1 sign convention kept here
(OCD toe positive = toe-out: front +value, rear -value). Ride height: 1 click = 1 mm is ASSUMED (no
source gives a unit; same research note section 1).
The same menu ranges hold in both games (S0): wings 0-50, suspension 1-41, anti-roll bars 1-21, ride
height F 15-35 / R 40-60, diff 10-100 %, brake bias 50-70 %.
"""
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from ..config import CarConfig, DataConfig, KNOB_ORDER
from ..knobs import knob_bounds, out_of_bounds, roll_from_bars

# The 15 game parameters the map reads (brake pressure and the four tyre pressures have no OCD knob).
GAME_COLUMNS = ('front_wing', 'rear_wing', 'differential_on_throttle_pct', 'differential_off_throttle_pct',
                'front_camber_deg', 'rear_camber_deg', 'front_toe_deg', 'rear_toe_deg', 'front_suspension',
                'rear_suspension', 'front_anti_roll_bar', 'rear_anti_roll_bar', 'front_ride_height',
                'rear_ride_height', 'front_brake_bias_pct')


@dataclass(frozen=True)
class F1GameMapConfig:
    """Review R1 constants. All of them: fitted to the OCD box, not measured."""
    freq_base_hz: float = 2.5               # fitted to the OCD box, not measured: f = base + span (c - 1) / (c_max - 1), both axles
    freq_span_hz: float = 2.5               # fitted to the OCD box, not measured
    suspension_clicks: int = 41             # menu range 1..41 (S0)
    bar_base_Nmprad: float = 10_000.        # fitted to the OCD box, not measured: K_b = base + step (c - 1), both axles
    bar_step_Nmprad: float = 12_500.        # fitted to the OCD box, not measured
    height_offset_mm: float = 5.            # fitted to the OCD box, not measured: h = click + offset; 1 click = 1 mm ASSUMED (no source)
    wing_max: float = 50.                   # menu range 0..50 (S0)
    aero_share_low: float = .35             # fitted to the OCD box, not measured: share at the lowest front-wing fraction
    aero_share_span: float = .25            # fitted to the OCD box, not measured
    # toe signs: label-supported; per-wheel vs total unverified. OCD positive = toe-out; the F1 25 menu
    # labels the values "Front Toe-Out" and "Rear Toe-In" (c2_game_setup_units.md section 4), so the
    # front value enters as +toe-out and the rear value as -toe-out (= the C1 convention). Whether the
    # game value is per wheel or the axle total, and whether F1 26 keeps the labels, is unverified.
    toe_front_sign: float = 1.
    toe_rear_sign: float = -1.


# Game track name -> geometry file stem (TUM racetrack-database name where TUM has the circuit; the
# same stem is used for the converted bacinger centre lines in data/tracks_extra/). S3 "track name map".
TRACK_GEOMETRY = {'abudhabi': 'YasMarina', 'australia': 'Melbourne', 'austria': 'Spielberg', 'azerbaijan': 'Baku',
                  'bahrain': 'Sakhir', 'brazil': 'SaoPaulo', 'canada': 'Montreal', 'china': 'Shanghai',
                  'hungary': 'Budapest', 'imola': 'Imola', 'japan': 'Suzuka', 'las_vegas': 'LasVegas',
                  'madrid': 'Madrid', 'mexico': 'MexicoCity', 'miami': 'Miami', 'monaco': 'Monaco', 'monza': 'Monza',
                  'netherlands': 'Zandvoort', 'qatar': 'Lusail', 'saudi_arabia': 'Jeddah', 'silverstone': 'Silverstone',
                  'singapore': 'Singapore', 'spa': 'Spa', 'spain': 'Catalunya', 'usa': 'Austin'}
# Geometry stem -> bacinger/f1-circuits id: the 9 circuits TUM lacks + the 5 TUM layouts that fail R2.
BACINGER_IDS = {'Baku': 'az-2016', 'Imola': 'it-1953', 'LasVegas': 'us-2023', 'Madrid': 'es-2026',
                'Miami': 'us-2022', 'Monaco': 'mc-1929', 'Lusail': 'qa-2004', 'Jeddah': 'sa-2021',
                'Singapore': 'sg-2008', 'YasMarina': 'ae-2009', 'MexicoCity': 'mx-1962', 'Zandvoort': 'nl-1948',
                'Melbourne': 'au-1953', 'Catalunya': 'es-1991'}


def game_to_knobs(game: Mapping[str, np.ndarray], cfg: F1GameMapConfig = F1GameMapConfig(),
                  car: CarConfig = CarConfig()) -> np.ndarray:
    """(N, 15) physical OCD knobs in KNOB_ORDER from game values (mapping column -> array).

    R1: f = 2.5 + 2.5 (c - 1)/40 Hz; K_b = 10,000 + 12,500 (c - 1) N m/rad; h = c + 5 mm (rake =
    c_R - c_F mm); aero_level = (fw + rw)/100; aero_front_share = 0.35 + 0.25 (s - 1/52)/(50/52) with
    s = (fw + 1)/(fw + rw + 2); brake bias and both LSD locks pct/100; camber direct; toe front +, rear - (label-supported).
    K_phi and chi follow from the frequency mediator (`knobs.roll_from_bars`), so K_b >= 0."""
    g = {k: np.asarray(game[k], dtype=float) for k in GAME_COLUMNS}
    f_f = cfg.freq_base_hz + cfg.freq_span_hz * (g['front_suspension'] - 1) / (cfg.suspension_clicks - 1)
    f_r = cfg.freq_base_hz + cfg.freq_span_hz * (g['rear_suspension'] - 1) / (cfg.suspension_clicks - 1)
    k_b_f = cfg.bar_base_Nmprad + cfg.bar_step_Nmprad * (g['front_anti_roll_bar'] - 1)
    k_b_r = cfg.bar_base_Nmprad + cfg.bar_step_Nmprad * (g['rear_anti_roll_bar'] - 1)
    k_phi, chi = roll_from_bars(f_f, f_r, k_b_f, k_b_r, car)
    h_f_mm = g['front_ride_height'] + cfg.height_offset_mm
    h_r_mm = g['rear_ride_height'] + cfg.height_offset_mm
    fw, rw = g['front_wing'], g['rear_wing']
    s = (fw + 1) / (fw + rw + 2)
    s_min, s_span = 1 / (cfg.wing_max + 2), cfg.wing_max / (cfg.wing_max + 2)
    values = dict(ride_freq_front_hz=f_f, ride_freq_rear_hz=f_r, roll_stiffness_Nm_per_rad=k_phi,
                  roll_stiffness_front_share=chi, ride_height_front_m=h_f_mm / 1000.,
                  rake_m=(h_r_mm - h_f_mm) / 1000., aero_level=(fw + rw) / (2 * cfg.wing_max),
                  aero_front_share=cfg.aero_share_low + cfg.aero_share_span * (s - s_min) / s_span,
                  brake_bias_front=g['front_brake_bias_pct'] / 100., camber_front_deg=g['front_camber_deg'],
                  camber_rear_deg=g['rear_camber_deg'], toe_front_deg=cfg.toe_front_sign * g['front_toe_deg'],
                  toe_rear_deg=cfg.toe_rear_sign * g['rear_toe_deg'],
                  lsd_lock_drive=g['differential_on_throttle_pct'] / 100.,
                  lsd_lock_coast=g['differential_off_throttle_pct'] / 100.)
    return np.column_stack([values[k] for k in KNOB_ORDER])


def box_check(knobs: np.ndarray, groups: np.ndarray, cfg: DataConfig = DataConfig(), log=print):
    """S2 step 4: (keep mask, report). Rows outside the OCD box by more than `bounds_tol` are dropped;
    raises ValueError when more than `max_bounds_drop_frac` of any group (game) goes (stop rule, R1).
    report = {'dropped', 'total', 'per_knob': {knob: rows}, 'per_group': {group: (dropped, total)}}."""
    oob = out_of_bounds(knobs, cfg.bounds_tol)
    keep = ~oob.any(axis=1)
    per_knob = {k: int(oob[:, i].sum()) for i, k in enumerate(KNOB_ORDER) if oob[:, i].any()}
    per_group = {str(g): (int((~keep[groups == g]).sum()), int((groups == g).sum())) for g in np.unique(groups)}
    report = {'dropped': int((~keep).sum()), 'total': int(len(keep)), 'per_knob': per_knob, 'per_group': per_group}
    log(f"box check: {report['dropped']} of {report['total']} rows outside the OCD box "
        f"({100 * report['dropped'] / max(report['total'], 1):.2f} %), per knob {per_knob}")
    for g, (d, n) in per_group.items():
        log(f'  {g}: {d} of {n} ({100 * d / max(n, 1):.2f} %)')
    bad = {g: d / n for g, (d, n) in per_group.items() if n and d / n > cfg.max_bounds_drop_frac}
    if bad:
        raise ValueError(f'box drop exceeds the {100 * cfg.max_bounds_drop_frac:g} % per-source stop rule: '
                         f'{ {g: round(100 * f, 2) for g, f in bad.items()} } % (per knob: {per_knob})')
    return keep, report


def clip_to_box(knobs: np.ndarray) -> np.ndarray:
    """Kept rows clipped onto the box (removes the <= tol rounding excess, e.g. rake = -7e-18)."""
    low, high = knob_bounds()
    return np.clip(knobs, low, high)
