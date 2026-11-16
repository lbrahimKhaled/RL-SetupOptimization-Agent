"""The 15 OCD knobs of the critic: bounds, [-1, 1] normalisation, frequency mediator, feasibility.

Source-independent (S4, S9): every data source maps its own parameters to these physical knobs with
its adapter (`sources/`). Sign conventions are OCD's (`ocd_lap/car_setup.py`): camber negative = top
inward; toe positive = toe-out; roll stiffness and share as in `translator.suspension_rates`. SI units.
"""
import hashlib
import numpy as np
from ocd_lap.car_setup import KNOB
from .config import CarConfig, KNOB_ORDER


def knob_bounds():
    """(low, high) physical bounds [knob units] in KNOB_ORDER, from `ocd_lap.car_setup.KNOB`."""
    return np.array([KNOB[k].low for k in KNOB_ORDER]), np.array([KNOB[k].high for k in KNOB_ORDER])


def knob_index(name) -> int:
    return KNOB_ORDER.index(name)


def wheel_rate(freq_hz, corner_mass_kg):
    """K_w = m_c (2 pi f)^2 [N/m] (PROJECT_CONTEXT 5)."""
    return corner_mass_kg * (2 * np.pi * freq_hz) ** 2


def roll_from_bars(f_front_hz, f_rear_hz, k_bar_front, k_bar_rear, car: CarConfig = CarConfig()):
    """Inverse frequency mediator (C1 Decision 8): (K_phi [N m/rad], chi) from ride frequencies [Hz]
    and bar roll rates [N m/rad]. Axle roll rate = K_b + 1/2 K_w t^2; K_phi = front + rear;
    chi = front / K_phi. This inverts K_b = chi K_phi - 1/2 K_w t^2, so K_b >= 0 whenever the bar
    rates passed in are >= 0."""
    front = k_bar_front + .5 * wheel_rate(f_front_hz, car.corner_sprung_mass_front_kg) * car.track_front_m ** 2
    rear = k_bar_rear + .5 * wheel_rate(f_rear_hz, car.corner_sprung_mass_rear_kg) * car.track_rear_m ** 2
    k_phi = front + rear
    return k_phi, front / k_phi


def out_of_bounds(knobs: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """(N, 15) bool: True where a knob is outside the OCD bounds by more than `tol` (S2 step 4)."""
    low, high = knob_bounds()
    return ~np.isfinite(knobs) | (knobs < low - tol) | (knobs > high + tol)


def normalise(knobs):
    """Physical knobs -> [-1, 1] with the KNOB bounds (numpy array or tensor, any shape (..., 15))."""
    low, high = _bounds_like(knobs)
    return 2 * (knobs - low) / (high - low) - 1


def denormalise(p):
    """Inverse of `normalise`."""
    low, high = _bounds_like(p)
    return low + (p + 1) * (high - low) / 2


def _bounds_like(x):
    low, high = knob_bounds()
    if isinstance(x, np.ndarray):
        return low, high
    return x.new_tensor(low), x.new_tensor(high)


def bar_roll_rates(p, car: CarConfig = CarConfig()):
    """(K_b_front, K_b_rear) [N m/rad] from normalised p (..., 15), differentiable when p is a tensor
    (C1-R2). K_b,a = chi_a K_phi - 1/2 K_w,a t_a^2, as `ocd_lap.translator.suspension_rates`: the input
    box contains points with K_b < 0, which the translator clamps to 0, so the later actor must keep
    these non-negative itself (C1 Decision 8)."""
    x = denormalise(p)
    f_f, f_r = x[..., knob_index('ride_freq_front_hz')], x[..., knob_index('ride_freq_rear_hz')]
    k_phi = x[..., knob_index('roll_stiffness_Nm_per_rad')]
    chi = x[..., knob_index('roll_stiffness_front_share')]
    k_w_f = wheel_rate(f_f, car.corner_sprung_mass_front_kg)
    k_w_r = wheel_rate(f_r, car.corner_sprung_mass_rear_kg)
    return chi * k_phi - .5 * k_w_f * car.track_front_m ** 2, (1 - chi) * k_phi - .5 * k_w_r * car.track_rear_m ** 2


def knob_feasibility(p, car: CarConfig = CarConfig()):
    """min(K_b_front, K_b_rear) [N m/rad]; >= 0 means the setup is realisable with real bars (S9)."""
    kb_f, kb_r = bar_roll_rates(p, car)
    return np.minimum(kb_f, kb_r) if isinstance(kb_f, np.ndarray) else kb_f.minimum(kb_r)


def support_box(p_train: np.ndarray):
    """(low, high) per-knob [min, max] of normalised training rows (C1-R2, C1-R9)."""
    return p_train.min(axis=0), p_train.max(axis=0)


def setup_hash(knobs: np.ndarray) -> np.ndarray:
    """C1-R3: hex digest per row of the 15 mapped knobs rounded to 1e-8 (track excluded); object array."""
    rounded = np.round(np.asarray(knobs, dtype=float), 8) + 0.0  # + 0.0 removes -0.0
    return np.array([hashlib.sha256(r.tobytes()).hexdigest()[:20] for r in rounded], dtype=object)
