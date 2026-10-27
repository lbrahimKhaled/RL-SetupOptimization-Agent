"""All numbers of the driving environment (D1). Nothing numeric lives in env.py.

Spec: docs/specs/D1_driving_env.md as amended by docs/specs/D1_review.md (the review wins).
Units SI; angles in rad; lateral offset e positive = left of the reference line (ocd_first_lap
convention). PLACEHOLDER marks numbers that are not yet justified by data or a decision.

Together, `EnvConfig.action_repeat`, `RewardConfig` (a, b, W, c, P0, kappa, T_rec) and
`RewardConfig.gamma` are the frozen "driver definition" (review R1): changing any of them
changes which driver is being trained.
"""
from dataclasses import dataclass, field, asdict
import math


REWARD_MODES = ('ref_tracking', 'time_pbrs')


@dataclass(frozen=True)
class RewardConfig:
    """Two reward modes (`mode`; DECISIONS #13, #19).

    'ref_tracking' = reward (1) (default; the full1 driver), per 50 Hz base tick:
        r = a ds - b dt (1 + du^T W du) - c e^2.
    du = action rate [action units / s]. The c e^2 term is charged per base tick (not times dt),
    as in `ocd_lap.env.OCDRacingEnv`. With action repeat the per-decision reward is the sum of
    the base-tick rewards.
    Failure (off track, non-finite state, stall, wrong way) ends the episode with
        P = P0 + kappa * (c / dt_tick) * (T_rec / 3) * e_exit^2      (review R6.3)
    where dt_tick = 0.02 s is the base tick of ocd_first_lap (CONTROL_STEPS * DT, not a free
    constant) and e_exit the lateral offset to the reference at the failing physics step.
    Defaults give P = 50 + 30 c e^2. No speed scaling (R6.3).

    'time_pbrs' = R1 (docs/specs/R1_time_optimal.md §1): per base tick r = a ds - b dt -
    [terminal failure] penalty_const, plus per decision the potential-based shaping
    F = gamma Phi(s') - Phi(s), Phi(s) = -shaping_k |e(s)| (Ng et al. 1999). `gamma` is also the
    shaping discount (it must be the learner's per-decision discount, else F is not potential-
    based). c and w are diagnostic only (smoothness / tracking still logged, not rewarded).
    Terminal failures = off track, non-finite, wrong way (Phi(s') = 0); a stall is a truncation
    (no penalty, real Phi(s')). `failure_penalty()` is the ref_tracking formula only.
    """
    a: float = 1.0                       # per metre of progress along Track.path. PLACEHOLDER
    b: float = 1.0                       # per second. PLACEHOLDER
    w: tuple = (.01, .001, .001)         # diag(W): steering, throttle, brake rates [s^2]. PLACEHOLDER
    c: float = .1                        # per m^2 of lateral offset to xi_ref. PLACEHOLDER (OPEN §11.8)
    penalty_base: float = 50.            # P0. PLACEHOLDER (R6.3: slow-speed living cost + slow recovery)
    penalty_kappa: float = 1.5           # kappa, safety factor on the recovery cost. PLACEHOLDER
    recovery_time_s: float = 1.2         # T_rec, linear recovery time from e_exit. PLACEHOLDER
    gamma: float = .992                  # agent discount per 25 Hz decision (used in D2, R10); in
                                         # time_pbrs also the shaping discount. PLACEHOLDER
    mode: str = 'ref_tracking'           # 'ref_tracking' = reward (1), full1 | 'time_pbrs' = R1
    shaping_k: float = 1.0               # [1/m] Phi(s) = -shaping_k |e(s)|; time_pbrs only. PLACEHOLDER
    penalty_const: float = 100.          # P for terminal failures in time_pbrs. PLACEHOLDER

    def __post_init__(self):
        if self.mode not in REWARD_MODES:
            raise ValueError(f'reward mode {self.mode!r} not in {REWARD_MODES}')
        if not all(math.isfinite(x) and x >= 0 for x in (self.shaping_k, self.penalty_const)):
            raise ValueError('shaping_k and penalty_const must be finite and nonnegative')
        if not 0. < self.gamma <= 1.:
            raise ValueError(f'gamma must be in (0, 1], got {self.gamma}')

    def failure_penalty(self, e_exit_m, dt_tick_s):
        """P [reward units] for a failure with lateral offset e_exit_m [m] to the reference."""
        return (self.penalty_base + self.penalty_kappa * (self.c / dt_tick_s)
                * (self.recovery_time_s / 3.) * e_exit_m ** 2)


@dataclass(frozen=True)
class ObsConfig:
    """Observation layout and fixed scales (review R4; no running normalisation, so a fine-tune
    on another setup sees the same input scaling). Scales per D1_signoff A5. The layout and
    scales are FROZEN from D2 on: changing them invalidates every trained driver."""
    n_preview: int = 20                  # stations ahead along Track.path arc length
    preview_horizon_m: float = 300.      # d_k = horizon * (k / N)^exponent, k = 1..N
    preview_exponent: float = 1.5
    preview_min_scale_m: float = 10.     # station k is divided by max(d_k, this)
    grid_m: float = 1.                   # precomputation grid on Track.path arc length
    history: int = 3                     # ticks (decisions) in the ego + action stack, incl. now
    speed_scale_mps: float = 80.         # speed and v_x
    vy_scale_mps: float = 5.             # lateral velocity v_y (D1_signoff A5; p99 |v_y| ~ 0.7 m/s)
    accel_scale_mps2: float = 30.
    yaw_rate_scale_radps: float = 1.
    steer_scale_rad: float = .6          # actual steer angle; = ocd_lap.env.MAX_STEER_RAD, so it is
                                         # in action units and last_action[0] - steer = PT1 lag (A5)
    heading_scale_rad: float = .5        # heading error to xi_ref; wrong way (pi/2) -> 3.14 (A5)
    slip_gain: float = 5.                # rear slip ratio * gain, then clipped to +-slip_clip
    slip_clip: float = 5.
    e_scale_m: float = 5.
    gear_scale: float = 8.               # gear / 8 when EnvConfig.obs_gear
    clip: float = 10.                    # final clip; also the Box bounds


@dataclass(frozen=True)
class DemoConfig:
    """Demo / settle controller (review R5): pure pursuit on xi_ref with `ScriptedDriver`
    geometry, speed PI on the pedals with the gains of `ocd_lap.manoeuvres._speed_pedals`."""
    speed_kp: float = .4                 # [pedal per m/s]   = manoeuvres._speed_pedals
    speed_ki: float = .2                 # [pedal per m]     = manoeuvres._speed_pedals
    integral_limit_m: float = 5.         # = manoeuvres.constant_radius / qss_lap settle loops
    brake_cap: float = .3                # 0.3 * 12 kN m = 3.6 kN m < 3.65 kN m no-aero lock limit
    lookahead_m: float = 3.              # pure pursuit lookahead = lookahead_m + lookahead_s * vx
    lookahead_s: float = .35             # (ScriptedDriver values)
    speed_preview_s: float = .5          # lapping target = f * v_qss(s + v * preview). PLACEHOLDER
    released_pedal_action: float = -.8   # action for a released pedal with the deadzone (R3)
    # Traction control (NOT in the review's R5; added because full throttle at low gear spins
    # the rears to slip 1.0 and the car off the track). ScriptedDriver's law on the rear slip
    # ratio: throttle *= max(0, 1 - (slip - start) / span). traction_control=False = R5 as written.
    traction_control: bool = True
    traction_slip_start: float = .08     # = ScriptedDriver
    traction_slip_span: float = .12      # = ScriptedDriver


@dataclass(frozen=True)
class EnvConfig:
    track: str = 'Shanghai'
    track_vmax_mps: float = 12.          # Track(name, vmax): only its scripted speed profile; unused here
    max_brake_torque_Nm: float = 12000.  # total, DECISIONS #5. PLACEHOLDER until SF70H data
    car: str = 'av21'                    # ocd_lap.cars name: 'av21' (OCD's shipped car) or 'sf70h_tum'
    ers: bool = False                    # the car's ERS. Off: OCDBackend's 4 MJ is per lap, and DrivingEnv's
                                         # multi-lap, random-start episodes neither refill nor observe it
    action_repeat: int = 2               # 2 x 50 Hz base ticks -> 25 Hz decisions (R2.3)
    pedal_deadzone: bool = True          # R3, pending Ibrahim: pedal = clip((a + .5) / 1.5, 0, 1)
    obs_gear: bool = False               # R4.3, pending Ibrahim: add gear / 8 to the observation
    max_episode_s: float = 60.           # training truncation after hand-over. PLACEHOLDER
    settle_s: float = 2.                 # settle phase before hand-over (R5); 0 = no settle
    start_speed_fraction: tuple = (.3, .7)   # f ~ U(lo, hi); v0 = f * min v_qss over the window
    settle_window_margin_m: float = 50.  # window = [s0, s0 + v_hat * settle_s + margin]
    max_start_attempts: int = 20         # resample the start if settle leaves the track
    stall_speed_mps: float = 3.          # R7: stall if speed < this ...
    stall_time_s: float = 2.             # ... for longer than this
    wrong_way_rad: float = math.pi / 2   # R7: |heading error to xi_ref| above this
    evaluation: bool = False             # flying-lap evaluation mode (also a reset option)
    eval_run_in_m: float = 200.          # evaluation start: this far before the line (s = 0)
    eval_speed_fraction: float = .5      # evaluation settle speed fraction f
    eval_laps: int = 2                   # timed laps; lap 2 is the flying lap that is reported
    eval_max_s: float = 900.             # evaluation time limit. PLACEHOLDER
    record_telemetry: bool = False       # per-wheel Fz etc. in info (costly signals() call, R8)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    def to_dict(self):
        return asdict(self)
