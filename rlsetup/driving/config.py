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
CORNER_BONUS_MODES = ('exit', 'entry_exit')
R4_REWARD_KEYS = ('speed_k', 'speed_sigma')                       # omitted from to_dict while speed_k = 0
R4_CORNER_KEYS = ('mode', 'bonus_in', 'bonus_out', 'sigma')        # omitted while mode = 'exit'


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

    R4 speed-profile potential (docs/specs/R4_review.md, DECISIONS #26; time_pbrs only, off with
    speed_k = 0): Phi_v(s, v) = speed_k g(v, v_qss(s)), g = max(0, 1 - ((v - v*) / (speed_sigma v*))^2)
    (speed_profile.closeness), v = |(v_x, v_y)| [m/s], v_qss = corner table speed on Track.path arc
    length s; added per decision as gamma Phi_v(s') - Phi_v(s) with the same terminal convention as
    Phi (0 after off track / non-finite / wrong way). speed_k = 0: not computed at all (bit-identical
    rewards); EnvConfig.to_dict then omits both fields.
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
    speed_k: float = 0.                  # R4: k_v of Phi_v [reward units]; 0 = off. f1_r4: 20 [ARCH]
    speed_sigma: float = .3              # R4: sigma of g (relative to v_qss) [ARCH]

    def __post_init__(self):
        if self.mode not in REWARD_MODES:
            raise ValueError(f'reward mode {self.mode!r} not in {REWARD_MODES}')
        if not all(math.isfinite(x) and x >= 0 for x in (self.shaping_k, self.penalty_const, self.speed_k)):
            raise ValueError('shaping_k, penalty_const and speed_k must be finite and nonnegative')
        if not (math.isfinite(self.speed_sigma) and self.speed_sigma > 0):
            raise ValueError(f'speed_sigma must be finite and positive, got {self.speed_sigma}')
        if self.speed_k > 0 and self.mode != 'time_pbrs':
            raise ValueError("speed_k > 0 (R4 speed-profile potential) needs mode 'time_pbrs'")
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
                                         # in action units and last_action[0] - steer = PT1 lag (A5).
                                         # Kept 0.6 for cars with another lock (sf70h_f1 0.30 rad:
                                         # actual steer / 0.6 in [-0.5, 0.5] = action / 2): the same
                                         # rad -> input scale for every car (comparable observations,
                                         # obs layout frozen); the lag is then last_action[0] / 2 - steer
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
    traction_slip_start: float = .08     # = ScriptedDriver (AV-21 tyre: Fx peak at kappa 0.14-0.27)
    traction_slip_span: float = .12      # = ScriptedDriver; throttle fully cut at start + span = 0.20.
                                         # Preset f1_sha (sf70h_f1, Fx peak kappa 0.075-0.081): 0.06 / 0.03


@dataclass(frozen=True)
class CornerBonusConfig:
    """R2 clean-corner exit bonus (docs/specs/R2_review.md §R2-2, R2-3; DECISIONS #22-23). A
    TRAINING AID: the env's reward stays time_pbrs bit for bit; the env only reports
    info['corner_bonus'] (unweighted, per decision) and the replay buffer adds w(t) x bonus at
    sampling time (buffers.py). bonus = B f(rho), f = clip((rho - floor_ratio) / (1 - floor_ratio),
    0, 1), rho = exit speed / v_qss at the exit. The detection fields (prominence .. grid_m) define
    the corner table (corners.py, cache key).
    R4 (`mode` 'entry_exit', docs/specs/R4_review.md req 3-5): instead of the exit bonus, per corner
    unit bonus_in g(v_entry) paid at the entry crossing (entry = window start = QSS braking point)
    and bonus_out g(v_exit) at the exit, g = speed_profile.closeness with `sigma`; an entry paid
    in this pass is taken back (negative info['corner_bonus']) when a terminal failure comes
    before that unit's exit (speed_profile.EntryExitTracker). bonus / floor_ratio are unused there.
    The R4 fields are left out of to_dict while mode is 'exit' and they are at their defaults."""
    enabled: bool = False
    bonus: float = 3.                    # B per corner [reward units] (Amendment 1 F4)
    floor_ratio: float = .75             # f = 0 at or below this exit-speed ratio [J]
    prominence_mps: float = 5.           # apex = local minimum of v_qss with at least this prominence [J]
    merge_m: float = 150.                # apexes closer than this form one unit [J]
    exit_after_apex_m: float = 150.      # exit point = last apex + this ...
    exit_before_brake_m: float = 30.     # ... but at least this before the next unit's braking point [J]
    grid_m: float = 1.                   # Track.path grid of the mapped v_qss
    mode: str = 'exit'                   # 'exit' = R2/R3 exit bonus | 'entry_exit' = R4 entry + exit closeness
    bonus_in: float = 10.                # R4 B_in per unit [reward units] [ARCH]
    bonus_out: float = 10.               # R4 B_out per unit [reward units] [ARCH]
    sigma: float = .3                    # R4 sigma of g at entries / exits [ARCH]

    def __post_init__(self):
        if self.mode not in CORNER_BONUS_MODES:
            raise ValueError(f'corner_bonus mode {self.mode!r} not in {CORNER_BONUS_MODES}')
        if not all(math.isfinite(x) and x >= 0 for x in (self.bonus_in, self.bonus_out)):
            raise ValueError('bonus_in and bonus_out must be finite and nonnegative')
        if not (math.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError(f'corner_bonus sigma must be finite and positive, got {self.sigma}')


@dataclass(frozen=True)
class FocusConfig:
    """R2 focused starts (§R2-7): with probability `fraction` a training reset (not evaluation,
    no explicit start option) starts lead_m before the window start of a corner drawn from the
    focus weights (FocusedStartCallback), settled at f ~ U(speed_fraction) of v_qss. A TRAINING
    AID (start distribution only; the reward is unchanged)."""
    enabled: bool = False
    fraction: float = .25                # share of training resets that are focused [P: TAV18, FLOR17]
    bin_m: float = 50.                   # Track.path bins of the focus weights
    lead_m: tuple = (50., 300.)          # hand-over target d ~ U(lo, hi) before the target [m]
    speed_fraction: tuple = (.5, .8)     # settle speed fraction f ~ U(lo, hi)
    max_attempts: int = 5                # focused settle attempts before a uniform start


TRAINING_AID_KEYS = ('corner_bonus', 'focus')   # EnvConfig blocks that are not part of the driver definition


@dataclass(frozen=True)
class EnvConfig:
    track: str = 'Shanghai'
    track_vmax_mps: float = 12.          # Track(name, vmax): only its scripted speed profile; unused here
    max_brake_torque_Nm: float = 12000.  # total, DECISIONS #5. PLACEHOLDER until SF70H data
    car: str = 'av21'                    # ocd_lap.cars name: 'av21' (OCD's shipped car), 'sf70h_tum' or
                                         # 'sf70h_f1' (ocd_first_lap branch sf70h-f1 only)
    ers: bool = False                    # the car's ERS. Off: OCDBackend's 4 MJ is per lap, and DrivingEnv's
                                         # multi-lap, random-start episodes neither refill nor observe it
    setup: str = 'p0'                    # car setup by name, rlsetup.driving.setups: 'p0' (Setup()), 'p1_TUM',
                                         # 'bcn_best', 'p1_TUM_f1', 'sha_f1_best' (sf70h_f1 only)
    reference: str = 'path'              # xi_ref: 'path' = Track.path (safe line), 'raceline' = Track.reference
                                         # (racetrack-database raceline = TUM's) or 'csv:<repo-relative path>'
                                         # (x, y CSV, ReferenceLine.from_csv). Track progress, the lap
                                         # line and the QSS profile stay on Track.path either way
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
    corner_bonus: CornerBonusConfig = field(default_factory=CornerBonusConfig)   # R2 training aid
    focus: FocusConfig = field(default_factory=FocusConfig)                      # R2 training aid

    @property
    def speed_on(self):
        """R4 speed-profile potential on (reward.speed_k > 0)."""
        return self.reward.speed_k > 0

    @property
    def entry_exit_on(self):
        """R4 corner entry + exit closeness terms on."""
        return self.corner_bonus.enabled and self.corner_bonus.mode == 'entry_exit'

    @property
    def r4(self):
        return self.speed_on or self.entry_exit_on

    def to_dict(self):
        """asdict without the TRAINING_AID_KEYS blocks that equal their default (disabled), so
        configs, demo keys and hashes written before R2 do not change (R2-6); a missing block
        loads as its default. R4: reward.speed_k / speed_sigma are left out while speed_k = 0 and
        speed_sigma is at its default, the corner_bonus R4 fields while mode is 'exit' and they
        are at their defaults (configs written before R4 round-trip byte-equal)."""
        d = asdict(self)
        for k in TRAINING_AID_KEYS:
            if getattr(self, k) == type(getattr(self, k))():
                del d[k]
        rd = RewardConfig()
        if self.reward.speed_k == 0 and self.reward.speed_sigma == rd.speed_sigma:
            for k in R4_REWARD_KEYS:
                del d['reward'][k]
        if 'corner_bonus' in d and all(getattr(self.corner_bonus, k) == getattr(CornerBonusConfig(), k)
                                       for k in R4_CORNER_KEYS):
            for k in R4_CORNER_KEYS:
                del d['corner_bonus'][k]
        return d
