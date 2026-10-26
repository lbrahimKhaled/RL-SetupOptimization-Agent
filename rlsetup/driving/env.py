"""DrivingEnv (D1): the xi_ref-following driving environment on `ocd_lap.env.OCDRacingEnv`.

Spec docs/specs/D1_driving_env.md as amended by docs/specs/D1_review.md (R1-R9); reward mode
'time_pbrs' per docs/specs/R1_time_optimal.md.

Reuses OCDRacingEnv.step for physics (10 x 2 ms OCD steps per 50 Hz base tick), off-track /
non-finite / lap termination, truncation and reward (1)
    r = a ds - b dt (1 + du^T W du) - c e^2        per base tick, du in action units / s
and overrides only `_line_errors` (offsets to xi_ref instead of Track.path, cached per physics
step), `_observation`, `to_inputs` (pedal deadzone), `reset` and the observation space.
RewardConfig.mode selects the reward:
  'ref_tracking' (default)  reward (1) above and the failure rules below; bit-identical to the
               code that trained full1 (tests/test_r1.py golden hash).
  'time_pbrs'  R1: per base tick r = a ds - b dt - [terminal failure] P (P = penalty_const), and
               per decision the shaping F = gamma Phi(s') - Phi(s), Phi(s) = -shaping_k |e(s)|, e at
               the decision-boundary state (= obs[9] * e_scale_m), gamma = RewardConfig.gamma (the
               learner's discount). Phi(s') = 0 after a terminal failure (off track, non-finite,
               wrong way); real Phi(s') after a truncation (time limit, stall) or lap_complete.
               A stall is a truncation (truncation 'stall', termination None, no penalty).
               Smoothness and tracking are still computed and logged (step_cost, laps), not
               rewarded; the episode return includes the shaping, so it is not a lap score.

action       [steering, throttle, brake] in [-1, 1]^3, absolute; steering * 0.6 rad (+ = left);
             pedals per `EnvConfig.pedal_deadzone` (see demo_driver.action_to_inputs). Each action
             is held for `action_repeat` base ticks (default 2 -> 25 Hz decisions); the decision's
             reward is the sum of the base-tick rewards; it stops early if a tick ends the episode.
progress     Track.path arc length (driver-independent; the same for every xi_ref). e and the
             heading error are relative to xi_ref (`ReferenceLine`), e positive = left.
failure      off track (4.2 x 2.0 m footprint), non-finite state, stall (speed < 3 m/s for > 2 s)
             or wrong way (|heading error| > 90 deg): terminated, reward - P,
             P = P0 + kappa (c / 0.02 s) (T_rec / 3) e_exit^2 (RewardConfig.failure_penalty).
             e_exit for a non-finite state is the last finite offset. (time_pbrs: see above;
             FAILURES keeps the stats meaning incl. stall, TERMINAL_FAILURES are the terminal ones.)
truncation   max_episode_s after hand-over (training) / eval_max_s (evaluation): truncated=True.
             Both must be whole multiples of the decision period (checked in __init__).
             The base env's lap-completion check is put out of reach (progress starts at -1e7 m):
             in training finishing a lap is not an MDP terminal state for the driver; in
             evaluation the episode ends ('lap_complete', terminated) when this env has recorded
             all eval_laps + 1 line crossings (one distance counter decides both, QA MINOR 3).
reset        training: s0 ~ U[0, L) on Track.path, car placed on xi_ref abeam s0 with xi_ref
             heading at v0 = f * min v_qss over [s0, s0 + f v_qss(s0) settle_s + 50 m],
             f ~ U(0.3, 0.7); then `settle_s` (2 s) of the demo driver holding v0 (not part of the
             episode); resampled if the settle leaves the track. evaluation: s0 = eval_run_in_m
             before the line (s = 0), f = 0.5, then `eval_laps` timed laps; line crossings are
             interpolated between base ticks (as ocd_lap.simulation.drive); lap 2 is the flying lap.
             options: 'setup', 'evaluation', 'start_pose' (StartPose, explicit), 'start_s',
             'start_speed', 'settle_trace' (sum of the four Fz per settle decision).
info         s_track, s_ref, e_ref, heading_error, speed, progress_m (since hand-over), time_s,
             gear, termination ('off_track' | 'nonfinite' | 'stall' | 'wrong_way' |
             'lap_complete' | None), truncation ('time_limit' | 'stall' (time_pbrs) | None),
             step_cost / episode_cost (progress a ds, time b dt, smoothness b dt du^T W du,
             tracking c e^2, penalty P; the same components in both modes, only P differs),
             shaping (F of this decision, 0 in ref_tracking), phi (Phi(s')), episode_shaping,
             reward_mode, obs_clipped; evaluation: line_crossings_s, laps (per-lap time and cost
             components, cost = time + smoothness + tracking in both modes (the legacy lap cost),
             cost_r1 = time (the R1 lap cost); shaping and P never enter laps),
             track_position (-1 right ... +1 left edge) and track_width_m (finite states only);
             telemetry (backend.get_telemetry()) only with record_telemetry. On a non-finite
             state e_ref, heading_error and speed are NaN.
Observation (fixed scales, clipped to +-ObsConfig.clip; ObsConfig documents the scales):
    ego (9): speed, v_x, v_y, yaw rate, a_x, a_y (body frame: a_x = dv_x/dt - r v_y,
             a_y = dv_y/dt + r v_x over the last decision, r and v at the interval midpoint),
             actual steer angle, rear slip ratios RL, RR
    e, heading error, lateral track position (-1 right edge ... +1 left edge)
    preview: N stations at d_k = 300 (k/N)^1.5 m ahead on Track.path: xi_ref, left-edge and
             right-edge points in the car frame (x forward, y left), divided by max(d_k, 10 m)
    last action (3); [gear / 8 if obs_gear]; history: (ego, last action) of the previous
    history - 1 decisions, newest first; entries not yet filled by the settle phase hold
    (ego at placement, nothing pressed [0, -1, -1]).
"""
from collections import deque
import hashlib
import math
import gymnasium as gym
import numpy as np
from ocd_lap.backend import OCDBackend, StartPose, DT
from ocd_lap.car_setup import Setup
from ocd_lap.env import OCDRacingEnv, RewardWeights
from ocd_lap.simulation import CONTROL_STEPS
from .config import EnvConfig
from .demo_driver import DemoDriver, action_to_inputs
from .reference import ReferenceLine, track_grid
from . import qss_profile

EGO_SIZE = 9
NO_LAP_PROGRESS_M = -1e7       # training: base progress start, lap completion out of reach
FAILURES = ('off_track', 'nonfinite', 'stall', 'wrong_way')    # stats meaning: episode ended badly
TERMINAL_FAILURES = ('off_track', 'nonfinite', 'wrong_way')    # time_pbrs: terminal, P, Phi(s') = 0
COST_KEYS = ('progress', 'time', 'smoothness', 'tracking', 'penalty')


def body_accelerations(prev, cur, dt):
    """Body-frame (a_x, a_y) [m/s^2] from two (v_x, v_y, yaw rate) samples dt [s] apart.
    In the rotating car frame a_x = dv_x/dt - r v_y and a_y = dv_y/dt + r v_x; r, v_x, v_y are
    taken at the interval midpoint. Steady circle: a_y = v r."""
    vx = (prev[0] + cur[0]) / 2
    vy = (prev[1] + cur[1]) / 2
    r = (prev[2] + cur[2]) / 2
    return (cur[0] - prev[0]) / dt - r * vy, (cur[1] - prev[1]) / dt + r * vx


def _zero_costs():
    return dict.fromkeys(COST_KEYS, 0.)


class DrivingEnv(OCDRacingEnv):
    def __init__(self, config=EnvConfig(), *, setup=None, track=None, reference=None, qss=None):
        """`setup`: default car setup (None = p0 = `Setup()`); `track`: optional prebuilt
        `ocd_lap.track.Track` (build it once in the main process before vector-env workers
        start, R1; OCDRacingEnv.__init__ still loads the track by name, which only reads the
        safe-line cache once it is valid); `reference`: `ReferenceLine` (default Track.path);
        `qss`: `QSSProfile` (default: the cached p0 profile)."""
        cfg = self.cfg = config
        if not (isinstance(cfg.action_repeat, int) and cfg.action_repeat >= 1):
            raise ValueError('action_repeat must be a positive integer')
        lo, hi = cfg.start_speed_fraction
        if not (0. < lo <= hi) or cfg.settle_s < 0 or cfg.eval_laps < 1 or cfg.max_start_attempts < 1:
            raise ValueError('invalid start/settle/evaluation settings')
        rw = cfg.reward
        weights = RewardWeights(a=rw.a, b=rw.b, w=tuple(rw.w), c=rw.c, off_track_penalty=0.)
        if not all(math.isfinite(x) and x >= 0 for x in (rw.penalty_base, rw.penalty_kappa, rw.recovery_time_s)):
            raise ValueError('penalty constants must be finite and nonnegative')
        self.dt_tick = CONTROL_STEPS * DT
        self.decision_dt = cfg.action_repeat * self.dt_tick
        for name in ('max_episode_s', 'eval_max_s'):   # QA MINOR 2: no partial last decision
            n = getattr(cfg, name) / self.decision_dt
            if not (math.isfinite(n) and n >= 1 and abs(n - round(n)) <= 1e-9 * max(1., n)):
                raise ValueError(f'{name} must be a positive multiple of the decision period '
                                 f'{self.decision_dt:g} s')
        setup = Setup() if setup is None else setup   # p0 (DECISIONS #4); never OCD's shipped car
        super().__init__(cfg.track if track is None else track.name, setup,
                         vmax=cfg.track_vmax_mps if track is None else track.vmax,
                         max_time_s=cfg.max_episode_s, weights=weights,
                         backend_factory=lambda: OCDBackend(powertrain='engine',
                                                            max_brake_torque_Nm=cfg.max_brake_torque_Nm))
        if track is not None:
            self.track = track
        self.ref = reference if reference is not None else ReferenceLine.from_track_path(self.track)
        self.grid = track_grid(self.track, self.ref, cfg.obs.grid_m)
        self.qss = qss if qss is not None else qss_profile.load_or_build(self.track)
        self.settle_driver = DemoDriver(self.ref, cfg.demo, pedal_deadzone=cfg.pedal_deadzone,
                                        control_dt_s=self.decision_dt)
        o = cfg.obs
        k = np.arange(1, o.n_preview + 1)
        self.preview_d = o.preview_horizon_m * (k / o.n_preview) ** o.preview_exponent
        self.preview_scale = np.maximum(self.preview_d, o.preview_min_scale_m)[:, None]
        n = (EGO_SIZE + 3 + 6 * o.n_preview + 3 + int(cfg.obs_gear)
             + (o.history - 1) * (EGO_SIZE + 3))
        self.observation_space = gym.spaces.Box(-o.clip, o.clip, (n,), np.float32)
        self._dummy_obs = np.zeros(n, np.float32)
        self._in_step = False
        self._episode = 0
        self._err_key = None
        self.obs_clipped_total = self.obs_values_total = 0
        self.evaluation = cfg.evaluation

    # --- run metadata ---------------------------------------------------------------------------
    def env_metadata(self):
        """JSON-ready identity of this env for run metadata (QA MINOR 9): full EnvConfig, default
        setup, track (+ sha256 of Track.path, which vector-env workers must match), reference
        line (name, length, digest) and the QSS profile's cache key. Store it with
        `rlsetup.provenance.run_provenance()`."""
        track = self.track
        return {'config': self.cfg.to_dict(), 'setup': self.setup.to_dict(),
                'track': {'name': track.name, 'length_m': float(track.length),
                          'path_sha256': hashlib.sha256(
                              np.ascontiguousarray(track.path).tobytes()).hexdigest()},
                'reference': {'name': self.ref.name, 'length_m': self.ref.length,
                              'digest': self.ref.digest},
                'qss': {'cache_key': self.qss.meta.get('key'), 'lap_time_s': float(self.qss.lap_time_s)},
                'observation_size': int(self.observation_space.shape[0]),
                'decision_dt_s': self.decision_dt}

    # --- actions ------------------------------------------------------------------------------
    def to_inputs(self, action):
        return action_to_inputs(self._validate_action(action).astype(float), self.cfg.pedal_deadzone)

    # --- geometry -------------------------------------------------------------------------------
    @property
    def s_ref(self):
        return self._s_ref

    def _line_errors(self):
        """(e [m], heading error [rad]) to xi_ref, matched once per physics step (cache, R2.2)."""
        key = (self._episode, self.physics_steps)
        if self._err_key != key:
            st = self.state
            self._s_ref = self.ref.match((st['x'], st['y']), self._s_ref)
            self._err = self.ref.errors_at(self._s_ref, st['x'], st['y'], st['yaw'])
            self._err_key = key
        return self._err

    def track_position(self):
        """Lateral position across the track at the car: -1 right edge ... +1 left edge."""
        _, left, right = self.grid.at(self.s)
        d = left - right
        p = np.array([self.state['x'], self.state['y']]) - right
        return float(2. * (p @ d) / (d @ d) - 1.)

    def track_width(self):
        """Track width w(s) = |left edge - right edge| [m] of the grid cross-section at the car."""
        _, left, right = self.grid.at(self.s)
        return float(np.linalg.norm(left - right))

    # --- observation ------------------------------------------------------------------------------
    def _ego(self):
        """Scaled ego block of the current state; accelerations over the last decision."""
        st, o = self.state, self.cfg.obs
        cur = (st['vx'], st['vy'], st['yaw_rate'])
        clock = self._clock()
        if self._kin is None or clock == self._kin[1]:
            ax = ay = 0.
        else:
            ax, ay = body_accelerations(self._kin[0], cur, (clock - self._kin[1]) * DT)
        self._kin_next = (cur, clock)
        slip = lambda x: min(o.slip_clip, max(-o.slip_clip, o.slip_gain * x))
        return np.array([math.hypot(st['vx'], st['vy']) / o.speed_scale_mps, st['vx'] / o.speed_scale_mps,
                         st['vy'] / o.vy_scale_mps, st['yaw_rate'] / o.yaw_rate_scale_radps,
                         ax / o.accel_scale_mps2, ay / o.accel_scale_mps2,
                         st['steer_actual'] / o.steer_scale_rad, slip(st['slip_rl']), slip(st['slip_rr'])])

    def _clock(self):
        return self._settle_steps + self.physics_steps

    def _push_history(self, ego):
        self._hist.appendleft(np.concatenate([ego, self.last_action]))
        self._kin = self._kin_next

    def _observation(self):
        if self._in_step:  # the base step's observation; built once per decision in step()
            return self._dummy_obs
        return self._decision_observation()

    def _decision_observation(self):
        st, o = self.state, self.cfg.obs
        ego = self._ego()
        e, heading = self._line_errors()
        ref, left, right = self.grid.at(self.s + self.preview_d)
        c, s = math.cos(st['yaw']), math.sin(st['yaw'])
        pts = np.stack([ref, left, right], axis=1) - [st['x'], st['y']]          # (N, 3, 2)
        local = np.stack([c * pts[..., 0] + s * pts[..., 1], -s * pts[..., 0] + c * pts[..., 1]], axis=-1)
        preview = (local / self.preview_scale[:, None]).ravel()
        parts = [ego, [e / o.e_scale_m, heading / o.heading_scale_rad, self.track_position()],
                 preview, self.last_action]
        if self.cfg.obs_gear:
            parts.append([self.backend.gear / o.gear_scale])
        parts += list(self._hist)
        raw = np.concatenate(parts)
        clipped = int(np.count_nonzero(np.abs(raw) > o.clip))
        self._last_clipped = clipped
        self.obs_clipped_total += clipped
        self.obs_values_total += raw.size
        self._push_history(ego)
        return np.clip(raw, -o.clip, o.clip).astype(np.float32)

    # --- reset ------------------------------------------------------------------------------------
    def _sample_start(self, options):
        """(s0 on Track.path, f, v0, explicit pose or None). Draws s0 then f from np_random."""
        L, cfg = self.track.length, self.cfg
        if 'start_pose' in options:
            pose = options['start_pose']
            return float(self.track.match([pose.x, pose.y])[0]), None, float(pose.speed), pose
        if self.evaluation:
            s0, f = (-cfg.eval_run_in_m) % L, cfg.eval_speed_fraction
        else:
            s0 = float(self.np_random.uniform(0., L))
            f = float(self.np_random.uniform(*cfg.start_speed_fraction))
        s0 = float(float(options.get('start_s', s0)) % L)
        if 'start_speed' in options:
            return s0, None, float(options['start_speed']), None
        window = f * float(self.qss.speed(s0)) * cfg.settle_s + cfg.settle_window_margin_m
        return s0, f, f * self.qss.min_speed(s0, window), None

    def _start_and_settle(self, setup, s0, v0, pose, trace):
        """Place the car and run the settle phase. Returns False if it left the track."""
        backend, track, ref = self.backend, self.track, self.ref
        if pose is None:
            s_ref0 = float(self.grid.s_ref_at(s0))
            x, y = ref.point(s_ref0)
            pose = StartPose(float(x), float(y), ref.heading(s_ref0), v0)
        else:
            s_ref0 = None
        self.state = st = backend.reset(setup, pose)
        if not track.contains_car(st['x'], st['y'], st['yaw']):
            return False
        self.s = float(track.match([st['x'], st['y']], s0)[0])
        self._s_ref = ref.match((st['x'], st['y']), s_ref0)
        self._settle_steps = self.physics_steps = 0
        self.last_action = np.array([0., -1., -1.], dtype=np.float32)  # nothing pressed
        self._kin = None
        placed = np.concatenate([self._ego(), self.last_action])     # QA MINOR 4 (not zeros)
        self._hist = deque([placed] * (self.cfg.obs.history - 1), maxlen=self.cfg.obs.history - 1)
        self.settle_driver.reset()
        n_decisions = round(self.cfg.settle_s / self.decision_dt)
        steps = self.cfg.action_repeat * CONTROL_STEPS
        for _ in range(n_decisions):
            self._push_history(self._ego())
            action = self.settle_driver.act(st, self._s_ref, v0)
            inputs = self.to_inputs(action)
            for _ in range(steps):
                st = backend.step(inputs)
                self._settle_steps += 1
                if not all(map(math.isfinite, st.values())) or not track.contains_car(st['x'], st['y'], st['yaw']):
                    return False
            self.state = st
            self.last_action = action
            self.s = float(track.match([st['x'], st['y']], self.s)[0])
            self._s_ref = ref.match((st['x'], st['y']), self._s_ref)
            if trace is not None:
                t = backend.get_telemetry()
                trace.append(sum(t[f'Fz_{w}_N'] for w in ('FL', 'FR', 'RL', 'RR')))
        return True

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)  # NOT OCDRacingEnv.reset (second backend.reset, R2.1)
        options = dict(options or {})
        setup = options.get('setup')
        setup = self.setup if setup is None else setup   # N1: None must never reach backend.reset
        self.evaluation = bool(options.get('evaluation', self.cfg.evaluation))
        for attempt in range(1, self.cfg.max_start_attempts + 1):
            s0, f, v0, pose = self._sample_start(options)
            trace = [] if options.get('settle_trace') else None
            if self._start_and_settle(setup, s0, v0, pose, trace):
                break
            if pose is not None or 'start_s' in options or self.evaluation:
                raise RuntimeError(f'fixed start at s = {s0:.1f} m leaves the track during settle')
        else:
            raise RuntimeError(f'no start settled on track in {self.cfg.max_start_attempts} attempts')
        # hand-over (R2.1): episode counters start here; settle steps are not counted
        self._episode += 1
        self._err_key = None
        self.steps, self.done, self.physics_steps = 0, False, 0
        self._stall_s = 0.
        self._last_e = self._line_errors()[0]
        self._trunc_cause = None
        rw = self.cfg.reward       # Phi at hand-over (time_pbrs; 0 in ref_tracking)
        self._phi = -rw.shaping_k * abs(self._last_e) if rw.mode == 'time_pbrs' else 0.
        self._episode_shaping = 0.
        L = self.track.length
        if self.evaluation:
            self.max_time_s = self.cfg.eval_max_s
            d_line = float((L - self.s) % L)                # distance to the next line crossing
            self._lines = [d_line + k * float(L) for k in range(self.cfg.eval_laps + 1)]
            self._crossings = [0.] if d_line == 0. else []
        else:
            self.max_time_s = self.cfg.max_episode_s
            self._lines, self._crossings = [], []
        self.progress = NO_LAP_PROGRESS_M     # the base lap check never fires; see step()
        self._dist = 0.
        self._laps, self._lap_cost = [], _zero_costs()
        self._episode_cost = _zero_costs()
        obs = self._decision_observation()
        e, heading = self._line_errors()
        info = {'setup': setup, 'setup_warnings': self.backend.run_metadata().get('warnings', []),
                'start': {'s0': s0, 'fraction': f, 'v0': v0, 'attempts': attempt,
                          'settle_s': self._settle_steps * DT},
                's_track': self.s, 's_ref': self._s_ref, 'e_ref': e, 'heading_error': heading,
                'speed': math.hypot(self.state['vx'], self.state['vy']), 'evaluation': self.evaluation,
                'phi': self._phi, 'reward_mode': rw.mode}
        if trace is not None:
            info['settle_sum_fz_N'] = trace
        return obs, info

    # --- step -------------------------------------------------------------------------------------
    def _distance(self):
        """Track.path distance driven since hand-over [m] (exact; independent of the offset
        given to the base env's progress counter)."""
        return self._dist

    def _add_costs(self, target, costs, weight=1.):
        for key in COST_KEYS:
            target[key] += weight * costs[key]

    def _time_laps(self, d0, d1, t0, t1, costs):
        """Interpolated line crossings between two base ticks; split the tick's costs by the
        crossing fraction so each lap's time component is exactly b * lap time."""
        j = len(self._crossings)
        if j < len(self._lines) and d0 < self._lines[j] <= d1:
            phi = (self._lines[j] - d0) / max(d1 - d0, 1e-12)
            t_cross = float(t0 + phi * (t1 - t0))
            self._crossings.append(t_cross)
            self._add_costs(self._lap_cost, costs, phi)
            if j >= 1:
                lap = dict(self._lap_cost, lap_time_s=t_cross - self._crossings[j - 1])
                lap['cost'] = lap['time'] + lap['smoothness'] + lap['tracking']   # legacy, both modes
                lap['cost_r1'] = lap['time']          # R1 lap cost; shaping and P never enter laps
                self._laps.append(lap)
            self._lap_cost = _zero_costs()
            self._add_costs(self._lap_cost, costs, 1. - phi)
        else:
            self._add_costs(self._lap_cost, costs)

    def step(self, action):
        if self.done:
            raise RuntimeError('episode has ended; call reset()')
        action = self._validate_action(action)
        rw, cfg = self.cfg.reward, self.cfg
        pbrs = rw.mode == 'time_pbrs'
        total, step_cost = 0., _zero_costs()
        terminated = truncated = False
        cause = None
        self._trunc_cause = None
        decision_dt = 0.
        self._in_step = True
        try:
            for _ in range(cfg.action_repeat):
                prev_action, d0, t0, s0 = self.last_action, self._dist, self.time_s, self.s
                _, r, terminated, truncated, base = super().step(action)
                dt = base['transition_dt_s']
                decision_dt += dt
                nonfinite = base['nonfinite']
                if not nonfinite:  # same ds as OCDRacingEnv.step
                    L = float(self.track.length)
                    self._dist += (self.s - s0 + L / 2) % L - L / 2
                # QA MINOR 1 (kept, as OCDRacingEnv): actual dt -> a shortened terminal tick inflates smoothness.
                du = (action - prev_action) / dt
                costs = dict(progress=rw.a * (self._dist - d0), time=rw.b * dt,
                             smoothness=rw.b * dt * float(np.sum(np.asarray(rw.w) * du ** 2)),
                             tracking=0., penalty=0.)
                if not nonfinite:
                    e, heading = self._line_errors()
                    self._last_e = e
                    costs['tracking'] = rw.c * e ** 2
                if base['off_track']:
                    cause = 'off_track'
                elif nonfinite:
                    cause = 'nonfinite'
                elif not terminated:
                    speed = math.hypot(self.state['vx'], self.state['vy'])
                    self._stall_s = self._stall_s + dt if speed < cfg.stall_speed_mps else 0.
                    if self._stall_s > cfg.stall_time_s:
                        cause = 'stall'
                    elif abs(heading) > cfg.wrong_way_rad:
                        cause = 'wrong_way'
                    if cause == 'stall' and pbrs:      # R1 §1.2: stall = truncation, no penalty
                        truncated, self.done, self._trunc_cause = True, True, 'stall'
                    elif cause:
                        terminated, truncated, self.done = True, False, True
                if base['lap_complete']:   # unreachable: progress starts at NO_LAP_PROGRESS_M
                    raise RuntimeError('base lap-completion check fired')
                if pbrs:                   # R1 per tick: a ds - b dt - [terminal failure] P
                    if cause in TERMINAL_FAILURES:
                        costs['penalty'] = rw.penalty_const
                    r = costs['progress'] - costs['time'] - costs['penalty']
                elif cause in FAILURES:    # ref_tracking: base r, unchanged (bit-identical to full1)
                    costs['penalty'] = rw.failure_penalty(self._last_e, self.dt_tick)
                    r -= costs['penalty']
                if self.evaluation and not nonfinite:
                    self._time_laps(d0, self._distance(), t0, self.time_s, costs)
                    if cause is None and len(self._crossings) == len(self._lines):
                        cause = 'lap_complete'   # QA MINOR 3: all eval_laps + 1 crossings recorded
                        terminated, truncated, self.done = True, False, True
                self._add_costs(step_cost, costs)
                total += r
                if terminated or truncated:
                    break
        finally:
            self._in_step = False
        shaping = 0.
        if pbrs:   # R1 §1.1: F = gamma Phi(s') - Phi(s) per decision; Phi = 0 after a terminal failure
            phi_next = (0. if terminated and cause in TERMINAL_FAILURES
                        else -rw.shaping_k * abs(self._line_errors()[0]))
            shaping = rw.gamma * phi_next - self._phi
            total += shaping
            self._phi = phi_next
            self._episode_shaping += shaping
        if truncated and self._trunc_cause is None:
            self._trunc_cause = 'time_limit'
        self._add_costs(self._episode_cost, step_cost)
        if cause == 'nonfinite':
            obs, self._last_clipped = np.zeros(self.observation_space.shape, np.float32), 0
        else:
            obs = self._decision_observation()
        info = {'progress_m': self._distance(), 'time_s': self.time_s, 'transition_dt_s': decision_dt,
                'termination': cause if terminated else None, 'truncation': self._trunc_cause,
                'step_cost': step_cost, 'episode_cost': dict(self._episode_cost),
                'obs_clipped': self._last_clipped, 'gear': self.backend.gear,
                's_track': self.s, 's_ref': self._s_ref, 'shaping': shaping, 'phi': self._phi,
                'episode_shaping': self._episode_shaping, 'reward_mode': rw.mode}
        if cause != 'nonfinite':
            e, heading = self._line_errors()
            info.update(e_ref=e, heading_error=heading,
                        speed=math.hypot(self.state['vx'], self.state['vy']))
            if cfg.record_telemetry:
                info['telemetry'] = self.backend.get_telemetry()
            if self.evaluation:   # D2 §6: edge margin = (1 - |track_position|) w / 2 - half car width
                info.update(track_position=self.track_position(), track_width_m=self.track_width())
        else:  # QA MINOR 5: same keys, so loggers need no special case
            info.update(e_ref=math.nan, heading_error=math.nan, speed=math.nan)
        if self.evaluation:
            info.update(line_crossings_s=list(self._crossings), laps=[dict(l) for l in self._laps])
        return obs, float(total), bool(terminated), bool(truncated), info
