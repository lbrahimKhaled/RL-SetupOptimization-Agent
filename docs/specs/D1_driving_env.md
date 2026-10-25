# D1 — Driving environment (DRAFT for rl-architect review)

Status: DRAFT. Decisions it builds on: `docs/DECISIONS.md` #1–9. Evidence: `docs/research/driving_agent_sota.md`, sim-runner baseline (summarised in §0).

## 0. Measured facts this spec relies on (sim-runner, 25 Oct)
- `OCDRacingEnv` throughput: ~1,950 env steps/s/core (random), ~1,830 (scripted). OCD physics is ~7 % of step time; `car.signals()` in `_sync_powertrain` is ~33 % (being fixed, DECISIONS #8); `contains_car` 16 %; observation 15 %.
- Shanghai: `Track.path` (safe line, ξref placeholder) 5388.8 m, footprint 100 % inside, worst corner margin +0.94 m. `Track.reference` (TUM raceline) leaves the track on 30 % of samples.
- Reset at speed causes a heave bounce (ζ_heave ≈ 0.06, aero load applied instantly; ΣFz 12–26 kN for seconds at 60 m/s) and a +0.16 m/s first-step speed jump.
- QSS reference lap (default setup, Shanghai): 133.9 s; v 10–80 m/s, mean 40 m/s. Scripted driver: 506 s (capped at 12 m/s).
- Brake 12 kN·m total → wheels lock at any speed at full pedal; tyre-limited ≈ 8.2 kN·m at 60 m/s, 3.65 kN·m with no aero.

## 1. Code layout (this repo)
```
rlsetup/
  __init__.py
  driving/
    config.py      EnvConfig, RewardConfig, ObsConfig dataclasses (all numbers live here)
    reference.py   ReferenceLine: ξref geometry (closed x,y polyline) -> arc length, point(s), tangent(s), match(xy, s_hint), signed lateral error e (+ = left), heading error
    env.py         DrivingEnv(gymnasium.Env)
tests/
  test_reference.py, test_env.py
scripts/
  bench_env.py     throughput; random + scripted-policy episode; prints steps/s, lap time, return
```
`ReferenceLine.from_track_path(track)` now; `ReferenceLine.from_csv(path)` (x, y [m] in the track's frame) for Jad's laps later.

## 2. Compliance with ocd_first_lap
- `DrivingEnv` **subclasses `ocd_lap.env.OCDRacingEnv`** and reuses its `step()` (physics loop, off-track/non-finite/lap termination, truncation, reward (1) with `RewardWeights`). It overrides only `_line_errors` (→ ξref), `_observation`, `reset`, and the observation space.
- Backend: `backend_factory=lambda: OCDBackend(powertrain='engine', max_brake_torque_Nm=cfg.max_brake_torque_Nm)` with 12 000 N·m (DECISIONS #5). Setup = `Setup()` (p0) unless `reset(options={'setup': …})`.
- Progress Δs stays on `Track.path` arc length via `Track.match` (driver-independent coordinate: identical for every ξref). e and heading error come from `ReferenceLine`.
- Units/conventions as ocd_first_lap: DT 0.002 s, 10 physics steps per 50 Hz tick (`CONTROL_STEPS`), lateral offset + = left, steering ±0.6 rad (`MAX_STEER_RAD`).

## 3. Action (DECISIONS #6)
`[steering, throttle, brake] ∈ [-1, 1]³`, absolute, mapped by `OCDRacingEnv.to_inputs` (steer × 0.6 rad; pedals (a+1)/2). OCD's PT1 actuator limits steer rate to 1.5 rad/s. No env-side change. Instructive init is a D2 (agent) matter.

## 4. Observation (all fixed scales, no running normalisation — so fine-tunes on new setups see the same input scaling)
Current tick:
- ego (car frame): speed, v_x, v_y, yaw rate, a_x, a_y (finite difference over the last tick), actual steer angle, rear slip ratios (slip_rl, slip_rr from the backend state)
- ξref: e, heading error
- track position: lateral position across the track as a fraction of width (−1 right edge … +1 left edge) at the car
- preview at N = 20 stations ahead along `Track.path` arc length, d_k = 300 m · (k/N)^1.5 (dense near, sparse far; ~5–6 s at mean speed): for each station, ξref point, left-edge point, right-edge point, in the car frame (x, y) → 120 numbers
- last action (3)
History: ego block + last action of the previous 2 ticks appended (3-tick stack).
Scaling: positions / 100 m, speeds / 80 m/s, accelerations / 30 m/s², yaw rate / 1 rad/s, angles / π, slips ×5 (clipped ±5), e / 5 m. Observation space is a finite Box (clip ±10).

## 5. Reset
- **Training:** start s uniform on [0, L) of `Track.path`, on ξref with ξref heading, speed v0 = U(0.3, 0.7) × QSS speed at s (from `qss_lap`, computed once per track/setup and cached; PLACEHOLDER fractions). Then a **settle phase of 1.0 s** (not part of the episode, not in the replay buffer) with a PI speed hold and pure-pursuit steering (`ScriptedDriver` primitives) so the heave bounce decays before the agent takes over. If the settle phase leaves the track, resample.
- **Evaluation:** fixed start 200 m before the start line, settle, then time one **flying lap** line-to-line (interpolated crossing, as `evaluate.py` `lap_times_s`). Deterministic policy (mean action).
- Training episodes truncate at `max_episode_s` = 60 s (PLACEHOLDER); truncation is `truncated=True` (bootstrapped), off-track is `terminated=True`.

## 6. Reward
Reward (1) as in `OCDRacingEnv`: `r = a·Δs − b·Δt·(1 + u̇ᵀWu̇) − c·e²`, PLACEHOLDER weights a = 1/m, b = 1/s, W = diag(0.01, 0.001, 0.001), c = 0.1/m² (c sweep in D3; c is OPEN §11.8). Off-track: terminate, penalty P.
Question for architect: P and whether it should scale with speed, given research implication #2 (termination must never be cheaper than continuing).

## 7. Info dict
Adds: `s_track`, `e_ref`, `lap_time_s` (when a flying lap completes), `speed`, per-wheel Fz (for later setup loss F(p)), off-track flag. Telemetry for plots via an optional recorder.

## 8. Tests (fast)
ReferenceLine: arc length of a circle, signed e on a known offset, wraparound match across s = L. Env: `gymnasium.utils.env_checker.check_env`; obs finite and within bounds over a scripted lap segment; e = 0 ± 0.05 m when placed on ξref; reward on a hand-computed transition; reset determinism with a seed; settle phase leaves heave transient < X.

## 9. Open points for rl-architect
1. Control rate 50 Hz (as ocd_first_lap) vs action-repeat 2 (25 Hz). GT Sophy: no gain above 10 Hz.
2. Off-track penalty P and speed scaling.
3. Preview stations / horizon; whether to add curvature or keep points only (GT Sophy ablation: points beat rays+curvature).
4. History depth (3 ticks) vs adding previous actions only.
5. Training start-speed distribution and settle phase.
