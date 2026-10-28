# comparison/ — TUM's lap vs the same lap in OCD vs the RL driver (Shanghai)

Data for a later comparison analysis. Nothing here is analysed; it is clean, documented data and
the scripts that make it. Three sources, one CSV per lap/episode, SI units, one row per sample,
same column names and units wherever the quantity exists in more than one source. Every CSV has a
`<name>.meta.json` next to it (provenance: code commits, settings, lap time, sha256 of the CSV and
of its inputs).

| Folder | What | Files |
|---|---|---|
| `tum/` | TUM laptime-simulation's own lap (QSS point mass), per raceline point | `tum_tum_default.csv` (91.238 s), `tum_ers.csv` (93.243 s), `tum_none.csv` (94.886 s) |
| `tum_pure_pursuit_ocd/` | TUM's lap driven in OCD (`sf70h_tum`, ERS on) by pure pursuit on TUM's raceline at TUM's `ers` speed profile, 50 Hz | `pp_f1.00.csv` (lap 2 = 104.700 s), `search.json` (speed-scale search and diagnostic sweep) |
| `tum_pure_pursuit_ocd_noers/` | The same, but **ERS off** (no refill) and TUM's `none` speed profile (no ERS, no DRS): the RL run's conditions. `scripts/comparison/tum_pure_pursuit.py --no-ers --tum-csv comparison/tum/tum_none.csv --out comparison/tum_pure_pursuit_ocd_noers` | `pp_f1.00.csv` (lap 2 = 105.720 s, 72.6 % of lap 2 on track), `search.json` (f = 1.00 ... 0.80: 105.72, 105.49, 106.08, 107.16, 108.44, 109.92, 111.60, 113.46, 115.53, 117.81, 120.28 s). Made on Linux x86_64; the ERS-on run reproduces `pp_f1.00.csv` (104.700 s) there exactly |
| `rl/` | The RL driver (`runs/mac/D2/sf70h_mac_r1/seed0/checkpoints/best.zip`, step 6.95M) on its own env | `rl_det_seed0.csv` (deterministic, lap 2 = 103.238 s, 50 Hz), `rl_sto_seed0..9.csv` (stochastic, 25 Hz), `summary.json` |
| `figures/` | Made from the CSVs above only (no simulation), **ERS and DRS off everywhere**: RL, TUM `none`, `tum_pure_pursuit_ocd_noers`. `scripts/comparison/compare_laps.py` and `ghost_video.py` | `lap_comparison.png` (speed, offset from the RL reference line xi_ref with the track edges, and time delta vs distance; time lost per corner group; g-g; lap times), `segments.json` (per corner group: time, min / max speed, braking point, every source), `ghost_lap.png` (one video frame); `ghost_lap.mp4` (real-time ghost lap with xi_ref and the 20 xi_ref preview points in the RL observation, 109 s, ~22 MB) is made locally and not committed |

Common conventions (OCD sources): x, y in the racetrack-database frame (the same frame as TUM's
raceline file); yaw counter-clockwise from +x; body frame x forward, y left; lateral offsets
positive = left of the line; `lap` = line crossings so far (0 run-in, 1 lap 1, 2 lap 2 = the
flying lap used as the reference, 3 = after it). The timing line is Track.path s = 0, crossing
times interpolated between 50 Hz ticks (as `DrivingEnv` and `ocd_lap.simulation.drive`); TUM's
raceline s = 0 is abeam Track.path s = -0.06 m. Floats have 7 significant digits; empty fields
are missing values (for example `ers_energy_J` with ERS off, or reward columns on non-decision
ticks). Lap times in the `.meta.json` files are full precision. How the data was built, checks and
open points: [BUILD_NOTES.md](BUILD_NOTES.md).

## Condition differences between the sources

These are not like-for-like laps. Read this table before comparing anything.

| | `tum/` | `tum_pure_pursuit_ocd/` | `rl/` |
|---|---|---|---|
| Model | TUM quasi-steady-state forward/backward solver (FB+): point mass on the line, per-wheel loads from static + steady load transfer + aero, axle-wise friction check, no slip, yaw or suspension dynamics | OCD double-track, MF5.2 tyres, PT1 steering actuator, suspension, 2 ms physics step | same as pure pursuit |
| Car | `F1_Shanghai.ini` (TUM @ `afa0aa7`): 733 kg, TUM linear tyre law, CzA 2.20/2.68, CxA 1.56, rho 1.18 | `sf70h_tum` = OCD double-track with every value TUM's ini gives (mass, geometry, tyre friction mapped onto MF5.2, aero, ERS); inertias, tyre slip stiffness and shape, suspension stay AV-21 PLACEHOLDERs (ocd_first_lap `docs/SF70H_TUM_CAR.md`); setup p0 = `Setup()` | same as pure pursuit |
| Tyre radius | grows with speed (+4.5 % per 200 km/h) | fixed 0.3299 m | same |
| Gearbox | ideal: at each speed the lowest gear whose rpm is below its shift rpm (TUM `find_gear`), instant; rotating-mass factor e_i on acceleration | ocd_lap powertrain: upshift at TUM's shift rpm, 1000 rpm downshift hysteresis, >= 0.1 s between shifts, follows rear-axle speed (wheelspin can upshift); engine inertia passed to OCD | same |
| ERS | `tum_default`: 4 MJ FCFB **with** recuperation (MGU-K 15 % of braking energy up to 2 MJ, turbo 10 % of engine power; 6.99 MJ deployed); `ers`: 4 MJ FCFB, no recuperation; `none`: off. TUM boosts only the torque above the engine maximum | **on**: 4 MJ per lap, refilled to 4 MJ at each line crossing, no recuperation; the motor adds its share to every throttle request above 27.8 m/s until empty | **off** (`EnvConfig.ers = False` in the run's config) |
| DRS | `tum_default`: on (TUM's two Shanghai zones, drag x 0.83 while accelerating); `ers`, `none`: off | none (the time-domain backend has no DRS) | none |
| Recuperation | `tum_default` only | no | no |
| Line | TUM's raceline (racetrack-database `Shanghai.csv`, spline re-interpolated to 5 m, curvature filtered over 10 m); length 5341.43 m (spline) | pure pursuit on the same raceline as a polyline (`Track.reference`, 5340.77 m). With the 4.6 x 2.0 m footprint placed exactly on it, only 70.1 % of the raceline (per metre) is inside the track | the driver's own line; reward `time_pbrs` shapes toward xi_ref = `Track.path`, the safe minimum-curvature line (4.5 m width allowance, 5388.77 m; footprint 100 % inside when on it) |
| Speed | QSS optimum minus TUM's 0.5 m/s cornering margin | target f x v_TUM(`ers`) 0.5 s ahead, speed PI on the pedals (`DemoDriver`, brake cap raised to 1.0) | learned policy |
| Track limits | none (point on the line) | not enforced: the car keeps driving, `on_track` records the env's footprint check | enforced: the episode ends at the first off-track 2 ms step (crashed stochastic laps are kept, flagged in `termination`) |
| Brakes | ideal distribution, tyre limited; front-axle braking limited by its weaker wheel (`limit_braking_weak_side = 'FA'`); friction ellipse exponent 2 | 12 kN m total pedal torque, bias 0.5, no ABS | same |
| Start | periodic flying lap (TUM finds the start speed that closes the lap) | placed on the raceline abeam Track.path s = -200 m at f x v_TUM there; lap 2 reported | 2 s settle from Track.path s = -200 m at 0.5 x min v_QSS (not part of the episode), then the policy; lap 2 reported (stochastic episodes: 1 lap, as the training evaluation) |
| Sample | one row per 5 m raceline point (1069 + closing point) | 50 Hz control ticks | deterministic: 50 Hz base ticks (2 per 25 Hz decision); stochastic: 25 Hz decision ends |
| Lap time | 91.238 / 93.243 / 94.886 s | 104.700 s (f = 1.00), 70.8 % of lap 2 ticks on track | 103.238 s deterministic lap 2; stochastic 7/10 clean, mean 105.574 s |

## How each source was produced

### `tum/` — `scripts/comparison/export_tum_lap.py`

TUM's code is LGPL-3.0 and is not vendored; it runs from a clone outside this repo, in its own
venv (NumPy 1.x), exactly as ocd_first_lap's `scripts/run_tum_laptimesim.py` documents. The script
imports that file's `TRACK`, `SOLVER`, `DRIVER`, `DEBUG`, `CASES` dictionaries (read only) and
builds TUM's `Track`, `CarHybrid`, `Driver`, `Lap` the same way, so the settings are
`main_laptimesim.py`'s with only the ERS/DRS switches changed. It stops with an error unless the
clone is clean at `afa0aa7` and `tum_default` reproduces 91.238 s. Each row is one raceline point
of TUM's `Lap` object: state at the point (`*_cl` arrays: t, v, gear, rpm, energy store, fuel,
energy used) and quantities of the 5 m step that starts there (torques, MGU-K recuperation, DRS,
tyre loads). Force potentials are TUM's tyre law (`car.tire_force_pots`) applied to the stored
tyre loads. TUM does not store per-step turbocharger harvest or the solver's internal a_x; `ax`
is the kinematic (v[i+1]^2 - v[i]^2) / (2 ds).

### `tum_pure_pursuit_ocd/` — `scripts/comparison/tum_pure_pursuit.py`

ocd_first_lap `OCDBackend(powertrain='engine', car='sf70h_tum', ers=True,
max_brake_torque_Nm=12000)`, setup `Setup()`. Controller: `rlsetup.driving.demo_driver.DemoDriver`
(no new control law) on a `ReferenceLine` of `Track.reference`, one command per 50 Hz tick:

- Steering: pure pursuit from the rear axle (ScriptedDriver geometry), goal 3 m + 0.35 s x vx
  ahead on the raceline, delta = atan(2 L sin(alpha) / d), limited to +-0.6 rad.
- Pedals: speed PI on vx with kp = 0.4 per m/s, ki = 0.2 per m, integral limited to +-5 m.
  Traction control scales the throttle by 1 - (slip - 0.08) / 0.12 above a rear slip of 0.08.
  Brake = clip(-u, 0, 1.0).
- Target: v_t = f x v_TUM(s + 0.5 s x vx), `DemoPolicy`'s preview rule. v_TUM is TUM `ers`
  mapped onto `Track.reference` by position; TUM's points lie within 0.1 mm of the polyline.
- Only change from `DemoConfig`: `brake_cap` 0.3 -> 1.0, because 0.3 x 12 kN m is about 1.5 g,
  against TUM's 4-5 g braking.

Stops only on spin (|heading error to the raceline| > pi/2, the env's wrong-way rule), non-finite
state, stall (< 3 m/s for > 2 s) or 400 s. Speed-scale rule: f* = highest f in 1.00, 0.98, ...
that completes lap 2 without a stop. f = 1.00 already completes it, so f* = 1.00 and only
`pp_f1.00.csv` is exported. `search.json` also holds a diagnostic sweep down to f = 0.80
(summaries only): lap 2 = 104.70, 104.34, 104.61, 105.70, 106.93, 108.34, 109.95, 111.76,
113.76, 115.97, 118.38 s for f = 1.00 ... 0.80. Lap-2 on-track fraction is 0.71-0.78, max |e|
to the raceline 17.7 m at f = 1.00 and 2.4 m at f = 0.80. The run is deterministic
(byte-identical CSV on a rerun). At f = 1.00 the closed loop settles into a periodic lap (laps 2
and 3 cross the line at the same tick phase), so lap 2 = 5235 ticks = 104.700 s exactly.

### `rl/` — `scripts/comparison/export_rl_laps.py`

The repo's own evaluation path: env config = the run's `config.json` (`env`), eval envs as
`rlsetup.driving.train.eval_envs` (deterministic: `evaluation=True`, 2 laps; stochastic: 1 lap)
with `record_telemetry=True`. The model is `StagedSAC.load(best.zip)` on CPU. Every episode is
`rlsetup.driving.evaluate.flying_lap` with the training evaluation's seeds: deterministic seed 0;
stochastic k = 0..9 with env reset seed and torch seed k. Recording is a read-only hook after
each 50 Hz base tick (a subclass inserted between `DrivingEnv` and `OCDRacingEnv` in the MRO; no
rlsetup or ocd_lap file is changed).

Reproduction against the training evaluation at step 6.95M (`evals.jsonl`): deterministic laps
103.89331652777723 / 103.23790594919477 s, identical to the last digit. All 10 stochastic
episodes are identical in termination and lap time: 7 clean, seeds 1, 2, 4 off track, clean mean
105.574 s. The recorder's own line crossings and safe-line offsets equal the env's: lap times to
1e-9 s, `e_path_m` minus `e_ref_env_m` = 0.

## Columns

Generated from `scripts/comparison/common.py` (`SCHEMAS`); `tests/test_comparison_schema.py`
checks that every CSV header and these tables match it.

### `tum/*.csv` (one row per raceline point)

| Column | Unit | Meaning |
|---|---|---|
| `i` | - | raceline point index after TUM's 5 m re-interpolation (0 = start/finish line); the last row is the closing point N = first point of the next lap |
| `closing_point` | 0/1 | 1 on the closing row; per-step columns are empty there |
| `s_m` | m | arc length along TUM's interpolated raceline (Track.dists_cl, spline length) |
| `x_m` | m | raceline point x (track frame of the racetrack-database, same as ocd_lap Track.reference) |
| `y_m` | m | raceline point y |
| `kappa_radpm` | 1/m | curvature of the raceline, filtered as in TUM (Track.kappa); positive = turning left |
| `t_s` | s | time at this point since the line (Lap.t_cl); the closing row holds the lap time |
| `speed_mps` | m/s | speed at this point (Lap.vel_cl) |
| `ax_mps2` | m/s^2 | longitudinal acceleration over the step i -> i+1: (v[i+1]^2 - v[i]^2) / (2 ds) (point-mass kinematics) |
| `ay_mps2` | m/s^2 | lateral acceleration v^2 kappa at this point; positive = left |
| `gear` | - | gear, 1-based (Lap.gear_cl) |
| `engine_rpm` | 1/min | engine speed (Lap.n_cl x 60) |
| `engine_torque_crank_Nm` | N m | combustion-engine torque used in the step (Lap.m_eng, crankshaft) |
| `e_motor_torque_crank_Nm` | N m | electric-motor (MGU-K) torque used in the step (Lap.m_e_motor, crankshaft) |
| `requested_torque_crank_Nm` | N m | powertrain torque the tyres could transmit in the step (Lap.m_requ, crankshaft) |
| `engine_power_W` | W | 2 pi n m_eng |
| `e_motor_power_W` | W | 2 pi n m_e_motor (mechanical) |
| `ers_energy_J` | J | energy-storage state at this point (Lap.es_cl); 0 for variant none |
| `ers_used_J` | J | electric energy drawn by the motor since the line, incl. efficiency (Lap.e_cons_cl) |
| `ers_recuperated_step_J` | J | MGU-K recuperation in the step (Lap.e_rec_e_motor; turbocharger harvest is not stored by TUM) |
| `fuel_used_kg` | kg | fuel used since the line (Lap.fuel_cons_cl) |
| `drs` | 0/1 | DRS open in this step (Track.drs; drag x (1 - drs_factor) while accelerating) |
| `em_boost_use` | 0/1 | driver allows the e-motor boost in this step (Driver.em_boost_use) |
| `throttle_pos` | - | driver throttle position in this step (Driver.throttle_pos; 1 unless lift-and-coast / yellow flags) |
| `mu` | - | track friction factor (Track.mu = mu_mean x mu_weather) |
| `Fz_FL_N` | N | tyre load FL as stored by the solver (Lap.tire_loads: static + load transfer + aero) |
| `Fz_FR_N` | N | tyre load FR as stored by the solver (Lap.tire_loads: static + load transfer + aero) |
| `Fz_RL_N` | N | tyre load RL as stored by the solver (Lap.tire_loads: static + load transfer + aero) |
| `Fz_RR_N` | N | tyre load RR as stored by the solver (Lap.tire_loads: static + load transfer + aero) |
| `fx_pot_FL_N` | N | longitudinal force potential FL: mu (mux + dmux_dfz (Fz - fz_0)) Fz, TUM car.tire_force_pots law on Fz_FL_N |
| `fx_pot_FR_N` | N | longitudinal force potential FR: mu (mux + dmux_dfz (Fz - fz_0)) Fz, TUM car.tire_force_pots law on Fz_FR_N |
| `fx_pot_RL_N` | N | longitudinal force potential RL: mu (mux + dmux_dfz (Fz - fz_0)) Fz, TUM car.tire_force_pots law on Fz_RL_N |
| `fx_pot_RR_N` | N | longitudinal force potential RR: mu (mux + dmux_dfz (Fz - fz_0)) Fz, TUM car.tire_force_pots law on Fz_RR_N |
| `fy_pot_FL_N` | N | lateral force potential FL: mu (muy + dmuy_dfz (Fz - fz_0)) Fz, same law |
| `fy_pot_FR_N` | N | lateral force potential FR: mu (muy + dmuy_dfz (Fz - fz_0)) Fz, same law |
| `fy_pot_RL_N` | N | lateral force potential RL: mu (muy + dmuy_dfz (Fz - fz_0)) Fz, same law |
| `fy_pot_RR_N` | N | lateral force potential RR: mu (muy + dmuy_dfz (Fz - fz_0)) Fz, same law |

### `tum_pure_pursuit_ocd/*.csv` (one row per 50 Hz tick)

| Column | Unit | Meaning |
|---|---|---|
| `t_s` | s | time since the start of the recording (car placed 200 m before the line; RL: since hand-over after the 2 s settle) |
| `lap` | - | line crossings before this sample: 0 = run-in, 1 = lap 1, 2 = lap 2 (the flying lap), 3 = after lap 2 |
| `t_lap_s` | s | time since the last line crossing (crossing interpolated between ticks); time since start on the run-in |
| `progress_m` | m | distance driven along Track.path since the start (unwrapped) |
| `s_path_m` | m | arc length on Track.path (ocd_first_lap safe minimum-curvature line), [0, 5388.77) |
| `s_raceline_m` | m | arc length on Track.reference (TUM raceline polyline), [0, 5340.77) |
| `x_m` | m | CoG x (track frame) |
| `y_m` | m | CoG y |
| `yaw_rad` | rad | yaw angle (OCD psi, counter-clockwise from +x; not wrapped) |
| `speed_mps` | m/s | hypot(vx, vy) |
| `vx_mps` | m/s | body-frame longitudinal velocity |
| `vy_mps` | m/s | body-frame lateral velocity, positive = left |
| `yaw_rate_radps` | rad/s | yaw rate, positive = counter-clockwise (left) |
| `ax_mps2` | m/s^2 | body-frame acceleration dvx/dt - r vy from OCD's state derivative at the sample |
| `ay_mps2` | m/s^2 | body-frame acceleration dvy/dt + r vx from OCD's state derivative, positive = left |
| `steer_cmd_rad` | rad | steering command (OCD steering_actuator target), positive = left |
| `steer_actual_rad` | rad | actual steering angle after OCD's PT1 actuator (0.6 rad, 1.5 rad/s limits) |
| `throttle` | - | throttle pedal [0, 1] |
| `brake` | - | brake pedal [0, 1] (x 12 kN m total, split by the setup brake bias 0.5) |
| `drive_torque_axle_Nm` | N m | rear-axle drive torque OCD received in the last 2 ms step (transmission output) |
| `brake_torque_FL_Nm` | N m | brake torque FL OCD received in the last 2 ms step |
| `brake_torque_FR_Nm` | N m | brake torque FR OCD received in the last 2 ms step |
| `brake_torque_RL_Nm` | N m | brake torque RL OCD received in the last 2 ms step |
| `brake_torque_RR_Nm` | N m | brake torque RR OCD received in the last 2 ms step |
| `gear` | - | gear, 1-based (ocd_lap.powertrain, TUM F1 ratios) |
| `engine_rpm` | 1/min | engine speed = rear-axle speed x overall ratio of the current gear |
| `engine_torque_crank_Nm` | N m | throttle x full-load engine torque at engine_rpm (ocd_lap.powertrain, recomputed at the sample) |
| `e_motor_torque_crank_Nm` | N m | throttle x e-motor torque at engine_rpm and speed (0 with ERS off or empty store; recomputed at the sample) |
| `ers_energy_J` | J | ERS energy left (refilled to 4 MJ at each line crossing); empty with ERS off |
| `slip_rl` | - | rear-left longitudinal slip ratio (backend state) |
| `slip_rr` | - | rear-right longitudinal slip ratio (backend state) |
| `kappa_FL` | - | longitudinal slip ratio FL (OCD x_vec) |
| `kappa_FR` | - | longitudinal slip ratio FR (OCD x_vec) |
| `kappa_RL` | - | longitudinal slip ratio RL (OCD x_vec) |
| `kappa_RR` | - | longitudinal slip ratio RR (OCD x_vec) |
| `alpha_FL_rad` | rad | slip angle FL (OCD x_vec) |
| `alpha_FR_rad` | rad | slip angle FR (OCD x_vec) |
| `alpha_RL_rad` | rad | slip angle RL (OCD x_vec) |
| `alpha_RR_rad` | rad | slip angle RR (OCD x_vec) |
| `Fz_FL_N` | N | vertical tyre force FL (OCD) |
| `Fz_FR_N` | N | vertical tyre force FR (OCD) |
| `Fz_RL_N` | N | vertical tyre force RL (OCD) |
| `Fz_RR_N` | N | vertical tyre force RR (OCD) |
| `e_raceline_m` | m | lateral offset to TUM's raceline (Track.reference), positive = left |
| `heading_err_raceline_rad` | rad | yaw minus raceline heading, wrapped to [-pi, pi) |
| `e_path_m` | m | lateral offset to Track.path (the RL driver's safe line xi_ref), positive = left |
| `track_position` | - | lateral position across the track at s_path: -1 right edge ... +1 left edge (DrivingEnv.track_position) |
| `on_track` | 0/1 | car footprint (4.6 x 2.0 m box) inside the track at every 2 ms physics step of the tick ending here (Track.contains_car, the env's own check) |
| `target_speed_mps` | m/s | speed-controller target: f x v_TUM(ers) at the raceline point 0.5 s x vx ahead |

### `rl/*.csv` (one row per 50 Hz base tick; stochastic files: decision ends only)

| Column | Unit | Meaning |
|---|---|---|
| `t_s` | s | time since the start of the recording (car placed 200 m before the line; RL: since hand-over after the 2 s settle) |
| `lap` | - | line crossings before this sample: 0 = run-in, 1 = lap 1, 2 = lap 2 (the flying lap), 3 = after lap 2 |
| `t_lap_s` | s | time since the last line crossing (crossing interpolated between ticks); time since start on the run-in |
| `progress_m` | m | distance driven along Track.path since the start (unwrapped) |
| `s_path_m` | m | arc length on Track.path (ocd_first_lap safe minimum-curvature line), [0, 5388.77) |
| `s_raceline_m` | m | arc length on Track.reference (TUM raceline polyline), [0, 5340.77) |
| `x_m` | m | CoG x (track frame) |
| `y_m` | m | CoG y |
| `yaw_rad` | rad | yaw angle (OCD psi, counter-clockwise from +x; not wrapped) |
| `speed_mps` | m/s | hypot(vx, vy) |
| `vx_mps` | m/s | body-frame longitudinal velocity |
| `vy_mps` | m/s | body-frame lateral velocity, positive = left |
| `yaw_rate_radps` | rad/s | yaw rate, positive = counter-clockwise (left) |
| `ax_mps2` | m/s^2 | body-frame acceleration dvx/dt - r vy from OCD's state derivative at the sample |
| `ay_mps2` | m/s^2 | body-frame acceleration dvy/dt + r vx from OCD's state derivative, positive = left |
| `steer_cmd_rad` | rad | steering command (OCD steering_actuator target), positive = left |
| `steer_actual_rad` | rad | actual steering angle after OCD's PT1 actuator (0.6 rad, 1.5 rad/s limits) |
| `throttle` | - | throttle pedal [0, 1] |
| `brake` | - | brake pedal [0, 1] (x 12 kN m total, split by the setup brake bias 0.5) |
| `drive_torque_axle_Nm` | N m | rear-axle drive torque OCD received in the last 2 ms step (transmission output) |
| `brake_torque_FL_Nm` | N m | brake torque FL OCD received in the last 2 ms step |
| `brake_torque_FR_Nm` | N m | brake torque FR OCD received in the last 2 ms step |
| `brake_torque_RL_Nm` | N m | brake torque RL OCD received in the last 2 ms step |
| `brake_torque_RR_Nm` | N m | brake torque RR OCD received in the last 2 ms step |
| `gear` | - | gear, 1-based (ocd_lap.powertrain, TUM F1 ratios) |
| `engine_rpm` | 1/min | engine speed = rear-axle speed x overall ratio of the current gear |
| `engine_torque_crank_Nm` | N m | throttle x full-load engine torque at engine_rpm (ocd_lap.powertrain, recomputed at the sample) |
| `e_motor_torque_crank_Nm` | N m | throttle x e-motor torque at engine_rpm and speed (0 with ERS off or empty store; recomputed at the sample) |
| `ers_energy_J` | J | ERS energy left (refilled to 4 MJ at each line crossing); empty with ERS off |
| `slip_rl` | - | rear-left longitudinal slip ratio (backend state) |
| `slip_rr` | - | rear-right longitudinal slip ratio (backend state) |
| `kappa_FL` | - | longitudinal slip ratio FL (OCD x_vec) |
| `kappa_FR` | - | longitudinal slip ratio FR (OCD x_vec) |
| `kappa_RL` | - | longitudinal slip ratio RL (OCD x_vec) |
| `kappa_RR` | - | longitudinal slip ratio RR (OCD x_vec) |
| `alpha_FL_rad` | rad | slip angle FL (OCD x_vec) |
| `alpha_FR_rad` | rad | slip angle FR (OCD x_vec) |
| `alpha_RL_rad` | rad | slip angle RL (OCD x_vec) |
| `alpha_RR_rad` | rad | slip angle RR (OCD x_vec) |
| `Fz_FL_N` | N | vertical tyre force FL (OCD) |
| `Fz_FR_N` | N | vertical tyre force FR (OCD) |
| `Fz_RL_N` | N | vertical tyre force RL (OCD) |
| `Fz_RR_N` | N | vertical tyre force RR (OCD) |
| `e_raceline_m` | m | lateral offset to TUM's raceline (Track.reference), positive = left |
| `heading_err_raceline_rad` | rad | yaw minus raceline heading, wrapped to [-pi, pi) |
| `e_path_m` | m | lateral offset to Track.path (the RL driver's safe line xi_ref), positive = left |
| `track_position` | - | lateral position across the track at s_path: -1 right edge ... +1 left edge (DrivingEnv.track_position) |
| `on_track` | 0/1 | car footprint (4.6 x 2.0 m box) inside the track at every 2 ms physics step of the tick ending here (Track.contains_car, the env's own check) |
| `decision` | - | index of the 25 Hz decision this tick belongs to (0 = first decision after hand-over) |
| `decision_end` | 0/1 | 1 on the tick that ends a decision; the columns from `reward` on are filled only there (stochastic files hold only these rows) |
| `a_steer` | - | raw policy action held over the decision, steering, [-1, 1] (x 0.6 rad) |
| `a_throttle` | - | raw policy action, throttle, [-1, 1] (pedal = clip((a + 0.5) / 1.5, 0, 1)) |
| `a_brake` | - | raw policy action, brake, [-1, 1] (same deadzone mapping) |
| `reward` | - | reward of the decision (time_pbrs: r_progress - r_time - r_penalty + r_shaping) |
| `r_progress` | - | a ds over the decision (a = 1 per m of Track.path progress) |
| `r_time` | - | b dt over the decision (b = 1 per s) |
| `r_smoothness` | - | b dt du^T W du (logged only, not rewarded in time_pbrs) |
| `r_tracking` | - | c e^2 to the safe line (logged only, not rewarded in time_pbrs) |
| `r_penalty` | - | failure penalty P (100 for off track / non-finite / wrong way) |
| `r_shaping` | - | potential-based shaping gamma Phi(s') - Phi(s), Phi = -|e_path| |
| `e_ref_env_m` | m | the env's own offset to the safe line at the decision end (info e_ref; equals e_path_m) |
| `termination` | text | env termination at this decision: off_track, nonfinite, wrong_way, lap_complete (empty otherwise) |
| `truncation` | text | env truncation at this decision: time_limit, stall (empty otherwise) |

## Regenerate

From the repo root, in this order (the pure-pursuit run reads `tum/tum_ers.csv`):

```sh
# 1. TUM (once: clone + venv outside the repo, see the script docstring)
git clone https://github.com/TUMFTM/laptime-simulation.git ~/Documents/tum/laptime-simulation
git -C ~/Documents/tum/laptime-simulation checkout afa0aa710adb9fdd23fa3d6fbdd6a13664253616
python3.12 -m venv ~/Documents/tum/tumenv
~/Documents/tum/tumenv/bin/pip install "numpy==1.26.4" "scipy<1.14" "matplotlib<3.9" quadprog
~/Documents/tum/tumenv/bin/pip install --no-deps "trajectory-planning-helpers==0.66"
~/Documents/tum/tumenv/bin/python scripts/comparison/export_tum_lap.py ~/Documents/tum/laptime-simulation   # ~1 min

# 2. Pure pursuit in OCD (~50 s incl. the diagnostic sweep)
.venv/bin/python scripts/comparison/tum_pure_pursuit.py

# 3. RL driver (~35 s; needs runs/mac/D2/sf70h_mac_r1/seed0, which is not in git)
.venv/bin/python scripts/comparison/export_rl_laps.py

# figures and the ghost-lap video (~10 min; needs ffmpeg and ocd_first_lap on PYTHONPATH)
(cd scripts/comparison && ../../.venv/bin/python compare_laps.py)
.venv/bin/python scripts/comparison/ghost_video.py

# column tables of this README
.venv/bin/python scripts/comparison/common.py
```

On a rerun with the same code the CSVs are byte-identical. Only `created_utc` and the wall-clock
fields (`wall_s`) in the JSON files change. The git state
of this repo goes into the metadata, so a rerun from a commit also changes the `commit` fields.

## Known caveats

- `data/qss_profile_Shanghai_p0_sf70h_tum.npz` was built on Windows. Its cache key does not match
  this Mac's code hashes and Track.path, so `qss_profile.load_or_build` recomputes the profile in
  memory with a warning. The training run hit the same warning, which is why the RL laps
  reproduce. The profile only sets the RL settle start speed.
- `s_m` in `tum/` is TUM's spline arc length (5341.43 m per lap). `s_raceline_m` in the OCD files
  is the polyline arc length of the same raceline (5340.77 m). Match by position (`x_m`, `y_m`) or
  rescale.
- OCD `engine_torque_crank_Nm` / `e_motor_torque_crank_Nm` are recomputed at the sample state
  (throttle x full-load curves at the sampled rpm). `drive_torque_axle_Nm` is the torque OCD
  actually received in the last 2 ms step. TUM's torques are the solver's per-step values.
- OCD `ax_mps2` / `ay_mps2` are instantaneous body-frame accelerations at the sample. TUM's `ax`
  is a 5 m step average, and its `ay` is v^2 kappa on the line.
