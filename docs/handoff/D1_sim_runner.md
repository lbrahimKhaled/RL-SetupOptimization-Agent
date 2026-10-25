# D1 sim-runner report (compliance + rerun + D2 capacity), 2026-10-25

Machine: Apple M4, 10 cores (hw.perflevel0.physicalcpu = 4 P, hw.perflevel1.physicalcpu = 6 E), 16 GB.
Python 3.12 (.venv), SB3 2.9.0, torch 2.14.1, gymnasium 1.4.0, numpy 2.5.0.
Logs: /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/runs/d1_verify/, runs/bench/, runs/d2_capacity/

## 1. Compliance with ocd_first_lap

**ocd_first_lap untouched: YES.**
- `git status --short` identical to the baseline given (9 modified, 9 untracked), before and after all runs.
- sha256 of all 25 modified/untracked files recorded before the runs (runs/d1_verify/ocd_dirty_files_before.sha256), unchanged after.
- `results/Shanghai/safe_line.csv` NOT rewritten: sha256 e114444c…cf4, mtime Oct 19 00:20:35 2026 (before and after the QA gate, pytest, benches, vec-env workers). `safe_line.meta.json` same (e6b4eaf9…, Oct 19).
- `ocd_native.cpython-312-darwin.so` sha256 2b62143d… unchanged by the QA gate rebuild (so the QSS cache key stays valid).
- ocd_first_lap HEAD 2147dcd3ed54e51b88ec48155720ff43e8cce3e5 (dirty: uncommitted Stage 6 + approved rear_axle_omega patch, DECISIONS #8/#9).

**Only public pieces / no monkeypatching: YES (no setattr/mock/module-attribute assignment anywhere in rlsetup/, scripts/, tests/).**
Imports used: `ocd_lap.backend.{OCDBackend, StartPose, Inputs, DT, CONFIG}`, `ocd_lap.env.{OCDRacingEnv, RewardWeights, MAX_STEER_RAD}`, `ocd_lap.track.{Track, closed_path}`, `ocd_lap.car_setup.Setup`, `ocd_lap.simulation.CONTROL_STEPS`, `ocd_lap.driver.{REAR_AXLE_M, WHEELBASE_M}`, `ocd_lap.qss_lap.{measure_ggv, machine_limit, lap_time}`, `ocd_lap.translator.P`, `ocd_lap.provenance.{capture_provenance, provenance_digest}`, `ocd_native` (only `__file__` for a hash). Tests also use `ocd_lap.manoeuvres._speed_pedals` (private, test-only, deliberate cross-check).
Backend attributes read: `backend.params` (qss_profile.py:89), `backend.gear` (env.py:195,399; property), `backend.run_metadata()`, `get_telemetry()`.

**Brake 12 000 N·m via backend_factory: YES** — env.py:103-104 `backend_factory=lambda: OCDBackend(powertrain='engine', max_brake_torque_Nm=cfg.max_brake_torque_Nm)`, config.py:88 default 12000. Measured: `env.backend.control_limits()` -> brake_torque_Nm 12000.0. ocd_first_lap's 360 N·m default untouched. No explicit test asserts env.backend.max_brake_torque_Nm == 12000; it is only implied by test_equivalence (base built with 12000, bit-equal rewards with braking actions).

**Coupling to OCDRacingEnv private/internal state** (DrivingEnv subclasses it; every base `self.<attr>` it touches):

| Base attr / method | DrivingEnv use (env.py lines) | Pinned by a test? |
|---|---|---|
| `track` | read everywhere; WRITTEN at 106 (replaces the Track built by base ctor) | yes: test_default_setup_is_p0_and_track_is_shared |
| `setup` | read 266 | yes: same test |
| `max_time_s` | WRITTEN in reset 285, 291 (base step reads it for truncation) | partly: test_time_limit_truncates (training), test_evaluation_reset_is_fixed (value only; eval truncation not exercised) |
| `weights` | via ctor arg only (off_track_penalty=0, P added by subclass) | yes: test_equivalence (bit-equal rewards) |
| `backend` | via backend_factory; read 225-260, 300, 399, 406 | implicitly (equivalence) |
| `state` | WRITTEN 232, 254 (reset/settle); read throughout | yes: equivalence |
| `s` | WRITTEN 235, 256; read 147, 187, 286, 349, 356 | yes: equivalence |
| `progress` | WRITTEN 294 (−1e7 in training, −(d_line+(laps−1)L) in eval, to steer the base lap_complete check) | yes: test_training_never_completes_a_lap, test_evaluation_reset_is_fixed, slow flying-lap test |
| `steps` | WRITTEN 280 | yes: test_settle_hands_over_with_its_last_action |
| `done` | WRITTEN 280, 377; read 338 | partly: test_off_track_… (step after end raises); not after stall/wrong_way |
| `physics_steps` | WRITTEN 237, 280; read 139 (cache key), 172 | yes: settle test (time_s == 0), equivalence |
| `last_action` | WRITTEN 238, 255; read 192, 349 | yes: settle test, test_reward_matches_hand_computation |
| `time_s` (property) | read 349, 384, 396 | yes |
| `_validate_action` (private static) | called 130, 340 | yes: test_bad_actions_rejected_before_advancing |
| `_line_errors` (private, OVERRIDDEN; base step uses it for c·e²) | 137 | yes: equivalence |
| `_observation` (private, OVERRIDDEN, returns a dummy inside step) | 178 | indirectly (obs-size tests); the dummy short-circuit itself is not asserted |
| `to_inputs` (OVERRIDDEN, deadzone) | 129 | yes: test_no_deadzone_mapping_equals_ocd_env + equivalence |
| base `step` info keys `transition_dt_s, nonfinite, off_track, lap_complete` | 351-378 | off_track/lap_complete yes; `nonfinite` path untested |
| base `reset` | NOT called (re-implements its field init, review R2.1) | equivalence test would catch a new base field only if it changes rewards |

Verdict: compliant with the rules (no edits, no monkeypatching, settings via ctor), but DrivingEnv writes 9 base attributes and overrides 2 private methods; the R2.5 equivalence test is the guard. Risks, not violations:
- (low) env.py:356 duplicates the base ds formula, and evaluation lap timing uses `_dist` while the base terminates on `progress`; both sum the same ds but from different float offsets, so in principle the base can terminate one tick before the last crossing is recorded (would give 1 lap instead of 2). Not observed (3 crossings in every run).
- (low) Missing pins: explicit brake-torque assert; eval-mode truncation; `nonfinite` termination path; `done` after stall/wrong_way.

**Units and signs: consistent with ocd_first_lap.**
- e positive = left: reference.py:85 is the same expression as ocd_lap/env.py:74; heading error = yaw − line heading wrapped (same as base). Equivalence test bit-equal.
- Steering +0.6 rad = left: MAX_STEER_RAD imported; OCD `steering_actuator.angle_max_rad` = 0.6 (measured). Positive steer gives positive yaw rate (test_body_frame_accelerations: v·r > 3).
- Car frame x forward, y left (env.py:190 rotation by −yaw); track_position −1 right … +1 left (env.py:147-152).
- Body accelerations a_x = dv_x/dt − r v_y, a_y = dv_y/dt + r v_x (z up), checked by test on a steady circle.
- DT = 0.002 s and CONTROL_STEPS = 10 imported, not hard-coded; dt_tick = 0.02 s, decision = 0.04 s (action_repeat 2).
- Pure-pursuit geometry WHEELBASE_M 3.2 = OCD `wheelbase_m` 3.2 (measured).
- Spring compression / pitch are not used by DrivingEnv (only Fz telemetry in the settle trace).

**Provenance:** bench JSON records config (EnvConfig.to_dict), ocd_first_lap commit + dirty/untracked flags, rlsetup commit + flags, `ocd_lap_code_version` (73d2c1637f6c), machine, python. Gaps:
- rlsetup commit is ddbc2b6 (initial) with ALL D1 code untracked -> the hash does not identify the code. Fix: commit D1 before D2 runs, or add a sha256 of rlsetup/*.py + scripts to `run_provenance()`.
- ocd_first_lap is dirty; its commit hash does not identify the code, and `ocd_lap.provenance.capture_provenance` does NOT hash `ocd_lap/env.py` (DrivingEnv's base class), `qss_lap.py`, `handling.py`, `manoeuvres.py` (ocd_lap/provenance.py:35-37 + simulation.SOURCES). qss_profile's own cache key covers qss_lap/handling/manoeuvres but not env.py. Fix (in rlsetup/provenance.py, no ocd_first_lap change): add sha256 of every ocd_lap/*.py actually imported (at least env.py) and of `git diff` in ocd_first_lap.

## 2. ocd_first_lap QA gate

`cd /Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap && ./scripts/qa_gate.sh` -> **QA GATE: PASS** (exit 0). Log: runs/d1_verify/qa_gate.log
- build PASS; unit tests 59 OK (9.2 s); Shanghai lap ×2: lap_time_s 504.72000000168185 both, max_deviation_m 0.2137216622769679 both; lap.csv sha256 d47063de…f1ef both (= PRE_PATCH hash); baseline |Δ| 0 s / 2.3e-14 m; vendor OCD clean, on ocd-first-lap, 2 commits = 2 patches, pin+patches tree = HEAD tree.

## 3. Independent rerun

`.venv/bin/python -m pytest -q` -> **50 passed in 16.13 s** (incl. slow tests). Settle statistic printed: first-attempt settles **200/200**, ΣFz within 5 %: **200/200**, worst 2.550 %. Log: runs/d1_verify/pytest.log

`.venv/bin/python scripts/bench_env.py --json runs/bench/bench_env_rerun.json` (22.1 s wall), then again to runs/bench/bench_env_rerun2.json:

| Quantity | Coder | Rerun 1 | Rerun 2 |
|---|---|---|---|
| random, decisions/s (step only) | 1,745 | 1,767 | 1,773 |
| random, decisions/s incl. resets | – | 1,240 | 1,245 |
| reset (settle 2 s) | – | 21.5 ms | 21.4 ms |
| random episodes: mean length / causes | – | 89.4 decisions; 44 stall, 12 off_track (56 resets) | same |
| demo f=0.5, decisions/s (step only / incl. policy) | – | 1,801 / 1,737 | 1,785 / 1,721 |
| flying lap f=0.5 (lap 2) | 268.375 s | **268.375 s** (lap 1 268.363) | identical |
| flying lap f=0.7 (lap 2) | 192.033 s | **192.033 s** (lap 1 192.005) | identical |

f=0.5 flying lap cost 275.412 = time 268.375 + smoothness 0.549 + tracking 6.487; return 10384.2; max |e| 0.298 m; clip fraction 0; 13,561 decisions in 8.0 s.
f=0.7: cost 217.356 = 192.033 + 0.716 + 24.607; return 10500.8; max |e| 0.830 m; clip 0; 9,719 decisions in 5.7 s.
QSS p0 lap 133.916 s; obs size 159.

**Determinism: identical.** All non-timing fields of the two JSONs (flying laps incl. line_crossings_s to the last bit, returns, random-episode statistics and causes, config, provenance) are equal.

## 4. D2 capacity (measured on the M4; throwaway scripts in runs/d2_capacity/, no source edited)

### 4a. DrivingEnv throughput, SB3 SubprocVecEnv(start_method='forkserver'), random actions, total decisions/s
Command: `.venv/bin/python runs/d2_capacity/vec_throughput.py --workers 1 2 4 6 8 10 --steps 2000` (+ repeats noted). Auto-resets (21 ms settle each) included.

| Workers | random, 2000 vec steps | random, 5000 steps rep 1 / rep 2 | random, thread caps =1 (OMP/VECLIB/OPENBLAS/MKL) | demo-driver actions (60 s episodes, few resets) |
|---|---|---|---|---|
| 1 (DummyVecEnv) | 1,207 | – | – | 1,574 |
| 1 (Subproc) | 1,136 | – | – | 1,218 |
| 2 | 1,669 | – | – | – |
| 4 | 1,875 | 1,340 / 1,484 | 2,355 | 3,143 |
| 6 | 2,175 | – | – | – |
| 8 | 2,652 | 1,537 / 1,771 | 2,369 | 4,309 |
| 10 | 2,638 | – | – | – |

Ceiling without a vec env (N independent processes, no lockstep; `runs/d2_capacity/indep_procs.py 1 4 8`): 1 → 1,257, 4 → 4,216, 8 → 5,667 decisions/s incl. resets (step-only 1,776 / 5,956 / 8,029).
Reading: SubprocVecEnv scales poorly with random actions (lockstep: a 21 ms reset in any worker stalls all; ~0.2 ms IPC per step; 6 of 10 cores are E-cores). Run-to-run noise is large (±25 %). With long episodes (demo actions) 4 → 3.1k, 8 → 4.3k.

### 4b. SB3 2.9.0 SAC `train()` gradient steps/s, CPU, 3×256 MLPs (actor + 2 critics, 868,362 policy params), batch 512, obs 159, act 3
Command: `.venv/bin/python runs/d2_capacity/sac_train_speed.py --threads T --steps 1000` (dummy env, 20k random transitions in the buffer, 100 warm-up steps, train() only).

| torch threads | grad steps/s (same process) | separate processes |
|---|---|---|
| 1 | 198 | 193 |
| 2 | 157 | 156, 159 |
| 4 | 149 | 157, 119 |
| MPS (extra) | – | 60 |

1 thread is fastest on this machine; more threads are slower. MPS is 3× slower than CPU at this size.

### 4c. End-to-end SAC.learn() (SubprocVecEnv forkserver, torch threads 1, train_freq 1 vec step, batch 512, 3×256, untrained policy, 8,000 env decisions after 1,024 learning_starts)
Command: `.venv/bin/python runs/d2_capacity/sac_e2e.py --workers 8 --grad 2 4 8` and `--workers 4 --grad 1 2 4`.

| Workers | UTD (grad steps per env decision) | env decisions/s | grad steps/s |
|---|---|---|---|
| 8 | 0.25 | 597 | 149 |
| 8 | 0.5 | 333 | 166 |
| 8 | 1.0 | 169 | 169 |
| 4 | 0.25 | 578 | 145 |
| 4 | 0.5 | 328 | 164 |
| 4 | 1.0 | 189 | 189 |

### Balance
- The learner is the bottleneck: env and learner are balanced at UTD ≈ grad rate / env rate ≈ 193 / 1,900–2,650 ≈ 0.07–0.10. At any UTD ≥ 0.25 SB3's synchronous loop spends most of its time in train(), and 8 workers give no gain over 4 (measured 597 vs 578 at UTD 0.25; 169 vs 189 at UTD 1).
- Suggested: 4 workers (= P-core count), torch.set_num_threads(1). The UTD choice is the architect's; its measured cost: UTD 1 ≈ 170–190 decisions/s, UTD 0.5 ≈ 330, UTD 0.25 ≈ 580–600.
- ESTIMATES (extrapolated from the rates above, not measured): 1 M decisions ≈ 1.5 h at UTD 1, ≈ 50 min at UTD 0.5, ≈ 28 min at UTD 0.25 per seed on the M4; 3 seeds in parallel would compete for the 4 P-cores.
- Higher env throughput only helps with an asynchronous learner (collection overlapping training), which SB3 does not do.
