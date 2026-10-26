# Decisions log (this repo)

Newest last. Each entry: date, decision, who decided, why. OPEN items stay in PROJECT_CONTEXT §11 until decided here.

## 2026-10-25 — Driving agent, phase start (decided by Ibrahim)

1. **ξref placeholder = the safe minimum-curvature line** (`Track.path` in ocd_first_lap: TUM optimizer, 4.5 m width allowance). Reason: footprint verified inside the track; already used by `env.py` and the scripted driver. Jad's laps replace it later through the same loader.
2. **ξref is geometry only.** The driving agent neither observes nor is rewarded on the ξref speed profile; the only ξref term is c·e² in reward (1). Reason: the agent must choose its own speed so each setup shows its own lap time.
3. **Brake torque = tyre-limited value, passed from this repo** via `OCDBackend(max_brake_torque_Nm=…)`. ocd_first_lap's default (360 N·m) and its regression baseline stay untouched. The value is a PLACEHOLDER until SF70H data exists (PROJECT_CONTEXT §11.6 stays open for the real number).
4. The driving agent is independent of the 26-vs-16 knob question (it never sees the setup); it trains on p0 = `Setup()` defaults.

## 2026-10-25 — Second round (decided by Ibrahim, after sim-runner + web-researcher reports)

5. **Brake torque = 12 kN·m total** ("the most realistic"). Measured lock limits (sim-runner, bias 0.5): ≈3.65 kN·m with no aero load, ≈8.2 kN·m at 60 m/s, ≈11 kN·m at 80 m/s (scaled). 12 kN·m can lock at any speed, so the driver must modulate (like a real F1 car, no ABS), and it equals the value `ocd_lap/qss_lap.py` uses for the 133.9 s QSS reference, so RL and QSS laps are comparable. PLACEHOLDER until SF70H data.
6. **Actions stay 3 separate absolute outputs [steering, throttle, brake]**, with an **instructive initialization** (output bias toward no steer / light throttle / no brake, plus a warm start from scripted-driver data). Exact form approved by rl-architect.
7. **Base driver trains on p0 first; randomized-setup base training is a later measured experiment** (fine-tune cost p0-base vs randomized-base), Ibrahim picks with numbers.
8. **ocd_first_lap performance patch approved:** direct getter for the rear-axle speed instead of `car.signals()` in `_sync_powertrain`; results must be bit-identical and `./scripts/qa_gate.sh` must stay green.
9. QSS reference for the default setup on Shanghai: **133.9 s** (shipped OCD), 133.4 s (AC-aligned). This is the driving agent's benchmark, not TUM's 91.24 s (different car/tyres).
   - **Done (25 Oct):** `Car.rear_axle_omega()` in ocd_first_lap `src/binding.cpp`, used by `_sync_powertrain`. Bit-identical (lap.csv sha `d47063de…`, env stream hashes equal), QA gate PASS before and after (59 tests). Env throughput 1,893 → 2,939 steps/s (random), 1,817 → 2,952 (scripted). Not committed in ocd_first_lap (mixed with uncommitted Stage 6 work); standalone diff in `docs/ocd_patches/0001-fast-rear-axle-omega.diff`.

## 2026-10-25 — Third round (decided by Ibrahim, after rl-architect D1 review)

10. **Pedal deadzone ON:** pedal = clip((a + 0.5)/1.5, 0, 1) for throttle and brake (D1_review R3). Still 3 absolute outputs.
11. **No gear in the observation** (strict "driver does not see the setup"; gear at a given speed leaks final drive).
12. **D2 done criterion = architect's proposal:** 3 seeds with a clean deterministic flying lap on p0; lap 2 ≤ 140.6 s (1.05 × 133.9 s QSS); < 0.2 s improvement over the last 20 % of steps; seed std ≤ 0.2 s; ≥ 9/10 stochastic laps clean; env steps + wall time logged.
13. Architect decisions adopted (technical, not OPEN): 25 Hz decisions (action repeat 2, rewards summed, reward (1) unchanged per 50 Hz tick); γ = 0.992; settle 2.0 s; off-track P = P0 + 30·c·e_exit² (P0 = 50); stall/wrong-way termination; setup agent scores the undiscounted evaluation-lap cost, never the training return.

## 2026-10-26 — D2 technical decisions (lead; not OPEN items)
14. Demo steering noise default **0.05** (measured clean fraction 0.41 at 0.1, 0.86–0.89 at 0.05); fallback 0.025 (PLACEHOLDER). Supersedes D2_agent.md §4's 0.1.
15. Demo-dataset cache key = sha256 of the six demo-relevant rlsetup sources (not the rlsetup commit), so unrelated commits don't force recollection. Supersedes D2_agent.md §4.
16. Crash-safe training: replay buffer saved atomically at every checkpoint (1.3 GB, 1.4 s each; 38 GiB free); resume without a buffer continues with a warning.
17. Unattended full run: `scripts/run_d2_full.sh`, 3 seeds sequential, default `MAX_WALL_H=3` per seed (Ibrahim asked for training to continue offline; no explicit hour budget given, so 3 h/seed cap until he says otherwise).
18. BC needs the full preset's 30 epochs (smoke preset's 10 stops before the val-NLL minimum at epoch 15; gate fails at 10, passes at 30 with a clean 340.6 s lap).

## 2026-10-26 — Reward redesign R1 (decided by Ibrahim)
19. **Fastest driver; ξref is initial guidance only** (Ibrahim: "I don't care about the line … I don't care about the driver too"; approved R1–R4). New reward mode `time_pbrs`: r = a·Δs − b·Δt + [γΦ(s′) − Φ(s)], Φ = −k|e| (potential-based shaping, Ng 1999; as Linesight), k = 1/m, γ_s = γ = 0.992 per decision; smoothness and tracking logged only; constant failure P = 100 for off-track/nonfinite/wrong-way; stall → truncation. Supersedes #13's P formula and stall termination **for this mode**; `ref_tracking` (reward (1)) stays the default so full1 is reproducible. Consequence (architect F1): the converged driver no longer depends on ξref (driver axis of PROJECT_CONTEXT §4); this is not v5's annealed c (§12). First run: warm start from full1/seed0 best.zip into a separate run `r1_warm` (own TensorBoard curve, step counter from 0); a from-scratch R1 run comes later. CMDP/Lagrangian deferred. Spec: `docs/specs/R1_time_optimal.md`.
