# D2 smoke run — sim-runner report (in progress, written incrementally)

Started 2026-10-26 ~08:15 local. Repo `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent`, branch `d1-driving-env`.
- rlsetup HEAD at start: `0dadadb5f3f1fce566a9c57ecb73dae9e92a63c3` ("D2 implemented"); tracked tree clean (untracked `.codex/`, `AGENTS.md` only).
- ocd_first_lap HEAD `2147dcd3ed54e51b88ec48155720ff43e8cce3e5`, **dirty** (uncommitted Stage 6 + speed patch: 9 modified, 9 untracked paths). `git diff HEAD` sha256 = `45f07e392e37…` = identical to the value recorded in `runs/bench/bench_env_d1_complete.json` (D1 complete) → unchanged since D1. Not clean in the strict sense; known/documented state (HANDOFF "Things Ibrahim should know").
- Disk at start: `df -h ~` → 460 Gi size, 385 Gi used, **38 Gi free (91 %)**.
- Machine load at start: load avg 2.4–2.9; two other agents' `pytest` processes were running concurrently (PIDs 66086 `pytest -q`, 67224 `pytest -q -m "not slow"`) — throughput numbers may be depressed while they run.
- Logs: `runs/D2/smoke_logs/`.

## Step 1 — collect_demos (smoke preset)
Command: `.venv/bin/python scripts/collect_demos.py --preset smoke` (logs `runs/D2/smoke_logs/collect_demos*.log`).

rl-coder committed twice while this step ran (`39d5ecd` steering noise default 0.05 / fallback 0.025; `2b32e3e` dataset key hashes demo sources instead of the commit). Because the key changed, collection was run three times:

| Run | rlsetup code | Result | Wall |
|---|---|---|---|
| 1 | 0dadadb + uncommitted 39d5ecd edits (dirty) | collected `demos_ae584b8a813a.npz` | 54 s total process (44 s collection) |
| 2 | 39d5ecd → 2b32e3e landed mid-run (key computed with 2b32e3e code) | collected `demos_1c612cde2770.npz` | 45 s (44 s collection) |
| 3 | 2b32e3e (clean) | cache hit `demos_1c612cde2770.npz` | 1 s |

- Noise used: **[0.05, 0.1, 0.1]** (steer, throttle, brake) — the new default; the fallback was **not** needed.
- **Gate: PASS** — 142 episodes, 200,006 transitions, clean fraction **0.887** (≥ 0.80). Failures 16: off_track 15, wrong_way 1; time_limit 126 (clean). Mean reward/decision 0.749, mean speed fraction 0.501, mean episode 1,408 decisions.
- Determinism: runs 1 and 2 produced byte-identical npz (sha256 `81962e3d…41129` both; only the key/metadata differ).
- Dataset used by training: `data/demos/demos_1c612cde2770.npz` (134 MB). The stale `demos_ae584b8a813a.*` (134 MB) remains in `data/demos/` and can be deleted.

## Step 2 — smoke training
Command: `.venv/bin/python -m rlsetup.driving.train --preset smoke --seed 0 --run-name smoke1` (wrapped in `/usr/bin/time -l`), started 08:19:42 local at rlsetup HEAD **`2b32e3ee2c95d4f2239f009e9fb78d30f60f092c`** (tracked tree clean). Log `runs/D2/smoke_logs/train.log`; run folder `runs/D2/smoke1/seed0/`.

**Result: stopped at the BC gate after 16 s wall** (`status.json`: state `bc_gate_failed`, stop_reason "BC deterministic lap: off_track"). No exception; behaviour is as the spec prescribes (§4 step 3: stop, lead decides on `bias_only`). Steps 4a/4b/RL never ran.
- Demos cached (0.7 s). BC: 10 epochs (smoke cap), best epoch 8, best val NLL −2.347 (val NLL still trending down: −2.32 / −2.35 / −2.32 / −2.35 at epochs 7–10), 3.8 s training, 10.8 s phase incl. gate (9,254 env steps).
- BC val MSE vs clean demo action per dim [steer, throttle, brake] = [1.1e-4, 0.0161, 0.0105]; mean σ per dim = [0.052, 0.287, 0.413].
- **Gate: deterministic lap off_track; lap 1/lap 2 = none. Stochastic 0/3 clean** (stall, off_track, off_track).

Diagnostic replay (script `/tmp/d2diag/bc_gate_diag.py`, outside the repo; reloads `bc/actor_bc.pt` into `build_model(cfg, DummyVecEnv)` and calls `evaluate.flying_lap`, same env config). It reproduces the gate exactly:
- Deterministic (seed 0): start s = 5,223.6 m (eval start), off_track at s = 1,598.8 m after 66.99 s / 1,675 decisions (lap 1 not completed; mean speed 26.4 m/s). |e| stays < 0.6 m until s ≈ 1,568 m (e = −0.15 at 18.4 m/s, gear 2), then |e| > 1 m at s = 1,578.8 m (v 17.9 m/s) and grows to 5.6 m within 1.2 s while the policy holds steer ≈ −0.65 with both pedals in the deadzone (throttle ≈ −0.45, brake ≈ −0.85, i.e. coasting). Fails in a slow corner at ~16–18 m/s.
- Stochastic seed 0: **stall** at s = 900 m, 52.5 s (speed fell to 2.2–2.4 m/s in gear 1 with throttle in the deadzone, −0.3 … −0.55).
- Stochastic seed 1: off_track at s = 4,661 m after 233.6 s (34→31 m/s, track_position ≈ +0.5…+0.66, braking).
- Stochastic seed 2: off_track at s = 449 m after 17.1 s (48→40 m/s, track_position ≈ +0.6…+0.7, braking).

## Step 3 on smoke1 (gate-failed run)
`.venv/bin/python scripts/d2_report.py runs/D2/smoke1` → **exit 1, unhandled `FileNotFoundError`** (`scripts/d2_report.py:56` `StagedSAC.load(seed_dir/'checkpoints'/'last.zip')`; a seed stopped at the BC gate has no `last.zip`). Log `runs/D2/smoke_logs/d2_report_smoke1.log`. Finding (minor, for rl-coder): the report should mark seeds whose `status.json` state is `bc_gate_failed`/`demo_gate_failed`/no `last.zip` as FAIL with the reason instead of crashing.

## BC-epochs diagnostic (offline, no RL)
Script `/tmp/d2diag/bc_epochs.py <run> <max_epochs>` (outside the repo) mirrors pipeline steps 1–3 (`build_model` on a DummyVecEnv, `init_output_bias`, `train_bc`, gate laps).
- max_epochs 10: actor **bit-identical** to `runs/D2/smoke1/seed0/bc/actor_bc.pt`; same gate result (det off_track at s = 1,598.8 m; stochastic stall / off_track / off_track). Reproducible.
- **max_epochs 30 (full-preset value): early stop after 18 epochs, best epoch 15 (val NLL −2.426), val MSE clean [1.0e-4, 0.0138, 0.0081], 7.4 s. Gate PASS: deterministic 2-lap run clean, lap 1 = 340.195 s, lap 2 = 340.578 s.** Stochastic 0/3 clean (stall at s = 1,556.5 m, stall at s = 4,738.8 m, off_track at s = 412.4 m) — informative only per spec.
- Reading: the smoke preset's `bc.max_epochs = 10` cuts BC before the val-NLL minimum (epoch 15); the gate outcome is sensitive to this.

## Step 2b — smoke run with `bc.max_epochs=30` (DEVIATION from the smoke preset, labelled)
Rationale: smoke1 stopped at the gate, so none of the RL/infra criteria were measured. This run changes only `bc.max_epochs` (10 → 30, the full-preset value; gate still required) to measure them. It does not replace the lead's decision on the gate/`bias_only`.
rl-coder had uncommitted edits to `train.py`, `callbacks.py`, `train_config.py` at launch, so the run uses a **detached git worktree pinned at `2b32e3e`** (`/Users/ibrahimkhaled/Documents/rlsetup-wt-smoke`, created with `git worktree add --detach`; no source edited), with absolute `runs_root` / `demo_dir` back into the main repo:
```sh
cd /Users/ibrahimkhaled/Documents/rlsetup-wt-smoke
/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/.venv/bin/python -m rlsetup.driving.train --preset smoke --seed 0 \
  --run-name smoke1_bc30 --set bc.max_epochs=30 \
  --set runs_root=/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/runs/D2 \
  --set demo_dir=/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/data/demos
```
Started 08:25:12. Log `runs/D2/smoke_logs/train_bc30.log`; run folder `runs/D2/smoke1_bc30/seed0/`. BC epochs 1–18 identical to the offline diagnostic.
(running)

### Provenance check (smoke1_bc30 `provenance.json`)
Recorded: rlsetup commit `2b32e3e` (dirty False, diff sha = empty-diff hash), ocd_first_lap commit `2147dcd` dirty True + diff sha `45f07e39…`, `ocd_lap_code_version`, rlsetup + extra ocd_lap source hashes, config sha, demo dataset sha `81962e3d…`, BC actor sha `ab95cc35…`, QSS cache-key sha + lap 133.916 s, reference digest, track path sha, packages, platform, hostname, torch threads 1, n_envs 4, argv, UTC start. Complete per spec §9.

**Finding (minor, provenance):** `ocd_lap_code_version` is `c1cdd4cae7cc` in every D2 artefact, vs `73d2c1637f6c` in the D1 bench. Nothing in ocd_first_lap changed (no file newer than the D1 bench; binary dated 25 Oct 22:27; same diff hash). Cause: `rlsetup/provenance.py:106` `training_provenance` calls `run_provenance(track)` with a **Track object**; `ocd_lap.provenance.capture_provenance` builds `f'{track}.csv'` paths, finds none and silently omits the track/raceline CSV hashes. Measured: `capture_provenance(None)` → `c1cdd4cae7cc`, `capture_provenance('Shanghai')` → `73d2c1637f6c`. `demos.py:64,179` and `qss_profile.py:108` call `run_provenance()` without a track (same digest). `scripts/bench_env.py:108` passes `track.name` (correct). Fix: pass `track.name` in `training_provenance` (and in the demo/qss calls if they should be comparable). The track path sha is still recorded separately (`track_path_sha256`), so no information about the run is lost. Changing it changes the demo dataset key (it contains the code version), so the demos would be recollected once (~45 s).

### Compliance (code read, D2 modules)
- D2 modules reach the simulator only through `DrivingEnv` (D1, previously verified) and `ocd_lap.track.Track` (train.py, collect_demos.py, d2_report.py). `evaluate.py` / `demos.py` read `env.unwrapped` attributes of rlsetup's own DrivingEnv (state, track length, demo driver), not OCDBackend internals. No `setattr`/monkeypatching of ocd_lap found. Settings flow via `EnvConfig`/`TrainConfig` and `--set` overrides.
- ocd_first_lap untouched by this work (`git status --short` unchanged, diff sha unchanged). It remains dirty from the known Stage 6 + speed-patch work.

### Interim (smoke1_bc30, as of 08:30)
- Gate in pipeline: deterministic clean, lap 2 = 340.578 s (= offline diagnostic); stochastic 0/3.
- Prefill: 46,732 rows × 4 columns = 186,928 transitions (13,078 dropped by column trimming), 3 flagged tails.
- Offline warm-up (5 × 1,000 grad steps, ~6 s per block ≈ 167 grad steps/s): critic loss 1.59 → 1.15 → 1.61 → 1.84 → 2.12; mean Q 12.2 → 38.6. Q vs soft MC on demos: Q_min 39.7 vs MC 63.7 (ratio 0.62).
- Unfreeze event at step 10,000 (08:27:17). Probes (baseline 463.3 m): 15k 445.2 m, 20k 440.0 m, 25k 615.9 m — all ok (≥ 0.5 × baseline).
- Eval step 0: lap 2 340.578 s clean, mean |e| 0.085 m, max |e| 0.63 m, min edge margin 0.96 m, front lock 0, spin upshifts 0, α 0.100, eval wall 19.7 s.
- Eval step 25k: **deterministic off_track after 1,037 decisions (det_regression event)**; stochastic 0/3 (all off_track); Q_min 106.1 vs MC 76.4 (ratio 1.39, overestimating); α 0.0535; entropy −2.94; obs clip 0.0; training causes so far: off_track 53, stall 9, wrong_way 6.
- **Throughput at 25k: 154 env steps/s, 77 grad steps/s** (wall_train 161.9 s, wall_eval 26.2 s cumulative). **CPU contention:** during this window other agents ran `pytest -q` (PID 75602), `pytest -q -x -m …` (PID 82663) and a qa-reviewer D2 training run (PID 81697, `--run-name a --set runs_root=/tmp/qa_d2/det/runs …`, 2 envs); load avg 5–7 on 4 P-cores. Throughput is therefore not a clean measurement.

## Final results — smoke1_bc30 (sim-runner, 26 Oct, run ended 08:48:06; resume 10:53–10:56)
Run ended on the wall-clock limit at **147,704 steps** (status `wall_clock_limit`, 0 rollbacks); process 1,377.5 s real, peak RSS 1.1 GB. Phases (status.json): demos 0.7 s (cached), BC 25.1 s, offline warm-up 31.7 s (5,000 grad steps), online warm-up 62.0 s (10,000 steps), RL 1,247.2 s (137,704 steps, 78,850 updates total).

### Evaluations (evals.jsonl)
| step | det result | lap 1 / lap 2 [s] | sto clean | Q_min / MC (ratio) | α | env/s | eval wall [s] |
|---|---|---|---|---|---|---|---|
| 0 | clean | 340.195 / 340.578 | 0/3 | 37.4 / 54.1 (0.69) | 0.100 | – | 19.7 |
| 25k | off_track @1,037 dec | – | 0/3 | 106.1 / 76.4 (1.39) | 0.054 | 154 | 6.6 |
| 50k | off_track @430 | – | 0/3 | 171.1 / 84.5 (2.02) | 0.072 | 131 | 1.7 |
| 75k | off_track @485 | – | 2/3 (145.87, 145.05) | 156.9 / 122.0 (1.29) | 0.079 | 160 | 7.7 |
| 100k | off_track @3,586 | – | 0/3 | 142.0 / 101.9 (1.39) | 0.069 | 142 | 9.7 |
| 125k | off_track @1,390 | – | 0/3 | 138.5 / 82.1 (1.69) | 0.083 | 141 | 5.9 |
| 147,704 | **clean** | **131.146 / 130.817** | **3/3** (137.45, 133.62, 137.22) | 161.9 / 150.0 (1.08) | 0.084 | 124 | 22.6 |
| 150k (resume) | off_track | – | 3/3 | – | 0.084 | 172 | – |
| 175k (resume) | off_track | – | 2/3 | – | 0.084 | 159 | – |

Final lap 2 (147,704): mean |e| 0.61 m, max |e| 3.53 m, min edge margin 0.08 m, front-lock fraction 0.058, spin upshifts 22, smoothness share 0.029, tracking cost 387.5 (BC: 19.9). **Lap 2 = 130.8 s is 3.1 s faster than the QSS reference lap (133.916 s) after 148k steps; this needs scrutiny by qa/architect** (it is legal under the env rules, since there is no off_track and the edge margin is > 0, but the agent leaves ξref by up to 3.5 m and runs within 8 cm of the edge; the QSS lap is on the fixed reference line, so a shorter/wider line can beat it). The deterministic policy is unstable between evals: 5 of 6 post-unfreeze evals before the last were off_track, and both resume evals were off_track again.

Training termination causes (cumulative to 147,704): off_track 127, time_limit 46, stall 12, wrong_way 8. Note: after resume, `train_causes_total` restarts from 0 (minor: not cumulative across resume).

### §11.1 pass criteria
| Criterion | Result | Evidence |
|---|---|---|
| No exceptions | **PASS** (smoke1_bc30 + resume). smoke1 (spec smoke preset) stopped cleanly at the BC gate. | logs `runs/D2/smoke_logs/train_bc30.log`, `resume_bc30.log` |
| Throughput within 20 % of benchmark (328 env steps/s, 4 workers, UTD 0.5) | **FAIL as measured / not attributable**: per-eval windows 124–160 env steps/s (62–80 grad steps/s); train-only 147,704 / 1,046.3 s = 141 /s; RL phase incl. evals/probes/checkpoints 110 /s; resume 159–172 /s. This is 47–66 % below the benchmark, but other agents ran pytest + a qa D2 training run during 25 of 30 samples (load 5–9), and during the resume the full1 run was also running (load 15–17). A clean uncontended measurement was not possible. | `contention.log`, evals.jsonl |
| Critic loss finite, |Q| < 2,000 | **PASS**: critic loss 1.06–73.2, all finite (53 progress rows); max |Q| over all 334,628 buffer transitions = 554.5 (Q(s,a_buffer) ∈ [−554.5, 308.7]; max |Q(s, a~π)| 553.4), all finite. Q overestimates soft MC by 1.08–2.02× after unfreeze (risk 3 to watch). | `/tmp/d2diag/qmax.py` |
| Probe progress after unfreeze ≥ 0.5 × BC | **PASS, 0 rollbacks**: BC baseline 463.3 m; 27 probes 15k–145k, minimum 440.0 m (20k), then 560–914 m. | probes.jsonl |
| α in [1e-4, 1] | **PASS**: 0.0451–0.1001 (progress.csv), 0.084 at the end | sb3/progress.csv |
| Obs clip < 1 % | **PASS**: 0.0 in every eval, incl. resume | evals.jsonl |
| Checkpoint save/load/resume | **PASS**: d2_report re-evaluation of last.zip reproduces logged lap 2 exactly (delta 0.000 s). Resume 147,704 → 175,000 steps in 191 s, exit 0, new SB3 log `sb3/resume1/`, last.zip and buffer rewritten. Resume adds no `phases` entry in status.json (only events); minor. | `resume_bc30.log` |
| Report script runs (criteria FAIL, no crash) | **PASS on smoke1_bc30** (exit 0, 155 s; i, ii, v, vi PASS; iii plateau and iv seed-std FAIL as expected for 1 seed / short run). **Crash on smoke1** (gate-failed run, no last.zip), see above. | `runs/D2/smoke1_bc30/d2_report.md` |

### Other requested numbers
- BC gate: spec smoke preset (10 epochs) **FAIL** (det off_track at s = 1,598.8 m); with 30 epochs (early stop 18, best 15) **PASS**, BC lap 2 = **340.578 s**; stochastic 0/3 in both.
- wall_eval_s vs wall_train_s (cumulative at 147,704): **73.7 s vs 1,046.3 s (7.0 %)** at a 25k interval with 3 stochastic laps. Probes + checkpoints + buffer save are in neither number: online+RL wall 1,309.2 s − 1,120.0 s = 189 s (14 %). Eval cost scales with how long the policy survives: a clean 2-lap det run + 3 clean stochastic laps (~16.9k decisions) took 22.6 s (~750 decisions/s). Estimate for a full-preset eval (det 2 laps + 10 stochastic laps of a lap-capable policy ≈ 40k decisions): ~55 s per 50k steps, i.e. ~15 % of the train time at 140 steps/s (estimate, not measured).
- `replay_buffer.pkl`: **1,296,003,190 bytes (1.30 GB)**. This is the full 1e6 preallocated capacity (250,000 × 4 × 159 float32 obs + next_obs), so it is the same size for every run regardless of fill. Run folder 1.3 GB total; checkpoints ~7.8 MB each.
- Free disk (`df -h ~`): 38 Gi at start, **36 Gi** at end (92 % used). 3 full seeds ≈ 4 GB of buffers + checkpoints: fits. The stale `data/demos/demos_ae584b8a813a.*` (134 MB) can be deleted.
- Steps/hour (measured, contended): RL phase incl. all overheads **110 steps/s ≈ 397k steps/h**; train-only 141 /s ≈ 508k/h; whole pipeline 107 /s ≈ 387k/h. Uncontended might approach the benchmark ratio, but that is not measured.

### Recommendation: total_timesteps for the full preset under a 3 h/seed cap
Full-preset fixed cost ≈ 3–4 min (BC ~25 s, offline 20k grad steps at ~167/s ≈ 2 min, online warm-up 20k at ~160/s ≈ 2 min). At the measured contended end-to-end rate (~397k/h), 3 h gives **~1.15–1.2M steps** (estimate). **Recommend `total_timesteps = 1.2e6` with `max_wall_h = 3` as a backstop.** If the machine is left uncontended, measure the rate from full1's first evals and raise it.
The running full1 uses `total_timesteps = 2e6, max_wall_h = 3`. At measured rates the wall cap will stop it before 2M (estimate ~1.2–1.5M). `d2_report.plateau` takes T = last eval step, so criterion (iii) is still evaluated correctly on a wall-capped run. Only the planned and actual T will differ.

### Housekeeping
- The detached worktree `/Users/ibrahimkhaled/Documents/rlsetup-wt-smoke` was removed after use (`git worktree remove`). smoke1_bc30's provenance records its path and commit `2b32e3e`.
- ocd_first_lap at the end: `git status --short` 18 entries, diff sha `45f07e39…`, unchanged from the start and from D1. qa_gate was not run (not requested).
- Diagnostic scripts (not in the repo): `/tmp/d2diag/bc_gate_diag.py`, `bc_onset.py`, `bc_epochs.py`, `qmax.py`.
