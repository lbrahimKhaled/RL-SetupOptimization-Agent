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
