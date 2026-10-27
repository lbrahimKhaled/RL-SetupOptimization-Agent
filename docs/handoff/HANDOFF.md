# HANDOFF — read this first to resume (last updated 25 Oct 2026)

## 27 Oct 2026 — Mac vs Windows (read first if you are on the Windows box)
- Two machines: the Mac (M4, CPU, forkserver, `.venv`) and Windows (i9-14900 + RTX 4000 Ada, CUDA, spawn, conda `C:\Users\imk14\ocdenv`). Per-machine launchers, overrides and run roots (`runs/mac/D2`, `runs/windows/D2`) are in [platforms/](../../platforms/README.md). Mac paths in the sections below (`/Users/ibrahimkhaled/...`, `caffeinate`) apply to the Mac only.
- Windows: launch long runs with `platforms\windows\start_detached.ps1` (hidden `Start-Process`); `nohup &` from Git Bash dies with its shell. Live run `runs/D2/full_win/seed0` (pre-split root); resume command in `platforms/windows/README.md`.
- Windows tuning: not measured yet (`platforms/windows/bench.ps1`). Test failures on Windows, their causes and proposals, plus the open question about the car (AV-21 default vs sf70h_tum): [W1_windows_port.md](W1_windows_port.md).

Goal of the current phase: build the **SAC driving agent** only; evaluate and tune it; then move on to the setup agent.

## Where everything is
| What | File |
|---|---|
| Project context (from Ibrahim) | `docs/PROJECT_CONTEXT.md` |
| Every decision so far (#1–13) | `docs/DECISIONS.md` |
| Literature survey (cited) | `docs/research/driving_agent_sota.md` |
| D1 env spec + architect review (review wins) | `docs/specs/D1_driving_env.md`, `docs/specs/D1_review.md` |
| Architect sign-off on D1 deviations (2 fixes needed: A5 obs scales, N1 setup=None guard) | `docs/specs/D1_signoff.md` (DONE) |
| **D2 agent spec** | `docs/specs/D2_agent.md` (DONE) |
| QA review of D1 (no blockers; M-1 add 5 test groups; 12 minors) | `docs/handoff/D1_qa_review.md` (DONE) |
| sim-runner compliance + D2 throughput benchmark (D1 PASS; use 4 workers, torch 1 thread; learner-bound above UTD 0.25) | `docs/handoff/D1_sim_runner.md` (DONE) |
| ocd_first_lap speed patch (applied there, uncommitted) | `docs/ocd_patches/0001-fast-rear-axle-omega.diff` |

If one of the "(if it finished)" files is missing or partial, rerun that agent with the same task (see "Resume prompts").

## Status
- **Done:** team; context docs; baseline measurements; literature; ocd_first_lap speed patch (1.6×, bit-identical, QA gate PASS); D1 driving env implemented (by rl-coder).
- **D1 coder report (25 Oct):** 50 tests pass (`.venv/bin/python -m pytest -q`, 16 s; fast suite `-m "not slow"` 3.4 s). 200/200 random-start settles OK (worst ΣFz deviation 2.55 %). 1,745 decisions/s/core (25 Hz decisions; 1,225/s incl. resets); obs size 159. Demo driver flying laps: f = 0.5 → 268.375 s, f = 0.7 → 192.033 s (QSS/f = 267.8 / 191.3), max |e| 0.83 m, clip fraction 0. Equivalence with `OCDRacingEnv` rewards exact.
- **D1 coder deviations awaiting architect sign-off:** (1) demo driver has traction control (without it 4/12 demo episodes spin out); (2) DrivingEnv defaults to `Setup()` (base env default = shipped OCD, zero downforce); (3) Track still loaded once per env by the parent ctor; (5) steer angle scaled by π (max 0.09); (6) random-action episodes mostly end in stall (deadzone → ~37 % brake), min start 3.05 m/s vs 3 m/s stall threshold; (7) several readings of ambiguous spec points (see D1_signoff.md).
- **Not done:** QA blockers fixed (if any) → commit D1 cleanly → D2 (SAC + instructive init) → D3 (c sweep) → D4 (fine-tune protocol, seed variance) → setup agent.

## Things Ibrahim should know / decide
- ocd_first_lap has **uncommitted Stage 6 work** + our speed patch in the same files. Recommend Ibrahim commits Stage 6 first, then applies/commits the speed patch, so runs can pin a hash.
- OPEN still: c value (D3 sweep), B laps / fine-tune budget (§11.8; architect notes OCD is deterministic, so B repeated laps add nothing — variance is across fine-tune seeds).

## Resume prompts
1. "Read docs/handoff/HANDOFF.md. Check which handoff/spec files exist. Rerun any missing review (QA: qa-reviewer on D1; compliance: sim-runner on D1; architect: D1 sign-off + D2 spec), then have rl-coder fix blockers, then commit D1."
2. After D1 is committed: "Have rl-coder implement docs/specs/D2_agent.md; then qa-reviewer + sim-runner verify; run the ~20 min smoke run."

## NEXT ACTION (as of all D1 reviews landing)
All four D1 reviews are in and committed. Next: rl-coder applies **D1 fix list** = D1_signoff A5 (obs scales) + N1 (setup=None guard) + D1_qa_review M-1 (5 test groups) + MINOR 2, 3, 4, 5, 8, 10 + provenance additions (D1_sim_runner §1 / QA minor 9). Then rerun full pytest + bench, commit "D1 complete", then implement D2_agent.md.

## D1 fixes applied (25 Oct 2026, rl-coder) — supersedes "NEXT ACTION" above
Commits on `d1-driving-env`: `c25e2b0` N1 + A5, `7f7a101` QA M-1 tests, `a738b2a` QA minors, `b02a831` provenance, then "D1 complete".
- **N1:** `reset(options={'setup': None})` = default setup (never reaches `backend.reset`). **A5:** `ObsConfig.vy_scale_mps = 5`, `steer_scale_rad = 0.6` (= MAX_STEER_RAD), `heading_scale_rad = 0.5` replace `angle_scale_rad`; obs layout frozen from D2 on. Demo laps f = 0.5 / 0.7: clip fraction 0; p99 |obs| v_y 0.11 / 0.14, steer 0.32 / 0.32, heading 0.20 / 0.20.
- **M-1:** 5 test groups added. All 8 QA mutants are killed by the fast suite (checked by mutating env.py one at a time). Deviation: the stall-reset test starts at 2.9 m/s, not 3.5 m/s; from 3.5 m/s the first tick zeroes an inherited timer, so the mutant survived.
- **Minors:** 2 (time limits must be whole decision periods, tolerance 1e-9), 3 (evaluation ends on this env's crossing count; base lap check out of reach in both modes), 4 (initial history = ego at placement + [0, −1, −1]), 5 (NaN `e_ref`/`heading_error`/`speed` on nonfinite), 8 (bench prints n/a; checked with a failing f = 0.85 lap), 10 (ReferenceLine rejects open path, s not starting at 0 / not strictly increasing, segments ≤ 1e-6 m). 1 documented only.
- **Provenance:** `git_state` adds `diff_sha256` (`git diff HEAD`); `run_provenance` adds `rlsetup_sources_sha256` and `ocd_lap_extra_sources_sha256` (env, qss_lap, handling, manoeuvres + rig, vertical_model, `__init__`, which are also imported); `DrivingEnv.env_metadata()` = EnvConfig dict, setup, track path sha256, ref name/digest, QSS cache key. Test fails if rlsetup imports an ocd_lap module covered by neither list.
- **Results:** full suite `.venv/bin/python -m pytest -q -s` → 71 passed in 18.2 s (fast suite 69 in ~4–5 s); settles 200/200 first attempt, ΣFz within 5 % 200/200, worst 2.550 %. `scripts/bench_env.py` (runs/bench/bench_env_d1_complete.json): random 1,693 decisions/s (1,189 incl. resets, reset 22.4 ms, 44 stall / 12 off_track); demo 1,698/s; flying lap f = 0.5 → 268.375 s, f = 0.7 → 192.033 s, identical to the pre-fix bench, clip 0.
- **Open finding:** `qss_profile.cache_key` hashes handling.py but not rig.py / vertical_model.py, which handling.py imports. Left unchanged, because changing the key invalidates `data/qss_profile_Shanghai_p0.npz`. Architect or Ibrahim to decide.
- **Next:** implement `docs/specs/D2_agent.md`.

## Ibrahim's instruction (26 Oct): unattended training
When D2 tests + QA + the smoke run pass, launch the full 3-seed D2 run **detached** so it keeps running offline and after the session ends. Local CPU only, no internet. Wall cap 3 h per seed until Ibrahim says otherwise. Implemented in `scripts/run_d2_full.sh` (follow-up commit 4); exact behaviour below.

**Launch** (from the repo root, because the redirect path is relative; `runs/D2/` must exist; the tracked tree must be clean, otherwise the `full` preset refuses with exit 4 for every seed):
```sh
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
scripts/run_d2_full.sh --dry-run                                     # prints the commands it would run, runs nothing
caffeinate -dimsu nohup scripts/run_d2_full.sh > runs/D2/full_stdout.log 2>&1 &
```
**Env vars** (defaults): `RUN_NAME=full1` (folder `runs/D2/$RUN_NAME/seed<k>`), `MAX_WALL_H=3`, `TOTAL_STEPS=2000000`. Example: `RUN_NAME=full2 MAX_WALL_H=4 caffeinate -dimsu nohup scripts/run_d2_full.sh > runs/D2/full2_stdout.log 2>&1 &`. `RUNS_ROOT` and `PYTHON_BIN` exist only for the tests.

**What it runs**, using `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/.venv/bin/python` and `cd` to the repo root, with no TTY needed:
1. `scripts/collect_demos.py --preset full` once (cache hit if the dataset exists; currently `data/demos/demos_1c612cde2770.npz`).
2. Seeds 0, 1, 2 in order, each decided from its folder:
   - `status.json` phase `done` (state completed / wall_clock_limit / collapsed): **skip**.
   - state `bc_gate_failed` / `demo_gate_failed`: **skip, counted as a failure** (rerunning is deterministic). Move the folder away to retry.
   - `checkpoints/last.zip` exists: **resume**, `-m rlsetup.driving.train --resume --seed k --preset full --run-name $RUN_NAME --set max_wall_h=$MAX_WALL_H --set total_timesteps=$TOTAL_STEPS`.
   - otherwise **fresh** with the same arguments minus `--resume`. A leftover folder without last.zip is first moved to `runs/D2/$RUN_NAME/_aborted/seed<k>.<timestamp>/` (never deleted). If step 1 failed, fresh starts are skipped as failures; resumes still run.
   A failing seed does not stop the next one.
3. `scripts/d2_report.py runs/D2/$RUN_NAME`.
Every seed logs timestamped start/end lines with the action, exit code and duration, followed by a summary. The script exits 1 if any step failed, else 0. Only one instance can run per run name: the lock file `runs/D2/$RUN_NAME/.run_d2_full.lock` holds the PID, and a stale lock from a dead process is taken over.

**Progress:** `tail -f runs/D2/full_stdout.log`; per seed `runs/D2/full1/seed*/{status.json,evals.jsonl,stdout.log}`.
**Stop:** `pkill -f run_d2_full.sh; pkill -f 'rlsetup.driving.train.*--run-name full1'`. The seed is resumable afterwards, like after a hard kill.

**What survives what:**
- **Terminal close / session end / logout of the shell:** survives (nohup, background).
- **Idle / display sleep:** prevented by caffeinate (`-s` prevents system sleep only on AC power).
- **Closed lid on battery = system sleep:** the run is suspended, not killed. The wall clock keeps counting towards `max_wall_h` (time.time()). Keep the Mac plugged in with the lid open.
- **Reboot, power loss, kill -9:** the script and the training die. **Recovery:** relaunch the same command. Finished seeds are skipped and the interrupted seed resumes:
  - Model: from `checkpoints/last.zip`, saved atomically at every evaluation (50k) and checkpoint (100k), so at most 50k steps are lost.
  - Replay buffer: from `replay_buffer.pkl`, saved atomically right after the demo prefill, every `replay_buffer_save_every` steps (default = `checkpoint_interval` = 100k) and at the end. It can lag last.zip by up to 50k transitions. `status.json` `replay_buffer_step` says when it was saved. *This repo since `dea9614` (not the pinned full1 run):* also saved at every evaluation together with last.zip, so it lags by at most one vector step; a larger lag adds a resume warning, and the last loaded row is marked as an episode end so n-step targets do not chain across the resume cut.
  - A kill before the first last.zip (during demos/BC/offline warm-up): fresh restart after the folder is moved to `_aborted/`.
  - last.zip present but no buffer: resume with an empty buffer. This adds a `warnings` entry and a `resume_without_replay_buffer` event to `status.json`. *This repo since `dea9614`:* no gradient updates until the buffer holds `agent.batch_size × resume_empty_buffer_min_batches` = 512 × 10 = 5,120 transitions.
  - *This repo since `dea9614` (not the pinned run):* resume compares the code with the seed's `provenance.json` (rlsetup commit + diff sha, ocd_lap code_version). A mismatch exits **5** unless `--allow-code-change`, and a dirty tree with the full preset exits **4** unless `--allow-dirty`. Either refusal adds a `resume_refused` event and changes nothing else. `run_d2_full.sh` counts the refusal as a failed seed. **So relaunch a crashed full1 from the pinned worktree (`rlsetup-wt-full`, f17be23), not from this repo.**
  - **A resumed seed gets a fresh `MAX_WALL_H` budget**: `max_wall_h` counts from the start of each process, so a crash-resume can use up to 2× the cap for that seed.
  - Rows of `train_stats.jsonl` / `probes.jsonl` written after the last save are not rolled back, so a resumed seed can have rows with steps above its resume step. `evals.jsonl` is not affected, because last.zip is saved with every eval.
- **Disk:** about 1.3 GB `replay_buffer.pkl` per seed, plus 1.3 GB transient during a save (measured 1.4 s per save), plus `step_<n>.zip` every 100k. 38 GiB was free on 26 Oct.
- **Known gap (sim-runner, D2_smoke.md), fixed in this repo (`68792bc`), still present in the pinned full1 run:** `d2_report.py` raised FileNotFoundError when a seed had no last.zip (gate-failed seed). The fixed script marks such a seed FAIL with its status reason and still reports the other seeds. If a full1 seed stops at a gate, rerun the report with this repo's script: `.venv/bin/python scripts/d2_report.py runs/D2/full1`.

## D2 implemented (26 Oct 2026, rl-coder) — supersedes "Next: implement D2" above
Spec `docs/specs/D2_agent.md`. Commits on `d1-driving-env`: `ef2c736` group 0, `0269b47` group 1, `dde334e` group 2, `4740643` group 3, `ae7c5eb` group 4, then "D2 implemented".
- **Group 0 (lead decision):** `qss_profile.cache_key` also hashes ocd_lap `rig.py` and `vertical_model.py`; `data/qss_profile_Shanghai_p0.npz` rebuilt after asserting s, v, lap time (133.91610367943798 s), ggv, traction and drag identical to the old file. SB3 / sb3-contrib pinned to 2.9.0 in pyproject; `data/demos/` ignored.
- **Code:** `rlsetup/driving/{train_config,policy,agent,vec,demos,bc,evaluate,callbacks,train}.py`, `rlsetup/runs.py`, `rlsetup/provenance.py` (`training_provenance`), `scripts/{collect_demos,d2_report}.py`; `env.py` adds `track_position` / `track_width_m` to eval-mode info (edge margin).
- **Tests:** `.venv/bin/python -m pytest -q` → **111 passed in 74.4 s** (incl. slow); fast suite `-m "not slow"` → 107 passed in 9.3 s. New: `test_agent.py`, `test_demos_bc.py`, `test_callbacks.py`, `test_report.py`, `test_d2_slow.py` (+ helper `tests/sb3_helpers.py`).
- **Mini pipeline (CLI, 2 workers, 36.6 s wall):** demos 5k (6 episodes) → BC 2 epochs → gate (fails, as expected after 2 epochs; `gate_required=False` in the test) → prefill 1,956 rows × 2 columns → offline 200 grad steps → online 3k steps (unfreeze at 1,000, 1 probe at 2,000: 107 m vs BC baseline 80 m) → evals at 0 and 3,000 → resume to 4,000 from `checkpoints/last.zip` + `replay_buffer.pkl` works → `d2_report.py` runs (criteria FAIL, no crash); deterministic re-eval of last.zip reproduces the logged deterministic return exactly (delta 0.0).
- **Demo gate finding (measured, 30k-transition samples, fresh starts):** steering noise 0.1 → 13/32 episodes clean (**0.41 < 0.80, gate fails**, all failures off_track); noise 0.05 → 19/22 clean (0.86, passes). `ensure_demos` recollects automatically once with `fallback_steer_noise = 0.05`, so the 0.1 dataset is collected and then discarded on every new dataset key. Architect may want 0.05 as the default steering noise.
- **Dataset key includes the rlsetup commit (spec §4)**, so every commit invalidates the cached demos (re-collect ~2–4 min, twice with the fallback). Collect them before the smoke run so the 20-min wall clock (counted from pipeline start) is spent on RL.
- **Deviations from the spec (all small; details in the coder report):** extra `actions_clean` array in the demo npz (BC val MSE vs the clean demo action); automatic one-shot steering-noise fallback; `init_output_bias(actor, *, pedal_deadzone, ...)` takes the deadzone flag (the actor cannot know it); `last.zip` / `best.zip` live in `checkpoints/`; extra files `probes.jsonl`, `warmup.jsonl`, `bc/throttle_vs_rear_slip.png`; resumed runs log SB3 output to `sb3/resume<k>/` (SB3's CSV logger truncates `progress.csv`); resume may override only `total_timesteps`, `max_wall_h`, `save_replay_buffer`, `checkpoint_interval`; the BC gate's 3 stochastic laps use the 1-lap eval env; `full` refuses only on tracked uncommitted changes (untracked `.codex/`, `AGENTS.md` are not ours); full preset `max_wall_h = None` until Ibrahim answers D2 question 2.
- **Unmeasured risk:** a full evaluation (2-lap deterministic + 10 stochastic laps) is ~40–80k decisions (~35–70 s, estimate); at ~330 decisions/s training that is 20–45 % overhead at a 50k interval. sim-runner should measure it in the smoke run (`wall_eval_s` vs `wall_train_s` in evals.jsonl).

### D2 follow-ups (26 Oct, rl-coder; lead decisions 1-2): these supersede the matching bullets above
- **Demo steering noise default 0.05** (`DemoDataConfig.noise_std = (0.05, 0.1, 0.1)`). The fallback is kept with `fallback_steer_noise = 0.025` (PLACEHOLDER, unmeasured) and only runs if 0.05 fails the gate. Smoke collection at 0.05 gave clean 0.887 and the fallback was not needed.
- **Demo dataset key** = sha256 of (EnvConfig, DemoDataConfig without workers, sha256 of `rlsetup/driving/{env,config,demo_driver,reference,qss_profile,demos}.py`, ocd_lap code_version, QSS cache-key sha). The rlsetup commit is no longer in the key, so commits that don't touch those files keep the cache (verified: `collect_demos.py --preset full` at `08abe04` hits `demos_1c612cde2770`).
- **Crash-safe resume:** `TrainConfig.replay_buffer_save_every` (None = `checkpoint_interval`, 0 = end only; also overridable on resume). The buffer is also saved after the prefill. All model/buffer saves are atomic. Resume without a buffer gives an empty buffer plus a warning instead of FileNotFoundError.
- **`scripts/run_d2_full.sh`**: see the unattended-training section above.

### Smoke-run commands (sim-runner)
```sh
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
.venv/bin/python scripts/collect_demos.py --preset smoke            # cache the demo dataset first (gate + fallback)
.venv/bin/python -m rlsetup.driving.train --preset smoke --seed 0 --run-name smoke1
.venv/bin/python scripts/d2_report.py runs/D2/smoke1                 # expect criteria FAIL, no crash
.venv/bin/python -m rlsetup.driving.train --preset smoke --resume --seed 0 --run-name smoke1 --set max_wall_h=0.05   # resume check (~3 min)
```
Run folder: `runs/D2/smoke1/seed0/` (status.json, evals.jsonl, train_stats.jsonl, probes.jsonl, warmup.jsonl, sb3/progress.csv, checkpoints/, bc/, laps/, stdout.log). Smoke pass criteria: spec §11 step 1.

## NEXT ACTION (as of D2 implemented)
sim-runner runs the smoke preset (commands above), then qa-reviewer reviews D2.

## FULL D2 RUN LAUNCHED (26 Oct 10:54 EET) — running unattended
- Code pinned: detached worktree `/Users/ibrahimkhaled/Documents/rlsetup-wt-full` at **f17be23** (its `.venv` is a symlink to this repo's venv). Edits in this repo do NOT affect the run.
- Command: `cd rlsetup-wt-full && RUNS_ROOT=<this repo>/runs/D2 nohup caffeinate -dimsu ./scripts/run_d2_full.sh > runs/D2/full_stdout.log &` (seeds 0,1,2 sequential, max_wall_h 3, total_timesteps 2e6, then d2_report).
- Results: `runs/D2/full1/seed{0,1,2}/` (evals.jsonl, status.json, checkpoints/), log `runs/D2/full_stdout.log`, report `runs/D2/full1/d2_report.md` at the end (~9 h worst case).
- Check: `tail runs/D2/full_stdout.log`; `pgrep -fl run_d2_full`. Stop: `pkill -f run_d2_full.sh; pkill -f rlsetup.driving.train`.
- Smoke result (smoke1_bc30, 148k steps, ~22 min, CPU-contended): BC lap 2 340.6 s → final deterministic lap 2 **130.8 s clean** (< 140.6 target, < 133.9 QSS). One lap only; intermediate evals 25k–125k went off track. Verify the sub-QSS lap (track usage vs safe line) before trusting it.
- Known issues in the pinned code (f17be23). **All are now fixed in this repo but NOT in the running full1**; details in the next section. QA M1: full1's `q_vs_mc` has only the per-step soft MC, which is biased by about +25 at α = 0.1, log π = 3, so do not read q_min_ratio > 1 as overestimation without correcting for this. M2/m1/m3/m4: resume provenance and resume edge cases; these matter only if a seed crashes and resumes. m2: d2_report crashes if a seed has no last.zip; if so, rerun the report with this repo's script. m6 and the provenance digest are cosmetic for full1.

### D2 QA fixes (26 Oct, rl-coder): in this repo only, NOT in the pinned full1 run
Source: `docs/handoff/D2_qa_review.md` + the provenance finding in `docs/handoff/D2_smoke.md`. Commits on `d1-driving-env`:
- `68792bc` **m2:** `scripts/d2_report.py` marks a seed as FAIL with `fail_reason` = its status state + stop_reason. This covers a seed stopped at the demo/BC gate, or any seed without `checkpoints/last.zip` when re-evaluating; the other seeds are still re-evaluated. A missing provenance.json is a consistency problem instead of a crash. Checked on a copy of `runs/D2/smoke1`: `seed0: FAIL (state bc_gate_failed: BC deterministic lap: off_track)`, exit 0, report written.
- `632c05d` **M1 (lead decision: log both):** `q_vs_mc` / `summarize_q_mc` report the per-step soft MC (`mc`, unchanged keys) **and** the n-step-consistent soft MC (`mc_nstep`, keys `*_nstep`, `n_steps`). The n-step MC puts entropy only at offsets n, 2n, …, which is the fixed point of SB3 2.9.0's n-step critic target. This lands in evals.jsonl `q_vs_mc` and warmup.jsonl `q_vs_mc_demo`. For over/underestimation, compare Q with `mc_nstep`. Test: brute-force fixed point built from a real `NStepReplayBuffer` on a synthetic chain (n = 1, 2, 3).
- `dea9614` **Resume robustness:**
  - **M2:** code check and refusal (exit 5 / 4, flags `--allow-code-change`, `--allow-dirty`). The `resume` event records `code`, `code_mismatches` and both flags; an allowed mismatch also adds a `warnings` entry.
  - **m1:** after `load_replay_buffer`, row `(pos − 1) % size` of every column gets done = timeout = 1, so the target bootstraps from that row's own next_obs. A true termination there stays terminal; the QA's literal fix ("set both to 1") would have turned it into a bootstrap.
  - **m3:** CollapseGuard saves last.zip atomically right after a rollback, before the event. A crash between the two allows at most one extra rollback.
  - **m4:** the buffer is saved at every eval, once per step, shared with the checkpoint callback. The resume event gets `replay_buffer_lag` and a warning when `replay_buffer_step < num_timesteps − n_envs`.
  - **Empty-buffer resume:** `StagedSAC.min_train_transitions` gate; new config field `TrainConfig.resume_empty_buffer_min_batches = 10`.
- `66c3b51` **m6 + provenance:** `EvalConfig.stats_slow_mps / stats_wide_m / stats_low_v0_mps` (15, 3, 5) replace TrainStatsCallback's constants. `training_provenance` and the resume check pass `track.name` to ocd_lap's `capture_provenance`. New runs therefore record `ocd_lap_code_version` **73d2c1637f6c** (with track CSVs) instead of c1cdd4cae7cc, which full1 records. The demo dataset key is unchanged (demos.py still calls `run_provenance()` without a track).
- `13e1e17` **Mutation gaps:** tests for A3, B1–B3, E1, E2, E4, E7, E8, C1, T3, R1–R3, R5, V1. All 16 QA mutants plus 15 mutants of the fixes above are killed by the fast suite (QA patterns re-applied one at a time to a throwaway copy).
- Not done: m5 (lap-2 tagging at decision granularity, document only) and m7 (`run_d2_full.sh` does not continue seeds stopped by the wall-clock limit) were not in this fix list.
- Tests (26 Oct, during the full1 run, fast suite only): `.venv/bin/python -m pytest -q -m "not slow"` → 131 passed, 4 deselected in 15.8 s. The slow suite (`test_d2_slow.py`, CLI mini pipeline + resume + report) was **not** run, to leave the CPU to full1. Run it after full1 finishes.

## 26 Oct 12:20 — Ibrahim: only seed 0 (13:30 stop CANCELLED 12:22, see below)
Launcher `run_d2_full.sh` killed (seeds 1, 2 will NOT run). Seed 0 training (PID 7154) kept alive under `caffeinate -w`. `runs/D2/stop_seed0_at.sh` (detached, log `runs/D2/stop_seed0.log`) stops it right after the first evaluation at/after 13:30 (SIGINT, then SIGTERM after 60 s), then refreshes `docs/results/D2_full1/SEEDS.md`. Driver to use: `runs/D2/full1/seed0/checkpoints/best.zip` (best clean deterministic lap). Ibrahim mentioned ~90 s laps: not reachable for this OCD car (QSS 133.9 s; real SF70H 1:31.9 needs SF70H tyres/aero).

## 26 Oct 12:22 — Ibrahim: train seed 0 until a clean lap ≤ 95 s
`runs/D2/stop_at_lap.sh` (detached; log `runs/D2/stop_at_lap.log`) replaces the 13:30 stopper. It stops seed 0 once evals.jsonl has a clean deterministic lap 2 ≤ 95 s. If the run first ends on its built-in caps (2M steps / 3 h, ~13:55), it resumes ONCE from the pinned worktree with total_timesteps 6M and max_wall_h 8, then stops when that ends. Lead's assessment, told to Ibrahim: 95 s is not reachable for this OCD car (QSS 133.9 s; tyre/aero limited; needs SF70H tyre + aero data), so in practice the run ends on the caps or when Ibrahim stops it: `pkill -f stop_at_lap.sh; pkill -f rlsetup.driving.train`.

## 26 Oct 12:55 — Ibrahim: seed 0 trains until 14:15 (CHANGED to 16:00, see below)
`runs/D2/stop_at_lap.sh` rewritten: stop right after the first evaluation at/after 14:15, or earlier on a clean lap 2 ≤ 95 s. The run's own 3 h wall cap ends it ~13:55 (started 10:54:55); the watcher then resumes it from the pinned worktree (`--resume --set total_timesteps=6000000 --set max_wall_h=8`) and keeps watching until 14:15. Log: `runs/D2/stop_at_lap.log`. Dashboard: TensorBoard on 127.0.0.1:6006 (`runs/D2/refresh_seed0.sh` mirrors evals.jsonl every 5 min, append-only).

## 26 Oct — Ibrahim: deadline moved to 16:00
Same watcher (`runs/D2/stop_at_lap.sh`), DEADLINE=1600: stop right after the first evaluation at/after 16:00 (or a clean lap 2 <= 95 s); resumes after the ~13:55 wall cap.


## 26 Oct 22:11 — R1 run `r1_warm` launched (DECISIONS #19, #20)
- Spec: `docs/specs/R1_time_optimal.md` (+ Amendment 1). Coder: `docs/handoff/R1_coder.md`. QA: `docs/handoff/R1_qa_review.md`.
- Run dir `runs/D2/r1_warm/seed0`, stdout `runs/D2/r1_warm_stdout.log`, TB refresher `runs/D2/refresh_r1.sh`.
- TensorBoard (127.0.0.1:6006) runs: `full1_old_reward` vs `r1_warm_new_reward`.
- Command: `caffeinate -dimsu nohup .venv/bin/python -m rlsetup.driving.train --preset r1_warm --seed 0 --run-name r1_warm --init-from runs/D2/full1/seed0/checkpoints/best.zip --set warm.load_optimizers=False --set warm.actor_lr=0.0001 --set guard.threshold=0.65 --set stop.plateau=False --set total_timesteps=3000000 --set max_wall_h=4.0`
- Resume after a crash: same command with `--resume` (no `--init-from` re-application).
- 1.0M check (DECISIONS #20). Open for Ibrahim: setup loss = lap time? z_drv dropped/deferred? seeds 1–2 from scratch or warm?
- Follow-up: `StopConfig.start_step` so plateau ignores pre-unfreeze evals.
- **1.0M check (23:36):** stop rule not met (post-unfreeze best clean det lap 2 = 123.04 s < 124.0; sto clean last 5 = 0.36), so the run continues to 3M / 4 h. Not faster yet: det clean in 7/20 post-unfreeze evals (123.04–124.29 s), sto mean 125.0–126.2 s (init 124.77), 1 rollback at 160k. Line freed: mean |e| 0.43 → 0.6–0.8 m, max 2.4 → 3–6 m, min edge margin 0.35 → 0.12–0.25 m; lap smoothness 10 → 22–39; progress_jumps 0.

## 27 Oct — Windows RTX workstation; driving agent switched to the `sf70h_tum` car
Machine: HP Z2 G9, i9-14900 (32 threads), 64 GB, RTX 4000 Ada 20 GB. Not admin, no WSL/MSVC.
- **Environment:** conda env `C:\Users\imk14\ocdenv` (Python 3.12, torch 2.11 cu128, SB3 2.9.0, sb3-contrib 2.9.0, gymnasium 1.4.0, tensorboard, pytest + ocd_first_lap requirements). ocd_native built with WinLibs MinGW GCC 16 + conda cmake/ninja/eigen/boost headers (ocd_first_lap `3945634`). `PYTHONPATH=C:\Users\imk14\Documents\ocd_first_lap` (branch `sf70h-tum-car`, which already has the rear-axle speed patch and `max_brake_torque_Nm`).
- **Code:** `5c90818` `TrainConfig.device` (default `cpu`; `cuda` on this box; BC stays on the CPU). `857706b` `EnvConfig.car` / `ers` (default `av21`, unchanged): backend, track-limit footprint, demo-driver pure-pursuit geometry and the QSS profile follow the car. Non-AV-21 profiles come from `qss_lap.lap_time` (DRS off: OCDBackend has none) in `data/qss_profile_Shanghai_p0_sf70h_tum.npz`; the AV-21 cache key and file are unchanged.
- **ERS is off** for `sf70h_tum` runs: OCDBackend's 4 MJ is per lap, and DrivingEnv's random-start, multi-lap episodes neither refill nor observe it. QSS reference without ERS/DRS: **104.78 s** (AV-21: 133.9 s; TUM laptime-sim with ERS + DRS: 91.24 s).
- **BC gate on sf70h_tum:** demos at the default `fraction_range` (0.35, 0.65) gave a BC driver at ~9 % throttle that stalled in T1–T2 (gate fail). With `demos.fraction_range=(0.5,0.8)` the noise-0.05 set is 66 % clean, the automatic 0.025-steering-noise fallback 92 % clean, and the BC gate passes (det lap 2 178.1 s).
- **Run `sf70h_win`** (runs/D2/sf70h_win/seed0), launched 27 Oct 17:04:
  `python -X faulthandler -m rlsetup.driving.train --preset full --seed 0 --run-name sf70h_win --set env.car=sf70h_tum --set "demos.fraction_range=(0.5,0.8)" --set total_timesteps=1000000000 --set device=cuda --set vec_start_method=spawn --set n_envs=16 --set gradient_steps=8 --set torch_threads=2` (UTD 0.5 as on the Mac; ~160 decisions/s, learner-bound).
  Ibrahim's rule: train until a **clean deterministic lap 2 < 95 s**, else indefinitely. A detached PowerShell watcher (log `runs/D2/sf70h_win_watch.log`) stops it on that lap and resumes from last.zip after a process crash. First evals: step 0 178.12 s, 50k off track, **100k 117.68 s** (10/10 stochastic clean, best 120.78 s).
- **Windows notes:** start long jobs with PowerShell `Start-Process -WindowStyle Hidden`; children of an interrupted Bash tool call die. `forkserver` does not exist on Windows: `--set vec_start_method=spawn`. TensorBoard: `runs/D2`, lap metrics mirrored every 2 min by a loop running `scripts/evals_to_tb.py runs/D2/sf70h_win/seed0`.
- **Hardware instability (not our code):** 4 bluescreens in 5 days (0xF7, 0x20001, 0x1E, 0x3B), 5 unexpected shutdowns in 30 days, and random access violations inside numpy, also in an unrelated Python 3.9 on 23 Oct. Consistent with the known i9-13th/14th-gen degradation (microcode is already 0x12F). 480 isolated evaluation laps (CPU and CUDA) did not crash. A bluescreen kills training and the watcher; relaunch by hand after logging in. Raised with Ibrahim (AUB IT / HP).
- **Not yet done:** the Mac/Windows split into per-platform folders, Windows tuning measurements, and the 7 Windows-only test failures (they also fail on the unmodified code: provenance x2, QSS cache key, R1 legacy rollout golden, run_d2_full.sh x3).
