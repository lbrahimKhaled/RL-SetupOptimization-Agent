# HANDOFF — read this first to resume (last updated 25 Oct 2026)

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
  - Replay buffer: from `replay_buffer.pkl`, saved atomically right after the demo prefill, every `replay_buffer_save_every` steps (default = `checkpoint_interval` = 100k) and at the end. It can lag last.zip by up to 50k transitions. `status.json` `replay_buffer_step` says when it was saved.
  - A kill before the first last.zip (during demos/BC/offline warm-up): fresh restart after the folder is moved to `_aborted/`.
  - last.zip present but no buffer: resume with an empty buffer. This adds a `warnings` entry and a `resume_without_replay_buffer` event to `status.json`.
  - **A resumed seed gets a fresh `MAX_WALL_H` budget**: `max_wall_h` counts from the start of each process, so a crash-resume can use up to 2× the cap for that seed.
  - Rows of `train_stats.jsonl` / `probes.jsonl` written after the last save are not rolled back, so a resumed seed can have rows with steps above its resume step. `evals.jsonl` is not affected, because last.zip is saved with every eval.
- **Disk:** about 1.3 GB `replay_buffer.pkl` per seed, plus 1.3 GB transient during a save (measured 1.4 s per save), plus `step_<n>.zip` every 100k. 38 GiB was free on 26 Oct.
- **Known gap (sim-runner, D2_smoke.md):** `d2_report.py` raises FileNotFoundError when a seed has no last.zip (gate-failed seed). The script records the report's non-zero exit; the seeds are unaffected.

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
- Known issues in the pinned code (fixed later in this repo, not in the run): QA M1 (Q-vs-MC diagnostic biased ~+25 by n-step entropy), M2/m1/m3/m4 (resume provenance + resume edge cases — only matter if a seed crashes and resumes), m2 (d2_report crashes if a seed has no last.zip → rerun report with fixed code if so).
