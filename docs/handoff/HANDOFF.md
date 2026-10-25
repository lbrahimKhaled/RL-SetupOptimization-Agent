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
