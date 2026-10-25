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
| QA review of D1 | `docs/handoff/D1_qa_review.md` (if it finished) |
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
