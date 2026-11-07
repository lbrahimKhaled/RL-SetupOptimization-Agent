# R2 QA review — commit c2791ea vs 6e97b40 (qa-reviewer, 7 Nov 2026; saved by the lead)

**Verdict: no BLOCKER; one MAJOR (spec-level, Ibrahim decides); MINORs below.**

## Tests (re-run by QA)
- Default path: 339 passed, 10 skipped, 2 failed (183 s). f1 path: 345 passed, 4 skipped, 2 failed (285 s, incl. test_r2_slow).
- The 2 failures — `test_provenance::test_every_imported_ocd_lap_module_is_hashed` ({'ocd_lap/cars.py'}) and `test_qss_profile::test_cached_file_matches_fresh_computation` (KeyError 'provenance') — fail identically at 6e97b40 (git archive extract, both PYTHONPATH variants): pre-existing.

## Verified
1. `buffers.py` `_get_samples` vs SB3 2.9.0 (normalised diff): only signature annotations + the 2-line bonus block (buffers.py:59-60) differ; float32 throughout; w = 0 bit-identical by construction and by test; `adopt()` in place (id() tested); pickle round trip; `check_sb3_version` in build_model (agent.py:132).
2. CornerTracker: once per pass; forfeit only on terminal failure in the paying decision (env.py:599), stall/time-limit don't forfeit; start inside a window paid; tracker reset at hand-over (env.py:445); exit-speed interpolation; multi-lap eval labelling.
3. Env reward bit-identical with the bonus on (tests + r1 golden unchanged); bonus only via info → buffer; q_vs_mc uses r + w·bonus; det_bonus_sum separate.
4. Schedule + Amendment-1 trigger: w(t); callback order; trigger condition; bonus_state.json; stop at report_from (`report_complete`); resume after trigger; best.zip only on reportable evals, best_aided.zip otherwise; `scripts/keeper_check.py`; local `runs/mac/keeper_f1.sh` uses it and never resumes a self-stopped run.
5. Focused starts never on eval/probe/explicit starts; fraction 0.25 ± 0.015 (N = 10,000); fallback reachable; weights/decay/floor; seed 166 crashes / 11 H0 bins; focus_state.json; 108 bins both sides. Not exercised end to end (push across SubprocVecEnv workers, S4 share) → sim-runner smoke.
6. Low-noise env 3: other envs untouched, no extra RNG when off; stored = executed action; scale 0 = deterministic; 0.3σ empirical.
7. Backward compatibility: aid blocks omitted at default; `_env_without_reward` tested on f1_best_w35's config; relabel refusal before loading the buffer.
8. Determinism (two env instances, aids on, same seeds/actions → identical hash over 6 episodes); numerical guards; provenance `r2` block; committed corner table c83c1c689508 is a cache hit, matches R2-2.7 within 1 m / 0.1 m/s.

## Findings
- **MAJOR (spec conflict, Ibrahim decides before launch):** after an Amendment-1 trigger the run stops at report_from with exactly one reportable eval; resume and keeper refuse to continue → R2-12.8 "median of the last 5 clean reportable lap 2" impossible. Options: (a) run more evals past report_from; (b) last 5 clean evals of [trigger, report_from]; (c) report the single lap.
- **MINOR — keeper pattern** (`runs/mac/keeper_f1.sh`, local): `PAT="rlsetup.driving.train .*--run-name $RUN"` also matches `--run-name f1_r2_smoke` → keeper thinks f1_r2 alive / pkill kills the other run. Fix: `--run-name $RUN( |$)`. Same for the launch-recipe pgrep.
- **MINOR — corners.py:281-282, 288:** hand-over exactly at exit_m → unit never paid for the whole episode (0 payments over 12 km by script). Fix: `self.next[self.next <= 0.] += self.L` in reset (or high = −inf). Probability ~0.
- **MINOR — demo keys:** DEMO_SOURCES hash env.py/config.py (demos.py:49-50, 65-69) → every preset's demo key changed; next fresh `bc` run recollects demos once (time only). Log under DECISIONS #15.
- **MINOR (pre-R2):** `ocd_lap/cars.py` not in QSS `cache_key` `_SOURCES` (qss_profile.py:33-34) nor provenance → corner-table key inherits the blind spot.
- **MINOR — clarity:** 95 s limit in two places (`keeper_check.LIMIT_S`, `BonusScheduleConfig.trigger_lap2_s`).
- Observation: backward projection jumps delay the next exit payment (only where progress_jumps ≠ 0; same weakness as the progress reward).

## Tests to add
1. `test_tracker_handover_exactly_at_exit` (fails today). 2. `test_env_stall_in_paying_decision_does_not_forfeit`. 3. `test_learn_and_finish_report_complete`. 4. `test_keeper_pattern_does_not_match_prefix`. 5. `test_focus_weights_pushed_to_subproc_workers`. 6. `test_bonus_buffer_weighted_wrapped_multi_env`.
