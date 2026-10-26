# R1 QA review: qa-reviewer, 26 Oct 2026, commit 475bbe2 vs docs/specs/R1_time_optimal.md §1–§8

Reviewed the committed code only (`git archive 475bbe2`). The lead saved this from the reviewer's reply.

**Verdict:** no BLOCKER. The reward is implemented as specified. ref_tracking is bit-identical to the pre-R1 code in training and in evaluation mode. There is one MAJOR finding (the timing of the critic check, which matters under Amendment 1) and eight MINOR ones.

## Tests
The fast suite on an extracted 475bbe2 tree gave 156 passed and 4 failed. All 4 failures come from the extracted copy (no `.git`, no `.venv`), not from R1 code; the same tests pass in the real repo (160 passed). The slow tests were not run.

## Verified by running code
1. **Golden hash:** `tests/r1_golden.legacy_rollout` gives the same hash `0007e2813d2d…c4c2` on the 47f1d8d tree and the 475bbe2 tree. The episode ends were off_track, stall, time_limit, wrong_way, off_track, stall.
2. **Legacy evaluation path:** an eval-mode rollout (lap_complete with lap splitting, then a wrong-way end) gives the same hash `35efc61d…` on both trees.
3. **Non-finite state under time_pbrs:** terminated, P = 100, Φ′ = 0, F = −Φ(s), obs all zero, r = progress − time − P + F exactly. The behaviour is correct, but no test covers it.
4. **SB3 2.9.0 `NStepReplayBuffer`:**
   - the n-step window stops at the first done or timeout;
   - the discount is γ^m;
   - the target is zeroed only on done without timeout;
   - row pos−1 is marked as a timeout.

   Given this, the shaping sum telescopes exactly.
5. **`CallbackList`** calls every callback even after one returns False.

## Verified by reading
- **Per-tick reward:** r = a·Δs − b·Δt − [terminal failure]·P.
- **Shaping:** added once per decision, using `rw.gamma`, which is the learner's γ.
- **Φ′:**
  - Φ′ = 0 only for terminal failures.
  - Φ′ is the real value for stall, time limit and lap_complete.
  - `_phi` and `_episode_shaping` are reset in `reset()`.
- **Stall under R1:** treated as a truncation, so `TimeLimit.truncated` is set.
- **Clipping:** shaping uses the true e. The clip at ±50 m is far above the 14.1 m maximum on track.
- **Lap costs:** `laps['cost']` stays the legacy cost; `cost_r1` = time.
- **Stall bookkeeping:** evaluate.py's `failed` treats an R1 stall as a truncation, and TrainStats counts it as 'stall'.
- **warm_start:** the checks, the in-place copy of log_α, the fresh optimizers and the freeze all match the spec.
- **Dirty check:** uses `strict_provenance` and looks at tracked files only.
- **`--init-from` and resume:**
  - `--init-from` is required iff init_mode is 'warm'.
  - On resume, frozen_phase covers 'bc' and 'warm'.
  - The plateau history and the critic-check flag are rebuilt from evals.jsonl.
  - init_from is not re-applied.
- **Stop-reason precedence:** collapsed > plateau > wall clock.
- **Determinism:** the same seed gives the same step-0 lap.
- **Provenance:** complete.

## MAJOR
**M1. The critic check runs at the first scheduled evaluation at or after unfreeze, not at unfreeze** (callbacks.py:360, train.py:381).
- Under Amendment 1 (critic phase 20k, evaluations every 50k), the check would land 30k steps after unfreeze, so S3 would measure a critic and an actor that are already moving.
- **Fix:** DrivingEvalCallback schedules an extra evaluation at `critic_check_at` (next_eval = min(next multiple of the interval, critic_check_at if it is not done yet and > n)).
- **Test:** with interval=50k and critic_check_at=20k, a record exists at step 20,000 and carries `critic_check`.

## MINOR
- **m1. A NaN critic check passes silently** (callbacks.py:364, 387–396), because `nan > tol` is False. Fix: treat a non-finite value as not converged, and have `critic_check_value` return None for non-finite input. Add a test.
- **m2. An InitMismatch leaves a run folder that blocks relaunching** (train.py:264 vs 367–371). Fix: before `make_run_dir`, check that the init file and its `parents[1]/config.json` exist, or allow reusing a folder whose state is `init_mismatch`.
- **m3. PlateauStop's stochastic-mean best is noisy** (callbacks.py:401, evaluate.py:390): it averages clean laps only. This is conservative; it can only delay a stop. Optional fix: require `sto_clean >= 5`, or use the median.
- **m4. The golden hash covers training mode only.** Fix: add an evaluation-mode episode and regenerate the hash on a `git archive 47f1d8d` tree.
- **m5. Untested paths:**
  - (a) non-finite state under time_pbrs (exact asserts as in item 3 above);
  - (b) warm_start refusals for action space, n_critics, critic LayerNorm and policy class;
  - (c) R1 stall through SubprocVecEnv.
- **m6. `assert` in production code** (agent.py:197). It disappears under `-O`; raise InitMismatch instead.
- **m7. warm_start does not compare target_entropy, n_steps, tau or activation_fn.** This is safe today. Optional fix: compare `target_entropy` and `n_steps`.
- **m8. Magic numbers in `progress_jumps`** (evaluate.py:198). Move them to EvalConfig.

## Not issues (checked)
- A stall in the same tick as the time limit is reported as 'stall'.
- Legacy `termination` is unchanged.
- The step-0 evaluation writes best.zip.
- PlateauStop's position in the callback order is correct.
- `lifetime_timesteps` is correct.
