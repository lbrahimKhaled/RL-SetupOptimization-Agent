# R6 QA report (21 Nov 2026)

**Verdict: NO-GO for the 4M launch as configured.** No implementation bug found; the code matches
the spec. Blocker is a runtime result: `bcn_f1_r6_smoke` stopped itself at the throttle gate
before `learn()`, as specified for a failed gate. The 4M preset uses the same offline settings for
10x more grad steps and will almost certainly fail the same gate. Changing those settings is a
decision for Ibrahim (with the architect), not a code fix.

## Tests
- Full suite under ocd_first_lap-f1: 1 failed, 856 passed, 4 skipped (279 s).
- The one failure, `tests/test_qss_profile.py::test_cached_file_matches_fresh_computation`, is
  unrelated (see check 6).

## BLOCKER
**B1. Smoke failed the throttle gate; the 4M run would stop the same way.** (Spec §2.2; preset
`rlsetup/driving/train_config.py:691`; logs `runs/mac/D2/bcn_f1_r6_smoke/seed0/status.json`,
`warmup.jsonl`.)
- `state: throttle_gate_failed`, `gate 0.0193 > tol 0.01`, 07:47:52 UTC.
- Gate per block: 0.0006 before training, then 0.0032, 0.0066, 0.0100, 0.0137, 0.0193 (rising
  every block and speeding up).
- Over the same 5 blocks: critic loss 52.5 -> 77.9, mean Q on the fixed batch 220.2 -> 211.7.
- TensorBoard confirms critic lr 3e-5 and lr_scale 0.1 during this phase, as specified.
- The 4M preset runs 50 blocks at the same lr.

Inference (not proven): before any training the gate is 0.06%, so the loaded 7M critic already
gives nearly the same Q for relabelled and original rows. Mean Q of random rows falls about as fast
as the gate grows, which looks like general drift of the converged critic when refit offline at 10x
its 7M lr, not damage from the relabel. As specified, the gate cannot tell these apart.

Diagnostic: the same 5k offline grad steps at scale 0.1 with no remap/relabel, same metric on the
same 4,096 rows; repeat at scale 0.01. If the no-relabel control climbs the same way, the gate
measures lr-driven drift.

Options (Ibrahim's choice): (a) offline lr scale 0.01 (3e-6); (b) far fewer or zero offline grad
steps, relying on the 100k online critic phase; (c) keep 3e-5 but define the gate relative to the
no-relabel control (spec change + architect review). After the decision, rerun the smoke: the later
smoke checks (lr_scale 0.01 at learn start, retries, crash factor 3 from step 0, unfreeze at 20k)
never ran.

## Checks
1. **Throttle map and mirrors (env, settle driver, DemoPolicy, bc, evaluate): correct.**
   pedal(0.70) = 0.8, pedal(nextafter 0.70) = 0.8, pedal(0.85) = pedal(1) = 1.0; non-decreasing on a
   2M grid; slope 4/3 above the knee. Below the knee bitwise equal to the R3 map; brake and steering
   bitwise unchanged. Round trips within 1.1e-16 (4e-8 via float32); pedal 0 -> -0.8. Invalid (k, s),
   deadzone off and `ref_tracking` all raise. `_pedals` and `bc.pedal` match `throttle_pedal`; env
   pedal equals eval pedal on 20k actions. Flag off bit-identical to HEAD: 0 mismatches on a 20k grid
   against a `git archive HEAD` copy; golden bcn_f1_r5 env stream regenerated with HEAD code gives
   `d0ffef43…c4c1`, same as the new code.
2. **Buffer relabel: exact; test edits did not weaken the spec.** Real buffer: obs slots
   [133, 145, 157], 496,060 action cells, max |dpedal| 3.18e-8 (abort 1e-6). `test_r6.py`
   tolerances still 1e-5 for obs/next_obs/rewards, 1e-6 for settle slots, bitwise for policy slots.
   Moving the bcn rollout to 60 m/s is justified: with the unchanged map a 1-ulp throttle change moves
   the obs by 1.04 (40 m/s start) and 1.47 (35 m/s) within 100 decisions, but 1.2e-7 at 60 m/s.
   Expected side effect: retry/settle hand-over states differ from R5's at float32 level and diverge
   at low speed; the spec accepts this.
3. **Exit-bonus refusal change (`agent.py:371`): correct and safe for R2/R3 buffers.** An R2/R3
   source (mode 'exit') vs an entry_exit target is still refused (tested); different B_in/B_out is
   refused (tested); the exit-to-exit path does not touch this check. For R6,
   `c.env.reward == r5.env.reward` (tested) and R5/R6 share the QSS profile (key `6e00b6fe…`,
   84.738 s, ocd commit `ee86683`).
4. **Lifetime schedules: correct.** With the 7M offset: lr scale 0.01, retries on (start 500k), crash
   factor 3 (start 500k). Offline 3e-5 confirmed in TensorBoard. Offset survives a resume (from
   status.json) and is 0 with the flag off. The 0.1 -> 0.01 hand-over at learn start is only
   unit-tested (m3).
5. **AC refusal (`actor.py`): correct.** R6 configs raise; R5 configs (no key or None) pass.
   `--critic-init` must be the bundle's own source checkpoint (sha-checked) and goes through
   `_config`, so an R6 critic cannot slip in. Running AC servers loaded the old module and are
   unaffected until restart.
6. **qss_profile failure unrelated to R6.** Fails on HEAD too. The test, `qss_profile.py`, conftest
   and `data/` are unchanged vs HEAD. Cause: `data/qss_profile_Shanghai_p0.npz` (26 Oct) was hashed
   against older ocd_lap sources (backend, handling, manoeuvres, powertrain, qss_lap, rig). Fix:
   rebuild with `scripts/build_qss_profile.py`.

## MINOR
- **m1** `rlsetup/driving/demo_driver.py:74`: with the remap on and a float32 action, the
  below-knee throttle is computed in float64; the old path is float32 (numpy 2), so results differ by
  float32 rounding (9,454 of 20k). Only in-repo caller `env.to_inputs` passes float64, so no effect
  today. Fix: use the R3 lambda when `a[1] <= k`, or document float64.
- **m2** `agent.py:371`: the new pass-through accepts the buffer when corner_bonus settings are
  equal but never checks `src_env.reward == cfg.env.reward`; `warm_start` compares only gamma. A
  future warm start with a changed line or exit_k would silently train on stale rewards. Fix: also
  require equal reward blocks there, plus a test.
- **m3** Missing test: tiny warm run with `throttle_relabel`, `lifetime_schedules`, offline scale 0.1,
  gate tol 1.0; assert critic lr 3e-5 offline, 3e-6 after the first train() in learn(), retries on
  at step 0, crash factor 3.
- **m4** Commit `tests/test_r6.py`, `tests/r6_golden.py`, `scripts/r6_prelaunch.py` with the change;
  `strict_provenance` requires the commit before the 4M run.
- Info: the demo-dataset key hashes `env.py`, `config.py`, `demo_driver.py`, so the next from-scratch
  BC run recollects demos. R6 unaffected.

Provenance and determinism: config, rlsetup commit and diff sha, ocd_first_lap commit, QSS key and
relabel stats are recorded; the gate batch is seeded with `cfg.seed`; the golden hash reproduces.

## 4M run collapse (21 Nov 2026, 18:02) and diagnosis
- `bcn_f1_r6` (offline phase removed, gate 0.0006 passed) collapsed at 190k (rollback 180k, collapse guard
  ~520 m vs 1433 m baseline); keeper resume collapsed again at 200k; keeper stopped.
- Cause: the 100k online critic-only phase. With the actor frozen, Q on the deterministic lap fell
  236 -> 190 (50k) -> 23 (100k) while MC stayed 235.6; critic loss 52 -> 283; twin-critic spread
  (Q_mean - Q_min) 0.33 -> 6.9. The collapse is uniform over throttle 0..1 and over old and new buffer rows.
- Control `diag_r5_frozen100k` (R5 map, no remap/relabel, same 100k freeze): Q 236 -> 228 -> 183 -> 116 -> 45,
  spread 0.31 -> 6.2. **The throttle remap is not the cause; a 100k frozen-actor phase is.**
- Smoke (`bcn_f1_r6_smoke2`, 20k freeze) is identical to the 4M run up to 18k; it unfroze at 20k, the spread
  stayed 0.5-0.9 and Q tracked MC (+-7) for 130k steps. R3/R4/R5 also used a 20k freeze (R1 Amendment 1 cut 100k
  -> 20k); the 100k phase had never run before.
- Likely mechanism (not proven): with a fixed policy nothing offsets clipped double-Q underestimation; the
  twin critics drift apart and min() pulls every bootstrap target down, accelerating.
