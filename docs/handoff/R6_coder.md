# R6 top-end throttle remap: coder report (rl-coder, 20-21 Nov 2026)

Spec: `docs/specs/R6_throttle_remap.md` (architect APPROVE). Ibrahim's call on the one open point: offline critic
lr 3e-5 (scale 0.1). Nothing is committed and no training was launched. ocd_first_lap is untouched. The only change
under `rlsetup/assetto` is the refusal in `actor.py`.

Status: implementation done; full suite green apart from one test that already fails on HEAD; pre-launch items done.

## 0. Follow-up 21 Nov 2026 (after the failed smoke gate and QA, `docs/handoff/R6_qa.md`)
- **Offline critic phase removed (Ibrahim):** `bcn_f1_r6` / `bcn_f1_r6_smoke` now use `offline_critic_grad_steps=0`
  and `offline_critic_lr_scale=None`. The new `train.warm_offline_phase` does the following with 0 steps:
  - no lr override and no grad step;
  - the throttle gate is measured once on the loaded critic after the relabel and logged as a warmup.jsonl row with
    `grad_steps 0` and status `throttle_gate` with `values` and `offline_grad_steps`, judged against tol 0.01;
  - on a pass, learn() starts as usual: `LrDecayCallback` sets scale 0.01 (critic 3e-6) before the first train(),
    retries and crash factor 3 are on from step 0, and the actor and alpha stay frozen until `critic_warmup_steps`
    (100k / 20k in the smoke).
  - The nonzero-offline path still exists for other presets; it now also counts the pre-training value in the
    "not rising" window.
- **QA m1** (`demo_driver.action_to_inputs`): below the knee the R3 lambda runs on `a[1]` as given, so float32
  input gives the old result bitwise (test).
- **QA m2** (`agent.preload_buffer`): the entry/exit pass-through now also requires `src_env.reward == cfg.env.reward`.
  Tested: R5 -> R6 accepted; changed exit_k / line_k / speed_k refused.
- **QA m3:** `test_tiny_warm_run_schedules_gate_and_freeze[0, 200]` runs `warm_offline_phase` + `learn()` with the
  run's schedule callbacks on a tiny warm run (throttle_relabel, lifetime_schedules, gate tol 1.0). It checks:
  - offline 0: one gate row, no update, critic lr 3e-6 from the first train() call;
  - offline 200 at scale 0.1: 3e-5 offline, then 3e-6;
  - both: retries on and crash factor 3 at step 0, actor frozen and unchanged until step 30, then trained.
- Spec amended (`docs/specs/R6_throttle_remap.md` §2.2, dated note).
- Tests after the follow-up (ocd_first_lap-f1): `tests/test_r6.py` 36 passed. R6 plus agent/r1-r5/demo_driver/f1_sha/
  lr_decay/retry/tune_tum: 257 passed, 1 skipped. **Full suite: 860 passed, 4 skipped, 1 failed**; the failure is the
  known `test_qss_profile::test_cached_file_matches_fresh_computation`, which also fails on HEAD.

## 1. What changed (files)

| File | Change |
|---|---|
| `rlsetup/driving/config.py` | `EnvConfig.throttle_remap: tuple \| None = None`; `check_throttle_remap(remap, pedal_deadzone)` (-0.5 < k < s <= 1, deadzone on); `EnvConfig.__post_init__` also refuses the remap with `reward.mode == 'ref_tracking'`; `to_dict` omits the key while None. |
| `rlsetup/driving/demo_driver.py` | `throttle_pedal(a, remap)` / `throttle_action(p, remap, released)` (scalar or array). `action_to_inputs`, `pedal_to_action` and `inputs_to_action` take a `throttle_remap=None` kwarg (throttle only). With None they run the old code path exactly as before. `DemoDriver(..., throttle_remap=)`; `DemoPolicy` passes the env's remap. |
| `rlsetup/driving/env.py` | `to_inputs` and the settle `DemoDriver` use `cfg.throttle_remap`; `throttle_obs_indices(cfg)` returns [133, 145, 157] for bcn_f1_r5 and handles obs_gear; the docstring notes that legacy `cost`/`smoothness` are not comparable under the remap, while `cost_r1` is. |
| `rlsetup/driving/evaluate.py` | `_pedals(actions, deadzone, throttle_remap=None)`; `lap_r4(..., throttle_remap=None)`, which `evaluate_model` calls with the run's remap. New per-straight metrics: `full_throttle_share_99`, `time_s`, `qss_time_s`, `time_lost_s`. New per-lap metrics: `straights_time_s` / `_qss_time_s` / `_time_lost_s`, `throttle_action_share_above_sat`, `throttle_sat_action` (0.85 without a remap). Helpers: `span_qss_time` and `straight_times`. |
| `rlsetup/driving/bc.py` | `pedal(a, deadzone, throttle_remap=None)`; `train_bc(..., throttle_remap=None)` (train.py passes it). |
| `rlsetup/driving/relabel.py` | `relabel_throttle_values`, `throttle_relabel_error`, `relabel_throttle(rb, src, dst, obs_indices)` (actions[...,1] + obs/next_obs slots, stats, abort above `THROTTLE_RELABEL_TOL` = 1e-6 with the buffer untouched). |
| `rlsetup/driving/agent.py` | `warm_start`: `throttle_remap` may differ from the source only with `warm.throttle_relabel`, and every other env field is still compared. `preload_buffer`: runs the relabel (source map from its config.json, absent = R3), puts the stats in `throttle_relabel`, raises InitMismatch on abort, and fills `aux['throttle_probe']` (gate batch of 4,096 rows with old throttle > 0.70, copied before the relabel). Also one fix of an existing check, see §4 D1. |
| `rlsetup/driving/train_config.py` | `WarmStartConfig.throttle_relabel / offline_critic_lr_scale / throttle_gate_batch (4096) / throttle_gate_tol (0.01) / throttle_gate_blocks (5)`, omitted while all at defaults; `TrainConfig.lifetime_schedules` (omitted while False); presets `bcn_f1_r6` and `bcn_f1_r6_smoke`; constants `BCN_F1_R6_INIT`, `BCN_F1_R6_BUFFER`, `R6_THROTTLE_REMAP`. |
| `rlsetup/driving/callbacks.py` | `LrDecayCallback(cfg, offset=0)`, `RetryCallback(start_step, offset=0)`, `CrashEmphasisCallback(cfg, offset=0)` read `num_timesteps + offset`. The default of 0 keeps the old behaviour. |
| `rlsetup/driving/train.py` | `schedule_offset` (status `lifetime_steps_offset` when `lifetime_schedules`); `offline_lr_scale`, which sets `model.lr_scale` for the offline phase only (`LrDecayCallback` sets 0.01 at learn start); throttle gate (`throttle_gate_value` / `throttle_gate_passed`, `throttle_gate` per block in warmup.jsonl plus a value at 0 grad steps). If the gate fails, the status gets `state='throttle_gate_failed'` and the run exits with code 2 before `learn()`. `provenance.json init_buffer.throttle_relabel`; status `init_buffer.throttle_relabel`, `throttle_gate`. |
| `rlsetup/assetto/actor.py` | `_config` raises `ValueError('unsupported source env.throttle_remap')` when `env.get('throttle_remap') is not None`. R5 configs pass (tested). |
| `scripts/evals_to_tb.py` | TB scalars for `full_throttle_share_99` / `time_lost_s` per straight and two lap-level keys. These are only written when present (small addition beyond the spec). |
| `scripts/r6_prelaunch.py` (new) | Pre-launch items (a) and (b), no training. Output: `runs/mac/r6_prelaunch.json`. |
| `tests/test_r6.py`, `tests/r6_golden.py` (new) | Spec §4 tests 1-6 (32 tests). |
| `tests/test_{lr_decay,corner_line,retry,r2,r4}.py` | Added `bcn_f1_r6`/`bcn_f1_r6_smoke` to the preset lists these tests enumerate. |

Gate rule as implemented: the last block's value must be <= 1 %, and not above the value 4 blocks earlier ("not
rising over the last 5 blocks").

## 2. Tests

- `tests/test_r6.py` under ocd_first_lap-f1: **32 passed** (5.7 s).
- R6-adjacent suites under ocd_first_lap-f1 (corner_line, lr_decay, r2, r4, retry, r5, r6, ac_actor, ac_r5,
  demo_driver, f1_sha): **287 passed, 1 skipped** (31 s).
- Full suite, default checkout (`.venv/bin/python -m pytest -q`): before the preset-list fixes, 6 failed / 837
  passed / 17 skipped. Five of the failures were preset-enumeration tests (fixed). `test_qss_profile::test_cached_file_matches_fresh_computation`
  fails on an `ocd_lap_sources_sha256` mismatch between the cached QSS file and the current ocd_first_lap, which
  does not involve rlsetup (it also fails on HEAD).
- **Full suite under ocd_first_lap-f1** (`OCD_FIRST_LAP=… PYTHONPATH=<f1>:$PWD .venv/bin/python -m pytest -q`):
  **856 passed, 4 skipped, 1 failed** (274 s). The one failure is `test_qss_profile::test_cached_file_matches_fresh_computation`,
  and it **also fails on HEAD 7bf3cd0** (checked with HEAD's rlsetup + tests from `git archive`): the cached QSS
  file's `ocd_lap_sources_sha256` no longer matches the ocd_first_lap sources. This is the same mismatch behind the
  "cache key does not match … recomputing in memory" warning that every run prints. It has nothing to do with R6.

Bit-identity with the remap off:
- Grid tests against verbatim copies of the HEAD expressions.
- The 200-decision bcn_f1_r5 env stream hash `d0ffef43…c4c1` was generated with HEAD 7bf3cd0's rlsetup
  (`git archive` into the scratchpad) and equals the new code's hash. With the remap on, the hash differs.
- No pre-R6 preset writes `throttle_remap`, `lifetime_schedules` or the R6 warm keys.

### The 3 failing relabel tests (fixed; the relabel code was right)
1. `test_synthetic_bonus_buffer_changes_only_throttle_cells`: a test bug. SB3 `buffer_size` counts transitions
   (300 / 2 envs = 150 rows), so 280 adds wrapped the buffer. The test now sizes the buffer as `n * n_envs`.
2. `..._shanghai` and `..._bcn_f1_r5`: the scripted rollout left the track before 100 decisions. Now pure pursuit on
   xi_ref, with throttle above the knee below f·v_qss.
3. bcn only: it then failed the 1e-5 match by up to 2.2. The cause is the simulator, not the relabel. **Under the
   unchanged R3 map**, a 1-ulp float32 throttle change moves the observation by 0.019 after one decision and by 1.2
   within 100 decisions at 35-50 m/s (rear slip, obs[8]). So no pedal difference (including the 2e-7 rounding the
   spec accepts) can be checked to 1e-5 there. At 60 m/s on the main straight into T1, the same 1-ulp change moves
   it by about 2e-7. The test uses that start (`start_s 0, start_speed 60`, f = 0.85, 90/100 throttle actions above
   the knee), and the exactness bounds stay as the spec wrote them (obs, next_obs and rewards within 1e-5; settle
   throttle slots within 1e-6; policy slots bitwise).
   The relabel's own error is 3.2e-8 in pedal, both in the test and on the real buffer.
   Worth knowing: at low speed in low gears the sf70h_f1 dynamics are chaotic at the float32 level.

## 3. Pre-launch (spec §3; `scripts/r6_prelaunch.py`, output `runs/mac/r6_prelaunch.json`)

The real R5 buffer through `warm_start` + `preload_buffer` (no training):
- Source: 7,000,000 steps, alpha 0.0630; lifetime offset 7,000,000.
- 250,000 rows (1M transitions, full).
- Cells changed: 496,060 action cells (49.6 % above the knee), 1,486,320 obs cells, 1,487,249 next_obs cells.
- **max |delta pedal| 3.18e-8** (abort level 1e-6). No stored throttle above 0.85 remains.
- Throttle gate before any grad step: **0.059 %** (4,096 rows, mean |Q_src| 250.9).

(a) R5 7M lap 2 under the old map (stored `step_7000000_det.npz`; a re-run with this code is identical, 85.27 s):

| | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | sum |
|---|---|---|---|---|---|---|---|---|---|
| pedal >= 0.99 share | 0 | 0.049 | 0 | 0 | 0 | 0 | 0 | 0 | |
| pedal >= 0.95 share | 0.034 | 0.588 | 0.164 | 0 | 0.138 | 0.063 | 0 | 0.416 | |
| brake share | 0 | 0.029 | 0.082 | 0 | 0 | 0.117 | 0.167 | 0.017 | |
| top speed [m/s] | 62.5 | 74.9 | 63.0 | 64.8 | 66.5 | 74.5 | 55.2 | 77.8 | |
| time lost vs QSS [s] | 0.127 | 0.304 | 0.276 | 0.320 | 0.215 | 0.281 | 0.103 | 0.934 | **2.56** |

Straights in total: 34.12 s driven vs 31.56 s QSS. The sum of 2.56 s is below the research note's ~3.6 s; the
definitions of the span differ. Throttle action share > 0.85: 0.374.

(b) Evaluation of the unchanged 7M actor (det 2 laps + 10 stochastic laps), same code, old vs new map:

| | old map (R5) | **new map (R6, step-0)** |
|---|---|---|
| det lap 2 | 85.27 s (clean) | **84.52 s** (clean) |
| det lap 1 | 86.33 s | 85.57 s |
| stochastic clean | 9/10 (one off track in U7) | **10/10** |
| stochastic mean lap | 87.61 s | 86.96 s |
| straights time lost (lap 2) | 2.56 s | 1.98 s |
| pedal >= 0.99 share S1/S2/S5/S6/S8 | 0 / 0.05 / 0 / 0 / 0 | 0.53 / 0.76 / 0.63 / 0.57 / 0.91 |
| top speed S8 | 77.8 m/s | 80.0 m/s |
| Q_min / MC^n (stochastic laps) | 0.958 | 0.942 |
| policy entropy | -2.917 | -2.899 |

There was no deterministic crash. The map change alone takes 0.75 s off lap 2 and fixes the one stochastic crash
in this sample. That is information, not a result: it is one deterministic lap and 10 stochastic laps.

## 4. Deviations / decisions taken as the coder (please check)
- **D1 (needed for the spec to run at all):** `preload_buffer` refused any source with an enabled corner bonus
  when the target uses the R4 entry/exit terms without `speed_relabel`. That check was written for R2/R3
  exit-bonus buffers. R5's buffer already holds R6's identical entry/exit terms (same `corner_bonus` block,
  B_in = B_out = 0), so the approved warm start failed with "holds exit bonuses". It now refuses only when the
  source `corner_bonus` differs from the target's. Exit-bonus buffers and buffers with different entry/exit terms
  are still refused (new test).
- D2: the demo dataset key. The `env` part is unchanged without the remap (tested). But the key also hashes the
  source files env.py, config.py and demo_driver.py, so any edit to them changes it. This is unavoidable and does
  not affect R6, which has no demos.
- D3: gate details not in the spec: a value is logged before the first block, and "not rising" means the last
  block's value is <= the value 4 blocks earlier. A gate failure uses exit code 2 (EXIT_GATE) and
  state `throttle_gate_failed`.
- D4: the offline phase lr scale is applied by setting `model.lr_scale` before `_offline_critic`.
  `offline_critic_lr_scale=None` together with `lifetime_schedules` gives the lifetime schedule's scale (0.01).
  Pre-R6 presets are unscaled, as before.
- D5: the straight time window runs from the end of the decision before the first in-span decision to the end of
  the last in-span decision; the QSS integral covers that same Track.path span. A straight across the line is
  counted once (its two lap-2 parts are summed).
- D6: `evals_to_tb.py` scalars (an addition beyond the spec).
- Smoke preset: `max_wall_h` 1.0 as a safety cap. The gate is also checked in the smoke, after only 5 blocks.

## 5. Launch and resume commands (updated 21 Nov 2026)

Smoke (150k; no offline phase; gate checked once; online critic 20k; dirty tree allowed). The new run name keeps
the failed `bcn_f1_r6_smoke` folder:

    cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
    PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 OCD_FIRST_LAP=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 \
      caffeinate -dimsu platforms/mac/train.sh --preset bcn_f1_r6_smoke --seed 0 --run-name bcn_f1_r6_smoke2 \
      >> runs/mac/bcn_f1_r6_smoke2_stdout.log 2>&1

What to check after the smoke:
- status: `init_buffer.throttle_relabel` (max_abs_dpedal ~3e-8), `throttle_gate.passed` true with one value ~0.0006.
- `phases.offline_warmup.grad_steps` 0 and no `lr_scale` there.
- TensorBoard `train/critic_lr` 3e-6 from step 0 and `lr_scale` 0.01 in evals.
- `retry.retry_episodes` > 0 and `crash_emphasis.factor` 3 from step 0.
- the unfreeze event at 20k.

The 4M run uses **preset `bcn_f1_r6` and run name `bcn_f1_r6b`**. After the original 100k-freeze run
collapsed, the amended run used a 20k freeze. Its checkpoints and logs are under
`runs/mac/D2/bcn_f1_r6b/seed0`.

The launch used the clean `9912dc0` checkout for strict provenance and supplied the 20k override
explicitly. The managed sandbox denied the forkserver socket. The successful launch used
`vec_start_method=spawn` with the same four workers. This is the recorded launch command.

    export R6B_MAIN=/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
    export R6B_OCD=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1
    cd "$R6B_MAIN/runs/mac/r6b_code_9912dc0"
    PYTHON_BIN="$R6B_MAIN/.venv/bin/python" RUNS_ROOT="$R6B_MAIN/runs/mac/D2" \
      PYTHONPATH="$R6B_OCD" OCD_FIRST_LAP="$R6B_OCD" \
      caffeinate -dimsu nohup platforms/mac/train.sh --preset bcn_f1_r6 --seed 0 --run-name bcn_f1_r6b \
      --set warm.critic_warmup_steps=20000 --set vec_start_method=spawn \
      --init-from "$R6B_MAIN/runs/mac/D2/bcn_f1_r5/seed0/checkpoints/step_7000000.zip" \
      --init-buffer "$R6B_MAIN/runs/mac/D2/bcn_f1_r5/seed0/replay_buffer.pkl" \
      >> "$R6B_MAIN/runs/mac/bcn_f1_r6b_stdout.log" 2>&1 &

The legacy keeper uses the run name as the preset, so it would try the nonexistent preset
`bcn_f1_r6b`. If training stops with a saved `last.zip`, resume manually from the same clean
checkout. Resume reads the 20k freeze and spawn method from the saved `config.json`.

    export R6B_MAIN=/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
    export R6B_OCD=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1
    cd "$R6B_MAIN/runs/mac/r6b_code_9912dc0"
    PYTHON_BIN="$R6B_MAIN/.venv/bin/python" RUNS_ROOT="$R6B_MAIN/runs/mac/D2" \
      PYTHONPATH="$R6B_OCD" OCD_FIRST_LAP="$R6B_OCD" \
      caffeinate -dimsu nohup platforms/mac/train.sh --resume --preset bcn_f1_r6 --seed 0 --run-name bcn_f1_r6b \
      >> "$R6B_MAIN/runs/mac/bcn_f1_r6b_stdout.log" 2>&1 &
