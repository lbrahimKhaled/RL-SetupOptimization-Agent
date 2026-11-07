# R2 implementation — rl-coder report (7 Nov 2026)

Spec: `docs/specs/R2_review.md` (Amendment 1 wins) > rest of the review > `docs/specs/R2_corner_bonus.md`.
Decisions #22-23. Nothing in ocd_first_lap / ocd_first_lap-f1 was touched; `runs/mac/D2/f1_best_w35/` only read.
No training run longer than the slow test's mini smoke was started; f1_r2 not launched. Not committed.

## Files (lines added / removed; new files: total lines)
| File | +/- | What |
|---|---|---|
| `rlsetup/driving/corners.py` (new) | 315 | corner table (R2-2: QSS on xi_ref, map to Track.path, find_peaks, merge, windows, exits), cache `data/corner_table_<key12>.json`, `CornerTracker`, `settle_events`, f(rho) |
| `rlsetup/driving/buffers.py` (new) | 87 | `BonusNStepReplayBuffer` (+ `adopt`, `labelled_fraction`) |
| `rlsetup/driving/config.py` | +45 / -1 | `CornerBonusConfig`, `FocusConfig`, `TRAINING_AID_KEYS`, `EnvConfig.to_dict` omits default aid blocks |
| `rlsetup/driving/train_config.py` | +99 / -4 | `BonusScheduleConfig` (+ `trigger_lap2_s`, `report_from`), `FocusUpdateConfig`, `AgentConfig.low_noise_*`, `TrainConfig.r2_aids` / `to_dict`, presets `f1_r2`, `f1_r2_smoke` |
| `rlsetup/driving/env.py` | +100 / -5 | tracker per 50 Hz tick + info, focused starts, `set_focus_weights`, `focus_start` |
| `rlsetup/driving/agent.py` | +76 / -16 | buffer class in `build_model`, `StagedSAC._sample_action` (low noise), R2 attrs, `_env_without_reward` drops aid keys, `preload_buffer` adopt + relabel refusal |
| `rlsetup/driving/callbacks.py` | +369 / -12 | `BonusScheduleCallback`, `bonus_weight`, `load_bonus_state`, `FocusedStartCallback`, `focus_seed`, TrainStats groups, eval-callback extras / schedule / best_aided |
| `rlsetup/driving/evaluate.py` | +47 / -5 | `corner_bonus` in TRACE_KEYS, corner events in LapReport, `lap_corners`, r + w b in Q vs MC, det_bonus_sum, lap2_corners, fail_s, sto_corners |
| `rlsetup/driving/train.py` | +118 / -17 | table built once, passed to workers / eval envs; R2 callbacks; w(0) before offline critic; provenance 'r2'; resume state; stop reason `report_complete` |
| `rlsetup/driving/vec.py` | +8 / -7 | `corners` through `make_env` / `env_fns` / `build_vec_env` |
| `scripts/evals_to_tb.py` | +86 / -36 | R2 scalars, `tb_evals_lifetime/` at step + `lifetime_steps_offset` |
| `scripts/keeper_check.py` (new) | 62 | keeper `reached` (reportable, <= 95 s) / `stopped` (ended itself after a trigger) |
| `runs/mac/keeper_f1.sh` (local, gitignored) | 33 | uses keeper_check.py; never resumes a self-ended run; mirrors TB also when stopped |
| `tests/test_r2.py` (new) | 882 | R2-10.1 - 10.11 (fast) |
| `tests/test_r2_slow.py` (new) | 111 | R2-10.12 mini f1_r2_smoke from best.zip + resume mid-fade |
| READMEs (driving, scripts, tests) | +5 / -1 | module / script / test tables |
| `data/corner_table_c83c1c689508.json` (new, generated) | — | cache of the f1_r2 corner table (like the tracked QSS caches; commit it or let the first run rebuild it, 0.4 s) |

## Verified by running (not recalled)
- Corner table (sf70h_f1, sha_f1_best, w3.5 line) = R2-2.7 within 1 m on every apex / window start / exit;
  v_qss_exit within 0.2 m/s; t_line end = line QSS lap 93.8252937271318 s = `line_lap_time`.
- best.zip of f1_best_w35, deterministic eval with the bonus ON: lap 2 = 101.87194035987982 s (S1), obs / rewards /
  laps bitwise equal to the bonus-OFF env; lap-2 rho (0.912, 0.910, 0.847, 0.897, 0.894, 0.897, 0.926) (S2 within
  0.015); lap-2 bonus sum 12.39 (spec estimate 12.7).
- Focus seed of f1_best_w35: 208 deterministic evals >= 2M, 166 crashes, H0 = 650, 700, 750, 800, 1550, 2350,
  2400, 2500, 3100, 4700, 5100 m (= R2-7.4).
- `config_json` of every pre-R2 preset byte-identical to HEAD (scripted comparison against a HEAD worktree);
  r1 golden rollout hash unchanged, also with corner_bonus + focus enabled (synthetic table, no weights pushed).
- Mini f1_r2_smoke (slow test, 69 s): init buffer adopted (labelled 0), step-0 lap 2 = S1, rho = S2,
  w = 1, 1, 2/3 | resume | 1/3, 0, 0, reportable only at report_from, best_aided.zip before / best.zip only at the
  reportable eval, bonus / focus state files, provenance 'r2'.

## Deviations / interpretations (closest to the spec's intent)
1. **Demo dataset keys.** The env dict that enters `demos.dataset_key` is unchanged (test: 12 local demo datasets
   round-trip to the same dict and key; the f1_best_w35 dataset's env dict equals the preset's). But the key also
   hashes the source of `env.py` and `config.py` (DEMO_SOURCES, DECISIONS #15), which R2 must edit, so a *fresh*
   `bc` run of an old preset will recollect its demos (as after any earlier env.py edit). Existing runs, resumes and
   warm starts (f1_r2) are unaffected.
2. `TrainConfig.to_dict` also omits `bonus_schedule`, `focus_update` and `agent.low_noise_*` when no R2 aid is on
   and they are at their defaults (the spec only said this for EnvConfig); needed for "config dicts unchanged".
3. Event kind `exit_unpaid` = an exit crossed without failure but rho <= floor (bonus 0); the spec lists the kind
   without defining it.
4. Focused starts draw nothing from `np_random` until weights are pushed (the spec's "uniform start if none yet"):
   the resets before `on_training_start` and any env without weights keep the exact pre-R2 start stream.
5. Focus counters: the seed prior (n = 200, c = 200 crashes_b / N_evals) is the *initial value* of the decayed
   counters (it decays with half-life 1M like live counts), not a permanent pseudo-count. While sum r = 0 the 0.8
   part is uniform; with an empty H0 the 0.2 floor is uniform over all bins. fallback_rate = fallbacks / resets
   drawn as focused.
6. Trigger record: `bonus_w` = weight in effect before the trigger, `report_from` = the new value,
   `aided_target_lap: true`, `reportable: false`; best_aided.zip is always written at the trigger. Trigger
   threshold is `BonusScheduleConfig.trigger_lap2_s = 95` (config, not a literal).
7. Resume of a run that already ended itself after a trigger returns "nothing to do" (exit 0) without touching it
   (defensive, in addition to the keeper rule).
8. The keeper's checks moved to the tracked, tested `scripts/keeper_check.py` (`runs/` is gitignored, so the edited
   `runs/mac/keeper_f1.sh` is a local file only). `reached` also applies `reportable` to pre-R2 runs (records without
   the key count as reportable) and uses `<= 95` (draft §5).
9. R2 eval metrics (det_bonus_sum, lap2_corners, lap2_bonus_sum, fail_s, sto_corners, q_vs_mc_bonus_w) appear only
   when the eval env pays the bonus; `corner_bonus` is always in TRACE_KEYS, so laps/*.npz of old presets gain one
   all-zero array. TrainStats `groups` only when an R2 aid is on.
10. The corner table cannot be built for car 'av21' (ocd_lap's `lap_time` drops `v_profile` for av21); fast tests on
    the main checkout use synthetic tables, the real table is tested with sf70h_f1.
11. Slow test: a 1500-row init buffer of real transitions of the source actor replaces the 1.3 GB replay_buffer.pkl,
    "kill" is a graceful stop at total_timesteps then `--resume` (as test_r1_slow), and the actor stays frozen
    (critic phase longer than the run) so every eval drives the same lap. Focused resets are not exercised there
    (no episode ends within the mini run); they are covered by the fast tests (N = 10,000 sampler, N = 20 resets).

## Question for the lead (not decided here)
- After an Amendment-1 trigger the run stops at report_from, so exactly one evaluation (the one at report_from) is
  `reportable`. R2-12.8's "median of the last 5 clean deterministic lap 2 (reportable only)" then has one sample.
  Options: take the last 5 clean evals of the bonus-free window [trigger, report_from] (w = 0 there, but < 4M
  bonus-free steps), or keep the run going past report_from. Ibrahim's call; the code currently stops at report_from.

## Test results (run 7 Nov, M4)
- Baseline = unmodified HEAD (worktree): 2 failures in both PYTHONPATH variants, pre-existing and unrelated to R2:
  `test_provenance.py::test_every_imported_ocd_lap_module_is_hashed` ({'ocd_lap/cars.py'} not hashed) and
  `test_qss_profile.py::test_cached_file_matches_fresh_computation` (KeyError 'provenance').
- `.venv/bin/python -m pytest -q`: 338 passed, 10 skipped, 3 failed in 140 s = the 2 pre-existing failures +
  `test_r1_slow.py` whose resume was refused (exit 5, rlsetup diff_sha256 changed) because I edited READMEs while
  it ran; rerun alone: `test_r1_slow.py` + `test_d2_slow.py` 3 passed (110 s). Skips: sf70h_f1 / f1_best_w35
  tests (incl. 3 R2 ones) and the Windows launcher.
- `PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python -m pytest -q`: 345 passed,
  4 skipped, 2 failed (the 2 pre-existing) in 232 s; includes `test_r2_slow.py` (69 s alone) and the real corner
  table / focus seed tests.
- `tests/test_r2.py` alone: 38 passed / 2 skipped (default), 40 passed (f1), ~7 s.
- Indicative env cost of the tracker (single env, best.zip actor, while other tests ran): 1501-1462 vs
  1555-1501 decisions/s (~2-3 %). S8 (>= 225 env steps/s end to end) is for the sim-runner smoke.

## Launch commands
Smoke (sim-runner, <= 30 min; strict_provenance False, so a dirty tree is fine):
```
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 platforms/mac/train.sh --preset f1_r2_smoke \
  --seed 0 --run-name f1_r2_smoke --init-from runs/mac/D2/f1_best_w35/seed0/checkpoints/best.zip \
  >> runs/mac/f1_r2_smoke_stdout.log 2>&1
```
f1_r2 (after the commit: the preset has strict_provenance; R2-9.8):
```
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 caffeinate -dimsu nohup platforms/mac/train.sh \
  --preset f1_r2 --seed 0 --run-name f1_r2 --init-from runs/mac/D2/f1_best_w35/seed0/checkpoints/best.zip \
  >> runs/mac/f1_r2_stdout.log 2>&1 &
# once `pgrep -f "run-name f1_r2"` shows it (else the keeper would try a resume):
nohup runs/mac/keeper_f1.sh f1_r2 >> runs/mac/keeper_f1_r2.log 2>&1 &
tensorboard --logdir_spec f1_r2:runs/mac/D2/f1_r2/seed0/tb_evals_lifetime,f1_best_w35:runs/mac/D2/f1_best_w35/seed0/tb_evals
```
The init buffer defaults to `runs/mac/D2/f1_best_w35/seed0/replay_buffer.pkl` (1.3 GB, adopted in place, relabel off).

## What QA / sim-runner should check first
1. `buffers.BonusNStepReplayBuffer._get_samples` against SB3 2.9.0 `NStepReplayBuffer._get_samples` (line-by-line;
   the only new line is the `bonus_weight` one).
2. Amendment-1 flow: `BonusScheduleCallback.on_eval` / `_on_step`, `DrivingEvalCallback` best vs best_aided,
   `resume_pipeline` early exit, `scripts/keeper_check.py` + `runs/mac/keeper_f1.sh` (local file).
3. Smoke S4 / S5: focused resets were not exercised in the mini slow run (no episode ended); check
   `evals.jsonl` 'focus' (share_resets ~0.25, fallback_rate < 0.1, share_transitions <= 0.30) and
   `train_stats.jsonl` 'groups' (low_noise vs normal) in f1_r2_smoke.
4. S3: paid bonus per pass (`corner_events` in eval traces; lap2_bonus_sum ~12.4 at step 0) and that VecMonitor
   returns / det_return are bonus-free.
5. S6 critic_check at the unfreeze (q_vs_mc now uses r + w b) and S8 throughput.

## Fix batch (8 Nov)
Spec: `docs/specs/R2_review.md` Amendment 2 (wins), `docs/handoff/R2_qa_review.md` (MINORs, tests 1-6),
DECISIONS #24. Nothing in ocd_first_lap / ocd_first_lap-f1 touched; `runs/mac/D2/f1_best_w35/` only read; no
training launched; not committed.

### Changes
| File | What |
|---|---|
| `rlsetup/driving/train_config.py` | `F1_R2_INIT` = `<repo>/runs/mac/D2/f1_best_w35/seed0/checkpoints/last.zip` (absolute, like `--init-from`); presets `f1_r2` and `f1_r2_smoke` default `init_from` to it (`--init-from` still overrides). Docstrings: Amendment 2. |
| `rlsetup/driving/callbacks.py` | Amendment 2: `result_eligible(rec, trigger_step, report_from)` (trigger_step <= step <= report_from and bonus_w == 0), `best_eligible(rec, ...)` (reportable or result stretch). `BonusScheduleCallback`: after a trigger every record gets `result_eval`; `result()` = {trigger_step, report_from, n_evals, best_lap2_s, best_step, clean_laps [{step, lap2_s}]} written to bonus_state.json `result` and passed to `on_result` at every evaluation; `history=` (resume) rebuilds the stretch from the records (also from records without `result_eval`). `DrivingEvalCallback`: best.zip from `best_eligible` records; best_aided.zip from the others and always at the trigger (a trigger with w already 0 writes both). |
| `rlsetup/driving/train.py` | `_callbacks(eval_history=...)`, `on_result` -> status.json `result`; resume rebuilds best_lap2 from `best_eligible` records and best_aided from the others + the trigger record; passes the records to the schedule. |
| `rlsetup/driving/corners.py` | QA MINOR (a): `CornerTracker.reset` sets a threshold of exactly 0 (hand-over at s_out) to L (exit counts as passed, paid one lap later). |
| `data/corner_table_6c61b28a4876.json` (new) / `data/corner_table_c83c1c689508.json` (deleted) | corners.py is part of the cache key (R2-2.8), so the key changed. Rebuilt with sf70h_f1: `s`, `v`, `units`, `lap_time_s`, `length` identical to the old file (only meta key / provenance / created differ). Commit the new file and the deletion together. |
| `scripts/keeper_check.py` | QA MINOR (b): `pattern(RUN)` / CLI `pattern RUN` prints the POSIX ERE `rlsetup\.driving\.train .*--run-name RUN( |$)` (RUN's ERE metacharacters escaped). QA MINOR (c): the limit is the run's `config.json` `bonus_schedule.trigger_lap2_s` (`limit_s`), else `LIMIT_S_DEFAULT` = 95 (pre-R2 configs omit the block), tested equal to `BonusScheduleConfig().trigger_lap2_s`; an explicit CLI limit still overrides. |
| `runs/mac/keeper_f1.sh` (local, gitignored) | `PAT="$(.venv/bin/python scripts/keeper_check.py pattern "$RUN")"`; messages say "limit" instead of 95 s; replaced atomically (no keeper was running). |
| `scripts/README.md` | keeper_check row. |
| `tests/test_r2.py` | +11 tests (below). |

Interpretation (flag for the lead): "w = 0 in [trigger, report_from]" is applied to the record's `bonus_w`. The
trigger evaluation's `bonus_w` is the weight in effect up to it (deviation 6 of the R2 report), so a trigger during
the bonus/fade (w > 0) is NOT in the result stretch (only best_aided.zip), while a trigger after the fade (w already
0, step < report_from) IS (best.zip and best_aided.zip). Without a trigger nothing changes (R2-5 / R2-9.3).

`lifetime_steps_offset` stays derived from the init checkpoint (`warm_start` -> `src_num_timesteps`); last.zip's
`num_timesteps` = 12,350,000 (read from the zip; `test_f1_r2_init_is_12_35M_steps`). Older presets: config dicts
unchanged (`test_config_dicts_without_aids_are_unchanged` round trip + pre-R2 key check pass).

### Tests added (`tests/test_r2.py`)
- `test_tracker_handover_exactly_at_exit` (QA 1; fails on c2791ea's corners.py, passes now)
- `test_env_stall_in_paying_decision_does_not_forfeit` (QA 2; real env, stall truncation, crossing paid)
- `test_learn_and_finish_report_complete` (QA 3; tiny model, trigger state -> stops at report_from, status 'report_complete')
- `test_keeper_pattern_does_not_match_prefix` (QA 4; Python re on train.sh-style cmdlines + the real `pgrep -f` on a sleeping process with `--run-name <tag>_smoke`)
- `test_keeper_limit_single_source`
- `test_focus_weights_pushed_to_subproc_workers` (QA 5; `build_vec_env` = VecMonitor(SubprocVecEnv) of 2 real DrivingEnvs, forkserver; weights arrive in both workers, focused plans drawn there)
- `test_bonus_buffer_weighted_wrapped_multi_env` (QA 6; 3 envs, wrapped, w = 0.7: per-sample ring sums + bitwise equal to a plain buffer holding r + w b)
- `test_result_stretch_after_trigger`, `test_resume_rebuilds_best_from_result_stretch` (Amendment 2 incl. resume and late trigger)
- `test_f1_r2_presets_warm_start_from_last_zip`, `test_f1_r2_init_is_12_35M_steps` (needs_src)

### Test results (8 Nov, M4)
- `.venv/bin/python -m pytest -q`: **349 passed, 11 skipped, 2 failed** in 159 s. The 2 failures are the known
  pre-existing ones (`test_provenance::test_every_imported_ocd_lap_module_is_hashed`,
  `test_qss_profile::test_cached_file_matches_fresh_computation`). Skips +1 vs 7 Nov = the new needs_src test.
- `PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python -m pytest -q`:
  **356 passed, 4 skipped, 2 failed** (same 2) in 232 s, incl. `test_r2_slow.py` (still `--init-from best.zip`
  explicitly, so S1 101.87194035987982 s stays its check).
- `tests/test_r2.py` alone: 48 passed, 3 skipped, ~9 s (default path).
- CLI check: `train.main(['--preset', 'f1_r2', ...])` without `--init-from` resolves init_from to
  `<repo>/runs/mac/D2/f1_best_w35/seed0/checkpoints/last.zip`, identical to passing it explicitly.
- ocd_first_lap and ocd_first_lap-f1: `git status` clean.

### Launch (after the commit: f1_r2 has strict_provenance; the new corner-table cache must be committed with it)
```
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 caffeinate -dimsu nohup platforms/mac/train.sh \
  --preset f1_r2 --seed 0 --run-name f1_r2 --init-from runs/mac/D2/f1_best_w35/seed0/checkpoints/last.zip \
  >> runs/mac/f1_r2_stdout.log 2>&1 &
# (--init-from equals the preset default; kept explicit for the record.) Init buffer =
# runs/mac/D2/f1_best_w35/seed0/replay_buffer.pkl (12.35M, adopted in place); lifetime_steps_offset 12,350,000.
# Start the keeper only once the training process is visible (else it would try a resume):
pgrep -fl "$(.venv/bin/python scripts/keeper_check.py pattern f1_r2)"
nohup runs/mac/keeper_f1.sh f1_r2 >> runs/mac/keeper_f1_r2.log 2>&1 &
.venv/bin/tensorboard --logdir_spec f1_r2:runs/mac/D2/f1_r2/seed0/tb_evals_lifetime,f1_best_w35:runs/mac/D2/f1_best_w35/seed0/tb_evals
```
Stop by hand: `pkill -f "keeper_f1.sh f1_r2( |$)"; pkill -INT -f "$(.venv/bin/python scripts/keeper_check.py pattern f1_r2)"`.
Smoke variant (if wanted): same training line with `--preset f1_r2_smoke --run-name f1_r2_smoke` (default init is
last.zip too; strict_provenance off).
After a trigger: `status.json` / `bonus_state.json` `result` = best clean deterministic lap 2 of
[trigger, report_from] with w = 0 + the list of all its clean laps; best.zip is that best lap's checkpoint.
