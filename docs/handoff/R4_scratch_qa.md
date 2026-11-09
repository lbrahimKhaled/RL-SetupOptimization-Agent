# QA: preset `f1_r4_scratch` (qa-reviewer, 9 Nov 2026, uncommitted diff before the B1 fix)

**Verdict: no blocker in the code.** Demo terms, n-step accounting and regression behaviour are correct. Three launch / keeper problems (M1–M3) to settle; the tree must be committed or the launch exits 4 (`train.py:511-514`).

## Tests
| Run | Result |
|---|---|
| `-m "not slow"` with PYTHONPATH = ocd_first_lap-f1 | 412 passed, 4 skipped, 2 failed, 6 deselected (58.2 s) |
| The 2 failures | `test_provenance::test_every_imported_ocd_lap_module_is_hashed`, `test_qss_profile::test_cached_file_matches_fresh_computation` (known) |
| `-m slow`, same PYTHONPATH | 6 passed, 418 deselected (261.7 s) |
| `-m "not slow"` without PYTHONPATH | 404 passed, 12 skipped, 2 failed (same 2) |

## Findings
**M1. Keeper resumes a collapsed run from collapsed weights** (`scripts/keeper_check.py:71-77`, `runs/mac/keeper_f1.sh:27-31`, `train.py:318,331`). Same as architect B2. Ibrahim's decision: start the keeper only after step 220k.

**M2. Coder's launch command has no step cap** (`docs/handoff/R4_scratch_coder.md:78`). Same as architect B3. Use `--set total_timesteps=12350000`.

**M3. Launch needs PYTHONPATH.** Without it `qss_profile.for_env` raises `unknown car 'sf70h_f1'` after `make_run_dir`, and a relaunch fails with FileExistsError until the folder is removed. Use:
`PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 caffeinate -dimsu nohup platforms/mac/train.sh --preset f1_r4_scratch --seed 0 --run-name f1_r4_scratch --set total_timesteps=12350000 >> runs/mac/f1_r4_scratch_stdout.log 2>&1 &`

**m1.** A completed run is resumed 5 times (harmless no-op: "nothing to do: total_timesteps already reached"), then the keeper exits 1 "giving up", a false alarm.

**m2.** The keeper's 95 s stop still applies (`keeper_check.py:64-68`); unlikely given the control's 101.87 s best.

**m3. Test gaps:** w(0) wiring checked by source-text search (`tests/test_r4_scratch.py:278-282`); take-back through `run_episode` on the dataset path; `resume_pipeline` with n >= total_timesteps; resume of a BC + bonus run.

**m4.** `corner_bonus` is outside `demos.ARRAYS`, so `tests/test_d2_slow.py:112` does not compare it across worker counts. Checked by hand: workers=1 equals the workers=4 prefix over 10 episodes (13,022 transitions).

## Checked
1. Demo terms come from the training code (`run_episode` stores `info['corner_bonus']` from `DrivingEnv.step`, tracker = `EntryExitTracker`). 60 episodes at f = 0.92 on the real car and table: same length as actions and rewards, every row equals the signed sum of that decision's events. 30 take-back episodes: negative terms only on the last row with a terminal failure. Real dataset: 71 nonzero of 200,575, sum 92.515, max 4.432, none negative.
2. Env reward holds Phi_v and no corner term; terms only in `info` and the `bonuses` column. Sampled n-step returns equal a hand computation at lags 0, 1, 2 before the terminal row. R4 and control datasets are equal in every array except `rewards`.
3. No regressions found. Dataset keys change (working-tree `demos.py` hash); an old file is not found under the new key and the dataset is recollected (51–62 s). The rebuilt `f1_best_w35` dataset is byte-identical to the control's (npz sha256 `7f67112c...`). Resumes and warm starts never reach the dataset code. All 10 existing `config.json` under `runs/mac/D2` load.
4. Launch argv verified. 12,350,000 is divisible by 4 envs, so the run ends exactly on the cap with `stop_reason` completed. Corner table key `6c61b28a4876`, 93.825 s, 7 units. Simulator digest for Shanghai `0efddcc91634`, equal to the control's provenance.

## Not verified
Any training or learning behaviour; a mid-run resume of a BC + R4 run (code read only); the no-op resume on a real completed folder; the keeper loop end to end; real resumes of f1_r2 / f1_r3 (config loading only); `best_aided.zip` being written; `evals_to_tb` on this run; the coder's mutation checks and smoke numbers apart from the dataset figures.
