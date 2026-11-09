# f1_r4_scratch: R4 reward from scratch (9 Nov 2026)

Preset `f1_r4_scratch` added and the BC path fixed so the R4 reward works without a warm start. Nothing committed, full run not launched, ocd_first_lap / ocd_first_lap-f1 and existing run folders untouched.

## Read first: f1_r4 is no longer running
PIDs 37020 and 37322 were alive at my first `ps` (12:05) and gone about a minute later. The last writes in `runs/mac/D2/f1_r4/seed0` are evals / `last.zip` / `replay_buffer.pkl` at 12:05:08-09 (13.3M steps); `status.json` still says `running`, `runs/mac/f1_r4_stdout.log` shows no error, and an orphaned Python child (37336) remains. Trainer and keeper both went at once, which points to a deliberate stop, but I cannot tell who did it. My commands up to then were reads only. I did not resume it. Until this work is committed, a resume needs `--allow-dirty` as well: the keeper passes only `--allow-code-change`, so it would be refused with exit 4 (`train.py` `resume_pipeline`, strict_provenance).

## What changed
| File | Change |
|---|---|
| `rlsetup/driving/train_config.py` | Preset `_f1_r4_scratch`, built from `_f1_best_w35()`, registered in `PRESETS`. |
| `rlsetup/driving/demos.py` | Demo episodes store `info['corner_bonus']` as array `corner_bonus` (only when the env reports it). `prefill_replay_buffer` passes it the way online transitions do, and a bonus buffer refuses a dataset without it (ValueError, nothing written). `dataset_key` also hashes `corners.py` and `speed_profile.py` when the corner bonus or speed potential is on (`REWARD_SOURCES`). |
| `rlsetup/driving/train.py` | `_set_bonus_w0` (shared with the warm path) sets w(0) before the BC offline warm-up. Its Q vs soft MC uses r + w b. |
| `tests/test_r4_scratch.py` | New, 14 tests. |
| `tests/test_r2.py`, `tests/test_r4.py` | Preset-enumerating checks allow the new preset (focus block absent; R4 keys present). |
| `rlsetup/driving/README.md`, `tests/README.md` | Table entries. |

Config comparison (`f1_best_w35` config.json vs preset + `platforms/mac/train.overrides`): shared keys differ only in `preset` and `runs_root` (a `--set` of train.sh). Every field added since f1_best_w35 is either the reward items, or off / unused in this path (low-noise empty, focus off, warm.* unused, R4 eval pedal thresholds = logging only). Pinned by `test_new_fields_against_the_real_control_config`.

## Findings on items 1-4
**1. Demo dataset and rewards: already correct; one gap fixed.** The key changes with the reward config (R4 fields are in `env.to_dict()`), and stored rewards come from the configured env, so they include the speed potential. Gap: the key did not hash `corners.py` / `speed_profile.py`, so a change there could have reused stale demos. Fixed for bonus / potential configs only; other configs hash the same inputs as before.
Check on real data: the smoke dataset has identical obs and actions to the control's `demos_603167e71116` (same 200,575 transitions), and rewards differ only by the speed shaping.

**2. Corner terms on demo transitions: was broken, fixed.** The prefill passed only `TimeLimit.truncated`, so every demo row got bonus 0 while being marked labelled. Now the dataset stores the terms and the prefill writes them. In the smoke's replay buffer, bonuses and rewards equal the dataset row for row in both columns (`np.array_equal`). n-step returns on prefilled rows match a hand computation for w = 1 and w = 0, including episode ends (`test_n_step_returns_on_prefilled_demos_include_the_terms`).

**3. Warm-start assumptions:**
- **Bug fixed:** the BC path never set w(0) before the offline warm-up, so offline critic training would have run with w = 0 and then jumped to w = 1 at `learn()`. Confirmed in a real pipeline run after the fix: at `_offline_warmup` entry `bonus_weight` 1.0, buffer rows 99,289, 69 nonzero terms.
- **Bug fixed:** the offline Q vs MC check ignored the terms.
- **No problem found (read, then exercised in the smoke unless noted):** relabel not reached in BC; `focus_update` with focus off (no seed, no callback; groups fall to `uniform`); schedule never fades (w = 1 at every eval, `report_from` null, `reportable` false, no trigger, `bonus_state.json` correct); lifetime offset absent (`lifetime_steps_offset` None, `evals_to_tb` writes `tb_evals` only); `keeper_check` pattern / reached / stopped work on the smoke folder; `r4_provenance` written.
- `best_aided.zip` logic is unchanged, but no clean lap occurred in the smoke, so the file itself was not produced there.

**4. R4 off stays bit-identical.** `test_golden_rollouts_unchanged`, `test_legacy_rollout_bit_identical` and `test_env_entry_exit_terms_and_reward_unchanged` pass. A plain env's dataset has no `corner_bonus` array. Dataset keys of all presets do change, because `demos.py` is itself a hashed source (DECISIONS #15), as any edit to it does.

## Tests
`PYTHONPATH=<ocd_first_lap-f1> .venv/bin/python -m pytest -q -m "not slow"`:
**412 passed, 4 skipped, 2 failed, 6 deselected in 57.5 s.** The 2 failures are the known ones (`test_every_imported_ocd_lap_module_is_hashed`, `test_cached_file_matches_fresh_computation`).
Count check: 14 new tests + 398 = 412, and 398 + 6 deselected slow tests = 404, so the "404 passed" baseline matches the full suite in `R4_coder.md`, not a `not slow` run (inferred from the counts).
Mutation check: breaking each of the 6 fixes in turn (terms dropped at prefill, not stored in the dataset, no refusal, reward sources not hashed, w(0) not set, Q-MC ignoring terms) made a new test fail each time; sources restored and checksums verified.

## Smoke (real pipeline, temp folder under the session scratchpad, deleted afterwards)
Preset `f1_r4_scratch`, 60k steps. **Full demo set and full BC + offline phase, not reduced.** Reduced for CPU only: `n_envs=2`, `gradient_steps=1` (same UTD 0.5), buffer 400k, eval every 30k with 3 stochastic laps, guard probe every 5k, `demos.workers=2`.
Wall: demos 83 s, BC 17 s, offline 20k grad steps 120 s, online warm-up 20k steps 82 s, RL 40k steps 191 s; about 8.5 min in total, exit 0, status `completed`, one rollback (below).

- **NaNs: none.** Dataset rewards / terms / obs and buffer rewards / bonuses / obs / next_obs all finite; no NaN in evals, train_stats, warmup.
- **Demo gate:** 150 episodes, clean fraction 0.80 (gate passed exactly at the limit, same as the control).
- **Do demo transitions carry the terms? Yes, but very few are nonzero.** 71 of 200,575 decisions (sum 92.5, 0.62 per episode, max 4.43, none negative). All nonzero episodes have f_demo in [0.7, 0.8] (terms 1.65 per episode, 71 % of those episodes); f_demo < 0.7 gives exactly 0, because g = 0 at or below 0.7 v_qss with sigma 0.3. This is the demo speed range doing what the spec says, not a bug, but the critic learns almost nothing about the terms from the demos.

| Per-lap size | speed shaping | corner terms |
|---|---|---|
| Demo dataset (about 50.5 laps of distance, inferred from the plain reward) | -24.9 | +1.83 |
| Scripted demo driver, clean lap at f = 0.7 | -20.4 | +2.58 |
| Scripted demo driver, clean lap at f = 0.8 | -99.9 | +13.71 |

(Progress is 5388.8 per lap. Scripted f = 0.9 and 0.95 crashed before 0.1 lap, so there is no per-lap figure near QSS speed. The sum is negative because each decision pays gamma Phi' - Phi with gamma < 1: about -(1 - gamma) x 20 x mean g x decisions.)

**Training numbers, per episode (no lap was completed):**

| At step | shaping_v | corner terms | taken back |
|---|---|---|---|
| 30k eval | -3.6 | +0.08 | 0.11 |
| 60k eval | -16.9 | +1.58 | 0.28 |
| last 10k-step window | -23.8 | +3.91 | 0.21 |

Online buffer rows: 64 nonzero, 16 negative (take-backs), sum 138.2, range -5.03 to 7.44.

**Critic, offline:** Q_min / soft MC = 1.005 per step and 0.866 n-step (control: 0.984, 0.849), critic loss 31.0 (control 28.1).

**Things to read plainly, both also true of the control:**
- **BC deterministic gate lap fails off track** (control identical, same 12,952 BC env steps, `gate_required` False). All evaluated deterministic laps at steps 0 / 30k / 60k went off track, so `lap2_r4` is empty, the same as the control's first 200k steps.
- **The guard rolled back once**, at the first probe after the unfreeze (25k: 450.3 m vs limit 450.4 = 0.5 x 900.8), lr to 1e-4. Later probes 464, 458, 463, 735, 862, 607, 783 m, all ok. The control also rolled back at its first probe (30k: 400.8 m). `max_rollbacks` is 1, so a second dip below 450 m within the 200k guard window stops the run as `collapsed`; two probes landed within about 3 % of that line.

## Not verified
Learning behaviour or any clean lap (the smoke is far too short to show the reward's effect). Resume of a BC + R4 run (code read, not run). The keeper loop end to end (only `keeper_check.py` was run). `best_aided.zip` being written. The slow tests (6 deselected) were not run.

## Actor learning rate from the unfreeze (review B1, owner's decision; docs/specs/R4_scratch_review.md)
The control trained at actor lr 1e-4 from its guard rollback at 30k on. Without a change this run would have stayed at 3e-4 unless its guard fired.

**Changes**
- `rlsetup/driving/callbacks.py`: `PhaseCallback(..., actor_lr=None)` sets `model.actor_lr` at the unfreeze only when a value is given.
- `rlsetup/driving/train.py`: `_callbacks` passes `cfg.warm.actor_lr` for `init_mode == 'bc'` and None otherwise. This is the single place both fresh runs and resumes build it. The warm path still gets the value from `warm_start`.
- `rlsetup/driving/train_config.py`: `f1_r4_scratch` has `warm.actor_lr=1e-4`, the one allowed difference from `f1_best_w35` outside the reward.
- BC (own Adam) and the frozen phases keep their settings (`model.actor_lr` is None at the offline warm-up). Guard thresholds, rollback logic and `max_rollbacks` are unchanged. Critic and alpha stay at 3e-4. Presets with `warm.actor_lr` None never enter the new branch. None of these files is hashed into the demo dataset key.

**Resume after the unfreeze keeps 1e-4: from the saved model, by test.** `actor_lr` is a pickled attribute of the model and `StagedSAC._update_learning_rate` re-applies it at every `train()`. It is neither config-driven nor taken from the saved Adam state; a resume does not re-apply the config value (`test_resume_after_the_unfreeze_keeps_the_lr_from_the_saved_model`). A resume still inside the frozen phase gets it from the config at the unfreeze (`test_resume_inside_the_frozen_phase_applies_it_at_the_unfreeze`).

**Tests:** `PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python -m pytest -q -m "not slow"`: **426 passed, 4 skipped, 2 failed, 6 deselected in 57.6 s.** The 2 failures are the known ones (`test_every_imported_ocd_lap_module_is_hashed`, `test_cached_file_matches_fresh_computation`); no others. `tests/test_r4_scratch.py` now has 28 tests. It checks actor lr 1e-4 after the unfreeze for this preset and 3e-4 for all 8 other BC presets, and the field test allows only this difference. The source-text check is replaced by a spy at `_offline_warmup` entry in a mocked `run_pipeline` (real `build_model` and prefill; w = 1, 0 without a corner bonus; QA m3). Mutation check: 6 breakages of the new wiring and the w(0) step each failed a test; sources restored and checksums verified.

**Real pipeline check** (temp folder, deleted; 6,000 steps with reduced demos and warm-up, guard probes off so a rollback cannot explain it): `rollbacks` 0, evals record actor_lr None at step 0 (frozen) and 1e-4 at 3,000 and 6,000; SB3 logs `train/actor_lr` 1e-4 from the first record after the unfreeze at 2,000, with the base `train/learning_rate` still 3e-4. This changes the learning rate after the unfreeze, so the first smoke's numbers after 20k (probes, rollback, training windows) were measured at 3e-4 until its rollback; the 60k smoke was not repeated.

## Launch (not run; needs the tree committed first, strict_provenance refuses a dirty tree with exit 4)
```
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 caffeinate -dimsu nohup platforms/mac/train.sh --preset f1_r4_scratch --seed 0 --run-name f1_r4_scratch --set total_timesteps=12350000 >> runs/mac/f1_r4_scratch_stdout.log 2>&1 &
```
`total_timesteps` must be on the first launch (a resume reads config.json). **The keeper starts only after step 220k** (owner's decision, review B2: the 200k guard window after the 20k unfreeze; with the keeper running, a `collapsed` run would be resumed from its collapsed weights). After step 220k:
```
pgrep -fl "$(.venv/bin/python scripts/keeper_check.py pattern f1_r4_scratch)" && nohup runs/mac/keeper_f1.sh f1_r4_scratch >> runs/mac/keeper_f1_r4_scratch.log 2>&1 &
```
The run name must equal the preset name (the keeper resumes with `--preset $RUN`). First launch collects its demos under key `ad366e16f357` in `data/demos`, about 1 min.
