# R3 implementation: rl-coder report (8 Nov 2026)

Spec: `docs/specs/R3_bigger_bonus.md` (Ibrahim's §1 decisions are fixed) + `docs/specs/R3_review.md` (binding, reqs 1-9),
DECISIONS #25. Nothing in ocd_first_lap / ocd_first_lap-f1 was touched. `runs/mac/D2/f1_r2/` was only read
(buffer mtime / size checked unchanged after the dry check). f1_r3 is not launched.

## Changes (file:line at the commit)
| Req | Where | What |
|---|---|---|
| preset | `rlsetup/driving/train_config.py:416-437`, `PRESETS` (:455) | `F1_R2_RUN`, `F1_R3_INIT` (absolute path to f1_r2 `checkpoints/last.zip`, like `F1_R2_INIT`), `_f1_r3()` = f1_r2 + `corner_bonus.bonus=30, floor_ratio=.85`, `bonus_schedule.hold_until=None`, `warm.offline_critic_grad_steps=50_000`, `focus_update.seed_run=None, init_state='runs/mac/D2/f1_r2/seed0/focus_state.json'`. `warm.init_buffer` stays None (resolves to f1_r2's `replay_buffer.pkl`). Everything else inherited (focus 25 %, low-noise env 3 at 0.3 sigma, actor lr 1e-4, guard 0.65, no plateau stop, no step / wall cap, strict_provenance). |
| 1 | `train_config.py:160-183` | `BonusScheduleConfig.hold_until: int \| None`; `never_fade` property; `report_from` None when never-fade. |
| 1 | `callbacks.py:610` | `bonus_weight` = 1 when `hold_until` is None. |
| 1 | `callbacks.py:646-648` | `BonusScheduleCallback.__init__`: no `int(None)`; `report_from` None for never-fade. |
| 1 | `callbacks.py:676` | `reportable()` False when `report_from` is None. |
| 1 | `callbacks.py:725` | trigger skipped when `report_from` is None (early trigger off). |
| 1, 8 | resume (`train.py` `load_bonus_state` path) | unchanged code: bonus_state.json `report_from: null` gives (None, None), the schedule rebuilt from it is never-fade again. Tested. |
| 2 | no code change | `best_eligible` is False for every never-fade record, so only best_aided.zip is written, also on resume rebuild. Tested. |
| 3 | `scripts/keeper_check.py:58-68` | `never_fade(seed_dir)` (config.json `bonus_schedule.hold_until` present and null); `reached` then counts every evaluation. Limit stays `trigger_lap2_s` = 95. f1_r2 / pre-R2 behaviour unchanged. |
| 4 | `agent.py:268-277` | `warm_start` returns `src_lifetime_steps_offset` (source status.json, else 0) and `lifetime_steps_offset` = that + `src_num_timesteps`. |
| 4 | `train.py:579-593` | status `lifetime_steps_offset` = `init['lifetime_steps_offset']`; provenance `init_from` records both parts; log line shows the sum. |
| 5 | `buffers.py:24-80` | `relabel_bonus_values` (float64 rho = f_s + (1 - f_s) b / B_s, b' = B clip((rho - f)/(1 - f), 0, 1), then float32), `check_bonus_relabel` (refuses f < f_s, B_s <= 0, floor outside [0, 1)), `relabel_bonuses` (in place on the stored rows; refuses b < 0 or b > B_s (1 + 1e-6) with nothing changed; returns stats: n_transitions, n_b_pos, n_to_zero, mean / max before and after, sha256 of the relabelled bonuses). |
| 5 | `agent.py:285-352` (refusal :310-322, relabel :341-345) | `preload_buffer`: with a bonus model, reads the source config.json; when the source has `corner_bonus.enabled` and a different (B, floor), the config-level refusal runs **before** the 1.3 GB load; after the type check it relabels (not for an adopted plain buffer, which holds no bonuses). Result key `bonus_relabel`, also in provenance `init_buffer.bonus_relabel` and status `init_buffer`. |
| 6 | `train_config.py:138-145`, `R3_WARM_KEYS` :209 | `WarmStartConfig.pre_exit_horizon=30`, `pre_exit_warn_q=.5`, `pre_exit_warn_blocks=5` (omitted from config.json of non-aid presets like the R2 blocks: `R3_WARM_KEYS`, `to_dict`). |
| 6 | `buffers.py:83-101`, `train.py:363-417` | `pre_exit_cells(rb, horizon)`: (row, env) cells with a nonzero bonus at offset 0..horizon-1 in the same column, no `dones` before it, in time order (never across the write position). `_offline_critic`: fixed batch (q_probe_batch = 512, rng seed + 1) of those cells, `mean_q_pre_exit` per block in warmup.jsonl / `warmup/mean_q_pre_exit`; status warning if its range (max - min) over the last 5 blocks is > 0.5. |
| 7 | `train_config.py:187-201`, `train.py:147-161` (loader), `:447-450` (pre-check), `:488`, `:616`, `:761-767` (resume) | `FocusUpdateConfig.init_state` (path, repo-relative unless absolute; wins over `seed_run`). `focus_init_state_for` loads it (refuses a different `bin_m`; the callback refuses a different bin count) and gives provenance `r2.focus_init_state` {path, sha256, step, updates, n_bins, hot}. Passed as `state=` to FocusedStartCallback. A missing file is refused before the run folder is created. On resume without focus_state.json the counters fall back to init_state (with a warning). |
| 9 | none | no other config change (the preset test asserts f1_r3 equals f1_r2 apart from the fields above). |
| docs | `rlsetup/driving/README.md`, `scripts/README.md`, `tests/README.md`, `scripts/evals_to_tb.py` docstring (f1_r3 logdir_spec), `train.py` / `train_config.py` docstrings | preset / module / test tables. |
| tests | `tests/test_r3.py` (new, 18 tests), `tests/test_r2.py` (aid-preset name check now includes `f1_r3`) | below. |

## Interpretations (flag if wrong)
1. **"Moves > 0.5 over the last 5 blocks"** is taken as the range (max - min) of `mean_q_pre_exit` over the last 5 blocks,
   checked once at the end of the offline phase. The warning goes to status.json `warnings` (step 0).
2. **Bonus relabel trigger** = source config has `corner_bonus.enabled` and (bonus, floor_ratio) differs. It is
   independent of `warm.relabel` (the R1 relabel), which stays False for f1_r3.
3. **Near the new floor.** A stored float32 b that corresponds to rho = 0.85 exactly (e.g. 1.2) maps to ~8e-7, not
   0, because float32(1.2) lies ~4e-9 above in rho. This is inside the review's rounding bound (<= 4e-6). No snapping
   was added, since the spec map has none. `n_to_zero` therefore counts only rows strictly below the floor after rounding.
4. In the never-fade schedule `fade_until` / `bonus_free_before_report` stay in config.json (unused); `trigger_lap2_s`
   = 95 is still read by the keeper.
5. For a never-fade run, `evals_to_tb` writes `bonus/reportable` = 0 at every evaluation (no code change, review req 8).

## Tests (`tests/test_r3.py`)
relabel map against CornerTracker outputs on a rho grid (0 ... 1.3 incl. 0.85 +- 1e-7, 1.0, 1.3, + 351-point
linspace), max |err| <= 4e-6, b = 0 -> 0, b = 3 -> 30, B = 0, identity; refusals (floor below, B_s = 0, floor 1,
summed b = 6, negative b; buffer unchanged); relabel stats on stored rows only; `preload_buffer` relabel + provenance
+ no relabel at equal (B, floor) + refusal before load (non-pickle file) + summed-bonus refusal; never-fade
schedule (w = 1 up to 1e9, report_from None, no trigger on 94 s / 90 s laps, never stops); w = 1 before every
`train()` call in `learn()`; DrivingEvalCallback loop: only best_aided.zip, never best.zip; never-fade resume
(bonus_state null -> (None, None), best inf / best_aided 94, focus fallback to init_state, evals_to_tb scalars);
old f1_r2 config.json loads (synthetic without R3 keys + the real file); keeper reached on aided evals for
never-fade and unchanged for f1_r2; lifetime offset chaining (0 + 2.6M, 12.35M + 2.6M); real f1_r2 offset
12,350,000 + last.zip 7,400,000 = 19,750,000; focus init_state (state equal, precedence, bin_m refusal, provenance,
missing file refused before the run folder exists); real f1_r2 focus_state (108 bins) accepted; `pre_exit_cells`
(horizon, dones, wrapped buffer); `_offline_critic` probe + warning on / off + no probe without a bonus buffer;
preset f1_r3.

## Test results (8 Nov, M4)
- `PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python -m pytest -q`:
  **374 passed, 4 skipped, 2 failed in 237 s** (also 374 / 4 / 2 before the last small pre-check edit). The 2 failures are the known pre-existing ones
  (`test_provenance::test_every_imported_ocd_lap_module_is_hashed` (cars.py), `test_qss_profile::test_cached_file_matches_fresh_computation`
  (KeyError 'provenance')).
- `tests/test_r3.py` alone: 18 passed in ~3 s (both path variants; without the f1 path `test_r2.py` + `test_r3.py`
  = 66 passed, 3 skipped).

## Dry check (real f1_r2 buffer, in memory, nothing written)
Script: build `--preset f1_r3` through `train.parse_args` + `PRESETS` + `apply_overrides`, resolve the init buffer,
load `runs/mac/D2/f1_r2/seed0/replay_buffer.pkl`, `relabel_bonuses` on the in-memory copy.
- Config: init_from `<repo>/runs/mac/D2/f1_r2/seed0/checkpoints/last.zip`, init_buffer (resolved)
  `<repo>/runs/mac/D2/f1_r2/seed0/replay_buffer.pkl`, B 30, floor 0.85, hold_until None, report_from None, offline
  50,000 grad steps, focus init_state `runs/mac/D2/f1_r2/seed0/focus_state.json`.
- Lifetime offset: 12,350,000 + 7,400,000 = **19,750,000**.
- Buffer: BonusNStepReplayBuffer, full, pos 236,389, 4 envs, labelled 1.0, stored bonus_weight 0.0 (w at 7.4M);
  loaded in 0.5 s.
- Relabel (3, 0.75) -> (30, 0.85) over 1,000,000 transitions:
  - rows b > 0: **2,496** (0.25 %)
  - rows b > 0 -> 0: **293** (11.7 % of b > 0; rho in (0.75, 0.85])
  - mean b before **0.004011**, after **0.018255** (over b > 0: 1.607 -> 7.314)
  - max b before **2.3087** (rho 0.942), after **18.478**; no row at b = 3 (rho >= 1)
  - rho of paid rows, 5/25/50/75/95 %: 0.830 / 0.868 / 0.885 / 0.905 / 0.928
  - sha256 of the relabelled bonuses: `05e073a2dd1ffdb63c121554f1134d7b8549a8ac413889b43c0aeb9b0f8f5c6d`
- Pre-exit cells (horizon 30): 66,075 (6.6 % of transitions), so the 512-row probe batch is well populated.
- Source buffer file mtime / size unchanged.

## Launch (after this commit: f1_r3 has strict_provenance)
```
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 caffeinate -dimsu nohup platforms/mac/train.sh \
  --preset f1_r3 --seed 0 --run-name f1_r3 \
  >> runs/mac/f1_r3_stdout.log 2>&1 &
# init_from = preset default (f1_r2 last.zip); init buffer = f1_r2 replay_buffer.pkl (relabelled in memory, the
# file is not modified); lifetime_steps_offset 19,750,000. Start the keeper once the process is visible:
pgrep -fl "$(.venv/bin/python scripts/keeper_check.py pattern f1_r3)"
nohup runs/mac/keeper_f1.sh f1_r3 >> runs/mac/keeper_f1_r3.log 2>&1 &
.venv/bin/tensorboard --logdir_spec f1_r3:runs/mac/D2/f1_r3/seed0/tb_evals_lifetime,f1_r2:runs/mac/D2/f1_r2/seed0/tb_evals_lifetime,f1_best_w35:runs/mac/D2/f1_best_w35/seed0/tb_evals
```
Stop by hand: `pkill -f "keeper_f1.sh f1_r3( |$)"; pkill -INT -f "$(.venv/bin/python scripts/keeper_check.py pattern f1_r3)"`.
The keeper stops the run at the first clean deterministic lap 2 <= 95 s of any evaluation (aided). Its local
messages still say "reportable"; `runs/mac/keeper_f1.sh` is gitignored and was not edited.

## What to watch at launch (from the review)
status.json `init_buffer.bonus_relabel` (should equal the dry-check numbers), warmup.jsonl `mean_q_pre_exit` per
block + any "pre-exit mean Q_min moved" warning, step-0 / 20k / unfreeze evals and critic_check, sto_crash_U*,
det clean rate vs f1_r2's 41 %.
