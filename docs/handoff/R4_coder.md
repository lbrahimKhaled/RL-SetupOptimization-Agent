# R4 implementation: rl-coder report (8 Nov 2026)

Spec: `docs/specs/R4_speed_profile.md` (Ibrahim's §1 decisions are fixed) + `docs/specs/R4_review.md` (binding, reqs 1-7,
[ARCH] values), DECISIONS #26. Entry term paid at the entry crossing with take-back on a terminal failure before that
unit's exit (lead's choice). Nothing in ocd_first_lap / ocd_first_lap-f1 was touched; `corners.py` unchanged (table key
`6c61b28a4876`); `runs/mac/D2/*` only read (f1_r3 kept running; the f1_best_w35 buffer is unchanged after the dry
check, checked by size + mtime). f1_r4 is not launched.

## Changes (file:line at the commit)
| Req | Where | What |
|---|---|---|
| 1 | `rlsetup/driving/config.py:16-18, 45-74, 156-167, 225-261` | `RewardConfig.speed_k=0` (k_v), `speed_sigma=.3` (σ) with validation (finite >= 0, σ > 0, speed_k > 0 needs time_pbrs). `CornerBonusConfig.mode` ('exit' \| 'entry_exit'), `bonus_in=bonus_out=10`, `sigma=.3`. `EnvConfig.speed_on / entry_exit_on / r4`. `to_dict` leaves the R4 keys out at their defaults, so every pre-R4 config.json is byte-identical. |
| 2 | `rlsetup/driving/speed_profile.py:23-51` (new) | `closeness` g, `closeness_rho`, `speed_potential` (Φ_v, g), `entry_points`. |
| 2 | `rlsetup/driving/env.py:80-88, 196-207, 448-450, 487, 600-644` | corner table also built for `speed_on`; Φ_v at hand-over; per decision `shaping_v = γΦ_v(s′, v′) − Φ_v`, added after the Φ_e shaping; Φ_v′ = 0 after off_track / nonfinite / wrong_way, real after time limit / stall / lap complete (as Φ_e); info `shaping_v, phi_v, g_v, episode_shaping_v`. speed_k = 0: nothing computed. |
| 3 | `speed_profile.py:54-112`, `env.py:201-204, 643-645` | `EntryExitTracker`: two unchanged `corners.CornerTracker`s (entry view: exit_m := window start). Entries paid at the crossing (B_in g) and kept pending; exit paid (B_out g) and clears pending; terminal failure forfeits the decision's crossings (`settle_events`) and takes back pending entries (negative term, kind `taken_back`); truncation takes nothing back; hand-over at / inside a window: only the exit. Signed `info['corner_bonus']` into `BonusNStepReplayBuffer`; info `episode_corner_taken_back`, `episode_corner_takebacks`. The env reward is unchanged. |
| 4 | `rlsetup/driving/relabel.py:134-392` | `relabel_speed_profile(rb, src, dst, table, grid, max_unresolved)`: chains per env column from `rb.pos`, split at dones / obs breaks; Δs per row from `(r + b dt − (γΦ_e′ − Φ_e))/a`; anchor = 2-D rigid fit (Kabsch) of the 60 preview points to `grid.at(s + d_k)` (1 m grid, two fine searches + parabola) at the sharpest-curvature row; verified at >= 3 rows within 0.05 m, any conflict -> chain unresolved; F_v added in float64 -> float32; entry / exit terms replayed per decision (interpolated in the decision). Above `max_unresolved` nothing is written (`aborted`). `_check_speed_relabel` refusals. |
| 4 | `rlsetup/driving/agent.py:55-60, 298-398` | `RelabelUnresolved`; `preload_buffer(..., table, grid, aux)`: refusals before the load (no table, R1+R4 combined, source already with Φ_v / different reward / action repeat; exit-bonus source into entry_exit without speed relabel), relabel after adoption, `aux['s']` = recovered s, result key `speed_relabel`. |
| 4 | `rlsetup/driving/train.py:201-212, 392-450, 610-716` | `r4_provenance`; `_offline_critic` probes `mean_q_pre_entry` / `mean_q_pre_exit` on rows <= 150 m before an entry / exit (recovered s, `buffers.pre_point_cells` :106) with the R3 warning rule per probe; fallback on `RelabelUnresolved`: fresh buffer, `fallback_steps` = 200k frozen-actor transitions, then `DeferredOfflineCritic` (`callbacks.py:644`), then the usual critic warm-up; status `relabel_fallback`, `planned_unfreeze_at`, event `relabel_fallback`; resume keeps a pending fallback. |
| 5 | `rlsetup/driving/train_config.py:122-123, 153-159, 224-225, 279-290, 461-480, 501` | `WarmStartConfig.speed_relabel`, `speed_relabel_max_unresolved=.001`, `fallback_steps=200_000`, `probe_before_m=150`; `EvalConfig.full_throttle_pedal=.95`, `brake_pedal=.1` (PLACEHOLDER); omitted from config.json unless an R4 term is on. Preset `f1_r4` = f1_r2 + speed_k 20, σ .3, entry_exit B_in = B_out = 10, never fade (`hold_until=None`), `relabel=False, speed_relabel=True`, offline critic 50k, focus seeded from f1_best_w35 (as f1_r2). Init = f1_best_w35 `last.zip` + its buffer. |
| 6 | `rlsetup/driving/callbacks.py:257-290`, `evaluate.py:59, 392-491`, `scripts/evals_to_tb.py:85-104` | training-window / per-eval `train_r4` (episode shaping_v, corner terms, take-backs); eval trace `g_v`, `shaping_v`; `lap2_entry_exit` per unit (rho / g / kind in + out, taken_back, brake_onset_m vs entry, v_min) and `lap2_r4` (entry / exit / taken-back sums, mean g, shaping sums, front-lock fraction, per straight mean g / full-throttle share / brake share / top speed); TensorBoard `r4/*`, `straight/*`, `corner/*_U*`, `train_r4/*`. In entry_exit mode `lap2_bonus_sum` = net terms. |
| 7 | `tests/test_r4.py` (new, 30 tests), `tests/speed_truth.py` (new helper), `tests/test_r2.py:254, 682` | below. |
| dry | `scripts/r4_dry_check.py` (new) | below. |
| docs | `rlsetup/driving/README.md`, `scripts/README.md`, `tests/README.md` | module / script / test tables. |

## Interpretations (flag if wrong)
1. **f1_r2's real config.json does not round-trip byte-equal, already at HEAD before R4**: it was written before R3,
   which added `focus_update.init_state` and `warm.pre_exit_*` to aid presets. The diff is exactly those 4 lines (a
   test pins this). f1_best_w35 and f1_r3 round-trip byte-equal. R4 adds no key to any of them.
2. **Anchor thresholds** (relabel.py:149-157): 24 anchor candidates per chain, match RMS <= 0.05 m, uniqueness (second
   minimum > 0.25 m RMS at > 10 m away), residual curvature >= 0.05 (a straight-only preview does not fix s), agreement
   <= 0.05 m at >= 3 rows (review). Unresolved rows keep their old reward / bonus (only matters below the 0.1 % limit).
3. The relabel does not set the R2 `labelled` flag (the rows were labelled by the source run's setting).
4. Entry / exit term relabel interpolates the crossing inside the stored decision (the env interpolates inside the 50 Hz
   tick): bonus error up to ~1.3e-3 per term in the ground-truth test, reward error <= 1.3e-5.
5. `brake_pedal = 0.1` for brake share / onset is a PLACEHOLDER.
6. The code change means f1_r3 resumes need `--allow-code-change` (as before).
7. A QSS cache-key warning (`qss_profile_Shanghai_p0.npz`) appears in tests; it is pre-existing (also at HEAD).

## Tests (`tests/test_r4.py`)
g (symmetry, zero at and beyond ±σ, 1 at v*, rho form); config validation; to_dict omission, all presets round-trip,
no R4 key in pre-R4 presets; real config round-trips (f1_best_w35, f1_r3 byte-equal; f1_r2 diff = the 4 R3 lines);
golden rollouts unchanged (legacy `0007e281…`, time_pbrs `3104af11…` from HEAD, also with entry_exit on and
speed_k 0); Φ_v telescopes (Σγ^t F = γ^T Φ_T − Φ_0) for time limit and crash (Φ_T = 0, F_T = −Φ_{T−1}),
r_on − r_off = shaping_v to 1e-9, obs identical; Φ_v from Track.path s and |v|; tracker: paid once per pass,
take-back, forfeit in the failing decision, crash after exit, truncation, hand-over at / inside / before the
entry and at the exit, unpaid outside σ, window through s = 0; env integration (crash take-back, stall no
take-back, wrap); relabel vs env ground truth (|Δs| <= 0.01 m, r <= 1e-4, bonus <= 2e-3, dones / timeouts
untouched); wrapped full buffer whose oldest row is mid-episode; abort writes nothing; relabel refusals;
`pre_point_cells`; `preload_buffer` speed relabel + refusals + `RelabelUnresolved` leaves the model buffer
untouched; `DeferredOfflineCritic` runs once with no grad step before it; `_offline_critic` entry / exit probes +
warnings; preset f1_r4 (differs from f1_r2 only in the listed fields, w = 1 to 1e9); keeper counts aided evals;
entry points / table key `6c61b28a4876` (needs_f1); lifetime offset 12,350,000 (needs_src); `lap_r4`;
evals_to_tb scalars; TrainStats R4 accumulation.

## Test results (8 Nov, M4, with f1_r3 training)
- `tests/test_r4.py`: 30 passed in 11 s (f1 path); 28 passed, 2 skipped without it.
- Full suite `PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python -m pytest -q`:
  **404 passed, 4 skipped, 2 failed in 329 s** (threads limited to 2, f1_r3 training alongside). The 2 failures are the known pre-existing ones (`test_provenance::test_every_imported_ocd_lap_module_is_hashed` (cars.py), `test_qss_profile::test_cached_file_matches_fresh_computation` (KeyError 'provenance')).
- Bit identity against HEAD (scratch worktree at ad3b879): all 13 pre-R4 presets' config.json identical; legacy and
  time_pbrs rollout digests identical, both ocd_first_lap paths.

## Dry check (real f1_best_w35 buffer, in memory, nothing written)
`VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python scripts/r4_dry_check.py`
- `--preset f1_r4`: init_from `runs/mac/D2/f1_best_w35/seed0/checkpoints/last.zip`, buffer `replay_buffer.pkl`,
  lifetime offset 0 + 12,350,000 = **12,350,000**; corner table key `6c61b28a4876`.
- Buffer: NStepReplayBuffer (adopted), full, pos 136,390, 4 envs; load 0.8 s, relabel 13.5 s, wall 15.8 s, peak 1.73 GB.
- Chains 765, resolved 765, **unresolved 0 (0.0 %)**, integrity breaks 0. Anchor RMS median 2.4e-6 m, max 1.5e-4 m;
  agreement max 2.6e-4 m; verified rows median 19; anchor curvature min 2.41.
- 1,000,000 transitions = 372.45 laps driven. F_v: mean −0.110, std 0.461, min −20.0, max 5.48, p1 −1.51, p99 1.30,
  sum per lap −296.0.
- Entry / exit terms: entry paid 2,444, entry unpaid 270, entry taken back 134, exit paid 2,487, exit unpaid 67;
  entry sum 9,900.9, exit sum 20,307.2, taken back 549.8, net 29,658.3 = **79.6 per lap**; rows nonzero 5,065,
  negative 134.
- Probe rows: pre-entry 163,286, pre-exit 259,660.
- sha256 relabelled rewards `ffc62262a3619a49b2fe4bdc2570d06feede3991c5614612f2f5725071a4b768`, bonuses
  `77458714c93059a172b537f1744b6db783d033f7938fbfb58e204122953d4b2c`. Source file unchanged.

## Launch (stop f1_r3 first, per DECISIONS #26)
```
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 caffeinate -dimsu nohup platforms/mac/train.sh --preset f1_r4 --seed 0 --run-name f1_r4 >> runs/mac/f1_r4_stdout.log 2>&1 &
pgrep -fl "$(.venv/bin/python scripts/keeper_check.py pattern f1_r4)" && nohup runs/mac/keeper_f1.sh f1_r4 >> runs/mac/keeper_f1_r4.log 2>&1 &
.venv/bin/tensorboard --logdir_spec "f1_r4:runs/mac/D2/f1_r4/seed0/tb_evals_lifetime,f1_r3:runs/mac/D2/f1_r3/seed0/tb_evals_lifetime,f1_r2:runs/mac/D2/f1_r2/seed0/tb_evals_lifetime,f1_best_w35:runs/mac/D2/f1_best_w35/seed0/tb_evals" --port 6006 --host 127.0.0.1 --reload_interval 30
```
Watch at start: status `init_buffer.speed_relabel.unresolved_fraction` (0 expected), no `relabel_fallback` event,
warmup.jsonl `mean_q_pre_entry` / `mean_q_pre_exit` settling, then `r4/*`, `straight/*`, `corner/*` in TensorBoard.
