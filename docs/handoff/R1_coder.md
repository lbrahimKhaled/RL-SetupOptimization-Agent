# R1 coder report (rl-coder, 26 Oct 2026)

Spec: `docs/specs/R1_time_optimal.md` (47f1d8d), §2 and the tests of §7. Not committed (lead commits after QA). ocd_first_lap untouched; `runs/D2/full1` only read.

## Files changed
| File | Change |
|---|---|
| `rlsetup/driving/config.py` | `RewardConfig.mode` ('ref_tracking' default / 'time_pbrs'), `shaping_k` = 1.0, `penalty_const` = 100 (both PLACEHOLDER); `REWARD_MODES`; `__post_init__` validation (mode in set; shaping_k, penalty_const finite >= 0; 0 < gamma <= 1); docstring for both modes. |
| `rlsetup/driving/env.py` | `TERMINAL_FAILURES`; time_pbrs branch in `step()` (per tick a ds - b dt - [terminal failure] P; stall = truncation, no P; per decision F = gamma Phi(s') - Phi(s), Phi = -k\|e\|, Phi(s') = 0 only after a terminal failure); `_phi` set in `reset()`; `_trunc_cause` ('time_limit' / 'stall' / None); info `shaping`, `phi`, `episode_shaping`, `reward_mode` (reset info: `phi`, `reward_mode`); `laps[...]['cost_r1'] = time`, `cost` stays legacy in both modes; module docstring. Legacy branch keeps the base `r` and the same op order. |
| `rlsetup/driving/evaluate.py` | `TRACE_KEYS += shaping, progress_m`; `progress_jumps()`; `lap_metrics` adds `cost_r1`, `share_abs_e_gt_1m`, `share_abs_e_gt_3m`, `shaping_sum`, `progress_jumps`; `evaluate_model` adds `reward_mode`. |
| `rlsetup/driving/train_config.py` | `WarmStartConfig`, `StopConfig`; `TrainConfig.warm`, `.stop`, `.init_from`, `.strict_provenance`; `init_mode` 'warm'; presets `r1_warm`, `r1_warm_smoke`; `smoke` gets `strict_provenance=False`; `full` unchanged. |
| `rlsetup/driving/agent.py` | `InitMismatch`, `warm_start()`, `param_digest()` (sha256 of a module's state_dict or a tensor, for A4). |
| `rlsetup/driving/callbacks.py` | `PlateauStop`, `plateau_reached`, `plateau_entry`, `critic_check_value`; `DrivingEvalCallback` critic check + `records`; `PhaseCallback` 'unfreeze' event carries actor / log-alpha digests; `TrainStatsCallback` logs `ep_shaping_mean`. |
| `rlsetup/driving/train.py` | `--init-from` (absolute path; required iff init_mode 'warm'), `EXIT_INIT_MISMATCH = 6`; warm branch (no output-bias init, no demos/BC/offline, baseline = probe of the loaded actor, `init_from` in provenance + status, unfreeze + critic check at `warm.critic_warmup_steps`); PlateauStop wired (stop reason 'plateau'); `critic_not_converged` -> `_warn`; dirty check on `strict_provenance` in run and resume; resume: frozen phase for 'bc' and 'warm', plateau history and critic-check state from evals.jsonl, init_from never re-applied; status `lifetime_steps_offset` / `lifetime_timesteps`. |
| `scripts/evals_to_tb.py` | `line/mean_abs_e_m`, `line/max_abs_e_m`, `line/share_abs_e_gt_3m`, `lap/min_edge_margin_m`, `cost/lap2_legacy`, `cost/lap2_tracking`, `cost/lap2_smoothness`, `cost/lap2_r1`, `critic/q_min_over_mc_nstep`, `diag/progress_jumps` (+ `critic/critic_check`). Checked on a copy of full1's evals.jsonl (keys missing in old records are skipped). |
| `tests/r1_golden.py` (new) | the legacy rollout used for the golden hash. |
| `tests/test_r1.py` (new) | spec tests 1-11 (+ evaluation-metric and CLI tests). |
| `tests/test_r1_slow.py` (new) | spec test 12. |
| `tests/test_callbacks.py` | unfreeze-event assertion includes the digests; `resume_fixture(cfg=...)`; fake `_callbacks` / `_learn_and_finish` follow the new signatures. |

## Golden hash (test 1)
`r1_golden.legacy_rollout` = 300 decisions on `EnvConfig(max_episode_s=4)`: steering sweep + pedal steps, forced off-track, stall from 3.5 m/s, a time-limit episode on the main straight, a wrong-way end (ends `off_track, stall, time_limit, wrong_way, off_track, stall`). sha256 over reset obs + per decision (obs f32, reward f64, step_cost f64, terminated, truncated). Generated on the unmodified tree at 47f1d8d before env.py was touched (`git diff aeab814 47f1d8d -- rlsetup tests scripts` is empty, so code-identical to aeab814), twice, identical: `0007e2813d2d28971223460790b45852513768865f08d6d0b3685f5868a1c4c2`. Re-checked after the env edits: same hash.

Mutation check (each applied to env.py, R1 tests run, reverted): Phi(s') = 0 also on truncation -> 3 fail; shaping gamma 0.99 != learner gamma -> 4 fail; stall terminal in time_pbrs -> 2 fail; P depending on \|e\| -> 1 fail.

## Test results (actual output, final code)
- `.venv/bin/python -m pytest -q -m "not slow"` -> **160 passed, 5 deselected in 14.88s** (was 131 before R1; +29 in test_r1.py).
- `.venv/bin/python -m pytest -q -m slow` -> **5 passed, 160 deselected in 120.59s** (existing D1/D2 slow tests + new `test_r1_slow.py`, 54 s alone).
- Plain `.venv/bin/python -m pytest -q` (runs slow too, pyproject has no default deselect) -> 164 passed in 133.56s (before the last two small edits; both subsets rerun after them, above).
- Spec §7 mapping: 1 `test_legacy_rollout_bit_identical`; 2 `test_shaping_exact_and_telescopes_{off_track,time_limit,stall}`; 3 `test_nstep_shaping_telescopes_in_sb3_buffer`; 4 `test_gamma_is_one_object`; 5 `test_stall_is_truncation_in_r1_and_terminal_in_legacy`; 6 `test_lap_costs_independent_of_mode_and_k`; 7 `test_crash_penalty_constant_and_last_shaping` (off-track at two different \|e\| + wrong-way); 8 `test_warm_start_copies_policy_and_alpha`, `test_warm_start_without_critics`, `test_warm_start_refuses_mismatches` (obs space, env config, gamma, net_arch, missing config.json); 9 `test_frozen_phase_trains_critic_only`; 10 `test_plateau_stops_exactly_when_both_bests_stall`, `test_plateau_counts_cumulative_and_threshold_gains`, `test_plateau_callback_and_entries`; 11 `test_old_config_loads_as_ref_tracking` (synthetic pre-R1 dict + the real full1 config.json if present), `test_reward_config_validation`, `test_r1_presets`, `test_dirty_tree_refused_by_strict_presets`, `test_resume_warm_frozen_phase_and_strict_rule`; 12 `test_r1_slow.py::test_mini_r1_warm_smoke_from_mini_pipeline` (mini bc pipeline -> last.zip -> mini r1_warm_smoke 1k steps (frozen, critics train, actor digest = source) -> resume to 3k: still frozen until 2k, unfreeze event at 2k with the step-0 digests, critic_check at the 3k eval; step-0 det eval termination / decisions / lap times equal the source's last eval). Extra: `test_progress_jumps_and_lap_metrics`, `test_critic_check_value`, `test_eval_callback_critic_check_once`, `test_cli_init_from_absolute`.

## Smoke (A2-A6): `--preset r1_warm_smoke --seed 0 --run-name r1_warm_smoke --init-from runs/D2/full1/seed0/checkpoints/best.zip`
Run folder `runs/D2/r1_warm_smoke/seed0`. **Exit 3: state `collapsed` at 20,000 steps** (CollapseGuard), not the 15-min wall limit.
| Check | Result |
|---|---|
| A3 step-0 det lap 2 | **122.84531256503526 s, clean** (`lap_complete`), exact. best.zip written at step 0. Sto 3/3 clean, mean 124.632 s. Lap-2 legacy cost 344.45 (tracking 211.13, smoothness 10.47), cost_r1 122.845, mean/max \|e\| 0.429 / 2.352 m, share > 3 m 0, min edge margin 0.350 m, progress_jumps 0, lap-2 shaping sum +10.53 (spec §1.5 estimate +9.9). |
| A4 actor / log alpha digests | step 0 (after warm_start): actor `2515c2e66384b441...`, log alpha `678fdeeda6daf7b4...`; at the unfreeze event (step 10,000): **identical**. |
| A5 critic_check | logged at the first eval >= 10k, which was the final eval at 20k (after the collapse): 0.0210 (tol 0.08, no warning). Step-0 Q vs MC (old critic, new reward): Q_min - MC^n = -15.72 on MC^n 208.80 (-7.5 %); per-step -5.36 on 198.45. |
| A6 | config.json `env.reward.mode = 'time_pbrs'`, `agent.init_mode = 'warm'`, `init_from` absolute, `strict_provenance` False; provenance.json `init_from` = {path, sha256 `90083f75...e395320`, src_num_timesteps 2,600,000, src_rlsetup_commit f17be23, src_config_sha256, src_alpha 0.04338}; status `lifetime_timesteps` 2,620,000. |
| A2 throughput | evals record env_steps_per_s 327.6 over 0-20k (mostly frozen; the first 5,120 transitions have no gradient step). From the 100k diagnostic below: **279-284 env steps/s in the critic-only phase**, 206 after the unfreeze **including a probe every 5k** (probe = 3.9 s measured) -> ~245 env steps/s without probes. Step-0 eval 11.1 s (det 2 laps + 3 sto laps). |
| Warnings / events | no `warnings` entry; events unfreeze (10k), rollback (15k: probe 659 m vs baseline 1,246.7 m x 0.75 = 935 m, actor_lr -> 1e-4), collapsed (20k: 633 m), det_regression (20k, off_track). Final eval: det off_track, sto 0/3 clean. Train stats 10-20k: 5 off_track / 11 episodes. |

### The collapse is real and not caused by R1 or by this code: diagnostics (all r1_warm_smoke via the CLI, separate run names)
| Run (runs/D2/...) | Change vs smoke | Post-unfreeze probes (baseline 1,246.7 m, threshold 935 m) | End |
|---|---|---|---|
| `r1_warm_smoke` | none (critic-only 10k = ~2,440 critic grad steps) | 15k: 659, 20k (after rollback, lr 1e-4): 633 | collapsed 20k |
| `r1_warm_smoke_cw100k` | `--set warm.critic_warmup_steps=100000` (= r1_warm) | 105k: 678, 110k: 855 | collapsed 110k |
| `r1_warm_smoke_ctrl_reftrack` | `--set env.reward.mode=ref_tracking` (source's own reward) | 15k: 1,028 ok, 20k: 751, 25k: 904 | collapsed 25k (+ critic_not_converged 0.149) |
| `r1_warm_smoke_diag_optstate` | + source Adam states (wrapper script `/tmp/r1diag/warm_with_opt.py`, no repo change) | 15k: 864, 20k: 980, 25k: 812 | collapsed 25k |
| `r1_warm_smoke_diag_buffer_reftrack` | ref_tracking + source Adam states + **full1's 1M-transition replay buffer** (read-only load, `/tmp/r1diag/warm_with_buffer.py`) | 12 probes 15k-70k: 949-1,245, all ok | wall limit 70k; det lap 2 122.802 s at 25k and 122.806 s at 70k (det off_track at 50k), sto 2-3/3 |

cw100k details: during the 100k critic-only phase det lap 2 stayed 122.84531256503526 s at 25/50/75/100k (actor frozen); critic Q_min - MC^n went -15.7 -> -9.6 -> -8.2 -> -8.5 -> -10.8 (MC^n 208.8); **critic_check at the unfreeze eval (100k) = 0.0517 < 0.08** (A5 as specified, passes); 47,440 critic grad steps by the unfreeze; collapse 5k steps later anyway.

Reading (mine, [J]): the post-unfreeze collapse comes from the fresh replay buffer (a narrow on-policy dataset of the near-deterministic loaded actor, so the critic's action gradients off that distribution are unreliable), not from the R1 reward (the ref_tracking control collapses the same way), not from the critic-phase length (10k and 100k both collapse; the critic check passes at 100k) and not from fresh optimizers alone (copying the Adam states does not help). With full1's buffer the same warm start is stable (tested together with the Adam states; buffer without Adam states not run separately; one seed, 60k post-unfreeze steps only). The two wrapper scripts live in /tmp and may vanish; each is ~25 lines that monkeypatch `agent.warm_start` and call `train.main`. Also note: even in the stable configuration the probe dips to 949 m = 0.76 x baseline, so threshold 0.75 sits close to the normal probe noise of this actor (one crashed probe start costs up to ~1/8 of the mean).
**Consequence: launching `r1_warm` as specified will very likely end `collapsed` at ~110k steps.** This needs an architect/lead decision (spec §3 fixes "replay buffer fresh"); options I can see, not chosen: seed the buffer with full1's buffer relabelled to R1 (spec §3 calls relabelling approximate: tick-1 e and e_exit not stored, stall vs off-track ambiguous), mix old and new data, a much lower actor lr / slower unfreeze, or a looser guard. I did not change any of this.

## Deviations from the spec / readings of ambiguous points
1. Golden hash generated on 47f1d8d (unmodified tree, before editing env.py), not by checking out aeab814; `git diff aeab814 47f1d8d -- rlsetup tests scripts` is empty. The rollout also covers stall, time limit and wrong way, not only one forced off-track.
2. `info['termination'] = cause if terminated else None` (a time_pbrs stall keeps `cause='stall'` internally so the eval `lap_complete` check is not taken, but reports termination None / truncation 'stall'). In ref_tracking every cause sets terminated, so the value is unchanged. Stall in the same tick as the time limit -> truncation 'stall'.
3. `warm_start` returns extra keys `src_rlsetup_commit` (from the source provenance.json; None if absent), `actor_sha256`, `log_ent_coef_sha256` (A4). Missing init_from file -> InitMismatch. Env-config comparison = JSON dicts without `reward` (strict: a future EnvConfig field makes old sources mismatch).
4. Critic check: value None (no Q-vs-MC pairs, MC mean 0) also raises `critic_not_converged`. Checked once per run (resume: skipped if any evals.jsonl record has `critic_check`). The record also gets `critic_check_tol`.
5. PlateauStop: "improved" = best(step <= n - window) - best(step <= n) >= min_improvement_s - 1e-9 (float slack, so exactly 0.1 s counts); a best whose first value lies inside the window counts as improved; never any value counts as not improved. PlateauStop is always in the callback list (no-op unless `stop.plateau`).
6. Resume: configs without `strict_provenance` (written before R1) keep the old rule (strict iff preset 'full'), so old smoke runs still resume on a dirty tree. `--init-from` on resume is accepted only if it resolves to the run's `init_from` (never re-applied), else ValueError.
7. CLI: `--init-from` without init_mode 'warm' (or warm without it) -> `sys.exit` with a message (exit 1); `run_pipeline` raises ValueError for the same.
8. Extras: status `lifetime_steps_offset` / `lifetime_timesteps`, state `init_mismatch`, `TrainStatsCallback` `ep_shaping_mean`, the 'unfreeze' event carries actor / log-alpha digests (also for bc runs), `evals_to_tb` also writes `critic/critic_check`, `evaluate_model` reward_mode None for env stubs, `progress_jumps` uses v = max(speed at start, end) of the decision.
9. Side effect (expected, spec §6 #8): env.py / config.py are in the demo-dataset key and EnvConfig gained keys, so the cached `data/demos/demos_1c612cde2770.npz` no longer matches; the next bc-mode run recollects demos (~2-4 min). QSS cache key unaffected.
10. Disk: the five smoke/diagnostic folders hold ~4.8 GB (SB3 pickles the whole preallocated 1M buffer, ~1.2 GB each, even for 20k transitions); 27 GiB free. Safe to delete the diagnostic folders' `replay_buffer.pkl`; I left them.

Not done (as instructed): no commit; full `r1_warm` not launched.

---

# Amendment 1 + QA fixes (rl-coder, 26 Oct 2026, evening; on top of 6226341, not committed)

## Summary
Amendment 1 (preload full1's buffer relabelled to R1, source Adam states, offline critic 25k, online critic 20k, guard 0.65) and the QA findings M1, m1, m2, m5a, m6 (+ the optional m5b, m7) are implemented and tested. On the §D smoke, **S1, S2 and S3 pass and S4 fails, both on the prescribed run and on the single fallback rerun. Per the spec this escalates to the architect.** The full run was not launched.

## Files
| File | Change |
|---|---|
| `rlsetup/driving/relabel.py` (new) | `relabel_to_time_pbrs(rb, env_cfg_src, env_cfg_dst)`, numpy, in place, returns stats; `decision_duration()`. |
| `rlsetup/driving/agent.py` | `warm_start` also copies actor / critic / ent_coef Adam states (`warm.load_optimizers`), sets `model.actor_lr = warm.actor_lr`, also compares `n_steps` and `target_entropy` (QA m7); InitMismatch instead of `assert` (QA m6). New `default_init_buffer()`, `preload_buffer()` (checks class, buffer_size, n_envs, n_steps, gamma, obs_shape, action_dim -> InitMismatch; relabel; `cut_nstep_chains`; `min_train_transitions = 0`; sha256 of the pkl). |
| `rlsetup/driving/train_config.py` | `WarmStartConfig`: `critic_warmup_steps` 20k (was 100k), `load_optimizers`, `init_buffer`, `relabel`, `offline_critic_grad_steps` 25k, `offline_block` 1k, `actor_lr`. `r1_warm`: guard 0.65. `r1_warm_smoke`: total 150k, max_wall_h 0.5 (safety cap only; the step count ends it), inherits warm settings. |
| `rlsetup/driving/train.py` | default init_buffer resolved before config.json is written; QA m2: missing init_from / its config.json / init_buffer -> exit 6 **before** the run folder exists; warm branch: preload + relabel (status `init_buffer`, provenance `init_buffer` path/sha256/transitions), `_offline_critic()` (blocks -> warmup.jsonl `block, grad_steps, critic_loss, mean_q_fixed_batch`; phase `offline_warmup`); CLI `--init-buffer`. |
| `rlsetup/driving/callbacks.py` | QA M1: `DrivingEvalCallback._schedule` adds an eval at `critic_check_at` (if pending and off the grid); QA m1: `critic_check_value` returns None for non-finite input -> `critic_not_converged`. |
| `tests/relabel_truth.py`, `tests/test_relabel.py` (new) | C1, C2, config refusal. |
| `tests/test_r1.py`, `tests/test_r1_slow.py` | C3, C4, preload, CLI `--init-buffer`; QA M1, m1, m2, m5a, m5b/m7 tests; presets / defaults updated; slow test 12 covers preload, relabel, offline blocks, the extra eval at the unfreeze. |

## Deviation: relabel formula refined (needed for C1)
Two parts of the spec formula left C1 errors above its tolerances, measured against env ground truth:
- **Linear first-tick offset (e+e')/2:** non-failure error up to **2.6e-3** (tolerance 1e-3).
- **Failure decisions** that ended after one tick or a shortened tick: errors up to **2.3** (tolerance 0.05). The formula charges a first-tick tracking term the env never charged, and S uses 0.02 s instead of the shortened tick.

Two refinements, both from stored data only:
- **Cubic Hermite e(t)** with de/dt = v_x sin h + v_y cos h, taken from obs and next_obs.
- **Duration D of failure decisions**, recovered from next_obs's body accelerations. env.py computes these over the true elapsed time: a_x' = dv_x/D − r̄ v̄_y and a_y' = dv_y/D + r̄ v̄_x. A least-squares fit gives D, rounded to 2 ms. Ticks then end at min(j·dt, D), and S uses the first tick's length. If |A| < 1 m/s² (e.g. a stalled car), the full-length formula is used.

Measured max errors after refinement: **non-failure 9.0e-5, failure 6.1e-5**. On full1's buffer, D was identified for 106 of the 108 failure rows. Everything else follows §A: P_old, stall rule, term_new, F, flags.

## Tests (actual output, final code)
- `.venv/bin/python -m pytest -q -m "not slow"` -> **172 passed, 5 deselected in 16.23s**
- `.venv/bin/python -m pytest -q -m slow` -> **5 passed, 172 deselected in 125.80s** (incl. updated test 12)
- C1 `test_relabel_matches_env_ground_truth`: 13 scenarios, 355 transitions, covering off-track ×6, stall ×3, time limit and wrong-way. dones and timeouts are exactly equal; reward max errors are 9.0e-5 (non-failure) and 6.1e-5 (failure).
- C2 `test_relabelled_nstep_samples_match`: SB3 n-step samples for the same indices have equal dones, discounts and next_obs; returns match within the C1 tolerances.
- C3 `test_preload_buffer_refuses_mismatches`: refuses buffer_size, n_envs, gamma, obs_shape and a missing file.
- C4 `test_warm_start_copies_optimizer_states`: Adam states equal the source's (the source was trained 3 steps so the moments are non-empty).
- QA: `test_critic_check_eval_scheduled_at_unfreeze` (interval 50k, check at 20k -> records at 0, 20k, 50k, 100k; the check only at 20k), `test_nan_critic_check_is_not_converged`, `test_missing_init_files_refused_before_run_dir`, `test_nonfinite_under_time_pbrs` (NaN at physics step 5 and 15; exact asserts as QA item 3), `test_warm_start_refuses_more_mismatches` (action space, n_critics, critic LayerNorm, n_steps, target_entropy).

## Relabel of full1's buffer (S1)
`runs/D2/full1/seed0/replay_buffer.pkl` (sha256 `b7da046f…c2de5`): 250,000 rows × 4 envs, full, saved at step 3,000,000, so the window is steps 2.0M–3.0M. Load 0.7–1.8 s, relabel 0.7 s.

| Quantity | Relabel | full1 train_stats, 2.0M < step ≤ 3.0M |
|---|---|---|
| failures (dones ∧ ¬timeouts) | 108 | 108 (103 off_track + 4 wrong_way + 1 stall) |
| stalls | 1 | 1 |
| timeouts (old) | 612 | 612 time_limit |
| nonfinite | 0 | 0 |

**S1 classification: exact.**

Mean R_new − R_old per non-failure row is **+0.0967** (S 0.0449 + T 0.0496 + F 0.0023), not the spec's ≈ +0.06 + F = 0.062. The spec figure used the deterministic eval lap's smoothness (0.0034 per decision). Stochastic driving by best.zip, measured in the env (6 training episodes, 7,946 decisions), gives smoothness 0.041 and tracking 0.074 per decision, so the buffer's 0.045 / 0.050 is consistent. C1 confirms the per-row exactness. I count S1 as passed with this explanation.

## §D smoke, prescribed run: `--preset r1_warm_smoke --seed 0 --run-name r1_warm_smoke_a1 --init-from runs/D2/full1/seed0/checkpoints/best.zip`
Exit 0, `completed` at 150,000 steps.
- **Offline critic:** 25,000 grad steps in 150.6 s; critic loss 11.9 -> 10.1 (full1's own late-run mean is 4.7, median 2.4); mean Q on the fixed batch 194.4 -> 198.4.
- **Online critic:** 20k env steps in 84 s.
- **RL:** 130k env steps in 652 s, about 220 env steps/s including a probe every 5k (≈ 255/s without probes).

| | Result |
|---|---|
| S2 | step-0 det lap 2 **122.84531256503526 s**, clean -> PASS |
| S3 | extra eval at **step 20,000** (actor still the loaded one: lap 2 unchanged): critic_check **0.0469** <= 0.08 -> PASS. Q_min − MC^n: −9.99 (step 0) -> −9.80 (20k). |
| S4 | **FAIL**: 1 rollback (115k: probe 775 m = 0.622 × 1,246.7); det clean 1/6 post-unfreeze evals (25k only; 50k–150k off_track); sto clean 2,1,1,0,0,3 of 3 at 25k–150k; best post-unfreeze clean lap 2 124.616 s > 123.85. No collapse; other probes 0.71–0.97. |

Post-unfreeze evals (det lap 2 / sto clean / lap-2 tracking cost / max \|e\| / min edge margin):
- 25k: 124.616 s, 2/3, 640, 5.59 m, 0.27 m
- 50k–125k: det off_track
- 150k: det off_track in lap 2 (lap 1 123.92 s), sto 3/3 (125.2–125.4 s)

Training failures: 19 off_track and 1 stall in 150k steps. The rate is about 1.3 per 10k, compared with 2 per 10k while still frozen.

## Fallback (the one prescribed rerun): `... --run-name r1_warm_smoke_a1_fallback --set warm.load_optimizers=False --set warm.actor_lr=0.0001`
Exit 0, `completed` at 150,000.

| | Result |
|---|---|
| S2 | 122.84531256503526 s -> PASS |
| S3 | critic_check **0.0478** at step 20,000 -> PASS |
| S4 | **FAIL**: 0 rollbacks, no collapse, all 26 probes ok (min 0.69 × baseline); best post-unfreeze clean lap 2 **123.789 s** (<= 123.85, ok); but det clean **2/6** (25k 123.789 s, 50k 123.881 s; 75k–150k off_track) and sto **1/3 at 75k, 0/3 at 100k** (3/3 at 25k, 50k, 125k, 150k). |

Trend of the lap-2 metrics:

| Step | Lap-2 tracking cost | max \|e\| | min edge margin | Sto laps |
|---|---|---|---|---|
| 20k | 211 | 2.35 m | 0.35 m | 124.5–124.8 s |
| 25k | 411 | 3.71 m | 0.24 m | |
| 50k | 523 | 5.39 m | 0.18 m | |
| 125k–150k | | | | 125.5–126.2 s |

Over the same span α went 0.0434 -> 0.0335 and Q_min − MC^n went −10.0 -> +5.6 (100k) -> −4.1 (150k). Training failures: 18 off_track and 1 wrong_way in 150k.

## Reading (mine, [J]); for the architect
- **Buffer preload fixed the fresh-buffer collapse.** With fresh optimizers and actor lr 1e-4, the guard never fired. The CollapseGuard criterion is no longer the problem.
- **What fails is S4's eval criteria.** Within about 30k steps of the unfreeze, R1 moves the driver off ξref, as intended: lap-2 tracking cost doubles, max \|e\| goes 2.4 -> 5.4 m and the edge margin shrinks. Over the same span the laps get slower, not faster (det 122.85 -> 123.8–123.9 s; sto 124.6 -> 125.5–126 s), and the deterministic lap starts crashing.
- Looks like an early re-optimisation transient of the new objective, or a real regression; 130k steps cannot separate the two. Nothing in the runs points to an implementation fault:
  - the relabel is exact to < 1e-4 against the env;
  - the critic check is 0.047 at the unfreeze;
  - the step-0 lap is reproduced exactly;
  - the actor stays frozen through both critic phases (digests identical).
- Choices I see, none of which I made: run longer before judging S4 (the plateau logic / 3M cap); loosen S4 for the transient; keep a small tracking term in R1 initially; lower the actor lr further; or a from-scratch R1 run. These are architect / Ibrahim calls (§11-style decisions).

## Housekeeping
- **Deleted** (as allowed): the four diagnostic folders `r1_warm_smoke_cw100k`, `r1_warm_smoke_ctrl_reftrack`, `r1_warm_smoke_diag_optstate`, `r1_warm_smoke_diag_buffer_reftrack`.
- **Kept:**
  - `r1_warm_smoke` (old results above), 1.2 GB;
  - `r1_warm_smoke_a1`, 1.3 GB;
  - `r1_warm_smoke_a1_fallback`, 1.3 GB.

  Each holds a ~1.3 GB replay_buffer.pkl that can be deleted if the lead does not want to resume them. 29 GiB free.
- `data/demos` cache is still invalidated, as noted before. Nothing committed; full `r1_warm` not launched.
