# R2 review — rl-architect (7 Nov 2026)

**Verdict: CHANGES-REQUIRED on the draft → this review is the spec (APPROVE-WITH-NOTES once Ibrahim answers F1–F2).**
Binding for rl-coder; where this review and `R2_corner_bonus.md` conflict, **this review wins**. Labels: [F] formulation/decision, [C] code or measurement in this session, [P] paper (via `docs/research/plateau_fixes.md`), [J] judgement. §1 decisions of the draft are not changed.

## Findings for Ibrahim (only he can decide)
- **F1. Keeper vs Decision 2.** A clean lap ≤ 95 s before 4M bonus-free steps can't be reported (Decision 2). The spec therefore only lets the keeper stop on a *reportable* eval (step ≥ 8.0M); earlier ≤ 95 s laps are logged as event `aided_target_lap` and the run continues. Alternative: stop at the first ≤ 95 s lap and then run ≥ 4M bonus-free steps from that checkpoint.
- **F2. "Stayed on track through the whole window" with a start inside the window.** Failures end the episode, so any exit crossing within an episode is clean; the pre-hand-over part is driven by the settle controller (always on track, resampled otherwise). Spec: pay iff the exit point is crossed after hand-over and the paying decision has no failure (reward stays a function of the state). Alternative "arm at window start" leaves ~41 % of first passes unpaid (windows cover 41 % of the lap) for a reason the observation can't see.
- **F3. No control run (Decision 3)** is fine for "get a faster driver"; not sufficient to claim "R2 caused the gain" [P: FREI24/TRACEY23 via research A.5.4]. Free weak comparison: f1_best_w35's own continuation 9.2M → 12.35M (best stayed 101.87 s, recent clean laps 102.2–102.6 s) [C]; f1_r2 is plotted on that same step axis.
- **F4. B = 3 per corner is limited by the crash-risk bound** (research A.3) [J]. Expected effect: ~10–20 % extra pull on exit speed on top of what time_pbrs already pays. B = 6 doubles the pull and tolerates up to ~1.8 % extra crash risk per corner. Default 3.
- **F5.** Focused starts and low-noise rollouts don't change the reward. The spec keeps them on in the bonus-free phase too, so the reported driver is trained with them. Keep on (default) or switch off at 4M?

Technical notes for the lead: (a) the corner list differs from the lead's diagnostic (§R2-2; the diagnostic missed the T14 hairpin at 4749 m); (b) best.zip is from step 9.2M, while the source replay_buffer.pkl is the end-of-run buffer (12.35M) [C]; same pattern as r1_warm, covered by the critic phases.

## R2-0. Facts used [C]
- Source: `runs/mac/D2/f1_best_w35/seed0`. best.zip num_timesteps 9,200,000, actor_lr 1e-4 (rollback at 30k), lap 2 101.87194035987982 s. Last eval 12.35M: α 0.0546, entropy −2.95, Q_min mean 236, Q_min − MC^n +1.3 (0.56 %). Training failures total: off_track 2448, wrong_way 35, stall 10. Throughput 263 env steps/s, 131 grad steps/s (UTD 0.5). Last 40 evals: deterministic clean 9/40, stochastic clean rate 0.665.
- Track.path L = 5388.77 m; w3.5 line 5364.67 m, 2684 nodes. `qss_lap.lap_time(path=line, ggv_data=sha_f1_best cache)` = 93.8253 s; Track.path profile 100.162 s.
- Source buffer is already `time_pbrs` → `relabel_to_time_pbrs` would raise (it requires ref_tracking). `preload_buffer` refuses on `type(rb) is not type(cur)`. `warm_start` compares the env dicts minus `reward` → new EnvConfig fields would refuse. `demos.dataset_key` hashes `env_cfg.to_dict()` → new fields would invalidate every demo cache.
- SB3 2.9.0 `OffPolicyAlgorithm._setup_model` adds `n_steps`/`gamma` to replay_buffer_kwargs **only** when `replay_buffer_class is None`. `NStepReplayBuffer._get_samples` builds n-step sums from `self.rewards[indices, env_indices]` at sample time. `ReplayBuffer.add(..., infos)` receives the step infos (terminal ones included).
- OCDBackend (ocd_first_lap-f1): no state snapshot/restore. `reset(setup, StartPose(x, y, yaw, speed))`. OCD has `initial_state.*` parameters but only reachable through construction-time `overrides`. Gear/rpm live in the Python Powertrain. D1 R5: ΣFz needs 1.7–2.4 s to settle after a reset at speed. **→ Restoring visited states is not feasible without editing ocd_first_lap; use settle.**
- The probe uses `options={'start_s': …}` and runs deterministically; eval uses `evaluation=True`; stochastic eval laps call the actor directly (not `_sample_action`).

## R2-1. Architecture of the bonus (env vs buffer) — CHANGES-REQUIRED
1. **The env's returned reward stays exactly time_pbrs** (bit-identical with the bonus on or off). The env only reports `info['corner_bonus']` (unweighted, summed over the decision's ticks, 0 when nothing is paid), `info['episode_corner_bonus']`, and `info['corner_events']` (list, only when an event happens: `{unit, kind: 'paid'|'forfeit'|'exit_unpaid', v_exit, v_qss_exit, rho, bonus}`). VecMonitor returns, eval returns and `laps` stay bonus-free and comparable with f1_best_w35; the bonus enters learning only through the buffer [J].
2. 25 Hz consistency: crossings detected per 50 Hz tick inside the action-repeat loop, summed per decision. At ≥ 2 m per decision and ≥ 150 m between exits, at most one payment per decision.

## R2-2. Corner table — APPROVE (rule below)
Module `rlsetup/driving/corners.py`, built once in the main process (like the QSS profile), passed picklable to workers and eval envs:
1. **v_qss on ξref:** `qss_lap.lap_time(setup, track, ggv_data=(qss.meta['ggv'], qss.meta['traction'], None), car, ers, drs=False, path=ref.path)['v_profile']` (same call as `qss_profile.line_lap_time`). Cumulative QSS time t_line by the trapezoid rule; test: t_line[-1] == lap_time_s to 1e-6.
2. **Map to Track.path:** `track.match(xy, hint)` per line node with a running hint; unwrap; interpolate v onto a 1 m Track.path grid with period L.
3. **Apexes:** `scipy.signal.find_peaks(-v, prominence=5 m/s)` on the grid rolled to start at the global max. [J: the 2–4.5 m/s kinks at 614, 2094 and 2630 m have no exit phase worth rewarding.]
4. **Merge** consecutive apexes closer than 150 m into one unit [J].
5. **Window start** b_k = argmax of v_qss on (last apex of unit k−1, first apex of unit k) (start of braking).
6. **Exit point** s_out,k = last apex + min(150 m, (b_{k+1} − last apex) − 30 m) [J: keeps the exit before the next braking point; only U3 affected]. Window_k = [b_k, s_out,k]; assert disjoint and each ≥ 100 m.
7. **Result [C]** (sha_f1_best, w3.5; ρ = median / range over the 17 clean deterministic f1_best_w35 laps since 8M, single sample):

| Unit | Apex(es) m (v_qss) | Window [b, s_out] m | v_qss at exit | ρ launch median (min–max) |
|---|---|---|---|---|
| U1 | 813, 877 (24.1, 21.2) | [517, 1027] | 55.2 | 0.912 (0.885–0.921) |
| U2 | 1585 (21.6) | [1467, 1735] | 60.9 | 0.899 (0.877–0.911) |
| U3 | 2329 (49.0) | [2062, 2419] | 57.5 | 0.856 (0.834–0.871) |
| U4 | 2531 (30.9) | [2449, 2681] | 57.5 | 0.924 (0.868–0.939) |
| U5 | 3112, 3167 (19.4, 27.9) | [3007, 3317] | 56.0 | 0.890 (0.869–0.910) |
| U6 | 4749 (13.5) | [4634, 4899] | 60.4 | 0.896 (0.877–0.907) |
| U7 | 5128 (30.8) | [5042, 5278] | 63.4 | 0.930 (0.921–0.941) |

Every crash cluster (650, 700–850, 1550, 2350–2500, 3100, 4700, 5100 m) lies inside a window. Test: table matches ±2 m.
8. **Cache** `data/corner_table_<key12>.json`. Key = sha256 of {qss.meta['key'], ref.digest, CornerBonusConfig detection fields, sha256 of corners.py}. Recompute in memory on mismatch (same policy as qss_profile).

## R2-3. Bonus value and payment rule — CHANGES-REQUIRED (definition fixed here)
1. **Exit speed:** single sample at the s_out crossing; |v| = hypot(v_x, v_y) linearly interpolated between the tick's start and end at φ = (X − d0)/(d1 − d0), d = the env's cumulative Track.path distance `_dist` [J].
   ρ = v_exit / v_qss(s_out); f(ρ) = clip((ρ − 0.75)/0.25, 0, 1); bonus = B·f, B = 3.0. Cap f(ρ ≥ 1) = 1 (Decision 1). Floor 0.75: launch ρ (0.83–0.94) in the graded range; botched exits (< 0.75) earn 0 [J].
2. **Payment, once per pass:** at hand-over set d_out,k = (s_out,k − s_handover) mod L per unit; thresholds d_out,k + jL. Pay when the high-water mark of `_dist` first exceeds the next threshold, then advance by L. Backward driving / re-crossing never pays twice; a projection jump pays at most B (ρ capped).
3. **Start inside a window** (F2): paid if the exit is crossed after hand-over.
4. **Forfeit:** if the decision that crosses s_out ends in a terminal failure (off_track / nonfinite / wrong_way) at any tick, bonus 0 (event 'forfeit'). Stall / time-limit truncation does not forfeit.
5. **Sanity [C/J]:** per-lap bonus at launch ≈ 12.7, max 21, vs lap return ≈ 5290 (0.24–0.40 %). B = 3 vs crash cost P + V ≈ 336 → can't justify > 0.9 % extra crash risk per corner [P: research A.3]. Q shift ≤ ~3–5 (≈ 2 % of Q).
6. Evaluation envs compute the same info (logging only).

## R2-4. Replay buffer — CHANGES-REQUIRED
New module `rlsetup/driving/buffers.py`, class `BonusNStepReplayBuffer(NStepReplayBuffer)`:
1. Arrays `bonuses` (float32, [buffer_size, n_envs]) and `labelled` (bool); attribute `bonus_weight`. `add(...)`: `bonuses[pos] = [info.get('corner_bonus', 0.) for info in infos]`, `labelled[pos] = True`, then `super().add`.
2. `_get_samples`: copy of SB3 2.9.0's (pinned; `check_sb3_version`) with **one change**: `rewards_seq = self._normalize_reward(self.rewards[idx, env], env_)`, then **if `self.bonus_weight != 0`**: `rewards_seq = rewards_seq + np.float32(self.bonus_weight) * self.bonuses[idx, env]` (float32). At w = 0 bit-identical to NStepReplayBuffer. Masks, timeouts and the pos − 1 trick unchanged.
3. `build_model`: when `env.corner_bonus.enabled`, pass `replay_buffer_class=BonusNStepReplayBuffer, replay_buffer_kwargs={'n_steps': a.n_steps, 'gamma': cfg.env.reward.gamma}`. Otherwise unchanged.
4. **Warm-start buffer:** `preload_buffer` accepts a plain `NStepReplayBuffer` for an R2 model and converts it **in place**: `BonusNStepReplayBuffer.adopt(rb)` sets `__class__`, adds `bonuses = 0`, `labelled = False`, `bonus_weight = 0` (no 1.3 GB copy). Recomputing is impossible (no s in the observation) [C]; a fresh buffer collapsed in R1 [C]. Bonus 0 for the 1M old rows: bias ≤ ~1.8 on Q at exits, falls to 0 as the FIFO turns over (~1M steps) — a natural ramp-in [J]. `cut_nstep_chains` as now; relabel.py not used.
5. Pickling: class in an importable module, so save/load and resume keep `bonuses`/`labelled`. Log `buffer/labelled_fraction`.

## R2-5. Schedule and checkpointing — APPROVE (within Decision 2)
1. w(t), t = this run's num_timesteps: 1.0 for t < 2.0M; linear 1 → 0 on [2.0M, 4.0M]; 0 after. `report_from = 8.0M` [P: research A.5.4; Decision 2].
2. `BonusScheduleCallback`, first in the callback list: sets `model.replay_buffer.bonus_weight = w(num_timesteps)` in `_on_training_start` and every `_on_step` (SB3 calls on_step before train()). train.py sets w(0) before `_offline_critic`. Mirror to `model.bonus_weight` (audit only).
3. **Resume:** w is a function of num_timesteps + `TrainConfig.bonus_schedule`. Test: resume at 3.0M gives 0.5.
4. Every eval record gets `bonus_w` and `reportable = step ≥ report_from`.

## R2-6. Warm start and critic phases — APPROVE-WITH-NOTES
- As r1_warm Amendment 1 §B: actor, critics, targets, log α and Adam states from best.zip; buffer = source `replay_buffer.pkl`, **`warm.relabel = False`**.
- Offline critic 25k grad steps (labels all 0; re-syncs the 9.2M critic with the 12.35M buffer), online critic 20k (frozen, rows labelled at w = 1), then unfreeze with **actor_lr 1e-4** [C; DECISIONS #20]. Guard threshold 0.65 × the loaded actor's probe; critic_check tol 0.08.
- `TRAINING_AID_KEYS = ('corner_bonus', 'focus')`: `EnvConfig.to_dict()` omits an aid block equal to its default (disabled), so old config dicts, demo keys and golden hashes don't change; `_env_without_reward` drops `reward` and TRAINING_AID_KEYS on both sides; missing keys load as defaults.
- `q_vs_mc` (critic_check included) uses r + w·`corner_bonus` per decision. `det_return` stays bonus-free; add `det_bonus_sum`.

## R2-7. Focused starts — CHANGES-REQUIRED (restoring states not feasible → settle)
1. **Where (`FocusConfig`):** only when `focus.enabled`, not evaluation, and none of `start_pose`/`start_s`/`start_speed` in options (eval, probe/guard, demos, explicit starts never focused).
   - With probability `fraction = 0.25`, draw a bin b ~ P (weights pushed by the callback; uniform start if none yet).
   - Target x = window start b_k of the unit containing the bin's centre (else the bin start).
   - Hand-over target s_h = x − d, d ~ U(50, 300) m; f ~ U(0.5, 0.8).
   - s0 = s_h − f·v_qss_path(s_h)·settle_s, then v0 by the existing rule.
   - Up to 5 attempts with new (d, f), then uniform start (count `focus_fallback`).
   - Reset info `start` += `focused`, `focus_bin`, `focus_target_s`, `handover_to_target_m`.
2. **Fraction** 0.25 [P: TAV18, FLOR17; research B.2]; ≥ 75 % uniform. All 4 envs including the low-noise one.
3. **Weights (`FocusedStartCallback`, main process):** 50 m bins on Track.path (108 bins). Count a pass when the bin of `info['s_track']` changes (first bin from `reset_infos`); count a crash at the termination s for off_track / wrong_way / nonfinite. Every 10k steps decay by 0.5^(10k/1M); r_b = c_b/n_b; P = 0.8·(r_b/Σr) + 0.2·Uniform(H0). Push with `vec.env_method('set_focus_weights', P)`.
4. **Seed [C]:** final s of failed deterministic eval laps in `<seed_run>/laps/step_*_det.npz`, step ≥ 2M (166 crashes). Prior n_b⁰ = 200, c_b⁰ = 200·crashes_b/N_evals. H0 = bins with ≥ 5 seed crashes: 650, 700, 750, 800, 1550, 2350, 2400, 2500, 3100, 4700, 5100.
5. Persist counters to `focus_state.json` (atomic) at every eval; reload on resume (re-seed with a warning if missing).
6. **Log:** focus/share_resets, share_transitions (≤ 0.30), fallback rate, top-5 bins with P, r_b for H0, actual hand-over distance.

## R2-8. Low-noise rollouts — APPROVE-WITH-NOTES
1. `AgentConfig.low_noise_envs = (3,)`, `low_noise_std_scale = 0.3` [P: research C.3; F]. Env 3 = 25 % of transitions.
2. `StagedSAC._sample_action`: call `super()`, then overwrite the low-noise rows under `no_grad`: `mean, log_std, _ = actor.get_action_dist_params(obs)`, a = tanh(mean + scale·exp(log_std)·ε). Scale 0 = `predict(deterministic=True)` exactly. Stored action = executed action.
3. **SAC effects [J]:** 1-step soft Bellman target is off-policy; α loss uses fresh samples from π (entropy tuning unchanged); only bias: uncorrected 3-step sums on 25 % of samples. Negligible.
4. **Log:** TrainStats by env group (normal / low_noise) and focused flag: episodes, failure rate, crash bins, bonus per episode.

## R2-9. Preset, keeper, evaluation, TensorBoard, provenance — APPROVE-WITH-NOTES
1. **Config:** `config.py`: `CornerBonusConfig(enabled=False, bonus=3.0, floor_ratio=.75, prominence_mps=5., merge_m=150., exit_after_apex_m=150., exit_before_brake_m=30., grid_m=1.)`; `FocusConfig(enabled=False, fraction=.25, bin_m=50., lead_m=(50., 300.), speed_fraction=(.5, .8), max_attempts=5)`; `EnvConfig += corner_bonus, focus`. `train_config.py`: `BonusScheduleConfig(hold_until=2_000_000, fade_until=4_000_000, bonus_free_before_report=4_000_000)`; `FocusUpdateConfig(update_every=10_000, half_life=1_000_000, prior_passes=200., seed_run=None, seed_min_step=2_000_000, hot_min_crashes=5, hot_floor_share=.2)`; `AgentConfig += low_noise_envs=(), low_noise_std_scale=.3`; `TrainConfig += bonus_schedule, focus_update`.
2. **Preset `f1_r2`** = `_f1_best_w35()` + corner_bonus/focus enabled, init_mode 'warm', low_noise_envs (3,), warm.relabel False, warm.actor_lr 1e-4, guard.threshold .65, focus_update.seed_run 'runs/mac/D2/f1_best_w35/seed0', total_timesteps 1e9, max_wall_h None, stop.plateau False. **`f1_r2_smoke`**: 150k steps; hold 60k / fade 120k / report gap 30k; eval every 25k with 3 stochastic laps; guard every 5k; strict_provenance False.
3. **Best model:** `best.zip` only from reportable evals; before that `best_aided.zip` (never reported). Resume rebuilds `best_lap2` from reportable records only.
4. **Keeper** (`runs/mac/keeper_f1.sh`): `e['lap2_s'] <= 95 and e.get('reportable', True)` (F1).
5. **Evaluation additions:** `bonus_w`, `reportable`; `lap2_corners` per unit {v_exit, v_qss_exit, rho, bonus} + `lap2_bonus_sum`; stochastic laps add `fail_s` and per-unit pass/crash counts; `corner_bonus` in TRACE_KEYS.
6. **TensorBoard:** `scripts/evals_to_tb.py` adds bonus/w, bonus/lap2_sum, corner/rho_U1..U7, corner/sto_crash_U*, focus/*, lownoise/*, buffer/labelled_fraction; also writes `tb_evals_lifetime/` at step + `lifetime_steps_offset` (9.2M) so f1_r2 overlays f1_best_w35's continuation (F3). `tensorboard --logdir_spec f1_r2:runs/mac/D2/f1_r2/seed0/tb_evals_lifetime,f1_best_w35:runs/mac/D2/f1_best_w35/seed0/tb_evals`.
7. **Provenance/status:** corner table + key, CornerBonusConfig, schedule (+ report_from), FocusConfig + FocusUpdateConfig + seed histogram, low-noise config, init_from/init_buffer sha, labelled fraction at start.
8. **Launch:** `PYTHONPATH=<ocd_first_lap-f1> … train --preset f1_r2 --seed 0 --run-name f1_r2 --init-from runs/mac/D2/f1_best_w35/seed0/checkpoints/best.zip`, then `runs/mac/keeper_f1.sh f1_r2`.

## R2-10. Tests (fast unless marked)
1. Corner table matches R2-2.7 ±2 m; windows disjoint; t_line lap equals `line_lap_time` (1e-6).
2. Env bit-identity: bonus on vs off, same seed/actions → obs, rewards, dones, `laps` bitwise equal; r1 golden hash and full1 / f1_best_w35 demo keys unchanged.
3. `CornerTracker` pure class on synthetic (d0, d1, t, v): one payment per exit per lap; back-and-forth paid once; start inside a window before s_out → paid; start past s_out → next lap; failure in the paying decision → forfeit; projection jump → ≤ B.
4. f(0.75) = 0, f(1) = f(1.3) = 1, linear between; bonus = B·f.
5. Buffer: w = 0 bitwise equal to `NStepReplayBuffer`; w ≠ 0: returns = Σγ^j (r_j + w b_j) masked at done/timeout; `adopt()` bit-identical with bonuses 0 / labelled False; pickle round trip.
6. Schedule: w(0)=1, w(2M)=1, w(3M)=.5, w(4M)=0; reportable iff step ≥ 8M; weight set before train(); resume value.
7. `_sample_action`: scale 0 → row 3 == deterministic predict; other rows from super(); scale 0.3 → empirical pre-tanh std ≈ 0.3·σ.
8. Focus sampler (N = 10,000): share 0.25 ± 0.015; hand-over targets in [x − 300, x − 50]. Integration (N = 20): within −330…−20 m. Eval, probe and start_s resets never focused.
9. FocusedStartCallback on synthetic infos: passes/crashes/decay/seed/persistence.
10. `warm_start` accepts f1_best_w35's config with the R2 blocks; refuses when a driver-definition key differs; `preload_buffer` refuses relabel=True on a time_pbrs source.
11. q_vs_mc with w·bonus; best.zip vs best_aided.zip gating; keeper `reached()` on synthetic evals.
12. Slow: mini `f1_r2_smoke` from best.zip; kill and resume mid-fade.

## R2-11. Acceptance (sim-runner smoke ≤ 30 min, then launch)
- S1 step-0 deterministic lap 2 = 101.87194035987982 s ± 1e-6, clean.
- S2 step-0 lap2_corners ρ within ±0.015 of (0.912, 0.910, 0.849, 0.897, 0.894, 0.897, 0.926).
- S3 mean paid bonus per paid pass in [0.3B, 0.9B]; per-episode bonus ≤ 7B per lap; forfeits only on failure decisions; time/progress terms unchanged.
- S4 focused share of resets 0.25 ± 0.05; fallback < 10 %; focused share of transitions ≤ 0.30.
- S5 low-noise env actions consistent with 0.3σ; its episodes logged separately.
- S6 critic_check ≤ 0.08 at unfreeze; 0 collapses.
- S7 w(t) and `reportable` follow the compressed schedule through a resume; best.zip not written before report_from.
- S8 ≥ 225 env steps/s.
- Full suite green; ocd_first_lap untouched.

## R2-12. Risks to watch (metric)
1. Bonus too weak (F4) — per-unit ρ at 2M vs launch medians.
2. Exit push buys crashes — per-unit crash rate per pass, deterministic clean rate (launch 9/40), stochastic clean rate (0.665).
3. Gains given back after the fade [P: TRACEY23; research A.5.3] — median of the last 5 clean deterministic lap 2 and ρ at 6–8M vs 1.5–2M.
4. Critic lag during the fade — Q_min − MC^n (with w·b), critic loss, 2M–4.5M.
5. Focus feedback/forgetting — crash rate outside H0, focused share of transitions.
6. Low-noise env stuck repeating crashes — its failure rate, share of terminal rows.
7. Old-row inconsistency — labelled_fraction vs Q at exit states, first 1M steps.
8. Winner's curse — report the median of the last 5 clean deterministic lap 2 (reportable only), not the minimum.
9. α / entropy drift from −2.9.

## R2-13. Implementation size
~900 lines of source (corners.py 150, env.py 90, buffers.py 90, agent.py 60, callbacks.py 220, train.py 70, configs 90, evaluate.py 50, evals_to_tb.py 50, keeper 5) + ~600 lines of tests; ~2 coder-days + ~0.5 day QA and smoke. First reportable lap after ≥ 8M steps (~9.3 h at 263 steps/s incl. evals).
