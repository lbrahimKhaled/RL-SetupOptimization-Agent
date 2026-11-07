# R2 smoke + compliance — sim-runner, 7 Nov 2026 (c2791ea; saved by the lead)

Smoke `f1_r2_smoke` in `runs/mac/R2_smoke/f1_r2_smoke/seed0`: start 21:06, `kill -9` at ~85k (21:16), resume from 75k, done at 150k (21:22); ~16.5 min wall.

**Verdicts: S1 PASS · S2 PASS · S3 PASS · S4 FAIL as measured (sample size) · S5 PASS · S6 FAIL · S7 PASS · S8 FAIL as measured (marginal).**

- **S1** step-0 and 20k (frozen actor) det lap 2 = 101.87194035987982 s, clean.
- **S2** lap-2 ρ U1–U7 = 0.9118, 0.9102, 0.8466, 0.8968, 0.8939, 0.8971, 0.9259 (worst Δ 0.0024); lap2_bonus_sum 12.387; line QSS 93.8252937271318 s.
- **S3** paid bonus per pass: training 352 passes, mean 1.434 (0.009–2.26); det eval 41 passes, mean 1.737. Per det lap 12.39–12.55 (cap 21). 0/44 terminal-failure rows carry a bonus (forfeit case itself not observed). Bonus-on vs bonus-off env, 6,000 steps same seeds/actions: obs, rewards, terminated, truncated bitwise equal.
- **S4** focused resets 21/127 = 0.165 (0.176 excluding 8 pre-weight resets); band 0.20–0.30. Own draw test 0.246 (N = 20,000); P(≤ 21 of 119 | p = 0.25) = 0.037 → most likely sample noise. Fallback 0/21; focused share of transitions ~0.17; hand-over −143 to −226 m.
- **S5** pre-tanh noise in σ units (frozen actor rows): env 3 0.301/0.301/0.304; envs 0–2 0.97–1.02. Logged separately.
- **S6** critic_check 0.0973 at unfreeze (> 0.08; warning `critic_not_converged`, run continued); 0.083 at step 0. Q_min − MC^n: −17.2 (0), −20.1 (20k), +5.5 (25k), +21.9 (50k). Offline critic phase moved mean Q 232.7 → 207.4 (re-fit of the 9.2M critic to the 12.35M buffer; bonus explains ~0.6). 0 rollbacks, 28/28 probes ok, lowest probe 1017 m vs threshold 977 m (4 % margin).
- **S7** w at evals 1, 1, 1, 1, 0.75, 0.333, 0, 0; per-step bonus/w matches schedule to 3e-8 across the resume; bonus_state.json updated every eval; only best_aided.zip written; reportable only at 150k. best.zip-at-reportable not exercised (150k det lap crashed).
- **S8** env steps/s: 247.6 (0–20k frozen), 182–183 (20–50k, QA pytest running), 220.1 (50–75k), 221.9–223.7 (after resume, no pytest; machine idle load 1.7–2.3). Smoke probes 2× and saves the 1.3 GB buffer 4× as often as f1_r2; estimate for f1_r2 245–260 (not measured). Source run 250–276.

## Not in S1–S8
- **Deterministic laps crashed at every eval after the unfreeze** (25k s = 353, 50k 2399, 75k 2569, 100k 1561, 125k 5250, 150k 5247). At 125k/150k: off the right edge ~0.6 s after hand-over, steering chatter −0.6…+0.8. **Policy entropy −3.1 → −4.29.** Stochastic clean rate per eval (3 laps): 0, 0, 0, 0, 1, 1, 0.33, 0. Training failure rate per episode 0.22 (20k), 0.67 (50k), 0–0.08 (100–125k), 0.44 (150k). Source run: 9/40 det clean in its last 40 evals.
- One transition lost per resume (buffer saved before the step is added; existing SB3 pattern).

## Compliance
ocd_lap used only via `qss_lap.lap_time(path=...)` (corners.py:108-113), `Track.match/length/path`, the existing `OCDRacingEnv` subclass; no monkeypatching. Units: v_qss m/s on the w3.5 line mapped to Track.path arc length (5388.77 m), reversed line raises; exit speed hypot(vx, vy). ocd_first_lap (3945634) and ocd_first_lap-f1 (ee86683) git status clean before/after. Provenance complete (ee86683, code_version 0efddcc91634, rlsetup c2791ea, config/init sha, r2 block). TensorBoard: tb_evals (0–150k) and tb_evals_lifetime (9.20M–9.35M), 87 tags incl. all R2 tags.

## Diagnostic smokes (7 Nov, 21:34–21:53, d0d9268, run in parallel, 150k each, `--set max_wall_h=1.5`)
- **A `f1_r2_smoke_noaids`** (best.zip, all R2 aids off, plain NStepReplayBuffer) and **B `f1_r2_smoke_last`** (R2 on, warm start from last.zip 12.35M). Both completed; 0 rollbacks; 26/26 probes ok. Folders `runs/mac/R2_smoke/f1_r2_smoke_{noaids,last}/seed0`.
- After unfreeze (25k–150k): det lap 2 clean previous 0/6, **A 2/6**, **B 3/6**; sto clean 7/18, 5/18, **9/18**; training failure/episode 0.37, 0.45, **0.24**; lap-0 hand-over crashes (s ≈ 5242–5250, 10–17 decisions) 2, **2**, **0** (source: 0 of 166 det crashes since 2M).
- Critic: offline phase best.zip loss 56.0 → 43.5, mean Q 232.7 → 207.4, critic_check 0.105 (A); last.zip loss 13.3 → 11.4, mean Q 234.7 → 232.4, critic_check **0.036** (B, passes).
- B step 0: last.zip det lap crashes at s = 838.494 m (bit-identical to the source's 12.35M eval). B clean laps: 102.44 (25k), 102.79 (50k), 101.92 (125k, w = 0).
- Entropy −4.3/−4.5 and Q_min − MC^n ≈ +300 at some evals are artefacts of 3 stochastic laps all crashing ~0.5 s after start (n = 3), not policy collapse.
- Base rate: source det clean 0.19 since 8M → P(0/6) = 0.26; det lap 2 alone cannot separate hypotheses at this length.
- **Verdict:** R2 features not the cause (A without aids reproduces the worst signature). Warm-start critic mismatch with best.zip is clear in the critic and plausibly causes the new lap-0 hand-over crash mode (4/12 best.zip evals vs 0/6 B; p ≈ 0.16). Not conclusive on laps.
- Next checks proposed: ~20 stochastic laps per checkpoint re-eval; replay the hand-over crash; best.zip actor + last.zip critic (coder change) or longer offline phase; seeds 1–2.
- Parallel throughput 161–176 env steps/s each (~330–345 combined).
