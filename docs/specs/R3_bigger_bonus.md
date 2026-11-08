# R3 — bigger corner exit bonus, kept on (run `f1_r3`)

Status: DRAFT by lead, 8 Nov 2026. Needs an rl-architect check, then rl-coder. Decisions: DECISIONS #25.

## 0. Why
f1_r2 (B = 3, floor 0.75, faded by 4M) ran to 7.40M steps (19.75M lifetime) and was stopped by Ibrahim.
- Clean deterministic lap 2 in 41 % of evals (56/136; f1_best_w35 after 8M: 19 %, 17/88); best 101.48 s
  at 6.55M (w = 0); stochastic clean ~0.63, mean ~103.9 s.
- Exit-speed ratio ρ = v_exit / v_qss(exit) did not move: U1 0.91→0.91, U2 0.91→0.90, U3 0.83→0.84,
  U4 0.92→0.92, U5 0.90→0.89, U6 0.90→0.91, U7 0.93→0.93 (det lap 2, 0–2M vs 4M+).
- Scale (lead): progress term a = 1/m; the critic horizon is ~5 s (γ 0.992 @ 25 Hz); +5 % exit speed earns
  ~+14 from progress over that horizon but only +0.6 from the B = 3 bonus (slope B/0.25 per unit ρ).
  A crash costs ~360. The bonus was too small to change the speed/risk trade-off.

## 1. Decisions by Ibrahim (8 Nov 2026)
1. **B = 30 per corner, floor 0.85**: f(ρ) = clip((ρ − 0.85)/0.15, 0, 1), bonus = 30 · f, capped at v_qss.
   Same 7 corner units, same exit points, same clean-corner rule (forfeit on a failure in the window).
   Max 210 per lap. Slope at ρ ≈ 0.91: 2.0 per +0.01 ρ per corner (B = 3: 0.12).
2. **Bonus stays ON (w = 1) for this run** — no fade, no bonus-free stretch: Ibrahim wants to see its
   effect. A later fade is a separate decision. While on, lap times are "aided" (DECISIONS #2 is
   suspended for this run only; the result is not reported as the setup's driver lap time).
3. **f1_r2 stopped now** (7.40M); f1_r3 warm-starts from f1_r2 `checkpoints/last.zip` + its replay buffer.
4. Same car / setup / line / track: sf70h_f1, sha_f1_best, IQP w3.5, Shanghai. Focused starts (25 %) and
   low-noise env 3 (0.3σ) stay on.
5. TensorBoard: f1_r3 next to f1_r2 and f1_best_w35, on the lifetime step axis.

## 2. Implementation requirements (lead draft; [ARCH] = architect to confirm/adjust)
- **Preset `f1_r3`** = f1_r2 with corner_bonus.bonus = 30, floor_ratio = 0.85, init_from = f1_r2 last.zip,
  focus seed from f1_r2 (its crashes / focus_state.json) [ARCH], and a schedule with w = 1 for the whole
  run (e.g. hold_until = None meaning "never fade"). report_from: never; best.zip never written,
  best_aided.zip as in R2.
- **Early trigger off** for this run (it would switch the bonus off, contradicting decision 2).
  **Keeper stop**: first clean deterministic lap 2 ≤ 95 s at any eval (aided) — Ibrahim's standing stop
  rule; keeper_check.py `reached` must count non-reportable evals for runs whose schedule never fades.
- **Replay-buffer relabel** of f1_r2's buffer (BonusNStepReplayBuffer, bonuses stored unweighted at
  B = 3, floor 0.75): exact map per stored bonus b:
  b = 0 → 0 (ρ ≤ 0.75 or forfeit/no pass); 0 < b < 3 → ρ = 0.75 + 0.25·b/3, b' = 30·clip((ρ − 0.85)/0.15);
  b = 3 → 30 (ρ ≥ 1). Unlabelled rows (if any) stay 0. Record counts in provenance (rows rescaled, max |err|
  vs a recompute on a sample if feasible). [ARCH: confirm exactness; float32 rounding near the floor.]
- **Lifetime step offset**: f1_r3's offset = f1_r2's offset (12,350,000) + f1_r2's num_timesteps
  (7,400,000) = 19,750,000, not just src_num_timesteps. Generalise: offset = source run's own
  lifetime_steps_offset (status.json) + src_num_timesteps.
- **Warm procedure**: as f1_r2 (offline critic 25k grad steps on the relabelled buffer with w = 1, online
  critic 20k steps with frozen actor, critic_check, then unfreeze, actor lr 1e-4, guard 0.65). [ARCH: is
  25k offline enough for a reward change of this size (bonus up to 30 per exit, ~10× before)?]
- **Logging** unchanged from R2 (corner ρ, bonus per lap, sto crashes per corner, w).
- **Keeper / TensorBoard**: `runs/mac/keeper_f1.sh f1_r3`; logdir_spec f1_r3 + f1_r2 lifetime +
  f1_best_w35.

## 3. Risks to watch (lead)
- Faster exits paid for with more crashes at U4 / U2 / U7 (already the top crash corners). Watch
  sto_crash_U* and det clean rate vs f1_r2 (41 %). The clean-corner rule makes a crash in a corner cost
  up to 30 more, which pushes the other way.
- The driver may slow the straights or entries to secure the exit bonus (bonus ≤ 210/lap vs progress
  ~5,390/lap); lap time is the check.

## 4. Acceptance before launch
New tests: relabel map (incl. b = 0, b = 3, interior, float32 near the floor), never-fade schedule (w = 1
at all t, no trigger, report_from never, best.zip never), keeper `reached` on aided evals for never-fade
runs, lifetime offset chaining. Full suite green (except the 2 known pre-existing failures). No long
smoke: the run's own step-0 / 20k / unfreeze evals and critic_check are the check (lead watches).
