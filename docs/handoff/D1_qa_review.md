# D1 QA review — qa-reviewer (25 Oct 2026)

**Verdict: no BLOCKERs.** Implementation matches D1_driving_env.md as amended by D1_review.md on reward/penalty math, terminated vs truncated, reset, observation frames and the pedal deadzone — each verified by running throwaway scripts (`/tmp/qa_d1`, temporary), not just reading. Weak point is test strength: mutation testing shows **7 of 20 injected bugs survive the full suite** (MAJOR M-1).

Tests (real output): `.venv/bin/python -m pytest -q` → 50 passed in 16.87 s; `-m slow -s` → 2 passed, "first-attempt settles 200/200, Fz within 5 %: 200/200, worst 2.550%".

## Verified correct (by running code)
1. **Reward (1) per base tick** comes from `OCDRacingEnv.step`; env.py adds no reward terms. R2.5 equivalence is exact. Second (repeated) tick has u̇ = 0 exactly (same float32 last_action). Decision reward = sum of tick rewards (`r == r1 + r2` exact); stops early on first-tick end via `break`.
2. **Penalty** `P = P0 + κ(c/0.02)(T_rec/3)e²` (= 50 + 30·c·e² default), uses dt_tick 0.02 s not 0.04. e_exit = `_last_e` (offset at the failing physics step) for off-track/stall/wrong-way; nonfinite uses last finite e (NaN injected in 1st and 2nd sub-step: penalty 50.002 = P(e_last), terminated=True, obs zeros, no exception).
3. **Terminated vs truncated:** off-track, nonfinite, stall, wrong-way, eval lap → terminated=True, truncated=False; time limit → truncated=True. Time limit lands exactly on tick/decision boundary for every multiple of 0.04 s up to 120 s (60 s = physics step 30000).
4. **Lap split at line crossing:** interpolated crossings; each tick's costs split by crossing fraction → lap time cost = b × lap time. Demo flying laps f = 0.5 complete; f = 0.7 complete 192.005/192.033 s, clip 0.
5. **Training lap disabling (progress = −1e7):** no side effect on reward or info (`info['progress_m']` from `_dist`, reward uses ds directly; −1e7 only read by base lap check).
6. **Reset:** same seed → bit-identical obs + rewards over a whole random episode; a "dirty" episode then reset with that seed is bit-identical to a fresh env (no leakage); after settle `last_action` = settle's last action, s from `Track.match`, (s0, v0) independent of setup; v0 never below 3 m/s (min 3.11 over 20k draws).
7. **Observation:** car 1.5 m left of ξref → e = +1.5 (obs 0.30), station-1 ξref point y = −1.52 in car frame; left-edge preview y > 0, right-edge y < 0; preview ξref points match independent `track.point(s + d_k)` rotation within 1 cm incl. start at s0 = L − 3 m (wrap); history newest-first, each entry pairs ego with the action that led to it, filled from settle at hand-over; body-frame a_y ≈ v·r tested.
8. **Pedal deadzone:** clip((a+0.5)/1.5, 0, 1) and inverse (−0.5 + 1.5p, or −0.8 at p = 0) round-trip; deadzone off → bitwise equal to `OCDRacingEnv.to_inputs`.

## MAJOR
**M-1: critical paths untested** — each mutant survives the full suite (current code correct; regressions would go unnoticed). Add to `tests/test_env.py`:
- **Sign flips survive** — preview y (`env.py:190`), track position (`env.py:152`), obs e (`env.py:192`). Test: reset with start_pose 1.5 m left of ξref at s = 500 (settle_s = 0); assert `obs[9] ≈ +0.3`, preview station-1 ref y < 0, left-edge y > 0, right-edge y < 0; `track_position()` larger at +1.5 m than −1.5 m; with s0 = L − 3, preview ξref points == `rotate(track.point(env.s + d_k) − xy)` within 0.02 m (covers wrap).
- **History oldest-first (`env.py:175`) and history not filled from settle (zeros) survive.** Test: reset (with settle), 3 `DemoPolicy` steps; assert `obs[135:147] == np.r_[prev[0:9], prev[132:135]]` and `obs[147:159] == prev[135:147]`; assert `obs0[135:]` nonzero at hand-over.
- **Nonfinite gets no penalty (drop 'nonfinite' from `FAILURES`, `env.py:64`) survives** — exploitable (ending cheaper than continuing). Test: monkeypatch `env.backend.step` to return vx = nan on physics step 5, and (second case) step 15; assert `term and not trunc`, `termination == 'nonfinite'`, `penalty == failure_penalty(e_before)`, obs all zeros.
- **Stall timer not reset at `reset()` (`env.py:281`) survives** — spurious early terminated, biases critic. Test: `max_episode_s = 1.9`, start_speed 3.5, brake until truncated while stalled; reset the same way; assert no 'stall' before 2 s after hand-over.
- **Time cost using nominal 0.02 s on a shortened tick (`env.py:358`) survives.** Test: in the off-track test assert `r == Σ step_cost components` and `episode_cost['time'] == info['time_s']` on the terminal step.

## MINOR
1. **Smoothness cost blows up on a shortened terminal tick** (inherited: base env.py:146, ours env.py:357): u̇ = Δu/dt_actual → up to 10× (24.0 vs 2.4 for full swings on all channels). Terminal reward varies P … P + 22 depending on which 2 ms step the footprint crosses (first sub-step only). Fix (architect's call): document, or use dt_tick in u̇ and exclude terminal tick from the equivalence test.
2. **`max_episode_s` / `eval_max_s` not checked to be multiples of the decision period** (env.py:90–94); `max_episode_s = 0.05` gives a last decision of 0.01 s. Fix: raise if `round(T/decision_dt)*decision_dt != T`.
3. **Eval lap end uses base `progress` while crossings use `_dist`** (env.py:288 vs 323) — two float sums; if they disagree at the last line, episode ends 'lap_complete' with one lap recorded and `lap1, lap2 = info['laps']` crashes. Suspected only (~1e-11 m). Fix: in evaluation terminate on `len(self._crossings) == eval_laps + 1` yourself.
4. **Initial history zeros when settle < 2 decisions** (env.py:239); action 0 ≈ 1/3 pedal with deadzone. Only settle_s ≈ 0 configs (tests). Fix: init entries with (ego at placement, [0, −1, −1]).
5. **Nonfinite step's info lacks `e_ref`, `heading_error`, `speed`** (env.py:401) → KeyError in loggers. Fix: NaN.
6. Eval run-in after settle is 165.9 m not 200 (start L − 200, settle drives 34 m). Matches D1 §5; R5 "200 m run-in" ambiguous. Lap 2 unaffected. Note only.
7. DemoDriver traction control beyond R5 (config.py:76–81) — architect signed off (D1_signoff A1).
8. `scripts/bench_env.py:121` formats `lap1_minus_lap2_s:.3f` → TypeError if the demo lap fails (verified: demo leaves track at f = 0.85). Fix: guard None.
9. Provenance doesn't identify code yet (rlsetup was untracked when built; QSS profile provenance names the initial commit; ocd_first_lap dirty). Fix: commit before D2; add `env.ref.digest`, QSS cache key, `EnvConfig.to_dict()` to D2 run metadata.
10. `ReferenceLine.__init__` (reference.py:30–37) doesn't check path closed / s strictly increasing; duplicate CSV row → zero tangent → NaN. Matters for Jad's laps.
11. Stall timer not in obs → stall termination slightly non-Markov. Accepted practice (ACGYM24). Note only.
12. Each step's info copies `episode_cost` (and `laps` in eval), pickled per step under SubprocVecEnv. Small perf cost.
