# R4 — QSS speed-profile potential + corner entry/exit closeness (run `f1_r4`)

Status: DRAFT by lead, 8 Nov 2026. Needs rl-architect review ([ARCH] items), then rl-coder. Decision: DECISIONS #26.

## 0. Why (f1_r3 telemetry, lead, 8 Nov; script in the session scratchpad `behaviour.py`)
f1_r3 (exit bonus B = 30, floor 0.85, never fades) got slower: clean det lap 2 101.27 s (0.45M) → ~102.2 s
(2.3–2.6M), while stochastic laps got safer (70–100 % clean). Lap-2 telemetry vs f1_r2 6.55M (101.48 s):
- Lifts on straights: full-throttle share on 0–517 m 37 % → 0 %, on 1735–2062 m 77 % → 19 %; top speed
  start straight 76.8 → 74.0 m/s (QSS 82.6). Straights cost +0.07…+0.17 s each.
- Slow-in to secure the exit bonus: U3 entry top speed 71.2 → 68.8 m/s, v_min 48.8 → 50.7, exit ρ 0.846 → 0.885,
  but the U3 window +0.09 s slower. U2 entry top speed 61.4 → 56.9 m/s.
- Calmer: steering chatter −40 %; mean |e| 0.32 → 0.37 of half-width.
The one-sided exit bonus rewards protecting the exit, not lap time; nothing rewards straight-line speed.

## 1. Decisions by Ibrahim (8 Nov 2026) — not to be changed by the team
1. **Replace** the R2/R3 corner exit bonus with:
   (a) a **speed-profile potential** along the whole lap: Φ_v(s, v) = k_v · g(v, v_qss(s)), added as
       γΦ_v(s′) − Φ_v(s) (with the existing Φ_e = −k|e|; potential-based, Ng 1999, so the optimal policy is
       still the fastest lap);
   (b) **corner entry and exit closeness terms**: B_in · g(v_entry, v_qss_entry) + B_out · g(v_exit, v_qss_exit)
       per corner unit.
   Closeness score, squared, symmetric, non-negative: **g(v, v*) = max(0, 1 − ((v − v*)/(σ v*))²)**
   (1 at v = v*, 0 at ±σ; positive so crashing never escapes a penalty).
2. **Size of (b): small — at most 10 % of the lap's progress return** (≈ 5,390 per lap → ≤ ~540 per lap for
   all 7 corners' entry + exit together). Crash must stay far more costly than any gain.
3. **Kept on (w = 1), never fades** (both (a) and (b)). (a) does not change the optimum; (b) does, so laps are
   "aided" in the DECISIONS #2 sense.
4. **Warm start from f1_best_w35** (Ibrahim: "f1_best_w35 … from this"); lead: its `last.zip` (12.35M) + its
   replay buffer, per DECISIONS #24 (best.zip's critic mismatched the buffer → hand-over crashes in R2 smokes).
5. Same car / setup / line / track (sf70h_f1, sha_f1_best, IQP w3.5, Shanghai). Focused starts (25 %) and
   low-noise env 3 (0.3σ) stay on (lead: carried over from R2/R3; not re-asked).
6. f1_r3 is stopped when f1_r4 launches. TensorBoard: f1_r4 next to f1_r3, f1_r2, f1_best_w35 (lifetime axis).

## 2. Requirements and [ARCH] items
- **v_qss(s)**: QSS speed on the w3.5 line mapped to Track.path arc length (exactly the profile corners.py
  already builds and caches); v = hypot(vx, vy) as for the exit speed.
- **σ** [ARCH]: lead proposal 0.15 for both terms (at today's exits ρ ≈ 0.9 → g ≈ 0.56; entries at
  ~1.2 × v_qss in braking zones → g = 0 there). Check against the f1_r2/f1_r3 telemetry distributions of
  v / v_qss along the lap.
- **k_v** [ARCH]: size relative to the progress term (a = 1/m, ~2.2 per decision at 55 m/s) and to Φ_e. A
  potential's size does not move the optimum but sets how strongly it steers learning. Note the steady-state
  drain −(1 − γ)Φ_v per decision for a positive Φ_v (≈ −0.008 k_v); terminal-state convention for Φ_v must match
  the existing Φ_e implementation (Φ(terminal) = ?) — state it.
- **Entry point** [ARCH]: lead proposal = the QSS braking point before each corner unit (local max of v_qss
  before the window; ≈ window start b in the corner table), so the entry term rewards reaching full speed and
  braking late. **Exit point** = the existing s_out. Both paid **at the exit, only if the car stayed on track
  through the window** (entry speed recorded at the entry point, forfeited with the exit on a failure), as R2.
  Start inside a window after hand-over: pay only the parts crossed after hand-over [ARCH].
- **B_in, B_out** [ARCH], within decision 2's cap; lead suggestion B_in = B_out ≈ 10 (max 140/lap ≈ 2.6 %),
  because R3's 210/lap (≈ 4 %) already changed behaviour. Justify with the f1_r3 lesson (marginal value of
  slowing for a safer exit must not exceed what the entry term loses).
- **Replay buffer**: f1_best_w35's buffer is pre-R2 (plain NStepReplayBuffer, no bonus field). [ARCH] how to
  label it for the new reward: exact relabel from stored obs/next_obs if (s, v) are recoverable (see
  `rlsetup/driving/relabel.py`, R1 relabel was exact to < 1e-4), else adopt with corner terms 0 and Φ_v
  relabelled, else fresh-buffer warm-up. The env reward must stay bit-identical to time_pbrs when the new
  terms are disabled. Decide whether Φ_v goes into the env reward (like Φ_e) or a buffer field (like the bonus).
- **Warm procedure**: as f1_r3 (offline critic 50k grad steps with a pre-exit/pre-entry probe, online critic
  20k frozen actor, critic_check, unfreeze, actor lr 1e-4, guard 0.65, lifetime offset 12,350,000).
- **Schedule**: never-fade (hold_until None), no Amendment-1 trigger; keeper stops at a clean det lap 2 ≤ 95 s
  (aided); only best_aided.zip.
- **Focus**: seeded from f1_best_w35 crashes as in R2 (not f1_r3's state: different driver lineage) [ARCH].
- **Logging**: per corner entry and exit v/v_qss and g; per-lap sum of entry/exit terms; mean g along the lap
  and per straight; full-throttle share per straight (the f1_r3 failure signature); Φ_v shaping sum per lap.
- **Preset `f1_r4`**, run name `f1_r4`. The R2/R3 exit bonus is off in this preset (replaced).

## 3. Acceptance before launch
Tests: g (symmetric, 0 beyond σ, 1 at v*); Φ_v potential telescopes over an episode (sum of shaping =
γ-discounted boundary terms); entry/exit paid once per pass, only if clean, forfeited on failure; disabled →
reward bit-identical to time_pbrs (r1 golden); buffer labelling exactness (whichever path the architect picks);
never-fade schedule; keeper. Full suite green (2 known pre-existing failures). Short smoke (sim-runner, ≤ 20 min):
term magnitudes per lap in the expected range, no NaN, critic_check.
