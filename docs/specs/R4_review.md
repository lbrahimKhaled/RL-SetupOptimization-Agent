# R4 architect review — docs/specs/R4_speed_profile.md (rl-architect, 8 Nov 2026; saved by the lead)

**Verdict: CHANGES-REQUIRED (design sound; σ 0.15 and entry-paid-at-exit must change). §1 untouched.**
Evidence script: session scratchpad `vprof.py` (17 f1_best_w35 ≥ 8M, 62 f1_r2, 57 f1_r3 clean det lap 2).

## Evidence
- ρ = v/v_qss over the lap: p1–p99 0.72–1.23; straights 0.87–0.94; braking zones 0.73–1.39. Entries (QSS braking points): U1 0.90, U3 0.87, U2/U4/U5/U6/U7 0.71–0.79. Exits 0.84–0.93.
- σ = 0.15 → g = 0 on 18 % of the lap and at 72 % of entries (no signal). σ = 0.30 → 0.5 % of lap, 0 % of entries (entry g min 0.04, median 0.40).
- QSS straight speed unreachable: at pedal ≥ 0.95 the car accelerates 3.9 m/s² at 60–70 m/s (QSS 5.8), 0.8 at 70–80 (QSS 2.5); ρ on straights tops near 0.94, so below v* symmetric g acts as "faster is better".
- Φ_e terminal convention (env.py:566-568, 429): Φ(s′) = 0 after off_track/nonfinite/wrong_way; real Φ after time limit, stall, lap_complete; decision boundaries, γ = 0.992.

## [ARCH] values
1. σ = 0.30 (Φ_v, entry, exit).
2. k_v = 20, Φ_v in the env reward: shaping = γ(Φ_e+Φ_v)′ − (Φ_e+Φ_v), same terminal convention. Mean Φ_v ≈ 16.8 → critic shift ~7 % of Q ≈ 235; drain 0.008Φ_v ≈ 0.13/decision (~340/lap, 6 %), optimum unchanged (telescopes to −Φ_0); shaping activity k_v·TV(g) ≈ 470/lap vs Φ_e 160; straights ~0.44 per +1 m/s immediate (true value ~2–5) → guides, doesn't dominate; crash costs an extra −Φ_v(s).
3. Entry points = window starts (corners.py step 5 = QSS braking point): U1 517.0 m / 82.56 m/s, U2 1467.9 / 80.15, U3 2061.9 / 78.73, U4 2449.9 / 61.32, U5 3006.9 / 78.11, U6 4634.8 / 83.60, U7 5041.8 / 72.00. Exits = existing exit_m. Hand-over at/after an entry inside a window → only the exit is paid.
4. B_in = B_out = 10: max 140/lap (2.6 %; cap 539); expected now ≈ 7·(10·0.40 + 10·0.89) ≈ 90/lap. Slope per +0.01 ρ: exit at 0.9 → 0.22 (R3 2.0, R2 0.12); entry at 0.75 → 0.56 (2.5× exit) → braking early to protect the exit is now net negative (f1_r3 failure). At risk per corner on a crash ≈ 13 vs crash ≈ 360.
5. Entry must not be paid at the exit (window 5–13 s → discount γ^k 0.07 (U1)–0.36 (U7)): pay B_in·g_in at the entry crossing and take it back on a terminal failure before that unit's exit. Same undiscounted value; crashing never pays.
6. Exact buffer relabel possible: v = obs[0]·80; s by chaining Δs = r + 0.04 − (γΦ_e(e′) − Φ_e(e)) for non-terminal rows (a = b = k = 1; 2e-14 on lap traces); next_obs == obs[t+1] on 100 % of 999,235 non-done rows (761 episodes, median 1,500 rows); float32 drift ≤ 1e-3 m/episode; one anchor per chain.

## Requirements for rl-coder
1. config.py RewardConfig `speed_k = 0.`, `speed_sigma = .3` (time_pbrs only, validated; to_dict omits defaults); test f1_best_w35/f1_r2/f1_r3 config.json round-trip byte-equal.
2. env.py: Φ_v at env.py:429 and :567-568 (s = self.s, v_qss = CornerTable.speed); build the corner table when speed_k > 0 (env.py:186); info shaping_v, phi_v, g_v, episode_shaping_v; speed_k = 0 → path skipped, r1 golden bit-identical.
3. g + EntryExitTracker in a NEW module (corners.py untouched → cache key 6c61b28a4876): reuse CornerTracker threshold/high-water/tick-interpolation and settle_events forfeits; take-back on terminal failure for units whose exit is not crossed; truncation takes nothing back; signed info['corner_bonus'] → BonusNStepReplayBuffer with w ≡ 1; CornerBonusConfig mode 'entry_exit' (bonus_in, bonus_out, sigma).
4. relabel.py relabel called from agent.py:327-330 after adopt: order each env column by time from rb.pos, split at dones; anchor per chain = 2-D rigid fit of the 60 preview points vs grid.at(s + d_k) over candidate s; verify at ≥ 3 rows (≤ 0.05 m); F_v added to r in float64 → float32; rebuild entry/exit events (per-decision interpolation, terminal = dones & ~timeouts, chain start = hand-over); stats in provenance; > 0.1 % unresolved → abort → fallback fresh buffer (200k from frozen actor, then usual warm-up). Test vs env ground truth: s ≤ 0.01 m, F_v ≤ 1e-4, bonus ≤ 2e-3.
5. Preset f1_r4 (train_config.py after :435) from _f1_r2(): init_from = F1_R2_INIT, init_buffer None, warm.relabel False + new speed-relabel flag; offline 50k with pre-entry and pre-exit probes (rows ≤ 150 m before, by recovered s); hold_until None; focus seed_run = F1_BEST_W35_RUN, init_state None; lifetime offset 0 + 12,350,000; exit bonus off.
6. Logging (lap_corners evaluate.py:369 + eval records): per unit ρ_in, g_in, ρ_out, g_out, paid/forfeit/taken back, brake-onset s vs entry s, apex v_min; per-lap entry/exit sum; mean g per lap and per straight (exit_k → start_{k+1}); full-throttle share per straight (pedal ≥ 0.95), brake share, top speed; Φ_v and Φ_e shaping sums; front-lock fraction; per training episode F_v, corner terms, take-backs.
7. Tests: all of §3 + take-backs, hand-over exactly at/inside an entry, table key unchanged.

## Risks
- Braking zones (ρ 1.0–1.3 mid-braking) → g pushes harder braking; lock-ups may come before later braking: front_lock_fraction, sto_crash_U*, apex v_min, brake-onset s.
- Critic shift ~−17: critic_check, both probes over offline 50k (warn > 0.5), q_vs_mc.
- Renewed caution from take-back/forfeit (~13/corner): full-throttle share per straight, entry ρ, median of last 5 clean det lap 2 vs 101.9 s.
- s unobserved (Φ_v inferred from the preview): minor; may show as Q − MC error on the back straight.

## Questions for Ibrahim
1. Entry paid at the entry with take-back on a crash, instead of at the exit (crashing still never pays) — lead adopts it as technical (the pay point was the lead's draft, not a §1 decision); reported to Ibrahim.
2. Setup agent later: v_qss is per setup, so corner terms must be off for J(p) laps; keep Φ_v there? Not blocking — OPEN for later.
