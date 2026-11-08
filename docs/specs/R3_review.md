# R3 architect review — docs/specs/R3_bigger_bonus.md (rl-architect, 8 Nov 2026; saved by the lead)

**Verdict: CHANGES-REQUIRED (design sound; 5 fade assumptions in code + exact relabel/offset definitions).** §1 decisions untouched.

## Checked
- Stored bonus = float32(info['corner_bonus']) (buffers.py:46) = 3·clip((ρ−0.75)/0.25,0,1) summed over a decision's crossings (env.py:599); forfeit / exit_unpaid → 0; ≤ 1 crossing per decision (exits ≥ 150 m apart, ≤ ~4 m per decision).
- f1_r2 buffer labelled fraction 1.0 at 7.4M; last.zip num_timesteps 7,400,000 = status replay_buffer_step.
- Offset chain: f1_best_w35 none, f1_r2 12,350,000. f1_r2 config.json pins warm.init_buffer to f1_best_w35's buffer → f1_r3 preset must leave it None (train.py:380 resolves to f1_r2's buffer).

## Requirements for rl-coder
1. Never-fade schedule: `BonusScheduleConfig.hold_until: int | None`, None = w ≡ 1. bonus_weight → 1 (callbacks.py:606); report_from → None (train_config.py:164); BonusScheduleCallback.__init__ no int(None) (callbacks.py:642-644); reportable() False when report_from None (callbacks.py:671); trigger skipped when report_from None (callbacks.py:721) = early trigger off; resume with bonus_state.json report_from null (load_bonus_state, train.py:681). Old configs (f1_r2 config.json) load unchanged (test).
2. best.zip / best_aided.zip: no change (best_eligible False when not reportable → only best_aided.zip, incl. resume rebuild train.py:684-686). Test.
3. Keeper: keeper_check.reached counts every eval when config.json bonus_schedule.hold_until is null; limit stays trigger_lap2_s = 95.
4. Lifetime offset (train.py:527): source status.json lifetime_steps_offset (or 0) + src_num_timesteps → 19,750,000; record both parts in provenance.
5. Relabel in preload_buffer after the type check (agent.py:320), only when source config.json env.corner_bonus (B, floor) differs: float64 ρ = f_s + (1−f_s)·b/B_s, b' = B·clip((ρ−f)/(1−f),0,1) → float32. Refuse if f < f_s (ρ below old floor lost) or any b > B_s(1+1e-6) (summed events). Exact up to rounding (≤ ~16.7·ulp(3) ≈ 4e-6). Test on CornerTracker outputs for the same ρ grid at (3, 0.75) and (30, 0.85) incl. 0.85 ± 1e-7, 1.0, 1.3. Provenance: rows b > 0, rows mapped to 0, mean before/after, relabelled buffer sha.
6. Offline critic 50k grad steps (was 25k) [judgement]: expected ~67 bonus/lap at current ρ (12.4 before) → mean Q +3 but +12–16 just before exits; global critic_check (tol 0.08) can't see that; propagation over 125 decisions with n = 3, τ = 0.005 needs ~8–20k steps. Keep tol 0.08. Add a pre-exit probe in _offline_critic (train.py:332): fixed batch of rows with nonzero b' within the next 30 decisions (same column, no done between); log mean Q_min per block; warn if it moves > 0.5 over the last 5 blocks.
7. Focus: continue from f1_r2 focus_state.json (7.4M, 108 bins, decayed counters, H0) as state=, not seed= — new FocusUpdateConfig.init_state (path), in provenance. Re-seeding from f1_r2 laps after 4M: 69 evals, too few crashes for H0.
8. evals_to_tb (line 61), result block (None without trigger), status report_from null: safe. Test a never-fade resume.
9. B = 30: no other config change (α/entropy target, guard (progress only), q_vs_mc (r + w·b) unaffected).

## Risks to watch
- Warm-start actor = 7.4M snapshot whose det lap 2 crashed (sto clean 0.40); guard baseline is its probe → watch step-0 and 20k evals.
- +0.05 ρ earns +10 → can pay ~2.6 % extra crash risk per corner: watch sto_crash_U*, det clean (41 %; 33/69 after 4M), sto clean (0.64).
- Critic lag near exits: pre-exit probe, Q_min − MC^n, critic loss over the first 100k.
- Lap time can worsen while ρ rises: median of the last 5 clean det lap 2.

## Questions for Ibrahim (non-blocking)
Keeper stop on an aided lap (best_aided.zip the only artefact) and whether ≤ 95 s still applies with the bonus permanently on — lead: yes, Ibrahim's standing rule (DECISIONS #22 decision 5).
