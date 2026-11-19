# R5 review (rl-architect, 19 Nov 2026): apex split + crash emphasis

Spec: `docs/specs/R5_apex_split.md` (DECISIONS #32). §1 (Ibrahim's decisions) is not reopened. I read the spec and the
code it changes; the rl-coder's working-tree diff was already in progress, so a few findings point at it.

**Verdict: CHANGES-REQUIRED.** Four small changes (C1-C4). The formulas and the q mixture are correct. No
OPEN/§12 item is touched.

Labels: [F] = from the formulation / §1, [C] = from the code or data, [P] = from a paper, [J] = my judgement.

## (a) Blend weight
- [C] The spec's formula `((s - A) mod L + 10) / 20`, "taken signed before A - 10", is ambiguous. A plain mod puts
  every s before A near L, which clips to w = 1 (loose before the apex). The working-tree version is correct and
  handles wrapping windows: d = ((s - b_k) mod L) - ((A_k - b_k) mod L), the offset to the last apex measured inside
  the window, and w = clip((d + 10) / 20, 0, 1). It is unambiguous because every window (at most 504 m) is far
  shorter than L/2 = 2301 m. **C1.**
- [C] Barcelona table 6313f2151e24: the distance from window start to last apex is 35-354 m, so it is at least 10 m
  everywhere and the blend never starts before b_k. Every apex is followed by 150 m to the exit, so w = 1 holds for
  the last 140 m. Neither U7 (3352.7-3856.7) nor U8 wraps through 0, but the code must still handle a wrapping
  window (test it).
- [C] Multi-apex units: the strict stretch is 344 m in U7 and 223 m in U8 (decided, item 1). In total 1070 of the
  2270 m of windows are pre-apex.
- [C] **Fact for Ibrahim (not a change to item 1).** The three #30 failure sites against this table:
  - T5 at 2084-2099 m: U4's apex is 2110.8 m, so w = 0 there (fully strict). This is the intended effect.
  - T7 at 2528 m: U5's apex is 2533.8 m, so w = 0.21. The centred blend relaxes the strict penalty by 21 % at this
    failure point.
  - **T2 at 875 m: this lies 35 m after U1's only apex (839.9 m), so w = 1.** The loose line is unchanged there, and
    the exit-speed term pays for speed through it. The QSS profile shows no T2 minimum at 875 m: v rises from
    34.4 m/s at 840 m to 45.4 m/s at 880 m. The small dip near 950 m (48.5 m/s) is below the 5 m/s prominence, so
    the table has no T2 apex. R5 does not target the T2 failure and may push it. §0 should say so (**C4**), and see Q1.

## (b) Exit-speed term
- **Gaming.** [C] The ds in the term is the Track.path projection progress (`self._dist - d0`, wrapped difference),
  not the distance driven. Weaving adds no ds, and the extra |e| costs line penalty past 0.5 m. With max(0, ds),
  re-driving the same metres would pay twice. That needs negative-ds ticks, which OCD cannot produce in a sustained
  way: there is no reverse gear, wrong way (more than 90°) terminates, and projection jitter at 50 Hz is
  sub-centimetre. The term is therefore not gameable in practice. Optional hardening: pay on the high-water mark of
  `_dist`, like `CornerTracker`.
- **Size.** [C/J]
  - Per corner: 0.1333/m × ∫w ds = 0.1333 × 150 = 20, which is exact, because the ramp contributes 10 m and the
    w = 1 stretch 140 m.
  - Per lap: at most 160 (3.5 % of 4602).
  - At 1 m past each band: the pre-apex penalty is 0.141 × 1070 = 151 and the post-apex penalty is
    0.0705 × 1200 = 85. Together with the exit term that is 396, or 8.6 %, which is within item 5's ~10 %.
  - Cap 0.5/m < a keeps net progress ≥ 0.5/m.
  - Crash 100 vs at most 20 per corner. Since nothing is taken back, a crash at the end of an exit keeps up to ~20 of
    exit reward. The crash penalty still dominates.
- **Marginal speed incentive.** [J] At rho 0.9, v_qss 60 m/s:
  - The exit term gives 0.133 × dh/drho (2.22) = 0.30 per unit rho per metre.
  - The time term b/v gives ~0.02 per unit rho per metre, so the exit term is ~15x the true lap-time incentive over
    those 150 m.
  - Per corner that is ~44 per unit rho. For comparison: R4's B_out = 10 gave ~22 (no effect on exit speeds) and R3's
    B = 30 at floor 0.85 gave 200 (made the driver cautious).
  - R5 sits between them, so it is plausible but watch for "slow in" (risk R1). It saturates at v_qss, so it never
    pushes past the QSS profile.
- **Phi_v (k_v 20).** [P: Ng, Harada & Russell 1999] Phi_v is potential-based, so it does not move the optimum. Below
  v_qss both terms push the same way. Above v_qss, Phi_v dips transiently and h stays at 1, with no conflict at the
  optimum. They are computed at different points (r_exit per tick, Phi_v per decision) and do not double count.
- [C] Lap costs are untouched: the term goes into `r` only, not into `costs`, so `laps` / `cost_r1` stay pure time.
  This matches the line penalty and is correct for the setup-agent signal.

## (c) Crash emphasis
- **q is correct.** [C] Take N stored cells and C marked cells. Each marked cell is drawn with probability
  (1-q)/N + q/C = m / (N + (m-1)C), and each other cell with 1 / (N + (m-1)C). The ratio is exactly m.
- **n-step.** [C] AgentConfig.n_steps = 3, and no Barcelona preset overrides it. The mixture only changes the start
  row. The n-step chain, the done/timeout masks, the `pos - 1` timeout trick and the wrap index `% buffer_size` are
  unchanged, so marked rows near `pos - 1` are handled as before. The crash penalty enters directly for the last
  3 marked rows and through bootstrap for the rest, which suits a 2 s horizon at γ 0.992.
- **Marking.** [C]
  - The criterion "done and not TimeLimit.truncated" is right for SB3 2.9.0: `DummyVecEnv` (l.66) and
    `SubprocVecEnv` (l.38) set `TimeLimit.truncated = truncated and not terminated`.
  - A time_pbrs stall is `truncated=True, terminated=False` (env.py l.649-650), so it is not marked. The time limit is
    not marked either. Terminal failures (off track, nonfinite, wrong way) are marked.
  - Equivalent and more explicit: `info['termination'] in TERMINAL_FAILURES`.
  - The backward walk stops at an earlier `dones` row. That also covers the resume cut (`cut_nstep_chains` sets
    done = 1 at `pos - 1`). Rows are cleared on overwrite.
- **Demos.** [C] Demo transitions are not marked: `prefill_replay_buffer` runs before `learn()` (train.py ~624 vs
  333), while `crash_horizon` is still 0. That is the right outcome, because demo crashes are a scripted driver's
  crashes. It should be stated and tested, because it holds only through call order (**C4**).
- **Warm-up interaction.** [C] The spec runs the mixture before `_drop_warmup`. A crash draw that lands on a warm-up
  row is then replaced by a *uniform* draw, and f counts warm-up cells. Marked warm-up rows exist whenever a crash
  comes within 2 s of the end of a retry run-up, and with `learn_before_m` 150 m that is rare but possible. So
  "exactly m×" does not hold. **C3.**
- **Bias.** [P: Schaul et al. 2016, PER] Non-uniform replay changes the expected update. PER corrects this with
  importance weights. Without them, the per-(s, a) Bellman target is unchanged, but under function approximation
  the regression weighting and the actor/alpha state distribution shift toward pre-crash states. The spec says this,
  and it is acceptable as a stated design choice.
- **Cost.** [C, measured] `np.flatnonzero` over 250k×4 cells takes ~0.5 ms per batch. That is about 165 batches/s,
  ~8 % CPU on the sampler. Optional: cache the index list and refresh it only when `add` changed a mark.

## (d) Off = bit-identical
[C] Checked on the working tree:
- `_line_penalty` with apex_split off returns the same expression.
- The exit term is not computed and adds no info keys when off.
- `to_dict` drops the R5 keys at defaults, and `crash_emphasis` likewise.
- The buffer makes no RNG draw at factor ≤ 1 or with no marks, and marking writes only `crash_near`.
- The `agent.py` change (bonus buffer also for retry or crash emphasis) leaves every existing preset unchanged:
  `bcn_f1_r4_retry` already had corner_bonus on.

**Exception: `corners.py` was edited** (`CornerTable.post_apex_weight`). Its sha256 is part of the corner-table
cache key (`cache_key`: `corners_py_sha256`). Every run with a corner table, including the paused control
`bcn_f1_r4_retry`, would rebuild the table in memory under a new key instead of loading 6313f2151e24. That is
probably value-identical, but it is not guaranteed across machines, and the control must not change mid-run.
**C2.**

## (e) Demos / BC
[C] Demo rewards are `env.step` rewards (`demos.run_episode`), so they carry the line split and exit term
automatically. BC uses actions only. The demo key includes `env.to_dict()`, which holds the R5 keys, so the dataset
is recollected. With corner_bonus off, demos have no `corner_bonus` array. The working tree handles that through
`rb.stores_bonus = False` (demos.py l.278); keep a test. The demo driver runs at 0.35-0.65 v_qss, where h = 0, so
demo rows carry no exit reward. That is expected and harmless.

## Required changes
- **C1.** Spec §2: replace the mod-L wording with the window-relative definition d = ((s - b_k) mod L) -
  ((A_k - b_k) mod L), w = clip((d + 10) / 20, 0, 1). Add a test with a synthetic window that wraps through s = 0,
  with the apex on each side of 0.
- **C2.** Do not edit `corners.py`. Move the blend weight to `speed_profile.py` (already in REWARD_SOURCES) or to
  `env.py`. Add a test that `corners.cache_key` for the Barcelona inputs still gives 6313f2151e24..., or at least
  that `corners.py` is byte-identical to HEAD.
- **C3.** Crash sampling order:
  1. Draw uniformly as before.
  2. `_drop_warmup`.
  3. Run the mixture over C = crash_near ∧ ¬warmup (stored rows), with f = |C| / |stored ∧ ¬warmup|.
  
  Add a test with a synthetic buffer where warm-up and marks overlap: the ratio is ≈ 3 and no warm-up cell is
  returned.
- **C4.** Spec text:
  - (i) §0/§1: under item 1 the T2 failure point (875 m) is post-apex (U1 apex 839.9 m; w = 1), and T7 (2528 m)
    has w = 0.21. R5 targets T5 fully, T7 mostly, and T2 not at all.
  - (ii) §3: demo rows are not marked (prefill before learn); test that the demo rows are not marked.
  - (iii) §3: the marking criterion is SB3's `TimeLimit.truncated = truncated and not terminated`; stall and time
    limit are unmarked; test both.

Recommended (not blocking):
- Log the pre and post line penalty separately in info and in the per-unit eval stats: post-apex mean rho per unit,
  and pre-apex p95 |e| at U1/U4/U5.
- Cache the crash index list.
- Optionally pay the exit term on the `_dist` high-water mark.

## Risks to watch (metric)
- **R1 Slow-in for the exit term** (the R3 failure mode): apex and entry speed ratios v/v_qss per unit, and
  full-throttle share on straights, vs `bcn_f1_r4_retry` at equal steps. Lap 2 within ±0.3 s counts as
  inconclusive (#27 rule).
- **R2 Strict line costs pace in U7/U8** (344 m / 223 m strict): U7/U8 sector times vs the control;
  `ep_line_penalty_mean`.
- **R3 T2 still fails:** det-lap termination s histogram in 850-900 m; the stochastic crash share of U1.
- **R4 Crash emphasis destabilises the critic or alpha after 500k:** critic loss, Q vs MC (q_mc) at evals,
  ent_coef, `crash_emphasis.marked_share`. Above ~0.2, q exceeds 0.29 and the batch skews strongly.
- **R5 Exit term saturates and gives no gradient:** `ep_exit_speed_reward_mean` per lap near 160 while lap time is
  flat.
- **R6 Downstream:** v_qss is the setup's own QSS profile, so the non-potential exit term makes the driver chase each
  setup's QSS exits. Lap time stays the setup signal, but this is an aided driver (as R4), which matters for
  fine-tunes later.

## Questions for Ibrahim
- **Q1.** Under item 1 R5 does not touch the T2 failure point (875 m, 35 m after U1's apex), and T7 (2528 m) keeps
  21 % of the loose penalty because the blend is centred. Do you accept this as decided, or do you want a manual T2
  split point (or a blend that starts at the apex)? Either option changes item 1, so it is your call.
