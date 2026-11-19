# R5: apex split (strict line before the apex, exit speed after it) and crash emphasis (run `bcn_f1_r5`)

Status: reviewed (docs/specs/R5_review.md: CHANGES-REQUIRED C1-C4, all applied below), implemented, 19 Nov 2026.
Decision: DECISIONS #32.

## 0. Why
- `bcn_f1_r4_lrd` (DECISIONS #30): the deterministic lap fails repeatably about 1 m inside the line just before
  the T2, T5 and T7 apexes. The current corner line penalty treats the whole window the same: 0.5 m free band,
  k = 0.0705.
- The R4 entry/exit closeness terms (10 + 10 per unit) and the R2/R3 exit bonuses never moved exit speeds
  (median laps of f1_r2 and f1_r3 equal: 102.08 vs 102.11 s).
- Ibrahim's idea (19 Nov): the line matters up to the apex and speed matters after it. So hold the line strictly
  before the apex, and reward exit speed after it. Train more on what happened just before crashes.

## 1. Decisions by Ibrahim (19 Nov 2026), not to be changed by the team
Ibrahim agreed to these points one by one, then said "go, you evaluate". Items marked (lead) are the lead's
choices under that delegation.
1. **The split is at the apex**, i.e. the unit's last apex (`CornerTable` `apexes_m[-1]`; for the multi-apex
   complexes U7 and U8 the whole complex counts as "before the apex"). A 20 m linear blend is centred on it.
2. **Before the apex: a strict line.** Free band 0.2 m, k = 2 x 0.0705 = 0.141 per metre driven per metre of
   |e| beyond the band, cap 0.5 per metre.
3. **After the apex: the line is loose.** The current corner penalty applies (k 0.0705, band 0.5 m, cap 0.5),
   so the car may use the track width on exit.
4. **After the apex: an exit-speed term**, one-sided and paid per metre driven (never per second). It
   **replaces** the R4 entry/exit closeness terms (set to 0).
5. **Total corner terms are small: at most about 10 % of a lap's progress return.**
6. **OCD first.** The next AC run continues from this OCD run.
7. (lead) **Apex bonus: none.** #30's failure mode is cutting inside before the apex; a reward for being near
   the apex point pushes the same way. Item 2 already targets that failure.
8. (lead) **Crash emphasis: yes.** Transitions in the last 2 s before a terminal failure are drawn 3x as often in
   training batches. **Exit-zone oversampling: no**, so that only one emphasis mechanism changes in this run.
9. (lead) **From scratch with the `bcn_f1_r4_retry` recipe.** Everything else is unchanged: retries from 500k,
   `learn_before_m` 150, lr decay 3M -> 6.2M to 1/100, IQP w3.0 line, `bcn_f1_best`, demos + BC, speed
   potential k_v 20 / sigma 0.3, guard. Control: `bcn_f1_r4_retry`. Note: the comparison measures the reward
   change and the crash emphasis together.

## 2. Formulas (time_pbrs; all lengths are Track.path metres; per 50 Hz base tick whose end s lies in a window)
For unit k: window [b_k, x_k] (braking point to exit), last apex A_k. Blend weight after the apex (C1):
    d = ((s - b_k) mod L) - ((A_k - b_k) mod L),   w(s) = clip((d + 10) / 20, 0, 1)
(offsets from the window start, so windows wrapping through s = 0 work; w = 0 at A_k - 10 m, 1 at A_k + 10 m).
Implemented as env.post_apex_weight, not in corners.py: corners.py's sha256 is part of the corner-table cache
key, which must stay 6313f2151e24 (C2).
Line penalty per tick (replaces the uniform one when `apex_split` is on):
    p = [(1 - w) min(k_pre max(0, |e| - band_pre), cap) + w min(k_post max(0, |e| - band_post), cap)] max(0, ds)
    k_pre 0.141, band_pre 0.2 m, k_post 0.0705, band_post 0.5 m, cap 0.5
Exit-speed term per tick (added):
    r_exit = k_exit w h(v / v_qss(s)) max(0, ds),   h(rho) = max(0, 1 - ((1 - min(rho, 1)) / sigma_exit)^2)
    k_exit = 20 / 150 = 0.1333 per metre (20 per corner at h = 1 over the 150 m apex-to-exit stretch, which is
    exactly 150 m for all 8 Barcelona units: max 160 per lap, 3.5 % of the 4602 m progress return)
    sigma_exit = 0.3 (as R4). One-sided: rho >= 1 gives h = 1 (no penalty for exceeding v_qss).
    v = |(vx, vy)| at the tick end, v_qss = the corner table's speed (the same profile Phi_v uses).
Not potential-based (it changes the optimum; laps are aided, as with R4). On a terminal failure the tick's own
terms are computed as usual; nothing is taken back. The 100 crash penalty dominates the 20 per corner.

## 3. Crash emphasis (replay)
- `BonusNStepReplayBuffer` gets a `crash_near` bool [buffer_size, n_envs]. When a terminal failure is added
  (done and not TimeLimit.truncated) in column c, that row and up to H - 1 earlier rows of column c are marked.
  H = round(2 s / decision_dt) = 50. Marking stops at an earlier episode end (dones) or at the oldest stored row.
  An overwritten row is cleared first.
- Sampling (C3): uniform draw, then the warm-up redraw (`_drop_warmup`), then the crash draw. Among the cells
  that are not retry run-up, with marked fraction f > 0 and factor m = 3, each batch cell is replaced with
  probability q = (m - 1) f / ((m - 1) f + 1) by a uniform draw from the marked, non-run-up cells. This gives
  exactly m x the weight per marked cell, and a batch never holds a run-up row. No importance weights:
  Q-learning's per-(s, a) fixed point is unchanged; only the state weighting of the regression and of the
  actor loss shifts (the same caveat as the retries).
- What is marked (C4): only terminal failures, i.e. done and not TimeLimit.truncated (off track, non-finite,
  wrong way). Stalls and time limits are truncations and stay unmarked (SB3 2.9 sets TimeLimit.truncated =
  truncated and not terminated). Demo rows are never marked: the demo prefill runs before learn(), while
  crash_horizon is still 0; CrashEmphasisCallback sets it at the training start. Tested.
- `crash_factor` = 1 until the run reaches `start_step` = 500k (with the retries; retries from step 0 collapsed
  the smoke run). Factor 1, or no marked cell: no RNG draw, so sampling is bit-identical.
- The warm-up mask (retry run-up) is applied after it (`_drop_warmup`), unchanged.

## 3b. The #30 failure sites under R5 (C4; from the review)
| #30 failure | Last apex | w there | R5 effect |
|---|---|---|---|
| T5, 2084-2099 m | U4, 2110.8 m | 0 | fully strict, as intended |
| T7, 2528 m | U5, 2533.8 m | 0.21 | 79 % strict; the blend is centred on the apex |
| T2, 875 m | U1, 839.9 m | 1 | loose line plus exit-speed reward: NOT addressed |

The corner table has no T2 apex: v_qss rises from 34.4 to 45.4 m/s between 840 and 880 m, and the dip near
950 m is below the 5 m/s detection prominence. Open for Ibrahim (decision 1 is his): accept this, or give T2 its
own split point.

## 4. Config and preset
- `RewardConfig.apex_split` (bool, default False) plus `line_k_pre` / `line_free_pre_m`, `exit_k` /
  `exit_sigma`. The current `line_k` / `line_free_m` / `line_cap` are the post-apex values. Defaults keep
  configs, hashes and rewards bit-identical; `to_dict` omits the new fields while off.
- `TrainConfig.crash_emphasis`: enabled, factor 3, horizon_s 2, start_step 500k. Omitted while off.
- Preset `bcn_f1_r5` = `bcn_f1_r4_retry` with the corner entry/exit terms set to 0 (bonus_in = bonus_out = 0; the
  corner-bonus machinery and its never-fade schedule stay, so laps are filed as aided -> best_aided.zip, as for
  the control), apex_split on (numbers above) and
  crash_emphasis on. Demos and BC carry the same reward terms (they are env rewards).

## 5. Acceptance
Tests:
- the blend weight at A - 10, A and A + 10, and for the multi-apex units;
- the pre/post penalty values;
- h is one-sided;
- the exit term is paid per metre (the same total for a given path at two speeds when h = 1);
- off: rewards bit-identical, configs unchanged;
- crash_near marking (horizon, episode boundary, overwrite, truncation not marked);
- the sampling ratio is about 3 for a synthetic buffer;
- factor 1 is bit-identical;
- the preset builds.

Full suite green, except for the known failures. A short smoke run on the Windows PC before launch.
