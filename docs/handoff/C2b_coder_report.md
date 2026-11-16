# C2b coder report (critics, training, evaluation)

Status: **implemented, R0 + R1 run; R2-R9 for the sim-runner** (16 Nov 2026). Nothing committed.
Binding: `docs/specs/C2b_addendum.md` (B1-B20, Runs and budget, Bars) > `C2_review.md` > `C2_style_critic.md` S1-S11; S12 of the spec on top.

## Progress log
- [x] Step 1: data changes (B20/B1/B2/B3/B4/B6): built and run on real data (results below)
- [x] Step 2: models, nuisance, training, guards, twins, groups, Barcelona head, metrics, ablation, baselines, fallback (B7-B16)
- [x] Step 3: `rlsetup/critic/bundle.py` (B17) incl. the dummy-adapter test
- [x] Step 4: `scripts/critic_c2.py` (R0-R9, `--runs`, `--smoke`, bars.json first, resumable per fit)
- [x] Step 5: tests B19 (`tests/test_c2b_units.py`, `tests/test_c2b_pipeline.py`), full suite
- [x] Step 6: R0 smoke + R1 baselines on real data, timing, budget projection (results below)
- [x] B21-B23 (binding addendum items of 16 Nov): implemented and tested (section below)
- [x] Coordinator additions: B21 extra-loss hook (not the term itself); spec 12.21 (TD3's Q always critic 1, e-gate hook)

## Step 1: data (B1, B2, B3, B6; B4 transform in `rlsetup/critic/style_transform.py`)
Build: `.venv/bin/python scripts/critic_c2_data.py` -> `runs/critic_c2_data/20261115_232051Z/` (11.7 s).

| Count | Addendum | Measured |
|---|---|---|
| rows passing the box (no label filter, no Mexico) | 20,706 | 20,706 |
| rows A | 20,192 | **20,190** |
| dropped low / high by B1 | 71 / 443 | 71 / **444** (the hard-cut row is counted separately) |
| exact rows | 586 | 586 |
| exact rows with style | 579 | 579 |
| rows S | 1,634 | **1,633** |
| Barcelona rows A / exact | 977 / 24 | 977 / 24 |

Why rows A differ by 2 (verified on the same frame, scratch `c2b_coder/variants.py`):
- B1 says "recompute m and d once on the kept rows, then drop |z| > 3.5 again". I read "drop again" as
  acting on the kept rows (a row dropped in pass 1 stays dropped). The architect's measurement script
  (`scratchpad/c2b/a4.py`) re-tests **every** row against the pass-2 m and d, which re-admits 2 rows.
  With that reading this code gives exactly 20,192 (`DataConfig.label_retest_all=True`). One of the two
  rows is an S row (1,634 vs 1,633). Exact and Barcelona counts are the same under both readings.
- The B1 real-world hard cut (lap < 0.92 T_ref) removes 1 row (F1 25 Brazil Quali, 63.387 s vs pole
  69.511 s, gap -6.9 %), applied before the label filter; that row would also be dropped by B1 (robust z
  < -3.5), so it does not change rows A. Under the architect's script it is among the 71 "dropped low";
  here it is counted by the hard cut, so the label filter's own low drops are 71 and its high drops 444.
- Label filter, dropped by session (sign of the final gap): Race 50 low / 118 high; Quali 21 / 55; TT 0 / 271.
- Groups: 138 (game, track, session), smallest 40 rows.

B1 real-world check (`rlsetup/critic/sources/f1_reference.py`, season-matched poles from
`docs/research/c2_real_lap_times.md`; F1 26 = F1 25 + 2026 Season Pack -> 2026 poles):
- r = fastest f1laps TT top-10 lap / T_ref is inside [0.97, 1.05] on 36 of 39 (game, track) pairs.
- **Flagged (r < 0.97), for review, not dropped:** F1 26 Canada 0.962, F1 26 Monaco 0.964, F1 26 Singapore 0.960.
  All three are 2026-pack laps faster than the 2026 real pole, matching the research note's reading that
  the 2026 pack is faster than the real 2026 cars.
- Skipped (no dry season-matched reference): F1 25 Las Vegas (wet pole), F1 26 Abu Dhabi, Bahrain, Brazil (wet),
  Las Vegas, Qatar, Saudi Arabia. 2,481 rows A have no reference and are not cut.
- Hard cut: 1 row (see above); no group above 1 %.

B2: `DataConfig.drop_tracks=('mexico',)`, applied in `load_exports` after the ids are made (755 setups,
37 laps removed) and again in `build_frame`. 24 tracks remain.

B3: `GeometryConfig.z_min_abs_corr=0.80`; `style_data.track_z_table` takes z for Imola and Las Vegas from
their bacinger candidates (length test passed, |corr| 0.840 / 0.817), `z_provisional=true` in dataset.npz
and summary.json. `TrackScaler` is fitted on non-provisional training tracks only (`train.fit_track_scaler`).
Style geometry numbers keep the full gate (S_corner_peak_angle stays NaN on these tracks).

B6: `person` = integer per username across games, made in `load_exports` before the names are discarded;
stored in dataset.npz as an integer array (like `driver`). 2,633 persons in rows A, 123 with rows in both
games. Fold sets in `rlsetup/critic/folds.py` (written per training run to `folds.json`, which stores row
positions -> folds and the data hash, never a person or driver id). data hash of this build:
`ababb09233ddc02f1021804ece7816d029f4a83b8642205bd3f0376fbbb9d135`.

B4: `rlsetup/critic/style_transform.TrackStyleTransform` (new class; the C2a `StyleTransform` is kept
unchanged for its tests): log rates -> track median of training cells (>= 5 cells, else global + flag)
-> robust z (IQR/1.349) of the centred training cells -> clip +-3 -> NaN -> 0; output the 9 B/T/S numbers.

## Step 6: R0 and R1 on real data (data build 20261115_232051Z, data hash ababb092..., 1 thread)
R0 (`--runs R0 --smoke`): `runs/critic_c2/20261116_072521Z/` (6.1 s, F_cell repeat 0 fold 0; smoke budgets:
20 epochs / patience 5 for pretrains, 100 steps for fine-tunes). Measured:

| Fit | s | per unit |
|---|---|---|
| nuisance-only A / ridge A | 0.08 / 0.24 | |
| nuisance-only exact / ridge exact | 0.006 / 0.016 | |
| critic 3, twin 0, 3 wd x 20 epochs | 2.07 | 0.034 s / epoch |
| critic 2 pretrain, 3 wd x 20 epochs | 2.16 | 0.036 s / epoch |
| critic 1 fine-tune, 4 L2 x 100 steps | 0.37 | 0.0009 s / step |
| critic 2 fine-tune | 0.26 | |
| Barcelona head (3 L2 x 20 epochs) | 0.03 | |

Projection of R1-R9 (3 repeats, 1 thread, B1-B20 runs only): **upper 2.65 h** (every fit runs 500 epochs /
2,000 steps for every grid value), **typical 0.73 h** (150 epochs / 300 steps per grid value; an exploratory
full critic-3 fit stopped at 115 epochs). Per heavy fit: critic 3 / critic-2 pretrain 52-54 s upper,
15-16 s typical. Within the 4 h budget, so no repeat cut. The B21-B23 runs (R3b, R3c, R3d, R5b) are added
to the projection below once implemented.

R1 (`--runs R1`, 3 repeats): `runs/critic_c2/20261116_072548Z/` (30.1 s, 220 fits; projected 19 s). This is
the directory the sim-runner continues with `--resume` (R1 fits are recomputed on resume because the code
digest changes with B21-B23; 30 s). Selected numbers (bars are not judged here; R1 is baselines):
- ridge on A, pair accuracy: F_cell 0.566 [0.540, 0.595], F_person 0.563 [0.536, 0.593] (architect's proxy 0.566);
  R2 gain F_cell 0.009, F_person 0.035, F_hash 0.009 [0.005, 0.013].
- ridge on exact rows vs nuisance-only: R2 gain F_cell -0.035 [-0.078, 0.000], F_person -0.029 [-0.070, 0.009] (579 rows).
- Barcelona (157 pairs): Barcelona-only 0.584, pooled 0.571, leave-out 0.561, pooled + delta(alpha=10) 0.580;
  pooled - Barcelona-only -0.013 [-0.107, 0.072].
No output of either run contains an account path or username (scanned); the logged run dir is now
repo-relative.

## Deviations and interpretations
1. **B1 pass 2** acts on the rows kept by pass 1 (textual reading); switch `label_retest_all` gives the
   architect's 20,192. Flagged for the architect.
2. **B1 order**: box (S2 step 4) -> hard cut -> label filter. The addendum computed its counts on the rows
   that pass the box, so the box moved before the label (it was after the 8 % cut in C2a).
3. **Nuisance refit for evaluation** (`train.refit_nuisance`): every fit ends with a closed-form refit of
   the nuisance on all its training rows (critic frozen). The jointly trained nuisance sees only the
   inner-fit rows and, for twins, a bootstrap of them; measured on F_cell fold 0, critic 3's R2 gain was
   -0.066 with the joint nuisance and +0.005 after the refit (ridge +0.004), i.e. the joint-nuisance R2
   measured the nuisance, not the critic. The comparators "critic 3 + exact-row nuisance" and "A-pretrained
   critic 2 + nuisance" are therefore the closed-form optimum of "only the nuisance trained". q is unaffected.
4. **Ridge solver**: block coordinate descent did not converge (3,000 sweeps still 0.4 above the optimum);
   `nuisance.ridge_backfit` is now an exact sparse direct solve (scipy SuperLU, 0.2 s per fit).
5. **B8 patience**: 30 (B8) is used; the C1 amendment's 200 is superseded by B8. Smoke (R0): 20 epochs,
   patience 5; critic 1/2 smoke fine-tune 100 steps.
6. **Weight-decay / L2 grids** are selected per fit (per fold and twin) on that fit's inner validation set.
7. **CV metrics of twinned critics** use the mean of the two twins' predictions; q_min (-max over twins) is
   only in the bundle.
8. **Pairs** are scored on the setup part of the prediction (J + Barcelona head = -q, what TD3 ranks);
   the nuisance's date / team terms can differ within a pair group and are left out. Within-group
   Spearman is reported on the full prediction, without CI.
9. **Barcelona head optimiser**: the B8 settings (Adam 1e-3, batch 256, <= 500 epochs, patience 30); B11
   gives only the architecture, grid and inner split.
10. **R6 shuffle ablation**: uses the real fit's selected L2 and its twin-0 bootstrap weights; s~ is
    shuffled within strata separately in the training and the test rows; m(s~) always gets the real s~.
11. **R2** compares PCA vs physics groups with one non-bootstrap critic 3 per fold (no twins).
12. **Strict / unguarded variants** (R3, R5) use the twin-0 configuration (seed, bootstrap) of the main fit.
13. **Leave-Barcelona-out** is trained on **all** non-Barcelona rows A (2 fits, not per fold; corrected wording,
    architect ruling 13): about 25 % more off-track data than a pooled fold fit, so pooled - leave-out is
    conservative. No Barcelona row is ever in its training. **Barcelona-only** uses F_cell repeat 0.
14. **R9 Barcelona style reference**: the OCD drivers' style numbers are read from style-rl's
    `targets/style12.json` at the pinned commit (`git show`), never from a working copy.
15. **B4 class**: the C2a `StyleTransform` is kept (its tests); C2b uses the new `TrackStyleTransform`.
16. **MDE** = 2.8 x bootstrap SE (80 % power, two-sided 5 %).
17. **`person`** is stored in dataset.npz as integers (like `driver`); folds.json stores row positions only.
18. **Pipeline robustness on tiny data** (only reachable on test fixtures): a ridge baseline with no training
    rows returns NaN predictions (treated as untested rows, e.g. the Barcelona-only variant on data without
    Barcelona); `pretrain` / `finetune` raise `ValueError('... no fit rows ...')` instead of training on
    nothing; the guard returns no term on zero rows.

## Coordinator additions
**B21 hook (extra loss on (p, track, game); the term itself is NOT implemented).**
`rlsetup/critic/train.py: ExtraLoss` with `spec() -> dict` and `bind(d, fit_rows, scaler, support) -> callable`.
`pretrain(..., extra=None)` binds it once per fit on the in-fold fit rows only (never held-out or inner-
validation rows); each minibatch calls `fn(model, batch, gen)` with p, z, g, track, game, rows. The term
has its own torch generator, so the batch order, guard draws and initialisation are bit-identical with and
without it. `scripts/critic_c2.py: c3_fit(..., extra=None)` adds `extra.spec()` to the fit params (resume
hash) only when given, so with/without fits are separate cached fits under identical folds and seeds.
Fit info records `extra_loss: {spec, mean_term}` (None without). Test
`test_extra_loss_hook_in_fold_and_identical_without`: a zero term reproduces critic 3 bit for bit, the hook
only sees training rows, a non-zero term changes the fit. Wiring a B21 term into R3/R9 runs is left for B21.

**Spec 12.21 (TD3's Q is always critic 1).**
- `bundle.ROLES`: critic1 -> `td3_q`, critic3 -> `style_free_reference`, critic2 -> `comparison`;
  `save_bundle(kind='critic3' | 'critic2', role='td3_q')` raises; `CriticBundle.load(path, require_td3_q=True)`
  raises unless the file is a critic-1 `td3_q` bundle (`b.role` exposed).
- R9 writes `metrics/R9.json: td3_q = {bundle: bundles/critic1_final.pt, kind: critic1, style_rows: exact,
  e_gate, rule}`; critic 3 stays a separate bundle. `fallback_trigger` never selects critic 3 and reports
  `td3_q: critic1`. No code path designated critic 3 as TD3's Q before either (the void B16 clause was
  not implemented as a switch).
- Hook for the B23 scalar gate / shrinkage on the style path: buffer `e_gate` (scalar, default 1) in
  `GroupCritic` and `FullCritic`, multiplying e where it enters the heads (guard, fine-tune, prediction and
  bundle all go through it). 1 = B9/B10 exactly (bit-identical), 0 = critic 1 equals its critic 3. Not
  trained or set anywhere in C2b. Test `test_td3_q_is_always_critic1_and_e_gate_hook`.
- Open for B23: whether the delivered critic 1 comes from exact rows or rows S, and how `e_gate` is chosen.

## B21-B23 progress (binding addendum items, 16 Nov)
- [x] Config: `c2_config.MetaPriorConfig` (fast 20 %, min 20 rows, kappa 300, c grid {0, .03, .1, .3}),
  `SessionConfig` (S22b alpha 10, Race/Quali; S22a drops Quali), `LadderConfig` (L1-L3); bars `b21_adoption`,
  `b21_trigger`, `b21_triggered_delivery`, `b21_grid`, `s22a`, `s22b`, `b23` added to `BARS` (bars.json).
- [x] `rlsetup/critic/meta_prior.py` (m^in fast set, cross-fitted w / r_bar, lambda, Pi, state);
  `rlsetup/critic/sources/f1_meta.py` (published meta -> p, sensitivity only).
- [x] Training: `pretrain(prior=, wd_grid=, session_dev=)`, `finetune(level=)` (B23 rung 2: e_gate = gamma,
  fixed L2, alpha of m(s~)), critic 1 inherits Pi frozen, `predict` returns 'prior' and 'session_dev';
  nuisance `session_dev` block (S22b); per-session subsets in every metric block (B22.2).
- [x] Bundle: stores the in-fold prior (lambda, c, m, w, r_bar), refuses a published prior, `ctx.flags['meta_prior']`,
  `b.J(p, ctx, prior=True)`, `b.prior(p, ctx)`.
- [ ] Script runs R3b / R3c / R3d / R5b, delivered-critic-3 resolver, ladder, R9 delivery, projection.
- [ ] Tests (B19 additions), R0 smoke rerun, full suite.

Measured on the real build (F_cell repeat 0, fold 0 training rows; not a result run): fit_meta_prior 0.6 s;
2,259 pairs scored in cross-fitting; pooled-m fallback (fewer than 20 training rows) for F1 25 Madrid and
F1 26 Imola (the addendum expected none); mean w over tracks camber F/R 0.17 / 0.19, toe F/R 0.11 / 0.18,
f_F 0.09, LSD drive / coast 0.08 / 0.05, rake 0 (E21.10: camber 0.17 / 0.19, toe 0.16 / 0.18, f_F 0.10, LSD
0.08 / 0.09); in-fold m vs published: mean |diff| / SD 0.17, 55 % of values equal.
- [x] Script: R3b, R3c, R3d, R5b, `c3_source` / `delivered_variant` / `delivered_prior` (delivered critic 3 for
  R4-R9; critic 2 inherits Pi), `remedy_ladder` (written to metrics.json), R9 delivery per ladder rung,
  R0 extended with B21-B23 wiring timings, projection incl. R3b/R3c/R3d/R5b. `--meta-sources`, `--drop-quali`.
- [x] Tests: `tests/test_c2b_units.py` (+8 B21-B23 tests), `tests/test_c2b_pipeline.py`
  (`test_forced_b21_delivery_and_b23_rung2`: R3c and R5b exercised by relaxing / tightening bars in the test only).
- [x] R0 smoke rerun on real data, R1 rerun, full suite (below).

### R0 / R1 after B21-B23 (these are the current run dirs)
- R0: `runs/critic_c2/20261116_075104Z/` (8.7 s). New wiring timings (smoke budgets, F_cell r0 f0): meta prior
  fit 0.11 s; critic 3 with Pi (one wd, 20 epochs) 0.75 s; S22b critic 3 (3 wd x 20 epochs) 1.5 s; critic 1 at L3
  (100 steps) 0.09 s; critic 1 from the B21 critic 3 0.37 s. Checks in `metrics/R0.json: b21_b23_wiring`:
  lambda(c = 0.1) = 0.247, SD of lambda Pi on training rows 0.016 %, Pi bit-identical after critic 1's fine-tune,
  critic-1 L3 e_gate 0.03, pooled-m flags F1 25 Madrid and F1 26 Imola.
- **Projection (3 repeats, 1 thread): upper 3.22 h, typical 0.90 h; with the conditional R3c and R5b: upper 3.69 h,
  typical 1.03 h.** Fits added: R3b 45 + 10 critic-3 fits at one wd (+1 final), R3d 20 (3 wd), R3c 75 (only if
  triggered/adopted), R5b 180 + 6 final (only if rung 0 fails). Per heavy fit (upper / typical): critic 3 53 / 16 s,
  B21 critic 3 (one wd) 18 / 5.3 s. Within the 4 h budget even in the upper bound; no cut needed.
- R1: `runs/critic_c2/20261116_075115Z/` (29.6 s; same numbers as above, now with per-session rows). **The
  sim-runner should resume this dir** (`--resume runs/critic_c2/20261116_075115Z`, default run list). The earlier
  dirs `20261116_072521Z` / `20261116_072548Z` have an older bars.json and cannot be resumed (by design).

### What B21-B23 do in the code (for QA)
- B21.1 m^in: `meta_prior.fast_set_m`, fast set = residual <= the 20 % quantile per (game, track) of the fold's
  training rows (B15 nuisance-only residual fitted on those rows); fewer than 20 rows -> the game's pooled fast
  set = union of its per-(game, track) fast sets (flagged; F1 25 Madrid and F1 26 Imola on F_cell r0 f0).
- B21.2 w: `cross_fit_pairs` (halves by sha256(data_hash|meta|cell) mod 2; the m of the other half; pairs within
  (game, driver, track, session) from `pairs_of`; win = closer setup has lower residual; |dist diff| > 1e-9);
  kappa 300; w = max(0, 2 r_hat - 1). The published variant uses the published m in both halves.
- B21.3: lambda = c SD_train(setup part of the R3 twin-0 fit, no head) / SD_train(Pi); c selected per fold on the
  B8 inner validation MSE (R3's c = 0 fit vs the three R3b fits, same seed / bootstrap / inner split, wd fixed to
  the c = 0 choice; tie -> smaller c). Pi enters J outside `scale` as a fixed per-row offset during training.
- B21.4/5: `metrics_R3b` (adoption, trigger from the R3 metrics), `run_R3c` / `metrics_R3c` (full grid at the
  modal c, strict variant with the prior, triggered-delivery rule, statement), `c3_source` makes R4-R9 use the
  delivered critic 3; critics 1 and 2 inherit Pi frozen (critic 2 via `delivered_prior`).
- B21.7: bundle stores lambda, c, m (per game, track), w, r_bar; `ctx.flags['meta_prior']` in {track, pooled_w
  (track has no scored pairs), none}; `b.J(p, ctx, prior=True)`, `b.prior(p, ctx)`; a published prior raises.
- B21.1 report text: the published meta is f1laps' "top 100 setups ranked by lap time" per track and game,
  half-steps consistent with a per-parameter median (`docs/research/c2_meta_and_tt_rules.md`): it is computed from
  the labels, held-out folds included, so it is a leak; it stays sensitivity-only (`metrics_R3b` writes this note
  and the in-fold vs published closeness).
- B22: per-session rows in every metric block (`Evaluator.subsets`); R3d S22a (`s22a_train`: Quali removed from the
  training rows only) and S22b (`NuisanceEncoder(session_dev=(10, ('Race', 'Quali')))`, `predict()['session_dev']`);
  fixed rules in `metrics_R3d`; `--drop-quali` applies the S22a outcome if it ever fires.
- B23: `finetune(level=(name, gamma, l2, alpha_m_s))` (e_gate = gamma at training and query, saved in the bundle);
  `run_R5b` / `metrics_R5b` (per-fit level selection among B9, L1, L2, L3 by inner-validation MSE, tie -> earlier;
  bars on the selected mixture and on L3); `remedy_ladder` (fixed order, in metrics.json); R9 delivers critic 1
  from the ladder's rung and refuses (RuntimeError) on 'pending' or 'implementation_fault' outside smoke.

### Deviations and interpretations (B21-B23)
19. **Pooled fast set** (B21.1 "the game's pooled fast-set median") = median of the union of that game's
    per-(game, track) fast sets; m is given only to tracks with training rows (unseen track -> Pi = 0).
20. **Measured differences to the addendum's E21 numbers**: 2,259 pairs scored in the cross-fitting on F_cell r0
    f0 (E21.6 quotes 2,865 over all pairs); toe F weight 0.11 vs 0.16, LSD coast 0.05 vs 0.09; two (game, track)
    use the pooled fallback (the addendum expected none). The architect's script skipped groups below 20 rows
    instead of pooling; mine follows the B21.1 text.
21. **Adoption rule item 3 (Barcelona)** is judged on F_cell (the fold set of the Barcelona delivery rule); the
    F_person Barcelona delta is reported too.
22. **B21.5 trigger**: "misses any R11 ranking bar on A" is evaluated on both F_cell and F_person (pair accuracy,
    CI, vs ridge) plus the strict variant on F_cell r0; a miss on either fold set triggers.
23. **Published sensitivity c**: the modal selected c of the fold set; if that is 0, the smallest non-zero c
    (0.03), flagged in `selection.published_c`.
24. **R3c with modal c = 0** for a fold set reuses the R3 fits (B21 at c = 0 is critic 3 bit for bit).
25. **R9 final c** = the modal selected c over all R3b folds (tie -> smaller).
26. **S22a / S22b** compare against plain R3 twin 0 (no prior), with R3's seed key.
27. **R8 sensitivity** stays plain critic 3 (and compares with plain R3), whatever critic 3 is delivered.
28. **R6 shuffle ablation** stays on the rung-0 (B9) critic 1; the run table has no shuffle runs for rung 2.
29. **Critic 2 rung 2** (B23 "rungs 1-2 apply to it in the same way") has no fits in the run table; `finetune(level=)`
    supports it, but no run is scheduled. Flagged for the architect.
30. **B23 level selection** is per fit (fold, twin) on that fit's inner validation; the bars are tested on the
    resulting mixture; R9 selects the level on the mean inner-validation MSE of its two twins.
31. **Rung 1** is only tried when the B16 trigger fires (the ladder says 'pending: R5 --rows S' until R5_S exists).
32. Output keys avoid personal names (`escalate_iff`, `escalate_TT_reference_decision`).

## Fix round (QA C2b review + architect rulings, 16 Nov)
Scope cut by the coordinator (limited usage): items 1-4, m3 and m14 done; the rest deferred (below).

| Item | Change | Test |
|---|---|---|
| QA M1 resume safety | `C2Data.build_sha256` = sha256 over dataset.npz + summary.json bytes (`c2data.load_c2`); stored in `folds.json` and asserted by `load_folds` (resume with another build raises "build hash ... start a fresh run directory"); Store base adds `build_sha256`, `threads`, torch/numpy/scipy versions; `CODE_FILES` now `rlsetup/critic/**/*.py` (incl. `sources/`) + `rlsetup/style/*.py` + the script | T2 `test_c2b_pipeline.py::test_resume_refuses_when_build_content_changes` (copy build, style x1.7, track_features reversed, resume raises; digest covers style + sources) |
| QA M2 e = 0 bar | R5 / R5b store `setupA_c3` = critic 3 **with the same delivered Barcelona head** on rows A (what TD3 ranks); `metrics_R5` / `metrics_R5b` compare `setupA_e0` with it, twin mean | T1 `test_c2b_units.py::test_e0_bar_compares_like_with_like` (head perturbed, equality on all rows incl. Barcelona) |
| QA M3 = architect must-change 1 | R5b fits a comparator per level, `c3E{L}` = `finetune(..., 'c3_exact', cfg with nuisance.alpha_style = level alpha)` per fold/repeat/twin; bars judged on c1 vs matched comparator, per level and for the per-fit selected mixture; the alpha = 10 comparison kept as `report_only_vs_alpha10_comparator` (with `credit_of_the_alpha_change`) | T4 `test_c2b_pipeline.py::test_rung2_matched_comparator_and_selection_set` (every `c3E{L}` fit carries `alpha_m_s` = the level's alpha; metrics use it) |
| Architect must-change 2-3 | `LEVEL_ORDER = ('L1','L2','L3')` in `r5b_selection` and R9 `rung2_selected_level`; `BARS['b23']` adds `comparator`, `selection_set`, `critic2` (rungs 0-1 only, ruling 29; also in `metrics_R5b.critic2_note`) | same T4 (B9 never in `selected_counts` / selection; BARS keys) |
| Architect item 13 | report wording corrected (deviation 13 above) | - |
| QA m3 bars_judged | R3: `bars_judged` {R11 ranking bars, `F_hash_r2_gain_gt_0`, `all_R11_met`}; R4: `barcelona_F_*.bars_judged.barcelona_pair_acc_ge_0.55` (delivered variant); R7: `critic2_{fs}.bars_judged.mae_below_nuisance_ci_excludes_0` | covered by the pipeline run (keys read in the results doc); no dedicated T10 (deferred) |
| QA m14 | B21.4 item 3 with no Barcelona pairs: `adoption.barcelona_item_evaluable = false`, item treated as pass (interpretation, recorded in metrics) | - |
| QA m1 (incidental, needed for M2) | `base_c3` attaches the R4 head only if R4 delivers `pooled_plus_head` (as R9) | T1 indirectly |
| QA m4 doc | fallback command documented as `--runs R1,R5,R6,R7 --rows S` | - |

**Deferred** (not done in this round, by coordinator decision): QA m2 (bundle td3_q structural guard), m5 (style transform one row per cell on rows S), m6 (no pooled m for absent (game, track)), m7 (prior fit on train minus inner val), m8 (R6 shuffle L2 reselection), m9 (Barcelona head in critic-1 fine-tune loss), m10 (twin m equality check in bundle), m11 (Barcelona fits under `--drop-quali`), m12 (NaN first-epoch validation error), m13 (source-grep tests), i1 (pickled encoder ids: docstring corrected only), T3, T5-T10, and the report-only script (q_min pair accuracy, same-c published vs in-fold, same-month pairs, R3c delta table).

## How to run
- Data build: `.venv/bin/python scripts/critic_c2_data.py` (-> `runs/critic_c2_data/<UTC>/`).
- Smoke: `.venv/bin/python scripts/critic_c2.py --runs R0 --smoke [--threads 1]`.
- Full: a fresh run dir (the fix round changed bars.json and the fit key), in two invocations per the rulings
  "Approved to run": `--runs R1,R2,R3,R3b,R3c,R3d`, check `metrics/R3d.json`, then `--resume <dir> --runs R4,...,R9`
  (default run list R1,R2,R3,R3b,R3c,R3d,R4,R5,R5b,R6,R7,R8,R9; R3c / R5b write `needed: false` when not needed).
  B16 fallback afterwards: `--runs R1,R5,R6,R7 --rows S` (R1 writes the `_S` baselines, QA m4) (refuses unless the trigger fired). Then rerun `R9` so the
  ladder sees `R5_S`. `--meta-sources` points at the exports with `meta_setups_wide.csv` (default `setup_data/*`).
- Tests: `.venv/bin/python -m pytest -q` (C2b only: `tests/test_c2b_units.py tests/test_c2b_pipeline.py`, ~35 s).

## Test results
- Fix round (16 Nov): full suite `735 passed, 14 skipped in 246.27s (0:04:06)`; C2b only `38 passed in 37.23s`.
- Full evaluation (coordinator instruction) in the fresh run dir `runs/critic_c2/20261116_081601Z`: part 1
  `R1,R2,R3,R3b,R3c,R3d` (S22a drop_quali false, S22b no escalation), part 2 `R4..R9` exit 0; TD3 Q = critic 1
  rung 0 (B9). Results: `docs/results/critic_c2.md`. `timing.json` is overwritten per invocation (holds part 2 only).
- Full suite `.venv/bin/python -m pytest -q` (16 Nov, after B21-B23): `714 passed, 14 skipped in 220.51s (0:03:40)`
  (the 714 include the other agent's AC tests; before C2b-B21 the line was `705 passed, 14 skipped in 207.83s`).
- C2b only: `tests/test_c2b_units.py` 26 passed (2.5 s), `tests/test_c2b_pipeline.py` 9 passed (~31 s).
- Changed C2a test: `tests/test_c2_data.py::test_gap_is_per_game_and_filtered` now checks the B1 label
  (per-session median, two passes, 6 + 0 dropped) instead of the removed 8 % step.
- READMEs (`scripts/README.md`, `tests/README.md`) were already modified by someone else in the working tree and
  are not edited by me; the run script's docstring and this report document the C2b commands.
