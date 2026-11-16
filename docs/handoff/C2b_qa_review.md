# C2b QA review: critics, training, evaluation, bundle

**Verdict: PASS-WITH-FIXES.** No blockers. There are 3 majors, and all three must be fixed before the sim-runner resumes R2–R9.

**Leakage is clean**, which was the biggest risk. I multiplied every held-out row by 1,000 (and shifted its date), including in fitted objects the shipped tests do not cover. Not one training-side quantity moved. Determinism holds on real data. Privacy is clean on every run output.

The 3 majors:
- **M1.** `--resume` silently reuses stale fits when the data build changes style, z or other non-key columns.
- **M2.** The "critic 1 at e = 0 vs critic 3" bar compares a critic with the Barcelona head against one without it.
- **M3.** The B23 rung-2 critic-1 fits are judged against a comparator whose m(s̃) penalty differs. On real F_person rows that difference alone is worth +0.019 R², the same size as the −0.02 bar.

Nothing in the repo was edited. My throwaway scripts and outputs are in `scratch/qa_c2b/`: `check_leak_head.py`, `check_resume.py`, `check_real.py`, `check_alpha.py`, `full_suite.txt`, `r0_a/`, `r0_b/`.

## What I ran (exact output lines)

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -q tests/test_c2b_*.py tests/test_c2_*.py -rs` | `95 passed, 1 skipped in 33.51s`. Skip: `tests/test_c2_knobs.py:115: sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path` |
| Full suite `.venv/bin/python -m pytest -q` | `732 passed, 14 skipped in 226.17s (0:03:46)`. The two pre-existing C2a failures are gone. |
| Leakage, `check_leak_head.py` (synthetic C2b build, F_cell and F_person fold 0). Held-out rows ×1000 on gap, p, lap_s and style_raw; held-out dates +5,000 days. | All `True` on both fold sets for: `nuisance_only` A and exact, `ridge_A`, `ridge_A` + Barcelona δ, `ridge_exact` (predictions on training rows); critic-3 twin with bootstrap; Barcelona head; critic-1 twin (model and refitted nuisance); `c3_exact` comparator nuisance; critic-2 pretrain and fine-tune; meta prior m, w and **λ**; `sst_w` training means. **Verified.** |
| Two R0 smoke runs on real data, `--threads 1`, into scratch | `R0 metrics identical except timing/projection: True`, `folds.json identical: True`, `bundle roundtrip: {'critic3': 0.0, 'critic1': 0.0, 'critic2': 0.0}`, `pi_identical_after_critic1: True`, `critic1_L3_e_gate: 0.0299999993`. Projection: upper 3.37 h / typical 0.95 h (3.86 / 1.08 with R3c and R5b). |
| `runs/critic_c2/20261116_075104Z` and `…075115Z` against the current code | `bars.json == current BARS: True`, `code digest same as now: True`. R1 numbers match the coder report: ridge A pairs F_cell 0.566 [0.540, 0.595], F_person 0.563 [0.536, 0.593], n_pairs 2,878; ridge exact R² gain −0.035 [−0.078, 0.000] / −0.029 [−0.070, 0.009]; F_hash 0.0089; Barcelona 157 pairs. |
| Privacy: `grep -i '/Users/' 'ibrahim'` over text and binary, plus a whole-token scan against 2,948 export usernames | 0 hits in the 12 + 446 files of the two run directories and in my two scratch R0 runs. **Verified.** |
| Stale resume, `check_resume.py` | `data_hash equal: True rows_sha256 equal: True style differs: True` → `fits recomputed on resume with the new build: 0 of 80` → `max \|stale - fresh\| = 0.00345` (M1). |
| Documented fallback `--runs R5,R6,R7 --rows S --force-fallback --resume <run>` | `FileNotFoundError nuisE_F_cell_r0_f0_S.npz` (m4). |
| e = 0 bar, `check_leak_head.py` | `e0 vs critic3: max \|diff\| off-Barcelona 0.0 on Barcelona 0.242` (M2). |
| Rung-2 comparator confound, `check_alpha.py`, real exact rows, 3 repeats, person-cluster CI. Nuisance-only with α_m = L vs α_m = 10. | F_person: α 100 **+0.0192 [+0.0021, +0.0382]**, α 300 +0.0165 [−0.008, +0.045], α 1000 +0.0032. F_cell: +0.0015 / −0.0005 / −0.0062 (M3). |
| Real data, `check_real.py` | F1 25 madrid and F1 26 imola have **0 rows A** but get a finite pooled m (m6). Inner-validation rows are **11.0 %** of the fast set that defines m (m7). On rows S, track medians computed over rows vs over cells differ by up to 4.12, and the scale ratio ranges 0.67–1.04 (m5). |

## Findings

| # | Sev | File:line | Failure scenario | Fix | Status |
|---|---|---|---|---|---|
| **M1** | **MAJOR** | `scripts/critic_c2.py:60-61` (CODE_FILES), `:135-137` (Store base), `:1802` (default `--data` = latest build); `rlsetup/critic/c2data.py:96,110` | The fit-cache key is data_hash (export CSV sha256s only) + rows_sha256 (setup hash + lap_s) + config + a code digest of `rlsetup/critic/*.py` and the script. Content of a build that changes neither of those keys is outside the key: style numbers, z_raw, gap/label, sessions, assists, dates, cell/exact flags, `has_style`. Example: Ibrahim records Barcelona edges (§12.18), the build is redone, and the sim-runner runs `--resume runs/critic_c2/20261116_075115Z` (the coder's instruction). `--data` defaults to the latest build, so every fit is reused with the old style and z. Proven: 0 of 80 fits recomputed, stale predictions differ by up to 0.0035. Also outside the code digest: `rlsetup/critic/sources/*.py` (`f1_meta.py` / `f1_game_map.py` feed the published-meta fits) and `rlsetup/style/*.py`. | Put the sha256 of `dataset.npz` + `summary.json` into the Store base (and into `folds.json`, asserted on resume). Make the code digest recursive over `rlsetup/critic/**` and `rlsetup/style/*`. Add torch/numpy/scipy versions and `--threads` to the base. | Verified |
| **M2** | **MAJOR** | `scripts/critic_c2.py:1090-1091` (`base_c3` attaches the R4 head), `:1114` (`predict(..., e_zero=True)` keeps `with_bcn=True`), `:1337-1339` (comparator `c3a` = R3/R3c predictions made before any head existed); same at `:1223` and `:1281-1284` (R5b) | B16 (iii) and the critic-1 / B23 bar "at e = 0, pair accuracy on A within 0.01 of critic 3" should be 0 by construction (frozen inherited weights). Instead, on Barcelona pairs it measures the Barcelona head. Proven: off-Barcelona diff 0.0, on Barcelona up to 0.24 % in J. Pooled, the effect is about (157/2,878) × the head's effect on Barcelona pairs. A head that moves Barcelona pair accuracy by about 0.18 trips the 0.01 bar, which fires B16 and moves TD3's Q down the ladder for no style reason. The `barcelona` subset row of this metric is meaningless today. | Compare like with like. Either predict the e = 0 rows with `with_bcn=False`, or build the critic-3 comparator on A rows with the same head (predict `f3` + head on `te` in R5 and store it). Then add the test T1 below. | Verified |
| **M3** | **MAJOR** | `rlsetup/critic/train.py:305-307, 425` (level sets α of m(s̃) = 100/300/1,000 for critic 1 and its refit); `scripts/critic_c2.py:1277-1278` (R5b compares with the R5 `c3E` comparator, refit at α = 10) | B23 rung 2 tests ΔR²(c1 − c3) > −0.02. The critic-1 side's nuisance shrinks m(s̃) and the comparator's does not, so the bar mixes "style path" with "nuisance penalty". On real F_person exact rows the α change alone is +0.019 [+0.002, +0.038] R² at L1. That is the size of the bar margin. A critic 1 whose style path hurts can pass rung 2 because its nuisance is better regularised. The B23 L3 "implementation fault" logic also assumes the two sides differ only by the style path. | Fit the comparator per level: `finetune(d, etr, f3, 'c3_exact', replace(cfg, nuisance=replace(cfg.nuisance, alpha_style=α_L)))`, stored as `c3E{L}`, and use it in `metrics_R5b`. If the architect intended the asymmetry, record that in the addendum. Ask the architect either way. | Verified (size measured with the nuisance-only proxy) |
| m1 | MINOR | `scripts/critic_c2.py:1085-1092` vs `:1601, 1618-1622` | If R4 decides `'pooled'` (head more than 0.03 below pooled), R9 delivers critic 1 and critic 3 without the head. R5, R5b and R6 still evaluate critic 1 with the head attached. The CV bars and ladder then describe a different critic from the delivered one. | In `base_c3`, attach the head only if `metrics/R4.json` says `pooled_plus_head`, and add the decision to the fit params. | Verified (code path) |
| m2 | MINOR | `rlsetup/critic/bundle.py:64-66, 147-148` | The TD3-Q guard checks only the label. Proven: a critic-3 `Fitted` saved with `kind='critic1'` loads with `require_td3_q=True` as `td3_q`, has `n_e 0`, and its q does not depend on style. A critic 1 with `e_gate = 0` is accepted too. Neither is reachable from the script today, but spec 12.21 is enforced only by convention. | In `save_bundle(kind='critic1')`, assert each twin has `f.kind == 'critic1'`, `model.n_e > 0`, `float(model.e_gate) > 0` and `style_tf is not None`. In `load(require_td3_q=True)`, assert `arch['n_e'] > 0` and every `e_gate > 0`. | Verified |
| m3 | MINOR | `scripts/critic_c2.py:761-774, 1051-1079, 1506-1544` | Three bars have no pass/fail anywhere in code, so the report writer must judge them by hand: (a) the R11 "held-out-setup (F_hash) R² gain > 0" bar (only the four ranking bars are evaluated, inside the B21 trigger); (b) "Barcelona pair accuracy ≥ 0.55 (point)"; (c) the critic-2 bar "MAE below nuisance-only, paired CI excluding 0, on F_cell and F_person". | Add `bars_judged` blocks to R3 (all five R11 bars), R4 (≥ 0.55 on the delivered variant) and R7 (CI upper bound of `mae_c2_minus_nuis` < 0 per fold set). | Verified |
| m4 | MINOR | `scripts/critic_c2.py:1318` (`nuisE…{tag}`); docstring `:20`; coder report "How to run" | The documented B16 fallback `--runs R5,R6,R7 --rows S` crashes in `metrics_R5` with `FileNotFoundError nuisE_F_cell_r0_f0_S.npz`, because the `_S` nuisance baselines are written only by R1 with `--rows S`. | Document `--runs R1,R5,R6,R7 --rows S`, or have `run_R5` require the baselines and raise a clear message. | Verified |
| m5 | MINOR | `rlsetup/critic/train.py:309` → `style_transform.py:52-73` | B4 says track medians and the robust z come from training **cells**. On rows S (rung 1) every setup row of a cell carries the cell's style, so cells with many setups weigh more. Real F_cell r0 f0: medians shift by up to 4.12, scale ratio 0.67–1.04. Exact rows are one row per cell, so rung 0 and rung 2 are unaffected. The bundle's `style_train` mean and covariance for rung 1 have the same problem. | Fit the transform (and `style_pca_init`, and the bundle's `style_train`) on one row per cell. | Verified |
| m6 | MINOR | `rlsetup/critic/meta_prior.py:56-63` | The pooled fallback assigns a game's pooled fast-set m to every track seen in training, including (game, track) combinations with **0 rows** (F1 25 madrid, F1 26 imola on real data). A `source=None` query at Madrid then averages the real F1 26 m with a generic F1 25 median. B21.1 meant "fewer than 20 rows of an existing (game, track)". This only matters if B21 is adopted. | Fall back only when the game has at least 1 training row on that track; otherwise leave NaN, so `m_query` uses the games that have the track. | Verified |
| m7 | MINOR | `scripts/critic_c2.py:709, 869` vs `rlsetup/critic/train.py:175` | m and w for a fold are fitted on all training rows. These include the critic's inner-validation rows (11.0 % of the fast set on real data), whose labels then drive the c selection on that same inner validation. That biases the chosen c upwards. Outer metrics and the adoption decision stay clean. | Fit the selection prior on `train_rows[~val]`, or state the caveat in the report. | Verified (count); size of bias not measured |
| m8 | MINOR | `scripts/critic_c2.py:1394, 1420-1421` | The R6 shuffle ablation reuses the real fit's L2, and that L2 was selected on the real-s̃ inner validation. The shuffled fit gets no such selection, so the comparison slightly favours "real" (anti-conservative for "style helps"). Coder deviation 10. | Select L2 on the grid for each shuffle fit too (4× the small fits), or report the caveat. | Reasoned |
| m9 | MINOR | `rlsetup/critic/train.py:337, 378` vs `:487, 530-533` | Critic 1's fine-tune loss and nuisance initialisation leave out the Barcelona head. Evaluation and the closed-form refit add it. The 24 Barcelona exact rows train W and W1e against a residual that differs from the one they are scored on. | Add `base.bcn(P)` on Barcelona rows to `j0` and `pred` in `finetune`, as `predict` does. | Verified (code) |
| m10 | MINOR | `rlsetup/critic/bundle.py:72-74` | The error message says "twins must share lambda, m and w", but m is not compared. | Add `np.array_equal(f.prior.m, prior.m, equal_nan=True)`. | Verified |
| m11 | MINOR | `scripts/critic_c2.py:1043, 1046` | With `--drop-quali`, the leave-Barcelona-out and Barcelona-only critic-3 fits still train on Quali rows: they use `~bcn` / `bcn`, not `c.A & …`. | Intersect with `c.A`. | Verified (code) |
| m12 | MINOR | `rlsetup/critic/train.py:251-258` | If the validation loss is NaN at epoch 1, `best_state` stays None and `load_state_dict(None[0])` raises TypeError. That is not a clear error. Not reachable on current data. | Raise a `ValueError` naming the fit when no finite validation loss was seen. | Suspected |
| m13 | MINOR | `tests/test_c2b_units.py:252-254, 284-285, 431-432, 544-545` | Four assertions only grep the script source (`"clipped_frac_test" in src`, `'boot_weights(d.setup_hash[train]' in src`, …). They pass on renamed or dead code and do not test behaviour. | Replace them with behavioural checks (see T5). | Verified |
| m14 | MINOR | `scripts/critic_c2.py:838` | B21.4 item 3: when there are no Barcelona pairs (`bdel is None`) the item passes. The addendum is silent on this case. | Confirm with the architect, or record it as an interpretation. | Verified |
| i1 | INFO | `scripts/critic_c2.py:14-15`, `:101` | The docstring says drivers and persons are "never written to any output". The pickled `fits/*/*.pt` contain `NuisanceEncoder.maps['driver']` (game×1e9 + driver id) and `driver_mean_month` (one mean date per driver). These are integers, gitignored and not names, but the claim is inaccurate. Not present in the current runs: R0 and R1 save no `.pt`. | Correct the docstring, or strip `encoder` before `torch.save`. | Verified (code) |
| i2 | INFO | B21.5 / coder deviation 22 | The trigger evaluates the R11 ranking bars on F_cell **and** F_person. That is stricter than one fold set, so B21 is more likely to become mandatory. | Architect to confirm. | — |

## Deviations I was asked to judge

- **Deviation 3 (closed-form nuisance refit): sound.** It does not inflate metrics, and critics and baselines get the same treatment.
  - The nuisance-only and ridge baselines are closed-form ridges on all training rows.
  - The comparators `c3_exact` and `c2pre_exact` are closed-form refits on all training rows.
  - Every critic is refitted the same way: same encoder spec, same penalties, same training rows. Only the offset J differs.
  - The refit sees training rows only (leakage check: `critic1_refit_nuisance`, `c3_exact_nuisance` both True).
  - q never contains the nuisance (`test_q_independent…`).
  - A J that is pure noise can only lower the test R², so the refit cannot create gain.
  - **One exception is M3:** in rung 2 the comparator's nuisance spec (α_m) differs.
- **Deviation 8 (pairs scored on the setup part): sound, and it is what spec §8 says** ("pair accuracy and Spearman of the setup part of the prediction").
  - The setup part is J + Barcelona head + λΠ, which is exactly −q, the quantity TD3 ranks.
  - The ridge baselines are scored on their p, p×ẑ and p×δ part, so like with like.
  - The game, track, session and driver terms are constant within a pair; only date, team, controller and assists can vary.
  - Leaving out the date trend is correct for a setup-quality test. Within a driver, the later setup is faster in 0.663 of pairs (E21.5), so including it would reward the trend, not the setup.
- **Deviation 7** (CV on the twin mean): consistent with `b.J` (mean over twins). **Deviation 21** (Barcelona adoption item judged on F_cell): consistent with the Barcelona delivery rule. **Deviation 24** (R3c reuses R3 when the modal c = 0): correct, because c = 0 is bit-identical (tested).

## Bars and decision rules: code vs addendum text

| Rule | Text | Code | Match |
|---|---|---|---|
| R11 pair accuracy / CI / vs ridge / strict | ≥ 0.56; > 0.52; ≥ ridge − 0.02; ≥ 0.54 | `>=`, `>`, `>= -0.02`, `>=` (`:768-772`) | Yes (both fold sets, i2) |
| R11 F_hash R² gain | > 0 | not evaluated | **No (m3a)** |
| Barcelona delivery | pooled iff head > 0.03 below pooled (point) | `pp - hp > 0.03` (`:1071`) | Yes |
| Barcelona bar | ≥ 0.55 (point) | not evaluated | **No (m3b)** |
| Critic 1 | ΔR² CI low > −0.02 on both; \|e0\| within 0.01 | `lo > -0.02`, `abs(e0) <= 0.01` (`:1151-1152`) | Inequalities yes; e0 input wrong (**M2**) |
| Style helps | F_person CI low > 0 | `ci[0] > 0` (`:1450`) | Yes |
| Critic 2 | MAE < nuisance-only, CI excludes 0, both fold sets | not evaluated | **No (m3c)** |
| Groups C1-R12 | CI entirely below 0, or R² trails by > 0.01 | `ci[1] < 0`, `(r2f - r2p) > 0.01` (`:583-584`) | Yes |
| B16 (i), (ii), (iii) | point < −0.02; same for critic 2; > 0.01 | `<`, `<`, `abs > ` (`:1790-1792`) | Yes |
| B21.4 adoption | CI low > 0; point ≥ 0; ≥ −0.01; ≥ −0.005 | `>`, `>=`, `>=`, `>=` (`:835-840`) | Yes (m14) |
| B21.5 trigger / triggered delivery | any ranking bar missed; CI low > −0.01 and Barcelona ≥ −0.01 | `not all_met`; `>`, `>=` (`:845, 940`) | Yes |
| B22.3 / B22.4 | CI low > 0 F_person AND point > 0 F_cell; CI low > 0 F_person | `>`, `>`; `>` (`:994-996`) | Yes |
| B23 ladder | rung 0 → 1 (only if the trigger fires) → selected level → L3 → fault (stop) | `remedy_ladder` (`:1158-1191`); R9 raises on `pending` / `implementation_fault` | Yes; rung-2 comparator wrong (**M3**) |

## Invariants: verified

- Critic 1 equals critic 3 at initialisation for s × {0, 1, 100}, to atol 1e-7 (test).
- W ≠ 0 at initialisation (test).
- Inherited weights are bit-identical after the fine-tune: they are frozen, not in the optimiser, and the test passes.
- Π never takes s or e: its signature is `(p, game, track)`, it has no parameters, and it is bit-identical after critic 1's fine-tune (test, and real-data R0 `pi_identical_after_critic1: True`).
- c = 0 gives critic 3 bit for bit (test).
- e is clipped in `predict` and in the bundle as γ·clip(e). γ is a buffer, saved and loaded, and applied at training and at query (test).
- Bundle round trip has abs diff 0.0 for all three kinds on real data.
- A published prior raises in `save_bundle`. R9 only ever builds the in-fold prior.
- `bars.json` is written in `Ctx.__init__` before any fit and made read-only (0444). On resume it is compared with BARS. Decision code reads BARS, which is asserted equal.
- **TD3 Q is critic 1 at every rung in the script.** The only way I found to break it is outside the script (m2).

## Tests to add (exact)

- **T1** `test_c2b_units.py::test_e0_bar_compares_like_with_like`
  - Setup: `f3` from the module fixture; a head with `l2.weight += .2` attached via `replace(f3, bcn=head)`; then `f1 = finetune(...)`.
  - Assert: `predict(f1, d, te, e_zero=True, nuisance=False)['setup']` equals the comparator the script uses for e0 exactly, on **all** rows including `spain`. This fails today.
- **T2** `test_c2b_pipeline.py::test_resume_refits_when_build_content_changes`
  - Run R1 with `--repeats 1`. Copy the build, multiply `cell_style` by 1.7 and reverse `track_features`, then resume with `--data` pointing at the copy.
  - Assert: every `nuisE_*` fit is recomputed (mtime changed), or the resume raises. This fails today.
- **T3** `test_c2b_units.py::test_td3_q_bundle_must_carry_a_style_path`
  - Assert `save_bundle(kind='critic1', twins=[f3])` raises.
  - Assert a critic 1 with `e_gate = 0` raises.
  - Assert `load(require_td3_q=True)` on a hand-edited `arch.n_e = 0` raises.
- **T4** `test_c2b_pipeline.py::test_rung2_comparator_uses_level_alpha`
  - In the forced-R5b run, assert that for each level L the comparator fit's info carries `alpha_m_s == L.alpha`.
  - Assert that `metrics_R5b` reads that per-level comparator.
- **T5** Replace the source greps (m13):
  - twins: run `c3_fit` / `run_R5` on the fixture with a spy on `T.boot_weights`, and assert the keys are `setup_hash[train]` for critic 3 and `cell_key[etr]` for critic 1;
  - clipped fraction: assert the stored fit json contains `clipped_frac_test`;
  - S22a: assert the R3d fit's `test_sha` equals the R3 fit's `test_sha` for each fold.
- **T6** `test_c2b_units.py::test_ridge_baselines_and_bcn_head_have_no_leak`: the pattern of `check_leak_head.py`. Assert unchanged under held-out ×1000:
  - `ridge_A` (with and without `delta_alpha`), `ridge_exact` and `nuisance_only` (style on and off) predictions on training rows;
  - the `fit_bcn_head` state;
  - the `c3_exact` nuisance;
  - λ from `with_lambda`.
- **T7** `test_c2b_pipeline.py::test_fallback_command_runs`: after `full_run`, run `--runs R1,R5,R6,R7 --rows S --force-fallback --resume`. Assert that `metrics/R5_S.json` has `pair_acc_S` and that `remedy_ladder` can read it.
- **T8** `test_c2b_units.py::test_meta_prior_no_m_for_absent_game_track`: drop all F1 25 rows of one track from the training rows. Assert `prior.m[F1 25, t]` is NaN and that `m_query(t, uniform)` equals the F1 26 m.
- **T9** `test_c2b_units.py::test_style_transform_rows_S_one_row_per_cell`: duplicate one cell's row 50 times in the training rows. Assert that the track median and scale do not change.
- **T10** `test_c2b_pipeline.py::test_bars_judged`: assert that the R3, R4 and R7 metrics contain booleans for the F_hash R² bar, the Barcelona ≥ 0.55 bar and the critic-2 MAE bar.

## Regressions against C2a

C2a M9 (NaN style → 0, B4), M10 (label stated as global, B1), M12 (`dataset.py` containers), I5 (folds keyed to the data hash) and the two pre-existing full-suite failures are all resolved. I did not re-audit the other C2a items (M1–M8, M11, M13–M16).

## Relevant files

- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/scripts/critic_c2.py`
- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/rlsetup/critic/train.py`
- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/rlsetup/critic/bundle.py`
- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/rlsetup/critic/meta_prior.py`
- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/rlsetup/critic/c2data.py`
- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/rlsetup/critic/style_transform.py`
- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/tests/test_c2b_units.py`
- `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/tests/test_c2b_pipeline.py`
- Scratch checks: `scratch/qa_c2b/check_leak_head.py`, `scratch/qa_c2b/check_resume.py`, `scratch/qa_c2b/check_real.py`, `scratch/qa_c2b/check_alpha.py`

## Bottom line

C2b implements B1–B23 faithfully, and the evaluation is honest where it matters most: no fitted object sees held-out rows, folds group persons across games, and runs are bit-reproducible. Before R2–R9:
- fix **M1**, so a resume can never mix builds;
- fix **M2**, so the e = 0 bar compares like with like;
- resolve **M3** with the architect, before any rung-2 decision is taken.

M3 only matters if rung 0 fails. M2 feeds the B16 trigger and every rung's bar.
