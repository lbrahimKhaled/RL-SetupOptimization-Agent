# C2b results: style-conditioned setup critics (B18)

Run `runs/critic_c2/20261116_081601Z` (gitignored). Data build `runs/critic_c2_data/20261115_232051Z`. 1 thread, 3 repeats.
Code: repo HEAD 5a74422 plus the uncommitted C2b working tree (fix round included). The run was done in two invocations, as the
rulings' "Approved to run" says: `R1,R2,R3,R3b,R3c,R3d` (0.56 h fit time), then `--resume … R4,R5,R5b,R6,R7,R8,R9` (0.18 h).
Total ≈ 0.74 h, against a 4 h budget. `timing.json` holds only the last invocation (0.18 h).

Conventions:
- Pair accuracy is scored on the setup part of the prediction (J + Barcelona head [+ λΠ]), i.e. −q, which is what TD3 ranks.
- Spearman is on the **full prediction (lap-time predictor), not the setup ranking**.
- R² uses the closed-form nuisance refitted on all training rows. It never enters q.
- CIs: 95 % cluster bootstrap by person (2,000 draws). MDE = 2.8 × SE.
- Bars are judged on the twin mean (pre-registered). Twin-0 blocks are in `metrics/R3.json`.
- Every table lists Barcelona first, then pooled, then per game.

## Bars: judged

| Bar (addendum) | Value | Result |
|---|---|---|
| R11 critic 3 pair acc ≥ 0.56, F_cell / F_person | 0.575 / 0.569 | pass |
| R11 pair acc CI low > 0.52 | 0.552 / 0.546 | pass |
| R11 critic 3 − ridge ≥ −0.02 | +0.009 [−0.013, 0.030] / +0.006 [−0.014, 0.024] | pass |
| R11 strict (F_cell r0) ≥ 0.54 | 0.577 | pass |
| R11 held-out-setup (F_hash) R² gain > 0 | 0.011 [0.007, 0.014] | pass |
| Barcelona pair acc ≥ 0.55 (delivered pooled + head, F_cell) | 0.584 | pass |
| Groups (C1-R12): keep PCA unless the pair CI is entirely < 0 or R² trails physics by > 0.01 | PCA − physics −0.007 [−0.026, 0.010] | PCA kept |
| Guard (C1 amendment item 4): no significant degradation, F_cell | R² guarded − unguarded **−0.0022 [−0.0037, −0.0008]**; pair +0.013 [−0.008, 0.035] | **fail: blocks combined readiness under the C1 rule** |
| Guard, F_person | R² −0.000 [−0.004, 0.003]; pair −0.013 [−0.029, 0.003] | pass |
| Critic 1 rung 0 (B9): ΔR²(c1 − c3) CI low > −0.02, F_cell / F_person | +0.018 [−0.003, 0.037] / +0.016 [−0.012, 0.043] | pass |
| Critic 1 at e = 0 vs critic 3 (same delivered head), within 0.01 | 0.000 / 0.000 | pass (0 by construction since fix M2) |
| B16 trigger (i) critic 1 ΔR² < −0.02, (ii) critic 2 < −0.02, (iii) e0 > 0.01 | +0.018/+0.016; −0.002/+0.000; 0/0 | not fired, no rows-S fallback |
| B23 ladder | rung 0 passes | TD3 Q = rung 0 (B9 exact); R5b not needed |
| Style helps (R6): F_person real − shuffle CI low > 0 | track +0.021 [−0.010, 0.050] (MDE 0.043); cross +0.015 [−0.021, 0.047] (MDE 0.049) | **fail: not shown** (null was expected) |
| Critic 2: MAE below nuisance-only, CI excluding 0, F_cell and F_person | +0.011 [0.000, 0.022] / +0.014 [−0.002, 0.028] (% MAE, critic 2 − nuisance) | **fail** (point above nuisance) |
| B21.4 meta-prior adoption | F_person CI low −0.053 (needs > 0); F_cell point −0.004 (needs ≥ 0) | not adopted |
| B21.5 trigger (any R11 ranking bar missed) | all met | not fired; R3c not needed; critic 3 delivered plain |
| B22.3 S22a (drop Quali) | `drop_quali` false | rows A keep Quali |
| B22.4 S22b (escalate the TT-reference decision) | false | no escalation |

Stalled fits: 1 R3 CV fit (`c3_F_cell_r1_f4_t0`). The final twins did not stall.

Critic 2 under ruling 29: in C2b it gets rungs 0–1 only. Rung 1 needs the B16 trigger, which did not fire. Critic 2 is therefore a **negative result**, and screening uses critic 1.

## §1 Barcelona

Critic 3 Barcelona pairs: 157 pairs, F_cell, twin mean, pair accuracy with CI and MDE.

| Variant | Pair acc | MDE |
|---|---|---|
| pooled | 0.569 [0.474, 0.658] | 0.133 |
| pooled + head (**delivered**) | 0.584 [0.492, 0.671] | 0.131 |
| head − pooled | +0.015 [−0.003, 0.035] | 0.027 |
| leave-Barcelona-out | 0.529 [0.430, 0.622] | 0.137 |
| pooled − leave-out | +0.040 [−0.028, 0.117] | 0.102 |
| Barcelona-only (r0) | 0.580 [0.473, 0.675] | 0.146 |
| pooled r0 − Barcelona-only r0 | −0.013 [−0.101, 0.076] | 0.125 |
| F_person: pooled / pooled + head | 0.550 / 0.556 (head − pooled +0.006 [−0.014, 0.024]) | 0.129 |
| ridge A (R1), F_cell / F_person | 0.571 / 0.590 | 0.126 / 0.134 |

- Leave-out is trained on **all** non-Barcelona rows A, about 25 % more off-track data than a pooled fold fit. Pooled − leave-out is therefore conservative.
- Delivery rule: the delivery is pooled only if the head is more than 0.03 below pooled. It is not, so `pooled_plus_head` is delivered.
- Head L2 chosen: 1e-4 ×31, 1e-3 ×6, 1e-2 ×23.

Critics on the exact Barcelona rows:
- 24 rows exist. 23 have style, and 1 is excluded from the style critics.
- Values are mean residual (%) and MAE (%).

| | F_cell residual | F_cell MAE | F_person residual | F_person MAE |
|---|---|---|---|---|
| critic 3 | −0.041 [−0.188, 0.111] | 0.293 | −0.128 [−0.295, 0.025] | 0.351 |
| critic 1 | −0.046 [−0.183, 0.093] | 0.271 | −0.119 [−0.272, 0.030] | 0.334 |
| critic 2 | −0.040 [−0.175, 0.092] | 0.274 | −0.139 [−0.293, 0.007] | 0.314 |
| nuisance only | – | 0.259 | – | 0.309 |

Critic 1 − critic 3 ΔR² on Barcelona exact rows:
- F_cell: +0.101 [0.009, 0.253]. F_person: +0.091 [−0.030, 0.236].
- n = 23 and the MDE is about 0.18. Descriptive only (see below).

Barcelona style reference (31 cells, centre = track): style-rl's OCD drivers, from R9.

| Driver | Mahalanobis to Barcelona cells | to training cells | s̃ dims clipped | e clipped (fold-twin) |
|---|---|---|---|---|
| attacker | 10.17 | 7.42 | 2 of 9 | none |
| smooth | 10.09 | 7.67 | 3 of 9 | 1 of 4 |
| neutral | 15.11 | 10.14 | 1 of 9 | 1 of 4 |

All three OCD drivers lie far outside the game cells. Mean clipped fraction of e on CV test rows: 0.015.

**What the 24 exact Barcelona rows can show:**
- the calibration of the pooled critics on Barcelona's fast TT laps (residuals above: all CIs include 0 on F_cell);
- the 31 Barcelona style cells as the reference for centring OCD/AC styles;
- descriptive cases.

**What they cannot show:**
- any style effect or style × setup effect on Barcelona. With n = 24, a correlation must exceed about 0.6 to be detected, and the rows contain no variation within a driver;
- any Barcelona-specific style training.

## §2 TD3 readiness: critic 3 on rows A (20,190 rows, 2,878 pairs, 260 persons)

| Subset | F_cell pair acc | F_person pair acc | F_cell R² gain | F_person R² gain | F_hash R² gain |
|---|---|---|---|---|---|
| Barcelona | 0.569 [0.474, 0.658] | 0.550 [0.461, 0.643] | 0.001 | 0.029 | −0.003 [−0.011, 0.005] |
| pooled | 0.575 [0.552, 0.600] | 0.569 [0.546, 0.593] | 0.010 | 0.036 | 0.011 [0.007, 0.014] |
| F1 25 | 0.570 | 0.566 | 0.010 | 0.043 | 0.012 |
| F1 26 | 0.585 | 0.573 | 0.009 | 0.030 | 0.010 |
| Race | 0.576 | 0.572 | 0.004 | 0.021 | 0.004 |
| Quali | 0.573 | 0.559 | 0.006 | 0.029 | 0.006 |
| TT | 0.577 | 0.576 | 0.024 | 0.071 | 0.028 |

- Ridge A (R1) pooled pair accuracy: 0.566 / 0.563.
- Spearman within (full prediction, not setup ranking): 0.73 on F_cell, 0.45 on F_person.
- Session facts (E22): the Quali − TT level shift is −1.2 % (median, same driver). Quali and Race lap noise is about 2.3× TT in SD.
- Meta prior (B21): the per-fold c selection was mostly 0, so the modal c is F_cell 0, F_person 0 and F_hash 0.03, and the final c is 0.
  - B21 − plain pair accuracy on A: F_cell −0.004 [−0.022, 0.011], F_person −0.025 [−0.053, −0.003].
  - On Barcelona: +0.029 / −0.029.
  - Not adopted.
  - Anomaly, not investigated because B21 was not adopted: the B21 − plain R² on F_person is +0.109 [0.090, 0.126], while its pair accuracy drops.
- In-fold m vs the published f1laps meta: mean distance 0.18 SD, 54 % of knobs equal. The published meta is a sensitivity analysis only, because it leaks (it is ranked on lap times that include held-out folds).
- S22a (no Quali in training) on TT/Race pairs: F_cell −0.003, F_person +0.013 [−0.004, 0.028].
- S22b, q_TT − plain on TT pairs: F_cell +0.015 [−0.013, 0.038], F_person +0.005 [−0.018, 0.033].

## §3 Style critics on exact rows (579 rows with style; F1 25 279, F1 26 300; all TT)

Critic 1 (B9) vs critic 3 on the same rows, ΔR²(c1 − c3):

| Subset | F_cell | F_person |
|---|---|---|
| Barcelona (23) | +0.101 [0.009, 0.253] | +0.091 [−0.030, 0.236] |
| pooled | +0.018 [−0.003, 0.037] (MDE 0.028) | +0.016 [−0.012, 0.043] (MDE 0.039) |
| F1 25 | −0.011 [−0.035, 0.011] | −0.025 [−0.066, 0.014] |
| F1 26 | +0.052 [0.023, 0.087] | +0.061 [0.027, 0.100] |
| assist-free F1 26 (125) | +0.062 [0.017, 0.140] | +0.058 [0.018, 0.118] |

Ablation (R6, B9 critic 1; real − shuffled s̃):

| Shuffle | F_cell | F_person |
|---|---|---|
| within track | +0.012 [−0.012, 0.033] | +0.021 [−0.010, 0.050] (MDE 0.043) |
| within game × track | +0.025 [−0.003, 0.049] | +0.015 [−0.021, 0.047] (MDE 0.049) |

**Style helps: not shown.** The F_person CI includes 0, and the effects up to about 0.04–0.05 R² are undetectable at this n.
- The F1 26 subsets point positive.
- F1 25 points slightly negative.

Critic 2, per game:

| | F_cell MAE c2 − nuisance (%) | F_person | ΔR² c2 − c2pre |
|---|---|---|---|
| Barcelona | +0.015 [−0.025, 0.052] | +0.005 [−0.037, 0.048] | +0.047 / +0.077 |
| pooled | +0.011 [0.000, 0.022] | +0.014 [−0.002, 0.028] | −0.002 / 0.000 |
| F1 25 | +0.014 [0.002, 0.027] | +0.016 [0.003, 0.030] | −0.006 / −0.006 |
| F1 26 | +0.008 [−0.010, 0.024] | +0.011 [−0.014, 0.039] | +0.002 / +0.007 |

Critic 2 real − shuffle: F_cell +0.001 [−0.016, 0.018], F_person −0.005 [−0.031, 0.019].

## §4 Groups, twins, guard, support, R8

- **Groups:** PCA (4 groups) is kept. The adjusted Rand index between fold groupings has mean 1.0 and min 1.0 (45 pairs).
  - R² gain on F_cell: PCA 0.011, physics 0.012.
  - R² gain on F_hash: 0.012 for both.
- **Twins:** bars use the twin mean.
  - Weight decay chosen: 0 ×38, 1e-5 ×6, 1e-4 ×26.
  - Critic 1 L2 chosen: 1e-4 ×31, 1e-3 ×17, 1e-2 ×9, 0.1 ×3.
  - Median ‖W1e‖ is 0.80.
- **Guard:** on F_cell the guarded critic 3 has a significantly lower held-out R² than the unguarded one: −0.0022 [−0.0037, −0.0008], on 20,183 on-support rows.
  - Under C1 amendment item 4 this blocks combined readiness.
  - Pair accuracy is not degraded: +0.013 [−0.008, 0.035].
  - F_person passes.
  - Critic 1 guarded − unguarded R²: F_cell −0.009 [−0.023, 0.004], F_person −0.001 [−0.011, 0.010].
- **R8:** with Imola and Las Vegas dropped, on the same rows:
  - critic 3 pooled pair accuracy is 0.569 vs 0.575;
  - Barcelona is 0.554 vs 0.548;
  - critic 1 ΔR² is −0.004 [−0.030, 0.018];
  - R9 `r8_disagreement` (per-track value, see `metrics/R9.json`): critic 3 mean 0.90 (min 0.86, Barcelona 0.91); critic 1 mean 0.90 (Barcelona 0.91).

## Delivered bundles (R9)

- **TD3 Q:** `runs/critic_c2/20261116_081601Z/bundles/critic1_final.pt`.
  - kind critic1, role `td3_q`, rung `rung0_B9_exact`, e_gate 1.0 on both twins.
  - Status: "style path present; effect not detectable on game data; to be fine-tuned on OCD/AC laps with controlled styles".
- Critic 3 (style-free reference, never TD3's Q): `bundles/critic3_final.pt`, plain (no meta prior), Barcelona `pooled_plus_head`.
- Critic 2: `bundles/critic2_final.pt`. It failed its bar and is not used for screening.

## §5 Caveats

- Provisional knob constants (C2a game → knob maps).
- Placeholder racing line and track width.
- Provisional z for tracks flagged `z_provisional`.
- Style is taken from the labelled lap.
- Style data is TT-only: every exact row is TT.
- The label is computed globally, per (game, track, session), robust two-pass (B1).
- R² uses the closed-form nuisance refit on training rows (it never enters q). The rung-2 matched comparator was not exercised because rung 0 passed.
- Leave-Barcelona-out uses about 25 % more off-track data than a pooled fold fit, so pooled − leave-out is conservative.
- R6 is the ablation of the B9 critic 1 only. Its L2 is reused from the real fit (anti-conservative for "style helps"; QA m8, deferred). The result is null anyway.
- Barcelona n = 23 exact rows: every Barcelona style number is descriptive.
- QA minors deferred in the fix round (m2, m5–m13, i1):
  - m7 (meta prior fitted with inner-validation rows) could bias c upward, but c = 0 was delivered.
  - m9 (head not in critic 1's fine-tune loss) affects only the 23 Barcelona exact rows.
  - m5 (rows S style transform) is irrelevant, because rows S were not used.
- Report-only items have not been computed: q_min pair accuracy, same-c published vs in-fold, same-month pairs, and the R3c Δ table. They need the deferred report script.
- **Open for Ibrahim / architect:** the F_cell guard result blocks combined readiness under the C1 rule. Whether that blocks TD3 use of the critic-1 bundle is not the coder's call.

## Post-hoc ensemble check (16 Nov, lead; NOT pre-registered, no retraining)
Script: `scripts/critic_c2_ensemble_check.py` (reads the saved per-fold test predictions of `runs/critic_c2/20261116_081601Z`; each model's setup score z-scored per fold over its test rows, i.e. without labels). Pair accuracy on rows A, 3 repeats, person-cluster bootstrap; critic 3 here = R3 twin mean without the Barcelona head.

| Model (setup score) | F_cell | F_person (unseen drivers) | Barcelona F_cell / F_person |
|---|---|---|---|
| critic 3 (grouped heads) | 0.575 | 0.566 | 0.563 / 0.539 |
| ridge (B15) | 0.566 | 0.563 | — |
| critic 2 pretrained on A (full MLP, no style) | **0.582** | **0.581** | — |
| 50/50 critic 3 + ridge | 0.575 (Δ −0.000 [−0.011, +0.010]) | 0.577 (Δ +0.010 [+0.001, +0.020]) | 0.582 / 0.622 |
| equal 3-way critic 3 + ridge + critic-2-pretrain | 0.581 (Δ +0.006 [−0.008, +0.020]) | **0.581 (Δ +0.015 [+0.002, +0.028])** | 0.599 / 0.620 |

Reading: (1) the full-interaction network ranks setups better than the grouped heads (grouping costs ≈ 0.007–0.015 pair accuracy); (2) an equal-weight ensemble improves ranking for unseen drivers with a CI excluding 0. Caveats: four variants were looked at on the test folds (mild selection), weights were not tuned (equal / 50-50), Barcelona numbers have MDE ≈ 0.13. Not yet a delivered bundle: an ensemble TD3 Q would be mean(critic 1, critic 2 fine-tuned with style, ridge) — each differentiable in p and keeping the style path; it needs a bundle class and a pre-registered confirmation run.

### Ensemble TD3 Q built (16 Nov, lead)
`rlsetup/critic/ensemble.py` (`EnsembleCritic`), manifest `runs/critic_c2/20261116_081601Z/bundles/ensemble_td3_q.json` = equal-weight mean of `critic1_final.pt` (grouped, style) and `critic2_final.pt` (full MLP, style). Both members keep the style path; critic 2's inherited weights are frozen, so at e = 0 it equals the pretrained network measured above (0.582 / 0.581). Offline, critic 3 + critic-2-pretrain 50/50: F_cell 0.581 (Δ +0.006 [−0.005, +0.017]), F_person **0.582 (Δ +0.016 [+0.005, +0.028])**, Barcelona 0.582 / 0.620. Sanity on Barcelona with style-rl's three OCD drivers: gradients finite, corr(ensemble q, critic-1 q) 0.94–0.95, style changes q (attacker vs smooth mean |Δq| 0.39, rank corr 0.92), q_min ≤ q, style-free members refused. Not pre-registered and not run through QA; `critic1_final.pt` stays the TD3 Q of record until Ibrahim chooses.

### Further tweaks tried (16 Nov, lead; none adopted)
- Pairwise ranking net trained on within-driver pairs (RankNet, date as training-only term): 0.548 / 0.539 (too few pairs).
- Critic-2 seed ensembles (5 seeds) and widths 32/64/128: alone 0.582–0.588; with critic 3 within ±0.004 of the current ensemble (all CIs include 0).
- Plateau ≈ 0.58. Diagnostic: "later setup is faster" alone scores 0.651 within driver (driver learning, not setup). The ensemble scores 0.61–0.63 on pairs > 1 % apart and on same-day pairs, 0.53 on pairs < 0.25 % apart. Date is not added to q: TD3 cannot change it, so it would grade setups by when they were made.
