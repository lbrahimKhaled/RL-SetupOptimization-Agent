# C1 critic warm start: architecture review (9 Nov 2026)

**Verdict: CHANGES-REQUIRED.** One blocker (the §3 constants); the rest are concrete fixes. §1 is untouched: Decision 8 stands, only the §3 constants change. Evidence tags: [F] formulation, [C] code read, [D] measured read-only on the CSV, [J] judgement. No literature is relied on; two facts are listed under "Needs research".

## Evidence
- Row counts [D]: 2,565 → 2,487 after `gap > 8 %` → 2,007 on 17 tracks, as §2 expects. `gap` percentiles: p95 6.3 %, p99 20 %, max 109 %; fastest −6.7 %.
- **§3 as written fails its own stop rule [D]: 1,142 of 2,007 rows (56.9 %) leave the box, all on χ (range 0.09–0.83; bound 0.35–0.80). On the sf70h_f1 geometry: 1,223 rows (60.9 %; 1,221 on χ, 2 on K_φ).** Rake is never truly out of bounds; 2 rows are rake = 0 with floating-point error of about −7e-18.
- Formula [C]: K_b = χK_φ − ½K_w t² matches `translator.py:69-70`, and the inverse is correct. Knob names and `low`/`high` match `car_setup.py`; the `ocd_first_lap-f1` worktree has the same 15 bounds. All other game ranges fit their OCD bounds.
- Geometry [C]: m_c 171/199 and t 1.6/1.5 are not the car in use. `CarGeometry` gives sf70h_f1 151.1/185.4 kg, t 1.6/1.6 m (AV-21 169.5/200.5, 1.6/1.5). The car is decided in DECISIONS #21.
- Corrected map (R1) [D]: 1,955 rows kept; 710 unique 15-knob setups (one appears 257 times, 364 singletons); 807 unique (track, setup); 186 drivers, 114 with ≥ 3 rows.
- Ranking evidence [D]: **194 same-driver/track/session groups, 492 laps, 390 pairs, 34 drivers** (§0's 238/593 predate track and map filtering).
- Ridge baseline, folds by (driver, track) [D]: pair accuracy **0.572, 95 % CI [0.507, 0.610]** (driver-cluster bootstrap, SE 0.026). Removing training rows that share a setup with the held-out group: 0.567. Copying leaks little.
- Nuisance shrinkage [D]: driver ridge α ≤ 1 gives pair accuracy 0.57 and nuisance-only R² 0.636; α = 100 gives 0.515 and 0.428. Over-shrunk driver offsets push driver skill into the setup heads. Held-out-setup R² gain over nuisance: +0.024.
- Identifiability [D]: driver–setup graph has 38 components, the largest covering 1,787 of 1,955 rows; 63 drivers (113 rows) have a single distinct setup and contribute nothing to setup effects beyond their offset.
- PCA [D]: on deduplicated data, eigenvalues 6.19 / 1.73 / 1.20 against thresholds 1.28 / 1.22 / 1.18, so k = 3.
- Support [C, D]: `sha_f1_best` lies outside the data range on 5 knobs (aero front share, camber F/R, toe F/R). Data camber is −3.5…−2.5 against OCD 0; toe is never 0 in the data.
- Toe in OCD [C]: positive = toe-out, confirmed in `vehicle_dynamics_equations_impl.hpp:45-48` (right-side sign flipped).

## [ARCH] items settled
| Item | Decision |
|---|---|
| Toe sign | Keep §3 (front +, rear −), marked `assumed` in provenance. **The data cannot verify it**: any fit is invariant to flipping a narrow-support input. Safe handling: R9 keeps the toe gradient from steering the actor off-support, and the first OCD batch must include toe on both sides of 0 (R11). |
| Smoothing | Resample the closed centerline to 1 m; moving average of x and y, **40 m window, applied twice**. |
| Corner rule | Corner = contiguous run with radius < 500 m (Table VIII straight limit). **Merge two runs only if same curvature sign and gap < 40 m**; drop runs shorter than 15 m. Sanity against corner counts from memory: mean error 2.5 corners, flat across the grid tried (window 20–80 m, gap 0–60 m); treat `n_corners` as coarse. |
| Head size | **One hidden layer, width 16, SiLU, for every head** (805 parameters for the physics groups). Small final-layer init (std 1e-3); `b` starts at the training mean. |
| Regularisation | Heads: weight-decay grid **{1e-3, 3e-3, 1e-2}**, chosen once on the inner validation set with physics + `joint`, then frozen for all runs. Nuisance: ridge **α = 1** (`wd = 2/n_train`, about 1.25e-3), same for the nuisance-only baseline. |
| p0 in critic state | **No.** §4 gives p0 to the actor only [F]; the game data has no p0. Revisit only if OCD fine-tune shows J depends on the distance from the base driver's setup. |
| Final bar | R13. |

## Binding changes
**R1 (blocking). Replace the §3 constants.** Take m_c and t from `CarGeometry` of the fine-tune car. Check bounds with a 1e-9 tolerance so rake = 0 rows survive (`Setup` raises on a strict violation). f_F unchanged; **f_R = 2.5 + 1.5 (c − 1)/10 Hz; K_b,F = 60,000 + 24,000 (c − 1); K_b,R = 21,000 + 3,900 (c − 1) N·m/rad.** Result [D]: 52 rows dropped (2.6 %, all χ), K_φ 187–639 kN·m/rad, median χ 0.607 (OCD default 0.613), K_b ≥ 0 by construction. These are fitted to the OCD box, not measured from the game; say so in every result. If the Assetto Corsa report gives better maps, swap the constants and keep the 5 % rule.

**R2. Feasibility export.** The input box contains K_b < 0 points, which the translator silently clamps (`translator.py:86-89`). `knobs.py` must expose differentiable K_b,F and K_b,R from p, plus the training support box per knob, so the later actor can respect Decision 8.

**R3. Define the setup hash:** the 15 mapped knobs rounded to 1e-8, track excluded. Use it for held-out-setup folds, inner validation and twin bootstraps.

**R4. Folds.** Pool rare drivers inside each fold; unseen driver or team maps to `rare`. Median-based `gap` may stay global: pair differences within a track do not depend on the median, and held-out R² is affected negligibly. State the > 8 % filter as a selection on the label. Also report the stricter ranking variant (training rows sharing a held-out setup removed).

**R5. Nuisance fitting.** Initialise from a closed-form ridge on each training fold, then train jointly with the heads. Centre the heads after training (mean Σh = 0 on training rows; `b` absorbs the rest). Report the `separate` mode as biased toward zero setup gradient, not as a competitor.

**R6. Training defaults.** Adam lr 1e-3, batch 256, up to 500 epochs, patience 30.

**R7. Twins.** Bootstrap by setup hash, not rows, with each twin owning its nuisance. Correct §5: `q_min = −max(Ĵ1, Ĵ2)`. SiLU, twin, no target networks are right for γ = 0: there is no bootstrapped target [F]; the minimum acts only as ensemble pessimism [J]. Later use is the TD3 design's call.

**R8. Output contract.** Add a frozen `scale` (= 1) next to `b`. The OCD target later is `y = 100 (J / J_ref − 1)`, J_ref = J at a reference setup on the same track, J from the evaluation-lap cost with no shaping terms (DECISIONS #13, #26) [F].

**R9. Off-support guard (main protection for the fine-tune).** SiLU extrapolates linearly, and the OCD operating point is off-support on 5 knobs. Each step, take 256 training rows and replace each knob with U[−1, 1] with probability 0.3; keep those outside the per-knob training [min, max] on at least one knob. Add `0.1 · mean((Σ_g h_g − 1.0)²)` (1.0 % gap, about the observed mean within-driver change). Acceptance metrics use on-support rows only and must stay within noise of the unguarded joint fit.

**R10. `lsd_lock_coast`.** No OCD effect [§3 caveat]: pin it to the data median during OCD fine-tune so a game-only gradient cannot move it.

**R11. Fine-tune contract (record in the C1 results; later spec owns it).** First OCD batch is space-filling over the feasible box, includes p_ref and toe on both signs. Fit `b` and `scale` first, then unfreeze heads. Measure warm vs cold runs-to-converge (PROJECT_CONTEXT §8); fall back to a cold critic if warm is worse.

**R12. Replace the PCA rule.** The drafted rule is not sound: it demands PCA be proven better than random, which inverts Decision 4, and a 95th percentile from 20 partitions is close to the sample maximum. It is not powered: pair-accuracy SE is 0.026 from 34 drivers, so differences below about 0.05 cannot be resolved, and the earlier random spread (SD 0.006) was well below that. **New rule, fixed now: keep PCA unless, in `joint` on identical folds, the paired driver-cluster bootstrap 95 % CI of (PCA − physics) pair accuracy lies entirely below 0, or held-out-setup R² gain trails physics by more than 0.01.** Random partitions and the one-group control are reported, not used in the decision. Report fold-to-fold PCA stability (adjusted Rand index), fit final groups on all rows, and state plainly when groups are indistinguishable.

**R13. Final warm-start bar (all, `joint`, chosen groups):** pair accuracy ≥ 0.55; driver-cluster 95 % CI lower bound > 0.50; ≥ ridge − 0.02 on the same pairs; held-out-setup R² gain > 0; strict variant (R4) ≥ 0.52. Held-out track is reported only (about 23 pairs per track). If missed, report it; do not tune to pass.

**R14. Compute budget.** Single-net (no twins) for selection; twins only for the final fit. Leave-one-track-out only for the chosen groups and ridge. Random partitions: `joint`, 1 repeat. `backfit`: 1 repeat. Single thread as specified.

**R15. Track features.** Compute the shares from Table VIII radius bands (< 50, 50–150, 150–500, > 500 m), by length. Medium share stays out (shares sum to 1). Clip standardised z to ±3.

**R16. Results caveats to add:** the recalibrated map (R1); 390 pairs / 34 drivers; off-support behaviour (R9).

## Risks to watch
| Risk | Metric |
|---|---|
| Actor exploits extrapolation | Fraction of proposals outside the support box; Ĵ vs realised J there |
| Scale mismatch game vs OCD | Fitted `scale`; RMSE on first OCD batch before and after refit |
| Warm start slower than cold | Runs-to-converge, warm vs cold |
| Driver skill leaking into heads | Pair accuracy vs nuisance α; correlation of head output with driver offset |
| z overfitting on 17 tracks | Held-out-track pair accuracy vs in-track |
| Unstable PCA groups | Adjusted Rand index across folds |

## Needs research
1. F1 2020 setup-menu labels for front and rear toe: toe-out or toe-in?
2. Any published physical values for F1 2020 spring and anti-roll-bar clicks (Assetto Corsa report may cover part of this).

## Ibrahim decides
1. PROJECT_CONTEXT §11.1: the 15 inputs match neither the 26- nor the 16-knob list (16 has dampers, no diff). For the OCD fine-tune, which fixed values hold the 11 dropped knobs (the sf70h_f1 defaults or `sha_f1_best`)?
2. §11.7: accept Table VIII thresholds as PLACEHOLDER for C1?
3. R1 changes rear spring and bar scales to fit the OCD box. Accept, given 57 % of rows are otherwise lost?

## Ibrahim's answers (9 Nov 2026)
1. Fixed values for the 11 dropped knobs: decide when the OCD fine-tune stage is specified. Not needed for C1.
2. Table VIII radius bands: accepted as PLACEHOLDER for C1; §11.7 stays open.
3. R1 recalibration: accepted. Row loss reproduced independently: 1,142 of 2,007 with the §3 constants, 52 with R1.

Assetto Corsa (`docs/research/ac_spring_frequency.md`) has no equivalent to the frequency mediator and gives no better click maps, so the R1 constants stand.
