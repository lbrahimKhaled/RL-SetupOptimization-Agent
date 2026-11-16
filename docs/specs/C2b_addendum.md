# C2b addendum: style-conditioned setup critic (critics, training, evaluation)

rl-architect, 16 Nov 2026. This addendum is binding for C2b. Where it differs from `C2_style_critic.md` §1–§11 or from `C2_review.md` R1–R13, this addendum wins. §12 of the spec (Ibrahim's decisions of 16 Nov) wins over everything.

Evidence tags:
- [D] measured read-only on `setup_data/` and `runs/critic_c2_data/20261115_223137Z/`;
- [C] read in the code;
- [F] from the formulation or a decision;
- [J] my judgement.

No literature is relied on. Measurement scripts are in `scratch/c2b/`:
- `build_raw.py`: unfiltered frame, in memory, written to scratch only;
- `a2.py`–`a4.py`: label and filters;
- `a5.py`–`a9.py`: exact rows;
- `b1.py`–`b4.py`: Barcelona and transfer;
- `users.py`: user overlap between the games, counts only.

`docs/research/c2_real_lap_times.md` does not exist yet, so B1 defines the rule to apply when it arrives.

## Verdict: READY-FOR-CODER

Nothing blocks C2b. Three facts shape the design, and the report must state them plainly.

1. **On exact rows, nothing predicts lap time beyond the nuisance for an unseen driver.** That covers setup, style, and the two together. [D]
   - Setup effects learned on rows A explain none of the differences between drivers among top TT laps: R² gain −0.008, CI [−0.022, +0.004].
   - Exact rows have one row per cell, so they cannot separate a setup effect from a driver effect.
   - Under §12.13, the style critics are therefore judged mainly on **safety**: critic 1 must be non-inferior to critic 3. "Style helps" is a claim with a stated minimum detectable effect (MDE), and a null result is the expected outcome.
2. **Session moves the label by about 2.7 %** (Race +1.40 %, Quali −1.35 % against the per-(game, track) median). The gap median and the lap filters move to (game, track, session). [D]
3. **Track identity accounts for 63 % of the variance of `S_corner_peak_angle`** and 28–38 % of three other style numbers. Style is centred per track before scaling. [D]

## Evidence

### E1. Label and lap filters [D]
Medians are taken over 48 (game, track) groups, with Mexico dropped.

| Session | Median gap vs (game, track) median | Range over tracks |
|---|---|---|
| Race | +1.40 % | +0.49 … +2.02 |
| Quali | −1.35 % | −2.38 … −0.82 |
| TT | −0.18 % | −1.03 … +0.74 |

- The old fast tail (586 rows below −4 %, 544 of them Quali) is mostly this session offset. Under per-(game, track, session) medians, only 255 rows fall below −4 %: Race 124, Quali 130.
- **2,037 rows are faster than f1laps' fastest top-10 TT lap on the same (game, track): 1,839 Quali, 181 Race, 17 TT.** 401 Quali rows are faster by more than 2 %. The TT leaderboard is therefore not a floor for Quali or Race laps. I do not know the game-side reason (Needs Ibrahim 3).
- There are 138 (game, track, session) groups, with 39 to 600+ rows each (median 162). Robust spread (1.4826·MAD) is 0.81–2.77 % (median 1.45 %).
- Robust z has a thin lower tail and a heavy upper tail:
  - z < −3.5: 67 rows, against ≈5 expected under a normal distribution;
  - z > +3.5: 245 rows, 182 of them TT casual laps.
- Applying the rule in B1 to the 20,706 rows that pass the box (no gap filter, no Mexico):
  - kept: **20,192**;
  - dropped low: 71 (Race 50, Quali 21);
  - dropped high: 443 (TT 271, Race 118, Quali 54);
  - exact rows kept: **586 / 586**; Barcelona: 977 rows, 24 of them exact; rows S: 1,634.
- QA M10: leaving one fold of 5 out shifts a group median by 0.04 % (median), 0.19 % (p95) and 0.37 % (max). The shift is a per-group constant: pairs within a group cancel it, and the in-fold (game, track, session) fixed effect absorbs it.

### E2. Exact rows [D]
- **586 rows after the Mexico drop** (F1 25: 280, F1 26: 306), all TT. **579 have style.** `S_corner_peak_angle` is missing on 45 of them: 38 on Imola/Las Vegas and 7 rows with no style at all.
- 239 (game, user) driver ids (counted before the Mexico drop); 137 appear on a single track.
- 24 telemetry users (and 129 setup users) appear in **both games** under different ids. Held-out-driver folds keyed by (game, user) would leak these people across folds.
- Within-(game, track) SD of the gap: 0.55 % (variance 0.307 %²).
- F1 26 rows with ABS off and TC off: 126 of 306.
- Size of the shuffle strata on 586 rows:

| Strata | Number of strata | Rows in singleton strata (shuffle = identity) |
|---|---|---|
| track × controller × all 6 assists | 300 | **185** |
| track × controller × ABS × TC | 224 | 93 |
| track × controller | 135 | 14 |

- Share of style variance explained by **track**: `S_corner_peak_angle` 0.63, `S_mean_abs_rate` 0.38, `S_reversals` 0.28, `B_event_peak` 0.27, `B_mean_active` 0.12, the rest ≤ 0.07. Explained by **game**: ≤ 0.01 for every number.
- Setup does not predict style: R² of p beyond the nuisance is ≤ 0.013 for all 9 numbers (held-out driver). So "style is an outcome of the setup" is not visible in the data.

### E3. Power at n = 586: ridge proxies on exact rows [D]
R² gain over the nuisance-only model, as a fraction of the within-track variance. CIs use a driver-cluster bootstrap. Ridge α applies to standardised blocks.

| Model | Unseen driver (person folds) | Seen driver (cell folds + driver offset) |
|---|---|---|
| nuisance-only MSE (%²) | 0.253 | 0.198 |
| + p, α = 10 | −0.006 | −0.003 |
| + s, α = 10 | −0.071 | +0.003 |
| + p + s + s×PC4(p), α = 1 / 10 / 100 | −0.185 / −0.089 / −0.012 | −0.024 / +0.012 / +0.025 |
| A-learned setup part (critic-3 proxy) | −0.008 [−0.022, +0.004] | −0.003 [−0.015, +0.006] |
| **Interaction only, real − stratified shuffle** (s main effect kept real; α = 10; track × controller × ABS × TC) | −0.010 [−0.037, +0.017]; SE 0.013 → **MDE ≈ 0.04** | +0.011 [−0.003, +0.027]; SE 0.008 → **MDE ≈ 0.02** |
| Same, cross-track shuffle on track-centred s | −0.001 [−0.036, +0.034] | +0.026 [+0.008, +0.044] |

Reading [J]:
- With weak shrinkage, style **hurts** predictions for unseen drivers. That repeats the review's S-row finding.
- The only positive signal is for seen drivers. That is the size of effect that s×p terms can produce by memorising driver-specific patterns.
- On exact rows the style comes from the labelled lap itself, so the s main effect partly measures how that lap went.
- Consequences:
  - the s main effect goes into the nuisance (B7);
  - the style claim is tested on the interaction only, with unseen-driver folds as the primary test (B14).

### E4. Barcelona (`spain`) [D]
- 977 rows A, 513 cells, **157 within cell × session pairs from 50 drivers**:
  - by game: F1 25 96, F1 26 61;
  - by session: Race 69, Quali 63, TT 25.
- Rows S: 85 rows, 29 pairs, 7 drivers.
- Exact rows: 24 (12 + 12), from 24 drivers; 23 of them also have exact rows on other tracks. Gap SD 0.43 % / 0.39 %. 31 cells have style.
- Ridge pair accuracy on the Barcelona pairs (cell folds, nuisance + driver offset + p, plus p×z when pooled):

| Training set | Barcelona pairs | All-track pairs (2,878 / 277 drivers) |
|---|---|---|
| Barcelona only | 0.580 [0.471, 0.670] | 0.568 |
| Pooled, Barcelona in | 0.573 [0.480, 0.673] | 0.581 [0.551, 0.611] |
| Leave Barcelona out | 0.592 [0.504, 0.682] | 0.583 |
| **Pooled + Barcelona deviation δ (α_δ = 3 / 10 / 30)** | **0.637 / 0.631 / 0.611** [CI ≈ ±0.09] | 0.582 |
| Pooled, Imola and Las Vegas dropped | 0.554 [0.457, 0.653] | 0.583 |

Paired differences:
- pooled − Barcelona-only: −0.006 [−0.097, +0.091];
- **pooled + δ(10) − pooled: +0.057 [−0.008, +0.131]** (SE 0.036);
- pooled − leave-out: −0.019 [−0.072, +0.024].

What this shows:
- Setup effects are largely shared across tracks: a Barcelona-only model scores 0.568 on all tracks.
- Barcelona alone supports a critic 3 (977 rows) but not a sharp test: the pair-accuracy SE is ≈ 0.05 and the MDE for a model difference is ≈ 0.10–0.13.
- **What the 24 exact Barcelona rows can show:**
  - the calibration of the pooled critics on Barcelona's fast TT laps (mean residual, SE ≈ 0.08 %);
  - the 31 Barcelona style cells as the reference population for centring OCD/AC driver styles (B4);
  - descriptive case rows.
- **What they cannot show:**
  - any style effect or style × setup effect on Barcelona. With n = 24, a correlation must exceed about 0.6 to be detected, and the rows contain no variation within a driver;
  - any Barcelona-specific style training.

### E5. Missing z [D]
- The failing-candidate z of Las Vegas standardises to longest straight = +3.39 SD (clipped to 3). Imola sits inside the range (+1.24).
- These two tracks hold 1,332 rows A and 38 exact rows.
- Dropping them moves Barcelona ridge accuracy by −0.02, which is within noise.

## What §12 changes in R1–R13

| Item | Status |
|---|---|
| R1 | Unchanged. The constants stay **provisional** until Ibrahim reads the menu values (§12.16). |
| R2 | Mexico dropped (§12.12). For **z only**, the gate is relaxed (B3). The gate for style geometry numbers is unchanged. |
| R3 | Unchanged. |
| R4 | Per-track centring added (B4). IQR/1.349 confirmed (QA I3). NaN handling fixed (B4). |
| R5 | Assists are nuisance only (§12.14, option a; my option b is withdrawn). The strata are made concrete in B14. The assist-free F1 26 report stays (126 exact rows). |
| R6 | Applies to rows A only. Exact rows have no pairs. |
| R7 | Style main effect added to the nuisance (B7). Folds use a cross-game person id (B6). |
| R8 | Unchanged. |
| R9 | Shared folds over A ∪ exact rows, keyed to the data hash (B6). Rows S enter only through the fallback (B16). |
| R10 | Unchanged, plus track centring at query (B17). |
| R11 | Critic-3 bars unchanged, on A. The bars for critic 1, critic 2 and style move from S to **exact-row prediction** (Bars). |
| R12 | Superseded. "TT-only S" belongs to the fallback; exact rows are primary for every style critic. |
| R13 | Counts updated (E1, E2). |
| Review "Needs Ibrahim" | 1–4 and 6 are answered by §12. Item 5 (radius bands) is still open (§11.7), so the bands stay PLACEHOLDER. |

## Binding changes

**B1. Label and lap filter** (replaces S2 steps 2–3; applied at build).
- Groups are (game, track, session).
- Computation:
  - m = median lap, d = 1.4826·median|100(lap/m − 1)|, and robust z = gap/d;
  - drop |z| > 3.5;
  - recompute m and d once on the kept rows, then drop |z| > 3.5 again;
  - final gap = 100(lap/m − 1).
- State it in the results as a selection on the label, computed globally (QA M10, E1 numbers).
- Every count is logged per step. Expected: 20,192 rows A, 586 exact.
- **Real-world check**, applied when `c2_real_lap_times.md` arrives:
  - per track, T_ref = real pole for the game's layout (season-matched);
  - report r = fastest f1laps TT top-10 lap / T_ref per (game, track), and flag tracks with r outside [0.97, 1.05] as possible layout or label mismatches for review, not for automatic dropping;
  - hard cut only for laps < 0.92·T_ref [J; expected count ≈ 0]. This cut is counted, and it raises an error if it removes more than 1 % of any group.

**B2. Track list.** `drop_tracks = ('mexico',)` in the config, applied at load before anything else. 24 tracks remain.

**B3. z for Imola and Las Vegas.**
- z does not use the alignment, only the layout. For z, the bar is therefore the length test plus median |corr| ≥ 0.80.
- Imola (0.840, dL −0.9999 %) and Las Vegas (0.817, −0.98 %) pass this bar and use the candidate z, flagged `z_provisional=true`.
- `TrackScaler` fits on gated tracks only and clips at ±3.
- Sensitivity run: both tracks dropped.

**B4. Style input** (transform only; the numbers stay style-rl's, per Decision 2).
- Input: the 9 B/T/S numbers.
- Step 1: log of the 6 rates.
- Step 2: **track centring**: subtract the median of that track's training cells, both games pooled. If the track has fewer than 5 training cells, use the global training median and flag it.
- Step 3: robust z with IQR/1.349, clipped to ±3.
- Step 4: NaN → 0, the training median (QA M9). There is **no** missingness flag input: a flag would be a per-track constant on 2 tracks.
- Exact rows without any style (7) are excluded from the style critics and counted. They stay in A.

**B5. Rows.**
- Critic 3: rows A (20,192).
- Critics 1 and 2: exact rows with style (579).
- Rows S: only through B16.

**B6. Folds** (QA I5). Add a cross-game integer `person` id in the loader, built before names are discarded. It is never written next to anything that could identify a user. Three fold sets:
- F_cell: 5 folds by sha256(data_hash‖game‖driver‖track) mod 5;
- F_person: 5 folds by sha256(data_hash‖person) mod 5;
- F_hash: 5 folds by setup hash, A only, used for critic 3's held-out-setup R².

Rules:
- data_hash = sha256 over the sorted input-file sha256 values in `summary.json`.
- F_cell and F_person are each repeated 3 times (salts 0, 1, 2); F_hash runs once.
- The same assignment covers A and exact rows. In each fold, every critic-3 pretrain excludes all held-out rows.
- `folds.json` stores the data hash. Any run asserts that it matches.

**B7. Nuisance** (training only, never in q). Additive ridge terms trained jointly; penalty α/n_train on the mean loss.
- (game, track, session) fixed effect, α = 1e-2.
- Driver offset by (game, user), α = 1. Drivers with fewer than 2 training rows are pooled into "rare" per fold.
- controller × game, each assist × game, team × game: α = 1.
- Date: within-driver trend (months from the driver's mean date) per game, plus game × calendar-month effects, α = 1.
- **Critics 1 and 2 only:** a linear style main effect m(s̃), 9 → 1, α = 10.
- Nuisance-only model = the same terms without p, and without style in the heads.

**B8. Critic 3.**
- Ĵ3 = scale·(b + Σ_g h_g(p_g, ẑ, ĝ)), with h_g = Linear(|g| + 5 + 2 → 16) → SiLU → Linear(16 → 1).
- `scale` = 1 and `b` are frozen for the OCD fine-tune (C1-R8).
- Training (C1-R6):
  - Adam, lr 1e-3, batch 256, ≤ 500 epochs, patience 30;
  - inner validation on 10 % of training cells;
  - weight-decay grid {0, 1e-5, 1e-4}, selected on the inner validation;
  - CPU, deterministic algorithms.

**B9. Critic 1 (TD3's Q).**
- Ĵ1 = scale·(b + Σ_g h_g), with h_g = w2_g·SiLU(W1p_g p_g + W1z_g ẑ + W1g_g ĝ + **W1e_g e** + b1_g) + b2_g, and e = W s̃, where W is 2 × 9 with no bias, shared by all heads.
- Initialisation:
  - copy the critic-3 weights from the same fold, repeat and twin;
  - W1e_g = 0, so critic 1 equals critic 3 at start;
  - **W = the top-2 PCA loadings of the training cells' s̃**. A zero W together with zero W1e_g would give zero gradient to both.
- Fine-tune on the exact training rows:
  - trainable: W, W1e_g and the exact-row nuisance (incl. m);
  - **everything inherited from critic 3 is frozen.** 579 rows cannot re-learn setup effects, and freezing protects the A ranking;
  - Adam, lr 3e-3, full batch, ≤ 2,000 steps;
  - inner validation on 20 % of training persons, patience 100 steps;
  - L2 on (W, W1e) chosen from {1e-4, 1e-3, 1e-2, 1e-1} on the inner validation.
- e is clipped to the training [min, max] per dimension in evaluation and at query (R10).

**B10. Critic 2.**
- MLP(p 15, e₂ 2, ẑ 5, ĝ 2 → 32 → 32 → 1), SiLU, with its own encoder W₂.
- Pre-train on A with the e₂ columns absent, using B8's settings.
- Then the same fine-tune scheme as B9: W₂ initialised by PCA, first-layer e₂ columns zero-initialised; trainable are W₂, the e₂ columns and the nuisance.
- The A-pretrained critic 2 against critic 3 on A is the "does grouping cost accuracy" test.

**B11. Barcelona residual head** (§12.20).
- r_BCN(p) = Linear(15 → 16) → SiLU → Linear(16 → 1), last layer zero-initialised.
- Trained on the Barcelona training rows of A with critic 3 frozen. L2 from {1e-4, 1e-3, 1e-2} on an inner split of 20 % of Barcelona training cells.
- Added to Ĵ1, Ĵ2 and Ĵ3 iff the context track is `spain`. Each twin has its own head.
- Training set is fixed now: **pooled + Barcelona head**. Pooled-only and Barcelona-only are reported for comparison.

**B12. Twins and guards.**
- Twins:
  - critic 3: 2 bootstrap members by setup hash, each with its own nuisance;
  - critic 1: twin k starts from critic-3 twin k and is bootstrapped by cell on the exact rows;
  - q_min = −max(Ĵa, Ĵb).
- p guard: the paired guard of C1 amendment item 1, unchanged (256 rows, probability 0.3, U[−1, 1], outside support, weight 0.1, target 1.0 %), on critic 3 and on critic 1's fine-tune. For critic 1, s is sampled from training cells.
- Guarded and unguarded fits are compared per C1 amendment item 4, on F_cell repeat 0 and F_person repeat 0.

**B13. Groups.**
- The C1-R12 rule decides, on critic 3, rows A, F_cell repeat 0 and F_hash: keep PCA (k = 4, varimax, in-fold, deduplicated) unless the paired CI of (PCA − physics) pair accuracy lies entirely below 0, or the held-out-setup R² gain trails physics by more than 0.01.
- Critics 1 and 3 use the same groups.
- Adjusted Rand index across folds is reported.

**B14. Metrics.**
- **R² gain** = (SSE_nuis − SSE_model)/SST_w, where SST_w = Σ_test (y − ȳ_(game, track, session), train)².
- MAE in % and in s (% × group median lap / 100).
- Within-(game, track) Spearman: mean weighted by n.
- **Pairs** (A): within (game, driver, track, session), different hash and different gap. A tie in prediction counts 0.5.
- Strict variant: C1-R4.
- All CIs: person-cluster bootstrap, B = 2,000, paired where two models are compared. Each metric is averaged over the 3 repeats before its CI.
- **Style ablation:**
  - retrain the fine-tune with the s fed to the **heads** shuffled within strata, with m(s) kept on the real s; 5 shuffle seeds, loss averaged;
  - strata: game × track × controller × ABS × TC × braking-assist (on/off) for F1 26; game × track × controller for F1 25;
  - report the rows in singleton strata;
  - secondary: the cross-track shuffle on centred s, within game × controller × ABS × TC (+ braking);
  - "s = training median" is the critic-3 + nuisance comparator.
- The R8 disagreement between g = F1 25 and g = F1 26 is reported as Spearman of Ĵ over 2,000 support-box setups.

**B15. Baselines.**
- Nuisance-only.
- Ridge on A: nuisance + driver + p + p×ẑ, α = 1.
- Ridge on exact rows: nuisance + m(s) + p + s × PC4(p), α = 10, PCs fitted in-fold.
- The Barcelona ridge variants of E4.

**B16. Fallback to rows S: fixed trigger.**
- The fallback fires iff any of these holds:
  - (i) critic 1's ΔR²(c1 − c3) point estimate < −0.02 on F_person or on F_cell;
  - (ii) the same test fails for the fine-tuned critic 2 against its own A-pretrained version;
  - (iii) critic 1 at e = 0 moves pair accuracy on A by more than 0.01 against critic 3.
- What does **not** trigger it:
  - a null style effect;
  - critic 2 missing its bar against nuisance-only;
  - a missed Barcelona bar.
- When it fires:
  - rerun B9/B10 on rows S (1,634 rows; style = the cell's TT style; session in the nuisance; critic 1's S pair accuracy added), with the same bars;
  - report both runs;
  - TD3's Q = the critic 1 that passes non-inferiority, exact rows preferred. If neither passes, TD3's Q = critic 3, and critic 1 is reported as a negative result.

**B17. Interface** (`rlsetup/critic/bundle.py`; pure torch, CPU, no pandas, differentiable in p):
```
b = CriticBundle.load(path)              # kind in {'critic1','critic2','critic3'}
ctx = b.context(track=None, geometry=None, style=None, source=None)
#  track: canonical key ('spain'); geometry: (N,2) centre line -> z via saved TrackScaler (unknown track)
#  style: the 12 raw style-rl numbers (None -> training median, e = 0); track-centred with the saved
#  per-track medians, global median + flag if the track is unknown; e clipped to training range
#  source: None -> uniform 1/n_sources (R8)
b.J(p, ctx) -> % gap (lower better); b.q(p, ctx) = -J; b.q_min(p, ctx) = -max over twins
b.knob_feasibility(p) -> min K_b [N m/rad]; b.support_box(); b.off_support(p) -> bool
b.style_report(style) -> {clipped per e-dim, mahalanobis, centre: 'track'|'global'}
b.groups, b.style_scaler, b.track_scaler, b.provenance   # run_provenance, data sha256, style-rl commit, geometry sha256, folds hash
```
- p is (…, 15), normalised, in `KNOB_ORDER`.
- The Assetto Corsa adapter implements only `SourceAdapter` (to_knobs, style_inputs, meta, vocabularies, steer_kappa_sign) and a map from its track ids to canonical keys. It never imports model code.

**B18. Outputs.**
- Run directory: `runs/critic_c2/<UTC>/`, containing `folds.json`, `bars.json` (constants from Bars, written before training), `metrics.json`, per-fold predictions `.npz`, and `bundles/{critic1,critic2,critic3}_final.pt` with a `bundle.json` each.
- Report: `docs/results/critic_c2.md`, sections in this order:
  - §1 **Barcelona**:
    - critic 3 pooled vs pooled + head vs leave-out vs Barcelona-only (157 pairs / 50 drivers, CI, MDE);
    - critics 1/2/3 residuals on the 24 exact rows;
    - the 31-cell Barcelona style reference, and where style-rl's OCD drivers fall in it (clipped fraction, Mahalanobis);
    - the "can / cannot show" statement of E4.
  - §2 TD3 readiness: critic 3 on A, the R11 bars.
  - §3 Style critics on exact rows: critic 1 vs critic 3, the ablation, critic 2, per game, assist-free F1 26.
  - §4 Groups, twins, guard, support, R8.
  - §5 Caveats:
    - provisional knob constants;
    - placeholder line and width;
    - provisional z;
    - style taken from the labelled lap;
    - TT-only style;
    - label computed globally.
- Every table: Barcelona row first, then pooled, then per game.

**B19. Tests C2b must ship.**
- Folds:
  - deterministic from the data hash, and a mismatch raises;
  - F_cell puts each exact row in its cell's fold;
  - F_person groups a person across games.
- Leakage: StyleTransform (incl. track medians), TrackScaler, PCA groups, the W initialisation, support box and nuisance do not change when held-out rows are multiplied by 1,000.
- B1 on a fixture: exact counts, two passes, Mexico absent.
- `q` does not change with the nuisance descriptors or with m(s).
- Critic 1 at initialisation equals critic 3 for any s (to 1e-7); W ≠ 0; the inherited weights are bit-identical after the fine-tune.
- ∂q/∂p is finite and non-zero; ∂q/∂s is finite; inputs stay finite with NaN style (M9).
- e is clipped and the clipped fraction is logged.
- The shuffle permutes only within strata; singleton strata are unchanged; it is seed-deterministic.
- Twins bootstrap by hash on A and by cell on exact rows; q_min = −max.
- Pairs never cross sessions; a tie counts 0.5.
- The Barcelona head is zero at initialisation and active only for `spain`.
- Bundle save → load gives identical q. A dummy adapter with an unseen source gives uniform g; an unseen track uses global centring and sets the flag.
- `bars.json` is written before training and is read-only to the report.
- Two seeded runs give identical metrics.
- No username in any output (extend T8).

**B20. C2a items to correct for C2b** (item 6 of the task).
1. The gap median per (game, track) mixes sessions, and the filter is one-sided. Replace with B1.
2. Mexico is still in `dataset.npz` (25 tracks). Apply B2.
3. My own R2 extension gated z by alignment, which was too strict for z. Apply B3.
4. There is no cross-game person id, so held-out-driver folds leak 24 people. Apply B6.
5. Style is not centred per track. Apply B4.
6. The fold key is missing (I5). Apply B6.
7. Accepted as built: IQR/1.349 (I3), the centre-line placeholder (I4), the Azerbaijan F1 25 per-game corr of 0.846 (I2), which is reported only.

## Runs and budget
Per-fit times are [J] until the smoke run measures them.

| Run | Content | Fits |
|---|---|---|
| R0 smoke | F_cell fold 0: one critic-3 twin (20 epochs), critic 1 (100 steps), critic 2, ridge; measures s/fit | ≤ 10 min |
| R1 | Ridge and nuisance baselines on all fold sets; Barcelona ridge variants | ≤ 10 min |
| R2 | Groups: PCA vs physics, critic 3, F_cell repeat 0 + F_hash, 1 twin | 20 |
| R3 | Critic 3 with the chosen groups: F_cell ×3 ×5 ×2, F_person ×3 ×5 ×2, F_hash ×5 ×2; unguarded on repeat 0 (10); strict variant (5) | ≈ 85 |
| R4 | Barcelona heads on every R3 fit; leave-Barcelona-out critic 3 (2); Barcelona-only critic 3 (5, report) | 60 small + 7 |
| R5 | Critic 1 fine-tunes: F_cell and F_person ×3 ×2 twins, 4 L2 values | 60 × 4 small |
| R6 | Critic 1 shuffle ablation, twin 0, 5 seeds × 2 schemes | 300 small |
| R7 | Critic 2: pretrain 30, fine-tune 30, shuffles 5 × 30 | 30 + 180 small |
| R8 | Sensitivity: Imola and Las Vegas dropped (critic 3 F_cell repeat 0 + critic 1) | 5 + 5 small |
| R9 | Final fits on all data → bundles | 6 |

- Heavy fits: about 130 at 30–60 s each, so 1–2 h. Small fits: about 800 at ~3 s each, about 40 min.
- **Budget ≤ 4 h wall, hard stop 6 h.** If R0 projects more than 6 h, cut the repeats from 3 to 1 (and say so). Never cut epochs or bars.
- Order: R0 → R1 → R2 → R3 … R9. R5 onwards waits on R3 for each fold.

## Bars (fixed now; written to `bars.json`; if missed, report, do not tune)
- **Critic 3, A, pooled (TD3 readiness, R11 unchanged):**
  - pair accuracy ≥ 0.56;
  - CI lower bound > 0.52;
  - ≥ ridge − 0.02 on the same pairs (ridge today ≈ 0.581–0.590);
  - strict variant ≥ 0.54;
  - held-out-setup (F_hash) R² gain > 0.
- **Barcelona:**
  - the delivered model is pooled + head, unless the head's Barcelona pair accuracy is more than 0.03 below pooled (point estimate); then pooled is delivered;
  - Barcelona pair accuracy ≥ 0.55 (point estimate);
  - CI and MDE (≈ 0.13) reported.
- **Critic 1 (exact rows):**
  - paired CI lower bound of ΔR²(c1 − c3) > −0.02 on both F_person and F_cell;
  - at e = 0, pair accuracy on A within 0.01 of critic 3.
- **Style helps:**
  - claimed only if the CI of ΔR² (real − stratified shuffle, heads only) on **F_person** lies above 0;
  - F_cell is reported as seen-driver evidence only;
  - every report states MDE ≈ 0.04 (F_person) and ≈ 0.02 (F_cell), and that a null result is expected.
- **Critic 2:**
  - MAE below nuisance-only, with the paired CI excluding 0, on F_cell and F_person. A miss on F_person is expected (E3); state that up front.
  - Grouping cost is reported only.
- **Groups:** C1-R12, on A.
- **Fallback trigger:** B16.

## Needs Ibrahim
1. **Does "really bad" mean only "style damages the critic" (B16), or also "no detectable style effect"?**
   - My recommendation: damage only.
   - Neither exact rows nor rows S can detect a realistic style × setup effect. Exact rows have no variation within a driver; S has an MDE of about 0.10 pair accuracy.
   - The test that can identify style × setup is OCD/AC: controlled-style drivers running many setups.
2. **Barcelona reading (§12.20):** confirm that style critics train on all tracks, Barcelona gets the pooled critic + Barcelona residual head, and every result is reported Barcelona first.
3. **Game question:** 1,839 Quali setup laps are faster than f1laps' fastest top-10 TT lap on the same track, by more than 2 % for 401 of them. Do online or career quali laps get tow, ERS or track evolution that TT lacks, or is the recorded "lap time" something else?
   - The answer decides whether Quali rows belong in A at all.
   - B1's per-session medians make the critic robust either way.

## Risks

| Risk | Metric that reveals it |
|---|---|
| Style acts as a skill or lap-quality proxy (style taken from the labelled lap) | m(s) coefficient size; F_cell vs F_person style gap; shuffle gain on F_cell but not F_person |
| Style acts as a device/assist proxy | gap between stratified and unstratified shuffle; corr(e, controller/assist codes); share of rows in singleton strata |
| Critic 1 drifts from critic 3 | pair accuracy on A at e = 0 (must be identical with frozen weights); ‖W1e‖ |
| Barcelona head overfits 977 rows | head vs pooled on Barcelona pairs; ‖r_BCN‖ on support vs off support |
| OCD/AC styles are off-support | fraction of e clipped; Mahalanobis distance to the 31 Barcelona cells |
| **OCD's Barcelona layout ≠ game layout** (TUM Catalunya failed the gate, 0.798) | the first OCD/AC Barcelona run must use the gated bacinger line or the recorded edges (§12.18); compare L_geo with the game's 4,661 m |
| Provisional z (Imola, Las Vegas) and knob constants | B3 sensitivity; first-OCD-batch RMSE before and after refitting `scale` |
| Date drift leaks into the heads | pair accuracy on same-month pairs vs all pairs |
| Label shift from global medians | reported max shift 0.37 %; pairs are unaffected |
| Fold leakage across games | person-fold assertion test; count of persons in both games (24 telemetry, 129 setup users) |


## Addendum items B21–B23 (rl-architect, 16 Nov 2026)

These items are binding in the same way as B1–B20. They are implemented after the coder's R0/R1. B23 carries out spec §12.21 (Ibrahim's 16 Nov decision that TD3's Q must always see the driver style), and it overrides the last clause of B16.

**Sources.**
- Data: measured read-only on `runs/critic_c2_data/20261115_232051Z/` (the coder's B1 build: 20,190 rows A, 2,878 within-session pairs).
- Meta files: `meta_setups_wide.csv` for both games, mapped through `game_to_knobs` + `normalise`.
- Scripts: `scratch/meta/arch/`:
  - `a1`–`a3`, `a7`–`a9`: what the meta is, and the leak;
  - `a5`, `a6`, `a10`: how important each knob is;
  - `a4`, `a11`: the ridge proxies;
  - `b1`–`b3`: sessions.

**About the proxy.** Wherever a model gain is quoted, it comes from the B15 ridge on A (nuisance + p + p×ẑ, α = 1). The ridge is fitted on the real fold sets (`make_folds` with this build's data hash), and every CI is a paired person-cluster bootstrap with B = 2,000. The ridge is linear in p, so it gains more from a quadratic prior than critic 3's SiLU heads will. Treat each proxy gain as an upper bound for critic 3.

**Verdict.**
- **B21: APPROVE-WITH-NOTES, as a pre-registered, gated prior.** The evidence predicts no gain for unseen drivers. It is specified exactly so that it can be run, judged and switched off.
- **B22: APPROVE.** Keep Quali at full weight, and run two sensitivity runs with fixed rules.
- **B23: CHANGES-REQUIRED to B16.** A style-preserving remedy ladder replaces the "critic 3 as TD3's Q" clause.

---

### E21. The meta setups [D]

**1. Every (game, track) in rows A has a published meta.** That is 23 tracks per game, covering all 20,190 rows.

**2. Four of the 15 knobs have a meta that is a menu extreme, the same on every track.**
- Camber is −3.5 / −2.0 everywhere, in both games.
- Toe is 0.00 / 0.10 everywhere (0.02 / 0.11 on a few F1 25 tracks).
- 59–92 % of rows already sit exactly at these values.
- F1 26 brake bias is 56–57 % on every track.
- So the knobs the meta looks "important" on are global, monotone main effects. They are not track-specific.

**3. The meta is close to the community's own in-fold median.** Distances are mean |·| / SD per knob, measured on the F_cell repeat 0, fold 0 training rows:

| Centre | Mean distance to published meta | Knob values exactly equal |
|---|---|---|
| All training rows, median | 0.236 | 51 % |
| Fastest 20 % by gap, median | 0.213 | 54 % |
| Fastest 20 % by nuisance residual, median | 0.208 | 55 % |
| Random row (reference) | 0.740 | 0 % |

- The fast-set median is closer to the meta than the all-rows median in 63–70 % of the 46 groups (Wilcoxon p = 0.005–0.04).
- The most recent 20 % of setups are *farther* from the meta (0.29), so the tilt is towards fast setups, not recent ones.
- Reading [J]: the published meta is most likely an aggregate of the same community setups, weighted by lap time. If so, a fixed external prior leaks held-out labels. How f1laps builds the meta needs checking (Needs web-researcher 1).

**4. The 21 rows that copy the meta exactly are not fast.**
- Mean gap +0.32 %, against +0.01 % for all rows.
- 20 of them are F1 25; 9 of those 20 fall in the last 25 days of the F1 25 data.

**5. A date effect is mixed into the lead's 0.533, but the 0.533 itself is not a date effect.**
- Within a driver, the later setup is faster in **0.663** of 2,524 pairs.
- The closer-to-meta setup is the later one in 0.493 of pairs, so the overall 0.533 is not produced by date.
- The split by date order is still large: closer-to-meta wins 0.702 when it is the later setup and 0.375 when it is the earlier one.
- After removing the nuisance (incl. the within-driver trend) on training rows, these knobs stand out:
  - camber F/R: 0.59;
  - toe F/R: 0.58–0.59;
  - all others: 0.46–0.56; rake 0.46–0.50.

**6. Per-track importance is not established.**
- A naive χ² test rejects equal per-track win rates for 10 of 15 knobs.
- But person-cluster CIs are about 1.8× wider than binomial ones (design effect ≈ 3.2). The pairs are clustered, so the χ² test is anti-conservative.
- The between-track SD of the win rate is at most 0.03–0.11. This is an upper bound.
- Only toe F/R (and, barely, front height and LSD drive) have pooled person-CIs that exclude 0.5.

**7. The ridge already beats the "closer to meta" rule on exactly the knobs the meta looks good on.**

| Pairs that differ in | Meta rule | Base ridge |
|---|---|---|
| toe F | 0.625 | 0.655 |
| toe R | 0.639 | 0.649 |
| camber F | 0.550 | 0.591 |
| camber R | 0.552 | 0.603 |
| all pairs (L2 distance rule) | 0.533 | 0.566 |

**8. Most of the ridge gain comes from curvature, not from where the meta sits.** Free per-knob quadratic features (p_k − c_{t,k})² added to the ridge, F_cell, 3 repeats:
- centred at the published meta: +0.010 [−0.003, +0.023];
- centred at the in-fold all-rows median: +0.008;
- published meta − in-fold median: **+0.0016 [−0.005, +0.009]**;
- on F_person: +0.003 / +0.001.

**9. The B21 loss form below, in the ridge proxy.** In-fold m, cross-fitted w, κ = 300, λ chosen on inner-validation MSE, repeat 0:

| Fold set | In-fold meta | Published meta |
|---|---|---|
| F_cell | +0.006 [+0.002, +0.010] | +0.007 [+0.001, +0.013] |
| F_person | **+0.001 [−0.010, +0.013]** | +0.002 [−0.002, +0.007] |

- Barcelona moves by −0.013 to +0.006 (n.s.).
- A prior as wide as the data's own setup effects (λ·SD(Π) ≈ 0.9·SD of the setup part) costs about −0.03 pooled and **−0.09 to −0.13 on Barcelona**.
- Inner validation picked λ = 0 or the smallest non-zero value in all 20 folds.

**10. Measured in-fold weights** (F_cell repeat 0, fold 0, mean over tracks):
- camber F/R: 0.17 / 0.19;
- toe F/R: 0.16 / 0.18;
- f_F: 0.10;
- LSD coast: 0.09; LSD drive: 0.08;
- everything else: ≤ 0.04; rake: 0.

**Conclusion [J].**
- The meta adds almost nothing beyond (a) monotone camber/toe effects the data term already learns and (b) generic curvature that critic 3's heads can already represent.
- Expected B21 effect on critic 3 for unseen drivers: about 0.
- B21 is still specified because Ibrahim asked for it as the response to "the critic is unsuccessful in grading". It is gated so that it cannot add bias unless the data select it.

### B21. Meta-setup prior

**B21.1 Which meta (leak).**
- **Primary: in-fold meta m^in.**
  - Per (game, track), m^in is the knob-wise median of the training rows whose nuisance-only residual is in the lowest 20 % of that (game, track).
  - The residual comes from the B15 nuisance-only fit on that fold's training rows.
  - If a (game, track) has fewer than 20 training rows, use the same game's pooled fast-set median and flag it. The smallest fast set measured is 28 rows, so this is not expected to happen.
  - The final fit (R9) uses all rows.
- **Published meta: sensitivity only.**
  - Mapped through `game_to_knobs` + `normalise`, using the 15 `GAME_COLUMNS` only.
  - Brake pressure, tyre pressures and engine braking have no OCD knob. Engine braking is out of scope (PROJECT_CONTEXT §3).
  - It is never delivered.
  - The report states the leak: the published meta is an aggregate of the community's setups and probably their lap times, held-out folds included.
  - The report also gives the closeness numbers of E21.3.

**B21.2 Importance per (track, knob).** All of it is in-fold and cross-fitted, so the fast rows that define m do not also score their own pairs.
1. Split the training cells in two by sha256(data_hash‖'meta'‖cell) mod 2.
2. Compute m from one half. Score the within-(game, driver, track, session) pairs of the other half on nuisance residuals. Then swap the halves.
3. For each knob k, a pair counts if the two setups' distances to m on k differ by more than 1e-9. A "win" means the setup closer to m has the lower residual.
4. Pool both games per track and both halves. This gives wins W_{t,k} and counts N_{t,k}.
5. Pooled rate: r̄_k = Σ_t W / Σ_t N.
6. Shrunk rate: r̂_{t,k} = (W_{t,k} + κ·r̄_k) / (N_{t,k} + κ), with **κ = 300 pairs** [J]. This is the design effect ≈ 3.2 times p(1−p)/τ² with τ ≤ 0.05, so a typical track keeps about 15 % of its own rate.
7. Weight: **w_{t,k} = max(0, 2·r̂_{t,k} − 1)**.
8. An unseen track uses w = max(0, 2·r̄_k − 1).

**B21.3 The loss term: option (a), an additive prior with a fixed weight.**

```
Ĵ3(p, c) = scale·(b + Σ_g h_g(p_g, ẑ, ĝ)) + λ·Π(p; t, ĝ)
Π = Σ_k w_{t,k} · (p_k − m_{t,k})²
```

- p is box-normalised. m_{t,·} is the ĝ-weighted mean of the games' m. A training row has a one-hot ĝ, so it uses its own game's m. A query with source = None (R8) uses the uniform average.
- Π has no trainable parameters, and λ is fixed during a fit. The heads train with Π present, so where the data contradict the prior, the heads cancel it, and the prior binds mainly where data are sparse.
- **Fixed grid:** λ = c·SD_train(setup part of the λ = 0 fit) / SD_train(Π), with **c ∈ {0, 0.03, 0.1, 0.3}**.
- c is selected on the B8 inner validation by MSE, with the weight decay already chosen for c = 0. A tie goes to the smaller c.
- c = 0 is exactly critic 3.
- The prior sits outside `scale`. `scale`, `b`, λ, m and w are all frozen for the OCD fine-tune (C1-R8).
- Π is quadratic, so ∂Π/∂p = 2λw(p − m) is smooth for TD3.
- Both twins share λ, m and w per fold; these are computed from the fold's training rows, not per bootstrap.

**Rejected options** [J]:
- (b) Pairwise hinge: with an uncalibrated margin it pushes Ĵ without bound in the meta direction, and with a calibrated target it flattens knobs where r̂ ≈ 0.5.
- (c) Gradient regulariser: needs second-order gradients and acts directly on the quantity TD3 follows, which gives the strongest bias exactly where it matters most.
- (d) Pseudo-labelled meta rows: they need an invented label.

Option (a) is explicit and removable: it is one term with one scalar.

**B21.4 How it is judged.**
- Critic 3 with B21 against plain critic 3, on identical folds, seeds and inner splits.
- Metrics: paired Δ pair accuracy on A (F_cell, F_person) and on the Barcelona pairs; Δ held-out-setup R² gain on F_hash; selected c per fold.
- Also reported: the per-knob w table (Barcelona row first), and the in-fold vs published comparison.

**Adoption rule (fixed now).** The delivered critic 3 includes B21 iff all of these hold:
1. F_person paired Δ CI lower bound > 0;
2. F_cell Δ point estimate ≥ 0;
3. Barcelona pair accuracy no more than 0.01 below plain (point estimate);
4. F_hash R² gain no more than 0.005 below plain.

Otherwise B21 is reported only. Expected from E21.9: not adopted.

**B21.5 Trigger ("unsuccessful in grading").** If plain critic 3 misses **any R11 ranking bar on A**, B21 becomes mandatory. The ranking bars are:
- pair accuracy < 0.56;
- CI lower bound ≤ 0.52;
- below ridge − 0.02;
- strict variant < 0.54.

The F_hash R² bar and the Barcelona bar are reported but do not trigger B21, because the Barcelona MDE is ≈ 0.13.

When B21 is triggered:
- run R3c (below);
- deliver B21 critic 3 iff it is non-inferior: F_person Δ CI lower bound > −0.01, and Barcelona no more than 0.01 below plain;
- report the R11 bars for both versions.
  - If only B21 passes, state "passes with the in-fold meta prior".
  - If neither passes, state "critic 3 does not grade setups at the R11 level on game data", and the lead flags TD3 readiness to Ibrahim.
- No further tuning in either case. B21 counts as pre-registered, not as tuning after a miss.

**Relation to B16/B23.**
- B21 decides the delivered critic 3 first. The B16/B23 bars for critics 1 and 2 are then measured against that delivered critic 3.
- Critics 1 and 2 inherit Π with λ, m and w frozen.
- The two triggers are independent:
  - "fails to grade" (R11) → B21;
  - "style damages the critic" (B16 conditions) → B23.
  - A null style effect triggers neither.

**B21.6 Π stays out of critic 1's style path.**
- No style-dependent w(s), and Π is never multiplied by e. Reasons:
  - the meta is a population aggregate with no style information;
  - per-track w is not even established from 2,865 pairs (E21.6), so w(s) from 579 single-row cells, or 29 Barcelona S-pairs, is not estimable;
  - it would be a second, unablated route for style into q, which would confound the B14 shuffle test;
  - it would break the B9 invariant that critic 1 equals critic 3 at initialisation.
- Π belongs to critic 3's setup part only.

**B21.7 Interface (B17).**
- The bundle stores: λ, the selected c, m per (game, track) in p, w per (track, knob), and r̄.
- `ctx` gets `meta_prior ∈ {'track','pooled_w','none'}`. An unseen track has no m, so Π = 0 and the flag is set.
- `b.J(p, ctx, prior=True)`.
- On OCD/AC, m lives in OCD knob space through the provisional map constants (§12.16). This is listed as a caveat.

### E22. Sessions [D]

**Rows and pairs per session:**

| Session | Rows | Pairs | Drivers |
|---|---|---|---|
| Race | 7,514 | 1,359 | 158 |
| Quali | 6,636 | 973 | 128 |
| TT | 6,040 | 546 | 134 |

Barcelona pairs: Race 69, Quali 63, TT 25 (E4).

**Level shift.** Comparing the best lap per session for the same (game, driver, track):
- Quali − TT: median −1.20 % (IQR −2.49 … −0.24); Quali is faster in 81 % of 1,495 cells.
- Race − TT: +1.65 %.
- Race − Quali: +2.75 %.

So for the same driver, Quali is a level shift of about −1.2 %. The B1 per-session medians and the (game, track, session) fixed effect absorb it.

**Noise within a session.** Within-(cell, session) SD of the gap:
- TT 0.39 %;
- Quali 0.89 %;
- Race 0.89 %.

Quali and Race carry about 2.3× TT's lap-to-lap noise in SD (about 5× in variance).

**Pair accuracy is still similar across sessions.** Base ridge, F_cell:
- TT 0.573 [0.527, 0.622];
- Quali 0.572 [0.536, 0.615];
- Race 0.559 [0.520, 0.600].

**Training-set variants, ridge proxy.** Δ against training on all rows:

| Variant | F_cell, all pairs | F_cell, other subsets | F_person |
|---|---|---|---|
| Drop Quali (TT+Race only) | −0.012 [−0.022, −0.003] | Race −0.015 [−0.027, −0.003]; TT +0.002 | −0.008 [−0.023, +0.005] |
| TT only | −0.013 | TT +0.023 (n.s.); Barcelona −0.049 | Barcelona −0.070 |
| Quali weight 0.5 | −0.003 | — | −0.001 |
| Quali and Race weights 0.2 (inverse variance) | −0.005 | TT +0.035 [+0.001, +0.066]; Race −0.011 | −0.003; TT +0.027 (n.s.) |
| p × {Race, Quali} deviation, α = 10 (session-specific prediction) | +0.015 [+0.002, +0.027] | TT +0.037 [+0.003, +0.067] | +0.006 [−0.013, +0.024]; TT +0.004 [−0.031, +0.036] |
| Same, TT-referenced q on all pairs | −0.007 | Race −0.016 | −0.015; Race −0.029 [−0.048, −0.009] |

**Which game effects are a level shift and which add noise.** The mechanisms are Ibrahim's forward and unverified; the split is my judgement [J] based on the measurements above.

| Effect (Quali vs TT) | Level shift (absorbed by the FE / per-session B1) | Noise within a session (pairs do not cancel it) | Can it bias setup effects? |
|---|---|---|---|
| Looser track limits | yes | partly, through lap-to-lap cutting; a driver's habitual cutting cancels in within-driver pairs | small |
| Manual ERS / overtake | yes | partly: strategy varies lap to lap | small |
| Slipstream / tow from traffic | partly | **yes**: random per lap | **yes**: the gain from a tow depends on drag, so low-wing setups are favoured. This is a candidate for the session-specific F_cell gain. |
| Track evolution | yes, within a session | **yes**: different setups usually come from different sessions | no |
| Tyre preparation | small | small | no |
| Race: fuel, tyre wear, traffic | yes | yes | yes (wear) |

### B22. Session handling

1. **Keep Quali and Race in rows A at weight 1.** Use the B1 per-(game, track, session) label and within-session pairs only (R6). B5, B7 and R3 do not change.
   - Down-weighting is rejected: it does not help unseen drivers, and it costs Race/Quali pairs.
   - Dropping Quali is rejected: it costs −0.012 on F_cell, with the CI excluding 0.
2. **Report:** every critic-3 table adds per-session pair accuracy (TT, Quali, Race), after the per-game rows (B18). The report states the level shift and the 2.3× noise ratio of E22.
3. **S22a, TT+Race sensitivity.** Critic 3 is trained without Quali rows and evaluated on all pairs, on F_cell repeat 0 and F_person repeat 0, with 1 twin.
   - Fixed rule: Quali is dropped from rows A for all delivered critics iff the paired Δ (TT+Race-only − all) on TT+Race pairs has CI lower bound > 0 on F_person **and** point > 0 on F_cell.
   - Otherwise nothing changes. Expected: no change, from the proxy (−0.014 / −0.010).
4. **S22b, TT-referenced session deviation.** Add training-only nuisance terms p × 1[Race] and p × 1[Quali] (linear, 15 + 15, α = 10). TT is the reference level, so the heads represent the TT setup effect, which is the solo-hotlap condition closest to OCD/AC.
   - Same folds as S22a.
   - Report the TT-pair accuracy of q, and the all-pair accuracy of the session-specific prediction.
   - Fixed rule: if the Δ on TT pairs (q_TT − plain q) has CI lower bound > 0 on F_person, the lead takes "switch q to the TT reference" to Ibrahim (Needs Ibrahim 3). Otherwise it is reported only.
   - Expected: no switch (F_person TT +0.004 [−0.031, +0.036]; MDE on TT pairs ≈ 0.05).
5. **Not in scope:** a TT-only critic. It costs Barcelona 0.05–0.07 in the proxy, and Barcelona has only 25 TT pairs.

### B23. Style-preserving remedy ladder (§12.21; replaces B16 "If neither passes, TD3's Q = critic 3")

**TD3's Q is critic 1 in every case.** Critic 3 is still delivered alongside it, as the pooled style-free critic. Every rung keeps the safety bars of the "Critic 1" bars:
- paired CI lower bound of ΔR²(c1 − c3) > −0.02 on F_person **and** F_cell;
- at e = 0, pair accuracy on A within 0.01 of critic 3 (the delivered critic 3, B21).

Rungs are tried in order, and the first rung that passes is delivered.

- **Rung 0:** critic 1 as in B9, on exact rows.
- **Rung 1:** the B16 fallback. Same trigger (i)–(iii), the rows S rerun as in B16, same bars.
- **Rung 2: shrunk style path, on exact rows.**
  - The style input becomes e′ = γ·clip(e). γ is a fixed constant per level, applied at training and at query, and stored in the bundle.
  - e stays an input, W1e stays trainable, and the B9 initialisation is unchanged (critic 1 = critic 3 at initialisation).
  - Three fixed shrinkage levels:

    | Level | γ | L2 on (W, W1e) | α of m(s) |
    |---|---|---|---|
    | L1 | 0.3 | 1 | 100 |
    | L2 | 0.1 | 10 | 300 |
    | L3 | 0.03 | 100 | 1,000 |

  - m(s) is shrunk too, because E3 showed the style main effect alone can fail unseen drivers (−0.071).
  - Choose the level on B9's inner validation (20 % of persons) by MSE, among {B9 setting, L1, L2, L3}, then test the bars on the outer folds.
  - If the chosen level fails, deliver L3 iff L3 passes. L3 is always fitted.
- **Rung 3, status:** for rungs 1–2, or any rung whose style-helps test is null, the report says "style path present; effect not detectable on game data; to be fine-tuned on OCD/AC laps with controlled styles".
  - The OCD/AC fine-tune may raise γ and refit W and W1e. That belongs to a later stage.
  - **If L3 still fails the bars,** that is an implementation fault, because critic 1 at L3 is critic 3 plus the exact-row nuisance up to about 3 % of the style signal. Stop and report to the lead. Critic 3 is never delivered as Q.

**Critic 2:** it is not TD3's Q. Rungs 1–2 apply to it in the same way. If it fails at every rung, it is reported as a negative result, and screening uses critic 1.

**What does not trigger the ladder:** unchanged from B16 (a null style effect, critic 2 missing its bar against nuisance-only, a missed Barcelona bar).

## Runs and budget delta
Fit times are [J] until R0 measures them.

| Run | Content | Fits |
|---|---|---|
| R3b (always) | B21 in-fold: F_cell repeat 0 + F_person repeat 0 + F_hash, twin 0, c ∈ {0.03, 0.1, 0.3} (c = 0 reused from R3); published-meta sensitivity at the selected c on F_cell repeat 0 + F_person repeat 0 | 45 + 10 heavy |
| R3c (only if triggered or adopted) | B21 at the selected c (modal value per fold set from R3b) on the full R3 grid, so critic 1 can inherit it | 70 heavy |
| R3d (always) | S22a and S22b, F_cell repeat 0 + F_person repeat 0, twin 0 | 20 heavy |
| R5b (only if rung 0 fails) | B23 rung 2: 3 levels × (F_cell + F_person) × 3 repeats × 5 folds × 2 twins | 180 small |

- Order: R3 → R3b → R3c if needed → R3d → R4 … R5 → R5b if needed.
- R5 waits on the decision about which critic 3 is delivered.
- Typical added time is about 75 heavy fits (+70 if triggered), roughly 40–75 min (+35–70 min), plus about 10 min of small fits.
- The 6 h hard stop is unchanged. If R0 projects beyond it, cut in this order:
  1. the published-meta sensitivity;
  2. S22b;
  3. the B3 repeats.
- Never cut epochs or bars.

## Bars added (to `bars.json`, written before training)
- B21 adoption: B21.4. B21 trigger and triggered delivery: B21.5.
- S22a: B22.3. S22b: B22.4.
- B23: the "Critic 1" bars apply to the delivered rung. The rung order and the γ levels are written to `bars.json`.

## Tests added to B19
- Π = 0 (c = 0) gives critic 3 bit for bit.
- m^in, w and r̄ do not change when held-out rows are multiplied by 1,000.
- In cross-fitting, a cell never scores pairs against an m computed from its own half.
- ∂Π/∂p is finite.
- Π is bit-identical after critic 1's fine-tune, and Π never takes s or e as input.
- An unseen track gives Π = 0 and `meta_prior='none'`.
- The published meta can never be written into a delivered bundle.
- S22a filters training rows only; the evaluation pairs are identical.
- B23: γ is applied at training and at query and is saved and loaded with the bundle; the bundle kind delivered as TD3 Q is always `critic1`; the ladder order is deterministic.

## Risks added

| Risk | Metric that reveals it |
|---|---|
| The prior pulls TD3 towards the community median on a car where that median means nothing (OCD/AC, provisional map) | λ·‖∂Π/∂p‖ against ‖∂(heads)/∂p‖ at the TD3 proposals; the selected c |
| Prior dominance on Barcelona | Barcelona Δ for each c (proxy: −0.09 to −0.13 at c ≈ 0.9) |
| Date drift mistaken for setup quality | pair accuracy on same-month pairs; residual-based vs raw win rates (E21.5) |
| Tow / wear distorts the aero effect in Quali and Race | S22b per-knob deviation sizes for aero and aero share |
| Rung 2 hides a style path that hurts | the selected γ level; the ΔR² trend from L1 to L3 |

## Needs Ibrahim
1. **"Unsuccessful in grading"** = critic 3 misses any R11 ranking bar on A (B21.5)? With §12.21, this replaces old Needs Ibrahim 1. A "really bad" style result now only moves critic 1 down the B23 ladder; style is never dropped.
2. **B21 in production:** if B21 is adopted, should Π stay in the critic used for OCD/AC queries, or be switched off there? My recommendation is to switch it off until the knob constants are measured (§12.16), because m comes through the provisional map.
3. **Target session:** is the OCD/AC condition a solo TT-style hotlap (fixed ERS, no tow)? If yes, S22b is the test that could justify a TT-referenced q. Today the data cannot show TT-specific rankings for unseen drivers.

## Needs web-researcher
1. How f1laps computes its "meta" setup page. Median or mean? Of which setups: all, top-N by lap time, a time window, which sessions? Is it weighted by lap time? This sets the size of the leak in B21.1.
2. F1 25 / F1 26: does Time Trial fix ERS deployment, and does it apply stricter track limits (lap invalidation) than online or career Qualifying? Does TT have any slipstream? This is for the report text only; B22's decisions rest on the data.
