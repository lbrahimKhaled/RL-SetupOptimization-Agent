# C2 style-conditioned critic: architecture review (15 Nov 2026)

**Verdict: CHANGES-REQUIRED.** Four items block the spec:
- **R1:** the knob constants are missing.
- **R2:** five of the 16 TUM geometries do not match the game layout.
- **R6/R11:** within-cell ranking has to be done within session. Rows S alone cannot show a setup ranking, so the TD3-readiness bar has to rest on rows A.
- **Assists (Needs Ibrahim 2):** ABS and TC move the style numbers more than drivers differ from each other.

The §1 decisions stand and none is overridden. Two of them cause real problems, stated below: the assist confound against Decision 2, and the §12 conflict.

Evidence tags: [F] formulation, [C] code read, [D] measured read-only on `setup_data/` (scripts in the session scratchpad `scratch/c2/`), [J] judgement. No literature is relied on. Open facts are listed under "Needs research".

## Evidence

**Counts [D]**
- Setups: 21,704 dry, no duplicate `setup_id` → 21,523 after `gap ≤ 8 %` (F1 25 12,948, F1 26 8,575).
- Rows S before filters: **857 + 877**, not 865 + 877. 9 F1 25 laps have no graph samples.
- Rows S after the gap filter and the R1 box filter: **1,697** (856 + 841); **1,686** of them have a styled lap.
- Exact telemetry rows after filters: **613**, one per cell. They contain **no within-cell pairs at all**.

**Within-cell signal in rows S [D]**
- Pairs inside a cell (game, driver, track) mostly cross sessions. The median |Δgap| is **2.09 %** across sessions against **0.74 %** within a session, so cross-session pairs measure the session, not the setup.
- Within cell × session: **295 pairs, 54 drivers** (rows with style). By game: F1 25 171 pairs / 37 drivers, F1 26 118 / 23.
- Noise floor: the same setup in the same cell and session still differs by rms **1.28 %** (191 pairs). Different setups differ by rms 1.62 %. The setup share of pair variance is 0.38, with a CI that touches 0.
- Ridge (nuisance + driver + p, cell folds) within cell × session on S: pair accuracy **0.539, 95 % CI [0.430, 0.613]** (SE 0.046).
- Adding s × top-4 PCs of p: **+0.010 [−0.056, 0.085]**.
- The paired SE is 0.036, so the minimum detectable style gain is about 0.10. At this sample size, S cannot detect a style effect on ranking.

**Rows A [D]**
- 21,277 rows inside the box; 1,584 cell × session groups; **3,029 pairs; 284 drivers** (F1 25 2,006 pairs, F1 26 1,023).
- Ridge pair accuracy **0.589 [0.561, 0.617]**. With a date-trend nuisance: 0.586 [0.557, 0.616].
- The setup signal exists, and it is measurable on A.

**Date drift [D]:** within a driver/track/session, the later setup is faster by **0.41 %** on average (64 % of 4,232 pairs). This is about the size of the setup effect.

**Driver offset vs style (§6, ridge, 8 geometry-free numbers) [D]**

| Folds | nuis + p | + s | + driver offset | + driver + s |
|---|---|---|---|---|
| By driver (unseen driver), all S | 0.451 | **0.441** | n/a | n/a |
| By cell (driver seen), all S | 0.481 | 0.495 | **0.573** | 0.573 |
| By driver, TT rows only | 0.220 | **0.156** | n/a | n/a |
| By cell, exact rows (609) | 0.384 | 0.399 | 0.501 | 0.535 |

- Style does not stand in for skill on unseen drivers; it makes those predictions worse. The driver offset adds +0.09 to +0.16.
- Once the offset is in, style adds 0 to 0.03 R².
- Across tracks, the style numbers do correlate with the gap (0.05 to 0.28 on exact rows), but that correlation is mostly the controller/assist information.

**Style extraction (854 of 859 laps; geometry numbers on the 556 TUM-track laps) [D]**
- A numpy port of `dstyle` at ea37559 equals the original to **0.0** on a 2,000-sample synthetic Monza lap (DataFrame shimmed only for the check).
- Time from `time_mapping` with (0 m, 0 s) prepended closes within 1 % on 854 laps; 4 are rejected.
- Sample spacing per lap ranges 1 to 11 m (outliers at 30, 48 and 110 m). The effective rate is **≈15 Hz at the slowest 5 %** of a lap and ≈20 Hz at the median, not 7–10 Hz.
- Game steering + = right turn (correlation with κ is −0.75 to −0.94). This does not affect the 12 numbers.

The 12 numbers. Units are style-rl's spread; game values use steer × 115.

| Number | Game median [p1, p99] | sd / spread | Within-track sd / spread (4 tracks) | Change when samples halved | attacker / smooth / neutral |
|---|---|---|---|---|---|
| B_mean_active | 0.63 [0.35, 0.95] | 1.32 | 1.20 | 0.17 | 0.73 / 0.43 / 0.59 |
| B_event_peak_mean | 0.88 [0.52, **1.00**] | 1.20 | 0.93 | 0.03 | 0.88 / 0.48 / 0.69 |
| B_apply_rate | 5.8 [1.5, 18.2] | 1.24 | 1.23 | **0.49** | 5.1 / 0.84 / 1.3 |
| B_release_rate | 1.35 [0.46, 4.7] | 1.48 | 1.35 | 0.14 | 0.69 / 0.37 / 0.45 |
| T_apply_rate | 0.85 [0.51, 4.6] | 2.33 | 2.16 | 0.12 | 0.78 / 0.36 / 0.48 |
| T_release_rate | 2.6 [1.3, 4.4] | 1.52 | 1.42 | **0.77** | 2.8 / 3.8 / 3.5 |
| S_reversals_5deg_per_km | 6.2 [2.9, 26.6] | 0.75 | 0.70 | 0.07 | 23.5 / 3.3 / 12.2 |
| S_mean_abs_rate | 18.5 [11.6, 37.0] | 0.65 | 0.51 | 0.06 | 24.2 / 15.2 / 18.9 |
| S_corner_peak_angle | 46.9 [30.8, 66.1] | 1.26 | 0.89 | 0.03 | 66.2 / 71.9 / 63.2 |
| L_border_mean_m | 2.20 [1.82, 2.66] | 0.57 | **0.03** | 0.00 | 2.12 / 2.44 / 2.39 |
| L_inside_dist_apex_m | 0.96 [0.86, 1.21] | 0.13 | **0.14** | 0.00 | 1.43 / 1.58 / 1.42 |
| L_outside_dist_exit_m | 3.52 [2.02, 5.40] | 1.33 | **0.04** | 0.00 | 4.07 / 3.91 / 4.05 |

What the table shows:
- **Scale:** the game numbers sit on the same scale as style-rl's. Driver variance within a track is about 1 spread unit for the 9 B/T/S numbers, so it is meaningful.
- **Degenerate numbers:**
  - The L_* are per-track constants (placeholder line; driver ICC −0.06 to 0.17).
  - B_event_peak_mean hits the 1.0 ceiling on 19 % of laps.
  - S_corner_peak_angle and L_* are NaN on the 298 laps without geometry.
  - B_apply_rate and T_release_rate are sensitive to sampling per lap. At population level, the correlation with sample spacing is small (|r| ≤ 0.11).
- **Trait:** each driver's style is consistent across tracks. Driver ICC on the 9 numbers is 0.54–0.95 overall and 0.31–0.95 within one controller type (28 drivers). Style dimension by parallel analysis: eigenvalues 4.67 / 1.60 / 0.88 against thresholds 1.26 / 1.17 / 1.11, so **k = 2**.
- **Device and assists (this conflicts with the intent of Decision 2):**
  - Controller laps sit +0.4 to +1.4 spread units above wheel laps.
  - In F1 26, **ABS on vs off moves the four brake numbers by 1.0–2.3 spread units and T_apply_rate by 2.5. TC on vs off moves T_apply_rate by 3.7** and the brake numbers by 0.8–2.2.
  - Only 166 of 410 F1 26 laps have ABS off and TC off. F1 25 laps carry no assist fields.
  - So part of "style" is the assist setting, not the driver.
- **Off-support in OCD:** style-rl's OCD drivers fall outside the game's p1–p99 on 2 numbers (attacker), 6 (smooth) and 4 (neutral).

**Geometry check [D]** (game lap distance vs TUM length, best |corr(steer, κ)|):

| Result | Tracks |
|---|---|
| Pass | Austria, Bahrain, Brazil, Canada, China, Hungary, Japan, Monza, Silverstone, Spa, USA: |ΔL| ≤ 0.3 %, |corr| 0.88–0.94, start offset −18 to +18 m |
| **Fail** | Abu Dhabi −4.8 % / 0.59; Mexico +5.3 % (0.90 only after a −230 m shift); Netherlands −1.4 % (+44 m); Melbourne −0.5 % / 0.80; Catalunya +0.2 % / 0.83 |

The likely cause is older layouts in TUM (unverified).

**Knob map [D, C]**
- C1-R1's constants rescaled to the new click ranges drop **96.9 %** of rows (χ 0.37–0.91; also diff (pct − 50)/50 < 0 for 10–40 %).
- The box rule does **not** fix the constants. Many families of maps pass with < 0.1 % dropped, so constants must be chosen by a stated rule (R1).
- Ride height read literally as mm: **1,106 rows (5.1 %, all front clicks 15–19) fall below OCD's 20 mm floor.** That alone breaks the 5 % rule.
- An mm reading is consistent with the 14 mm static front height that `cars.py` cites from F1DOC §5. The game rake of 5–45 mm is ground-effect era; the SF70H's ~97 mm rake (`cars.py` comment) lies outside game support.
- OCD's LSD [C]: friction torque = `LSD_lock_* × input torque` (`drivetrain_rwd_lsd_eqns.cpp:15-21`), a 0..1 ratio.
- Geometry constants from the f1 worktree [C]: m_c 151.1 / 185.4 kg, t 1.6 / 1.6 m.

**PCA on rows A [D]**
- 8,895 unique (track, setup) rows. Eigenvalues 4.15 / 1.77 / 1.38 / 1.29 / 1.01 against thresholds 1.08 / 1.06 / 1.05 / 1.04 / 1.03, so **k = 4**.
- The varimax groups mix physics. One group holds {f_R, h_F, bias, toe F/R, diff drive/coast}, a likely "default vs meta" copying axis: 8,284 unique setups in 21,277 rows, one setup used 512 times.

**Repo [C]:** `setup_data/` is not gitignored (`git check-ignore` returns nothing; only `*.zip` is covered). Neither the venv nor system Python has pandas.

## [ARCH] items settled

| Item | Decision |
|---|---|
| Import vs copy dstyle | **Copy, as a numpy-only port** (logic verbatim, DataFrames removed) into `rlsetup/style/`, with the commit hash ea37559. The equality test runs against the pinned original. |
| pandas | **Not a runtime dependency.** Dev/test-only (`requirements-dev`), used only for the equality test. scipy is already in the venv. |
| Placeholder line where no TUM raceline exists | Minimum-curvature line computed once with the TUM helpers vendored in ocd_first_lap, cached and flagged. Centre line only as a flagged last resort. Low stakes, because L_* are held out. |
| L_* as inputs | **Hold out.** They are per-track constants that duplicate z and invite track memorisation. Keep the slot, masked. |
| Diff map | **pct / 100** (0.1–1.0). No fitted constant; OCD's lock is a 0..1 ratio [C]. |
| Ride height | **h = click + 5 mm** on both axles (1 click = 1 mm assumed; offset fitted to the box). Rake = c_R − c_F mm. |
| Head size | One hidden layer of width 16, SiLU, for every group head (C1-R13). Critic 2: 2 × 32 SiLU. |
| s into heads | A **shared linear encoder e = W s, 9 → 2, no bias** (parallel analysis k = 2). Each head of critics 1 and 2 sees e. |
| Critic 1 from critic 3 | **Yes.** Copy the same-fold critic 3 weights and add zero-initialised e-input columns, so critic 1 starts equal to critic 3. |
| Driver offset | **On for all three critics** (training only, never in q). Data in §6 table. |
| Off-support guard for s | **Yes, by clipping, not by penalty.** TD3 acts on p, not on s, so the risk is extrapolation, not exploitation (R10). |
| §8 ridge products | 9 transformed B/T/S numbers × top-4 in-fold PCs of p. |
| Ranking groups | **Cell × session.** |
| Bars | R11. |

## Binding changes

**R1 (blocking). F1 25/26 → OCD constants** (one config, all marked "fitted to the OCD box, not measured"):
- Ride frequency, both axles: f = 2.5 + 2.5 (c − 1)/40 Hz.
- Bar rate, both axles: **K_b = 10,000 + 12,500 (c − 1) N·m/rad**.
- Ride heights: h = c + 5 mm.
- Aero: aero_level = (fw + rw)/100; aero_front_share = 0.35 + 0.25 (s − 1/52)/(50/52), with s = (fw + 1)/(fw + rw + 2).
- Brake bias pct/100; LSD pct/100; camber direct; toe signs as C1 (assumed).

Selection rule, stated in the results: one map per click type for both axles; the p1–p99 support covers sha_f1_best's χ 0.40 and K_φ 4e5 and the OCD default χ 0.613; bars carry about half of K_φ (median 0.50); ≤ 5 % dropped **per game**.

Result: **246 of 21,523 dropped (1.14 %; F1 25 0.32 %, F1 26 2.38 %), all on χ plus 1 on K_φ.** χ ranges 0.352–0.796 (median 0.561); K_φ 164–945 kN·m/rad (median 557); K_b ≥ 0 by construction (minimum 10 kN·m/rad).

sha_f1_best remains outside support on rake (50 vs ≤ 40 mm), camber F/R (0 vs ≤ −2.5 / −1.0), rear toe and coast lock. Coast lock is pinned per C1-R10; the R9 guard covers the others.

**R2 (blocking). Geometry gate and alignment.**
- Per track, fit a shift (and a scale) of game distance to geometry progress by maximising |corr(steer, κ)|.
- Accept a geometry only if the median |corr| ≥ 0.85 and |L_game / L_geo − 1| ≤ 1 %.
- Abu Dhabi, Mexico, Netherlands, Melbourne and Catalunya fail and join the sourcing list. Decision 7 is kept: nothing is dropped.
- The same gate applies to the z track features.
- Store the shift, scale and correlation in provenance.

**R3. Lap QC.** Prepend (0, 0) to `time_mapping`. Reject laps with |t_end / lap − 1| > 1 % or median sample spacing > 8 m (13 laps). Store spacing per lap and the steering sign.

**R4. Style input transform.** The 9 B/T/S numbers enter. Take log of the six rate numbers. Then robust z (median and IQR on training-fold cells) and clip to ±3. The numbers themselves are unchanged (Decision 2). D distances are reported in style-rl's raw spread units.

**R5. Device/assist control.**
- Controller and assists stay in the nuisance.
- The "shuffled style" ablation shuffles **within track × controller × assist stratum** (F1 25: track × controller). Otherwise "style helps" can mean "assist information helps".
- Report the style effect separately on assist-free F1 26 cells.
- Whether assists also enter the heads is Ibrahim's call (Needs Ibrahim 2).

**R6 (blocking). Ranking pairs within cell × session only.** Cross-session pairs stay out of every ranking metric.

**R7. Nuisance.** Driver offset keyed by (game, user), α = 1, rare drivers pooled per fold (C1-R4/R13). Add a within-driver time trend (months relative to the driver's mean date) and game × month. Also team, session, controller, assists and game.

**R8. Game code and query.** g is the source one-hot. At an OCD/AC query, use the uniform 1/n_sources vector. Report how far the setup rankings disagree between g = F1 25 and g = F1 26.

**R9. Shared folds.** One cell-level fold assignment over rows A ∪ S, because S ⊂ A. Critic 3 is pre-trained per fold on that fold's training cells only. PCA, scalers, the style scale and the nuisance are all fitted inside the fold.

**R10. s guard.** At query time, clip e(s) to the training [min, max] per dimension. Log the Mahalanobis distance of s to the training cells and the fraction of queries clipped. The C1-R9 p-guard draws s from training cells.

**R11 (blocking). Bars, fixed now. Report if missed; do not tune.**
- **Critic 3 on A (the TD3-readiness bar):** pair accuracy ≥ 0.56; driver-cluster CI lower bound > 0.52; ≥ ridge − 0.02 on the same pairs (ridge is 0.589 today); strict variant ≥ 0.54; held-out-setup R² gain > 0.
- **Critic 1 on S:** non-inferior to critic 3 on the same pairs (paired CI lower bound of c1 − c3 > −0.05). Held-out-driver R² no more than 0.01 below critic 3. With s fixed at the training mean, critic 1's pair accuracy on A stays within 0.01 of critic 3.
- **Style helps:** claimed only if the paired CI of (real s − stratified-shuffled s) on S pair accuracy lies above 0. Held-out-setup R² is reported with its CI. Every report states that the minimum detectable gain is about 0.10, so a null result is the expected outcome.
- **Critic 2:** held-out-setup and held-out-driver MAE below the nuisance-only model, with paired CIs excluding 0. Within-track Spearman is reported.
- **Groups:** C1-R12 applied on A (critic 3), not on S.
- Per game and pooled; held-out track is reported only.

**R12. Sensitivity runs.** Replace "625 exact-only" with two runs:
- **TT-only S** (838 rows, 104 pairs, 36 drivers) as the session-consistency check.
- **Exact rows** (609) for critic 2 only. State that exact rows cannot test ranking: one row per cell.

**R13. Spec facts to correct.** S counts as in Evidence. Sampling is 1–11 m per lap, ≈15 Hz in the slowest 5 %. TUM covers 11 game layouts, not 16. Gitignore `setup_data/` and `data/tracks_extra/raw/` before anything else.

## Risks to watch

| Risk | Metric |
|---|---|
| Style acts as a device/assist proxy | Stratified vs unstratified shuffle gap; corr(e, controller/assist codes) |
| TT style is wrong for Race/Quali | Style gain on TT pairs vs Race/Quali pairs (a mismatch attenuates toward 0) |
| Extrapolation in s on OCD/AC drivers | Fraction of queries clipped; Mahalanobis distance |
| Date drift leaking into the setup heads | Pair accuracy on same-month pairs vs all pairs |
| Map mis-scaling | Fitted `scale`; RMSE on the first OCD batch before and after refit |
| Sampling artefacts | corr(style, sample spacing) < 0.2 (now ≤ 0.11) |
| Critic 1 drifting from critic 3 | A pair accuracy of critic 1 at mean s vs critic 3 |
| Copied-setup PCA groups | Adjusted Rand index across folds |

## Needs Ibrahim
1. **TT style for Race/Quali setups.** Recommend **yes**, with session as nuisance and pairs within session only. Evidence:
   - Style is a stable driver trait (cross-track ICC up to 0.95).
   - TT-only gives just 104 pairs, too few to rank.
   - A wrongly assigned style acts as noise in s, which biases toward "no style effect" rather than producing a false positive.
2. **Assisted laps.** ABS/TC move the style numbers by 1–3.7 spread units; F1 25 assists are unknown. Options:
   - (a) assists as nuisance only (as the spec has it);
   - (b) an assist code also as head context, queried at "off" for OCD/AC (OCD has no ABS, DECISIONS #5). **My recommendation;**
   - (c) style rows restricted to assist-free laps (F1 26: 166 of 410; F1 25: none verifiable).
3. **PROJECT_CONTEXT §12** lists "Driver style vector z as input" as superseded. Your 15 Nov Decision 2 brings it back for the setup critic. Please confirm, and update §12 and §4 (z_drv) so agents do not reject it.
4. **Accept the R1 constants** (fitted to the box, not measured) and the mm reading of ride height (+5 mm offset)?
5. **§11.7:** Table VIII radius bands again as the PLACEHOLDER for z in C2?
6. **Before the 14 failing or missing geometries are sourced:** smoke runs only (recommended), or interim fits on the 11 tracks that pass?

## Needs research (web-researcher)
1. F1 25/26 setup menu: are the ride-height values 15–35 / 40–60 in mm, and what is one click? Are any physical values published for the suspension (1–41) and anti-roll-bar (1–21) clicks?
2. Which layout versions do F1 25/26 use for Albert Park (post-2022?), Catalunya (no final chicane, 2023+?), Yas Marina (post-2021?) and Zandvoort (4.259 km)? Which version is each TUM file? Why does the game's Mexico lap distance read 4.52 km against about 4.30 km real?
3. Is f1laps' `source_normalized` steering the raw device input or the post-assist/linearity value, and what wheel angle corresponds to 1.0?
4. Carried over from C1: F1 game toe labels, toe-in or toe-out.
