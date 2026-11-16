# C2 — Style-conditioned setup critic (replaces C1)

Status: reviewed by rl-architect 15 Nov 2026 (`C2_review.md`, CHANGES-REQUIRED with binding changes R1–R13). **Where this spec and the review differ, the review wins**; every [ARCH] item is settled there. Open items for Ibrahim are listed in the review ("Needs Ibrahim").
Stages: **C2a** data + knob map + style extraction + geometry gate (no model); **C2b** critics, training, evaluation. C2b starts after Ibrahim's answers and the geometry sourcing.
Scope: the critics of the TD3 setup agent. The TD3 actor and its training are NOT part of C2, but every critic must expose what TD3 needs later (§9).
C1 (`C1_critic_warmstart.md`, `C1_review.md`) is superseded; its review findings are reused here where they still apply and are cited as C1-R<n>.

## 0. Facts (measured by the lead on 15 Nov, read-only)
| | F1 25 | F1 26 |
|---|---|---|
| Dry setups, 20 game parameters + lap time (`setups.csv`) | 12,982 | 8,722 |
| Public leaderboard laps with telemetry (`laps.csv` + `graph_*.csv`) | 458 | 410 |
| ... matched exactly to a setup (`has_telemetry=true`) | 295 | 330 |
| Laps per (driver, track) | always 1 | always 1 |
| Setups by a driver on a track where that driver has a telemetry lap | 865 | 877 |
| Recorded car positions / racing lines | 0 | 0 |

- Telemetry laps are all Time Trial, from top-10 leaderboard views: fast drivers only; lap spread within a track 1.4–1.7 % (median).
- Telemetry is sampled every 3 m of distance (median). `time_mapping.csv` gives elapsed time in ms every 10 m (Australia F1 26: 77,237 ms at 5,270 m for a 77.323 s lap). Steering is `source_normalized` −1..1 (median per-lap |max| 0.69).
- F1 25/26 menu ranges differ from F1 2020, so C1's click map does not apply: wings 0–50, suspension 1–41, anti-roll bars 1–21, ride height F 15–35 / R 40–60, diff 10–100 %, camber F −3.5…−2.5 / R −2.0…−1.0, toe F 0–0.2 / R 0.1–0.25, brake bias 50–70 %. Same ranges in both games.
- F1 26 laps also carry assists (traction control, ABS, steering/braking assist, gearbox, racing-line assist); F1 25 laps only controller.
- Geometry: TUM (`ocd_first_lap/vendor/racetrack-database`) covers 16 of the 24 game tracks. Missing: Azerbaijan, Imola, Las Vegas, Madrid, Miami, Monaco, Qatar, Saudi Arabia, Singapore — being sourced by web-researcher (`docs/research/c2_track_geometry.md`).

## 1. Decisions by Ibrahim (15 Nov 2026) — not to be changed by the team
1. C2 **replaces C1**. C1 code and tests are deleted; C1 docs stay as superseded reference. Take intuition from C1.
2. **Driver style is the most important new input**: the 12 numbers of `style-rl` (branch `style12-whole-lap`, `style_rl/dstyle/features.py`). Definitions, thresholds and the per (driver, track) median come from that repo, unchanged.
3. Setup inputs are **OCD knobs** (15, as C1 Decision 2); brake pressure and the four tyre pressures are dropped (no OCD knob). The game→OCD map is rebuilt for F1 25/26 ranges with the **frequency-mediator rule** of C1 Decision 8.
4. **All data** in `setup_data/` (F1 25 + F1 26). More scrapes will follow: loaders take new export folders without code changes.
5. **Steering: 115 °/unit** (style-rl's OCD conversion, 0.15 rad × ratio 13.4), so game and OCD styles share one scale.
6. **Line numbers are a placeholder**: computed on the optimal line of the track (same for every driver), flagged as placeholder, to be replaced by measured values in OCD / Assetto Corsa.
7. **No track is dropped**; missing geometry is sourced.
8. Target = **% gap to the track median lap**. **Three critics**: (1) group critic conditioned on style — the TD3 Q; (2) full lap-time predictor conditioned on style — screens setups before costly OCD / Assetto Corsa runs; (3) setup-only critic on all setups. Groups: PCA first, physics groups as fallback (C1 Decision 4).
9. **Reusable for Assetto Corsa** (an F1-like car): critics work in OCD physical knobs; every data source gets its own adapter to those knobs.
10. TD3 is later; keep its needs in mind (smooth ∂Q/∂p, feasibility, support box, twins).
11. No drivers/teams as network inputs (C1 Decision 7); the style vector is the driver descriptor.

## 2. Data (`rlsetup/critic/sources/f1laps.py`)
Input: one or more export folders (CLI list or glob). Each folder is an f1laps export with `setups.csv`, `laps.csv`, `setup_lap_map.csv`, `time_mapping.csv`, `graph_{throttle,brake,steer,speed}.csv` (zip archives are read in place; nothing is extracted into the repo). The game is a column (`F1 25`, `F1 26`, later more). Record each file's sha256 in provenance. Usernames → integer ids at load; no username is written to any output. CSVs stay out of git (`setup_data/` must be gitignored — check).

Setup pipeline, row counts logged per step (C1 §2 adapted):
1. Dry, all 15 mapped game parameters present. Deduplicate by `setup_id` across folders.
2. `gap` = 100·(lap / median lap of (game, track) − 1). Median per game: the 2026 car is a different car.
3. Drop `gap > 8 %`, recompute median and gap once (selection on the label, stated as such, C1-R4).
4. Map to OCD knobs (§4); drop rows outside the OCD box with 1e-9 tolerance; stop and report if > 5 % go.
5. Normalise session {Race, Quali, TT}; controller {Wheel, Controller, blank}; assists where present, else `unknown`.

## 3. Driver style (`rlsetup/style/`)
**Code.** style-rl is a dependency like ocd_first_lap: do not edit it. [ARCH] import it from `~/Documents/style-rl` at a pinned commit, or copy `dstyle/features.py` + `dstyle/track.py` into `rlsetup/style/` with the commit hash and a test that both give identical numbers on a fixture lap. Proposal: copy + equality test (no runtime path dependency, works on RunPod / Windows).

**f1laps lap → style-rl input.**
- `time_s`: from `time_mapping.csv` (ms → s, interpolated to the 3 m samples), checked against `lap_time` (|end − lap| < 1 %, else the lap is rejected and counted).
- Resample throttle, brake, steer to a uniform **25 Hz** grid in time (linear), as style-rl expects (`SMOOTH_SAMPLES = 3` = 0.12 s). Real rate is ~7–10 Hz in slow corners: rates there read smoothed — state in results.
- `steer_deg` = steer × 115.
- Position: the game gives none. Pass `q=` (style-rl's documented path for simulators): `progress_m` = game distance × (track length / game lap length); `lateral_m` = lateral offset of the **placeholder line** at that progress; widths from the geometry. Corners come from `Track.find_corners()` on the geometry.
- Placeholder line: TUM raceline where it exists; else the line delivered by the research step; else [ARCH] (a minimum-curvature line computed once, or the centre line). Line numbers (`L_*`) are then per-track constants. `is_placeholder` flag stored.
- Geometry: TUM `x_m,y_m,w_tr_right_m,w_tr_left_m` → style-rl's `x_center … y_right` (edges = centre ± width × left normal). Track name map F1 game → geometry file in config.

**Cells.** Style of (game, driver, track) = median over that driver's laps there (currently 1 lap each). Laps that leave the road cannot be checked (no positions): all public leaderboard laps are valid laps, accepted as on-track. Style is cached per lap (`runs/style_cache/`, keyed by lap_id + code hash).

**Standardisation.** Each number divided by its spread across real cells (style-rl's `D` unit), spread fitted on training folds only. `L_*` are constants within a track: [ARCH] keep them as inputs (they then act as track features) or hold them out of the network until real line data exists, keeping the slot.

## 4. Game → OCD knobs (`rlsetup/critic/sources/f1_game_map.py`)
Same 15 knobs and order as C1 §3; normalised to [−1, 1] with `ocd_lap.car_setup.KNOB`. Same frequency mediator (K_w = m_c(2πf)², K_b ≥ 0 by construction, χ and K_φ derived), m_c and t from `CarGeometry` of the fine-tune car (C1-R1). New constants for the F1 25/26 ranges [ARCH]: frequency per suspension click (1–41), K_b per bar click (1–21), ride heights (15–35, 40–60 — in mm?), aero level and front share from wings 0–50, diff (pct − 55)/45 or as C1 [ARCH]. Fit them to the OCD box as C1-R1 did and say so. Toe signs as C1 (assumed). `lsd_lock_coast` pinned in OCD fine-tune (C1-R10). Differentiable K_b,F/R and the training support box exported (C1-R2). Setup hash as C1-R3.

**Adapter contract** (`rlsetup/critic/sources/base.py`): `to_knobs(df) -> (n, 15) array`, `style_inputs(lap) -> style-rl input`, `meta(df)` (game/sim, session, controller, assists). F1 game = first adapter. Assetto Corsa = a later adapter with its own map; nothing in model/train/evaluate may depend on F1 column names.

## 5. Rows for each critic
- **S (style rows):** setups by a driver on a track where that driver has a telemetry lap; style = that cell's style. F1 25 + F1 26: 865 + 877 = 1,742 before filters, of which 625 are the exact telemetry laps. **Within-cell pairs exist only through these extra setups** (each cell has 1 telemetry lap). Session and assists enter as nuisance. Sensitivity run: exact telemetry rows only (625).
  > **Needs Ibrahim's confirmation:** applying a driver's Time-Trial style to their Race/Quali setups on the same track.
- **A (all rows):** all setups after §2, no style. Used by critic 3.

## 6. Critics (`rlsetup/critic/model.py`)
Inputs: p (15 knobs, [−1, 1]), s (12 standardised style numbers), z (track features, C1 §5 / C1-R15, from the geometry: `n_corners, longest_straight_m, straight_share, slow_corner_share, fast_corner_share`), g (game one-hot). Output Ĵ = % gap, lower is better; `q = −Ĵ`. SiLU everywhere (smooth ∂q/∂p).
1. **GroupCritic** (TD3's Q): Ĵ = b + Σ_g h_g(p_g, s, z, g). Each head sees its own knobs plus the context (s, z, g), so a group's effect can depend on the driver. [ARCH] head width (C1-R13 had 1 × 16 without s); whether every head gets the full s or a learned low-dim projection of it (shared encoder) given 1,742 rows.
2. **FullCritic** (lap-time predictor): one MLP on (p, s, z, g); all cross-group interactions allowed. Also reports whether grouping costs accuracy.
3. **SetupCritic**: GroupCritic without s on rows A. Larger data; context for critics 1–2 and a fallback when no style is known. [ARCH] whether critic 1 is initialised from critic 3 (pre-train heads without s, then add s).
- Nuisance (training only, never in `q`): game, session, controller, assists, team. **[ARCH] driver offset:** C1 needed it (driver skill dominates). With a driver offset, the style main effect is absorbed and style acts only through its interaction with p — which is what TD3 needs; without it, style must also explain skill. Proposal: driver offset kept for critic 3; for critics 1–2 report both and pick by the §8 rule.
- Twins for critic 1 (C1-R7: bootstrap by setup hash, `q_min = −max(Ĵ1, Ĵ2)`, no target networks at γ = 0). Frozen `scale` + `b` for the later OCD fine-tune (C1-R8, R11).
- Off-support guard (C1-R9) on p for critics 1 and 3; [ARCH] whether s needs one too (TD3 will see OCD styles that may fall outside the game cells).

## 7. Groups
As C1 §4 with the C1-R12 rule (keep PCA unless the paired driver-cluster CI of PCA − physics pair accuracy lies below 0, or held-out-setup R² gain trails physics by > 0.01). Fitted on deduplicated rows A inside each training fold. Physics groups: aero + heights; springs + roll; alignment; diff; brake bias. The same groups are used by critics 1 and 3.

## 8. Evaluation (`scripts/critic_c2.py`, report `docs/results/critic_c2.md`)
Everything (scalers, PCA, style spread, nuisance) fitted on training folds only.
- **Held-out setups** (folds by setup hash): R² gain over nuisance-only.
- **Within-cell ranking** (folds by driver × track): pair accuracy and Spearman of the setup part of the prediction, driver-cluster bootstrap CI. Headline, as C1.
- **Does style help?** (new, the reason for C2) Critics 1 and 2 against the same model with s replaced by (a) the training-mean style and (b) a style shuffled across cells within a track. Paired driver-cluster CI on pair accuracy and on held-out-setup R². Also **held-out driver** folds (style must generalise to unseen drivers; this is the OCD/AC situation).
- **Lap-time predictor** (critic 2): MAE in % and in s on held-out setups and held-out drivers; rank correlation within track.
- Held-out track: reported only. Per game reported separately and pooled.
- Baselines: nuisance only; ridge on 15 knobs; ridge on 15 knobs + 12 style + their products with the 4 largest PCA loadings [ARCH].
- Bars fixed before running [ARCH]; if missed, report, do not tune to pass (C1-R13).

## 9. Interface for TD3 and Assetto Corsa
`critic.q(p, s, z, g)` and `q_min` for twins; `knob_feasibility(p)` (K_b ≥ 0), `support_box()`, `style_scaler`, `track_features(geometry)`. Saved with `run_provenance()`, data sha256, style-rl commit, geometry sha256. TD3 later: actor state = (s, z) (+ p0 per C1 §4 for the actor only); action = p.

## 10. Cleanup
Delete `rlsetup/critic/*` (C1), `scripts/critic_warmstart.py`, `tests/test_critic_*.py`, `docs/handoff/C1_coder_report.md`, `docs/handoff/C1_qa_review.md` (uncommitted). Mark `docs/specs/C1_*.md` superseded by C2 at the top. Reuse C1 code ideas freely; nothing imports C1.

## 11. Tests and outputs
Fixture export folder (synthetic, 3 tracks, few laps) — no real usernames in tests. Tests: loader on several folders + dedup; gap per game; knob map known values, K_b ≥ 0, bounds; style adapter: time reconstruction, 25 Hz resample, steering scale, equality with style-rl on a fixture lap; placeholder flag; folds never leak (scalers, PCA, style spread); determinism; `q` ignores nuisance; ∂q/∂p finite and non-zero; style ablation code path. Weights to `runs/critic_c2/` (gitignored). CPU, one thread unless the machine is free.
**Dependencies (checked):** `pandas` is NOT in the venv; style-rl's `dstyle` uses it (DataFrame in/out) and scipy (`cKDTree`, `gaussian_filter1d`). [ARCH] add pandas to the venv, or the copied `dstyle` takes numpy arrays (equality test still against the original).
**Git (checked):** `setup_data/` is NOT gitignored; it contains public usernames. Add `setup_data/` and `data/tracks_extra/raw/` to `.gitignore` first.
style-rl pinned commit: `ea37559005b4758e0c03bc3ced2359bab9894781` (branch `style12-whole-lap`).

## 12. Decisions by Ibrahim, 16 Nov 2026 (after C2a) — binding, override §1/§5 and the review where they differ
12. **Mexico is dropped** (no accepted geometry; "drop it").
13. **Style rows: exact telemetry laps first** (613 rows, one per cell). The Time-Trial-style extension to the driver's other setups (rows S, 1,697) is used **only if** the exact-row results are really bad. Consequence to state in results: exact rows contain no within-cell pairs, so style critics are judged on prediction metrics, not ranking; ranking is judged on rows A.
14. **Assists are not part of the style vector and not a network input.** They stay a training-only correction term (never in `q`) and the stratification variable of the shuffled-style ablation (review R5).
15. PROJECT_CONTEXT §12 "driver style vector as input" is **revived for the setup critic** (confirmed).
16. Knob-map constants: Ibrahim owns the game and will read menu values on request (list in the lead's message of 16 Nov); R1 constants stay provisional until then.
17. **Real fits now** on the tracks with accepted geometry (no smoke-only phase).
18. **Track edges are to be recorded in the game** (starting with Barcelona); the constant width stays only until then.
19. Lap filters are the team's call: cut implausible laps on both sides, and sanity-check against real-world lap times.
20. **Focus on Barcelona now.** Lead's reading (to be confirmed by Ibrahim): Barcelona (`spain`, 999 rows A, 31 telemetry laps, 24 exact rows) is the target track — every result is reported for Barcelona first, the first OCD/AC use is Barcelona — while style critics still train on all tracks, because 24 Barcelona rows cannot train a style-conditioned network.
21. **(16 Nov, later) A driver-style-aware critic is mandatory.** A pooled critic (style-free) may exist alongside it, but TD3's Q must always see the driver style. This overrides the part of C2b B16 that falls back to critic 3 as TD3's Q: if critic 1 fails non-inferiority, the fix is to make critic 1 safe (stronger regularisation / shrinkage of the style path, fallback rows, more data from OCD/AC), never to drop style from TD3's Q. A null style effect on game data is reported honestly; the style path stays and is fine-tuned on OCD/AC laps with controlled driving styles.
