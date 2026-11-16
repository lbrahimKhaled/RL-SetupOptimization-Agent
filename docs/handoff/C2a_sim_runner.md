# C2a sim-runner verification (16 Nov 2026)

**Overall: PASS.** Every count reproduces, two runs give bit-identical outputs, an independent re-implementation of the R1 knob map matches the pipeline exactly, all 21,277 rows A build an `ocd_lap.car_setup.Setup`, both ocd_first_lap trees are unmodified, and no username appears in any output. Nothing blocks C2a. There are three small reporting or provenance points (items 6–8); none changes a result.

Repo state: this repo at `e616f94` (dirty, C2a uncommitted). ocd_first_lap at `3945634` (`sf70h-tum-car`); the f1 worktree at `ee86683` (`sf70h-f1`). I edited no source.

## 1. Runs: PASS

| Command (cwd = repo root) | Wall time | Result |
|---|---|---|
| `.venv/bin/python scripts/fetch_track_geometry.py --offline` | 12.5 s | exit 0, 14 tracks converted; the gate table matches the coder's |
| `.venv/bin/python scripts/critic_c2_data.py` (cache) | 15.6 s | exit 0 → `runs/critic_c2_data/20261115_220208Z/` (849/849 laps from cache) |
| `.venv/bin/python scripts/critic_c2_data.py --no-cache` | 15.7 s | exit 0 → `runs/critic_c2_data/20261115_220223Z/` (0 from cache) |
| `.venv/bin/python -m pytest -q tests/test_c2_*.py` | 1.6 s | 50 passed, 1 skipped |

**Determinism**
- I hashed all 44 arrays of each `dataset.npz` (dtype + shape + bytes). Run 1 and run 2 are identical; the combined digest is `5a4e628cad95bd0c` for both.
- Stdout is identical apart from the cache-hit count.
- `summary.json` differs only in `created_utc` and `argv`.
- Against the coder's earlier run (`20261115_214827Z`), one array differs. `lap_median_spacing_m` of the single `lap_no_samples` lap is now 110 m instead of NaN. This is the coder's stated change ("spacing recorded even when the time map is missing"), so it is benign.

**Geometry conversion**
- The 14 CSVs are byte-identical to the files present before the run.
- The 14 JSON sidecars change on every run, but only in `created_utc` (I confirmed this with a third run and a field diff). Note that the offline re-run rewrote these untracked sidecars.

## 2. Counts: PASS (all match)

I reproduced these two ways: from the script's log, and from my own CSV parsing plus my own R1 formulas (`indep_recount.py`, which does not use the repo loader or `game_to_knobs`).

| Count | Expected | Measured |
|---|---|---|
| after gap ≤ 8 % | 21,523 | 21,523 (F1 25 12,948 / F1 26 8,575) |
| dropped by the box | 246 (≤ 5 % per game) | 246: F1 25 42/12,948 = 0.32 %, F1 26 204/8,575 = 2.38 %; χ 245, K_φ 1 |
| rows A | 21,277 | 21,277 |
| rows S | 1,697 | 1,697 (856 + 841) |
| rows S with style | 1,680 | 1,680 |
| exact rows | 613 | 613 |
| laps with style | 849 | 849 |
| QC rejections | 10 | 10 |
| gate | 22/25; Imola, Las Vegas, Mexico fail | 22/25: Imola `gate_low_corr`, Las Vegas `gate_low_corr`, Mexico `gate_length` |

**QC rejections in detail:** 4 `lap_time_closure` (median spacing 48, 3, 30.5, 8.5 m), 5 `lap_spacing` (9, 9, 10, 11, 11 m), 1 `lap_no_samples` (110 m). There are also 9 `lap_no_graph` laps, which are not counted as QC.

**Other figures that match the review:** rows S before filters 857 + 877; A pairs 3,029 with 284 drivers. χ in A runs 0.352 / 0.561 / 0.796 (min / median / max) and K_φ 164 / 557 / 945 kN·m/rad, as in R1.

**Ranking basis note (not a mismatch):** the script reports S ranking as 295 pairs / **60** drivers over all rows S. The review's "295 pairs, 54 drivers" is for rows with style.

## 3. Compliance with ocd_first_lap: PASS

- **Knob names, order, bounds and units.**
  - All 15 `KNOB_ORDER` names exist in `ocd_lap.car_setup.KNOB`, in the same relative order as `KNOBS`.
  - `knobs.knob_bounds()` reads `low`/`high` directly from `KNOB`, so the bounds cannot drift.
  - `car_setup.py` and `translator.py` are byte-identical between the main tree and the f1 worktree.
  - Units are SI: ride height in rows A runs 0.020–0.040 m and rake 0.005–0.040 m.
- **Independent map:** my R1 implementation, using the CarGeometry of `sf70h_f1`, gives a knob matrix equal to the `dataset.npz` knobs row by row (max |diff| = 0.0).
- **Setup construction:**
  - 200 random A rows (seed 0): 200 OK, 0 failures.
  - All 21,277 A rows: 21,277 OK, 0 failures.
- **K_b against the translator:**
  - `translator.translate(Setup, OCDBackend(car='sf70h_f1').base_params)` on the 200 rows emits 0 "clamped" warnings.
  - The minimum K_b from `translator.suspension_rates` is exactly 10,000 N·m/rad.
  - The largest difference from the click formula 10,000 + 12,500(c − 1) is 1.2e-10 N·m/rad.
  - K_b ≥ 0 holds on all 21,277 rows.
- **Toe sign:**
  - Game ranges: front 0.00–0.20 (labelled toe-out), rear 0.10–0.25 (labelled toe-in).
  - The translator output `toe_out_rad.front_*` = +rad(front) and `toe_out_rad.rear_*` = −rad(rear), which matches OCD's "positive = toe-out" on all 200 rows.
  - Whether the game value is per wheel or the axle total is still unverified, as the coder flagged.
- **Car constants:**
  - `PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python -m pytest -q -rA tests/test_c2_knobs.py` gives 9 passed, including `test_car_constants_match_the_f1_worktree`.
  - The worktree's CarGeometry has m_c 151.1467 / 185.3533 kg and t 1.6 / 1.6 m, equal to `CarConfig`.
- **Interface use:**
  - The C2a code imports only `ocd_lap.car_setup.KNOB`, `ocd_lap.track.closed_path` and `ocd_lap.track.ROOT`; the last is used to read the vendored TUM files, read-only.
  - No monkeypatching or `setattr` on ocd_lap (grep).
  - A C2a run loads only `ocd_lap`, `ocd_lap.car_setup` and `ocd_lap.track`, and the last two are covered by `code_version`.
- **Unmodified:** `git status --short` is empty in both `ocd_first_lap` and `ocd_first_lap-f1`, before and after my runs.
- **QA gate:** I did not run `./scripts/qa_gate.sh`. It was not requested, and nothing touched ocd_first_lap.
- **Provenance:**
  - `summary.json` records: the ocd_first_lap commit `3945634` (dirty False) and `ocd_lap_code_version` `70c48ed08c49`; this repo's commit `e616f94` (dirty True, plus a diff sha256); the style-rl commit `ea37559` with source hashes; the sha256 of every input file and geometry file; and the full config and map_config.
  - There is one gap; see item 6.

## 4. Style output sanity: PASS, with notes

Values are median [p1, p99] over the 849 laps, computed directly from `lap_style`.

| Number | Measured | Review | |
|---|---|---|---|
| B_mean_active | 0.633 [0.352, 0.945] | 0.63 [0.35, 0.95] | ok |
| B_event_peak_mean | 0.884 [0.523, 1.00] | 0.88 [0.52, 1.00] | ok; 21.4 % of laps at the ceiling (review 19 %) |
| B_apply_rate | 5.83 [1.51, 18.2] | 5.8 [1.5, 18.2] | ok |
| B_release_rate | 1.34 [0.454, 4.74] | 1.35 [0.46, 4.7] | ok |
| T_apply_rate | 0.850 [0.506, 4.58] | 0.85 [0.51, 4.6] | ok |
| T_release_rate | 2.63 [1.27, 4.41] | 2.6 [1.3, 4.4] | ok |
| S_reversals_5deg_per_km | 6.17 [2.94, **27.8**] | 6.2 [2.9, 26.6] | p99 is 4.5 % higher |
| S_mean_abs_rate | 18.5 [11.5, 37.0] | 18.5 [11.6, 37.0] | ok |
| S_corner_peak_angle | 49.0 [33.0, 73.6] (n 753) | 46.9 [30.8, 66.1] | different lap set (see below) |
| L_* | 2.56 / 1.03 / 5.39, p99 ≈ 5.82 | 2.20 / 0.96 / 3.52 | placeholder-width artefact; held out |

- **S_corner_peak_angle:** on the 381 laps with a TUM raceline it is 48.8 [33.1, 67.6]. The review's figure came from 556 laps on 16 TUM files, including the 5 that now fail, so the two are not directly comparable.
- **L_*:** on the bacinger centre lines these sit at the 5.823 m half width.
- **Sampling artefacts:**
  - The largest |corr(style, median spacing)| is **0.146** (S_mean_abs_rate), against "≤ 0.11" in the review. That is still below the 0.2 risk bar.
  - Median spacing of the accepted laps is 1–8 m.
- **Steering sign:** corr(steer, κ) < 0 on all 753 gated laps (median −0.896, maximum −0.529), so game steer + = right turn.

**Plots** (in `scratch/c2a/`):

- `scratch/c2a/align_spa.png`
  - Spa, TUM raceline; shift +2 m, scale 0.9993.
  - Every steering peak sits on a −κ peak, including La Source, Eau Rouge, Les Combes, Pouhon and the Bus Stop.
  - My brute-force corr on raw samples: median |corr| 0.906 (gate 0.906); largest per-lap difference from the stored value 0.016; all 28 laps negative.
  - **Looks right.**
- `scratch/c2a/align_singapore.png`
  - Singapore, bacinger centre line; this is the file whose direction was reversed. Shift +2 m, scale 1.0033.
  - Peaks align along the whole lap: corr −0.900 after alignment against −0.873 on raw distance.
  - Brute force: median |corr| 0.900 (gate 0.891); largest per-lap difference 0.079; all 38 laps negative.
  - **Looks right.** The traced line is visibly smoother than the steering trace, which is expected.
- `scratch/c2a/pedals_spa_lap99.png`
  - Spa lap #99: lap time 103.23 s, reconstructed t_end 103.11 s, 25 Hz uniform, 3 m raw spacing.
  - 7 brake zones, each starting 55–130 m before the next corner apex: La Source, Les Combes, Bruxelles, Pouhon (a light 0.8 lift), Fagnes, Stavelot and the Bus Stop. There is no brake through Eau Rouge or Blanchimont, and the throttle is flat on the straights.
  - Recomputing `style_vector` for this lap reproduces the cached 8 B/T/S numbers exactly.
  - **Looks right.** The brake input is near on/off (peak 1.0), consistent with the B_event_peak_mean ceiling.

**Tracks without geometry:** rows are kept, with NaN geometry numbers and reason codes.

| Track | Laps with numbers | Geometry numbers | B/T/S numbers | Reason | Rows A kept | Rows S kept |
|---|---|---|---|---|---|---|
| Imola | 18 | all NaN | all finite | `gate_low_corr` | 569 | 30 |
| Las Vegas | 41 | all NaN | all finite | `gate_low_corr` | 763 | 82 |
| Mexico | 37 | all NaN | all finite | `gate_length` | 747 | 61 |

For each track the cell geometry reason is the same code on every S row.

## 5. Privacy: PASS (0 hits)

- **Name list:** 2,824 distinct `user` values from every setups/laps/map/meta CSV in `setup_data/`, held in memory and never printed. The coder counted 2,948, from a different file set.
- **Search:** whole-token search at any length, plus substring search for names of 6 or more characters. Inside `.npz` files I searched every string array and the array names.

| Location | Files / arrays | Hits |
|---|---|---|
| `runs/critic_c2_data/` (all three runs) | 48 | **0** |
| `data/tracks_extra/` (CSV, JSON, raw GeoJSON, LICENSE) | 43 | **0** |
| `runs/style_cache/` (extra check) | 1,698 | **0** |

`setup_data/`, `runs/` and `data/tracks_extra/raw/` are gitignored (`git check-ignore -v`).

## Findings to fix or note (none blocking)

6. **Provenance gap (minor).**
   - The m_c and t constants in `rlsetup/critic/config.py:45-55` come from the `ocd_first_lap-f1` worktree (`ee86683`). `summary.json` records only the main ocd_first_lap tree (`3945634`).
   - Rule: runs must record where their ocd_first_lap inputs came from.
   - Fix: add the f1 worktree commit (or a hash of the CarGeometry values it was checked against) to the `provenance` block in `scripts/critic_c2_data.py:166-170`.
7. **Coder report slip (cosmetic).**
   - `docs/handoff/C2a_coder_report.md` gives Imola as "FAIL corr, length" with dL −1.00 (−1.01) %.
   - The code computes length_rel = −0.99986 % (L_geo 4,958.6 m), so `length_ok` is True and Imola fails on corr only (0.840). The outcome does not change.
8. **Sidecars not byte-reproducible (minor).** `fetch_track_geometry.py` writes `created_utc` into each `data/tracks_extra/*.json`, so a re-run changes all 14 files although the content is the same. If byte-stable files matter, drop the field or write only when the content changes.
9. **Observation for Ibrahim, not a defect.** The critic's 15 knobs (no damping knobs, LSD locks included) are not the 16-knob list in PROJECT_CONTEXT §5. That list is OPEN item §11.1, so this is his call.

## Files
- Logs:
  - `scratch/c2a/fetch_offline.log`
  - `scratch/c2a/data_run1.log`
  - `scratch/c2a/data_run2.log`
  - `scratch/c2a/indep_recount.log`
- Check scripts:
  - `scratch/c2a/indep_recount.py`
  - `scratch/c2a/hash_npz.py`
  - `scratch/c2a/plots.py`
  - `scratch/c2a/privacy.py`
- Outputs:
  - `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/runs/critic_c2_data/20261115_220208Z/`
  - `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/runs/critic_c2_data/20261115_220223Z/`
