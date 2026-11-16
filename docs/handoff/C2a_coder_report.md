# C2a coder report (foundations: data, knob map, style extraction, geometry gate)

Status: **done** (15 Nov 2026). No model, training or evaluation. Nothing committed.
Specs:
- `docs/specs/C2_style_critic.md`;
- `docs/specs/C2_review.md` (R1–R13, wins where the two differ).

## Built

### Setups and loading (previous coder, reviewed and kept)
- `.gitignore`: `setup_data/` and `data/tracks_extra/raw/`.
- C1 code removed. It was never committed, so git shows no deletions. Both C1 specs are marked superseded.

`rlsetup/critic/config.py`:
- Config dataclasses.
- Lap/gate reason codes; added `LAP_NO_GRAPH`.
- Added `ExtraGeometryConfig`: bacinger repo, pinned commit, PLACEHOLDER width.
- Added `StyleTransformConfig.iqr_to_sd`.

`rlsetup/critic/knobs.py` (source-independent):
- 15 OCD knobs, bounds, normalisation;
- frequency mediator, differentiable K_b;
- support box, setup hash.

`rlsetup/critic/sources/base.py`: adapter contract and `StyleInput`.

`rlsetup/critic/sources/f1_game_map.py`:
- R1 constants. All are "fitted to the OCD box, not measured".
- Box check with a per-game stop rule.
- New: `TRACK_GEOMETRY` (25 game tracks → geometry stem) and `BACINGER_IDS` (14).
- Toe and ride-height labels (item 5, below).

`rlsetup/critic/sources/f1laps.py`:
- Loader for folders, zips read in place and bare setups CSVs.
- Dedup by `setup_id` / `lap_id` and by identical file bytes.
- Usernames → integer ids keyed by (game, user).
- S2 pipeline, rows A and S, ranking stats.
- `lap_style_input` (S3 + R3).

New in this pass, in `f1laps.py`:
- graph `unit` guard (`fraction` / `source_normalized`, anything else stops the load);
- median spacing recorded even when the time map is missing.

### Style port (previous coder, reviewed and kept)
`rlsetup/style/`: numpy port of style-rl `dstyle` at `ea37559005b4758e0c03bc3ced2359bab9894781`.
- I diffed it against `git show` of the pinned files: only the DataFrame lines differ.
- The sha256 values in `rlsetup/style/__init__.py` match the pinned files.
- The test also compares numbers: exact equality, see Tests.

### New modules
`rlsetup/critic/geometry.py`:
- TUM files → centre/left/right (edges = centre ± width × left normal).
- Candidates: `data/tracks_extra/<stem>.csv` first, then the TUM file.

R2 gate in `geometry.py`:
- progress = (shift + scale·d) mod L_geo.
- Shift: exact circular cross-correlation by FFT on style-rl's ~2 m κ grid.
- Scale: grid of ±1 % around L_geo/L_game.
- Objective: median over laps of |corr|.
- Accept only if median |corr| ≥ 0.85 and |L_game/L_geo − 1| ≤ 1 %.
- Stored per candidate: shift, scale, corr, both pass flags, per-lap signed corr, and the best shift at scale 1 (report only).
- `scale_scan` is report-only.

Placeholder line, in `geometry.py`:
- TUM raceline where the chosen geometry is TUM, else the centre line.
- `is_placeholder` is always True; `line_kind` says which line.

`rlsetup/critic/style_data.py`:
- Per-lap style:
  - gated track: placeholder line, `q=` passed to `style_vector`;
  - ungated track: B/T/S rate numbers on a stub track, with `S_corner_peak_angle` and `L_*` set to NaN plus a reason code.
- Per-lap JSON cache under `runs/style_cache/`. The key combines:
  - raw-sample digest and lap time;
  - code digest;
  - geometry digest and alignment;
  - config and the style-rl commit.
- Cells: (game, driver, track) median.
- `StyleTransform` (R4):
  - log of the six rate numbers, floored at 1e-3;
  - robust z = (x − median)/(IQR/1.349), from training cells only;
  - clip ±3;
  - `L_*` held in a masked slot (written as 0, `mask` False);
  - NaN inputs stay NaN.
- `track_feature_table`: z (C1-R15) from gated centre lines only; NaN rows otherwise.

### Scripts and data
`scripts/fetch_track_geometry.py`:
- Downloads each GeoJSON into its own `data/tracks_extra/raw/bacinger/<id>/`: size cap, JSON-only parsing, sha256 recorded. LICENSE.md is fetched too.
- Projection: ENU tangent plane on WGS84 (exact ellipsoid formulas, no pyproj).
- Closes the loop.
- Periodic smoothing spline with s = n·(2 m)², resampled at 5 m.
- Direction: both orientations go through the R2 gate; the larger median |corr| wins.
- Start: index 0 = game distance 0.
- Width: one PLACEHOLDER constant, `width_assumed=true`.
- Writes `data/tracks_extra/<stem>.csv` + `<stem>.json` sidecar, then prints the gate table.
- `--offline` re-converts from the raw files on disk.

`scripts/critic_c2_data.py`:
- Full build from `setup_data/*`.
- Prints the row-count log, gate table, free-scale scans, style summary, z table and measured vs review.
- Writes `runs/critic_c2_data/<UTC>/{dataset.npz, summary.json}`. summary.json holds:
  - sha256 of every input file;
  - geometry sha256;
  - style-rl commit and source hashes;
  - `run_provenance()`;
  - full config.

`data/tracks_extra/`: 14 converted tracks (CSV + JSON). `raw/` is gitignored and `git check-ignore` confirms it.

`pyproject.toml`: optional `dev = ["pandas"]`, used only by the port-equality test. `rlsetup` never imports pandas (grep-checked).

### Tests and READMEs
New tests:
- `tests/c2_fixture.py`: synthetic export with 3 tracks and 2 games (folder, zip, bare CSV), plus geometry; usernames `drv_*`.
- `tests/test_c2_data.py`, `test_c2_knobs.py`, `test_c2_style_input.py`, `test_c2_geometry.py`, `test_c2_style.py`.

READMEs: `scripts/README.md`, `tests/README.md`, `data/README.md`.

## How to run
```
.venv/bin/python scripts/fetch_track_geometry.py          # download + convert + gate table (~30 s; --offline to reuse raw files)
.venv/bin/python scripts/critic_c2_data.py                 # full build, ~15 s (cache) / ~16 s (--no-cache)
.venv/bin/python -m pytest -q tests/test_c2_*.py           # C2a tests, ~1.5 s
PYTHONPATH=<ocd_first_lap-f1> .venv/bin/python -m pytest -q tests/test_c2_knobs.py   # + CarConfig vs sf70h_f1
```

## Measured vs review Evidence (`scripts/critic_c2_data.py`, setup_data as of 15 Nov)

| Count | Review | Measured |
|---|---|---|
| dry setups (unique `setup_id`) | 21,704 | 21,704 |
| after gap ≤ 8 % | 21,523 | 21,523 (F1 25 12,948, F1 26 8,575) |
| dropped by the box | 246 | 246 (1.14 %; F1 25 0.32 %, F1 26 2.38 %; χ 245, K_φ 1) |
| rows A | 21,277 | 21,277 |
| rows S before filters | 857 + 877 | 857 + 877 |
| rows S after filters | 1,697 | 1,697 (856 + 841) |
| exact telemetry rows | 613 | 613 |
| A pairs / drivers (cell × session) | 3,029 / 284 | 3,029 / 284 |
| laps with style | 854 | **849** |
| QC rejections | 13 | **10** |
| rows S with a styled lap | 1,686 | **1,680** |

The three differences are one cause: R3 now applies the spacing rule. The review's 854 was measured before that rule, as 859 laps with samples minus 4 time-closure failures minus 1 lap without a time map.

The review's "13" adds 4 closure failures and 9 spacing failures. These overlap:
- 3 laps fail both rules;
- 1 of the 9 spacing laps has no time map (median spacing 110 m).

The union is therefore 10 laps:
- 4 `lap_time_closure`;
- 5 `lap_spacing`;
- 1 `lap_no_samples` (no time map).

There are also 9 laps without graph samples (`lap_no_graph`), not counted as QC.

The 5 spacing-only laps take the style away from 5 cells, and those cells hold 6 S rows: 1,686 − 6 = 1,680. I verified this from the saved arrays.

Other facts:
- Steering sign: on all 753 laps with gated geometry, corr(steer, κ) < 0, so game steer + = right turn.
- Spacing: 1–11 m per lap, with outliers at 30.5, 48 and 110 m.

## Geometry gate (median |corr| ≥ 0.85 and |dL| ≤ 1 %)

| Game track | Candidate | Laps | L_game | L_geo | dL % | Scale | Shift m | \|corr\| | \|corr\| at scale 1 | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| abudhabi | bacinger | 37 | 5279 | 5303 | −0.44 | 0.9994 | +8 | 0.909 | 0.910 | PASS (used) |
| abudhabi | TUM YasMarina | 37 | 5279 | 5547 | −4.82 | 1.0612 | −34 | 0.581 | 0.360 | FAIL corr, length |
| australia | bacinger | 30 | 5274 | 5282 | −0.16 | 0.9965 | −2 | 0.861 | 0.847 | PASS (used) |
| australia | TUM Melbourne | 30 | 5274 | 5299 | −0.48 | 1.0048 | +4 | 0.768 | 0.747 | FAIL corr |
| austria | TUM | 36 | 4322 | 4315 | +0.15 | 0.9985 | −6 | 0.890 | 0.887 | PASS |
| azerbaijan | bacinger Baku | 38 | 5991 | 6017 | −0.44 | 0.9994 | +0 | 0.855 | 0.855 | PASS |
| bahrain | TUM | 39 | 5406 | 5406 | +0.00 | 0.9975 | +6 | 0.888 | 0.886 | PASS |
| brazil | TUM | 36 | 4292 | 4305 | −0.29 | 1.0029 | −18 | 0.930 | 0.927 | PASS |
| canada | TUM | 34 | 4370 | 4358 | +0.28 | 0.9973 | +4 | 0.904 | 0.902 | PASS |
| china | TUM | 32 | 5440 | 5445 | −0.10 | 0.9985 | +16 | 0.891 | 0.891 | PASS |
| hungary | TUM | 34 | 4376 | 4377 | −0.02 | 1.0002 | +18 | 0.909 | 0.909 | PASS |
| imola | bacinger | 18 | 4909 | 4959 | −0.99986 | 1.0101 | +0 | **0.840** | 0.796 | **FAIL corr** (corrected in the fix round: the length passes) |
| japan | TUM | 37 | 5808 | 5803 | +0.09 | 0.9966 | +6 | 0.912 | 0.903 | PASS |
| las_vegas | bacinger | 41 | 6198 | 6260 | −0.98 | 1.0024 | +0 | **0.817** | 0.815 | **FAIL corr** |
| madrid | bacinger | 18 | 5410 | 5449 | −0.72 | 1.0048 | +0 | 0.895 | 0.865 | PASS |
| mexico | bacinger | 37 | 4523 | 4310 | **+4.95** | 0.9576 | +2 | 0.893 | 0.366 | **FAIL length** |
| mexico | TUM | 37 | 4523 | 4297 | **+5.25** | 0.9525 | −238 | 0.891 | 0.371 | **FAIL length** |
| miami | bacinger | 36 | 5409 | 5439 | −0.54 | 1.0030 | +0 | 0.893 | 0.886 | PASS |
| monaco | bacinger | 36 | 3322 | 3299 | +0.69 | 0.9857 | +0 | 0.918 | 0.837 | PASS |
| monza | TUM | 32 | 5795 | 5790 | +0.08 | 0.9992 | +2 | 0.880 | 0.878 | PASS |
| netherlands | bacinger | 36 | 4257 | 4261 | −0.09 | 0.9984 | +0 | 0.938 | 0.936 | PASS (used) |
| netherlands | TUM Zandvoort | 36 | 4257 | 4316 | −1.38 | 1.0140 | +46 | 0.910 | 0.865 | FAIL length |
| qatar | bacinger Lusail | 34 | 5415 | 5451 | −0.66 | 1.0041 | −2 | 0.915 | 0.913 | PASS |
| saudi_arabia | bacinger Jeddah | 38 | 6174 | 6207 | −0.53 | 1.0028 | +0 | 0.887 | 0.879 | PASS |
| silverstone | TUM | 32 | 5889 | 5887 | +0.04 | 0.9971 | +0 | 0.886 | 0.884 | PASS |
| singapore | bacinger | 38 | 4927 | 4943 | −0.33 | 1.0033 | +2 | 0.891 | 0.884 | PASS |
| spa | TUM | 28 | 7005 | 7000 | +0.07 | 0.9993 | +2 | 0.906 | 0.903 | PASS |
| spain | bacinger | 31 | 4661 | 4680 | −0.41 | 1.0067 | +2 | 0.914 | 0.909 | PASS (used) |
| spain | TUM Catalunya | 31 | 4661 | 4650 | +0.24 | 0.9926 | +14 | 0.798 | 0.786 | FAIL corr |
| usa | TUM | 41 | 5513 | 5508 | +0.10 | 0.9990 | +6 | 0.865 | 0.865 | PASS |

**Result:** 22 of 25 tracks have gated geometry; 3 do not (Imola, Las Vegas, Mexico).
- The TUM-only part reproduces the review's table: 11 pass; the 5 named fail with the same numbers.
- The review gave Mexico 0.90 after a −230 m shift; here it is 0.891 at −238 m and scale 0.9525.
- 12 of the 14 bacinger conversions pass.
- The 3 tracks without geometry keep their rows (Decision 7):
  - 96 laps have B/T/S rate numbers and NaN geometry numbers;
  - reason codes `gate_low_corr` (59 laps) and `gate_length` (37 laps).

**Mexico** (report only, gate not relaxed):
- With the gate's fitted scale, both geometries clear the correlation bar: TUM 0.891, bacinger 0.893.
- A free scale scan from 0.90 to 1.10 peaks exactly at the closed-lap ratio: 0.9525 TUM / 0.9575 bacinger, against L_geo/L_game 0.950 / 0.953. At scale 1 the corr is only 0.37.
- So the game's Mexico distance channel is a uniform ~5 % stretch of the real 4.30 km layout. It is not a different layout and not a local insertion.
- Both geometries fail only the length test.

**Madrid**:
- bacinger measured 5,450 m;
- game lap 5,410 m;
- official 5,414 m;
- dL −0.72 %, |corr| 0.895: passes.

**Lusail**:
- measured 5,451 m;
- game 5,415 m;
- official 5,419 m;
- −0.66 %, 0.915: passes.

**Length bias of the converted lines.** The bacinger lines run +0.25 to +1.02 % longer than the official lengths (Monaco is the exception at −1.09 %). Likely cause: lateral jitter in the hand-traced vertices (my reading). Against the game lap this puts Imola (dL −0.99986 %, passes) and Las Vegas (−0.98 %, passes) just inside the 1 % bar; both fail on corr only.

**Direction:** bacinger's Singapore file was reversed and the direction check fixed it. Every other file was already in driving direction. As-file vs reversed corr is 0.82–0.94 vs 0.33–0.60, so the decision is clear on every track.

**Width:** one constant per side, 5.823 m. That is half of 11.646 m, the median over the 25 vendored TUM tracks of each track's median total width. It is PLACEHOLDER, provisional pending Ibrahim, with `width_assumed=true` in every sidecar. On these tracks the placeholder is the centre line, so `L_border_mean_m`, `L_inside_dist_apex_m` and `L_outside_dist_exit_m` sit at the half width (p99 5.82). These are artefacts of the assumption, consistent with L_* being held out.

## Style summary (849 laps; style-rl units; steer × 115)

| Number | n | Median | p1 | p99 |
|---|---|---|---|---|
| B_mean_active | 849 | 0.633 | 0.352 | 0.945 |
| B_event_peak_mean | 849 | 0.884 | 0.523 | 1.00 |
| B_apply_rate | 849 | 5.83 | 1.51 | 18.2 |
| B_release_rate | 849 | 1.34 | 0.454 | 4.74 |
| T_apply_rate | 849 | 0.85 | 0.506 | 4.58 |
| T_release_rate | 849 | 2.63 | 1.27 | 4.41 |
| S_reversals_5deg_per_km | 849 | 6.17 | 2.94 | 27.8 |
| S_mean_abs_rate | 849 | 18.5 | 11.5 | 37.0 |
| S_corner_peak_angle | 753 | 49.0 | 33.0 | 73.6 |
| L_border_mean_m | 753 | 2.56 | 1.82 | 5.82 |
| L_inside_dist_apex_m | 753 | 1.03 | 0.859 | 5.81 |
| L_outside_dist_exit_m | 753 | 5.39 | 2.46 | 5.82 |

- The B/T/S rows match the review's table.
- The geometry rows differ from the review because they now cover the 22 gated tracks, including the bacinger centre lines with assumed width, instead of the 16 ungated TUM tracks.

## Item 5 (toe, ride height)

Toe:
- The F1 25 menu labels are "Front Toe-Out" / "Rear Toe-In".
- OCD's `toe_*_deg` is "positive = toe-out", applied per wheel (`car_setup.py`, `toe_out_rad.{wheel}`).
- The C1 convention is kept: front +value, rear −value.
- Marked "label-supported; per-wheel vs total unverified" in `f1_game_map.py`.

Ride height: 1 click = 1 mm is still marked ASSUMED.

## Tests
- C2a tests: `50 passed, 1 skipped in 1.46s`.
- The skip is CarConfig vs the sf70h_f1 worktree; it passes with `PYTHONPATH=<ocd_first_lap-f1>`.
- Full suite (`.venv/bin/python -m pytest -q`): `2 failed, 473 passed, 14 skipped, 1 warning in 182.79s`.
  - The same 2 tests failed before any C2a change (baseline `2 failed, 423 passed, 13 skipped`).
  - C2a adds 50 passed and 1 skipped.
  - The provenance failure set is unchanged: `{'ocd_lap/cars.py'}`.

Coverage:
- multi-folder + zip + bare CSV loading and dedup;
- per-game gap (F1 26 fixture laps are 5 % slower);
- knob known values, K_b round trip ≥ 0, box drop and stop rule;
- time reconstruction (exact at constant speed, including before the first key);
- 25 Hz resample, steering × 115, QC codes;
- FFT corr equal to brute force;
- gate accept, reject on length, reject on corr, no laps; shift recovered;
- candidate order, placeholder line (raceline lateral 2 m, centre line 0);
- every TUM raceline lies inside its edges;
- width constant = TUM median / 2;
- ENU projection against WGS84 radii; JSON-only parsing; periodic spline;
- port equality with the pinned original (exact, via `git show` + pandas; skipped with a reason if either is missing; checks the pinned sha256);
- placeholder flag; ungated rows kept with NaN geometry numbers; cells;
- R4 transform definition and fitted on training cells only;
- determinism and cache; no fixture username in any output of the data script.

Real-data checks:
- Two builds (cache and `--no-cache`) give bit-identical arrays (44) and stdout. A third run takes 849/849 laps from the cache, also identical.
- No username appears in stdout, summary.json, npz strings, the style cache or `data/tracks_extra/` (whole-token check against 2,948 names from the exports, ≥ 4 characters).

## Deviations and open points (not decided here)
1. **Placeholder line where TUM has no raceline: the centre line** (lead's instruction). The review's [ARCH] item preferred a minimum-curvature line, with the centre line as a flagged last resort. Low stakes because L_* are held out; `line_kind` records it.
2. **Robust z uses IQR/1.349** (SD-equivalent), so the ±3 clip means 3 SD. The review says "median and IQR" without a factor. `StyleTransformConfig.iqr_to_sd = 1.0` gives raw IQR. Architect to confirm.
3. **Style cache key** uses a digest of the lap's raw samples instead of `lap_id`. The loader discards source ids for privacy, and the digest is stricter.
4. **Gate scale grid** is ±1 % around L_geo/L_game, and the gate takes the best median |corr| over (scale, shift). The scale-1 result is stored for reference.
5. **Bacinger loops start at game distance 0** (fitted shift), not at an OSM/FastF1 start node.
6. **Pre-existing failures outside C2a** (also in the baseline run before any C2a change): `test_provenance::test_every_imported_ocd_lap_module_is_hashed` and `test_qss_profile::test_cached_file_matches_fresh_computation`. Not fixed (outside C2a).
   - The provenance test lists `ocd_lap/cars.py` as unhashed, and C2 also loads it. A one-line fix would add `'cars.py'` to `OCD_LAP_EXTRA_SOURCES`, but that changes driving-run provenance.
   - The QSS test fails on a stale tracked cache: the cache key no longer matches the current ocd_first_lap build. It would need `scripts/build_qss_profile.py`.
7. For Ibrahim:
   - the width constant (provisional);
   - Mexico has no geometry: the game distance is stretched ~5 %;
   - Imola and Las Vegas fail;
   - a fitted-scale exception for Mexico or better geometry sources (OSM / UDP recording, c2_track_geometry §4) are his call.

## Fix round (QA M1–M16, T1–T9, I1/I2; sim-runner items 6–8)

Behaviour on the real data is unchanged: every array of `dataset.npz` that existed before is bit-identical to the pre-fix run (fresh run and cache-read run both checked), and all review counts are the same (10 of 13 equal, the same 3 differences as before: laps with style 849 vs 854, QC rejections 10 vs 13, rows S with a styled lap 1,680 vs 1,686, all from the R3 spacing rule). New outputs only.

### Code changes
| Item | Change | Where |
|---|---|---|
| M1 | Telemetry (each graph channel, the time map) comes from the first export that has it for a lap. A lap listed earlier without samples is filled from a later export; its laps.csv ids are read even when the bytes duplicate an earlier file. New count "laps given telemetry by a later export" (0 on the real data). | `sources/f1laps.py::load_exports` |
| M2 | The gate checks the sign of the median signed corr against the source's declared `steer_kappa_sign` (F1: −1). Reason order: corr → sign → length (`GATE_SIGN = 'gate_sign'`). All 22 gated tracks keep passing (median signed corr −0.855 to −0.938). | `geometry.gate`, `config.GATE_SIGN`, `SourceAdapter.steer_kappa_sign` |
| M3 | `GateResult.lap_index` = input positions of the laps used. `lap_styles` maps corr through it and asserts that every gated lap has its corr (no zip). | `geometry.gate`, `style_data.lap_styles` |
| M4 | Cache in `runs/style_cache/<cache_env>/`. `cache_env` = sha256 over the code digest (now including the adapter's own source file), numpy and scipy versions, and the style-rl commit. `prune_cache` deletes other 16-hex env directories and old flat `<sha256>.json` files only (1,698 flat files removed on the first run). | `style_data.cache_env`, `prune_cache` |
| M5 | `_increasing` returns (kept, dropped). `StyleInput.dropped_ties` per channel and time map; `LapStyles.n_dropped_ties`, npz `lap_n_dropped_ties`. Measured: 295 of 859 laps have steer ties (40,897 samples); 78 lose > 1 % (64 F1 25), 42 lose > 10 % (all F1 25), max 30.5 %; 0 time-map ties. Behaviour unchanged (first kept). | `sources/f1laps.py`, `sources/base.py` |
| M6 | Dead `feature_table` deleted. | `track_features.py` |
| M7 | `build_frame` takes knobs from `adapter.to_knobs` and session/controller/assists from `adapter.meta` (raw columns `META_COLUMNS` kept per setup as `meta_raw`); asserts `meta['source']` equals the game. | `sources/f1laps.build_frame` |
| M9/M10 | Not implemented (C2b). API allows it: `dataset.gaps(lap_s, group, reference=mask)` (NaN for groups without reference rows); `Frame.lap_s` kept; StyleTransform passes NaN through (unchanged). | `dataset.py` |
| M11 | summary.json provenance gets `libraries` (python 3.12.13, numpy 2.5.0, scipy 1.18.0) and `placeholder_line_sha256` per candidate (16 of 30 have a TUM raceline). The gate table carries `line_sha256`. | `critic/provenance.py`, `scripts/critic_c2_data.py` |
| M12 | New `critic/dataset.py` with `Frame`, `gaps`, `cells_of`, `ranking_stats`, and the `LapLike`/`LapSource` Protocols. SESSIONS/CONTROLLERS moved out of config into the F1 adapter; `Frame` carries `sessions`, `controllers`, `assist_levels` and `source_row`. `SourceAdapter` declares the vocabularies and `steer_kappa_sign`. `lap_styles(..., adapter=)` is required (no F1 default). | `dataset.py`, `sources/base.py`, `config.py`, `style_data.py` |
| M13 | The fetch script stops when a raw file's sha256 differs from the existing sidecar's `source.sha256`, unless `--update`. Checked by tampering a sidecar: it stops with "Monaco: raw mc-1929.geojson sha256 daf184fc7d09 != sidecar 000000000000"; the original was restored, byte-identical. | `scripts/fetch_track_geometry.py` |
| M14 | `data/tracks_extra/LICENSE-bacinger-f1-circuits.md` is a verbatim copy of the upstream LICENSE.md (cmp identical); sidecars name it under `licence.copy`. | same |
| M15 | Docstring now points to `tests/test_c2_style.py`. | `rlsetup/style/__init__.py` |
| M16 | `--out` defaults to `REPO_ROOT/runs/critic_c2_data` and `--sources` to `REPO_ROOT/setup_data/*` (both scripts). `--no-cache` help: "neither read nor written" (test checks nothing is written). | scripts |
| sim 6 | `provenance.car_constants` = CarConfig values + sha256 + git state of the ocd_first_lap worktree on branch `sf70h-f1`, found via `git worktree list` (commit ee86683, clean). The worktree path is not stored because it contains the account name. | `critic/provenance.py` |
| sim 7 | Imola corrected above: fails corr only (dL −0.99986 % passes). | this report |
| sim 8 | Sidecars have no `created_utc`. Re-running `--offline` gives byte-identical CSVs (all 14) and sidecars; the only sidecar diff from before is the added `licence.copy`. | fetch script |
| I1/I2 | Gate table and summary carry `closure_gap_m` = scale·L_game − L_geo and `group_median_abs_corr` per game. | `GateResult`, both scripts |

### Tests added (T1–T9 and supporting)
- T1 `test_c2_data.py::test_lap_samples_from_a_later_export`
- T2 `test_c2_style_input.py::test_distance_ties_keep_first_and_are_counted` (the specified array plus a lap-level count on StyleInput)
- T3 `test_c2_knobs.py::test_stop_rule_is_per_game`
- T4 `test_c2_data.py::test_ranking_pairs_within_session_only` (2 pairs; equal hash → 1; equal gap → 1; mask)
- T5 `test_c2_geometry.py::test_gate_rejects_mirror_image`
- T6 `test_c2_style.py::test_track_scaler_uses_training_tracks_only`
- T7 `test_c2_data.py::test_adapter_contract`
- T8 `test_no_username_in_any_output` now also scans the cache JSONs (rglob)
- T9 `test_c2_style.py::test_lap_corr_alignment`
- Supporting: `test_c2_geometry.py::test_gate_reports_lap_index_closure_and_groups`, `test_c2_style.py::test_cache_prune_removes_stale_entries_only`; `test_determinism_and_cache` now checks the env sub-directory and that `use_cache=False` writes nothing.

### Lead's addition (a): rows without track z (report only)
- z has 15 NaN entries (3 tracks × 5). They affect 2,079 of 21,277 rows A and 173 of 1,697 rows S.
- z the failing candidates would give, against the 22 gated tracks' range:

| | n_corners | longest straight m | straight share | slow share | fast share |
|---|---|---|---|---|---|
| gated min | 10 | 250.9 | 0.248 | 0.023 | 0.092 |
| gated max | 24 | 1439.0 | 0.719 | 0.102 | 0.401 |
| Imola bacinger (corr) | 14 | 1178.7 | 0.526 | 0.026 | 0.286 |
| Las Vegas bacinger (corr) | 14 | **1781.8** (outside) | 0.656 | 0.037 | 0.178 |
| Mexico bacinger (length) | 16 | 1197.9 | 0.524 | 0.073 | 0.189 |
| Mexico TUM (length) | 17 | 1210.0 | 0.619 | 0.085 | 0.138 |

- Only Las Vegas' longest straight falls outside the gated range. Everything else is inside it. Nothing here is used.

### Lead's addition (b): fast tail of gap (report only)
- Quantiles (%):
  - min −8.94, p0.1 −7.08, p0.5 −5.66, p1 −5.03, p5 −3.33;
  - p50 0.00, p95 +3.41, p99 +5.64, max +8.00 (the one-sided filter).
- Rows below −3/−4/−5/−6/−7/−8 %: 1,366 / 586 / 224 / 70 / 23 / 7.
- Below −4 %: 586 rows.
  - By game: F1 25 365, F1 26 221.
  - By session: Quali 544, Race 41, TT 1.
- So the fast tail is almost entirely Quali rows. It spreads over all tracks: largest groups are F1 25 canada Quali 28 of 182, austria 27/186, qatar 27/128; each comes from many drivers.
- Race rows below −4 % are 1–3 per (game, track), usually from 1 driver, down to −8.5 %.
- Full table in summary.json `gap_fast_tail`.

### Test results (fix round)
- `.venv/bin/python -m pytest -q tests/test_c2_*.py`: `60 passed, 1 skipped in 1.61s`. The skip is the sf70h_f1 car-constant check, which needs the f1 worktree on PYTHONPATH.
- `.venv/bin/python -m pytest -q`: `2 failed, 505 passed, 14 skipped, 1 warning in 184.05s (0:03:04)`. The 2 failures are the same pre-existing ones as at baseline (`test_provenance::test_every_imported_ocd_lap_module_is_hashed`, `test_qss_profile::test_cached_file_matches_fresh_computation`).
- The suite now also collects `tests/test_ac_preflight.py`, which is not ours and was not touched.
