# C2a QA review: data, knob map, style extraction, geometry gate

**Verdict: PASS-WITH-FIXES.** No blockers and no majors. 16 minor findings: some are code or test fixes, some are hooks C2b needs. There are also 6 notes for the architect and Ibrahim (I1–I6).

Read first: PROJECT_CONTEXT, DECISIONS, `docs/specs/C2_style_critic.md`, `docs/specs/C2_review.md` (R1–R13 binding) and `docs/handoff/C2a_coder_report.md`. Nothing in the repo was edited. All throwaway scripts and outputs are in `scratch/qa_c2a/`.

## What I ran (exact output lines)

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -q tests/test_c2_*.py -rs` | `50 passed, 1 skipped in 1.44s`. Skip: `tests/test_c2_knobs.py:104: sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path` |
| Same file with `PYTHONPATH=…/ocd_first_lap-f1` | `9 passed in 0.01s`. The CarConfig m_c and t match the f1 worktree (verified). |
| Full suite `.venv/bin/python -m pytest -q` | `2 failed, 473 passed, 14 skipped, 1 warning in 188.33s (0:03:08)`. Failures: `test_provenance.py::test_every_imported_ocd_lap_module_is_hashed` and `test_qss_profile.py::test_cached_file_matches_fresh_computation` |
| Port equality test, `-v -k "port or pinned"` | `2 passed`. It runs here and does **not** skip: pandas 3.1.1 is in the venv and `~/Documents/style-rl` is on `style12-whole-lap` at HEAD `ea37559…`. |
| `git show ea37559:style_rl/dstyle/{features,track}.py \| shasum -a 256` | `bc16b00c…f1a` and `1af3ae81…491`. Both equal `STYLE_RL_SOURCES_SHA256`. |
| `diff` of the pinned originals vs `rlsetup/style/{features,track}.py` | Only the DataFrame lines, docstrings, the `Corner` namedtuple and `from_csv` differ. Every number-producing line is identical (details below). |
| `scripts/critic_c2_data.py --no-cache --out <scratch>` | rc=0. `box check: 246 of 21523 rows outside the OCD box (1.14 %)`, `F1 25: 42 of 12948 (0.32 %)`, `F1 26: 204 of 8575 (2.38 %)`, `tracks with gated geometry: 22 of 25`, steer sign `{'negative': 753, 'positive': 0}`. The measured-vs-review table matches the coder report: 10 counts same; 1680/849/10 differ for the stated R3-spacing reason. |
| My no-cache build vs the coder's cached run `20261115_220223Z` | `44 arrays; differing: []`: bit-identical (determinism verified). |
| `runs/style_cache` file count before and after my run | 1698 and 1698. A `--no-cache` run neither reads nor writes the cache (verified). |
| Username scan: 2,824 export usernames (2,802 with ≥ 4 characters), whole-token match against 1,762 files | Files scanned: `data/tracks_extra/*`, all of `runs/style_cache/*`, every `runs/critic_c2_data/*/{summary.json, npz strings}`, my build's stdout and summary, C2 code, tests, fixture, READMEs, C2 docs. **One hit, a false positive:** the English word "Example" in `docs/research/c2_track_geometry.md`, which is also a username. No username reaches any output (verified). |
| `git check-ignore -v` | `setup_data/` (.gitignore:29), `data/tracks_extra/raw/` (:30), `runs/` (:5) are all ignored. `data/tracks_extra/*.csv\|json` are not ignored, as intended. |

## 1. Correctness against the spec and review

R1 constants, `rlsetup/critic/sources/f1_game_map.py:71-89`. I checked every formula term by term against R1:
- f = 2.5 + 2.5(c−1)/40.
- K_b = 10,000 + 12,500(c−1).
- h = c + 5 mm; rake = c_R − c_F.
- aero_level = (fw+rw)/100.
- share = 0.35 + 0.25(s − 1/52)/(50/52), with s = (fw+1)/(fw+rw+2).
- Bias and both LSD locks are pct/100; camber is passed through; toe is front +, rear −.

All of these match R1. The frequency mediator (`knobs.py:27-35`) gives K_φ = Σ(K_b + ½K_w t²) and χ = front/K_φ. The test round-trips it exactly, so K_b ≥ 0 holds by construction (minimum 10 kN·m/rad).

Box check:
- Tolerance is 1e-9 (`knobs.py:38-41`); NaN counts as out of the box.
- The stop rule is applied per game (`f1_game_map.py:100-109`). I proved this with a throwaway check: game A 6/100 out, game B 0/1000 out is 0.55 % pooled, and it raises `{'A': 6.0} %`.
- The OCD bounds of the 15 knobs are identical in the main and f1 worktrees (verified).
- By the click ranges, only χ and K_φ can leave the box, which matches "χ 245, K_φ 1".

Gap filter (`f1laps.py:367-397`):
- The median is taken per (game, track).
- Rows with gap > 8 % are dropped, then the median and gap are recomputed once.
- The fixture test checks the final gap against the kept rows' median.

Rows A and S, exact rows and R6 grouping: the definitions match the review, and the counts reproduce (21,277 / 1,697 / 613 / 3,029 / 284).

R2 gate (`geometry.py:186-278`):
- I checked the FFT circular cross-correlation index algebra: shift k·ds ↔ progress = shift + scale·d. A brute-force test confirms it, including wrap-around.
- Bars are median |corr| ≥ 0.85 and |L_game/L_geo − 1| ≤ 1 %.
- The length test uses the closed-lap lengths, not the fitted scale, so the scale cannot game it.
- The same `gate()` is used for TUM and bacinger candidates (extra first, first pass wins).
- z is gated as well (`style_data.py:270-278`).
- The direction fix is in `fetch_track_geometry.py:133-144`.
- Shift, scale and corr are stored.

R3:
- (0, 0) is prepended.
- Closure > 1 % and median spacing > 8 m are rejected.
- Spacing and steer sign are stored.

Other style-input items:
- Resampling is 25 Hz linear (`base.py:38-42`) and steer is ×115. Both are tested.
- L_* sit in a masked slot written as 0. Placeholder flag and `line_kind` are stored.
- Cells are the nanmedian per (game, driver, track).

R4: log of the six rates, robust z fitted on the indices passed in, clip ±3.

**Judgement on IQR/1.349.** Acceptable; keep it configurable and have the architect confirm.
- It is the normal-consistent robust z, so ±3 means "3 SD", the same meaning as the TrackScaler's ±3. (Note: sklearn's RobustScaler uses raw IQR by default, so both readings exist.)
- I measured both choices on the 849 real cells:

| | Cells clipped, worst number | Other numbers |
|---|---|---|
| IQR/1.349 | 1.8 % (S_reversals) | 0.0–0.9 % |
| raw IQR | 0.5 % | ≤ 0.2 % |

- No cell sits at the 1e-3 log floor.
- So the choice is immaterial for clipping. It only rescales the inputs by 1.349, which the 9→2 encoder absorbs.

## 2. Port fidelity
Verified by `diff`.
- **features.py:** the only changes are pandas accessors → `np.asarray`, and `corners.itertuples()` → iterating a list. `style_table` now returns an array.
- **track.py:** the constructor takes arrays (CSV read moved to `from_csv`, plus a shape check). `find_corners` builds `Corner` tuples with a stable sort instead of pandas `sort_values`. The only behavioural difference: two corners with identical `s_apex` could come out in a different order, which changes nothing beyond float summation order. It also returns `[]` where the original raises.
- **Thresholds, constants and arithmetic** are byte-identical.
- **The equality test is real.** It compares exact equality on Monza, more than 5 corners, through all three paths (xy, `lap_dist` and `q`), and checks the pinned sha256. It ran here.

## 3. Privacy
- **Verified clean:** the token scan above found no username anywhere.
- **Ids:** usernames are turned into ids keyed by (game, user) and the strings are discarded (`f1laps.py:305-314`).
- **Cache:** it holds numbers only, keyed by a sample digest, not by lap_id.
- **Fixture:** uses `drv_*` names.
- **`data/tracks_extra/*.csv`:** ENU coordinates plus constant widths only.
- **`*.json` sidecars:** public URL, sha, MIT licence, the copyright line, centroid lon/lat and parameters. They also contain Ibrahim's first name in `width_method` ("pending Ibrahim"). That is harmless, and the docs already carry it.
- **Absolute paths:** `runs/` (gitignored) contains absolute paths with the macOS account name, in the gate rows' `path` and in `run_provenance`. This is not an f1laps username.

## 4. Fold safety (hooks for C2b)
These are fold-safe by API:
- `StyleTransform.fit(values, train_idx)`, which is also tested;
- `TrackScaler.fit(table, train_tracks)` (untested, M6);
- `support_box(p_train)`;
- `normalise`, which uses fixed KNOB bounds, not data.

These are fitted globally and C2b must decide on them (see M9, M10):
- the per-(game, track) median that defines `gap`, and the 8 % filter;
- the gate's shift/scale, pooled over all laps including held-out drivers' laps (input alignment, not label);
- the ungated-track stub length;
- driver ids, which are the enumeration of sorted (game, user). They change whenever a new export adds a user, so fold assignments must be stored with the data hash.

`Frame.lap_s`, `game` and `track` are kept, so C2b can recompute the gap per fold.

## 5. Determinism and numerical safety
- **Determinism:** verified bit-identical (above).
- **NaN path:** 96 laps on the 3 ungated tracks get B/T/S numbers on a stub track. The geometry numbers are NaN, and NaN passes through `StyleTransform` (M9).
- **Division by zero is guarded** in: `corr_over_shifts` (var > 0), `time_closure_rel`, `B_mean_active` (max 1e-9), empty brake zones (0.0, then the log floor), and `scale` (falls back to 1 when IQR = 0).
- **Time base:** strictly increasing on all OK laps. Real data has no negative distances (verified).
- **Monotonic filter:** `_increasing` drops 40,895 tied steer samples. They sit on 78 laps (64 F1 25) whose distance is quantised to 1 m; at most 31 % of one lap is dropped (M5).
- **Fetch script:** JSON-only parsing (`json.loads`, float coercion, range check), a 2 MB cap via `read(max+1)`, atomic `.part` → replace, a fixed path per id, nothing executed. On a network failure `urlopen` raises and the script stops. Tracks already written stay complete, because the licence is fetched before anything else.

## 6. Pre-existing failures: confirmed unrelated (reasoned from code, tree not modified)
- **`test_every_imported_ocd_lap_module_is_hashed`**
  - Fails in isolation: `1 failed in 0.72s`, `assert {'ocd_lap/cars.py'} == set()`.
  - The test imports only `rlsetup.driving.*`. After `import rlsetup.driving.env, rlsetup.driving.train`, the check printed `'ocd_lap.cars' in sys.modules` → True and `'rlsetup.critic' in sys.modules` → False.
  - `cars.py` is pulled in by `ocd_first_lap/ocd_lap/backend.py:9` (`from . import cars`). It is in neither `SOURCES` nor `OCD_LAP_EXTRA_SOURCES`.
- **`test_cached_file_matches_fresh_computation`**
  - Fails in isolation with `KeyError: 'provenance'` on the tracked `data/qss_profile_Shanghai_p0.npz`, plus the warning "cache key does not match the current code/track".
- **C2a touches neither cause.** `git status --porcelain rlsetup/ data/*.npz tests/conftest.py tests/test_qss_profile.py tests/test_provenance.py` shows only `?? rlsetup/critic/` and `?? rlsetup/style/`.

## 7. Adapter contract
- No F1 column name appears outside `sources/` (grep verified).
- `isinstance(F1LapsAdapter(), SourceAdapter)` is True.
- The coupling is structural, not by column name (M12): `Frame`, `build_frame` and `ranking_stats` live in `sources/f1laps.py`; `lap_styles` expects `data.laps[k].{track, game, driver, has_samples}` and defaults to `F1LapsAdapter`; `config.SESSIONS`/`CONTROLLERS` hold f1laps vocabulary (`'blank'`).

## Findings

| # | Sev | File:line | Failure scenario | Fix | Status |
|---|---|---|---|---|---|
| M1 | MINOR | `rlsetup/critic/sources/f1laps.py:278-279, 291-299` | Laps are deduplicated by lap_id with "first export wins", and graph/time files are read only for lap_ids new in that export. A later scrape that re-lists the same laps *with* telemetry, sorted after an export without graph files, loses the samples silently. Only the count shows it. Today the union over exports is 859 laps with samples and the loader also has 859, so nothing is lost now. | When a known lap has no channels or time map, fill them from the later export. Write the test T1. | Verified latent |
| M2 | MINOR | `rlsetup/critic/geometry.py:260, 275-277` | The gate uses \|corr\| only, so a mirror-image geometry passes. Proven: an x → −x copy of the fixture track passes with median \|corr\| 0.997 and signed corr **+0.996…+0.998**. An east/north swap in a future conversion would be accepted with left/right flipped, which affects L_* and the inside/outside choice. Real data: 753/753 laps negative. | Reject (or flag `gate_sign`) when the median signed corr has the wrong sign for the source's declared steer convention. Write the test T5. | Verified |
| M3 | MINOR | `rlsetup/critic/style_data.py:123-126` | `zip(lap_keys, lap_corr)` relies on `gate()` keeping every lap. `gate()` drops laps with `len(d) < 4` (`geometry.py:252`), and `zip` truncates silently, which would misassign `steer_kappa_corr`. Unreachable today, because R3 needs ≥ 4 samples per channel. | `assert len(lap_keys) == len(lap_corr)`, or have the gate return per-lap results keyed by lap. | Verified fragile |
| M4 | MINOR | `rlsetup/critic/style_data.py:145-157` | The cache is never pruned: 1,698 files for 849 laps. The key also leaves out the numpy/scipy versions, and `gaussian_filter1d` or `cKDTree` changes would be served stale. | Add numpy and scipy `__version__` to the key; prune entries whose code digest is not current, or document. | Verified |
| M5 | MINOR | `rlsetup/critic/sources/f1laps.py:468-473` | Ties in distance keep the first sample. This is necessary with a distance-based time base, but it is undocumented and uncounted: 78 laps lose up to 31 % of their samples. | Document it, store the dropped-sample count per lap, optionally average ties. Write the test T2. | Verified |
| M6 | MINOR | `rlsetup/critic/track_features.py:93-117` | `TrackScaler`, which is fold-relevant, has no test. `feature_table` is dead code, duplicated by `style_data.track_feature_table`. | Write the test T6; delete `feature_table`. | Verified |
| M7 | MINOR | `rlsetup/critic/sources/f1laps.py:536-543` | `F1LapsAdapter.meta` is the contract method, but nothing calls or tests it. `build_frame` normalises session, controller and assists through a separate path, so the two can drift. | Have `build_frame` use `meta`, or write the test T7. | Verified |
| M8 | MINOR | `tests/test_c2_data.py:79-85`; `tests/test_c2_knobs.py:75-86` | `ranking_stats` is tested only with `pairs > 0`, so the R6 within-session rule is unchecked. The stop-rule test uses one group, so "per game" is unchecked. The code is right; I proved it above. | Write T3 and T4 with exact counts. | Verified |
| M9 | MINOR (C2b hook) | `rlsetup/critic/style_data.py:258-263` | NaN `S_corner_peak_angle` on the 3 ungated tracks (96 laps) passes through `transform`. A network fed this would produce NaN. | C2b: fold-safe imputation (training median, i.e. z = 0) plus a missingness indicator, and a test that the model inputs are finite. | Verified |
| M10 | MINOR (C2b hook) | `rlsetup/critic/sources/f1laps.py:367-369, 390-397` | The gap median and the 8 % selection use all rows, including held-out drivers and setups. The effect is small (hundreds of rows per game and track) but it touches the label. | C2b: either state it in the results, or recompute the median from training rows per fold (`lap_s`, `game`, `track` are available). | Reasoned |
| M11 | MINOR | `scripts/critic_c2_data.py:46-47, 166-170` | Provenance stores the geometry file sha but not the TUM raceline sha (`Geometry.line_sha256`), which the L_* depend on. It also stores no numpy/scipy versions. | Add `line_sha256` to the gate rows and the library versions to `summary.json`. | Verified |
| M12 | MINOR | `rlsetup/critic/sources/f1laps.py:332-462`; `rlsetup/critic/config.py:18-19`; `rlsetup/critic/style_data.py:95-104` | An Assetto Corsa adapter would have to import `Frame`, `build_frame` and `ranking_stats` from the F1 module, or re-implement them. `lap_styles` depends on the shape of `TelemetryLap`. | Before C2b, move `Frame`, `ranking_stats` and a lap-container Protocol into `sources/base.py` (or `critic/dataset.py`). Treat SESSIONS/CONTROLLERS as adapter output vocabularies. | Verified |
| M13 | MINOR | `scripts/fetch_track_geometry.py:185, 194` | `--offline` and re-downloads do not check the raw sha against the committed sidecar's `source.sha256`, so a changed raw file is converted silently. | Compare against the existing sidecar sha and stop on a mismatch. | Verified |
| M14 | MINOR | `data/tracks_extra/` (README row) | The derived bacinger data will be committed, but the MIT permission notice lives only in the gitignored `raw/LICENSE/LICENSE.md`. MIT asks for the notice to go with substantial portions. | Commit `data/tracks_extra/LICENSE-bacinger-f1-circuits.md` (a verbatim copy). | Verified |
| M15 | MINOR | `rlsetup/style/__init__.py:8` | The docstring points to `tests/test_c2_style_port.py`, which does not exist; the test is in `tests/test_c2_style.py`. | Fix the reference. | Verified |
| M16 | MINOR | `scripts/critic_c2_data.py:69-70` | `--out` defaults to a cwd-relative `runs/critic_c2_data`: run from elsewhere, it writes outside the gitignored tree. The `--no-cache` help says "the cache is still not read", but the cache is also not written. | Default to `REPO_ROOT / 'runs/critic_c2_data'`; fix the help text. | Verified |
| I1 | INFO | `rlsetup/critic/geometry.py:265-270` | With the fitted scale, the progress at lap end does not meet the start. Gap = scale·L_game − L_geo, all negative: Baku −30 m, Abu Dhabi −26, Australia −26, Monaco −25. Corner windows are 180 m, so S_corner_peak_angle is barely affected; the effect is mostly on L_* (held out). | Report the closure gap in the gate rows. | Measured |
| I2 | INFO | gate pooled over games | Azerbaijan passes pooled (0.855), but F1 25 alone has median \|corr\| 0.846 (18 laps). | Report per-game \|corr\| in the gate table. For Ibrahim and the architect. | Measured |
| I3 | INFO | `rlsetup/critic/config.py:128` | IQR/1.349: judged acceptable (see §1); the architect confirms. | — | Measured |
| I4 | INFO | `rlsetup/critic/geometry.py` docstring | The placeholder is the centre line where TUM has no raceline (the lead's instruction; the review preferred a min-curvature line). Low stakes, since L_* are held out; recorded in `line_kind`. | — | Accepted deviation |
| I5 | INFO | `rlsetup/critic/sources/f1laps.py:306` | Driver ids follow sorted (game, user), so they renumber when exports are added. | C2b: key folds to the data hash. | Reasoned |
| I6 | INFO | `rlsetup/critic/style_data.py:145` | The cache key is a sample digest instead of lap_id. Accepted: stricter, and better for privacy. | — | Accepted deviation |

## Tests the coder should add (exact)
- **T1** `test_c2_data.py::test_lap_samples_from_a_later_export`: export A has `laps.csv` with lap L and no `graph_*` or `time_mapping`. Export B, listed after A, has the same lap L with graphs and a time map. Assert that `load_exports([A, B]).laps[...]` has samples and a time map. This fails today.
- **T2** `test_c2_style_input.py::test_distance_ties_keep_first_and_are_counted`: steer `[[1,.1],[2,.2],[2,.9],[3,.3],[4,.4]]` keeps `[1,2,3,4]` with value 0.2 at 2 m, and the dropped count of 1 is recorded on `StyleInput`.
- **T3** `test_c2_knobs.py::test_stop_rule_is_per_game`: game A has 6/100 rows out, game B 0/1000. Assert `ValueError` with `'A'` in the message (pooled 0.55 %).
- **T4** `test_c2_data.py::test_ranking_pairs_within_session_only`: a hand-built Frame with 1 cell, 2 Race rows (different hash and gap) and 2 TT rows. Assert `pairs == 2`, not 6, and that equal-hash or equal-gap pairs are excluded.
- **T5** `test_c2_geometry.py::test_gate_rejects_mirror_image`: `geom_from(resample(shape_xy(1.),5.) * [-1, 1])` with the fixture laps. Assert `not passed`, or reason `gate_sign`. It passes today, which is the bug.
- **T6** `test_c2_style.py::test_track_scaler_uses_training_tracks_only`: the same pattern as the StyleTransform leakage test (change held-out rows ×1000; mean and std unchanged; NaN rows stay NaN).
- **T7** `test_c2_data.py::test_adapter_contract`: `isinstance(F1LapsAdapter(), SourceAdapter)`; `meta()` on a 3-row table returns sources, sessions, controllers and assist levels equal to `build_frame`'s for those rows; `to_knobs` equals `game_to_knobs`.
- **T8** Extend `test_no_username_in_any_output` so it also scans `tmp_path/'cache'/*.json`.
- **T9** `test_c2_style.py::test_lap_corr_alignment`: on the fixture, for each gated track, `len(lap_keys) == len(tg.chosen[1].lap_corr)`, and `steer_kappa_corr` equals a per-lap recomputation at the fitted shift and scale.

## Bottom line
C2a implements the spec and R1–R4 faithfully, and the gate is applied the same way to TUM and converted tracks. The port is exact. Privacy is clean on every output, cache and fixture. The build is bit-identical across a cached and a no-cache run. The two full-suite failures are pre-existing and unrelated. Before C2b, fix M1, M2, M3, M8 and M12 and add T1–T9. M9 and M10 are decisions for C2b.
