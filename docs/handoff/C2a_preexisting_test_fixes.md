# Two pre-existing test failures fixed (16 Nov 2026)

Fixing these was approved by Ibrahim. Changed: `rlsetup/provenance.py` (one tuple entry plus its comment) and
`data/qss_profile_Shanghai_p0.npz` (rebuilt). Nothing in ocd_first_lap was edited. Nothing is committed.
Before the fix, the two tests failed with the ocd_first_lap on the `.pth` path (main checkout, branch `sf70h-tum-car`, 3945634, clean).

## 1. `test_provenance.py::test_every_imported_ocd_lap_module_is_hashed`

**Cause.** `ocd_lap/backend.py` imports `ocd_lap/cars.py` (added in ocd_first_lap 846e2f2, 27 Oct). `cars.py` is not hashed by
`ocd_lap.provenance.capture_provenance` (it is missing from `simulation.SOURCES`), and it was missing from `OCD_LAP_EXTRA_SOURCES`.
So the definitions of the cars, including `sf70h_f1` used by every f1_* run, were not identified by any hash in run provenance.

**Impact analysis (before the change).** `OCD_LAP_EXTRA_SOURCES` only feeds `run_provenance()['ocd_lap_extra_sources_sha256']`.
- Resume check (`train.py` → `code_state` / `code_mismatches`): compares rlsetup commit, diff_sha256 and `ocd_lap_code_version`
  only. `ocd_lap_code_version` is ocd_lap's own digest, which this list does not change. **No effect**.
- QSS profile cache key (`qss_profile.cache_key`): uses its own `_SOURCES` tuple, not this list. **No effect** (all caches keep their keys).
- Demo dataset key (`demos.dataset_key`): `ocd_lap_code_version` + QSS key sha + rlsetup demo sources. **No effect**.
- Corner table key (`corners.cache_key`): QSS key + corners.py sha. **No effect**.
- `scripts/d2_report.py`: reads `ocd_lap_code_version` only. **No effect**.
- `training_provenance`, `qss_profile.save`, `corners.save`, `critic_c2_data.py`, bench/sweep scripts: record the dict only.
  **New outputs record one more hash.**
- `scripts/comparison/comparison_state.source_differences` (used by export_fair_rl_laps, fair_tum_baseline, tune_*): compares
  `ocd_lap_extra_sources_sha256` key by key, so a saved run without the cars.py key now shows one more difference.
  These scripts already refuse every saved run when run from this working tree: run against f1_r4_scratch before the edit, they
  reported 14 differences (the untracked `rlsetup/critic/*`, `rlsetup/style/*` files appear in `rlsetup_sources_sha256`), and
  any edit to `rlsetup/provenance.py` also changes that file's own hash in `rlsetup_sources_sha256`. The comparison workflow runs
  from a pinned worktree (`/Users/ibrahimkhaled/Documents/rlsetup-wt-full`, f17be23) that has its own `provenance.py`, so it is
  not affected. **No added blocker in practice.**
- No training process was running (checked with `ps`), only TensorBoard.

So in practice the only effect is that new runs record one more file hash. Change: `'cars.py'` added to `OCD_LAP_EXTRA_SOURCES`.

Separate gap, not changed: `cars.py` is also missing from the QSS cache key `_SOURCES` in `rlsetup/driving/qss_profile.py`. For
non-av21 cars, editing a car's parameters in cars.py would therefore not invalidate its `qss_profile_*_<car>*.npz`. Adding it
would invalidate every tracked cache, so that is Ibrahim's decision.

## 2. `test_qss_profile.py::test_cached_file_matches_fresh_computation`

**Cause.** The cache key stored in `data/qss_profile_Shanghai_p0.npz` was stale. `conftest.qss` → `load_or_build` saw the key
mismatch, warned, and returned an in-memory recomputation. `compute()` does not attach `provenance` (only `save()` does), so
`qss.meta['provenance']` raised `KeyError`. The file itself does contain provenance.
- The file was built 2026-10-26T05:44 UTC (commit 717fa89, "D2: QSS cache key, pins") from ocd_first_lap 2147dcd with uncommitted changes.
- On 27 Oct, ocd_first_lap committed Stage 6 and the selectable cars (1c5ea98, 846e2f2, 76e235f, 9fb0253, 5e65cff). These commits changed
  `qss_lap.py, handling.py, manoeuvres.py, powertrain.py, backend.py, rig.py`, whose sha256 values are part of the key. The native binary,
  config, track path and setup keys still matched.
- `scripts/build_qss_profile.py` was last changed in 4c9d958 (25 Oct). `rlsetup/driving/qss_profile.py` changed later (ea73f0d, 604efe7,
  4533fe9), but only in ways that keep the av21 / 12 kN·m key format the same.

**Readers.** `data/qss_profile_Shanghai_p0.npz` is the default path for `car='av21', setup='p0', brake 12 kN·m` (the EnvConfig defaults).
It is read by `tests/conftest.py` (the session `qss` fixture used across the suite), `scripts/bench_env.py`, `scripts/bench/*`, and
any run on the default env: `runs/D2/full1`, `r1_warm` and the smokes (Oct 26). No f1_*, p0_TUMref, p1_TUM or bcn run uses it. They use
other cars, setups or tracks. Since 27 Oct, every reader has received the in-memory recomputation with the current key. The rebuilt
file stores that same key, so no downstream key (demo dataset, corner table, provenance `qss.cache_key_sha256`) changes compared with
today's behaviour.

**Numbers (old file vs fresh, then old file vs rebuilt file).** The `s`, `v`, `ggv`, `traction` and `drag_coeff` arrays are all
bit-identical: max |Δv| = 0.0 m/s over 2696 nodes. The lap time is 133.91610367943798 s in both, Δ = 0.0. That is within the test's
own tolerance (`assert_array_equal`, exact `==`), so the file was rebuilt with `.venv/bin/python scripts/build_qss_profile.py`.
Output: `.../data/qss_profile_Shanghai_p0.npz: lap 133.916 s, v 10.18-80.08 m/s`. The new provenance records ocd_first_lap 3945634
(clean), code_version 70c48ed08c49, and includes the cars.py hash. The old file is kept at
`scratch/qss_profile_Shanghai_p0.old.npz`.

**Other tracked caches (each checked against its own key).**

| file | main ocd_first_lap (.pth) | ocd_first_lap-f1 (PYTHONPATH) |
|---|---|---|
| Catalunya_bcn_best_sf70h_tum_brake15000 | match | stale (qss_lap.py, backend.py) |
| Shanghai_p0 (av21) | stale → rebuilt | stale (the f1 tree's sources differ) |
| Shanghai_p0_sf70h_tum | **stale**: built on Windows (`C:\...\ocd_first_lap`), native binary, config, track-path sha and all sources differ | stale |
| Shanghai_p1_TUM_f1_sf70h_f1_brake15000 | n/a (no sf70h_f1 car) | match |
| Shanghai_p1_TUM_sf70h_tum_brake15000 | match | stale (qss_lap.py, backend.py) |
| Shanghai_sha_f1_best_sf70h_f1_brake15000 | n/a | match |

Every cache matches in the worktree its runs use. The one exception is `Shanghai_p0_sf70h_tum` (Windows build). Its readers recompute
it in memory with a warning. It was not rebuilt here because no test failed on it and it was not in scope.

## Tests
```
.venv/bin/python -m pytest -q tests/test_provenance.py::test_every_imported_ocd_lap_module_is_hashed tests/test_qss_profile.py::test_cached_file_matches_fresh_computation
2 passed in 1.18s
.venv/bin/python -m pytest -q
507 passed, 14 skipped in 179.42s (0:02:59)
```
