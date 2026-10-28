# Build notes: comparison data (TUM vs pure pursuit in OCD vs RL)

Branch `comparison-tum-vs-rl` from `main` (ef80ca5). Not committed. ocd_first_lap was not modified: clean at 7d6f2aa, branch sf70h-tum-car. Data was generated from the scripts below before they were committed: the metadata records the rlsetup commit ef80ca5 with `untracked: true` plus the sha256 of each script.

## Files created

| Path | What |
|---|---|
| `scripts/comparison/common.py` | Column schemas (single source for the CSV headers and the README tables), CSV and metadata writers. Standard library and numpy only, so it runs in both venvs |
| `scripts/comparison/export_tum_lap.py` | TUM's own lap, run in TUM's venv. Imports the settings dictionaries of ocd_first_lap's `run_tum_laptimesim.py` (read only) |
| `scripts/comparison/ocd_trace.py` | Read-only OCD sample row, line geometry with its own match hints, lap clock (the env's line-crossing rule) |
| `scripts/comparison/tum_pure_pursuit.py` | TUM's lap in OCD: `DemoDriver` pure pursuit on `Track.reference` at f x v_TUM(`ers`), plus the speed-scale search |
| `scripts/comparison/export_rl_laps.py` | The RL driver's laps through `evaluate.flying_lap`, with a read-only per-tick hook (a subclass in the script; no rlsetup change) |
| `tests/test_comparison_schema.py` | CSV headers equal the schema, every CSV has its meta, the README tables equal the schema, writer behaviour |
| `scripts/README.md` | One row added for `comparison/` |
| `comparison/README.md` | What each file is, columns and units, how it was produced, the condition-difference table, regeneration commands, caveats |
| `comparison/tum/` | `tum_tum_default.csv`, `tum_ers.csv`, `tum_none.csv` + meta (0.8 MB) |
| `comparison/tum_pure_pursuit_ocd/` | `pp_f1.00.csv` + meta, `search.json` (4.3 MB) |
| `comparison/rl/` | `rl_det_seed0.csv` (50 Hz), `rl_sto_seed0..9.csv` (25 Hz) + meta, `summary.json` (17 MB) |

Total 22 MB, CSV only (no models). TUM's code is cloned outside the repo at `~/Documents/tum/laptime-simulation` (clean, afa0aa7), with its venv at `~/Documents/tum/tumenv` (Python 3.12.13, numpy 1.26.4, scipy 1.13.1, matplotlib 3.8.4, quadprog 0.1.14, trajectory-planning-helpers 0.66 installed without dependencies).

## Lap times reproduced

| Source | Result | Target |
|---|---|---|
| TUM `tum_default` (ERS + recuperation + DRS) | 91.238 s (6.99 MJ deployed, 318.4 km/h) | 91.24 s: reproduced |
| TUM `ers` (ERS, no recuperation, no DRS) | 93.243 s | 93.24 s (ocd_first_lap `tum_laptimesim.json`): reproduced |
| TUM `none` | 94.886 s | 94.89 s: reproduced |
| RL deterministic, lap 2 | 103.23790594919477 s (lap 1 103.893 s) | 103.24 s at step 6.95M: identical to the training eval to the last digit |
| RL stochastic, seeds 0-9 | 7/10 clean, mean 105.574 s; seeds 1, 2, 4 off track at Track.path s = 4690, 2556, 3131 m (lap 1) | all 10 identical to the training eval in termination and lap time |

The TUM `raceline_csv` sha256 (`db621f25…`) equals ocd_first_lap's vendored raceline, so the two simulators use the same line.

## Pure-pursuit result

- f* = 1.00. At f = 1.00 the car completes lap 2 without a spin. Lap 2 = 104.700 s (lap 1 = 104.623 s), against TUM `ers` at 93.243 s.
- Lap 2 behaviour: on_track fraction 0.708 (0.704 over the whole run). Mean |e| to the raceline 2.11 m, max 17.7 m. Max |rear slip| 0.37, max sideslip 0.148 rad, v_max 297.2 km/h.
- The car does not hold TUM's speeds. Lap-2 mean speed is 52.1 m/s against a mean target of 56.1 m/s. It runs wide in several corners, and the traction control is active on about a third of lap 2.
- No spin occurred at any f, so the search rule ("highest f that completes lap 2 without spin") gives f* = 1.00 and only `pp_f1.00.csv` is exported.
- As a diagnostic (summaries only, `search.json`), the sweep went on down to f = 0.80:

  | f | Lap 2 [s] | On track, lap 2 | Max \|e\| [m] |
  |---|---|---|---|
  | 1.00 | 104.70 | 0.708 | 17.7 |
  | 0.98 | 104.34 | 0.753 | 9.1 |
  | 0.96 | 104.61 | 0.776 | 7.2 |
  | 0.94 | 105.70 | 0.779 | 5.7 |
  | 0.92 | 106.93 | 0.781 | 4.4 |
  | 0.90 | 108.34 | 0.784 | 3.7 |
  | 0.80 | 118.38 | 0.775 | 2.4 |

- The on_track fraction never exceeds about 0.78, because of the line rather than the controller: with the 4.6 x 2.0 m sf70h_tum footprint placed exactly on TUM's raceline, only 70.1 % of the raceline (per metre) is inside the track. On the safe line the same check gives 100 %. This ceiling is stored in the meta and in `search.json`.
- Lap 2 = 104.700 s exactly is not a rounding artefact. The deterministic closed loop (with the ERS refilled at the line) settles into a periodic lap, and laps 2 and 3 cross the line at the same tick phase (5235 ticks).

## Deviations and decisions to confirm

1. **Brake cap of the reused controller.** `DemoConfig.brake_cap` is raised from 0.3 to 1.0 (12 kN·m). 0.3 limits braking to about 1.5 g, far below TUM's profile. All other `DemoDriver` gains are the defaults, documented in the README and the meta. The lead may prefer another value.
2. **f\* criterion.** The spec's criterion (lap 2 without spin) is met at f = 1.00 even though the car clearly cannot hold TUM's speeds (large offsets and slip). If a stricter criterion is wanted, it would need a decision; the sweep in `search.json` already holds the numbers for it. Examples: max |e| below some value, or on_track close to the 70 % line ceiling.
3. **Spin definition.** |heading error to the raceline| > pi/2 (the env's wrong-way rule). Non-finite state, stall (EnvConfig thresholds) and a 400 s time limit also stop the run. None of them fired.
4. **ERS in pure pursuit.** Refilled to 4 MJ at each line crossing, as `ocd_lap.simulation.drive` does. The backend's ERS also differs from TUM's: it boosts every throttle request, while TUM boosts only above the engine maximum (ocd_first_lap `docs/SF70H_TUM_CAR.md`).
5. **RL sample rates.** The deterministic run is exported at 50 Hz base ticks (two per decision; decision columns on the decision-end tick). Stochastic runs are exported at the 25 Hz decision ends only, to keep the data small (17 MB instead of about 35 MB). The training eval's stochastic env had `record_telemetry=False`; the export uses True. The reproduced lap times show this changes nothing.
6. **Engine and e-motor crank torques in the OCD files** are recomputed at the sample state, because the backend does not store the split. `drive_torque_axle_Nm` is OCD's actual input.
7. **Timing line.** For both OCD sources it is Track.path s = 0, the env's definition. TUM's raceline s = 0 is abeam Track.path s = -0.06 m.
8. **CSV precision.** 7 significant digits. The pure-pursuit run reads `tum_ers.csv` as written, so its lap times depend on that file (sha256 in the meta). With the earlier 9-digit file, lap 2 was 104.712 s instead of 104.700 s, a measure of the closed loop's sensitivity at the limit.

## Determinism

Every CSV of the three sources was regenerated into a temporary folder and is byte-identical to the one in `comparison/`: TUM 3/3, RL 11/11, pure pursuit 1/1. RL inference runs on CPU with 1 torch thread, with the same seeds as the training eval.

## Pre-existing issues seen (not caused by this work, not fixed)

- `data/qss_profile_Shanghai_p0_sf70h_tum.npz` cache key mismatch. It was built on Windows: the ocd_native binary, the ocd_lap source hashes and the Track.path sha differ from this Mac's. `load_or_build` therefore recomputes the profile in memory with a warning. The sf70h_mac_r1 training run hit the same warning (its `stdout.log` line 1), which is why the export reproduces it.
- The fast suite has 3 failures that also fail on a clean `main` worktree:
  - `test_provenance::test_every_imported_ocd_lap_module_is_hashed`: `ocd_lap/cars.py` is not in `OCD_LAP_EXTRA_SOURCES`.
  - `test_qss_profile::test_cached_file_matches_fresh_computation`: KeyError 'provenance' on the cached npz.
  - `test_run_d2_full::test_mac_launcher_dry_run`: the test expects `device=mps`, but the launcher now uses `cpu` (commit 7eb1663).

## Tests

`.venv/bin/python -m pytest -q -m "not slow"`: 178 passed, 3 failed (the pre-existing ones above), 3 skipped, 5 deselected, in 22.6 s. `tests/test_comparison_schema.py`: 5 passed in 0.15 s.
