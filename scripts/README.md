# scripts

Command-line tools. Run from the repo root.

| Script | Use |
|---|---|
| `build_qss_profile.py` | Rebuild the cached p0 QSS speed profile in `data/` |
| `collect_demos.py` | Collect or reuse the demonstration dataset for a preset (with the clean-fraction gate) |
| `run_d2_full.sh` | Unattended D2 run: demos once, seeds 0 to 2 in sequence, then the report. Launch it through `platforms/mac/run_d2_full.sh` (Windows port: `platforms/windows/run_d2_full.ps1`) |
| `check_line_starts.py` | Settle check every 100 m along a preset's xi_ref (drive line) and, with `--demos`, its demo dataset clean rate (fills the demo cache the run reuses) |
| `seed_summary.py` | Per-seed progress table of a run (read-only); produces `SEEDS.md` |
| `d2_report.py` | D2 done-criterion report (DECISIONS #12) |
| `evals_to_tb.py` | Mirror `evals.jsonl` into TensorBoard (`tb_evals/`; warm-started runs also `tb_evals_lifetime/` on the lifetime step axis, chained over warm starts) |
| `r4_dry_check.py` | R4 dry check: builds `--preset f1_r4` (lifetime offset) and relabels the f1_best_w35 buffer in memory (chain / anchor / F_v / entry-exit stats; the file is not written) |
| `keeper_check.py` | Keeper stop / no-resume checks (reportable clean lap 2 <= the run's `trigger_lap2_s`, 95 s; any evaluation for a never-fade run (R3); R2 run ended itself) and the `pgrep -f` pattern of a run (`pattern RUN`, no prefix matches) |
| `fetch_track_geometry.py` | C2a: bacinger/f1-circuits centre lines (pinned commit, MIT) -> TUM-format files in `data/tracks_extra/` (+ JSON provenance), direction and start from the R2 gate, PLACEHOLDER width; prints the gate table. Stops when a raw file's sha256 differs from its sidecar unless `--update`; `--offline` re-converts from disk. Raw downloads in `data/tracks_extra/raw/` (gitignored) |
| `critic_c2_data.py` | C2a data build: setups -> rows A / S, laps -> style numbers, R2 gate table (closure gap, per-game |corr|, sign), track z (+ report-only z of the failing candidates), fast tail of `gap`, style summary, counts vs the C2 review; writes `runs/critic_c2_data/<UTC>/`; style cache in `runs/style_cache/<env>/` (`--no-cache`: neither read nor written) |
| `bench_env.py` | D1 env throughput and demo-driver lap benchmark |
| `comparison/` | Data for the TUM vs OCD vs RL comparison: `export_tum_lap.py` (TUM's own lap, run in TUM's venv), `tum_pure_pursuit.py` (TUM's lap in OCD by pure pursuit), `export_rl_laps.py` (the RL driver's laps). Output and docs in [comparison/](../comparison/README.md) |
| `bench/` | Throwaway throughput measurements (vec env, SAC speed); not part of the pipeline. Windows driver: `platforms/windows/bench.ps1` |
