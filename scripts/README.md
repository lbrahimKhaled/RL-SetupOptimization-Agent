# scripts

Command-line tools. Run from the repo root.

| Script | Use |
|---|---|
| `build_qss_profile.py` | Rebuild the cached p0 QSS speed profile in `data/` |
| `collect_demos.py` | Collect or reuse the demonstration dataset for a preset (with the clean-fraction gate) |
| `run_d2_full.sh` | Unattended D2 run: demos once, seeds 0 to 2 in sequence, then the report. Launch it through `platforms/mac/run_d2_full.sh` (Windows port: `platforms/windows/run_d2_full.ps1`) |
| `seed_summary.py` | Per-seed progress table of a run (read-only); produces `SEEDS.md` |
| `d2_report.py` | D2 done-criterion report (DECISIONS #12) |
| `evals_to_tb.py` | Mirror `evals.jsonl` into TensorBoard |
| `bench_env.py` | D1 env throughput and demo-driver lap benchmark |
| `bench/` | Throwaway throughput measurements (vec env, SAC speed); not part of the pipeline. Windows driver: `platforms/windows/bench.ps1` |
