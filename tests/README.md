# tests

```bash
python -m pytest -q -m "not slow"    # fast suite
python -m pytest -q                  # everything (long runs, 200-reset statistics)
```

| Area | Files |
|---|---|
| Env and reference | `test_env.py`, `test_env_slow.py`, `test_reference.py`, `test_line_reference.py`, `test_qss_profile.py` |
| Demos, BC, agent | `test_demo_driver.py`, `test_demos_bc.py`, `test_agent.py`, `test_callbacks.py` |
| D2 pipeline | `test_d2_slow.py`, `test_report.py`, `test_run_d2_full.py` (bash, skipped on Windows), `test_run_d2_full_ps1.py` (Windows only), `test_provenance.py` |
| R1 reward | `test_r1.py`, `test_r1_slow.py`, `test_relabel.py` |
| R2 training aids (corner bonus, focus, low noise) | `test_r2.py`, `test_r2_slow.py` (sf70h_f1 + f1_best_w35 run folder) |
| R3 bigger bonus (relabel, never-fade schedule, keeper, offset chaining, focus init state) | `test_r3.py` (a few tests need the f1_r2 run folder) |
| R4 speed profile (g, Φ_v telescoping / bit identity, entry / exit tracker and take-back, relabel vs env ground truth, preset f1_r4, fallback pieces, logging) | `test_r4.py` (a few tests need sf70h_f1 / the f1_best_w35 run folder) |
| Helpers and golden data | `conftest.py`, `sb3_helpers.py`, `r1_golden.py`, `relabel_truth.py`, `speed_truth.py` |
