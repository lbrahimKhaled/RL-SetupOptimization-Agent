# tests

```bash
python -m pytest -q -m "not slow"    # fast suite
python -m pytest -q                  # everything (long runs, 200-reset statistics)
```

| Area | Files |
|---|---|
| Env and reference | `test_env.py`, `test_env_slow.py`, `test_reference.py`, `test_qss_profile.py` |
| Demos, BC, agent | `test_demo_driver.py`, `test_demos_bc.py`, `test_agent.py`, `test_callbacks.py` |
| D2 pipeline | `test_d2_slow.py`, `test_report.py`, `test_run_d2_full.py`, `test_provenance.py` |
| R1 reward | `test_r1.py`, `test_r1_slow.py`, `test_relabel.py` |
| Helpers and golden data | `conftest.py`, `sb3_helpers.py`, `r1_golden.py`, `relabel_truth.py` |
