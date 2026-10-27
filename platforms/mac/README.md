# Mac (MacBook Air M4)

The machine the project was developed and run on: CPU training with forkserver env workers.

## Environment
- Python 3.12 venv at `<repo>/.venv` (`.venv/bin/python`). The launchers and `scripts/run_d2_full.sh` use it.
- `ocd_first_lap` is a sibling checkout (`~/Documents/ocd_first_lap`), set up with its own `./setup.sh`: Homebrew cmake/eigen/boost/pybind11, Apple clang, `scripts/build.sh` → `ocd_native.cpython-312-darwin.so`. It must be importable from `.venv`. If it is not on the venv's path, set `PYTHONPATH=~/Documents/ocd_first_lap`.
- Packages: stable-baselines3 / sb3-contrib 2.9.0 (pinned in `pyproject.toml`), torch, tensorboard, pytest. Run everything from the repo root.

## Defaults (`train.overrides`)
| Key | Value | Why |
|---|---|---|
| `device` | `cpu` | Measured M4 optimum. `mps` was not measured; do not switch without a benchmark |
| `vec_start_method` | `forkserver` | |
| `n_envs` | 4 | 4 performance cores (D1_sim_runner §4) |
| `gradient_steps` | 2 | UTD 0.5 |
| `torch_threads` | 1 | Fastest learner on the M4 (D1_sim_runner §4) |

These match the `TrainConfig` defaults, so a plain `python -m rlsetup.driving.train` behaves the same apart from the run root.

## Launch
```sh
platforms/mac/train.sh --preset full --seed 0 --run-name NAME          # -> runs/mac/D2/NAME/seed0
platforms/mac/train.sh --resume --preset full --seed 0 --run-name NAME
platforms/mac/run_d2_full.sh --dry-run
caffeinate -dimsu nohup platforms/mac/run_d2_full.sh > runs/mac/full_stdout.log 2>&1 &
```
Runs from before the split (`runs/D2/full1`, `r1_warm`, ...) stay in `runs/D2`. To resume one of them, use the old command or set `RUNS_ROOT=$PWD/runs/D2`.
