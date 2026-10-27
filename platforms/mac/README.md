# Mac (MacBook Air M4)

The machine the project was developed and run on: CPU training with forkserver env workers.

## Environment
- Python 3.12 venv at `<repo>/.venv` (`.venv/bin/python`). The launchers and `scripts/run_d2_full.sh` use it.
- `ocd_first_lap` is a sibling checkout (`~/Documents/ocd_first_lap`), set up with its own `./setup.sh`: Homebrew cmake/eigen/boost/pybind11, Apple clang, `scripts/build.sh` → `ocd_native.cpython-312-darwin.so`. It must be importable from `.venv`. If it is not on the venv's path, set `PYTHONPATH=~/Documents/ocd_first_lap`.
- Packages: stable-baselines3 / sb3-contrib 2.9.0 (pinned in `pyproject.toml`), torch, tensorboard, pytest. Run everything from the repo root.

## Defaults (`train.overrides`)
| Key | Value | Why |
|---|---|---|
| `device` | `mps` | Apple GPU, by request (27 Oct). The only measurement so far says it is **slower**; see below |
| `vec_start_method` | `forkserver` | |
| `n_envs` | 4 | 4 performance cores (D1_sim_runner §4) |
| `gradient_steps` | 2 | UTD 0.5 |
| `torch_threads` | 1 | Fastest learner on the M4 (D1_sim_runner §4) |

Apart from `device` (the `TrainConfig` default is `cpu`), these match the `TrainConfig` defaults.

## GPU (MPS)
`device=mps` puts the SB3 model (actor, critics, entropy coefficient, replay-buffer samples) on the Apple GPU, so gradient steps, acting and evaluation run there. Env workers (physics) and BC stay on the CPU. BC runs on the CPU on every machine, so its result does not depend on the device. Runs from before this change have no `device` in `config.json`, so they resume on the CPU.

The only measurement (D1_sim_runner §4b, M4, SB3 `train()`, 3×256 MLPs, batch 512) is **60 grad steps/s on MPS vs 198 on 1 CPU thread**. These networks are too small for the GPU to pay off: every step pays kernel-launch and sync overhead. Before a long run, measure it on the Mac:
```sh
.venv/bin/python scripts/bench/sac_train_speed.py --device cpu --threads 1 --steps 1000
.venv/bin/python scripts/bench/sac_train_speed.py --device mps --threads 1 --steps 1000
.venv/bin/python scripts/bench/sac_e2e.py --workers 4 --grad 2 --device mps     # end to end, UTD 0.5
.venv/bin/python scripts/bench/sac_e2e.py --workers 4 --grad 2 --device cpu
.venv/bin/python -m pytest -q tests/test_agent.py -k accelerator               # MPS code path
```
If MPS is slower end to end, set `device=cpu` in `train.overrides`, or pass `--set device=cpu` for a single run.

## Launch
```sh
platforms/mac/train.sh --preset full --seed 0 --run-name NAME          # -> runs/mac/D2/NAME/seed0
platforms/mac/train.sh --resume --preset full --seed 0 --run-name NAME
platforms/mac/run_d2_full.sh --dry-run
caffeinate -dimsu nohup platforms/mac/run_d2_full.sh > runs/mac/full_stdout.log 2>&1 &
```
Runs from before the split (`runs/D2/full1`, `r1_warm`, ...) stay in `runs/D2`. To resume one of them, use the old command or set `RUNS_ROOT=$PWD/runs/D2`.
