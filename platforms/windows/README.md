# Windows (i9-14900, RTX 4000 Ada)

24 cores / 32 threads, 64 GB RAM, NVIDIA RTX 4000 Ada 20 GB (CUDA 13.2 driver). No admin rights, no WSL, no MSVC.

## Environment
- **Python:** conda env `C:\Users\imk14\ocdenv` (Python 3.12, torch 2.11.0+cu128, stable-baselines3 / sb3-contrib 2.9.0, gymnasium 1.4.0, tensorboard, pytest, plus `ocd_first_lap/requirements.txt`). `env.ps1` puts `ocdenv` and `ocdenv\Library\bin` on PATH and `ocd_first_lap` on PYTHONPATH. Every launcher here dot-sources it. Override the defaults with `OCD_ENV` (default `$HOME\ocdenv`) and `OCD_FIRST_LAP` (default `..\ocd_first_lap`, next to this repo).
- **ocd_native:** WinLibs MinGW GCC 16.2 (winget, user scope) plus cmake / ninja / eigen / boost headers from conda. In `ocd_first_lap`, with `vendor/Open-Car-Dynamics` at pin 94f8fb1 + `patches/*.patch` and `vendor/racetrack-database` at e59595d:
  ```powershell
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ `
    -DPython_EXECUTABLE=C:/Users/imk14/ocdenv/python.exe -Dpybind11_DIR=$(python -m pybind11 --cmakedir) `
    -DCMAKE_PREFIX_PATH=C:/Users/imk14/ocdenv/Library
  cmake --build build -j 28                  # -> ocd_native.cp312-win_amd64.pyd (static MinGW runtime)
  ```
- **Scripts:** `powershell -ExecutionPolicy Bypass -File <script>.ps1` (no admin needed). Inside a PowerShell prompt, `.\platforms\windows\<script>.ps1` also works if the execution policy allows it.

## Defaults (`train.overrides`): PROVISIONAL
| Key | Value | Source |
|---|---|---|
| `device` | `cuda` | Runs `full_win`, `sf70h_win` |
| `vec_start_method` | `spawn` | No fork / forkserver on Windows |
| `n_envs` | 16 | Runs `full_win`, `sf70h_win`; not tuned |
| `gradient_steps` | 8 | UTD 0.5 (same as the Mac) |
| `torch_threads` | 2 | Runs `full_win`, `sf70h_win`; not tuned |

`full_win` measured about 175 decisions/s (`sf70h_win` about 160) and is learner-bound: GPU 25–45 % busy, because small MLPs on CUDA are dominated by kernel-launch overhead. Replace these values with the results of "Tuning" below.

## Launch
Long runs must be detached with `start_detached.ps1` (`Start-Process -WindowStyle Hidden`, stdout and stderr in files). A process started with `nohup ... &` from Git Bash dies when its parent shell or tool call ends; that already killed one run.
```powershell
platforms\windows\train.ps1 --preset full --seed 0 --run-name NAME            # foreground, -> runs\windows\D2\NAME\seed0
platforms\windows\start_detached.ps1 -Log runs\windows\NAME_stdout.log platforms\windows\train.ps1 --preset full --seed 0 --run-name NAME
platforms\windows\start_detached.ps1 -Log runs\windows\NAME_stdout.log platforms\windows\train.ps1 --resume --preset full --seed 0 --run-name NAME --allow-code-change --allow-dirty
powershell -ExecutionPolicy Bypass -File platforms\windows\run_d2_full.ps1 --dry-run
platforms\windows\start_detached.ps1 -Log runs\windows\full_stdout.log platforms\windows\run_d2_full.ps1
```
`run_d2_full.ps1` is the port of `scripts/run_d2_full.sh`. It uses the same environment variables (`RUN_NAME`, `MAX_WALL_H`, `TOTAL_STEPS`, `RUNS_ROOT`, `OVERRIDES_FILE`, `PYTHON_BIN`) and makes the same per-seed decisions. It keeps the same lock file, report and exit code. `tests/test_run_d2_full_ps1.py` tests it (runs on Windows only).

**Pre-split runs** (`runs/D2/full_win`, the live `runs/D2/sf70h_win` with its watcher, HANDOFF 27 Oct) keep the old root. Don't move them while they run. To resume one by hand (NAME = `full_win` or `sf70h_win`):
```powershell
python -m rlsetup.driving.train --preset full --seed 0 --run-name NAME --resume --allow-code-change --allow-dirty   # with env.ps1's PATH / PYTHONPATH
$env:RUNS_ROOT = 'runs/D2'; platforms\windows\train.ps1 --resume --preset full --seed 0 --run-name NAME --allow-code-change --allow-dirty
```
`sf70h_win` was started with `--set env.car=sf70h_tum --set "demos.fraction_range=(0.5,0.8)"` on top of these overrides; its config.json keeps them. A new sf70h_tum run: `platforms\windows\train.ps1 --preset full --seed 0 --run-name NAME --set env.car=sf70h_tum --set "demos.fraction_range=(0.5,0.8)"`.
TensorBoard: `tensorboard --logdir runs/windows/D2 --host 127.0.0.1 --port 6006` (the live instance uses `runs/D2`).

## Tuning (task 2): NOT MEASURED YET
`bench.ps1` runs these steps at BelowNormal CPU priority and writes JSON under `runs\windows\bench\<stamp>\`:
1. `learner`: `sac_train_speed.py`, SAC `train()` gradient steps/s at batch 512, 3×256 MLPs. CPU with 1, 2, 4 and 8 threads, and CUDA with 1, 2 and 4 threads.
2. `vec`: `vec_throughput.py`, spawn env workers 4 to 24, random actions, decisions/s.
3. `e2e`: `sac_e2e.py` at UTD 0.5 (gradient_steps = workers / 2), 8, 12, 16 and 24 workers, on CPU and on CUDA. It uses a plain SB3 SAC (no n-step, no critic LayerNorm), so it ranks the configurations; check the absolute rate against the next real run's `wall_train_s`.

While a run is live, `bench.ps1` skips the CUDA steps, because GPU sharing would slow both. It also records in `meta.json` that the CPU numbers are contended. For final numbers, run it when no training run is live. Then choose the fastest end-to-end configuration at UTD 0.5, put it in `train.overrides`, and record the numbers in the table below.

| Measurement | Result |
|---|---|
| Learner CPU, grad steps/s at 1, 2, 4, 8 threads | not measured |
| Learner CUDA, grad steps/s | not measured |
| Env workers (spawn), decisions/s at 4–24 | not measured |
| End to end at UTD 0.5, decisions/s | not measured |

## Tests
`python -m pytest -q -m "not slow"` (PYTHONPATH must include ocd_first_lap). See `docs/handoff/W1_windows_port.md` for which tests still fail on Windows, and why.
