# platforms

Per-machine setup. The code and presets are shared. A platform contributes only its launcher, its environment notes, its TrainConfig overrides and its run root, so results from the two machines never mix.

| | [mac/](mac/README.md) (MacBook Air M4) | [windows/](windows/README.md) (i9-14900 + RTX 4000 Ada) |
|---|---|---|
| Python | `.venv/bin/python` (venv in the repo) | conda env `C:\Users\imk14\ocdenv` (`env.ps1`) |
| ocd_native | Apple clang, `ocd_first_lap/scripts/build.sh` | MinGW GCC 16.2 + Ninja (windows/README.md) |
| Overrides | device cpu, forkserver, n_envs 4, gradient_steps 2, torch_threads 1 | device cuda, spawn, n_envs 16, gradient_steps 8, torch_threads 2 (provisional) |
| Run root | `runs/mac/D2` | `runs/windows/D2` |
| One run | `platforms/mac/train.sh ...` | `platforms\windows\train.ps1 ...` |
| Unattended D2 | `caffeinate -dimsu nohup platforms/mac/run_d2_full.sh > ... &` | `platforms\windows\start_detached.ps1 -Log ... platforms\windows\run_d2_full.ps1` |

`train.overrides`: one `KEY=VALUE` per line, applied as `--set` to fresh runs and to demo collection. A resume keeps the run's own `config.json`. Both machines keep UTD 0.5 (`gradient_steps / n_envs`). `device` needs `TrainConfig.device` (commit "TrainConfig.device, default 'cpu'").

Runs from before the split live in `runs/D2/` (Mac: `full1`, `r1_warm`, ...; Windows: `full_win`). Resume them with `RUNS_ROOT=runs/D2` or the plain `python -m rlsetup.driving.train` command.
