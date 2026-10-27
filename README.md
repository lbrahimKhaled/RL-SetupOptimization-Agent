# RL-SetupOptimization-Agent

Thesis project (Ibrahim Khaled, AUB): find the F1 car setup that gives a driver the fastest lap, model-free with RL.
Two agents: a **setup agent** (TD3, outputs setup p) and a **driving agent** (SAC, outputs steering / throttle / brake, never sees the setup).
Simulator: Open Car Dynamics via the sibling repo `ocd_first_lap`. Track: Shanghai. Car: SF70H-like.

**Branch `d1-driving-env` covers the driving agent only.** The setup agent is not started.

## Status (27 Oct 2026)

| Step | What | State |
|---|---|---|
| D1 | Driving env (`DrivingEnv`, obs 159, 3 actions, 25 Hz) | Done, 50+ tests |
| D2 | SAC driver: behaviour-cloning init, then SAC | Done. `full1` seed 0: best clean deterministic lap **122.85 s** (QSS reference 133.9 s) |
| R1 | Time-optimal reward (`time_pbrs`), warm start from `full1` | Done, **negative result**: 123.04 s, more crashes. No gain over `full1` |
| D2 seeds 1, 2 | Seed variance for the done criterion | Not run |
| D3, D4, setup agent | c sweep, fine-tune protocol, setup agent | Not started |

Results: [docs/results/](docs/results/). Decisions (#1 to #20): [docs/DECISIONS.md](docs/DECISIONS.md).

## Where to look first

| To review | Read |
|---|---|
| What the project is and why | [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) |
| Every decision and its reason | [docs/DECISIONS.md](docs/DECISIONS.md) |
| What was run and what came out | [docs/results/](docs/results/README.md) |
| Specs and architect reviews | [docs/specs/](docs/specs/README.md) |
| Resume instructions | [docs/handoff/HANDOFF.md](docs/handoff/HANDOFF.md) |
| The code | [rlsetup/driving/](rlsetup/driving/README.md) |

## Layout

| Folder | Use |
|---|---|
| [rlsetup/](rlsetup/driving/README.md) | Package: driving env, SAC agent, training, evaluation; run folders and provenance |
| [scripts/](scripts/README.md) | CLIs: build QSS profile, collect demos, run the full D2 job, reports, benchmarks |
| [platforms/](platforms/README.md) | Per-machine launchers, environment notes and overrides: `mac/`, `windows/` |
| [tests/](tests/README.md) | pytest suite (fast + slow) |
| [data/](data/README.md) | Cached QSS speed profile and demo datasets |
| [docs/](docs/README.md) | Context, decisions, specs, research, handoff notes, results |
| `runs/` | Training output (git-ignored): `runs/mac/D2`, `runs/windows/D2`; `runs/D2` holds runs from before the split |

## Quick start

Needs Python 3.12, `ocd_first_lap` built and importable, and SB3 2.9.0 (pinned in [pyproject.toml](pyproject.toml)).

```bash
python -m pytest -q -m "not slow"                       # fast tests, a few seconds
python -m pytest -q                                      # full suite
python -m rlsetup.driving.train --preset full --seed 0 --run-name NAME
python -m rlsetup.driving.train --preset r1_warm --seed 0 --run-name r1_warm \
    --init-from runs/D2/full1/seed0/checkpoints/best.zip
python scripts/seed_summary.py runs/D2/NAME             # progress table
```

## Mac vs Windows

Same code and presets on both machines. Each machine gets its own launcher, overrides and run root ([platforms/](platforms/README.md)):

| | Mac (M4) | Windows (i9-14900 + RTX 4000 Ada) |
|---|---|---|
| Python | `.venv/bin/python` | conda `C:\Users\imk14\ocdenv` |
| Overrides | cpu, forkserver, n_envs 4, gradient_steps 2, torch_threads 1 | cuda, spawn, n_envs 16, gradient_steps 8, torch_threads 2 (provisional, not tuned yet) |
| Runs | `runs/mac/D2` | `runs/windows/D2` |
| Launch | `platforms/mac/train.sh`, `platforms/mac/run_d2_full.sh` | `platforms\windows\train.ps1`, `run_d2_full.ps1`, detached via `start_detached.ps1` |

Windows test status and the open car question: [docs/handoff/W1_windows_port.md](docs/handoff/W1_windows_port.md).

## Caveats for the reviewer

- Working tree has **uncommitted edits** in `rlsetup/driving/agent.py` and `train.py`: device switched from `cpu` to `cuda`. Revert or commit before pinning a result to a hash.
- Brake torque (12 kN·m) is a placeholder until real SF70H data exists.
- The ξref is the safe minimum-curvature line, a placeholder for Jad's laps.
- `ocd_first_lap` carries a speed patch ([docs/ocd_patches/](docs/ocd_patches/0001-fast-rear-axle-omega.diff)) that must be applied there; it is bit-identical.
