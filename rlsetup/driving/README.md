# rlsetup/driving

The driving agent: environment, SAC agent, training pipeline, evaluation. Specs: [docs/specs/](../../docs/specs/README.md).

Pipeline: demo driver collects data, behaviour cloning (BC) initialises the actor, then SAC trains on a vector env, with periodic evaluation.

| File | Role |
|---|---|
| `config.py` | All numbers of the env (D1). Nothing numeric in `env.py` |
| `env.py` | `DrivingEnv`: ξref-following driving env on OCD. Reward modes `ref_tracking` (default) and `time_pbrs` (R1) |
| `reference.py` | ξref geometry and precomputed preview grid per track |
| `qss_profile.py` | Quasi-steady-state speed profile (the 133.9 s benchmark) |
| `demo_driver.py` | Scripted driver for demos and settling; pedal/action mapping |
| `demos.py` | Demonstration dataset, cached by hash of its source files |
| `bc.py` | Behaviour cloning of the SAC actor |
| `policy.py` | SAC policy with LayerNorm critics |
| `agent.py` | `StagedSAC` (freezable actor and entropy coefficient), model factory, output-bias init |
| `train_config.py` | All numbers of D2 and the presets (`smoke`, `full`, `r1_warm`, ...) |
| `train.py` | Training CLI and pipeline: demos, BC gate, critic warm-up, SAC, rollback, resume |
| `callbacks.py` | SB3 callbacks: periodic eval, checkpoints, plateau and regression guards |
| `evaluate.py` | Flying-lap evaluation, probe episodes, Q vs Monte Carlo, lap metrics |
| `vec.py` | Subprocess vector env (must not import torch) |
| `relabel.py` | Relabel an old replay buffer to the R1 reward |

Siblings in [rlsetup/](..): `runs.py` (run folders and JSON files), `provenance.py` (code hashes recorded with each result).
