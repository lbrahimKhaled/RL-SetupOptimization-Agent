# AC actor transfer and offline learning

The Windows collector runs a fixed portable actor for a lap, releases controls,
and exports numeric NPZ transitions. Copy the chunk to the Mac. The Mac learner
never connects to AC and never constructs an OCD simulator. This is a real SAC
update path; driving performance still requires calibration and actual AC laps.

Start from the verified Barcelona `sf70h_tum` portable actor. Only its ten actor
tensors transfer. AC critics, target critics, Adam optimizers, alpha (initial
0.1), and replay start fresh. The existing `StagedSAC` and `LNSACPolicy` provide
the source three-layer 256-unit actor and LayerNorm critics. Reward mode is R1
`time_pbrs`, gamma 0.992. Replay is deliberately **one step**, so chunk/lap
boundaries never join into artificial multi-step returns.

```sh
.venv/bin/python -m rlsetup.assetto.learn \
  --actor-bundle /path/to/source_actor \
  --chunk /path/to/lap.npz \
  --output /path/to/new_transaction \
  --critic-updates K --updates N
```

Replace K and N with explicit nonnegative integer budgets; neither has an implicit
training default. Critics train with the actor and entropy coefficient frozen
for K updates, then all train for N updates. These budgets remain experimental;
no production schedule is being selected here. The default batch is 512; a
chunk/replay must hold at least one batch. `--batch-size`, `--buffer-size`,
`--seed`, and `--device cpu|mps` are available. CPU with one torch thread is the
default, following the measured local update benchmark (about 152 versus 50
updates/s on the compared GPU path); it avoids GPU overhead for this small net.
Learning rates retain 3e-4 for critics/alpha and the approved R1 1e-4 actor rate.

Each output is a new directory containing `model.zip`, `replay.pkl`, RNG state,
`actor/`, and a completion `manifest.json` written last. Do not modify these
files. Incomplete transactions are rejected. The full config, source manifest,
git hash, file hashes, replay chunk hashes, target identity, and update counts
are stored alongside the checkpoint. These hashes detect corruption; they do
not authenticate files from untrusted sources. **Resume only your own trusted
local outputs**: SB3 model and replay serialization uses pickle internally.
The loader validates the completion manifest and every serialized-file hash
before invoking SB3 loaders. It never accepts an OCD/source policy checkpoint
as a resumable AC learner.

Resume with the previous transaction and its exported actor:

```sh
.venv/bin/python -m rlsetup.assetto.learn \
  --resume /path/to/previous_transaction \
  --actor-bundle /path/to/previous_transaction/actor \
  --chunk /path/to/new_lap.npz \
  --output /path/to/next_transaction \
  --critic-updates K --updates N
```

Keep config flags identical across resumes. Replay carries forward, and the same
chunk SHA is never imported twice. New chunks must name the actor used for their
collection and retain the same target/setup/reference/timing contract. Episodes
may change only after a terminated/truncated row; continuing rows require exact
next-observation continuity and sequence increments. Every chunk must finish at
a boundary. A true termination suppresses bootstrap; a pure truncation bootstraps
from the stored final next observation via SB3 `TimeLimit.truncated`.

Copy only the complete `actor/` portable bundle back to Windows. Select it between
laps; the collector keeps weights fixed during a lap. It contains no critics,
optimizers, replay, or SB3 pickles. Its optional hashed `training.json` explicitly
labels the weights as AC-trained and records their source contract, parent actor,
chunks, target and update budgets. Legacy source bundles still load unchanged.

Seeds cover Python, NumPy, torch and model initialization. CPU RNG states are
saved for resumed runs. MPS is optional: its kernels and RNG continuation are
not promised bit-identical. No actual-game result follows from synthetic unit
checks. Run isolated checks with:

```sh
.venv/bin/python -m pytest -q tests/test_ac_actor.py tests/test_ac_learn.py
```
