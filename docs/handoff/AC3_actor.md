# AC3 source actor bundle

Implemented the approved bounded source-actor export and CPU runtime. No game
control, observation adapter, AC control mapping, critic/replay transfer or training
is included in this stage. The actor and collector are intended to run on the
Windows AC PC; Mac inference is only a temporary test. Subsequent AC learning uses
fresh critics/replay and R1 first, as approved by Ibrahim.

## Files

- `rlsetup/assetto/actor.py`: CLI export, strict manifest/weight validation and
  standalone CPU inference with NumPy/Torch.
- `tests/test_ac_actor.py`: actual SB3 `Actor` fixture parity and rejection tests.
- This handoff.

No source checkpoint/configuration or `ocd_first_lap` files were changed. No commit
was made by this coder.

## Export

From the repository root, with an output directory that does not yet exist:

```sh
.venv/bin/python -m rlsetup.assetto.actor export \
  --checkpoint runs/mac/D2/bcn/seed0/checkpoints/best.zip \
  --config runs/mac/D2/bcn/seed0/config.json \
  --output /private/tmp/bcn_actor_new
```

The bundle has three files: `actor.pt`, the byte-identical source `config.json`,
and `manifest.json`. The manifest is written last. It records the source checkpoint,
source config and actor-weight SHA256 hashes; source training step/car/track/setup/
seed/preset; network; every source observation scale and column; history ordering;
and the source action contract. The directory must be created exclusively; failed
writes may leave an incomplete directory that the loader rejects.

The export reads JSON literals in the ZIP `data` member and loads `policy.pth`
using `torch.load(weights_only=True, map_location='cpu')`. It never decodes the
SB3 `:serialized:` metadata fields or unpickles them. Only the ten supported actor
tensors enter the output. Neither optimizers nor the replay buffer are read. The
export reads policy tensors from the checkpoint but constructs no critics or SB3
objects. Runtime imports neither SB3 nor OCD.

The supported source is deliberately bounded to the Catalunya `sf70h_tum` source
contract: ReLU MLP 159→256→256→256, two 256→3 heads, no gSDE, no learned features,
fixed source ObsConfig, gear absent, 25 Hz decisions and pedal deadzone enabled.
Other contracts are rejected, including missing observation fields. No activation
is inferred from weight shapes.

## Runtime

Copy all three bundle files and the repository's `rlsetup` package to the Windows
PC. The runtime module needs Python, NumPy and Torch; it does not require a loaded
simulator or SB3. The repository-wide installation still lists its existing RL
dependencies; AC3 does not change packaging.

```python
import numpy as np
import torch
from rlsetup.assetto.actor import ActorRuntime

actor = ActorRuntime('bcn_actor_new')
# obs must already contain all 159 correctly constructed, scaled source inputs.
obs = np.zeros(159, dtype=np.float32)  # shape-only example, not a valid driving state
policy_action = actor.predict(obs)
rng = torch.Generator(device='cpu').manual_seed(123)
sampled_action = actor.predict(obs, deterministic=False, generator=rng)
```

Observations must be finite `np.float32[159]` within [-10, 10]. The API does not
construct, pad or rescale them. The three outputs are finite float32 normalized
policy actions in [-1, 1], ordered steering/throttle/brake. **They are not AC
controls.** The manifest's 0.6 rad steering lock (positive left) and
`clip((a+0.5)/1.5,0,1)` pedal formula describe the source environment only; AC
calibration and telemetry-to-observation mapping remain separate work.

The CPU actor is frozen, in evaluation mode and uses `torch.inference_mode()`.
Every Linear and stochastic noise allocation specifies CPU float32 explicitly,
including when the caller has changed Torch's default dtype/device.
Its distribution clamps log standard deviations to [-20, 2]; deterministic output
is `tanh(mu)`, stochastic output is `tanh(mu + exp(log_std)*epsilon)`. Stochastic
calls require an explicit seeded CPU Torch generator. Construction and inference
preserve the global Torch RNG. Matching floating-point results across different
Torch builds/CPU architectures has not been measured. No MPS/CUDA execution is used.

## Actual verification

```text
$ .venv/bin/python -m pytest -q tests/test_ac_actor.py
..............................................                           [100%]
46 passed in 2.35s

$ .venv/bin/python -m pytest -q tests/test_ac_actor.py tests/test_ac_telemetry.py tests/test_ac_management.py tests/test_ac_preflight.py
........................................................................ [ 51%]
...................................................................      [100%]
139 passed in 1.78s
```

Tests compare pre-squash means, clamped log standard deviations, deterministic
actions and same-seeded stochastic actions with SB3's real `Actor`, including
observations at both clip limits and head values that force clamps/saturation.
They also exercise unsupported source architectures/features, missing scales,
invalid manifests/versions/dimensions/hashes/tensors/inputs, incomplete bundles,
exclusive output creation and RNG preservation. Six regressions change the
caller's defaults to float64, meta device, or both, during or after construction;
parameters and actions remain CPU float32 with identical predictions, and caller
defaults remain unchanged. A subprocess imports and predicts
while forbidding SB3/OCD imports.

Actual source CLI export and reload succeeded at:

```text
/private/tmp/ac3-source-smoke-w_nymdeh/bcn_actor
source checkpoint SHA256: d690a6fb89da8cdb1cc9addeee7058264b095498fb194fc0c3829ed1a14d4e05
source config SHA256: 49262f508549a56d9447813533ba4650d676875f1a9759a44af3fb4ad21632bf
actor weights SHA256: 738814604b2f0b860e337321117219c407646f3efa2de8b1b3aa611f5a6352ba
source num_timesteps: 5050000
zero observation action: [-0.10398370027542114, 0.5637003183364868, -0.8526345491409302]
source_files_unchanged: True
```

After the defaults-isolation correction, this same actual-source bundle reloaded
under simultaneous float64/meta defaults. All parameters remained CPU float32,
the deterministic zero-observation action remained identical, and stochastic
inference with CPU generator seed 123 returned float32
`[-0.11121567338705063, 0.5856908559799194, -0.887861430644989]`.

The reported source best lap is 83.968529 s; no lap was driven or remeasured in
AC3. The zero-vector prediction verifies loading/inference only. This stage does
not claim that the source actor can drive AC without an observation adapter and
measured calibration. No Windows execution was run by this coder.
