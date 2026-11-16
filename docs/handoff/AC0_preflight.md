# AC0: read-only Assetto Corsa transfer inventory

The architect approved this bounded diagnostic stage on branch
`codex/ac-barcelona-transfer`. `scripts/ac_preflight.py` inspects explicitly supplied
installation and run directories and prints JSON. It requires Python 3.8 or later
and the standard library only; Git is needed to identify an AC Gym checkout.

## Use on the Lenovo

Copy `scripts/ac_preflight.py` to the Lenovo and run it with the Python used for AC
Gym. Its installation path was supplied as `C:\Users\B-Test\assetto_corsa_gym`:

```powershell
python .\ac_preflight.py --ac-gym-root "C:\Users\B-Test\assetto_corsa_gym"
```

To inspect a run on any Windows machine, supply its actual run directory:

```powershell
python .\ac_preflight.py --run-root "C:\path\to\run\seed0"
```

`--run-root` can be repeated, or can name a runs tree containing multiple
`config.json` files. There is no default directory scan. Both path options may be
used together. The command reports package versions from the Python environment
that executes it, so use the intended AC Gym environment when inspecting Lenovo.

## Use on the Mac

From this repository:

```sh
.venv/bin/python scripts/ac_preflight.py --run-root runs/mac/D2/bcn/seed0
```

JSON goes to stdout. Exit status 0 means inspection completed without reported
errors or invalid evaluation rows; 1 means the JSON contains issues to review;
2 means invalid command-line usage. Missing optional checkpoints, provenance, or
evaluation logs are reported as absent and do not alone cause status 1.

## What the report establishes

- Host OS, architecture, Python and selected package versions, without importing
  PyTorch, Gym, the simulator, or AC Gym.
- AC Gym Git commit and dirty flag. A checkout does not establish that the game
  plugin is installed or active; that is a separate live observation. YAML config
  candidates are listed by path and their contents are not parsed.
- Explicitly saved car, track, setup, reference, reward, observation and agent
  fields. Omitted fields remain `null`; no current defaults are substituted and
  no identity is inferred from a folder or preset name.
- Presence, size and SHA-256 of `best.zip`, `best_aided.zip` and `last.zip`.
  Only literal numeric `num_timesteps`, `gamma`, `n_steps` and `batch_size` from
  the ZIP's JSON `data` member are inspected. Serialized policy classes, spaces,
  pickle/base64 fields and model tensors are never loaded or executed.
- Best logged positive clean deterministic `lap2_s` and its `step`, using the
  existing `evals.jsonl` schema. Invalid rows are counted. This does **not** prove
  that a checkpoint corresponds to that row; the association remains explicitly
  unverified in the report.

The script performs no training, action selection, game control, network
discovery, dependency installation or writes to the inspected trees. Git runs
with optional locks disabled and the filesystem monitor disabled. It does not
report Git remotes, arbitrary config fields or full provenance contents.

## Validation performed

```sh
.venv/bin/python -m pytest -q --noconftest tests/test_ac_preflight.py
```

Result: **22 passed in 0.19 s**. Tests cover missing/malformed input, omitted
identity, R1/R3/R4 field preservation, safe checkpoint inspection including
`best_aided.zip`, strict clean-lap filtering, read-only behavior, and a standalone
subprocess with site packages disabled. `--noconftest` keeps this diagnostic test
independent of the existing OCD fixtures. Windows execution has not been tested
by this implementation agent.

A read-only check of the actual local artifacts returned:

| Run | Saved car / track | Best logged clean lap 2 | Checkpoints present |
|---|---|---|---|
| `runs/mac/D2/bcn/seed0` | `sf70h_tum` / Catalunya | 83.968529 s at 5,050,000 | `best.zip`, `last.zip` |
| `runs/mac/D2/f1_r4_scratch/seed0` | `sf70h_f1` / Shanghai | 101.001791 s at 7,900,000 | `best_aided.zip`, `last.zip` |

Both inspected runs had zero inspection errors and zero invalid evaluation
rows. These are source-run measurements, not Assetto Corsa results. Ibrahim has
since accepted the local `bcn/seed0/checkpoints/best.zip` (`sf70h_tum`, Catalunya)
as the transfer source; this script does not configure or select it automatically.

## Remaining transfer work

The AC observation/action mapping, telemetry units and timing, policy execution
location, reward/target profile, transfer scope and CPU/MPS measurement belong
to the subsequent wrapper and fine-tuning specification. Live connection and
checkpoint transfer have not been implemented in AC0. The supplied Lenovo host
address must remain configurable in that later stage.
