# AC transfer quickstart

## 1. Prepare the actor and connection

Run commands from this repo root. The current test runs inference on the **Mac**
and connects to the AC PC. The source bundle already exists; export only if missing:

```sh
mkdir -p runs/ac
if [ ! -d runs/ac/source_actor ]; then
  .venv/bin/python -m rlsetup.assetto.actor export \
    --checkpoint runs/mac/D2/bcn/seed0/checkpoints/best.zip \
    --config runs/mac/D2/bcn/seed0/config.json --output runs/ac/source_actor
fi
export AC_HOST=172.20.10.2
.venv/bin/python -m rlsetup.assetto inspect > runs/ac/inspect-001.json
```

Check the inspection identifies `ks_ferrari_sf70h`, `ks_barcelona`, `layout_gp`,
25 Hz ego telemetry, and server-side vJoy. Use the PC's current address if it changes.
Only one AC Gym subscriber may be connected.

For production, copy the repo and actor bundle to the **Windows AC PC**. Use a
separate Python 3.12 terminal environment, not AC's embedded Python 3.3:

```powershell
py -3.12 -m venv .venv-ac
.\.venv-ac\Scripts\Activate.ps1
python -m pip install numpy torch
$env:AC_HOST='127.0.0.1'
```

The PC collector needs NumPy/PyTorch; no OCD, SB3, or project installation is needed
when launched from the repo root. Below, replace `.venv/bin/python` with `python`
on Windows and put each command on one line. Learning remains on the Mac.

## 2. Calibrate, then collect one bounded chunk

Load the confirmed default setup. **Before calibration, visibly confirm N and zero
speed**, release pedals, and stay at the controls ready to stop AC. Disconnect other
clients. This command actuates the full stationary steering grid and brake:

```sh
.venv/bin/python -m rlsetup.assetto.calibration --full-range \
  --output runs/ac/steering.json --trace runs/ac/steering-trace.jsonl \
  --execute --neutral-confirmed --at-controls
```

Continue only if the result reports `validated: true`. A small probe cannot produce
a validated map. The stationary map verifies steering response and range; moving
handling is still untested.

Now select a forward gear manually, face forwards with all wheels on track, and
release both pedals. Remain at the controls; the server has no control watchdog.
Removing `--execute` performs local validation without connecting or controlling.

```sh
.venv/bin/python -m rlsetup.assetto.driving collect \
  --actor-bundle runs/ac/source_actor --calibration runs/ac/steering.json \
  --reference centerline --steering-saturation clip \
  --max-steps 2500 --max-wall-s 120 --output runs/ac/chunk-001.npz --execute
```

The actor stays fixed throughout collection. Explicit `clip` saturates steering at
measured endpoints and records saturation; it never extrapolates. The collector
stops at a lap, failure, timing fault, or budget. Wheel-off-track means AC reports
at least one tyre out; it does not certify the whole body stays inside. On a timing
gap or invalid next observation, the pending transition is discarded. Check the
reported transition count and stop reason; startup failures can produce empty chunks.
If the car stalls, stop manually or let the short budget end, then investigate.

## 3. Learn on Mac and transfer the next actor

Copy the NPZ from Windows to the Mac when collecting on the PC. This **smoke example**
uses 100 critic-only updates followed by 100 joint updates, batch size 64. These are
test budgets, not a selected production training schedule. Supply at least 64 valid
transitions; repeat `--chunk PATH` to combine compatible chunks from the same actor.
The AC learner starts fresh critics/replay and uses one-step targets (`n_steps=1`).

```sh
.venv/bin/python -m rlsetup.assetto.learn \
  --actor-bundle runs/ac/source_actor --chunk runs/ac/chunk-001.npz \
  --critic-updates 100 --updates 100 --batch-size 64 \
  --output runs/ac/update-001
```

The command saves learner/replay state and exports `runs/ac/update-001/actor`.
Copy that entire actor directory to the PC. Manually pause/reset/reposition between
chunks as needed, restore the forward-gear/released-pedal handover, then repeat the
collector command with `--actor-bundle runs/ac/update-001/actor` and a new output
`runs/ac/chunk-002.npz`. The wrapper does not reset or pause AC. Never replace a
bundle during collection.

Copy the next chunk to Mac and resume the same learner transaction:

```sh
.venv/bin/python -m rlsetup.assetto.learn \
  --resume runs/ac/update-001 --actor-bundle runs/ac/update-001/actor \
  --chunk runs/ac/chunk-002.npz --critic-updates 100 --updates 100 --batch-size 64 \
  --output runs/ac/update-002
```

Every output path must be new. Use the newly exported `update-002/actor` for the
next chunk. Offline tests and a 71-sample live telemetry capture have been verified;
no live policy-driven AC lap or successful transfer training has been demonstrated.
