# AC transfer quickstart

## Automatic Lenovo worker and Mac learner

`python -m rlsetup.assetto.distributed server` runs the Mac learner TCP endpoint;
`worker` runs inference and AC controls on Lenovo. Each needs the same private
`--token-file`, a persistent `--work-dir`, and explicit `--max-cycles`. The server
also requires `--actor-bundle`, `--critic-updates`, `--updates`, `--batch-size`;
the worker requires `--server`, `--calibration`, `--max-steps`, `--max-wall-s` and
`--execute`. The packaged launchers provide a finite pilot configuration.

The worker automatically brakes to stationary, sends AC's management reset once,
verifies on-track forward-facing telemetry, observes each shift into first, then
releases pedals for collection. While TCP transfer/learning waits, a single local
UDP session holds full brake and consumes telemetry. Failed readiness stops the
run. Final braking is best effort; the upstream server has no control watchdog.
The game must stay running and the matching default setup/calibration must be
loaded; calibration is not repeated automatically.

Readiness uses separate phases. Reset transients are braked and allowed to settle
within the preparation timeout; collection requires at least 0.4 s after reset and
three consecutive samples below 0.05 m/s, released gas, full brake, zero tyres out
and forward heading. Each shift is observed and followed by another stationary
check. Once ready, movement or loss of these conditions aborts the parked wait.
Released-pedal handoff has a one-second deadline and allows less than 1 m/s creep
while confirming first gear and two released-brake samples. This creep limit is a
diagnostic tolerance, not a validated vehicle parameter. Each attempt writes an
exclusive `readiness-*.jsonl` in the worker directory, including raw telemetry,
commands, phase changes, peak speeds, displacement and startup-failure evidence.
AC's road reset can invalidate a Hotlap; that game message alone is separate from
the collector's off-track failure. Game penalties and driving thresholds are unchanged.
Automatic collection retains full brake on normal exit, error and interruption,
including the gap while the next parked session performs management queries. The
manual collector retains its original released-pedal cleanup. TCP frames have an
absolute receive deadline as well as payload bounds; invalid chunks cannot commit
a cycle or establish incompatible pending replay. Shared tokens are omitted from
status logs.
Braked waits in `BRAKE_TO_STOP`, `RESET_SETTLING` and `READY_HOLD` permit bounded telemetry coalescing only when no gear pulse or cleared-flags confirmation is pending: an
advancing packet must arrive within 0.25 s, with increasing counters/timestamps.
Duplicates do not refresh that deadline, and every drained packet is checked and
recorded. A gap or skipped packet clears release qualification; three subsequent
normal contiguous samples are required before a shift, readiness or handoff. Reset
motion, released brakes and geometry transients may settle within the existing
preparation timeout; certification still uses the original **3D speed below
0.05 m/s**, full brake and released gas, with on-track forward heading after reset.
Vertical bouncing is not treated as stationary based on a smaller planar speed.
Longer silence aborts with full brake retained. Pending gear pulses and their
observed confirmation/cleared interval, released-pedal handoff and active driving
keep their original strict 20–80 ms timing and contiguous counters. No replay
transition is reconstructed across a timing gap.
The executed worker limits PyTorch inference to one CPU thread and prints the
actual count once. This limits later inference load; it does not establish that
inference caused a reset-time telemetry pause, which can occur before actor loading.

Chunks and actor versions transfer automatically. Pending chunks survive restart,
and committed cycle receipts prevent duplicate training after reconnect. Small
chunks accumulate under the same actor until a batch is available. Restart each
side using the same commands/work directories; do not manually copy per-cycle files.
The existing manual workflow below remains available.

### Full training (run until stopped, resumable)

Launchers: `runs/ac/tcp-launch/START_MAC_FULL.command` (Mac learner, work dir
`runs/ac/tcp-full-001`) and `START_LENOVO_FULL.cmd` (in `lenovo-tcp-full-v9.zip`,
work dir `tcp-worker-full-001`). There is no cycle cap; stop either side with
Ctrl+C and re-run the same launcher to continue from the last committed actor,
learner, replay and RNG state. `--max-cycles` is optional and is not part of the
work-dir identity, so a later restart may add or change it. Schedule (experimental,
not tuned): source actor fixed until 5,000 valid transitions; then 1,000
critic-only updates; afterwards one joint update per 4 new transitions
(`--transitions-per-update 4`, remainder carried). Training rollouts sample with
0.3 x the policy's own pre-squash std (`--explore-std-scale`); every 10th cycle is a
deterministic evaluation rollout (`--eval-every`). Episodes are up to 3,750 steps
(150 s at 25 Hz). Each committed cycle appends a line to `progress.jsonl` (mode,
stop reason, return, progress in metres, updates). Only the newest two learner
snapshots keep their 380 MB replay/model files (`--keep-learners 2`); every
cycle's actor, chunk and receipt are kept. Only consecutive empty chunks count
toward `--max-no-update`.
With `--max-consecutive-failures 5` (Lenovo launcher, package v10) the worker restarts
brake/reset/gear preparation after telemetry stalls, readiness timeouts or lost
AC/learner connections, waiting 5, 10, 20, 40, 60 s; the car keeps the last
full-brake command meanwhile. A completed drive resets the count; the sixth
consecutive failure stops the worker. Server rejections, contract errors and
Ctrl+C are never retried. Each failure is appended to `retries.jsonl` in the
worker directory. The Mac launcher runs the learner under `caffeinate -ims`.

Reference `ocd_path` (launchers from package v11, work dirs `tcp-full-002` /
`tcp-worker-full-002`) gives the driver its OCD training line. On the Mac,
`scripts/ac_track_compare.py` measures AC vs OCD geometry, and
`scripts/ac_ocd_reference.py runs/ac/ac-track.json --output ... --margin 2.0` maps
`Track.path` into AC coordinates by the rigid alignment (0.70 m RMS), one point per AC
border row, at least 2 m from either AC border (OCD's own minimum is 2.19 m). The file is
bound to the exact AC border rows by `ac_borders_sha256`; the collector, readiness
check and learner reject it for other track data, and its replay cannot mix with
other references. `ac-track.json` comes from
`python -m rlsetup.assetto inspect --include-track` on the AC PC.

`--critic-init runs/mac/D2/bcn/seed0/checkpoints/best.zip` (Mac launcher, run
`tcp-full-002`) starts both critics, their targets and alpha (OCD-trained 0.063, not the
fresh 0.1) from the actor's own OCD checkpoint: same R1 reward, gamma and now the same
line; OCD used 3-step targets, AC uses 1-step. Only that exact checkpoint (SHA-256 in the
actor manifest, identical actor tensors) is accepted; no OCD replay or optimizer state is
used. The warmup (5,000 AC transitions, 1,000 critic-only updates with alpha frozen) still
runs. On the first fast-lane AC drives the OCD critic had the right scale on clean driving
but stayed optimistic (+240..270) just before AC crashes whose actual return was about -25;
the critic-only phase is what has to correct that before the actor moves.

The localhost integration harness launches the real worker and learner as separate
processes against a 25 Hz fake AC management/telemetry server, using isolated ports:

```sh
.venv/bin/python scripts/ac_loopback_smoke.py --output runs/ac/loopback-NEW
```

Its new output directory holds `report.json`, subprocess logs, raw simulated
commands/telemetry, chunks, versioned actors and learner receipts. The harness runs
small real gradient updates and checks tensor changes, short/empty accumulation,
pending-chunk restart, a dropped committed response, learner restart, malformed
network input, a reset pedal-release transient, one live UDP owner and final brake
state. It also delays a management reply to check braking between sessions.
This verifies the software loop on synthetic telemetry; it does not establish
real-game driving performance. Runtime/launcher packaging must use matching source
hashes; keep calibration, shared token and existing worker/learner state unchanged
when applying a runtime patch.

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
  --reference fast_lane --steering-saturation clip \
  --max-steps 2500 --max-wall-s 120 --output runs/ac/chunk-001.npz --execute
```

`fast_lane` uses the track's AC AI racing-line geometry, with its corresponding
borders; it does not use AI speed or pedal targets. This is not a certified optimal
or body-safe SF70H line. `--reference centerline` retains the border midpoint.
When changing references, start a fresh AC learner/replay/output rather than
resuming the centerline transaction; references and fast-lane geometry cannot mix.

The actor stays fixed throughout collection. Explicit `clip` saturates steering at
measured endpoints and records saturation; it never extrapolates. The collector
stops at a lap, failure, timing fault, or budget. Wheel-off-track means AC reports
at least one tyre out; it does not certify the whole body stays inside. On a timing
gap or invalid next observation, the pending transition is discarded. Check the
reported transition count and stop reason; startup failures can produce empty chunks.
Failures now report `wheel_off_track`, `wrong_way`, or both, with the native tyre
count and heading error saved as `terminal_evidence` in chunk metadata.
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
