# R5 AC/OCD mixed-replay fork

This experiment continues the current R5 AC learner from a **committed** cycle. Its model,
critic and target networks, entropy coefficient, optimizers, RNG state, imported chunk hashes,
cycle and scheduling carry are retained. The parent run stays separate. The fork does not
perform an update until a new AC chunk arrives.

The total replay capacity is 300,000 transitions. All AC transitions retained by the parent
snapshot are placed after a uniform sample of eligible OCD transitions filling the remaining
slots. Later AC transitions overwrite OCD first; once none remain the buffer is ordinary FIFO
AC replay. The learner still samples the combined buffer uniformly. Per-slot labels and
counts identify each domain and survive save/resume. Progress records include `replay_ac` and
`replay_ocd` for mixed cycles. This is an experiment: neither transferring from OCD to AC nor
mixing the two reward distributions is guaranteed to improve performance.

The actor/critic initialization used for the original AC run was the OCD R5 **7M** checkpoint.
The OCD replay selected here is the original R5 training buffer saved at **10.35M** steps;
these are different points in training. The source config matches the AC actor's source config.
OCD's stored one-step observations, actions, rewards, dones and timeout flags are copied
without reward relabelling. Its three-step sampler is not copied. Retry warmup rows are
excluded, and effective source corner bonuses must be zero. The mix uses independent seeded
sampling without replacement. The manifest records source file/config/status hashes, source
replay step, selection, parent receipt/cycle, and initial and current domain counts.

## Cutover

The Lenovo's original worker must stop first so its last cycle is committed. Confirm with the
Mac learner that no chunk is pending; the builder refuses a snapshot with pending chunks.
The snapshot captures one committed receipt, so any later parent cycles are not part of the
fork. Source and fork files are local trusted artifacts; the replay source is a Python pickle.

On the Mac, from the repository root, create the fork once:

```sh
.venv/bin/python -m rlsetup.assetto.mixed_replay \
  --parent runs/ac/tcp-f1r5-001 \
  --output runs/ac/tcp-f1r5-mixed-001 \
  --source-replay runs/mac/D2/bcn_f1_r5/seed0/replay_buffer.pkl \
  --source-config runs/mac/D2/bcn_f1_r5/seed0/config.json
```

The command prints the exact parent cycle and retained AC/OCD counts. It refuses an existing
destination, validates the saved learner hashes, source config/replay, reward bonus, replay
shapes and flags, and installs the new folder only after the bootstrap has been written.
The fork genesis records and verifies the parent's exact receipt and state; no historical
parent receipt is rewritten. Re-run failures only after checking the printed error.

Start `runs/ac/tcp-launch/START_MAC_F1R5_MIXED.command` on the Mac. It serves the fork on
port **2351** and refuses to start without the genesis. The original R5 learner uses port
2350. On the Lenovo, extract `runs/ac/lenovo-f1r5-mixed-launch.zip` into the parent of the
existing `ac-lenovo-tcp` folder. In a Command Prompt in that folder, set `AC_LEARNER_HOST` to
the Mac's current IP and run `START_LENOVO_F1R5_MIXED.cmd`. The CMD uses the existing v15
worker, steering calibration, Python environment and reference; it creates a separate
`tcp-worker-f1r5-mixed-001` work directory. If its heights file is missing and the R5 one
exists, the script copies it. A fresh worker performs `hello` to receive the fork's current
cycle and actor; never point the old worker state at the new server.

## Verification and current state

Focused QA reported **176 tests passed**. A disposable real-data fork at parent cycle 203
retained 54,977 AC transitions and added 245,023 OCD transitions. A synthetic seven-transition
worker chunk produced one update, saved and resumed as cycle 204, with 54,984 AC and 245,016
OCD transitions; checked parent hashes remained unchanged. These are smoke-test values,
not the final cutover counts. The final cutover and live mixed drive were pending at handoff.
