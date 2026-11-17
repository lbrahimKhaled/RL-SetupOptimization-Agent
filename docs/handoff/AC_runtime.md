# Thin AC runtime

The Windows runtime constructs the source actor's 159 normalized inputs from AC
telemetry and the explicitly selected midpoint centerline or AC AI fast-lane geometry. It runs one fixed actor
for a bounded chunk and exports numeric NPZ for Mac SAC updates. No speed targets,
game resets, automatic shifts, or actor swaps occur during collection.

Install this repo and its Python dependencies on the AC Windows computer; actor-only
inference needs NumPy/PyTorch, not OCD. Use `--host localhost` there. A Mac diagnostic
against the PC uses its current LAN hostname/IP or `AC_HOST`; addresses are never fixed
in code. Do not run two AC Gym subscribers together.

Before collection: load the confirmed default setup, select a forward gear manually
(AC exported `actualGear >= 2`), place the car on track facing forwards, and release
throttle/brake. Leave automatic shifting enabled if desired; this wrapper never shifts.
Keep the game running during the bounded chunk. The steering JSON must have a
validated full measured range and matching car/track/layout/setup/sampling identity.
A small stationary probe is not a full calibration.

```sh
python -m rlsetup.assetto.driving collect \
  --actor-bundle PATH_TO_BUNDLE --calibration PATH_TO_CALIBRATION_JSON \
  --host localhost --reference fast_lane --steering-saturation clip \
  --max-steps 250 --max-wall-s 20 --output NEW_CHUNK.npz
```

Without `--execute`, this validates the local bundle/calibration and does not connect.
After reviewing the command and completing readiness checks, append `--execute` for
control. Output must be a new path. Default steering saturation is `reject`; explicit
`clip` saturates requested road angle at measured endpoints, never extrapolates.

Copy the chunk to Mac, run the bounded AC learner, and copy the exported new actor
bundle back to Windows. Use the learner CLI's required gradient/warmup budgets; see
`python -m rlsetup.assetto.learn --help`. Between chunks, pause/reset/reposition the
car manually as needed, restore the deliberate handover state, and launch a fresh
collector with the chosen actor. Never replace a bundle during a collector run.

Both references are geometry guidance only; `fast_lane` does not consume AI speed
or pedal targets and is not a certified optimal or body-safe SF70H line. It uses
its own arc-length stations and corresponding border rows for previews. The
metadata stores reference length and a geometry hash. `centerline` preserves the
previous midpoint reference. Switching reference requires fresh learner replay;
mixed references or changed fast-lane geometry are rejected, including on resume.

Native `numberOfTyresOut >= 1` terminates
as **wheel-off-track**; it is not a whole-body footprint guarantee. Wrong-way heading
also terminates. R1 is progress minus actual elapsed seconds plus potential shaping
with gamma=.992 and Phi=-abs(e); terminal failures use zero next potential and P=100.
Chunk/lap/step/time/interruption truncations preserve the real next potential.
The stop reason distinguishes `wheel_off_track`, `wrong_way` and
`wheel_off_track_and_wrong_way`; `terminal_evidence` retains the native tyre count
and signed heading error in radians. Termination thresholds are unchanged.

Raw world velocity in `(AC z, AC x)` is rotated into the source body frame; local
velocity sign assumptions are avoided. Heading differences give yaw rate, and the
source midpoint rotating-frame formula gives acceleration. Road steering is the
front-minus-rear circular tyre-heading mean minus measured neutral baseline.
Policy steering requests `.6*a0` radians; pedals use the source deadzone followed
by vJoy `2*p-1`. Commands, policy actions, desired/executed steering, server sequence,
and server timestamps are all retained. Policy actions remain the original actions
when the actuator saturates. Metadata records saturation count and calibration hash.

History uses real telemetry only: first sample anchors heading, second anchors body
kinematics, third/fourth fill the two prior ego/action blocks, fifth provides the first
complete observation. Pedals must remain released throughout this handover. Their
source-equivalent history action is explicitly `[0,-.8,-.8]` (both pedals map to zero).
There is no fabricated stationary history.

A sequence gap/reset/duplicate, stale sample, invalid next observation, projection
jump, or timing violation stops collection and discards that pending transition.
The last valid transition is truncated. An empty startup-failure chunk is diagnostic
and cannot train. Every shutdown attempts explicit released steering/pedals before
disconnect because the inspected AC Gym server has no control watchdog. A lost
network link can prevent delivery; maintain manual control access.

NPZ files use only numeric arrays and a Unicode `metadata_json` scalar, loaded with
`allow_pickle=False`. The learner must not connect discontinuous chunks. Publication
uses an exclusive atomic same-directory hard link; filesystems without hard-link
support fail rather than overwrite an existing chunk.

Offline validation: `.venv/bin/python -m pytest -q tests/test_ac_driving.py`.
No live AC lap is established by these synthetic tests.
