# AC4 stationary steering calibration

This module implements the approved bounded stationary probe, independently of
actor loading, driving, and learning. It has not been run against the live game.
Default operation prints an offline plan without opening sockets:

```sh
.venv/bin/python -m rlsetup.assetto.calibration --full-range
```

When the user is at the PC controls, the car is stationary in visibly displayed
**N**, the intended default setup is selected, and other AC Gym clients are stopped:

```sh
.venv/bin/python -m rlsetup.assetto.calibration \
  --host HOST --output /private/tmp/ac_steering_NEW.json \
  --trace /private/tmp/ac_steering_NEW.jsonl --full-range \
  --execute --neutral-confirmed --at-controls \
  --setup-id default_user_confirmed
```

Use the current Windows LAN address (or `AC_HOST`); no IP is embedded. Both output
paths must be new. On Windows, use Windows paths and the chosen Python executable.
No live action is authorized by this handoff. `--neutral-confirmed` records the
user's visible N check; `--at-controls` records readiness to stop AC manually.
The upstream server has **no control watchdog**: the final center/released pedals
are best effort and cannot protect against a process/network/PC failure.
There are no reset or shift commands. A subscription replaces the current client.

Before UDP, read-only management queries require the measured 100 Hz server,
25 Hz telemetry, server-side vJoy configuration. The map binds car, track,
configuration, user-supplied setup label, management dictionary SHA-256 hashes,
protocol source commit and raw trace SHA-256. Installed source revision stays
unknown. `--source-config FILE` and `--source-checkpoint FILE` optionally hash
local artifacts without loading them. A setup label records user confirmation;
it is not a measured setup-file hash.

Baseline collection requires at least 10 advancing samples spanning 0.4 server
seconds within two local seconds, with speed below 0.05 m/s, actualGear=1,
gas at most 0.01, valid four-wheel heading vectors and stationary world position.
The first command centers/releases both pedals. The next centers/applies full
brake; measured gas <=0.01 and brake >=0.9 must verify within 0.4 server seconds
and 0.8 local seconds before any steering. All subsequent probes retain released
gas/full brake.

Full range uses 0, +.1, +.25, +.5, +1, 0, -.1, -.25, -.5, -1, 0, each held for
0.8 server seconds; the control session has a 15 local second limit. The default
small probe uses 0, +.1, 0, -.1, 0 and **cannot certify a full map**. The last 0.25
server seconds of each phase need at least four usable samples. The 0.005 rad
settling/center/rear-heading tolerances are technical measurement guards, not
identified vehicle parameters. Motion over 0.02 m, speed >=0.05 m/s, gear changes,
gas, invalid/missing/nonfinite fields, malformed packets, counter/timestamp
regression, stale telemetry over 0.2 local seconds or write failure aborts.
Local receive deadlines and server phase clocks have separate origins. UDP is
connected to the explicit resolved endpoint and draining is bounded to 32 packets.

Wheel headings are vectors, projected into `(AC z, AC x)`. The reported angle is
the front circular mean minus rear circular mean, with neutral baseline subtracted
exactly once. Backward vectors cancel in this relative angle. Repeated centers
must agree before merging. Nonzero responses must be monotonic, allowing reversed
command sign. The map stores the complete measured domain, sorted with its paired
commands; no fit, fill or extrapolation supplies unmeasured angles. Small probes
and failed runs have `validated=false`.

The driver's shared JSON contract is `schema_version=1`, `validated=true`,
`angle_rad` (ascending baseline-subtracted measured radians), paired `command`,
`baseline_angle_rad` (raw neutral metadata), `car_name`, `track_name`,
`track_configuration`, `setup_id`, `sampling_hz=25`. Full-range calibration does
not imply the target car reaches the source policy's 0.6 rad steering limit.
Trace JSONL contains raw telemetry, command intent with both clock values, phases,
management replies and stop outcome. Cleanup attempts center/release on the same
UDP socket with its latest accepted counter, then disconnects.

Offline verification:

```sh
.venv/bin/python -m pytest -q --noconftest tests/test_ac_calibration.py
```

Implementation result: `30 passed in 0.07s`. Fake sockets cover complete
full-range measurement, reversed signs, baseline/toe/rotation/vector rejection,
map pairing, monotonicity, guards, cleanup after malformed data and handshake
failure (including identification received after its deadline), failed final release,
exclusive output and offline plan. No game actions were performed.

Protocol inspected locally at pinned upstream commit
`ef0fa502bdd257fc2affad9361c02b3dbdfe1632`: `ego_server.py`, `car_control.py` and
`structures.py` in `assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/`.
