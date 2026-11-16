# AC1: read-only AC Gym management client

Implemented the architect-approved inspection stage in `rlsetup.assetto`. It uses
the standard library and the existing AC Gym management server. It imports no AC
controller, simulator, PyTorch or other RL runtime packages.

## Run on the Mac

From this repository, replace `LENOVO_HOST` with the current hostname or LAN IP:

```sh
.venv/bin/python -m rlsetup.assetto inspect --host LENOVO_HOST
.venv/bin/python -m rlsetup.assetto inspect --host LENOVO_HOST --include-track
```

Alternatively set a host for the current shell:

```sh
export AC_HOST=LENOVO_HOST
.venv/bin/python -m rlsetup.assetto inspect --include-track
```

`--host` overrides `AC_HOST`. There is no default host or embedded laptop IP.
Every request resolves the supplied hostname anew, so a later DNS address change
can be used on the next query. This does not discover a laptop's new IP when no
resolvable hostname is available. `--port` defaults to the upstream management
TCP port 2347. `--timeout` sets a deadline per query, default 5 seconds, with a
validated diagnostic range `(0, 300]` seconds.

Output is JSON on stdout. Exit 0 means all requested queries succeeded; exit 1
means the report contains query errors; exit 2 means invalid CLI/configuration.
The CLI does not write output files; shell redirection can save a report to a
chosen destination. No checkpoints or AC configuration files are changed.

## Protocol and bounds

The only accepted commands are `get_config`, `get_static_info`, and
`get_track_info`. Configuration and static information are requested by default;
track geometry is requested only with `--include-track`.

Each request uses a fresh TCP connection and sends the exact UTF-8 command with
no newline or length prefix. The upstream server keeps the socket open. This
client accumulates fragments until a complete dictionary parses, then closes
its socket without waiting for server EOF. Content trailing a dictionary in the
received buffer is rejected. The upstream protocol cannot distinguish later
unsolicited bytes from a future reply because it has no response framing.

Config replies are Python literals, parsed with `ast.literal_eval` after byte,
AST-node and depth limits. Static/track replies are strict JSON. All replies must
be dictionaries containing finite, JSON-compatible data; code expressions,
nonfinite numbers and unsupported literal objects are rejected. Truncated or
malformed messages never produce accepted partial geometry.

Default byte caps are 1 MiB for config/static and 8 MiB for track data. These are
diagnostic transport limits, not RL hyperparameters. Library callers may set
validated caps up to 64 MiB through `ConnectionConfig`. Config ASTs are limited
to 50,000 nodes; parsed replies to 1,000,000 values/containers and depth 64.
Hostname resolution is waited for in a daemon resolver with the same deadline;
an OS DNS operation may finish in that daemon after the caller times out, but it
owns no sockets. Connection address attempts and receive operations consume the
remaining deadline, rather than resetting the timeout for every fragment.

Config output uses a whitelist of sampling, remote-control and relevant
transport settings. Static output uses a whitelist of car/track identity,
dimensions, tyre contact points, environment and relevant assist fields. The
track dictionary is preserved in full without coordinate or speed conversion.
Reports include the endpoint, query durations, UTC start/end timestamps and
errors. Installed AC Gym revision remains `null`: protocol compatibility alone
does not establish the installed Git commit.

Source inspected:
[AC Gym management handler at ef0fa502](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/sensors_par.py#L111-L156)
and [static/track exports](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/structures.py).
The upstream server uses `send`, so a large response may be truncated. Such a
response is reported as an error rather than repaired or accepted silently.

## Verification and scope

Run the isolated tests:

```sh
.venv/bin/python -m pytest -q --noconftest tests/test_ac_management.py
```

The tests use controlled socket streams and cover fragmented UTF-8/literals,
exact requests, completion without EOF, malformed/incomplete/oversized replies,
nonfinite/non-dictionary data, AST bounds, code-expression rejection, deadline
consumption including DNS, fresh resolution, host precedence and imports with
site packages disabled. Live network and real-loopback verification are separate
integration checks to be recorded by the lead/sim-runner.

Implementation validation: **65 passed in 0.41 s** for the combined isolated
AC1 and AC0 suites (43 AC1 tests and 22 AC0 tests). Offline parsing also preserved
all three captured management dictionaries exactly, including the complete
7-key track export. Those snapshot checks are not a new live-network test.

AC1 performs no UDP registration, reset, action selection, game control,
checkpoint deserialization or training. Subsequent stages still need calibrated
steering/telemetry transforms, aligned AC geometry, transition timing and action
attribution, and the selected saved AC setup name. Ibrahim accepted the local
Barcelona `sf70h_tum` best checkpoint as source and chose R1 for the initial AC
transfer, with R4 a later experiment. Those decisions do not rename the source
model or establish equivalence of the two vehicle dynamics.

## Accepted deployment direction (subsequent stages)

Ibrahim subsequently chose production actor inference and the real-time
collector on the Windows computer running AC, whether Lenovo or a lab PC. The
Mac performs actor/critic backpropagation. Experience may be transferred and
learned from after a lap; updated actor weights are applied between laps. Mac
remote inference is a temporary test mode only. Action inference requires the
SAC actor, not the critics.

The accepted initial transfer keeps the source actor, creates fresh AC critics
and a fresh AC replay buffer, and uses R1 first. This deployment decision does
not change AC1's read-only management client. The complete environment,
calibrated observation/action mapping, Windows collector, actor export/load
contract, lap-boundary synchronization and fine-tuning protocol still belong to
the subsequent approved implementation stages.
