# AC2: bounded telemetry capture

Architect-approved diagnostic subscription using AC Gym's existing UDP server.
The command sends only `connect` and `disconnect`; it sends no driving actions,
gear commands or resets. A subscription replaces any existing client and resets
the server's connection counter, so stop other AC Gym clients first.

## Run from the Mac or Windows Python environment

From this repository, supply the current Lenovo hostname or LAN address:

```sh
.venv/bin/python -m rlsetup.assetto.telemetry --host HOST --output /private/tmp/ac_capture.jsonl --duration 3
```

`AC_HOST` may supply the host instead; `--host` takes precedence. There is no
default IP address. On Windows use the intended Python executable and a new
Windows output path. The UDP port defaults to upstream's 2345 and is configurable
with `--port`. `--timeout` is the total handshake deadline, `--duration` bounds
capture after identification, and `--max-packets` provides a second bound.
Output uses exclusive creation: choose a new name if the file already exists.

Optional `--source-config FILE` and `--source-checkpoint FILE` record SHA-256
hashes without executing or deserializing those files. No source identity,
installed revision, saved setup or telemetry value is guessed when absent.

## Output and interpretation

The standard-library-only module writes metadata, raw valid telemetry samples
with local receive-monotonic timestamps, and a final summary to JSONL. Metadata
records the queried host, transport bounds, UTC identity and verified protocol
source commit. The installed AC Gym revision remains unknown until checked on
Lenovo.

Summary fields separate malformed/oversized packets, missing and invalid numeric
fields, duplicate sequences, out-of-order timestamps, counter regressions,
possible resets, and forward sequence gaps. A counter regression with a
nondecreasing server timestamp is marked a *possible* reset, not a proven one.
`steps` increments on every `Car.update()`, including recording-related updates:
sequence gaps are not proof of network packet loss.

Receipt intervals and nonnegative adjacent server timestamp intervals report
p50/p95/p99/max milliseconds. Local `receive_monotonic` and Windows
`timestamp_ac` have different origins. These statistics describe interval jitter;
they do **not** measure one-way latency, action latency, or prove driving-loop
adequacy. Steering units/signs and the source driver's observation mapping remain
uncalibrated.

Exit 0 means valid complete-field samples with no reported errors; 1 means a
capture/output/data issue; 2 means invalid CLI configuration. Ctrl+C still closes
the subscription and writes a final summary when the output remains writable.

## Validation

```sh
.venv/bin/python -m pytest -q --noconftest tests/test_ac_telemetry.py tests/test_ac_management.py tests/test_ac_preflight.py
```

Result: **93 passed in 0.27 s** (28 telemetry tests and 65 management/preflight
tests). The standalone module's `--help` command also previously completed
successfully without contacting the game.

Fake-socket tests cover raw sample preservation, connected endpoint selection,
bounded handshake retries and packet count, late identification rejection,
cleanup after errors/interruption,
malformed/nonfinite/oversized data, unknown fields, sequence diagnostics, host
precedence, source hashing and overwrite refusal. Final QA passed all 93 tests;
sim-runner's offline integration check also passed.

The handshake deadline is checked immediately after receive, before accepting
`identified`. A fake-clock regression test delivers identification at 4 s against
a 3 s deadline and verifies the timeout summary, zero captured packets and
disconnect/close cleanup. It failed before the fix and passes with the check.

### Lead's live capture

On 16 Nov 2026 at 00:46:56 UTC, the lead ran a 3 s capture against the then-current
endpoint `172.20.10.2:2345`, writing
`/private/tmp/ac_telemetry_172_20261116_01.jsonl`. The initial live attempt was
not executed because automatic approval review failed; a subsequent compact CLI
invocation passed automatic review and completed. The endpoint records this
measurement only; it is not a default or a persistent host setting.

The capture contains **71 valid samples**, with no errors, malformed/oversized
packets, missing/invalid numeric fields, duplicate sequences, out-of-order
timestamps, counter regressions or forward sequence gaps. Across 70 intervals,
local receipt p50/p95/p99/max were **39.911/45.262/60.420/72.662 ms**; server
timestamp p50/max were **39.989/65.496 ms**. These measure intervals and jitter,
not one-way or action latency, and do not establish driving-loop adequacy.

Metadata records hashes for `runs/mac/D2/bcn/seed0/config.json` and
`runs/mac/D2/bcn/seed0/checkpoints/best.zip` without loading the policy. The
installed AC Gym revision remains unknown. This was a diagnostic subscription;
no actor, driving controls or reset was run.

Protocol evidence: upstream commit
`ef0fa502bdd257fc2affad9361c02b3dbdfe1632`,
[`ego_server.py`](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/ego_server.py)
and
[`structures.py`](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/structures.py).

The later production architecture is actor/collector on the Windows AC PC and
learning/backpropagation on Mac, with experience transferred after a lap. Mac
remote inference is a temporary test only. AC2 is diagnostic capture and does
not implement either runtime architecture or fine-tuning.
