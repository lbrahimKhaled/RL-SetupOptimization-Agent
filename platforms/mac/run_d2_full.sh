#!/usr/bin/env bash
# Mac launcher of the unattended D2 run: scripts/run_d2_full.sh with the run root runs/mac/D2 and
# the Mac overrides (train.overrides) for demos and fresh starts. Same environment variables and
# --dry-run as scripts/run_d2_full.sh (RUNS_ROOT / OVERRIDES_FILE override the two defaults).
#
#   caffeinate -dimsu nohup platforms/mac/run_d2_full.sh > runs/mac/full_stdout.log 2>&1 &
HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../.." && pwd -P)"
export RUNS_ROOT="${RUNS_ROOT:-$REPO/runs/mac/D2}"
export OVERRIDES_FILE="${OVERRIDES_FILE:-$HERE/train.overrides}"
exec /bin/bash "$REPO/scripts/run_d2_full.sh" "$@"
