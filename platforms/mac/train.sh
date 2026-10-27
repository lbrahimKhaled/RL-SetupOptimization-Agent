#!/usr/bin/env bash
# python -m rlsetup.driving.train with the Mac defaults (platforms/README.md).
#
#   platforms/mac/train.sh --preset full --seed 0 --run-name NAME [--set k=v ...]
#   platforms/mac/train.sh --resume --preset full --seed 0 --run-name NAME
#
# Fresh run: --set runs_root=$RUNS_ROOT, then every line of train.overrides, then the caller's
# arguments (a later --set wins). Resume: only runs_root (config.json keeps the rest; train.py
# refuses other overrides on resume). RUNS_ROOT defaults to <repo>/runs/mac/D2.
set -u
HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../.." && pwd -P)"
cd "$REPO" || exit 1
PY="${PYTHON_BIN:-$REPO/.venv/bin/python}"
ROOT="${RUNS_ROOT:-$REPO/runs/mac/D2}"
SETS="--set runs_root=$ROOT"
case " $* " in
  *" --resume "*) ;;
  *) while IFS= read -r line || [ -n "$line" ]; do
       line="$(printf '%s' "${line%%#*}" | tr -d '[:space:]')"
       if [ -n "$line" ]; then SETS="$SETS --set $line"; fi
     done < "$HERE/train.overrides" ;;
esac
export PYTHONUNBUFFERED=1
# shellcheck disable=SC2086
exec "$PY" -m rlsetup.driving.train $SETS "$@"
