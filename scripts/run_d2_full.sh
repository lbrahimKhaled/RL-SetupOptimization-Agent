#!/usr/bin/env bash
# D2 full run, unattended (docs/handoff/HANDOFF.md "unattended training"): collect the demo
# dataset once, train seeds 0, 1, 2 sequentially, then scripts/d2_report.py.
#
#   caffeinate -dimsu nohup scripts/run_d2_full.sh > runs/D2/full_stdout.log 2>&1 &
#   scripts/run_d2_full.sh --dry-run          # print the commands, run nothing
#
# Environment (defaults):
#   RUN_NAME=full1         run folder runs/D2/$RUN_NAME/seed<k>
#   MAX_WALL_H=3           wall-clock cap per seed and per invocation, h (a resumed seed gets a
#                          fresh cap: the training code counts it from the start of the process)
#   TOTAL_STEPS=2000000    total env steps per seed
#   RUNS_ROOT              run root instead of <repo>/runs/D2 (passed as runs_root); set by
#                          platforms/mac/run_d2_full.sh (runs/mac/D2) and by the tests
#   OVERRIDES_FILE         platform overrides, one KEY=VALUE per line ('#' comments), e.g.
#                          platforms/mac/train.overrides: passed as --set to collect_demos.py and to
#                          fresh starts only (a resume keeps the run's config.json)
#   PYTHON_BIN             only for tests: python for the pipeline commands (default <repo>/.venv)
#
# Per seed: status.json phase 'done' (completed / wall_clock_limit / collapsed) -> skip;
# state bc_gate_failed / demo_gate_failed -> skip, counted as a failure (rerunning is
# deterministic; move the folder away to retry); checkpoints/last.zip exists -> --resume;
# otherwise a fresh start (a stale folder without last.zip is first moved to
# runs/D2/$RUN_NAME/_aborted/, never deleted). A failing seed does not stop the next one; the
# script exits 1 if any step failed. bash 3.2 compatible (macOS /bin/bash); no TTY needed.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$REPO" || exit 1
VENV_PY="$REPO/.venv/bin/python"
PY="${PYTHON_BIN:-$VENV_PY}"
RUN_NAME="${RUN_NAME:-full1}"
MAX_WALL_H="${MAX_WALL_H:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
SEEDS="0 1 2"
export PYTHONUNBUFFERED=1

DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,/^set -u/p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ -n "${RUNS_ROOT:-}" ]; then
  ROOT="$RUNS_ROOT"
  ROOT_SET="--set runs_root=$RUNS_ROOT"
else
  ROOT="$REPO/runs/D2"
  ROOT_SET=""
fi
RUN_DIR="$ROOT/$RUN_NAME"
PLAT_SET=""
if [ -n "${OVERRIDES_FILE:-}" ]; then
  if [ ! -f "$OVERRIDES_FILE" ]; then echo "missing overrides file: $OVERRIDES_FILE" >&2; exit 2; fi
  while IFS= read -r line || [ -n "$line" ]; do
    line="$(printf '%s' "${line%%#*}" | tr -d '[:space:]')"
    if [ -n "$line" ]; then PLAT_SET="$PLAT_SET --set $line"; fi
  done < "$OVERRIDES_FILE"
fi
COMMON="--preset full --run-name $RUN_NAME --set max_wall_h=$MAX_WALL_H --set total_timesteps=$TOTAL_STEPS"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }

# run CMD...: print it; execute unless --dry-run; returns its exit code
run() {
  log "+ $*"
  if [ "$DRY" -eq 1 ]; then return 0; fi
  "$@" < /dev/null
}

# status_field FILE KEY: top-level string field of a status.json ('' if missing / unreadable)
status_field() {
  "$VENV_PY" -c 'import json, sys
try:
    v = json.load(open(sys.argv[1])).get(sys.argv[2])
except Exception:
    v = None
print(v if isinstance(v, str) else "")' "$1" "$2"
}

if [ ! -x "$VENV_PY" ]; then echo "missing venv python: $VENV_PY" >&2; exit 1; fi

# one invocation per run name at a time (stale lock of a dead process is taken over)
LOCK="$RUN_DIR/.run_d2_full.lock"
if [ "$DRY" -eq 0 ]; then
  mkdir -p "$RUN_DIR"
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    log "another run_d2_full.sh (pid $(cat "$LOCK")) is running $RUN_NAME; exiting"
    exit 1
  fi
  echo $$ > "$LOCK"
  trap 'rm -f "$LOCK"' EXIT
fi

FAILED=0
SUMMARY=""
log "run_d2_full: repo $REPO, run $RUN_DIR, seeds $SEEDS, max_wall_h $MAX_WALL_H, total_steps $TOTAL_STEPS, dry_run $DRY"
log "platform overrides: ${OVERRIDES_FILE:-none}${PLAT_SET}"
log "git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) $(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ') tracked changes"

# --- demos once (cached per dataset key; the seeds reuse the file) ---
T0=$SECONDS
# shellcheck disable=SC2086
run "$PY" scripts/collect_demos.py --preset full $PLAT_SET
DEMO_RC=$?
log "demos: exit $DEMO_RC ($((SECONDS - T0)) s)"
if [ "$DEMO_RC" -ne 0 ]; then FAILED=1; fi
SUMMARY="$SUMMARY
  demos: exit $DEMO_RC"

# --- seeds ---
for s in $SEEDS; do
  SD="$RUN_DIR/seed$s"
  T0=$SECONDS
  log "seed $s: start ($SD)"
  phase=""; state=""
  if [ -f "$SD/status.json" ]; then
    phase="$(status_field "$SD/status.json" phase)"
    state="$(status_field "$SD/status.json" state)"
  fi
  rc=0
  if [ "$phase" = "done" ]; then
    action="skip (finished: $state)"
    log "seed $s: $action"
  elif [ "$state" = "bc_gate_failed" ] || [ "$state" = "demo_gate_failed" ]; then
    action="skip ($state; move $SD away to retry)"
    rc=1
    log "seed $s: $action"
  elif [ -f "$SD/checkpoints/last.zip" ]; then
    action="resume (state ${state:-?})"
    log "seed $s: $action"
    # shellcheck disable=SC2086
    run "$PY" -m rlsetup.driving.train --resume --seed "$s" $COMMON $ROOT_SET
    rc=$?
  elif [ "$DEMO_RC" -ne 0 ]; then
    action="skip fresh start (demo collection failed)"
    rc=1
    log "seed $s: $action"
  else
    if [ -e "$SD" ]; then
      dest="$RUN_DIR/_aborted/seed$s.$(date '+%Y%m%dT%H%M%S')"
      log "seed $s: no last.zip (state ${state:-none}); moving the stale folder to $dest"
      if [ "$DRY" -eq 0 ]; then mkdir -p "$RUN_DIR/_aborted"; fi
      run mv "$SD" "$dest"
    fi
    action="fresh"
    # shellcheck disable=SC2086
    run "$PY" -m rlsetup.driving.train --seed "$s" $COMMON $ROOT_SET $PLAT_SET
    rc=$?
  fi
  if [ "$rc" -ne 0 ]; then FAILED=1; fi
  log "seed $s: end, $action, exit $rc, $((SECONDS - T0)) s"
  SUMMARY="$SUMMARY
  seed $s: $action, exit $rc, $((SECONDS - T0)) s"
done

# --- report ---
T0=$SECONDS
run "$PY" scripts/d2_report.py "$RUN_DIR"
REPORT_RC=$?
log "report: exit $REPORT_RC ($((SECONDS - T0)) s)"
if [ "$REPORT_RC" -ne 0 ]; then FAILED=1; fi
SUMMARY="$SUMMARY
  report: exit $REPORT_RC"

log "run_d2_full: summary$SUMMARY"
log "run_d2_full: done, exit $FAILED"
exit "$FAILED"
