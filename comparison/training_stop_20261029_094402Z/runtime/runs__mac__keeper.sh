#!/bin/bash
# Usage: runs/mac/keeper.sh RUN_NAME   (Ibrahim: run until stopped by hand or a clean lap < 95 s)
# - mirror evals into TensorBoard every 3 min;
# - stop training at the first clean deterministic lap 2 < 95 s;
# - if training dies otherwise (crash), resume it from last.zip (at most 5 times).
# Stop everything by hand: pkill -f "keeper.sh RUN_NAME"; pkill -INT -f "run-name $RUN"
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent || exit 1
RUN="$1"; shift; EXTRA="$*"; R=runs/mac/D2/$RUN/seed0
PAT="rlsetup.driving.train .*--run-name $RUN"
reached() { .venv/bin/python -c "
import json,sys
for l in open('$R/evals.jsonl'):
    try: e=json.loads(l)
    except Exception: continue
    if e.get('det_clean') and e.get('lap2_s') and e['lap2_s']<95: sys.exit(0)
sys.exit(1)"; }
resumes=0
while true; do
  if pgrep -f "$PAT" >/dev/null; then
    .venv/bin/python scripts/evals_to_tb.py $R >/dev/null 2>&1
    if reached; then
      echo "$(date) clean lap < 95 s reached: stopping"; pkill -INT -f "$PAT"; sleep 60; pkill -TERM -f "$PAT"
      .venv/bin/python scripts/evals_to_tb.py $R >/dev/null 2>&1; exit 0
    fi
  else
    reached && { echo "$(date) done (< 95 s)"; exit 0; }
    [ $resumes -ge 5 ] && { echo "$(date) training stopped; 5 resumes used, giving up"; exit 1; }
    resumes=$((resumes+1)); echo "$(date) training not running: resume #$resumes"
    caffeinate -dimsu nohup platforms/mac/train.sh --resume --preset $RUN --seed 0 --run-name $RUN --allow-code-change >> runs/mac/${RUN}_stdout.log 2>&1 &
    sleep 120
  fi
  sleep 180
done
