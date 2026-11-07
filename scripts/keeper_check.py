"""Stop / resume checks of runs/mac/keeper*.sh (R2-9.4, Amendments 1-2). Exit code 0 = yes, 1 = no.

    .venv/bin/python scripts/keeper_check.py reached runs/mac/D2/f1_r2/seed0 [LIMIT_S]
    .venv/bin/python scripts/keeper_check.py stopped runs/mac/D2/f1_r2/seed0
    .venv/bin/python scripts/keeper_check.py pattern f1_r2      # prints the pgrep / pkill -f regex

reached: an evaluation with a clean deterministic lap 2 <= the limit that is reportable (record
         'reportable'; records without the key, i.e. pre-R2 runs, count as reportable). Earlier
         aided laps and the Amendment-2 result stretch after a trigger never stop the keeper: the
         run handles them itself (it stops at report_from). Limit: LIMIT_S if given, else the
         run's config.json bonus_schedule.trigger_lap2_s (the same number the trigger uses), else
         LIMIT_S_DEFAULT (95 s, Decision 5; tested equal to BonusScheduleConfig.trigger_lap2_s).
stopped: the run ended itself after an Amendment-1 trigger (status.json stop_reason
         'report_complete', or bonus_state.json has a trigger_step and an evaluation at or past its
         report_from exists): the keeper must not resume it.
pattern: extended regex (pgrep / pkill -f, POSIX ERE) matching the training process of run RUN
         (`python -m rlsetup.driving.train ... --run-name RUN`) but not of a run whose name only
         starts with RUN (f1_r2 vs f1_r2_smoke): RUN must be followed by a space or the end.
numpy-free, stdlib only.
"""
import json
import sys
from pathlib import Path

LIMIT_S_DEFAULT = 95.
_ERE_SPECIAL = set('.[]()*+?{}|^$\\')


def _evals(seed_dir):
    p = Path(seed_dir) / 'evals.jsonl'
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:      # line being written right now
                continue
    return out


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def limit_s(seed_dir):
    """Keeper limit [s] of the run in `seed_dir`: config.json bonus_schedule.trigger_lap2_s (R2
    runs), else LIMIT_S_DEFAULT (pre-R2 configs omit the block)."""
    value = (_json(Path(seed_dir) / 'config.json').get('bonus_schedule') or {}).get('trigger_lap2_s')
    return float(value) if value is not None else LIMIT_S_DEFAULT


def reached(seed_dir, limit=None):
    limit = limit_s(seed_dir) if limit is None else float(limit)
    return any(e.get('det_clean') and e.get('lap2_s') is not None and e['lap2_s'] <= limit
               and e.get('reportable', True) for e in _evals(seed_dir))


def stopped(seed_dir):
    if _json(Path(seed_dir) / 'status.json').get('stop_reason') == 'report_complete':
        return True
    st = _json(Path(seed_dir) / 'bonus_state.json')
    if st.get('trigger_step') is None or st.get('report_from') is None:
        return False
    return any(e.get('step', -1) >= st['report_from'] for e in _evals(seed_dir))


def pattern(run_name):
    """POSIX ERE for `pgrep -f` / `pkill -f`: the training process of run `run_name` only."""
    name = ''.join('\\' + ch if ch in _ERE_SPECIAL else ch for ch in run_name)
    return f'rlsetup\\.driving\\.train .*--run-name {name}( |$)'


if __name__ == '__main__':
    what, arg = sys.argv[1], sys.argv[2]
    if what == 'pattern':
        print(pattern(arg))
        sys.exit(0)
    if what == 'reached':
        ok = reached(arg, float(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif what == 'stopped':
        ok = stopped(arg)
    else:
        sys.exit(f'unknown check {what!r}')
    sys.exit(0 if ok else 1)
