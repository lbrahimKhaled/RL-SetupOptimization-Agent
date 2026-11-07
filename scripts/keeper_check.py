"""Stop / resume checks of runs/mac/keeper*.sh (R2-9.4, Amendment 1). Exit code 0 = yes, 1 = no.

    .venv/bin/python scripts/keeper_check.py reached runs/mac/D2/f1_r2/seed0 [LIMIT_S]
    .venv/bin/python scripts/keeper_check.py stopped runs/mac/D2/f1_r2/seed0

reached: an evaluation with a clean deterministic lap 2 <= LIMIT_S (default 95 s, Decision 5) that
         is reportable (record 'reportable'; records without the key, i.e. pre-R2 runs, count as
         reportable). Earlier aided laps never stop the keeper: the run handles them itself.
stopped: the run ended itself after an Amendment-1 trigger (status.json stop_reason
         'report_complete', or bonus_state.json has a trigger_step and an evaluation at or past its
         report_from exists): the keeper must not resume it.
numpy-free, stdlib only.
"""
import json
import sys
from pathlib import Path

LIMIT_S = 95.


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


def reached(seed_dir, limit_s=LIMIT_S):
    return any(e.get('det_clean') and e.get('lap2_s') is not None and e['lap2_s'] <= limit_s
               and e.get('reportable', True) for e in _evals(seed_dir))


def stopped(seed_dir):
    if _json(Path(seed_dir) / 'status.json').get('stop_reason') == 'report_complete':
        return True
    st = _json(Path(seed_dir) / 'bonus_state.json')
    if st.get('trigger_step') is None or st.get('report_from') is None:
        return False
    return any(e.get('step', -1) >= st['report_from'] for e in _evals(seed_dir))


if __name__ == '__main__':
    what, seed = sys.argv[1], sys.argv[2]
    if what == 'reached':
        ok = reached(seed, float(sys.argv[3]) if len(sys.argv) > 3 else LIMIT_S)
    elif what == 'stopped':
        ok = stopped(seed)
    else:
        sys.exit(f'unknown check {what!r}')
    sys.exit(0 if ok else 1)
