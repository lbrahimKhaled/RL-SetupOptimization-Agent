"""Run folders and their JSON files (D2 spec §7). No torch.

Layout: <runs_root>/<run_name>/seed<k>/ with config.json, provenance.json, status.json, ...
`status.json` = {'phase', 'state', 'events': [...], 'phases': {name: {wall_s, env_steps}},
'created_utc', 'updated_utc', ...}; `update_status` merges fields, `add_event` appends.
JSON writes are atomic (temporary file + rename); NaN / inf become null.
"""
from pathlib import Path
import datetime
import json
import math
import os
import numpy as np
from .provenance import REPO_ROOT


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def to_json(obj, **kw):
    return json.dumps(_clean(obj), allow_nan=False, **kw)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(to_json(obj, indent=1))
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(to_json(obj) + '\n')


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def make_run_dir(runs_root, run_name, seed, *, resume=False):
    """<runs_root>/<run_name>/seed<seed>. A new run refuses an existing folder with a
    status.json (use resume); resume requires it."""
    d = resolve(runs_root) / run_name / f'seed{seed}'
    exists = (d / 'status.json').exists()
    if resume and not exists:
        raise FileNotFoundError(f'{d}: nothing to resume (no status.json)')
    if not resume and exists:
        raise FileExistsError(f'{d} already holds a run; pick another --run-name or use --resume')
    for sub in ('', 'bc', 'sb3', 'laps', 'checkpoints'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def update_status(run_dir, **fields):
    path = Path(run_dir) / 'status.json'
    s = read_json(path, {}) or {}
    s.setdefault('created_utc', utc_now())
    s.setdefault('events', [])
    s.update(fields)
    s['updated_utc'] = utc_now()
    write_json(path, s)
    return s


def add_event(run_dir, kind, **data):
    path = Path(run_dir) / 'status.json'
    s = read_json(path, {}) or {}
    s.setdefault('events', []).append(dict(kind=kind, utc=utc_now(), **data))
    write_json(path, s)
    return update_status(run_dir)


def record_phase(run_dir, name, *, wall_s, env_steps=0, **extra):
    """status['phases'][name] = {wall_s, env_steps, ...} (D2 done criterion (vi))."""
    path = Path(run_dir) / 'status.json'
    s = read_json(path, {}) or {}
    phases = s.get('phases', {})
    phases[name] = dict(wall_s=float(wall_s), env_steps=int(env_steps), **extra)
    return update_status(run_dir, phases=phases)
