"""Run provenance: which code of this repo and of ocd_first_lap produced a result.

`run_provenance()` is what every training/evaluation run stores next to its outputs, together
with `DrivingEnv.env_metadata()` (config, reference-line digest, QSS cache key).

Code identity is recorded three ways, because a commit hash alone does not identify dirty or
untracked code (D1_sim_runner §1, QA MINOR 9):
  - git state of both repos, incl. sha256 of `git diff HEAD` (tracked changes; untracked files
    are not in it);
  - sha256 of every rlsetup/**/*.py file;
  - ocd_lap's own `code_version` digest plus sha256 of the ocd_lap modules rlsetup imports that
    `ocd_lap.provenance.capture_provenance` does not hash (OCD_LAP_EXTRA_SOURCES).
"""
from pathlib import Path
import hashlib
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
# ocd_lap modules loaded by rlsetup but not in ocd_lap's code_version (simulation.SOURCES +
# evaluate.py + provenance.py). env.py: DrivingEnv's base class; qss_lap.py, handling.py,
# manoeuvres.py: QSS profile; rig.py, vertical_model.py: imported by handling.py; __init__.py.
# tests/test_provenance.py fails if rlsetup starts importing a module covered by neither list.
OCD_LAP_EXTRA_SOURCES = ('__init__.py', 'env.py', 'qss_lap.py', 'handling.py', 'manoeuvres.py',
                         'rig.py', 'vertical_model.py')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state(path):
    """{'path', 'commit', 'dirty' (tracked files modified), 'untracked' (untracked, non-ignored
    files), 'diff_sha256' (sha256 of `git diff HEAD`; e3b0c442... = empty diff)}.
    Values are None when `path` is not a git work tree."""
    def git(*args, text=True):
        r = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=text, check=False)
        return r.stdout if r.returncode == 0 else None
    commit = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain')
    diff = git('diff', 'HEAD', '--no-color', '--no-ext-diff', text=False)
    if commit is None or status is None or diff is None:
        return {'path': str(path), 'commit': None, 'dirty': None, 'untracked': None, 'diff_sha256': None}
    lines = [l for l in status.splitlines() if l]
    return {'path': str(path), 'commit': commit.strip(),
            'dirty': any(not l.startswith('??') for l in lines),
            'untracked': any(l.startswith('??') for l in lines),
            'diff_sha256': hashlib.sha256(diff).hexdigest()}


def ocd_first_lap_root():
    import ocd_lap
    return Path(ocd_lap.__file__).resolve().parents[1]


def rlsetup_sources_sha256():
    """{path relative to the repo root: sha256} of every .py file in the rlsetup package."""
    pkg = REPO_ROOT / 'rlsetup'
    return {str(p.relative_to(REPO_ROOT)): sha256_file(p)
            for p in sorted(pkg.rglob('*.py')) if '__pycache__' not in p.parts}


def ocd_lap_extra_sources_sha256():
    """{'ocd_lap/<module>.py': sha256} for OCD_LAP_EXTRA_SOURCES (None if a file is missing)."""
    lap_dir = ocd_first_lap_root() / 'ocd_lap'
    return {f'ocd_lap/{f}': sha256_file(lap_dir / f) if (lap_dir / f).is_file() else None
            for f in OCD_LAP_EXTRA_SOURCES}


def ocd_lap_code_version(track=None):
    """ocd_lap's content digest (12 hex) of `capture_provenance(track)`: its sources, OCD sources,
    the loaded ocd_native binary, packages and, for a track NAME, the track/raceline CSVs."""
    from ocd_lap.provenance import capture_provenance, provenance_digest
    return provenance_digest(capture_provenance(track))[:12]


def run_provenance(track=None):
    """Git state of this repo and ocd_first_lap, sha256 of rlsetup sources, ocd_lap's content
    digest `code_version` (its sources, OCD sources, the loaded ocd_native binary, packages; see
    ocd_lap/provenance.py) and sha256 of the ocd_lap modules that digest omits."""
    return {'rlsetup': git_state(REPO_ROOT),
            'rlsetup_sources_sha256': rlsetup_sources_sha256(),
            'ocd_first_lap': git_state(ocd_first_lap_root()),
            'ocd_lap_code_version': ocd_lap_code_version(track),
            'ocd_lap_extra_sources_sha256': ocd_lap_extra_sources_sha256()}


def track_name(track):
    """Track name for ocd_lap's capture_provenance, which builds '<name>.csv' paths: a Track
    object gives its .name (passing the object silently drops the track/raceline CSV hashes,
    D2_smoke.md provenance finding), a str or None passes through."""
    return getattr(track, 'name', track)


def code_state(track=None):
    """The code identity a resumed run is checked against (QA D2 M2): rlsetup git state and the
    ocd_lap code_version, computed for the same track as `training_provenance`."""
    return {'rlsetup': git_state(REPO_ROOT), 'ocd_lap_code_version': ocd_lap_code_version(track_name(track))}


def code_mismatches(recorded, current):
    """Differences between a run's provenance.json (`recorded`, {} if missing) and `code_state()`
    (`current`): rlsetup commit and diff_sha256 (uncommitted tracked changes), ocd_lap
    code_version. Returns a list of readable strings (empty = same code)."""
    rec, cur = recorded.get('rlsetup') or {}, current.get('rlsetup') or {}
    out = [f'rlsetup {k}: run {rec.get(k)} != now {cur.get(k)}'
           for k in ('commit', 'diff_sha256') if rec.get(k) != cur.get(k)]
    if recorded.get('ocd_lap_code_version') != current.get('ocd_lap_code_version'):
        out.append(f"ocd_lap_code_version: run {recorded.get('ocd_lap_code_version')} != "
                   f"now {current.get('ocd_lap_code_version')}")
    return out


def package_versions():
    """Versions of the packages a training run depends on (imports torch / SB3)."""
    import importlib.metadata as md
    out = {}
    for name in ('torch', 'stable-baselines3', 'sb3-contrib', 'gymnasium', 'numpy'):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def training_provenance(track, *, n_envs, qss, reference_digest, track_sha256, config_json,
                        argv=None, demo_sha256=None, bc_actor_sha256=None, allow_dirty=False):
    """provenance.json of a D2 training run (spec §9): `run_provenance(track.name)` plus package
    versions, machine, torch threads, n_envs, argv, UTC start, QSS cache-key sha + lap time,
    reference-line digest, track sha, demo dataset / BC actor sha (filled in as they exist) and
    the sha256 of the config JSON text."""
    import datetime
    import json
    import platform
    import socket
    import sys
    import torch
    return {**run_provenance(track_name(track)),
            'python': sys.version.split()[0], 'packages': package_versions(),
            'platform': platform.platform(), 'machine': platform.machine(), 'hostname': socket.gethostname(),
            'torch_num_threads': torch.get_num_threads(), 'n_envs': int(n_envs),
            'argv': list(sys.argv if argv is None else argv),
            'utc_start': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'qss': {'cache_key_sha256': hashlib.sha256(json.dumps(qss.meta['key'], sort_keys=True).encode()).hexdigest(),
                    'lap_time_s': float(qss.lap_time_s)},
            'reference_digest': reference_digest, 'track_path_sha256': track_sha256,
            'demo_dataset_sha256': demo_sha256, 'bc_actor_sha256': bc_actor_sha256,
            'config_sha256': hashlib.sha256(config_json.encode()).hexdigest(),
            'allow_dirty': bool(allow_dirty)}
