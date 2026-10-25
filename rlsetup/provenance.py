"""Run provenance: which code of this repo and of ocd_first_lap produced a result.

`run_provenance()` is what every training/evaluation run stores next to its outputs.
"""
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_state(path):
    """{'commit', 'dirty' (tracked files modified), 'untracked' (untracked, non-ignored files)}.
    Values are None when `path` is not a git work tree."""
    def git(*args):
        r = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True, check=False)
        return r.stdout if r.returncode == 0 else None
    commit = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain')
    if commit is None or status is None:
        return {'path': str(path), 'commit': None, 'dirty': None, 'untracked': None}
    lines = [l for l in status.splitlines() if l]
    return {'path': str(path), 'commit': commit.strip(),
            'dirty': any(not l.startswith('??') for l in lines),
            'untracked': any(l.startswith('??') for l in lines)}


def ocd_first_lap_root():
    import ocd_lap
    return Path(ocd_lap.__file__).resolve().parents[1]


def run_provenance(track=None):
    """This repo's and ocd_first_lap's git state plus ocd_lap's content digest `code_version`
    (sources, OCD sources, the loaded ocd_native binary, packages; see ocd_lap/provenance.py).
    Note: ocd_lap's digest does not cover every ocd_lap module (e.g. qss_lap.py, handling.py),
    so the ocd_first_lap git state is recorded as well."""
    from ocd_lap.provenance import capture_provenance, provenance_digest
    ocd = capture_provenance(track)
    return {'rlsetup': git_state(REPO_ROOT),
            'ocd_first_lap': git_state(ocd_first_lap_root()),
            'ocd_lap_code_version': provenance_digest(ocd)[:12]}
