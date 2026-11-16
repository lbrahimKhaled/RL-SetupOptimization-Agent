"""C2 provenance beyond `rlsetup.provenance.run_provenance` (QA M11, sim-runner item 6).

- `library_versions()`: python, numpy, scipy (the style numbers depend on scipy's gaussian_filter1d
  and cKDTree, the gate on numpy's FFT).
- `car_constants_provenance(car)`: the CarConfig values (m_c, t), their sha256, and the git state of the
  ocd_first_lap worktree on branch `CAR_SOURCE_BRANCH` they were taken from (found with `git worktree
  list` of the ocd_first_lap on the path; None when absent). The worktree path is not stored (it holds
  the local account name); branch, commit, dirty flag and diff sha256 are.
"""
from dataclasses import asdict
import hashlib
import json
import platform
import subprocess
import numpy as np
from ..provenance import git_state, ocd_first_lap_root
from .config import CAR_SOURCE_BRANCH, CarConfig


def library_versions() -> dict:
    import scipy
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}


def _worktree(branch: str):
    r = subprocess.run(['git', '-C', str(ocd_first_lap_root()), 'worktree', 'list', '--porcelain'],
                       capture_output=True, text=True, check=False)
    if r.returncode:
        return None
    path = None
    for line in r.stdout.splitlines():
        if line.startswith('worktree '):
            path = line[len('worktree '):]
        elif line == f'branch refs/heads/{branch}':
            return path
    return None


def car_constants_provenance(car: CarConfig = CarConfig()) -> dict:
    values = asdict(car)
    path = _worktree(CAR_SOURCE_BRANCH)
    state = None
    if path is not None:
        g = git_state(path)
        state = {k: g[k] for k in ('commit', 'dirty', 'untracked', 'diff_sha256')}
    return {'values': values, 'values_sha256': hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
            'source': f"CarGeometry.from_params(OCDBackend(car='sf70h_f1').base_params), ocd_first_lap branch "
                      f"{CAR_SOURCE_BRANCH}; checked by tests/test_c2_knobs.py::test_car_constants_match_the_f1_worktree",
            'branch': CAR_SOURCE_BRANCH, 'worktree_git': state}
