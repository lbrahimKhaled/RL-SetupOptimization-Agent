"""Where the sibling ocd_first_lap checkout is, on any machine.

Order: $OCD_FIRST_LAP; the checkout `ocd_lap` is importable from (PYTHONPATH, which the
platform launchers set); `ocd_first_lap` next to this repo. Nothing here imports ocd_lap.
"""
from pathlib import Path
import importlib.util
import os

REPO_ROOT = Path(__file__).resolve().parents[1]


def ocd_first_lap_root():
    if os.environ.get('OCD_FIRST_LAP'):
        return Path(os.environ['OCD_FIRST_LAP']).resolve()
    try:
        spec = importlib.util.find_spec('ocd_lap')
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parents[1]
    return (REPO_ROOT.parent / 'ocd_first_lap').resolve()


def sibling_checkout(name):
    """Another worktree next to ocd_first_lap, e.g. 'ocd_first_lap-tumdebug'."""
    return ocd_first_lap_root().parent / name
