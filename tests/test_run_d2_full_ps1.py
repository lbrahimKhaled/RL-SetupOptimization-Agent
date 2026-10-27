"""platforms/windows/run_d2_full.ps1: the Windows port of scripts/run_d2_full.sh, same cases as
test_run_d2_full.py (per-seed decisions, --dry-run, continuing after a failing seed, the lock).
Runs only where Windows PowerShell or pwsh exists, with RUNS_ROOT = a temp dir and a temp
OVERRIDES_FILE; the non-dry run uses PYTHON_BIN = a fake python (.cmd) that records its arguments."""
import json
import os
import shutil
import subprocess
import sys
import pytest
from rlsetup.provenance import REPO_ROOT

SCRIPT = REPO_ROOT / 'platforms' / 'windows' / 'run_d2_full.ps1'
PS = shutil.which('powershell') or shutil.which('pwsh')
pytestmark = pytest.mark.skipif(sys.platform != 'win32' or PS is None,
                                reason='Windows PowerShell launcher: Windows only')
SETS = '--set a=1 --set b=x'


def seed_dir(root, s, status=None, last_zip=False):
    d = root / 'r' / f'seed{s}'
    (d / 'checkpoints').mkdir(parents=True)
    if status is not None:
        (d / 'status.json').write_text(json.dumps(status, indent=1))
    if last_zip:
        (d / 'checkpoints' / 'last.zip').write_bytes(b'x')
    return d


def run(root, *args, **env):
    overrides = root.parent / f'{root.name}.overrides'
    overrides.write_text('# comment\na=1\n\n b=x  # trailing\n')
    e = dict(os.environ, RUN_NAME='r', RUNS_ROOT=str(root), OVERRIDES_FILE=str(overrides), MAX_WALL_H='0.5',
             TOTAL_STEPS='1000', **env)
    return subprocess.run([PS, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT), *args],
                          cwd=str(root.parent), env=e, capture_output=True, text=True, timeout=120)


def commands(out):
    return [l.split('] + ', 1)[1] for l in out.splitlines() if '] + ' in l]


def fake_python(tmp_path, fail_seed1=True):
    """.cmd that appends its arguments to calls.txt (and exits 3 for --seed 1 if fail_seed1)."""
    calls = tmp_path / 'calls.txt'
    fake = tmp_path / 'fakepy.cmd'
    fail = 'echo %* | findstr /C:"--seed 1 " >nul && exit /b 3\n' if fail_seed1 else ''
    fake.write_text(f'@echo off\n>>"{calls}" echo %*\n{fail}exit /b 0\n')
    return fake, calls


def test_dry_run_decisions(tmp_path):
    root = tmp_path / 'runs'
    seed_dir(root, 0, {'phase': 'done', 'state': 'wall_clock_limit'}, last_zip=True)
    seed_dir(root, 1, {'phase': 'rl', 'state': 'running'}, last_zip=True)       # hard-killed
    stale = seed_dir(root, 2, {'phase': 'bc', 'state': 'error'})                 # no last.zip
    py = sys.executable
    r = run(root, '--dry-run', PYTHON_BIN=py)
    assert r.returncode == 0, r.stdout + r.stderr
    common = f'--preset full --run-name r --set max_wall_h=0.5 --set total_timesteps=1000 --set runs_root={root}'
    cmds = commands(r.stdout)
    assert cmds[0] == f'{py} scripts/collect_demos.py --preset full {SETS}'
    assert cmds[1] == f'{py} -m rlsetup.driving.train --resume --seed 1 {common}'
    assert cmds[2].startswith(f'Move-Item {stale} {root / "r" / "_aborted" / "seed2."}')
    assert cmds[3] == f'{py} -m rlsetup.driving.train --seed 2 {common} {SETS}'
    assert cmds[4] == f'{py} scripts/d2_report.py {root / "r"}'
    assert len(cmds) == 5 and 'seed 0: skip (finished: wall_clock_limit)' in r.stdout
    assert stale.exists() and not (root / 'r' / '_aborted').exists()             # dry run: nothing moved
    assert not (root / 'r' / '.run_d2_full.lock').exists()


def test_continues_after_failing_seed(tmp_path):
    fake, calls = fake_python(tmp_path)
    root = tmp_path / 'runs'
    seed_dir(root, 0, {'phase': 'bc', 'state': 'bc_gate_failed'})                  # skipped, counted as failure
    stale = seed_dir(root, 2, {'phase': 'demos', 'state': 'running'})
    r = run(root, PYTHON_BIN=str(fake))
    assert r.returncode == 1, r.stdout + r.stderr
    lines = [l.strip() for l in calls.read_text().splitlines()]
    assert lines[0] == f'scripts/collect_demos.py --preset full {SETS}'
    assert lines[1].startswith('-m rlsetup.driving.train --seed 1 ') and lines[1].endswith(SETS)
    assert lines[2].startswith('-m rlsetup.driving.train --seed 2 ')
    assert lines[3] == f'scripts/d2_report.py {root / "r"}' and len(lines) == 4
    assert 'seed 1: end, fresh, exit 3' in r.stdout and 'seed 2: end, fresh, exit 0' in r.stdout
    assert 'seed 0: skip (bc_gate_failed' in r.stdout
    assert not stale.exists() and len(list((root / 'r' / '_aborted').iterdir())) == 1
    assert not (root / 'r' / '.run_d2_full.lock').exists()


def test_refuses_second_instance(tmp_path):
    fake, calls = fake_python(tmp_path, fail_seed1=False)
    root = tmp_path / 'runs'
    (root / 'r').mkdir(parents=True)
    (root / 'r' / '.run_d2_full.lock').write_text(str(os.getpid()))             # a live process holds it
    r = run(root, PYTHON_BIN=str(fake))
    assert r.returncode == 1 and 'is running r' in r.stdout and not calls.exists()
    (root / 'r' / '.run_d2_full.lock').write_text('999999')                      # stale lock: taken over
    assert run(root, PYTHON_BIN=str(fake)).returncode == 0 and calls.exists()
