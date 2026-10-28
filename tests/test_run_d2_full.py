"""scripts/run_d2_full.sh: per-seed decisions (skip / resume / fresh), --dry-run, continuing after
a failing seed. Runs under /bin/bash (3.2 on macOS) with RUNS_ROOT = a temp dir; the non-dry
run uses PYTHON_BIN = a fake python that records its arguments."""
import json
import os
import subprocess
import sys
import pytest
from rlsetup.provenance import REPO_ROOT

# The bash launcher is the Mac one (platforms/mac/README.md); Windows uses run_d2_full.ps1,
# tested in test_run_d2_full_ps1.py.
pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='bash launcher: macOS / POSIX only')

SCRIPT = REPO_ROOT / 'scripts' / 'run_d2_full.sh'
PY = str(REPO_ROOT / '.venv' / 'bin' / 'python')


def seed_dir(root, s, status=None, last_zip=False):
    d = root / 'r' / f'seed{s}'
    (d / 'checkpoints').mkdir(parents=True)
    if status is not None:
        (d / 'status.json').write_text(json.dumps(status, indent=1))
    if last_zip:
        (d / 'checkpoints' / 'last.zip').write_bytes(b'x')
    return d


def run(root, *args, **env):
    e = dict(os.environ, RUN_NAME='r', RUNS_ROOT=str(root), MAX_WALL_H='0.5', TOTAL_STEPS='1000', **env)
    return subprocess.run(['/bin/bash', str(SCRIPT), *args], cwd='/', env=e, capture_output=True, text=True,
                          timeout=60)


def commands(out):
    return [l.split('] + ', 1)[1] for l in out.splitlines() if '] + ' in l]


def test_dry_run_decisions(tmp_path):
    seed_dir(tmp_path, 0, {'phase': 'done', 'state': 'wall_clock_limit'}, last_zip=True)
    seed_dir(tmp_path, 1, {'phase': 'rl', 'state': 'running'}, last_zip=True)       # hard-killed
    stale = seed_dir(tmp_path, 2, {'phase': 'bc', 'state': 'error'})                 # no last.zip
    r = run(tmp_path, '--dry-run')
    assert r.returncode == 0, r.stdout + r.stderr
    common = f'--preset full --run-name r --set max_wall_h=0.5 --set total_timesteps=1000 --set runs_root={tmp_path}'
    cmds = commands(r.stdout)
    assert cmds[0] == f'{PY} scripts/collect_demos.py --preset full'
    assert cmds[1] == f'{PY} -m rlsetup.driving.train --resume --seed 1 {common}'
    assert cmds[2].startswith(f'mv {stale} {tmp_path}/r/_aborted/seed2.')
    assert cmds[3] == f'{PY} -m rlsetup.driving.train --seed 2 {common}'
    assert cmds[4] == f'{PY} scripts/d2_report.py {tmp_path}/r'
    assert len(cmds) == 5 and 'seed 0: skip (finished: wall_clock_limit)' in r.stdout
    assert stale.exists() and not (tmp_path / 'r' / '_aborted').exists()               # dry run: nothing moved
    assert not (tmp_path / 'r' / '.run_d2_full.lock').exists()


def test_continues_after_failing_seed(tmp_path):
    calls = tmp_path / 'calls.txt'
    fake = tmp_path / 'fakepy'
    fake.write_text(f'#!/bin/bash\necho "$*" >> {calls}\ncase " $* " in *" --seed 1 "*) exit 3 ;; esac\nexit 0\n')
    fake.chmod(0o755)
    root = tmp_path / 'runs'
    seed_dir(root, 0, {'phase': 'bc', 'state': 'bc_gate_failed'})                  # skipped, counted as failure
    stale = seed_dir(root, 2, {'phase': 'demos', 'state': 'running'})
    r = run(root, PYTHON_BIN=str(fake))
    assert r.returncode == 1, r.stdout + r.stderr
    lines = calls.read_text().splitlines()
    assert lines[0] == 'scripts/collect_demos.py --preset full'
    assert lines[1].startswith('-m rlsetup.driving.train --seed 1 ')
    assert lines[2].startswith('-m rlsetup.driving.train --seed 2 ')
    assert lines[3] == f'scripts/d2_report.py {root}/r' and len(lines) == 4
    assert 'seed 1: end, fresh, exit 3' in r.stdout and 'seed 2: end, fresh, exit 0' in r.stdout
    assert 'seed 0: skip (bc_gate_failed' in r.stdout
    assert not stale.exists() and len(list((root / 'r' / '_aborted').iterdir())) == 1
    assert not (root / 'r' / '.run_d2_full.lock').exists()


def test_refuses_second_instance(tmp_path):
    calls = tmp_path / 'calls.txt'
    fake = tmp_path / 'fakepy'
    fake.write_text(f'#!/bin/bash\necho "$*" >> {calls}\n')
    fake.chmod(0o755)
    (tmp_path / 'r').mkdir()
    (tmp_path / 'r' / '.run_d2_full.lock').write_text(str(os.getpid()))         # a live process holds it
    r = run(tmp_path, PYTHON_BIN=str(fake))
    assert r.returncode == 1 and 'is running r' in r.stdout and not calls.exists()
    (tmp_path / 'r' / '.run_d2_full.lock').write_text('999999')                  # stale lock: taken over
    assert run(tmp_path, PYTHON_BIN=str(fake)).returncode == 0 and calls.exists()


def test_overrides_file_reaches_demos_and_fresh_starts_only(tmp_path):
    """OVERRIDES_FILE (platforms/<os>/train.overrides): --set for collect_demos.py and fresh
    starts; a resume keeps the run's config.json (train.py refuses other overrides on resume)."""
    seed_dir(tmp_path, 0, {'phase': 'rl', 'state': 'running'}, last_zip=True)
    overrides = tmp_path / 'train.overrides'
    overrides.write_text('# comment\na=1\n\n b=x  # trailing\r\n')
    r = run(tmp_path, '--dry-run', OVERRIDES_FILE=str(overrides))
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = commands(r.stdout)
    assert cmds[0] == f'{PY} scripts/collect_demos.py --preset full --set a=1 --set b=x'
    assert cmds[1].startswith(f'{PY} -m rlsetup.driving.train --resume --seed 0 ') and '--set a=1' not in cmds[1]
    assert cmds[2].endswith(f'--set runs_root={tmp_path} --set a=1 --set b=x')
    assert run(tmp_path, '--dry-run', OVERRIDES_FILE=str(tmp_path / 'missing')).returncode == 2


def test_mac_launcher_dry_run(tmp_path):
    """platforms/mac/run_d2_full.sh = scripts/run_d2_full.sh + RUNS_ROOT + the Mac overrides."""
    e = dict(os.environ, RUN_NAME='r', RUNS_ROOT=str(tmp_path))
    r = subprocess.run(['/bin/bash', str(REPO_ROOT / 'platforms' / 'mac' / 'run_d2_full.sh'), '--dry-run'],
                       cwd='/', env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    mac = ('--set device=cpu --set vec_start_method=forkserver --set n_envs=4 --set gradient_steps=2 '
           '--set torch_threads=1')
    cmds = commands(r.stdout)
    assert cmds[0] == f'{PY} scripts/collect_demos.py --preset full {mac}'
    assert cmds[1].endswith(f'--set runs_root={tmp_path} {mac}')
