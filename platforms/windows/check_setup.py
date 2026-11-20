"""Import and file checks for a PC set up with setup_pc.ps1 (AC fine-tuning and OCD training on one PC).

    python platforms/windows/check_setup.py            # all checks
    python platforms/windows/check_setup.py --no-ocd   # skip ocd_native / ocd_lap (no .pyd yet)

Run with the env's python and env.ps1's PYTHONPATH (setup_pc.ps1 does both). One line per check
(OK / FAIL / MISSING); exit 1 if a required check fails. Nothing is changed.
- AC side (learner + worker, rlsetup.assetto): torch, SB3 2.9.0; does not import ocd_lap.
- OCD side (rlsetup.driving.train): ocd_native (the Windows .pyd in ocd_first_lap) and ocd_lap.
"""
from pathlib import Path
import argparse
import importlib
import os
import sys

REPO = Path(__file__).resolve().parents[2]
OCD_FIRST_LAP = Path(os.environ.get('OCD_FIRST_LAP', REPO.parent / 'ocd_first_lap'))
failed = []


def check(name, ok, detail='', required=True):
    tag = 'OK' if ok else ('FAIL' if required else 'MISSING')
    print(f'[{tag:7}] {name}' + (f': {detail}' if detail else ''), flush=True)
    if required and not ok:
        failed.append(name)
    return ok


def try_import(mod, required=True):
    try:
        m = importlib.import_module(mod)
    except Exception as e:   # noqa: BLE001 - report every import problem
        check(f'import {mod}', False, f'{type(e).__name__}: {e}', required)
        return None
    check(f'import {mod}', True, getattr(m, '__version__', ''))
    return m


def versions():
    check('python 3.12', sys.version_info[:2] == (3, 12), sys.version.split()[0])
    torch = try_import('torch')
    if torch is not None:
        cuda = torch.cuda.is_available()
        check('torch CUDA', cuda, torch.cuda.get_device_name(0) if cuda else
              'not available: train with --set device=cpu (CPU torch, or no NVIDIA driver)', required=False)
    for mod, want in (('stable_baselines3', '2.9.0'), ('sb3_contrib', '2.9.0'), ('gymnasium', '1.4.0')):
        m = try_import(mod)
        if m is not None:
            check(f'{mod} == {want}', m.__version__ == want, m.__version__)
    try_import('tensorboard', required=False)


def ac_side():
    try_import('rlsetup.assetto.distributed')
    try_import('rlsetup.assetto.learn')
    for rel in ('runs/ac/source_actor_bcn_f1_r5_7m/manifest.json', 'runs/ac/ocd_line_reference_bcn_f1_r5_7m.json',
                'models/bcn_f1_r5_step7000k/seed0/checkpoints/last.zip',
                'models/bcn_f1_r5_step7000k/seed0/config.json'):
        check(f'file {rel}', (REPO / rel).exists())


def ocd_side():
    pyd = sorted(OCD_FIRST_LAP.glob('ocd_native*.pyd' if sys.platform == 'win32' else 'ocd_native*.so'))
    check('ocd_native binary in ocd_first_lap', bool(pyd), pyd[0].name if pyd else
          f'copy ocd_native.cp312-win_amd64.pyd into {OCD_FIRST_LAP} (README_FIRST.md)')
    if try_import('ocd_native') is None:
        return
    try_import('ocd_lap.backend')
    track_mod = try_import('ocd_lap.track')
    if track_mod is not None:
        try:
            t = track_mod.Track('Catalunya')
            check('Track("Catalunya")', True, f'{len(t.path)} path points')
        except Exception as e:   # noqa: BLE001
            check('Track("Catalunya")', False, f'{type(e).__name__}: {e}')
    try:
        from ocd_lap.cars import CARS   # noqa: F401  (the sf70h_f1 car exists only on ocd_first_lap sf70h-f1)
        check('ocd_first_lap car sf70h_f1 (branch sf70h-f1)', 'sf70h_f1' in CARS)
    except Exception as e:   # noqa: BLE001
        check('ocd_first_lap car sf70h_f1 (branch sf70h-f1)', False, f'{type(e).__name__}: {e}')
    try_import('rlsetup.driving.train')


def line_endings():
    """A CRLF checkout changes every source hash (platforms/windows/README.md, 'Line endings')."""
    for p in (REPO / 'rlsetup' / 'driving' / 'train.py', OCD_FIRST_LAP / 'ocd_lap' / 'track.py'):
        if p.exists():
            lf = b'\r\n' not in p.read_bytes()
            check(f'LF line endings {p.name}', lf,
                  '' if lf else 'CRLF: re-checkout with core.autocrlf false (README_FIRST.md, troubleshooting)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--no-ocd', action='store_true', help='skip the ocd_native / ocd_lap checks')
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    print(f'repo {REPO}\nocd_first_lap {OCD_FIRST_LAP}\npython {sys.executable}')
    versions()
    ac_side()
    if not a.no_ocd:
        ocd_side()
    line_endings()
    print('\nall required checks passed' if not failed else f'\n{len(failed)} required check(s) failed')
    sys.exit(1 if failed else 0)
