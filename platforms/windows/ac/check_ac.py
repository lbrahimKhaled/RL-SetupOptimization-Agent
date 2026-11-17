"""Preflight for Assetto Corsa training on this PC (game and learner on the same machine).

    python platforms/windows/ac/check_ac.py           # static checks: env, files, vJoy, plugin
    python platforms/windows/ac/check_ac.py --live    # + ask the running plugin for its static info

Run with acenv's python and env_ac.ps1's PYTHONPATH (setup_acenv.ps1 does both). Prints one line
per check (OK / MISSING / FAIL) and exits 1 if a required check fails. Nothing is changed.
"""
from pathlib import Path
import argparse
import ast
import importlib
import os
import socket
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AC_GYM = Path(os.environ.get('AC_GYM', REPO.parent / 'assetto_corsa_gym'))
AC_INSTALL = Path(os.environ.get('AC_INSTALL', r'C:\Program Files (x86)\Steam\steamapps\common\assettocorsa'))
def _documents():
    """The Windows Documents known folder, which AC uses for cfg/ and logs/. On AUB machines it is
    redirected to a network share, not %USERPROFILE%\\Documents."""
    if sys.platform == 'win32':
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:   # CSIDL_PERSONAL
            return Path(buf.value)
    return Path.home() / 'Documents'


AC_DOCS = _documents() / 'Assetto Corsa'
VJOY_DLL = Path(r'C:\Program Files\vJoy\x64\vJoyInterface.dll')   # path hard-coded in AssettoCorsaEnv/vjoy.py
HOST, MGMT_PORT = '127.0.0.1', 2347
CAR, TRACK = 'ks_ferrari_sf70h', 'ks_barcelona'   # rlsetup.assetto: SF70H on Barcelona GP (Parked guard)

failed = []


def check(name, ok, detail='', required=True):
    tag = 'OK' if ok else ('FAIL' if required else 'MISSING')
    print(f'[{tag:7}] {name}' + (f': {detail}' if detail else ''))
    if required and not ok:
        failed.append(name)
    return ok


def python_side():
    check('python 3.12', sys.version_info[:2] == (3, 12), sys.version.split()[0])
    for p in (REPO, AC_GYM, AC_GYM / 'assetto_corsa_gym'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    check('assetto_corsa_gym checkout (plugin source)', (AC_GYM / 'assetto_corsa_gym').is_dir(), str(AC_GYM))
    # rlsetup.assetto (the Lenovo worker / learner) needs only numpy + torch: ocdenv or acenv.
    # The upstream assetto_corsa_gym trainer (train_ac.ps1) needs acenv: optional.
    for mod, required in (('torch', True), ('rlsetup.assetto.distributed', True), ('rlsetup.driving.agent', True),
                          ('AssettoCorsaEnv.ac_env', False), ('AssettoCorsaEnv.assettoCorsa', False)):
        try:
            importlib.import_module(mod)
            check(f'import {mod}', True)
        except Exception as e:   # noqa: BLE001 - report every import problem
            check(f'import {mod}', False, f'{type(e).__name__}: {e}' + ('' if required else ' (acenv only)'),
                  required=required)
    try:
        import torch
        check('torch CUDA', torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available', required=False)
    except Exception:   # noqa: BLE001
        pass


def game_side():
    check('vJoy driver (vJoyInterface.dll)', VJOY_DLL.exists(), 'install vJoy 2.x (needs admin)' if not VJOY_DLL.exists() else '')
    acs = AC_INSTALL / 'acs.exe'
    check('Assetto Corsa installed', acs.exists(), str(AC_INSTALL) if acs.exists() else 'acs.exe missing: finish the Steam download')
    if not acs.exists():
        return
    csp = (AC_INSTALL / 'extension').is_dir()
    check('Custom Shaders Patch', csp, '' if csp else 'Content Manager > Settings > CSP > Install')
    cfg = AC_INSTALL / 'apps' / 'python' / 'sensors_par' / 'config.py'
    if check('sensors_par plugin', cfg.exists(), 'run install_ac_plugin.ps1' if not cfg.exists() else ''):
        text = cfg.read_text(errors='replace')
        check('plugin binds 127.0.0.1 (single PC)', '"0.0.0.0"' not in text)
        check('plugin: drives vJoy itself, 100 Hz (rlsetup.assetto)', 'self.vjoy_executed_by_server = True' in text
              and 'self.sampling_freq = 100' in text)
    check('AC socket libs in system\\x64', (AC_INSTALL / 'system' / 'x64' / 'Lib').is_dir())
    check('Vjoy.ini controller preset', (AC_DOCS / 'cfg' / 'controllers' / 'savedsetups' / 'Vjoy.ini').exists())
    # X86=1 makes the launcher start acs_x86.exe: the plugin then fails to import the system\x64 socket/ctypes
    # libs ("DLL load failed: %1 is not a valid Win32 application" in logs\py_log.txt) and never opens its ports.
    launcher = AC_DOCS / 'cfg' / 'launcher.ini'
    x86 = launcher.exists() and any(l.strip() == 'X86=1' for l in launcher.read_text(errors='replace').splitlines())
    check('AC runs 64-bit (launcher.ini X86=0)', not x86, 'set X86=0 in ' + str(launcher) if x86 else '')
    check(f'car {CAR}', (AC_INSTALL / 'content' / 'cars' / CAR).is_dir())
    check(f'track {TRACK}', (AC_INSTALL / 'content' / 'tracks' / TRACK).is_dir())


def rlsetup_side():
    work = REPO / 'runs' / 'windows' / 'ac'
    check('steering calibration (this PC)', (work / 'steering.json').exists(),
          'run the calibration (README walkthrough step 6)', required=False)
    key = (work / 'tcp-session.key').exists()
    check('TCP token', key, '' if key else 'ac_server.ps1 / ac_worker.ps1 create it', required=False)
    check('learner rlsetup/assetto/learn.py', (REPO / 'rlsetup' / 'assetto' / 'learn.py').exists())
    missing = 'copy from the Mac (README "Model files from the Mac")'
    check('source actor runs/ac/source_actor', (REPO / 'runs' / 'ac' / 'source_actor' / 'manifest.json').exists(),
          missing, required=False)
    check('--critic-init checkpoint runs/mac/D2/bcn/seed0/checkpoints/best.zip',
          (REPO / 'runs' / 'mac' / 'D2' / 'bcn' / 'seed0' / 'checkpoints' / 'best.zip').exists(), missing, required=False)


def live():
    try:
        with socket.create_connection((HOST, MGMT_PORT), timeout=5) as s:
            s.sendall(b'get_static_info')
            s.settimeout(5)
            info = ast.literal_eval(s.recv(2**20).decode())
        check('plugin reachable on 127.0.0.1:2347', True,
              f"track={info.get('TrackName')} {info.get('TrackConfiguration') or ''}".strip())   # static info has no car name
    except Exception as e:   # noqa: BLE001
        check('plugin reachable on 127.0.0.1:2347', False, f'{type(e).__name__}: {e} (is a Hotlap session running?)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--live', action='store_true', help='also query the running plugin')
    a = ap.parse_args()
    python_side()
    game_side()
    rlsetup_side()
    if a.live:
        live()
    print('\nall required checks passed' if not failed else f'\n{len(failed)} required check(s) failed')
    sys.exit(1 if failed else 0)
