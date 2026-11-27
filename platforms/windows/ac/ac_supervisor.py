"""Unattended AC training: restart Assetto Corsa and the worker whenever the worker stops.

    python platforms/windows/ac/ac_supervisor.py --setup r6b --port 2352 \
        --work-dir runs/windows/ac/tcp-worker-r6b-pert4-001 -- <ac_worker_f1r5.ps1 arguments>

Why: a domain policy refresh can make Windows re-enumerate devices (25 Nov 2026 04:22: CSP logged
WM_DEVICECHANGE remove + arrive). AC binds controllers only at start-up, so it loses vJoy for good; the
worker's preparations then time out and it exits after 5 failures. Only a fresh acs.exe recovers.

Loop: while the worker runs, watch it (and kill it if nothing it writes changes for --stall-min minutes).
When it is gone and the learner is listening: close AC (WM_CLOSE, then kill), pin the track grip in
race.ini, start acs.exe (race.ini otherwise as Content Manager left it: SF70H, Barcelona GP, Hotlap),
check the grip is 100 %, load the named setup in the pit menu, check the loaded setup against the .ini
(fuel, brake bias, ERS: from AC's shared memory), click Drive, start the worker again.
Grip: Content Manager's race.ini had a random dynamic track (RANDOMNESS=200); on 25 Nov 2026 a restart
came up at 85 % grip and every drive crashed. Training ran at 100 %, so 100 % is pinned and checked. Every click is posted to AC's window, so this works while the PC is locked. Nothing is
started on a setup or grip mismatch. Events go to runs/windows/ac/logs/supervisor.jsonl with a screenshot of AC's
window per step (logs/supervisor-shots, last 40 kept).
"""
from __future__ import annotations

import argparse
import ctypes as ct
import ctypes.wintypes as wt
import configparser
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import ac_shm  # noqa: E402

AC_INSTALL = Path(os.environ.get('AC_INSTALL', r'C:\Program Files (x86)\Steam\steamapps\common\assettocorsa'))
CAR, TRACK = 'ks_ferrari_sf70h', 'ks_barcelona'
LOGS = REPO / 'runs' / 'windows' / 'ac' / 'logs'

# Pit-menu layout at AC's 1024x768 client area (measured 25 Nov 2026 on this PC; video.ini 1024x768 windowed).
CLIENT = (1024, 768)
DRIVE, SETUP_TAB, LOAD = (50, 173), (50, 273), (218, 502)
TRACK_LIST_X, SETUP_LIST_X, LIST_Y0, ROW_H = 230, 423, 185, 19
MENU_BAR = (5, 95, 130, 620)   # x0, x1, y0, y1: the red sidebar, only drawn in the pit menu


def documents():
    buf = ct.create_unicode_buffer(260)
    if ct.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:   # CSIDL_PERSONAL (redirected on AUB PCs)
        return Path(buf.value)
    return Path.home() / 'Documents'


# ---------------------------------------------------------------- pure helpers (unit-tested)
def setup_expectation(ini_text):
    """Shared-memory values a setup .ini fixes: {name: value} for the keys present."""
    cp = configparser.ConfigParser(strict=False)
    cp.read_string(ini_text)
    get = lambda k: float(cp[k]['VALUE']) if cp.has_section(k) and 'VALUE' in cp[k] else None
    want = dict(fuel=get('FUEL'), brake_bias=None if get('FRONT_BIAS') is None else get('FRONT_BIAS')/100,
                ers_recovery=get('MGUK_RECOVERY'), ers_power=get('MGUK_DELIVERY'), ers_heat_charging=get('MGUH_MODE'))
    return {k: v for k, v in want.items() if v is not None}


def setup_matches(want, have, tol=1e-3):
    """[] if every expected value is loaded, else the mismatching names."""
    return [k for k, v in want.items() if have.get(k) is None or abs(float(have[k])-v) > tol]


def list_rows(setups_dir, track, setup):
    """Row indices of the track folder and of the setup in AC's Setup I/O lists (alphabetical). AC lists only
    folders that hold a setup (25 Nov 2026: 'generic', 'ks_barcelona' out of 22 SF70H folders)."""
    tracks = sorted((p.name for p in setups_dir.iterdir() if p.is_dir() and any(p.glob('*.ini'))), key=str.lower)
    names = sorted((p.stem for p in (setups_dir/track).glob('*.ini')), key=str.lower)
    return tracks.index(track), names.index(setup)


GRIP = dict(SESSION_START='100', RANDOMNESS='0', LAP_GAIN='0')   # [DYNAMIC_TRACK]: 100 %, constant


def pin_grip(text):
    """race.ini text with [DYNAMIC_TRACK] set to GRIP (other sections and keys unchanged, line ends kept)."""
    nl = '\r\n' if '\r\n' in text else '\n'
    lines = text.splitlines()
    key = lambda s: s.split('=', 1)[0].strip()
    section, present = None, set()
    for line in lines:
        s = line.strip()
        if s.startswith('[') and s.endswith(']'):
            section = s[1:-1]
        elif section == 'DYNAMIC_TRACK' and '=' in s:
            present.add(key(s))
    out, section = [], None
    for line in lines:
        s = line.strip()
        if s.startswith('[') and s.endswith(']'):
            section = s[1:-1]
            out.append(line)
            if section == 'DYNAMIC_TRACK':
                out += [f'{k}={v}' for k, v in GRIP.items() if k not in present]
            continue
        if section == 'DYNAMIC_TRACK' and '=' in s and key(s) in GRIP:
            line = f'{key(s)}={GRIP[key(s)]}'
        out.append(line)
    if not any(line.strip() == '[DYNAMIC_TRACK]' for line in lines):
        out += ['[DYNAMIC_TRACK]'] + [f'{k}={v}' for k, v in GRIP.items()]
    return nl.join(out)+nl


def menu_visible(rgb):
    """True if the pit menu's red sidebar fills most of its strip (HxWx3 uint8 image of the client area)."""
    x0, x1, y0, y1 = MENU_BAR
    h, w = rgb.shape[:2]
    if w < x1 or h < y1:
        return False
    s = rgb[y0:y1, x0:x1].astype(int)
    red = (s[..., 0] > 150) & (s[..., 1] < 60) & (s[..., 2] < 60)
    return red.mean() > 0.5


# ---------------------------------------------------------------- Windows plumbing
u32, gdi = (ct.windll.user32, ct.windll.gdi32) if os.name == 'nt' else (None, None)
if u32:
    u32.SetProcessDPIAware()


def log(event, **kw):
    LOGS.mkdir(parents=True, exist_ok=True)
    row = dict(unix_time_s=time.time(), local=time.strftime('%Y-%m-%d %H:%M:%S'), event=event, **kw)
    with open(LOGS/'supervisor.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def ac_window():
    return u32.FindWindowW('acsW', None) or None


def client_size(h):
    r = wt.RECT()
    u32.GetClientRect(h, ct.byref(r))
    return r.right, r.bottom


def ensure_client_size(h):
    """Resize AC's window so its client area is 1024x768 (a lock can shrink it); True if it is."""
    if client_size(h) == CLIENT:
        return True
    style, ex = u32.GetWindowLongW(h, -16), u32.GetWindowLongW(h, -20)
    r = wt.RECT(0, 0, *CLIENT)
    u32.AdjustWindowRectEx(ct.byref(r), style, False, ex)
    u32.ShowWindow(h, 9)   # SW_RESTORE
    u32.SetWindowPos(h, None, 0, 0, r.right-r.left, r.bottom-r.top, 0x0002 | 0x0004 | 0x0010)   # NOMOVE|NOZORDER|NOACTIVATE
    time.sleep(1.)
    return client_size(h) == CLIENT


def click(h, xy):
    x, y = xy
    lp = (y << 16) | x
    u32.PostMessageW(h, 0x200, 0, lp)        # WM_MOUSEMOVE
    time.sleep(.15)
    u32.PostMessageW(h, 0x201, 1, lp)        # WM_LBUTTONDOWN
    time.sleep(.12)
    u32.PostMessageW(h, 0x202, 0, lp)        # WM_LBUTTONUP
    time.sleep(1.)


def capture(h):
    """HxWx3 RGB image of AC's client area (PrintWindow works on the locked desktop)."""
    import numpy as np
    w, hh = client_size(h)
    hdc = u32.GetDC(h)
    mdc, bmp = gdi.CreateCompatibleDC(hdc), gdi.CreateCompatibleBitmap(hdc, w, hh)
    gdi.SelectObject(mdc, bmp)
    u32.PrintWindow(h, mdc, 3)   # PW_CLIENTONLY | PW_RENDERFULLCONTENT

    class BIH(ct.Structure):
        _fields_ = [('biSize', wt.DWORD), ('biWidth', wt.LONG), ('biHeight', wt.LONG), ('biPlanes', wt.WORD),
                    ('biBitCount', wt.WORD), ('biCompression', wt.DWORD), ('biSizeImage', wt.DWORD),
                    ('x', wt.LONG), ('y', wt.LONG), ('used', wt.DWORD), ('important', wt.DWORD)]
    bi = BIH(ct.sizeof(BIH), w, -hh, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = (ct.c_ubyte*(w*hh*4))()
    gdi.GetDIBits(mdc, bmp, 0, hh, buf, ct.byref(bi), 0)
    gdi.DeleteObject(bmp)
    gdi.DeleteDC(mdc)
    u32.ReleaseDC(h, hdc)
    return np.frombuffer(buf, np.uint8).reshape(hh, w, 4)[:, :, 2::-1].copy()


def shot(h, tag):
    try:
        from PIL import Image
        d = LOGS/'supervisor-shots'
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(capture(h)).save(d/f'{time.strftime("%Y%m%d-%H%M%S")}-{tag}.png')
        for old in sorted(d.glob('*.png'))[:-40]:
            old.unlink()
    except Exception as e:   # noqa: BLE001 - screenshots are diagnostics only
        log('screenshot_failed', error=repr(e))


def alive(pid):
    if not pid:
        return False
    hp = ct.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not hp:
        return False
    code = wt.DWORD()
    ct.windll.kernel32.GetExitCodeProcess(hp, ct.byref(code))
    ct.windll.kernel32.CloseHandle(hp)
    return code.value == 259   # STILL_ACTIVE


def ps(cmd):
    return subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True,
                          creationflags=0x08000000).stdout.strip()


def running_worker(work_dir):
    """PID of a worker python already writing to work_dir, else None."""
    name = Path(work_dir).name
    out = ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
             f"'rlsetup.assetto.distributed worker' -and $_.CommandLine -match '{name}' }} | "
             "Select-Object -ExpandProperty ProcessId")
    pids = [int(x) for x in out.split() if x.isdigit()]
    return pids[0] if pids else None


def kill_tree(pid):
    subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], capture_output=True, creationflags=0x08000000)


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=3):
            return True
    except OSError:
        return False


def acs_pids():
    return [int(x) for x in ps('(Get-Process acs -ErrorAction SilentlyContinue).Id').split() if x.isdigit()]


def newest_activity(paths):
    """Latest write time among the files a working worker keeps touching (one readiness trace per drive)."""
    files = []
    for p in map(Path, paths):
        files += [p] if p.suffix else [p/'worker.json', p/'retries.jsonl', *sorted(p.glob('readiness-*.jsonl'))[-1:]]
    return max((f.stat().st_mtime for f in files if f.is_file()), default=0.)


# ---------------------------------------------------------------- recovery
def close_ac():
    for pid in acs_pids():
        h = ac_window()
        if h:
            u32.PostMessageW(h, 0x0010, 0, 0)   # WM_CLOSE: AC saves generic/last.ini and the replay
        for _ in range(60):
            if not alive(pid):
                break
            time.sleep(1.)
        if alive(pid):
            log('ac_kill', pid=pid)
            kill_tree(pid)
            time.sleep(3.)


def write_race_ini():
    path = documents()/'Assetto Corsa'/'cfg'/'race.ini'
    text = path.read_text(encoding='utf-8', errors='replace')
    pinned = pin_grip(text)
    if pinned != text:
        path.write_text(pinned, encoding='utf-8', newline='')
        log('race_ini_grip_pinned', **GRIP)


def start_ac(timeout_s=300.):
    write_race_ini()
    env = dict(os.environ)
    vc = Path(os.environ['LOCALAPPDATA'])/'ac-vcruntime-x64'   # x64 VC++ 2013 runtime without admin (README step 2)
    env['PATH'] = f'{vc};{env["PATH"]}'
    p = subprocess.Popen([str(AC_INSTALL/'acs.exe')], cwd=str(AC_INSTALL), env=env)
    end = time.time()+timeout_s
    while time.time() < end:
        s = ac_shm.snapshot()
        if ac_window() and listening(2347) and s and s['status'] == 2:
            time.sleep(8.)   # let the pit menu finish drawing
            return p.pid
        if p.poll() is not None:
            raise RuntimeError(f'acs.exe exited with {p.returncode}')
        time.sleep(2.)
    raise TimeoutError('AC did not reach the pit menu')


def load_setup(h, setup, want):
    setups = documents()/'Assetto Corsa'/'setups'/CAR
    ti, si = list_rows(setups, TRACK, setup)
    click(h, SETUP_TAB)
    click(h, (TRACK_LIST_X, LIST_Y0+ROW_H*ti))
    click(h, (SETUP_LIST_X, LIST_Y0+ROW_H*si))
    click(h, LOAD)
    time.sleep(1.)
    shot(h, 'setup')
    s = ac_shm.snapshot() or {}
    bad = setup_matches(want, s.get('setup', {}))
    return bad, s.get('setup')


def recover(args, want):
    log('recovery_start')
    close_ac()
    pid = start_ac()
    h = ac_window()
    log('ac_started', pid=pid, client=client_size(h))
    if not ensure_client_size(h):
        raise RuntimeError(f'AC client area is {client_size(h)}, clicks are laid out for {CLIENT}')
    shot(h, 'menu')
    if not menu_visible(capture(h)):
        raise RuntimeError('pit menu not visible after start')
    grip = (ac_shm.snapshot() or {}).get('surface_grip')
    if grip is None or abs(grip-1.) > 1e-3:
        raise RuntimeError(f'track grip is {grip}, not 1.0 (race.ini [DYNAMIC_TRACK])')
    bad, have = load_setup(h, args.setup, want)
    if bad:
        raise RuntimeError(f'setup {args.setup} not loaded: {bad} differ (expected {want}, AC has {have})')
    log('setup_loaded', setup=args.setup, values=have, surface_grip=grip)
    click(h, DRIVE)
    time.sleep(3.)
    shot(h, 'drive')
    if menu_visible(capture(h)):
        raise RuntimeError('still in the pit menu after clicking Drive')
    log('on_track')


def start_worker(args, attempt):
    stem = LOGS/f'supervised-worker-{time.strftime("%Y%m%d-%H%M%S")}'
    out, err = open(f'{stem}.out.log', 'w'), open(f'{stem}.err.log', 'w')
    cmd = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
           str(HERE/'ac_worker_f1r5.ps1'), *args.worker_args]
    p = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=err, creationflags=0x08000000)
    log('worker_started', pid=p.pid, attempt=attempt)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--setup', required=True, help=f'setup name in setups/{CAR}/{TRACK} (e.g. r6b)')
    ap.add_argument('--port', type=int, required=True, help='learner port on 127.0.0.1')
    ap.add_argument('--work-dir', type=Path, required=True, help='the worker --WorkDir (watched for activity)')
    ap.add_argument('--progress', type=Path, help='learner progress.jsonl (also watched for activity)')
    ap.add_argument('--stall-min', type=float, default=10., help='kill a worker that writes nothing for this long')
    ap.add_argument('worker_args', nargs=argparse.REMAINDER, help='after --: arguments for ac_worker_f1r5.ps1')
    args = ap.parse_args()
    if args.worker_args[:1] == ['--']:
        args.worker_args = args.worker_args[1:]
    ini = documents()/'Assetto Corsa'/'setups'/CAR/TRACK/f'{args.setup}.ini'
    want = setup_expectation(ini.read_text(errors='replace'))
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS/'supervisor.pid').write_text(str(os.getpid()))
    log('supervisor_start', setup=args.setup, expected=want, pid=os.getpid())

    worker_pid, proc, attempt, failures = running_worker(args.work_dir), None, 0, 0
    started_at = time.time()
    if worker_pid:
        log('worker_adopted', pid=worker_pid)
    watch = [args.work_dir] + ([args.progress] if args.progress else [])
    while True:
        try:
            if worker_pid and alive(worker_pid):
                idle = time.time()-max(newest_activity(watch), started_at)
                if idle > 60*args.stall_min:
                    log('worker_stalled', pid=worker_pid, idle_s=round(idle))
                    kill_tree(worker_pid)
                time.sleep(15.)
                continue
            if worker_pid:
                log('worker_exited', pid=worker_pid, exit_code=proc.poll() if proc else None)
                worker_pid = None
            if not listening(args.port):
                log('learner_not_listening', port=args.port)
                time.sleep(60.)
                continue
            try:
                recover(args, want)
                failures = 0
            except Exception as e:   # noqa: BLE001 - every failure is retried after a back-off
                failures += 1
                wait = min(60*failures, 900)
                log('recovery_failed', error=repr(e), failures=failures, retry_in_s=wait)
                h = ac_window()
                if h:
                    shot(h, 'failed')
                time.sleep(wait)
                continue
            attempt += 1
            proc = start_worker(args, attempt)
            worker_pid, started_at = proc.pid, time.time()
            time.sleep(30.)
        except Exception as e:   # noqa: BLE001 - the supervisor must outlive any one bad poll
            log('supervisor_error', error=repr(e))
            time.sleep(30.)


if __name__ == '__main__':
    main()
