"""Read-only AC training dashboard; no AC sockets or control commands.

python scripts/ac_dashboard.py --learner runs/windows/ac/tcp-r6b-pert4-001 \
    --worker runs/windows/ac/tcp-worker-r6b-pert4-001 --port 8765

Only the dashboard and its JSON API are served, never the repository or logs.
Live instruments use Windows shared memory, independently of the driver's UDP
connection. Completed drive statistics come from the learner's progress.jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes as ct
import json
import math
import mmap
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class Physics(ct.Structure):
    _pack_ = 4
    _fields_ = [('packet', ct.c_int32), ('gas', ct.c_float), ('brake', ct.c_float),
               ('fuel', ct.c_float), ('gear', ct.c_int32), ('rpm', ct.c_int32),
               ('steer', ct.c_float), ('speed', ct.c_float), ('velocity', ct.c_float*3),
               ('acceleration', ct.c_float*3), ('slip', ct.c_float*4),
               ('load', ct.c_float*4), ('pressure', ct.c_float*4),
               ('angular_speed', ct.c_float*4), ('wear', ct.c_float*4),
               ('dirt', ct.c_float*4), ('temperature', ct.c_float*4)]


class Graphics(ct.Structure):
    _pack_ = 4
    _fields_ = [('packet', ct.c_int32), ('status', ct.c_int32), ('session', ct.c_int32),
               ('current_text', ct.c_wchar*15), ('last_text', ct.c_wchar*15),
               ('best_text', ct.c_wchar*15), ('split', ct.c_wchar*15),
               ('laps', ct.c_int32), ('position', ct.c_int32),
               ('current_ms', ct.c_int32), ('last_ms', ct.c_int32), ('best_ms', ct.c_int32),
               ('session_left', ct.c_float), ('distance', ct.c_float),
               ('in_pit', ct.c_int32), ('sector', ct.c_int32),
               ('last_sector_ms', ct.c_int32), ('number_laps', ct.c_int32),
               ('tyre_compound', ct.c_wchar*33), ('replay_multiplier', ct.c_float),
               ('normalized_position', ct.c_float), ('coordinates', ct.c_float*3)]


class LiveMemory:
    def __init__(self):
        self.maps = {}
        self.last_packet = None
        self.changed_at = 0.

    def snapshot(self):
        if os.name != 'nt':
            return {'available': False}
        try:
            for tag, shape in [('physics', Physics), ('graphics', Graphics)]:
                if tag not in self.maps:
                    self.maps[tag] = mmap.mmap(-1, ct.sizeof(shape), tagname='acpmf_'+tag,
                                               access=mmap.ACCESS_READ)
            p = Physics.from_buffer_copy(self.maps['physics'][:ct.sizeof(Physics)])
            g = Graphics.from_buffer_copy(self.maps['graphics'][:ct.sizeof(Graphics)])
            now = time.time()
            if p.packet != self.last_packet:
                self.last_packet, self.changed_at = p.packet, now
            return dict(available=g.status == 2 and now-self.changed_at < 5,
                        status={0: 'Off', 1: 'Replay', 2: 'Live', 3: 'Paused'}.get(g.status, 'Unknown'),
                        speed_kmh=round(p.speed, 1), gear='R' if p.gear == 0 else 'N' if p.gear == 1 else str(p.gear-1),
                        rpm=p.rpm, throttle=p.gas, brake=p.brake, fuel_l=p.fuel,
                        tyre_temperatures=list(p.temperature),
                        current_lap_s=g.current_ms/1000 if g.current_ms > 0 else None,
                        ac_last_lap_s=g.last_ms/1000 if g.last_ms > 0 else None,
                        ac_best_lap_s=g.best_ms/1000 if g.best_ms > 0 else None,
                        completed_laps=g.laps, track_position=g.normalized_position,
                        coordinates=[g.coordinates[2], g.coordinates[0]])
        except (OSError, ValueError):
            return {'available': False}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def rows_from(path):
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get('cycle'), int):
                rows.append(row)
        except ValueError:
            pass  # A writer may still be appending the final line.
    return list({row['cycle']: row for row in rows}.values())


def summarize(rows):
    """A full reference length is a completion; teleport times stay separate."""
    drives = [r for r in rows if r.get('transitions', 0) > 0]
    complete = [r for r in drives if r.get('stop_reason') == 'lap_complete']
    lap_times = [r['seconds'] for r in complete
                 if r.get('start_mode') != 'teleport' and (r.get('seconds') or 0) > 0]
    evaluation = [r for r in drives if r.get('mode') == 'deterministic']
    recent = drives[-20:]
    fraction = lambda rr: 100*sum(r.get('stop_reason') == 'lap_complete' for r in rr)/len(rr) if rr else None
    return dict(drives=len(drives), attempts=len(rows), completions=len(complete),
                finish_rate=fraction(drives), recent_finish_rate=fraction(recent),
                eval_finish_rate=fraction(evaluation), evaluation_drives=len(evaluation),
                best_lap_s=min(lap_times) if lap_times else None,
                last_lap_s=lap_times[-1] if lap_times else None,
                timed_completions=len(lap_times),
                teleport_completions=sum(r.get('start_mode') == 'teleport' for r in complete),
                transitions=sum(r.get('transitions', 0) for r in rows),
                actor_updates=sum(r.get('updates', 0) for r in rows),
                critic_updates=sum(r.get('critic_updates', 0) for r in rows),
                best_progress_m=max((r.get('progress_m') or 0 for r in drives), default=0),
                failures=dict(Counter(r.get('stop_reason', 'unknown') for r in rows
                                      if r.get('stop_reason') != 'lap_complete')))


class Dashboard:
    def __init__(self, learner, worker, reference, label, learner_pid=None, worker_pid=None):
        self.learner, self.worker, self.label = learner, worker, label
        self.learner_pid, self.worker_pid = learner_pid, worker_pid
        self.live = LiveMemory()
        self.lock = threading.Lock()
        self.cached_at, self.cached = 0., {}
        self.durations = {}
        ref = read_json(reference) if reference else {}
        self.track = ref.get('rows', [])[::10]
        rows = ref.get('rows', [])
        self.length = (sum(math.dist(a, b) for a, b in zip(rows, rows[1:]))
                       + math.dist(rows[-1], rows[0])) if rows else 4602.

    def duration(self, cycle):
        if cycle in self.durations:
            return self.durations[cycle]
        candidates = [self.learner/f'cycle-{cycle:08d}'/'chunk.npz',
                      self.worker/f'chunk-{cycle:08d}.npz']
        for path in candidates:
            try:
                import numpy as np
                with np.load(path, allow_pickle=False) as chunk:
                    seconds = float(chunk['dt'].sum())
                self.durations[cycle] = seconds
                return seconds
            except (OSError, ValueError, KeyError):
                pass
        return None

    def snapshot(self):
        with self.lock:
            now = time.time()
            if now-self.cached_at >= 2:
                raw = rows_from(self.learner/'progress.jsonl')
                rows = [dict(row, seconds=self.duration(row['cycle'])) for row in raw]
                schedule = read_json(self.learner/'server.json')
                state = read_json(self.worker/'worker.json')
                stats = summarize(rows)
                self.cached = dict(label=self.label, updated_at=now, stats=stats,
                                   recent=rows[-50:], history=rows[-1000:],
                                   worker_cycle=state.get('cycle', 0),
                                   last_update=rows[-1].get('unix_time_s') if rows else None,
                                   warmup_transitions=schedule.get('warmup_transitions', 5000),
                                   track=self.track, track_length_m=self.length,
                                   learner_running=process_alive(self.learner_pid),
                                   worker_running=process_alive(self.worker_pid))
                self.cached_at = now
            return dict(self.cached, live=self.live.snapshot(), server_time=time.time())


def process_alive(pid):
    if not pid or os.name != 'nt':
        return None
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ct.c_void_p
    kernel.OpenProcess.argtypes = [ct.c_uint32, ct.c_int, ct.c_uint32]
    kernel.GetExitCodeProcess.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32)]
    kernel.CloseHandle.argtypes = [ct.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ct.c_uint32()
        return bool(kernel.GetExitCodeProcess(handle, ct.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learner', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--label', default='R6b / pert4')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--learner-pid', type=int)
    parser.add_argument('--worker-pid', type=int)
    args = parser.parse_args()
    dashboard = Dashboard(args.learner, args.worker, args.reference, args.label,
                          args.learner_pid, args.worker_pid)
    html = Path(__file__).with_suffix('.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                body, content_type = html, 'text/html; charset=utf-8'
            elif path == '/api/stats':
                body = json.dumps(dashboard.snapshot(), allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            elif path == '/health':
                body, content_type = b'{"ok":true}', 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        print(f'AC dashboard: http://{args.host}:{args.port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
