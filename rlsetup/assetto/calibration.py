"""Bounded stationary steering calibration. Offline plan unless --execute.

Angles are radians, positions metres, speeds m/s. Heading vectors are projected
into (AC z, AC x); front-minus-rear is positive left. No gear or reset command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

from .management import ConnectionConfig, ManagementClient, PROTOCOL_SOURCE_COMMIT
from .telemetry import CaptureConfig, _hash_sources, _numeric, _open_socket, _packet, _utc


@dataclass(frozen=True)
class CalibrationConfig:
    """Technical measurement guards, not vehicle parameters; time units seconds."""
    host: str
    full_range: bool = False
    setup_id: str = 'default_user_confirmed'
    port: int = 2345
    management_port: int = 2347
    handshake_s: float = 2.
    overall_s: float = 15.
    baseline_local_s: float = 2.
    baseline_server_s: float = .4
    baseline_samples: int = 10
    phase_server_s: float = .8
    tail_server_s: float = .25
    settled_samples: int = 4
    settling_rad: float = .005
    motion_m: float = .02
    speed_mps: float = .05
    freshness_s: float = .2
    brake_server_s: float = .4
    brake_local_s: float = .8
    max_drain: int = 32
    max_packets: int = 3000

    def __post_init__(self):
        CaptureConfig(self.host, port=self.port)
        if not self.setup_id:
            raise ValueError('setup identity is required')
        for key, value in asdict(self).items():
            if key in ('host', 'setup_id', 'full_range'):
                continue
            if not _numeric(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.tail_server_s >= self.phase_server_s:
            raise ValueError('settling tail must be shorter than phase')

    @property
    def phases(self):
        return ((0., .1, .25, .5, 1., 0., -.1, -.25, -.5, -1., 0.)
                if self.full_range else (0., .1, 0., -.1, 0.))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def number(packet, key):
    value = packet.get(key)
    if not _numeric(value):
        raise ValueError(f'missing/nonfinite telemetry: {key}')
    return float(value)


def circular_mean(angles):
    x, y = sum(math.cos(a) for a in angles), sum(math.sin(a) for a in angles)
    if math.hypot(x, y) < 1e-8:
        raise ValueError('degenerate circular mean')
    return math.atan2(y, x)


def heading(packet):
    """Return (front-minus-rear angle, rear mean), radians in (AC z, AC x)."""
    angles = []
    for wheel in ('FL', 'FR', 'RL', 'RR'):
        x = number(packet, f'tyreContactHeading_{wheel}_x')
        y = number(packet, f'tyreContactHeading_{wheel}_y')
        z = number(packet, f'tyreContactHeading_{wheel}_z')
        if math.hypot(x, z) < 1e-8:
            raise ValueError('degenerate tyre heading vector')
        angles.append(math.atan2(x, z))
    front, rear = circular_mean(angles[:2]), circular_mean(angles[2:])
    return wrap(front - rear), rear


def command_payload(steps, steer=0., brake=-1.):
    """Seven-key AC Gym command; released gas=-1, brake=-1 or applied brake=+1."""
    if type(steps) is not int or steps < 0:
        raise ValueError('nonnegative integer server_steps required')
    if not _numeric(steer) or not -1 <= steer <= 1 or brake not in (-1., 1.):
        raise ValueError('invalid stationary control')
    return {'server_steps': steps, 'steer': float(steer), 'acc': -1.,
            'brake': float(brake), 'enable_gear_shift': 0, 'shift_up': 0, 'shift_down': 0}


class Guard:
    """Validate every received sample before it can affect an outbound command."""
    def __init__(self, config):
        self.config, self.previous, self.origin, self.rear = config, None, None, None

    def check(self, packet):
        stamp = number(packet, 'timestamp_ac')
        steps = packet.get('steps')
        if type(steps) is not int or steps < 0:
            raise ValueError('invalid steps counter')
        speed, gear = number(packet, 'speed'), number(packet, 'actualGear')
        gas, brake = number(packet, 'accStatus'), number(packet, 'brakeStatus')
        position = tuple(number(packet, f'world_position_{axis}') for axis in ('x', 'y', 'z'))
        angle, rear = heading(packet)
        if speed < 0 or speed >= self.config.speed_mps or gear != 1:
            raise ValueError(f'car moved or left neutral (actualGear must be 1, speed below '
                             f'{self.config.speed_mps:g} m/s; got actualGear {gear:g}, speed {speed:.3f} m/s)')
        if not 0 <= gas <= .01 or not 0 <= brake <= 1.01:
            raise ValueError('unexpected gas or invalid pedal status')
        if self.origin is None:
            self.origin, self.rear = position, rear
        if math.dist(position, self.origin) > self.config.motion_m:
            raise ValueError(f'world displacement exceeded stationary guard ('
                             f'{math.dist(position, self.origin):.3f} m > {self.config.motion_m:g} m)')
        if abs(wrap(rear - self.rear)) > self.config.settling_rad:
            raise ValueError('rear heading changed during stationary probe')
        if self.previous is not None:
            old_steps, old_stamp = self.previous
            if steps < old_steps or stamp < old_stamp:
                raise ValueError('telemetry counter/timestamp regression')
            if steps == old_steps or stamp == old_stamp:
                return False
        self.previous = steps, stamp
        return True


def measured_map(phases, baseline, config):
    """Merge agreeing centers and sort measured angle/command pairs together."""
    if [row['command'] for row in phases] != list(config.phases):
        raise ValueError('incomplete phase sequence')
    centers = [row['angle_rad'] for row in phases if row['command'] == 0]
    if (max(centers) - min(centers) > config.settling_rad
            or any(abs(wrap(c - baseline)) > config.settling_rad for c in centers)):
        raise ValueError('repeated centers inconsistent with neutral baseline')
    # Baseline is subtracted exactly once; metadata retains the raw neutral angle.
    pairs = [(wrap(row['angle_rad'] - baseline), row['command'])
             for row in phases if row['command'] != 0]
    pairs.append((wrap(circular_mean(centers) - baseline), 0.))
    # AC's steering lock can be reached at half the controller range. A full
    # command then repeats the same angle with small settling noise; retain the
    # half-range sample so inversion uses the first measured lock command.
    saturated_commands = []
    if config.full_range:
        angles = {command: angle for angle, command in pairs}
        for command in (-1., 1.):
            if abs(wrap(angles[command]-angles[command/2])) <= config.settling_rad:
                saturated_commands.append(command)
        pairs = [pair for pair in pairs if pair[1] not in saturated_commands]
    by_command = sorted(pairs, key=lambda item: item[1])
    differences = [b[0] - a[0] for a, b in zip(by_command, by_command[1:])]
    if not (all(d > 1e-5 for d in differences) or all(d < -1e-5 for d in differences)):
        raise ValueError('steering response is zero or nonmonotonic')
    pairs.sort()
    if not pairs[0][0] < 0 < pairs[-1][0]:
        raise ValueError('measured steering does not straddle neutral')
    return {'angle_rad': [p[0] for p in pairs], 'command': [p[1] for p in pairs],
            'baseline_angle_rad': baseline, 'validated': config.full_range,
            'saturated_commands': saturated_commands}


class Receiver:
    """Bounded latest-queue drain; validate and log even superseded samples."""
    def __init__(self, sock, config, write):
        self.sock, self.config, self.write = sock, config, write
        self.guard = Guard(config)
        self.last_fresh = time.monotonic()
        self.count, self.latest = 0, None

    def next(self, deadline):
        latest = None
        for index in range(self.config.max_drain):
            remaining = min(deadline, self.last_fresh + self.config.freshness_s) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('no fresh sample within local deadline')
            self.sock.settimeout(remaining if index == 0 else 0.)
            try:
                payload = self.sock.recv(65508)
            except (BlockingIOError, socket.timeout):
                if latest is not None:
                    break
                if index:
                    break
                raise TimeoutError('no fresh sample within local deadline')
            received = time.monotonic()
            if received > min(deadline, self.last_fresh + self.config.freshness_s):
                raise TimeoutError('late telemetry')
            self.count += 1
            if self.count > self.config.max_packets:
                raise ValueError('packet bound exceeded')
            if payload == b'identified':
                self.write({'type': 'protocol', 'receive_monotonic': received, 'message': 'identified'})
                continue
            if len(payload) > 65507:
                raise ValueError('oversized telemetry')
            packet = _packet(payload)
            self.write({'type': 'sample', 'receive_monotonic': received, 'telemetry': packet})
            if self.guard.check(packet):
                self.last_fresh = received
                latest = self.latest = packet
        if latest is None:
            return self.next(deadline)
        return latest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(config, output, trace, *, neutral_confirmed=False, at_controls=False, source_files=None):
    """Run only the bounded probe. Explicit readiness flags required before sockets."""
    if not neutral_confirmed or not at_controls:
        raise ValueError('execute requires --neutral-confirmed and --at-controls')
    output, trace = Path(output).expanduser(), Path(trace).expanduser()
    sources = _hash_sources(source_files)
    result = {'schema_version': 1, 'validated': False, 'setup_id': config.setup_id,
              'sampling_hz': 25, 'protocol_source_commit': PROTOCOL_SOURCE_COMMIT,
              'installed_source_revision': None, 'source_files': sources,
              'readiness': {'visible_neutral_confirmed': True, 'user_at_controls': True},
              'config': asdict(config), 'started_at': _utc()}
    # Reserve both paths before any network activity. Existing files are untouched.
    with output.open('x', encoding='utf-8') as map_stream:
        with trace.open('x', encoding='utf-8') as stream:
            def write(row):
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                stream.flush()

            sock, receiver, latest = None, None, None
            write({'type': 'metadata', **result})
            try:
                management = ManagementClient(ConnectionConfig(config.host, port=config.management_port,
                                                                timeout_s=2.))
                server, static = management.query('get_config'), management.query('get_static_info')
                if (server.get('sampling_freq') != 100 or server.get('ego_sampling_freq') != 25
                        or server.get('vjoy_executed_by_server') is not True):
                    raise ValueError('requires verified 100/25 Hz server-side vJoy configuration')
                for key, source in (('car_name', 'CarName'), ('track_name', 'TrackName'),
                                    ('track_configuration', 'TrackConfiguration')):
                    if not isinstance(static.get(source), str) or (source != 'TrackConfiguration' and not static[source]):
                        raise ValueError(f'missing identity: {source}')
                    result[key] = static[source]
                for name, data in (('server_config', server), ('static_info', static)):
                    encoded = json.dumps(data, sort_keys=True, allow_nan=False).encode()
                    result[f'{name}_sha256'] = hashlib.sha256(encoded).hexdigest()
                    write({'type': name, 'data': data})
                deadline = time.monotonic() + config.handshake_s
                sock = _open_socket(CaptureConfig(config.host, port=config.port), deadline)
                sock.send(b'connect')
                sock.settimeout(max(.001, deadline - time.monotonic()))
                reply = sock.recv(65508)
                if time.monotonic() > deadline or reply != b'identified':
                    raise ValueError('missing or late identified handshake')
                overall = time.monotonic() + config.overall_s
                receiver = Receiver(sock, config, write)
                baseline_deadline = min(overall, time.monotonic() + config.baseline_local_s)
                baseline_samples = []
                while True:
                    latest = receiver.next(baseline_deadline)
                    baseline_samples.append(latest)
                    if (len(baseline_samples) >= config.baseline_samples and
                            latest['timestamp_ac'] - baseline_samples[0]['timestamp_ac'] >= config.baseline_server_s):
                        break
                angles = [heading(p)[0] for p in baseline_samples]
                baseline = circular_mean(angles)
                if max(abs(wrap(a-baseline)) for a in angles) > config.settling_rad:
                    raise ValueError('neutral heading baseline unstable')

                def send(steer=0., brake=-1., phase='release'):
                    packet = receiver.latest
                    command = command_payload(packet['steps'], steer, brake)
                    # Log intent before transmission; a log write failure prevents the command.
                    write({'type': 'command', 'send_monotonic': time.monotonic(),
                           'server_timestamp': packet['timestamp_ac'], 'phase': phase, 'command': command})
                    payload = json.dumps(command, allow_nan=False).encode()
                    if sock.send(payload) != len(payload):
                        raise OSError('short control datagram write')

                send()
                send(brake=1., phase='brake_verify')
                start = latest['timestamp_ac']
                brake_deadline = min(overall, time.monotonic() + config.brake_local_s)
                while True:
                    latest = receiver.next(brake_deadline)
                    if latest['timestamp_ac'] - start > config.brake_server_s:
                        raise ValueError('brake did not verify within server deadline')
                    if latest['brakeStatus'] >= .9 and latest['accStatus'] <= .01:
                        break
                phases = []
                for index, command in enumerate(config.phases):
                    send(command, 1., f'steer_{index}')
                    start = latest['timestamp_ac']
                    samples = []
                    while True:
                        latest = receiver.next(overall)
                        if latest['brakeStatus'] < .9:
                            raise ValueError('brake verification lost')
                        elapsed = latest['timestamp_ac'] - start
                        samples.append((elapsed, heading(latest)[0]))
                        if elapsed >= config.phase_server_s:
                            break
                    tail = [a for t, a in samples if t >= elapsed-config.tail_server_s]
                    mean = circular_mean(tail)
                    if len(tail) < config.settled_samples or any(abs(wrap(a-mean)) > config.settling_rad for a in tail):
                        raise ValueError('insufficient settled steering tail')
                    phase = {'command': command, 'angle_rad': mean, 'tail_samples': len(tail)}
                    phases.append(phase)
                    write({'type': 'phase', 'index': index, **phase})
                result.update(measured_map(phases, baseline, config))
                result['phases'] = phases
            except (OSError, ValueError, KeyError, RecursionError, MemoryError, KeyboardInterrupt) as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
                result['validated'] = False
            finally:
                if sock is not None:
                    # Same socket and freshest accepted counter, even after a failed log/write.
                    if receiver is not None and receiver.latest is not None:
                        release = command_payload(receiver.latest['steps'])
                        try:
                            write({'type': 'command', 'send_monotonic': time.monotonic(),
                                   'server_timestamp': receiver.latest['timestamp_ac'],
                                   'phase': 'final_release', 'command': release})
                        except OSError:
                            pass  # Log failure must not prevent the best-effort release.
                        try:
                            payload = json.dumps(release).encode()
                            if sock.send(payload) != len(payload):
                                raise OSError('short release write')
                            result['release_sent'] = True
                        except OSError as exc:
                            result['release_error'] = str(exc)
                            result['validated'] = False
                    try:
                        sock.send(b'disconnect')
                    except OSError as exc:
                        result['disconnect_error'] = str(exc)
                        result['validated'] = False
                    sock.close()
                result['completed_at'] = _utc()
                write({'type': 'stop', **result})
        result['raw_trace_sha256'] = _sha(trace)
        map_stream.write(json.dumps(result, allow_nan=False, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='current host; overrides AC_HOST')
    parser.add_argument('--output', type=Path, help='new map JSON; no overwrite')
    parser.add_argument('--trace', type=Path, help='new raw JSONL; no overwrite')
    parser.add_argument('--setup-id', default='default_user_confirmed')
    parser.add_argument('--full-range', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--neutral-confirmed', action='store_true', help='user sees N in AC')
    parser.add_argument('--at-controls', action='store_true', help='user ready to stop game: no server watchdog')
    parser.add_argument('--source-config', type=Path)
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--speed-mps', type=float, default=CalibrationConfig.speed_mps,
                        help='stationary speed limit (default 0.05 m/s); the 2 cm position guard is unchanged')
    parser.add_argument('--motion-m', type=float, default=CalibrationConfig.motion_m,
                        help='stationary displacement limit (default 0.02 m); the rear-heading guard is unchanged')
    args = parser.parse_args(argv)
    if not _numeric(args.speed_mps) or not 0 < args.speed_mps <= .2:
        parser.error('--speed-mps must be in (0, 0.2] m/s')
    if not _numeric(args.motion_m) or not 0 < args.motion_m <= .2:
        parser.error('--motion-m must be in (0, 0.2] m')
    if not args.execute:
        phases = (0., .1, .25, .5, 1., 0., -.1, -.25, -.5, -1., 0.) if args.full_range else (0., .1, 0., -.1, 0.)
        print(json.dumps({'offline_plan': True, 'phases': phases, 'phase_server_s': .8,
                          'full_range': args.full_range, 'requires': ['explicit host/output/trace',
                          '--execute', '--neutral-confirmed', '--at-controls'],
                          'note': 'Small probe cannot produce a validated full map. No sockets opened.'}))
        return 0
    if not args.output or not args.trace or not args.neutral_confirmed or not args.at_controls:
        parser.error('execute requires --output, --trace, --neutral-confirmed and --at-controls')
    try:
        config = CalibrationConfig(args.host or os.environ.get('AC_HOST'), args.full_range, args.setup_id,
                                   speed_mps=args.speed_mps, motion_m=args.motion_m)
        sources = {k: v for k, v in (('config', args.source_config), ('checkpoint', args.source_checkpoint)) if v}
        result = execute(config, args.output, args.trace, neutral_confirmed=True, at_controls=True, source_files=sources)
    except (OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 1
    print(json.dumps({'output': str(args.output), 'trace': str(args.trace),
                      'validated': result['validated'], 'error': result.get('error')}))
    return int(bool(result.get('error') or result.get('release_error') or result.get('disconnect_error')))


if __name__ == '__main__':
    raise SystemExit(main())
