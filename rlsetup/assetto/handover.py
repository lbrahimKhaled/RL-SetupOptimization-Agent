"""Bounded stationary neutral-to-first handover; offline unless --execute.

AC actualGear 1 means neutral, 2 means first. Commands use released gas=-1,
centered steering=0, applied brake=+1. UDP cleanup is attempted, not guaranteed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import socket
import time

from .calibration import command_payload, number
from .management import ConnectionConfig, ManagementClient, ManagementError, PROTOCOL_SOURCE_COMMIT
from .telemetry import CaptureConfig, _numeric, _open_socket, _packet, _utc


@dataclass(frozen=True)
class HandoverConfig:
    """Approved technical guards; times seconds, speed metres/second."""
    host: str
    port: int = 2345
    management_port: int = 2347
    management_s: float = 2.
    handshake_s: float = 1.
    overall_s: float = 4.
    freshness_s: float = .2
    baseline_samples: int = 3
    brake_s: float = .8
    pulse_s: float = .1
    confirmation_s: float = 1.
    speed_mps: float = .05
    max_packets: int = 1000
    max_drain: int = 32

    def __post_init__(self):
        CaptureConfig(self.host, port=self.port)
        ConnectionConfig(self.host, port=self.management_port)
        for key, value in asdict(self).items():
            if key == 'host':
                continue
            if not _numeric(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('baseline_samples', 'max_packets', 'max_drain'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'{key} must be an integer')
        if (self.overall_s > 4 or self.pulse_s > .1 or self.freshness_s > .2
                or self.confirmation_s > 1 or self.speed_mps > .05 or self.max_drain > 32):
            raise ValueError('handover bounds may only be tightened')


class Guard:
    """Require finite stationary, gas-free samples and advancing counters."""
    def __init__(self, config):
        self.config, self.previous = config, None

    def check(self, packet, *, first_allowed=False):
        steps = packet.get('steps')
        stamp = number(packet, 'timestamp_ac')
        if type(steps) is not int or steps < 0 or stamp < 0:
            raise ValueError('invalid telemetry counter/timestamp')
        speed, gear = number(packet, 'speed'), number(packet, 'actualGear')
        gas, brake = number(packet, 'accStatus'), number(packet, 'brakeStatus')
        if not 0 <= speed < self.config.speed_mps or not 0 <= gas <= .01:
            raise ValueError('requires stationary car and released throttle')
        if gear not in ((1, 2) if first_allowed else (1,)):
            raise ValueError('requires neutral (1), or already first (2); no reverse/other gear')
        if not 0 <= brake <= 1.01:
            raise ValueError('invalid brake status')
        if self.previous is not None:
            old_steps, old_stamp = self.previous
            if steps < old_steps or stamp < old_stamp:
                raise ValueError('telemetry regression')
            if stamp-old_stamp > self.config.freshness_s:
                raise ValueError('server telemetry gap exceeded freshness bound')
            if steps == old_steps or stamp == old_stamp:
                return False
        self.previous = steps, stamp
        return True


def execute(config: HandoverConfig, output):
    """Write a new JSONL trace; emit at most one shift-up edge, never throttle."""
    result = {'success': False, 'pulse_count': 0, 'release_attempted': False,
              'release_sent': False, 'final_gear': None, 'pulse_duration_s': None,
              'started_at': _utc(), 'config': asdict(config),
              'protocol_source_commit': PROTOCOL_SOURCE_COMMIT,
              'cleanup_note': 'UDP release transmission is best effort, not acknowledgement.'}
    sock, latest = None, None
    with Path(output).expanduser().open('x', encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
            stream.flush()

        write({'type': 'metadata', **result})
        try:
            client = ManagementClient(ConnectionConfig(config.host, port=config.management_port,
                                                        timeout_s=config.management_s))
            server, static = client.query('get_config'), client.query('get_static_info')
            write({'type': 'management', 'server': server, 'static': static})
            if (server.get('sampling_freq') != 100 or server.get('ego_sampling_freq') != 25
                    or server.get('vjoy_executed_by_server') is not True):
                raise ValueError('requires verified 100/25 Hz server-side vJoy')
            if (static.get('CarName'), static.get('TrackName'), static.get('TrackConfiguration')) != (
                    'ks_ferrari_sf70h', 'ks_barcelona', 'layout_gp'):
                raise ValueError('requires ks_ferrari_sf70h / ks_barcelona / layout_gp')
            overall = time.monotonic() + config.overall_s
            handshake = min(overall, time.monotonic() + config.handshake_s)
            sock = _open_socket(CaptureConfig(config.host, port=config.port), handshake)
            sock.send(b'connect')
            sock.settimeout(max(.001, handshake-time.monotonic()))
            if sock.recv(65508) != b'identified' or time.monotonic() > handshake:
                raise ValueError('missing or late identified handshake')
            guard, last_fresh, count = Guard(config), time.monotonic(), 0
            clear_at = None

            def send(phase, *, pulse=False, release=False):
                steps = latest['steps'] if latest is not None else 0
                command = command_payload(steps, brake=-1. if release else 1.)
                if pulse:
                    command.update(enable_gear_shift=True, shift_up=True)
                write({'type': 'command', 'phase': phase,
                       'intent_monotonic': time.monotonic(), 'command': command})
                payload = json.dumps(command, allow_nan=False).encode()
                sock.settimeout(max(.001, min(.1, overall-time.monotonic())))
                sent = time.monotonic()
                if pulse:
                    result['pulse_send_monotonic'] = sent
                if sock.send(payload) != len(payload):
                    raise OSError('short command datagram')
                if phase == 'clear_shift':
                    result['pulse_duration_s'] = sent-result['pulse_send_monotonic']
                return sent

            def receive(deadline, *, first_allowed=False, brake_required=False):
                nonlocal latest, last_fresh, count, clear_at
                while True:
                    now = time.monotonic()
                    if clear_at is not None and now >= clear_at:
                        send('clear_shift')
                        clear_at = None
                    end = min(overall, deadline, last_fresh+config.freshness_s)
                    if now >= end:
                        raise TimeoutError('no fresh stationary telemetry within deadline')
                    wait_end = min(end, clear_at) if clear_at is not None else end
                    sock.settimeout(wait_end-now)
                    try:
                        payload = sock.recv(65508)
                    except socket.timeout:
                        if clear_at is not None and time.monotonic() >= clear_at:
                            send('clear_shift')
                            clear_at = None
                            continue
                        raise TimeoutError('no fresh stationary telemetry within deadline')
                    received = time.monotonic()
                    # Clear on the next datagram even when malformed/duplicate/late.
                    if clear_at is not None:
                        send('clear_shift')
                        clear_at = None
                    pending = [(received, payload)]
                    # Drain queued telemetry before any sample can authorize actuation.
                    # Exhausting the bound aborts rather than trusting a stale queue.
                    for _ in range(config.max_drain):
                        if time.monotonic() > end:
                            raise TimeoutError('late telemetry during queue drain')
                        sock.settimeout(0.)
                        try:
                            queued = sock.recv(65508)
                        except (BlockingIOError, socket.timeout):
                            break
                        pending.append((time.monotonic(), queued))
                    else:
                        raise ValueError('telemetry queue drain bound exceeded')
                    accepted = None
                    for received, payload in pending:
                        if received > end:
                            raise TimeoutError('late telemetry')
                        count += 1
                        if count > config.max_packets or len(payload) > 65507:
                            raise ValueError('telemetry packet bound exceeded')
                        packet = _packet(payload)
                        write({'type': 'sample', 'receive_monotonic': received, 'telemetry': packet})
                        if not guard.check(packet, first_allowed=first_allowed):
                            continue
                        latest, last_fresh, accepted = packet, received, packet
                        result['final_gear'] = packet['actualGear']
                        if brake_required and packet['brakeStatus'] < .9:
                            raise ValueError('brake verification lost')
                    if accepted is not None:
                        return accepted

            baseline_gear = None
            for _ in range(config.baseline_samples):
                packet = receive(overall, first_allowed=True)
                if baseline_gear is not None and packet['actualGear'] != baseline_gear:
                    raise ValueError('gear changed during baseline')
                baseline_gear = packet['actualGear']
            if baseline_gear == 2:
                result.update(success=True, already_first=True)
            else:
                send('apply_brake')
                brake_deadline = min(overall, time.monotonic()+config.brake_s)
                while receive(brake_deadline)['brakeStatus'] < .9:
                    pass
                # Exactly one attempt; uncertain send outcomes are never retried.
                result['pulse_count'] = 1
                pulse_start = send('shift_up', pulse=True)
                clear_at = pulse_start+config.pulse_s
                confirmation = min(overall, pulse_start+config.confirmation_s)
                while receive(confirmation, first_allowed=True, brake_required=True)['actualGear'] != 2:
                    pass
                result['success'] = True
        except (OSError, ManagementError, ValueError, KeyError, RecursionError, MemoryError, KeyboardInterrupt) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            result['success'] = False
        finally:
            if sock is not None:
                result['release_attempted'] = True
                release = command_payload(latest['steps'] if latest is not None else 0)
                try:
                    write({'type': 'command', 'phase': 'final_release',
                           'send_monotonic': time.monotonic(), 'command': release})
                except OSError:
                    pass
                try:
                    sock.settimeout(.1)
                    payload = json.dumps(release).encode()
                    release_start = time.monotonic()
                    if sock.send(payload) != len(payload):
                        raise OSError('short release datagram')
                    result['release_sent'] = True
                    if result.get('pulse_send_monotonic') is not None and result['pulse_duration_s'] is None:
                        result['pulse_duration_s'] = release_start-result['pulse_send_monotonic']
                except (OSError, KeyboardInterrupt) as exc:
                    result['release_error'] = str(exc)
                    result['success'] = False
                try:
                    sock.send(b'disconnect')
                except (OSError, KeyboardInterrupt) as exc:
                    result['disconnect_error'] = str(exc)
                    result['success'] = False
                sock.close()
            result['completed_at'] = _utc()
            write({'type': 'summary', **result})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='current host; overrides AC_HOST')
    parser.add_argument('--output', type=Path, help='new JSONL trace; no overwrite')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'offline_plan': True, 'max_shift_up_pulses': 1,
                          'pulse_max_s': .1, 'udp_max_s': 4.,
                          'requires': ['--execute', '--output', '--host or AC_HOST'],
                          'note': 'No sockets opened. Stationary neutral to first only.'}))
        return 0
    if args.output is None:
        parser.error('--execute requires --output')
    try:
        config = HandoverConfig(args.host if args.host is not None else os.environ.get('AC_HOST'))
        result = execute(config, args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 1
    print(json.dumps(result, allow_nan=False))
    return int(not result['success'])


if __name__ == '__main__':
    raise SystemExit(main())
