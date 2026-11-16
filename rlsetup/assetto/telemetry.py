"""Bounded AC Gym telemetry subscription; sends only connect/disconnect, never controls.

Subscription replaces the server's current client and resets its connection counter.
Only run while no other AC Gym client is connected. Clock origins on the two computers
differ: this tool measures intervals/jitter, not one-way or action latency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import socket
import threading
import time

from .management import PROTOCOL_SOURCE_COMMIT


REQUIRED_FIELDS = ('steps', 'packetId', 'timestamp_ac', 'world_position_x',
                   'world_position_y', 'speed', 'local_velocity_x', 'local_velocity_y',
                   'yaw', 'angular_velocity_y', 'steerAngle',
                   'tyre_slip_ratio_rl', 'tyre_slip_ratio_rr')


@dataclass(frozen=True)
class CaptureConfig:
    """Diagnostic transport bounds; seconds, packets and bytes, not RL settings."""
    host: str
    port: int = 2345
    duration_s: float = 3.
    handshake_timeout_s: float = 3.
    handshake_attempts: int = 2
    max_packets: int = 10000
    max_datagram_bytes: int = 65507

    def __post_init__(self):
        if not isinstance(self.host, str) or not self.host or any(c.isspace() for c in self.host):
            raise ValueError('provide --host HOST or set AC_HOST; whitespace is not allowed')
        for name, value, upper in (('port', self.port, 65535),
                                   ('handshake attempts', self.handshake_attempts, 5),
                                   ('max packets', self.max_packets, 100000),
                                   ('datagram bytes', self.max_datagram_bytes, 262144)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{name} must be an integer in 1..{upper}')
        for name, value in (('duration', self.duration_s), ('handshake timeout', self.handshake_timeout_s)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 300:
                raise ValueError(f'{name} must be finite and in (0, 300] seconds')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError('capture deadline expired')
    return left


def _open_socket(config, deadline):
    # Bound the caller's wait for synchronous DNS; resolver owns no sockets.
    result = queue.Queue(maxsize=1)

    def resolve():
        try:
            result.put((socket.getaddrinfo(config.host, config.port, socket.AF_INET,
                                           socket.SOCK_DGRAM), None))
        except OSError as exc:
            result.put((None, exc))

    threading.Thread(target=resolve, daemon=True).start()
    try:
        addresses, error = result.get(timeout=_remaining(deadline))
    except queue.Empty as exc:
        raise TimeoutError('hostname resolution exceeded handshake deadline') from exc
    if error is not None:
        raise error
    if not addresses:
        raise OSError('hostname has no IPv4 UDP address')
    family, kind, protocol, _name, address = addresses[0]
    sock = socket.socket(family, kind, protocol)
    try:
        sock.settimeout(_remaining(deadline))
        # Connected UDP constrains received datagrams to this resolved IP and port.
        sock.connect(address)
        return sock
    except BaseException:
        sock.close()
        raise


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('nonfinite JSON number')
    return value


def _reject_constant(_text):
    raise ValueError('nonfinite JSON constant')


def _packet(payload):
    data = json.loads(payload, parse_float=_finite_float, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError('telemetry must be a JSON object')
    pending = [(data, 0)]
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if depth > 64 or count > 100000:
            raise ValueError('telemetry structural bound exceeded')
        if isinstance(value, dict):
            pending.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            pending.extend((v, depth + 1) for v in value)
    return data


def _numeric(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _intervals(values):
    intervals = [(b - a) * 1000 for a, b in zip(values, values[1:]) if b >= a]
    intervals = sorted(value for value in intervals if math.isfinite(value))
    if not intervals:
        return None

    def percentile(p):
        index = (len(intervals) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(intervals) - 1)
        return intervals[lo] + (intervals[hi] - intervals[lo]) * (index - lo)

    return {'count': len(intervals), 'p50_ms': percentile(.5), 'p95_ms': percentile(.95),
            'p99_ms': percentile(.99), 'max_ms': intervals[-1]}


def _hash_sources(source_files):
    result = {}
    for name, raw in (source_files or {}).items():
        path = Path(raw).expanduser().resolve()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result[name] = {'path': str(path), 'sha256': digest.hexdigest()}
    return result


def capture(config: CaptureConfig, output, source_files=None):
    """Write metadata, raw valid samples and summary to a new explicit JSONL file."""
    sources = _hash_sources(source_files)
    output = Path(output).expanduser()
    summary = {'type': 'summary', 'packets_received': 0, 'valid_samples': 0,
               'malformed_packets': 0, 'oversized_packets': 0, 'protocol_messages': 0,
               'missing_fields': {}, 'invalid_numeric_fields': {}, 'duplicate_sequences': 0,
               'out_of_order': 0, 'counter_regressions': 0, 'possible_counter_resets': 0,
               'forward_sequence_gaps': 0, 'skipped_sequence_values': 0,
               'receipt_intervals': None, 'server_intervals': None, 'errors': [],
               'note': 'Intervals/jitter only, not one-way/action latency. Sequence gaps are not proven packet loss.'}
    receipts, timestamps = [], []
    previous = None
    sock = None
    with output.open('x', encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')

        write({'type': 'metadata', 'schema_version': 1, 'started_at': _utc(),
               'host': config.host, 'port': config.port, 'duration_s': config.duration_s,
               'max_packets': config.max_packets, 'max_datagram_bytes': config.max_datagram_bytes,
               'protocol_source_commit': PROTOCOL_SOURCE_COMMIT, 'installed_source_revision': None,
               'source_files': sources, 'required_fields': list(REQUIRED_FIELDS),
               'outbound_commands': ['connect', 'disconnect'],
               'side_effects': 'Subscription replaces current client and resets server connection counter.',
               'clocks': 'Local receive_monotonic and server timestamp_ac have different origins.'})
        try:
            deadline = time.monotonic() + config.handshake_timeout_s
            sock = _open_socket(config, deadline)
            identified = False
            for attempt in range(config.handshake_attempts):
                _remaining(deadline)
                sock.send(b'connect')
                # Attempts share the original deadline; each gets a bounded slice.
                sock.settimeout(_remaining(deadline) / (config.handshake_attempts - attempt))
                try:
                    reply = sock.recv(config.max_datagram_bytes + 1)
                except socket.timeout:
                    continue
                _remaining(deadline)
                if reply == b'identified':
                    identified = True
                    break
                raise ValueError('unexpected handshake response (expected identified)')
            if not identified:
                raise TimeoutError('no identified reply before handshake deadline')
            deadline = time.monotonic() + config.duration_s
            while summary['packets_received'] < config.max_packets:
                try:
                    sock.settimeout(_remaining(deadline))
                    payload = sock.recv(config.max_datagram_bytes + 1)
                except TimeoutError:
                    break
                received = time.monotonic()
                if received > deadline:
                    break
                summary['packets_received'] += 1
                if payload in (b'identified', b'disconnect'):
                    summary['protocol_messages'] += 1
                    if payload == b'disconnect':
                        raise ConnectionError('server disconnected the subscription')
                    continue
                if len(payload) > config.max_datagram_bytes:
                    summary['oversized_packets'] += 1
                    continue
                try:
                    data = _packet(payload)
                except (ValueError, UnicodeError, RecursionError, MemoryError):
                    summary['malformed_packets'] += 1
                    continue
                missing = [key for key in REQUIRED_FIELDS if key not in data]
                invalid = [key for key in REQUIRED_FIELDS if key in data and not _numeric(data[key])]
                for key in missing:
                    summary['missing_fields'][key] = summary['missing_fields'].get(key, 0) + 1
                for key in invalid:
                    summary['invalid_numeric_fields'][key] = summary['invalid_numeric_fields'].get(key, 0) + 1
                receipts.append(received)
                stamp, seq = data.get('timestamp_ac'), data.get('steps')
                if _numeric(stamp):
                    timestamps.append(stamp)
                if type(seq) is int and seq >= 0 and _numeric(stamp):
                    if previous is not None:
                        old_seq, old_stamp = previous
                        if seq == old_seq:
                            summary['duplicate_sequences'] += 1
                        elif stamp < old_stamp:
                            summary['out_of_order'] += 1
                        elif seq < old_seq:
                            summary['counter_regressions'] += 1
                            summary['possible_counter_resets'] += 1
                        elif seq > old_seq + 1:
                            summary['forward_sequence_gaps'] += 1
                            summary['skipped_sequence_values'] += seq - old_seq - 1
                    # An older datagram must not move the comparison anchor backwards.
                    if previous is None or stamp >= previous[1]:
                        previous = seq, stamp
                summary['valid_samples'] += 1
                write({'type': 'sample', 'receive_monotonic': received, 'telemetry': data,
                       'missing_fields': missing, 'invalid_numeric_fields': invalid})
        except (OSError, ValueError) as exc:
            summary['errors'].append(f'{type(exc).__name__}: {exc}')
        except KeyboardInterrupt:
            summary['errors'].append('capture interrupted by user')
        finally:
            if sock is not None:
                try:
                    sock.send(b'disconnect')
                except OSError as exc:
                    summary['errors'].append(f'disconnect failed: {exc}')
                sock.close()
            summary['receipt_intervals'] = _intervals(receipts)
            summary['server_intervals'] = _intervals(timestamps)
            if not summary['valid_samples']:
                summary['errors'].append('no valid telemetry samples received')
            summary['completed_at'] = _utc()
            write(summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='hostname or LAN IP; overrides AC_HOST')
    parser.add_argument('--port', type=int, default=2345)
    parser.add_argument('--output', type=Path, required=True, help='new JSONL file; overwrite refused')
    parser.add_argument('--duration', type=float, default=3., help='capture seconds after handshake')
    parser.add_argument('--timeout', type=float, default=3., help='total handshake deadline seconds')
    parser.add_argument('--max-packets', type=int, default=10000)
    parser.add_argument('--source-config', type=Path, help='optional config file to hash, never execute')
    parser.add_argument('--source-checkpoint', type=Path, help='optional checkpoint to hash, never load')
    args = parser.parse_args(argv)
    try:
        config = CaptureConfig(host=args.host if args.host is not None else os.environ.get('AC_HOST'),
                               port=args.port, duration_s=args.duration,
                               handshake_timeout_s=args.timeout, max_packets=args.max_packets)
    except ValueError as exc:
        parser.error(str(exc))
    sources = {key: value for key, value in (('config', args.source_config),
               ('checkpoint', args.source_checkpoint)) if value is not None}
    try:
        summary = capture(config, args.output, sources)
    except OSError as exc:
        print(json.dumps({'output': str(args.output), 'error': str(exc)}))
        return 1
    print(json.dumps({'output': str(args.output), **summary}, allow_nan=False))
    return int(bool(summary['errors'] or summary['malformed_packets'] or summary['oversized_packets']
                    or summary['missing_fields'] or summary['invalid_numeric_fields']))


if __name__ == '__main__':
    raise SystemExit(main())
