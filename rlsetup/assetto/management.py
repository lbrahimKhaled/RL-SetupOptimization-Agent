"""Bounded, read-only access to AC Gym's existing TCP management server.

Protocol source: dasGringuen/assetto_corsa_gym commit
ef0fa502bdd257fc2affad9361c02b3dbdfe1632, sensors_par.py management handler.
Requests have no delimiter. Replies are complete dictionaries without framing or EOF;
get_config uses Python literal syntax, while static/track replies use JSON.
"""
from __future__ import annotations

import ast
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import queue
import socket
import threading
import time


PROTOCOL_SOURCE_COMMIT = 'ef0fa502bdd257fc2affad9361c02b3dbdfe1632'
COMMANDS = ('get_config', 'get_static_info', 'get_track_info')
CONFIG_FIELDS = ('sampling_freq', 'ego_sampling_freq', 'vjoy_executed_by_server',
                 'ego_server_host_name', 'ego_server_port',
                 'simulation_management_server_host_name', 'simulation_management_server_port',
                 'screen_capture_enable', 'enable_alternative_python_interpreter')
STATIC_FIELDS = ('CarName', 'TrackName', 'TrackConfiguration', 'TrackLength',
                 'autoShifterOn', 'penaltiesEnabled', 'WindSpeed', 'WindDirection',
                 'CAR_WHEEL_R', 'CAR_TRACK', 'CAR_WHEELBASE', 'CAR_MASS',
                 'CAR_DIM_X', 'CAR_DIM_Y', 'CAR_DIM_Z',
                 'TyreContactPoint_FL', 'TyreContactPoint_FR',
                 'TyreContactPoint_RL', 'TyreContactPoint_RR', 'WorldPosition')
MAX_LIMIT_BYTES = 64 * 1024 * 1024
MAX_AST_NODES = 50_000
MAX_DATA_NODES = 1_000_000
MAX_DEPTH = 64
RECV_BYTES = 65_536
_INCOMPLETE = object()


class ManagementError(RuntimeError):
    """An inspection failed; no partial dictionary is accepted."""


@dataclass(frozen=True)
class ConnectionConfig:
    """Transport settings only; byte caps and seconds are diagnostic bounds."""
    host: str
    port: int = 2347
    timeout_s: float = 5.
    small_response_bytes: int = 1024 * 1024
    track_response_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError('provide --host HOST or set AC_HOST')
        if self.host != self.host.strip() or any(c.isspace() for c in self.host):
            raise ValueError('host must be a hostname or IP without whitespace')
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError('port must be an integer in 1..65535')
        if (type(self.timeout_s) not in (float, int)
                or not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 300):
            raise ValueError('timeout must be finite and in (0, 300] seconds')
        for value in (self.small_response_bytes, self.track_response_bytes):
            if type(value) is not int or not 1 <= value <= MAX_LIMIT_BYTES:
                raise ValueError('response byte caps must be integers in 1..64 MiB')


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('inspection deadline expired')
    return remaining


def _resolve(host, port, deadline):
    # getaddrinfo has no Python timeout. A daemon resolver bounds the caller's wait;
    # it owns no sockets and may finish in the background after a DNS timeout.
    result = queue.Queue(maxsize=1)

    def resolve():
        try:
            result.put((socket.getaddrinfo(host, port, type=socket.SOCK_STREAM), None))
        except OSError as exc:
            result.put((None, exc))

    threading.Thread(target=resolve, daemon=True).start()
    try:
        addresses, error = result.get(timeout=_remaining(deadline))
    except queue.Empty as exc:
        raise TimeoutError('hostname resolution exceeded inspection deadline') from exc
    if error is not None:
        raise error
    return addresses


def _connect(config, deadline):
    error = None
    for family, kind, proto, _name, address in _resolve(config.host, config.port, deadline):
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(_remaining(deadline))
            sock.connect(address)
            return sock
        except (OSError, TimeoutError) as exc:
            sock.close()
            error = exc
    if error is not None:
        raise error
    raise OSError('hostname resolution returned no stream addresses')


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite JSON number')
    return result


def _bound_ast(tree):
    pending, count = [(tree, 0)], 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_DEPTH:
            raise ManagementError('configuration literal exceeds structural limits')
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _validate_dictionary(data):
    if not isinstance(data, dict):
        raise ManagementError('response must be a top-level dictionary')
    pending, count = [(data, 0)], 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > MAX_DATA_NODES or depth > MAX_DEPTH:
            raise ManagementError('response exceeds structural limits')
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ManagementError('response dictionary keys must be strings')
            pending.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend((v, depth + 1) for v in value)
        elif type(value) is float:
            if not math.isfinite(value):
                raise ManagementError('response contains a nonfinite number')
        elif value is not None and type(value) not in (str, bool, int):
            raise ManagementError('response contains unsupported literal data')
    return data


def _parse(data, command):
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        if exc.reason == 'unexpected end of data':
            return _INCOMPLETE
        raise ManagementError('response is not valid UTF-8') from exc
    try:
        if command == 'get_config':
            try:
                tree = ast.parse(text, mode='eval')
            except SyntaxError:
                return _INCOMPLETE
            _bound_ast(tree)
            result = ast.literal_eval(tree)
        else:
            try:
                result = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
            except json.JSONDecodeError:
                return _INCOMPLETE
        return _validate_dictionary(result)
    except (ValueError, TypeError, RecursionError, MemoryError) as exc:
        raise ManagementError('invalid response literal or structural limit exceeded') from exc


class ManagementClient:
    """Query only the three inspection commands; every query resolves/connects anew."""

    def __init__(self, config: ConnectionConfig):
        self.config = config

    def query(self, command):
        if command not in COMMANDS:
            raise ValueError('only read-only management inspection commands are allowed')
        limit = (self.config.track_response_bytes if command == 'get_track_info'
                 else self.config.small_response_bytes)
        deadline = time.monotonic() + self.config.timeout_s
        payload = bytearray()
        try:
            with closing(_connect(self.config, deadline)) as sock:
                sock.settimeout(_remaining(deadline))
                sock.sendall(command.encode('utf-8'))
                while True:
                    sock.settimeout(_remaining(deadline))
                    chunk = sock.recv(min(RECV_BYTES, limit - len(payload) + 1))
                    if not chunk:
                        raise ManagementError(
                            f'server closed before a complete dictionary ({len(payload)} bytes); '
                            'reply may be incomplete or malformed')
                    payload.extend(chunk)
                    if len(payload) > limit:
                        raise ManagementError(f'response exceeds {limit}-byte inspection cap')
                    parsed = _parse(payload, command)
                    _remaining(deadline)
                    if parsed is not _INCOMPLETE:
                        return parsed
        except TimeoutError as exc:
            raise ManagementError(
                f'inspection timed out after {self.config.timeout_s:g} s '
                f'({len(payload)} response bytes); reply may be incomplete or malformed') from exc
        except OSError as exc:
            raise ManagementError(f'connection to {self.config.host}:{self.config.port} failed: {exc}') from exc


def inspect_server(config: ConnectionConfig, include_track=False):
    """Return JSON-compatible inspection results, including failures per query."""
    report = {'schema_version': 1, 'inspection_only': True,
              'host': config.host, 'port': config.port,
              'protocol_source_commit': PROTOCOL_SOURCE_COMMIT,
              'installed_source_revision': None,
              'started_at': datetime.now(timezone.utc).isoformat(), 'queries': {}, 'errors': []}
    client = ManagementClient(config)
    commands = COMMANDS if include_track else COMMANDS[:2]
    for command in commands:
        started = time.monotonic()
        result = {'ok': False, 'data': None}
        try:
            data = client.query(command)
            fields = CONFIG_FIELDS if command == 'get_config' else STATIC_FIELDS
            result['data'] = data if command == 'get_track_info' else {k: data[k] for k in fields if k in data}
            result['ok'] = True
        except ManagementError as exc:
            result['error'] = str(exc)
            report['errors'].append(f'{command}: {exc}')
        result['elapsed_s'] = time.monotonic() - started
        report['queries'][command] = result
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    return report
