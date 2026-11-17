"""AC1 protocol tests: controlled streams, no game or live network required."""
import ast
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from rlsetup.assetto import management as M
from rlsetup.assetto.__main__ import main


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.sent = []
        self.timeouts = []
        self.closed = False

    def connect(self, address):
        self.address = address

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def sendall(self, value):
        self.sent.append(value)

    def recv(self, size):
        assert self.chunks, 'client incorrectly waited for EOF after the full dictionary'
        result = self.chunks.pop(0)
        if isinstance(result, BaseException):
            raise result
        if len(result) > size:
            self.chunks.insert(0, result[size:])
        return result[:size]

    def close(self):
        self.closed = True


def streams(monkeypatch, *chunks):
    sockets = [FakeSocket(c) for c in chunks]
    pending = iter(sockets)
    monkeypatch.setattr(M.socket, 'socket', lambda *_a: next(pending))
    monkeypatch.setattr(M.socket, 'getaddrinfo',
                        lambda host, port, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, '', (host, port))])
    return sockets


def test_fragmented_json_and_utf8_returns_without_eof(monkeypatch):
    saved = streams(monkeypatch, [b'{"CarName":"caf\xc3', b'\xa9","TrackName":', b'"gp"}'])
    result = M.ManagementClient(M.ConnectionConfig('lenovo.local')).query('get_static_info')
    assert result == {'CarName': 'café', 'TrackName': 'gp'}
    assert saved[0].sent == [b'get_static_info']  # Exact request, no newline or prefix.
    assert saved[0].address == ('lenovo.local', 2347)
    assert saved[0].closed


def test_config_python_literals_are_not_json_or_executed(monkeypatch):
    saved = streams(monkeypatch, [b"{'vjoy_executed_by_server': Tr", b"ue, 'optional': None}"])
    client = M.ManagementClient(M.ConnectionConfig('game-pc'))
    assert client.query('get_config') == {'vjoy_executed_by_server': True, 'optional': None}
    assert saved[0].closed


@pytest.mark.parametrize('command', ['reset', 'connect', 'arbitrary', 'get_config\n', ''])
def test_forbidden_commands_never_connect(monkeypatch, command):
    monkeypatch.setattr(M, '_connect', lambda *_a: pytest.fail('forbidden command attempted network I/O'))
    with pytest.raises(ValueError, match='read-only'):
        M.ManagementClient(M.ConnectionConfig('game-pc')).query(command)


@pytest.mark.parametrize('reply', [b'[]', b'42', b'null', b'{"x":NaN}', b'{"x":1e999}', b'\xff'])
def test_nonobject_nonfinite_and_invalid_utf8_fail(monkeypatch, reply):
    saved = streams(monkeypatch, [reply])
    with pytest.raises(M.ManagementError):
        M.ManagementClient(M.ConnectionConfig('game-pc')).query('get_static_info')
    assert saved[0].closed


@pytest.mark.parametrize('reply', [b'{"x":', b'{} trailing', b'{not-json}', b'\xc3'])
def test_eof_before_complete_valid_dictionary_is_explicit(monkeypatch, reply):
    saved = streams(monkeypatch, [reply, b''])
    with pytest.raises(M.ManagementError, match='incomplete or malformed'):
        M.ManagementClient(M.ConnectionConfig('game-pc')).query('get_static_info')
    assert saved[0].closed


def test_partial_response_timeout_closes_socket(monkeypatch):
    saved = streams(monkeypatch, [b'{"x":', socket.timeout('no more bytes')])
    with pytest.raises(M.ManagementError, match='timed out.*5 response bytes'):
        M.ManagementClient(M.ConnectionConfig('game-pc')).query('get_static_info')
    assert saved[0].closed


def test_cap_stops_receive_before_accepting_large_dictionary(monkeypatch):
    saved = streams(monkeypatch, [b'{"long":"too long"}'])
    with pytest.raises(M.ManagementError, match='8-byte'):
        M.ManagementClient(M.ConnectionConfig('game-pc', small_response_bytes=8)).query('get_config')
    assert saved[0].closed


def test_config_ast_is_bounded_before_literal_evaluation(monkeypatch):
    monkeypatch.setattr(M, 'MAX_AST_NODES', 4)
    monkeypatch.setattr(ast, 'literal_eval', lambda *_a: pytest.fail('oversized AST was evaluated'))
    with pytest.raises(M.ManagementError, match='structural'):
        M._parse(b"{'x':[1,2,3]}", 'get_config')


@pytest.mark.parametrize('reply', [b"{'x': __import__('os').system('anything')}",
                                  b"{'x':1e999}", b"{'x':{1,2}}"])
def test_config_refuses_code_nonfinite_and_non_json_literals(reply):
    with pytest.raises(M.ManagementError):
        M._parse(reply, 'get_config')


def test_json_structural_bound_and_recursion_error_are_explicit(monkeypatch):
    monkeypatch.setattr(M, 'MAX_DEPTH', 2)
    with pytest.raises(M.ManagementError, match='structural'):
        M._parse(b'{"a":[[[1]]]}', 'get_track_info')
    monkeypatch.setattr(M.json, 'loads', lambda *_a, **_k: (_ for _ in ()).throw(RecursionError()))
    with pytest.raises(M.ManagementError, match='structural'):
        M._parse(b'{}', 'get_track_info')


def test_report_resolves_each_query_whitelists_and_keeps_raw_track(monkeypatch):
    config = {'sampling_freq': 100, 'vjoy_executed_by_server': True, 'unrelated_private_field': 'omit'}
    static = {'CarName': 'ks_ferrari_sf70h', 'TrackName': 'ks_barcelona', 'driverName': 'omit'}
    track = {'fast_lane': [[1., 2.]], 'left_lane': [], 'right_lane': [], 'speed_arr': [3.]}
    streams(monkeypatch, [repr(config).encode()], [json.dumps(static).encode()], [json.dumps(track).encode()])
    resolved = []
    monkeypatch.setattr(M.socket, 'getaddrinfo', lambda host, port, **_k:
                        resolved.append(host) or [(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('resolved', port))])
    report = M.inspect_server(M.ConnectionConfig('changing-host.local'), include_track=True)
    assert resolved == ['changing-host.local'] * 3
    assert report['installed_source_revision'] is None
    assert report['protocol_source_commit'] == M.PROTOCOL_SOURCE_COMMIT
    assert report['errors'] == []
    assert report['queries']['get_config']['data'] == {'sampling_freq': 100, 'vjoy_executed_by_server': True}
    assert report['queries']['get_static_info']['data'] == {'CarName': 'ks_ferrari_sf70h', 'TrackName': 'ks_barcelona'}
    assert report['queries']['get_track_info']['data'] == track
    assert 'omit' not in json.dumps(report)
    assert report['started_at'] and report['completed_at']
    assert all(r['elapsed_s'] >= 0 for r in report['queries'].values())


def test_report_keeps_other_results_when_query_fails(monkeypatch):
    streams(monkeypatch, [b'', ], [b'{"CarName":"sf70h"}'])
    report = M.inspect_server(M.ConnectionConfig('game-pc'))
    assert not report['queries']['get_config']['ok']
    assert report['queries']['get_static_info']['ok']
    assert len(report['errors']) == 1
    assert 'get_track_info' not in report['queries']


@pytest.mark.parametrize('settings', [dict(host=''), dict(host=None), dict(host=' bad'),
    dict(host='game pc'), dict(host='pc', port=0), dict(host='pc', port=True),
    dict(host='pc', timeout_s=float('nan')), dict(host='pc', timeout_s=float('inf')),
    dict(host='pc', timeout_s=0), dict(host='pc', timeout_s=1e100),
    dict(host='pc', track_response_bytes=0), dict(host='pc', small_response_bytes=2**30)])
def test_invalid_transport_config(settings):
    with pytest.raises(ValueError):
        M.ConnectionConfig(**settings)


def test_cli_host_precedence_and_failure_status(monkeypatch, capsys):
    configs = []

    def inspect(config, include_track):
        configs.append((config, include_track))
        return {'errors': [] if config.host == 'explicit' else ['failed']}

    monkeypatch.setenv('AC_HOST', 'from-env')
    monkeypatch.setattr('rlsetup.assetto.__main__.inspect_server', inspect)
    assert main(['inspect', '--host', 'explicit', '--port', '1234', '--include-track']) == 0
    assert json.loads(capsys.readouterr().out) == {'errors': []}
    assert configs[0][0].host == 'explicit' and configs[0][0].port == 1234 and configs[0][1]
    assert main(['inspect']) == 1
    assert configs[1][0].host == 'from-env'
    assert json.loads(capsys.readouterr().out)['errors']


def test_missing_host_is_usage_error(monkeypatch, capsys):
    monkeypatch.delenv('AC_HOST', raising=False)
    with pytest.raises(SystemExit) as error:
        main(['inspect'])
    assert error.value.code == 2
    assert '--host HOST or set AC_HOST' in capsys.readouterr().err


def test_connection_and_receive_share_one_deadline(monkeypatch):
    saved = streams(monkeypatch, [b'{}'])
    clock = iter([0., 0.1, 0.2, 0.3, 2.])
    monkeypatch.setattr(M.time, 'monotonic', lambda: next(clock))
    # Bypass asynchronous resolver so the clock sequence is deterministic.
    monkeypatch.setattr(M, '_resolve', lambda *_a: [(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('pc', 2347))])
    with pytest.raises(M.ManagementError, match='timed out'):
        M.ManagementClient(M.ConnectionConfig('pc', timeout_s=1.)).query('get_config')
    assert saved[0].closed
    assert all(0 < t <= 1. for t in saved[0].timeouts)


def test_dns_timeout_does_not_wait_for_resolver(monkeypatch):
    release = threading.Event()

    def resolve(*_a, **_k):
        release.wait(2.)
        return []

    monkeypatch.setattr(M.socket, 'getaddrinfo', resolve)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match='resolution'):
            M._resolve('slow-dns', 2347, time.monotonic() + .03)
        assert time.monotonic() - started < .5
    finally:
        release.set()


def test_import_without_site_packages_loads_no_upstream_or_rl_modules():
    root = Path(__file__).resolve().parents[1]
    code = ('import sys; import rlsetup.assetto; '
            'assert not any(n.startswith(("torch", "ocd_lap", "ocd_native", '
            '"AssettoCorsaEnv", "AssettoCorsaPlugin")) for n in sys.modules)')
    completed = subprocess.run([sys.executable, '-S', '-c', code], cwd=root,
                               capture_output=True, text=True, check=True)
    assert completed.stdout == '' and completed.stderr == ''
