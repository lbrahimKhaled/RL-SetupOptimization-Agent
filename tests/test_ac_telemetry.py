"""AC2 subscription tests use socket fakes; no game or live network is contacted."""
import json
from pathlib import Path
import socket

import pytest

from rlsetup.assetto import telemetry as T


class FakeSocket:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.closed = False
        self.peer = None

    def settimeout(self, value):
        assert value > 0

    def connect(self, peer):
        self.peer = peer

    def send(self, data):
        self.sent.append(data)
        return len(data)

    def recv(self, limit):
        if not self.replies:
            raise socket.timeout()
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply[:limit]

    def close(self):
        self.closed = True


def packet(step=1, timestamp=1., **changes):
    row = {key: 0. for key in T.REQUIRED_FIELDS}
    row.update(steps=step, packetId=step, timestamp_ac=timestamp)
    row.update(changes)
    return json.dumps(row).encode()


def install(monkeypatch, replies):
    sock = FakeSocket(replies)
    monkeypatch.setattr(T, '_open_socket', lambda config, deadline: sock)
    return sock


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_capture_preserves_raw_data_and_sends_only_handshake_cleanup(tmp_path, monkeypatch):
    payload = packet(1, 1000., extra={'keep': [1, 2]})
    sock = install(monkeypatch, [b'identified', payload, packet(2, 1000.04)])
    output = tmp_path / 'capture.jsonl'
    summary = T.capture(T.CaptureConfig('lenovo.local'), output)
    saved = records(output)
    assert sock.sent == [b'connect', b'disconnect']
    assert sock.closed
    assert saved[0]['type'] == 'metadata'
    assert saved[0]['host'] == 'lenovo.local'
    assert saved[0]['installed_source_revision'] is None
    assert saved[1]['telemetry'] == json.loads(payload)
    assert saved[1]['missing_fields'] == []
    assert saved[-1] == summary
    assert summary['valid_samples'] == 2
    assert summary['errors'] == []
    assert summary['server_intervals']['p50_ms'] == pytest.approx(40.)
    assert 'not proven packet loss' in summary['note']


def test_does_not_overwrite_or_connect_when_output_exists(tmp_path, monkeypatch):
    path = tmp_path / 'capture.jsonl'
    path.write_text('original')
    monkeypatch.setattr(T, '_open_socket', lambda *_: pytest.fail('network started'))
    with pytest.raises(FileExistsError):
        T.capture(T.CaptureConfig('host'), path)
    assert path.read_text() == 'original'


def test_sequence_diagnostics_do_not_treat_regression_as_huge_gap(tmp_path, monkeypatch):
    seq = [(1, 1.), (3, 1.08), (3, 1.08), (2, 1.04), (1, 1.12), (2, 1.16)]
    install(monkeypatch, [b'identified'] + [packet(step, stamp) for step, stamp in seq])
    result = T.capture(T.CaptureConfig('host'), tmp_path / 'trace')
    assert result['duplicate_sequences'] == 1
    assert result['out_of_order'] == 1
    assert result['counter_regressions'] == 1
    assert result['possible_counter_resets'] == 1
    assert result['forward_sequence_gaps'] == 1
    assert result['skipped_sequence_values'] == 1


def test_invalid_packets_counted_and_missing_fields_remain_missing(tmp_path, monkeypatch):
    sock = install(monkeypatch, [b'identified', b'broken', b'[]', b'{"speed":NaN}',
                                b'{"speed":1e999}', b'\xff', b'{}', b'x' * 301])
    result = T.capture(T.CaptureConfig('host', max_datagram_bytes=300), tmp_path / 'trace')
    assert result['malformed_packets'] == 5
    assert result['oversized_packets'] == 1
    assert result['valid_samples'] == 1
    assert result['missing_fields'] == {key: 1 for key in T.REQUIRED_FIELDS}
    sample = records(tmp_path / 'trace')[1]
    assert sample['telemetry'] == {}
    assert sock.sent[-1] == b'disconnect'


@pytest.mark.parametrize('reply', [b'wrong', OSError('network failed'), KeyboardInterrupt()])
def test_cleanup_and_final_summary_after_handshake_error(tmp_path, monkeypatch, reply):
    sock = install(monkeypatch, [reply])
    summary = T.capture(T.CaptureConfig('host'), tmp_path / 'trace')
    assert summary['errors']
    assert records(tmp_path / 'trace')[-1]['type'] == 'summary'
    assert sock.sent == [b'connect', b'disconnect']
    assert sock.closed


def test_handshake_retries_bounded(tmp_path, monkeypatch):
    sock = install(monkeypatch, [socket.timeout(), socket.timeout(), b'identified'])
    result = T.capture(T.CaptureConfig('host', handshake_attempts=2), tmp_path / 'trace')
    assert sock.sent == [b'connect', b'connect', b'disconnect']
    assert result['errors']


def test_max_packets_stops_while_server_has_more_data(tmp_path, monkeypatch):
    sock = install(monkeypatch, [b'identified'] + [packet(i, float(i)) for i in range(1, 5)])
    result = T.capture(T.CaptureConfig('host', max_packets=2), tmp_path / 'trace')
    assert result['packets_received'] == 2
    assert len(sock.replies) == 2
    assert sock.closed


def test_duration_deadline_stops_without_accepting_late_packet(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(T.time, 'monotonic', lambda: clock[0])
    sock = install(monkeypatch, [b'identified', packet(1, 1.), packet(2, 1.04), packet(3, 1.08)])
    receive = sock.recv

    def delayed_receive(limit):
        value = receive(limit)
        if value != b'identified':
            clock[0] += .6
        return value

    sock.recv = delayed_receive
    result = T.capture(T.CaptureConfig('host', duration_s=1.), tmp_path / 'trace')
    assert result['valid_samples'] == 1
    assert len(sock.replies) == 1
    assert sock.sent == [b'connect', b'disconnect']


def test_connect_and_handshake_share_one_deadline(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(T.time, 'monotonic', lambda: clock[0])
    sock = FakeSocket([])

    def late_open(config, deadline):
        clock[0] = deadline + .01
        return sock

    monkeypatch.setattr(T, '_open_socket', late_open)
    result = T.capture(T.CaptureConfig('host'), tmp_path / 'trace')
    assert result['errors']
    assert sock.sent == [b'disconnect']
    assert sock.closed


def test_late_identification_expires_handshake_and_disconnects(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(T.time, 'monotonic', lambda: clock[0])
    sock = install(monkeypatch, [b'identified', packet()])
    receive = sock.recv

    def delayed_receive(limit):
        reply = receive(limit)
        clock[0] = 4.
        return reply

    sock.recv = delayed_receive
    output = tmp_path / 'trace'
    result = T.capture(T.CaptureConfig('host', handshake_timeout_s=3.), output)
    assert 'TimeoutError: capture deadline expired' in result['errors']
    assert result['packets_received'] == 0
    assert result['valid_samples'] == 0
    assert sock.replies == [packet()]
    assert sock.sent == [b'connect', b'disconnect']
    assert sock.closed
    assert [row['type'] for row in records(output)] == ['metadata', 'summary']


def test_server_disconnect_is_reported(tmp_path, monkeypatch):
    sock = install(monkeypatch, [b'identified', packet(), b'disconnect'])
    result = T.capture(T.CaptureConfig('host'), tmp_path / 'trace')
    assert result['valid_samples'] == 1
    assert result['protocol_messages'] == 1
    assert any('server disconnected' in error for error in result['errors'])
    assert sock.sent[-1] == b'disconnect'


def test_open_socket_connects_exact_resolved_udp_peer(monkeypatch):
    sock = FakeSocket([])
    monkeypatch.setattr(T.socket, 'getaddrinfo', lambda host, port, family, kind:
                        [(socket.AF_INET, socket.SOCK_DGRAM, 17, '', ('10.0.0.7', port))])
    monkeypatch.setattr(T.socket, 'socket', lambda *args: sock)
    assert T._open_socket(T.CaptureConfig('lenovo.example'), T.time.monotonic() + 1) is sock
    assert sock.peer == ('10.0.0.7', 2345)


def test_explicit_source_files_hashed_without_loading(tmp_path, monkeypatch):
    source = tmp_path / 'checkpoint'
    source.write_bytes(b'not a model; must not load')
    install(monkeypatch, [b'identified', packet()])
    T.capture(T.CaptureConfig('host'), tmp_path / 'trace', {'checkpoint': source})
    entry = records(tmp_path / 'trace')[0]['source_files']['checkpoint']
    assert len(entry['sha256']) == 64
    assert entry['path'] == str(source.resolve())
    assert source.read_bytes() == b'not a model; must not load'


@pytest.mark.parametrize('kwargs', [dict(host=None), dict(host='a b'), dict(port=0),
    dict(duration_s=float('nan')), dict(duration_s=0), dict(handshake_attempts=6),
    dict(max_packets=True), dict(max_datagram_bytes=0)])
def test_configuration_bounds(kwargs):
    with pytest.raises(ValueError):
        T.CaptureConfig(**{'host': 'host', **kwargs})


def test_cli_host_precedence_and_missing_host(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('AC_HOST', 'environment-host')
    selected = []

    def fake_capture(config, output, sources):
        selected.append(config.host)
        return dict(errors=[], malformed_packets=0, oversized_packets=0,
                    missing_fields={}, invalid_numeric_fields={})

    monkeypatch.setattr(T, 'capture', fake_capture)
    assert T.main(['--output', str(tmp_path / 'a')]) == 0
    assert T.main(['--host', 'explicit-host', '--output', str(tmp_path / 'b')]) == 0
    assert selected == ['environment-host', 'explicit-host']
    monkeypatch.delenv('AC_HOST')
    with pytest.raises(SystemExit) as exc:
        T.main(['--output', str(tmp_path / 'c')])
    assert exc.value.code == 2


def test_cli_failure_status_for_missing_fields(tmp_path, monkeypatch, capsys):
    install(monkeypatch, [b'identified', b'{}'])
    assert T.main(['--host', 'host', '--output', str(tmp_path / 'trace')]) == 1
    assert json.loads(capsys.readouterr().out)['missing_fields']


def test_interval_summary_labels_and_values():
    assert T._intervals([]) is None
    result = T._intervals([0., .04, .08, .12])
    assert result['p99_ms'] == pytest.approx(40.)
    assert result['max_ms'] == pytest.approx(40.)


def test_numeric_extremes_do_not_break_summary():
    assert T._numeric(10 ** 1000) is False
    assert T._intervals([-1e308, 1e308]) is None


def test_invalid_numeric_field_is_preserved_and_reported(tmp_path, monkeypatch):
    install(monkeypatch, [b'identified', packet(speed='unknown')])
    result = T.capture(T.CaptureConfig('host'), tmp_path / 'trace')
    assert result['invalid_numeric_fields'] == {'speed': 1}
    assert records(tmp_path / 'trace')[1]['telemetry']['speed'] == 'unknown'
