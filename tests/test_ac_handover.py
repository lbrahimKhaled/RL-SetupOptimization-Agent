"""First-gear handover protocol tests; fake transport, never vehicle actuation."""
import json

import pytest

from rlsetup.assetto import handover as h


class Clock:
    value = 0.

    def now(self):
        return self.value


class FakeManagement:
    def __init__(self, config):
        pass

    def query(self, command):
        if command == 'get_config':
            return {'sampling_freq': 100, 'ego_sampling_freq': 25, 'vjoy_executed_by_server': True}
        return {'CarName': 'ks_ferrari_sf70h', 'TrackName': 'ks_barcelona',
                'TrackConfiguration': 'layout_gp'}


class FakeSocket:
    def __init__(self, clock, mode='success'):
        self.clock, self.mode = clock, mode
        self.sent, self.i, self.timeout = [], 0, 0.
        self.brake, self.pulsed, self.closed = 0., False, False

    def settimeout(self, value):
        self.timeout = value

    def send(self, payload):
        self.sent.append((self.clock.now(), payload))
        if payload.startswith(b'{'):
            command = json.loads(payload)
            self.brake = 1. if command['brake'] == 1. else 0.
            if command['shift_up']:
                self.pulsed = True
        return len(payload)

    def recv(self, count):
        if self.timeout == 0.:
            raise BlockingIOError()
        if self.i == 0:
            self.i += 1
            return b'identified'
        if self.pulsed and self.mode == 'timeout':
            self.clock.value += self.timeout
            raise TimeoutError('no packet')
        self.clock.value += .04
        self.i += 1
        if self.mode == 'malformed' and self.pulsed:
            return b'{bad'
        if self.mode == 'interrupt' and self.pulsed:
            raise KeyboardInterrupt()
        packet = {'steps': self.i, 'timestamp_ac': 100+self.i*.04, 'speed': 0.,
                  'actualGear': 2 if self.pulsed or self.mode == 'already' else 1,
                  'accStatus': 0., 'brakeStatus': self.brake}
        if self.mode == 'brakefail':
            packet['brakeStatus'] = 0.
        if self.mode == 'reverse':
            packet['actualGear'] = 0
        if self.mode == 'motion':
            packet['speed'] = .05
        if self.mode == 'gas':
            packet['accStatus'] = .02
        if self.mode == 'nonfinite':
            packet['speed'] = 'nan'
        return json.dumps(packet).encode()

    def close(self):
        self.closed = True


def run(tmp_path, monkeypatch, mode='success'):
    clock = Clock()
    sock = FakeSocket(clock, mode)
    monkeypatch.setattr(h.time, 'monotonic', clock.now)
    monkeypatch.setattr(h, '_open_socket', lambda *args: sock)
    monkeypatch.setattr(h, 'ManagementClient', FakeManagement)
    trace = tmp_path/'handover.jsonl'
    result = h.execute(h.HandoverConfig('host'), trace)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows[-1]['type'] == 'summary'
    assert rows[-1]['success'] == result['success']
    return result, sock


def commands(sock):
    return [(stamp, json.loads(data)) for stamp, data in sock.sent if data.startswith(b'{')]


def assert_cleanup(sock):
    assert sock.closed and sock.sent[-1][1] == b'disconnect'
    command = commands(sock)[-1][1]
    assert command['brake'] == -1. and command['acc'] == -1. and command['steer'] == 0.
    assert not any(command[key] for key in ('enable_gear_shift', 'shift_up', 'shift_down'))


def test_success_exactly_one_edge_and_brake_through_confirmation(tmp_path, monkeypatch):
    result, sock = run(tmp_path, monkeypatch)
    assert result['success'] and result['pulse_count'] == 1 and result['release_sent']
    controls = commands(sock)
    assert sum(bool(p['shift_up']) for _, p in controls) == 1
    pulse_index = next(i for i, (_, p) in enumerate(controls) if p['shift_up'])
    assert controls[pulse_index][1]['enable_gear_shift'] is True
    assert controls[pulse_index+1][0]-controls[pulse_index][0] <= .1
    assert all(p['brake'] == 1. and p['acc'] == -1. and p['steer'] == 0.
               for _, p in controls[:-1])
    assert_cleanup(sock)


def test_already_first_stationary_never_pulses(tmp_path, monkeypatch):
    result, sock = run(tmp_path, monkeypatch, 'already')
    assert result['success'] and result['already_first'] and result['pulse_count'] == 0
    assert not any(p['shift_up'] for _, p in commands(sock))
    assert_cleanup(sock)


@pytest.mark.parametrize('mode', ['reverse', 'motion', 'gas', 'nonfinite', 'brakefail'])
def test_unsafe_baseline_or_unverified_brake_never_pulses(tmp_path, monkeypatch, mode):
    result, sock = run(tmp_path, monkeypatch, mode)
    assert not result['success'] and result['error'] and result['pulse_count'] == 0
    assert not any(p['shift_up'] for _, p in commands(sock))
    assert_cleanup(sock)


def test_timeout_after_pulse_clears_at_point_one_without_packet(tmp_path, monkeypatch):
    result, sock = run(tmp_path, monkeypatch, 'timeout')
    assert not result['success'] and result['pulse_count'] == 1
    controls = commands(sock)
    pulse = next(i for i, (_, p) in enumerate(controls) if p['shift_up'])
    assert controls[pulse+1][0]-controls[pulse][0] == pytest.approx(.1)
    assert controls[pulse+1][1]['brake'] == 1.
    assert not controls[pulse+1][1]['shift_up']
    assert sock.sent[-1][0] <= 4.
    assert_cleanup(sock)


@pytest.mark.parametrize('mode', ['malformed', 'interrupt'])
def test_malformed_and_interrupt_cleanup(tmp_path, monkeypatch, mode):
    result, sock = run(tmp_path, monkeypatch, mode)
    assert not result['success'] and result['error'] and result['release_attempted']
    assert_cleanup(sock)


def test_wrong_management_identity_never_opens_udp(tmp_path, monkeypatch):
    class WrongManagement(FakeManagement):
        def query(self, command):
            value = super().query(command)
            if command == 'get_static_info':
                value['CarName'] = 'other'
            return value
    monkeypatch.setattr(h, 'ManagementClient', WrongManagement)
    monkeypatch.setattr(h, '_open_socket', lambda *a: pytest.fail('UDP forbidden'))
    result = h.execute(h.HandoverConfig('host'), tmp_path/'trace')
    assert not result['success'] and not result['release_attempted']


def test_dry_run_no_network(monkeypatch, capsys):
    monkeypatch.setattr(h, 'execute', lambda *a: pytest.fail('execute forbidden'))
    monkeypatch.setattr(h.socket, 'socket', lambda *a: pytest.fail('socket forbidden'))
    assert h.main([]) == 0
    assert json.loads(capsys.readouterr().out)['offline_plan']


def test_duplicate_cannot_refresh_guard():
    guard = h.Guard(h.HandoverConfig('host'))
    packet = {'steps': 1, 'timestamp_ac': 1., 'speed': 0., 'actualGear': 1,
              'accStatus': 0., 'brakeStatus': 0.}
    assert guard.check(packet)
    assert not guard.check(packet)
    packet['timestamp_ac'] = 0.
    with pytest.raises(ValueError, match='regression'):
        guard.check(packet)


def test_management_failure_returns_recorded_summary_without_udp(tmp_path, monkeypatch):
    class BrokenManagement(FakeManagement):
        def query(self, command):
            raise h.ManagementError('inspection timed out')
    monkeypatch.setattr(h, 'ManagementClient', BrokenManagement)
    monkeypatch.setattr(h, '_open_socket', lambda *a: pytest.fail('UDP forbidden'))
    path = tmp_path/'trace'
    result = h.execute(h.HandoverConfig('host'), path)
    assert not result['success'] and 'ManagementError' in result['error']
    assert json.loads(path.read_text().splitlines()[-1])['type'] == 'summary'


def test_queued_unsafe_latest_cannot_authorize_pulse(tmp_path, monkeypatch):
    clock = Clock()
    class BurstSocket(FakeSocket):
        delivered = False
        def recv(self, count):
            if self.timeout == 0. and not self.delivered:
                self.delivered = True
                return json.dumps({'steps': 3, 'timestamp_ac': 100.12, 'speed': .2,
                                   'actualGear': 1, 'accStatus': 0., 'brakeStatus': 1.}).encode()
            return super().recv(count)
    sock = BurstSocket(clock)
    monkeypatch.setattr(h.time, 'monotonic', clock.now)
    monkeypatch.setattr(h, '_open_socket', lambda *args: sock)
    monkeypatch.setattr(h, 'ManagementClient', FakeManagement)
    result = h.execute(h.HandoverConfig('host'), tmp_path/'trace')
    assert not result['success'] and result['pulse_count'] == 0
    assert 'stationary' in result['error']
    assert_cleanup(sock)


def test_server_gap_and_loosened_speed_guard_rejected():
    guard = h.Guard(h.HandoverConfig('host'))
    packet = {'steps': 1, 'timestamp_ac': 1., 'speed': 0., 'actualGear': 1,
              'accStatus': 0., 'brakeStatus': 0.}
    guard.check(packet)
    with pytest.raises(ValueError, match='freshness'):
        guard.check({**packet, 'steps': 2, 'timestamp_ac': 1.21})
    with pytest.raises(ValueError, match='tightened'):
        h.HandoverConfig('host', speed_mps=.1)
