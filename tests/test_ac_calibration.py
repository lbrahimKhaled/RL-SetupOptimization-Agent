"""Offline AC calibration guards and stationary protocol, fake sockets only."""
import json
import math
from dataclasses import replace

import pytest

from rlsetup.assetto import calibration as c


def packet(i=1, angle=.012, rear=2.9):
    p = {'steps': i, 'timestamp_ac': 100+i*.04, 'speed': 0., 'actualGear': 1,
         'accStatus': 0., 'brakeStatus': 1., 'world_position_x': 1.,
         'world_position_y': 2., 'world_position_z': 3.}
    for wheel in ('FL', 'FR', 'RL', 'RR'):
        a = rear + (angle if wheel[0] == 'F' else 0.)
        for axis, value in (('x', math.sin(a)), ('y', 0.), ('z', math.cos(a))):
            p[f'tyreContactHeading_{wheel}_{axis}'] = value
    return p


def phases(config, sign=1., baseline=.012):
    return [{'command': v, 'angle_rad': baseline+sign*.3*v} for v in config.phases]


@pytest.mark.parametrize('rear', [0., 2.9, -3.1, math.pi])
def test_heading_rotations_and_backwards_vectors(rear):
    assert c.heading(packet(angle=.03, rear=rear))[0] == pytest.approx(.03)


def test_toe_circular_mean_and_baseline_once():
    p = packet(angle=.012)
    for wheel, toe in [('FL', .02), ('FR', -.02)]:
        a = 2.9+.012+toe
        p[f'tyreContactHeading_{wheel}_x'] = math.sin(a)
        p[f'tyreContactHeading_{wheel}_z'] = math.cos(a)
    assert c.heading(p)[0] == pytest.approx(.012)
    config = c.CalibrationConfig('host', True)
    result = c.measured_map(phases(config), .012, config)
    assert result['baseline_angle_rad'] == .012
    assert result['angle_rad'][4] == pytest.approx(0.)


def test_degenerate_vectors():
    p = packet()
    p['tyreContactHeading_FL_x'] = p['tyreContactHeading_FL_z'] = 0.
    with pytest.raises(ValueError, match='degenerate'):
        c.heading(p)


@pytest.mark.parametrize('sign', [-1, 1])
def test_pair_sort_and_measured_full_domain(sign):
    config = c.CalibrationConfig('host', True)
    result = c.measured_map(phases(config, sign), .012, config)
    assert result['validated']
    assert result['angle_rad'] == sorted(result['angle_rad'])
    assert result['command'] == [sign*v for v in [-1., -.5, -.25, -.1, 0., .1, .25, .5, 1.]]
    assert result['angle_rad'][0] == pytest.approx(-.3)
    assert result['angle_rad'][-1] == pytest.approx(.3)


def test_small_probe_not_validated():
    config = c.CalibrationConfig('host')
    assert not c.measured_map(phases(config), .012, config)['validated']


@pytest.mark.parametrize('change', ['incomplete', 'zero', 'nonmonotonic', 'center'])
def test_bad_map_rejected(change):
    config = c.CalibrationConfig('host', True)
    rows = phases(config)
    if change == 'incomplete':
        rows.pop()
    elif change == 'zero':
        rows[1]['angle_rad'] = .012
    elif change == 'nonmonotonic':
        rows[2]['angle_rad'] = -.2
    else:
        rows[-1]['angle_rad'] += .02
    with pytest.raises(ValueError):
        c.measured_map(rows, .012, config)


def test_command_exact_keys_and_released_pedals():
    p = c.command_payload(4)
    assert p == {'server_steps': 4, 'steer': 0., 'acc': -1., 'brake': -1.,
                 'enable_gear_shift': 0, 'shift_up': 0, 'shift_down': 0}
    assert c.command_payload(5, -.1, 1.)['brake'] == 1.
    for steps, steer in [(1., 0.), (1, float('nan')), (1, 1.1)]:
        with pytest.raises(ValueError):
            c.command_payload(steps, steer)


@pytest.mark.parametrize('key,value', [('speed', .05), ('actualGear', 2), ('accStatus', .1),
                                      ('world_position_x', 1.03), ('speed', float('nan')),
                                      ('steps', -1), ('timestamp_ac', 0.)])
def test_guard_abort(key, value):
    guard = c.Guard(c.CalibrationConfig('host'))
    guard.check(packet())
    p = packet(2)
    p[key] = value
    with pytest.raises(ValueError):
        guard.check(p)


def test_guard_speed_limit_is_configurable():
    # A full-lock-to-centre snap twitched a stationary car to 0.061 m/s with no displacement.
    p = packet(2)
    p['speed'] = .061
    strict = c.Guard(c.CalibrationConfig('host'))
    strict.check(packet())
    with pytest.raises(ValueError, match=r'actualGear 1, speed 0\.061'):
        strict.check(p)
    loose = c.Guard(c.CalibrationConfig('host', speed_mps=.1))
    loose.check(packet())
    loose.check(p)
    p = packet(3)
    p['speed'], p['world_position_x'] = .061, 1.03  # real movement still caught by the 2 cm guard
    with pytest.raises(ValueError, match=r'displacement.*0\.030 m > 0\.02 m'):
        loose.check(p)
    looser = c.Guard(c.CalibrationConfig('host', speed_mps=.2, motion_m=.1))
    looser.check(packet())
    looser.check(p)
    with pytest.raises(ValueError, match='rear heading'):  # rotation guard unchanged
        looser.check(packet(4, rear=3.))
    for flag in ('--speed-mps', '--motion-m'):
        with pytest.raises(SystemExit):
            c.main(['--execute', flag, '0.5'])


def test_guard_missing_heading_and_rotating_rear():
    guard = c.Guard(c.CalibrationConfig('host'))
    guard.check(packet())
    with pytest.raises(ValueError, match='rear heading'):
        guard.check(packet(2, rear=3.))
    p = packet(2)
    del p['brakeStatus']
    with pytest.raises(ValueError, match='missing'):
        guard.check(p)


class FakeSocket:
    def __init__(self, *, bad=False, late=False):
        self.sent, self.i, self.timeout, self.closed = [], 0, None, False
        self.steer, self.bad, self.late = 0., bad, late

    def settimeout(self, timeout):
        self.timeout = timeout

    def send(self, data):
        self.sent.append(data)
        if data.startswith(b'{'):
            self.steer = json.loads(data)['steer']
        return len(data)

    def recv(self, count):
        if self.timeout == 0:
            raise BlockingIOError()
        if self.i == 0:
            self.i += 1
            if self.late:
                raise TimeoutError('late identified')
            return b'identified'
        self.i += 1
        if self.bad and self.i > 15:
            return b'{malformed'
        return json.dumps(packet(self.i, angle=.012+.3*self.steer)).encode()

    def close(self):
        self.closed = True


class FakeManagement:
    def __init__(self, config):
        pass

    def query(self, command):
        if command == 'get_config':
            return {'sampling_freq': 100, 'ego_sampling_freq': 25, 'vjoy_executed_by_server': True}
        return {'CarName': 'sf70h', 'TrackName': 'barcelona', 'TrackConfiguration': 'layout_gp'}


def setup_fake(monkeypatch, sock):
    monkeypatch.setattr(c, 'ManagementClient', FakeManagement)
    monkeypatch.setattr(c, '_open_socket', lambda *a: sock)


def test_execute_full_fake_socket(tmp_path, monkeypatch):
    sock = FakeSocket()
    setup_fake(monkeypatch, sock)
    output, trace = tmp_path/'map.json', tmp_path/'trace.jsonl'
    result = c.execute(c.CalibrationConfig('host', True), output, trace,
                       neutral_confirmed=True, at_controls=True)
    assert result.get('error') is None
    assert result['validated'] and result['release_sent'] and sock.closed
    assert result['raw_trace_sha256'] == c._sha(trace)
    assert json.loads(output.read_text()) == result
    commands = [json.loads(data) for data in sock.sent if data.startswith(b'{')]
    assert commands[0]['brake'] == -1. and commands[1]['brake'] == 1.
    assert all(p['acc'] == -1. and not p['enable_gear_shift'] for p in commands)
    assert commands[-1] == c.command_payload(commands[-1]['server_steps'])
    assert sock.sent[-1] == b'disconnect'
    assert len(result['angle_rad']) == 9
    # Every command uses advancing telemetry, except the initial brake probe pair.
    assert commands[-1]['server_steps'] >= commands[-2]['server_steps']


@pytest.mark.parametrize('kind', ['bad', 'late'])
def test_abort_cleanup(tmp_path, monkeypatch, kind):
    sock = FakeSocket(**{kind: True})
    setup_fake(monkeypatch, sock)
    result = c.execute(c.CalibrationConfig('host', True), tmp_path/'map', tmp_path/'trace',
                       neutral_confirmed=True, at_controls=True)
    assert result['error'] and not result['validated']
    assert sock.closed and sock.sent[-1] == b'disconnect'
    if kind == 'bad':
        assert result['release_sent']
        assert json.loads(sock.sent[-2])['brake'] == -1.


def test_exclusive_outputs_and_readiness_before_socket(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail('network must stay closed')
    monkeypatch.setattr(c, '_open_socket', forbidden)
    monkeypatch.setattr(c, 'ManagementClient', forbidden)
    output, trace = tmp_path/'map', tmp_path/'trace'
    output.write_text('keep')
    with pytest.raises(FileExistsError):
        c.execute(c.CalibrationConfig('host'), output, trace, neutral_confirmed=True, at_controls=True)
    assert output.read_text() == 'keep'
    with pytest.raises(ValueError, match='requires'):
        c.execute(c.CalibrationConfig('host'), tmp_path/'other', trace)


def test_dry_plan_no_network(monkeypatch, capsys):
    monkeypatch.setattr(c, 'execute', lambda *a, **kw: pytest.fail('execute called'))
    assert c.main([]) == 0
    assert json.loads(capsys.readouterr().out)['offline_plan']
    assert c.main(['--full-range']) == 0
    assert len(json.loads(capsys.readouterr().out)['phases']) == 11


def test_receiver_freshness_and_late_identified(monkeypatch):
    sock = FakeSocket()
    sock.i = 1
    receiver = c.Receiver(sock, c.CalibrationConfig('host'), lambda row: None)
    receiver.last_fresh = -10.
    with pytest.raises(TimeoutError):
        receiver.next(1e30)


def test_identified_received_after_deadline_rejected(tmp_path, monkeypatch):
    clock = [0.]
    class LateReply(FakeSocket):
        def recv(self, count):
            clock[0] = 2.1
            return b'identified'
    sock = LateReply()
    setup_fake(monkeypatch, sock)
    monkeypatch.setattr(c.time, 'monotonic', lambda: clock[0])
    result = c.execute(c.CalibrationConfig('host'), tmp_path/'map', tmp_path/'trace',
                       neutral_confirmed=True, at_controls=True)
    assert 'late identified' in result['error']
    assert sock.sent == [b'connect', b'disconnect'] and sock.closed


def test_failed_release_write_invalidates_map(tmp_path, monkeypatch):
    class ReleaseFailure(FakeSocket):
        def __init__(self):
            super().__init__()
            self.releases = 0

        def send(self, data):
            count = super().send(data)
            if data.startswith(b'{') and json.loads(data)['brake'] == -1.:
                self.releases += 1
                if self.releases == 2:
                    raise OSError('release transport failed')
            return count
    sock = ReleaseFailure()
    setup_fake(monkeypatch, sock)
    result = c.execute(c.CalibrationConfig('host', True), tmp_path/'map', tmp_path/'trace',
                       neutral_confirmed=True, at_controls=True)
    assert not result['validated']
    assert result['release_error'] == 'release transport failed'
    assert sock.sent[-1] == b'disconnect' and sock.closed
