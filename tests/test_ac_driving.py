"""Fast synthetic AC transfer checks; no game/network access."""
import json
import math

import numpy as np
import pytest

from rlsetup.assetto.driving import (CollectConfig, Geometry, Observation, RELEASE_ACTION,
                                     Session, SteeringMap, controls, reward, write_chunk, GAMMA)


def geometry():
    a = np.arange(720)*2*np.pi/720
    return Geometry(dict(left_lane=np.c_[90*np.cos(a), 90*np.sin(a)],
                         right_lane=np.c_[110*np.cos(a), 110*np.sin(a)]), reference='centerline')


def steering():
    return SteeringMap(dict(schema_version=1, validated=True, angle_rad=[-.2, 0, .2],
                            command=[.8, 0, -.8], baseline_angle_rad=.01,
                            car_name='car', track_name='track', track_configuration='gp',
                            setup_id='default_user_confirmed', sampling_hz=25))


def packet(i, *, yaw=math.pi/2):
    p = dict(steps=i, timestamp_ac=i*.04, world_position_x=100., world_position_y=0.,
             velocity_z=0., velocity_x=8., yaw=yaw, tyre_slip_ratio_rl=.02, tyre_slip_ratio_rr=-.03)
    for name in ('FL', 'FR', 'RL', 'RR'):
        angle = yaw+(.01 if name in ('FL', 'FR') else 0)
        p[f'tyreContactHeading_{name}_z'] = math.cos(angle)
        p[f'tyreContactHeading_{name}_x'] = math.sin(angle)
    return p


def test_closed_geometry_projection_preview_labels_and_footprint():
    g = geometry()
    station, e, heading = g.project(np.array([100., 0.]))
    assert station == 0 and e == 0
    assert heading == pytest.approx(math.pi/2, abs=.005)
    np.testing.assert_allclose(g.at(g.length)[0], [100, 0])
    assert g.progress(g.length-.5, .5) == pytest.approx(1)
    assert g.inside(np.array([100., 0]), math.pi/2, 5., 2.)
    assert not g.inside(np.array([110., 0]), math.pi/2, 5., 2.)
    with pytest.raises(ValueError, match='labelled'):
        Geometry(dict(left_lane=g.right, right_lane=g.left), reference='centerline')


def test_observation_source_columns_real_history_and_body_frame():
    g = geometry()
    builder = Observation(g, .01)
    for i in range(1, 5):
        assert builder.build(packet(i), RELEASE_ACTION) is None
    obs = builder.build(packet(5), RELEASE_ACTION)
    assert obs.shape == (159,) and obs.dtype == np.float32
    expected = [8/80, 8/80, 0, 0, 0, 0, 0, .1, -.15]
    np.testing.assert_allclose(obs[:9], expected, atol=1e-6)
    np.testing.assert_allclose(obs[132:135], RELEASE_ACTION)
    np.testing.assert_allclose(obs[135:144], expected, atol=1e-6)
    np.testing.assert_allclose(obs[147:156], expected, atol=1e-6)
    ref, left, right = g.at(g.preview)
    pts = np.stack([ref, left, right], axis=1)-[100, 0]
    local = np.stack([pts[..., 1], -pts[..., 0]], axis=-1)
    np.testing.assert_allclose(obs[12:132], (local/np.maximum(g.preview, 10)[:, None, None]).ravel(), atol=1e-6)
    bad = packet(6)
    del bad['velocity_x']
    with pytest.raises(ValueError, match='velocity_x'):
        builder.build(bad, RELEASE_ACTION)


def test_acceleration_and_wrapped_yaw_rate():
    builder = Observation(geometry(), .01)
    for i in range(1, 5):
        builder.build(packet(i, yaw=math.pi-.01), RELEASE_ACTION)
    p = packet(5, yaw=-math.pi+.01)
    obs = builder.build(p, RELEASE_ACTION)
    assert obs[3] == pytest.approx(.5)
    # Source midpoint rotating-frame formula, independently evaluated.
    old = np.array([8*math.sin(.01), -8*math.cos(.01), 0.])
    cur = np.array([-8*math.sin(.01), -8*math.cos(.01), .5])
    mid = (old+cur)/2
    expected = (cur[:2]-old[:2])/.04+[-mid[2]*mid[1], mid[2]*mid[0]]
    np.testing.assert_allclose(obs[4:6]*30, expected, atol=1e-5)


def test_r1_terminal_truncation_and_telescoping():
    assert reward(2, .04, 1, 3) == pytest.approx(2-.04-GAMMA*3+1)
    assert reward(2, .04, 1, 3, failure=True) == pytest.approx(2-.04+1-100)
    es = [1., 2., .5, 3.]
    shaped = sum(GAMMA**i*(reward(0, 0, es[i], es[i+1])) for i in range(3))
    assert shaped == pytest.approx(GAMMA**3*(-es[-1])+es[0])


def test_controls_deadzone_calibration_range_and_explicit_saturation():
    s = steering()
    result = controls([.1, -.8, 1], s, 43)
    assert result == dict(steer=pytest.approx(-.24), acc=-1., brake=1., server_steps=43,
                          enable_gear_shift=False, shift_up=False, shift_down=False)
    with pytest.raises(ValueError, match='domain'):
        controls([1, 0, 0], s, 43)
    assert controls([1, 0, 0], s, 43, saturation='clip')['steer'] == -.8
    s.verify(dict(CarName='car', TrackName='track', TrackConfiguration='gp'),
             dict(ego_sampling_freq=25, vjoy_executed_by_server=True), 'default_user_confirmed')
    with pytest.raises(ValueError, match='identity'):
        s.verify({}, {}, 'default_user_confirmed')


class FakeSocket:
    def __init__(self, packets):
        self.packets = list(packets)
        self.sent = []
        self.closed = False
        self.blocking_receives = 0
    def settimeout(self, timeout):
        assert 0 <= timeout <= (3. if self.blocking_receives == 0 else .08)
        self.nonblocking = timeout == 0
    def recv(self, _size):
        if self.nonblocking:
            raise BlockingIOError()
        self.blocking_receives += 1
        return json.dumps(self.packets.pop(0)).encode()
    def send(self, value):
        self.sent.append(value)
    def close(self):
        self.closed = True


def test_udp_counter_echo_gap_and_cleanup():
    import time
    sess = Session(CollectConfig('localhost', 1), steering())
    sock = FakeSocket([packet(1), packet(3)])
    sess.sock = sock
    sess.receive(time.monotonic()+1)
    sess.send(RELEASE_ACTION)
    assert json.loads(sock.sent[0])['server_steps'] == 1
    with pytest.raises(ValueError, match='gap'):
        sess.receive(time.monotonic()+1)
    sess.close()
    assert json.loads(sock.sent[-2])['acc'] == -1
    assert json.loads(sock.sent[-2])['brake'] == -1
    assert sock.sent[-1] == b'disconnect' and sock.closed


def test_numeric_chunk_roundtrip_exclusive(tmp_path):
    row = dict(obs=np.zeros(159), next_obs=np.ones(159), actions=np.zeros(3), rewards=1.,
               terminated=False, truncated=True, dt=.04, episode_id=0, seq=5,
               timestamp_ac=1., next_timestamp_ac=1.04, command=[0, -1, -1],
               desired_road_angle=0., executed_road_angle=0.)
    path = tmp_path/'chunk.npz'
    write_chunk(path, [row], {'schema_version': 1})
    with np.load(path, allow_pickle=False) as chunk:
        assert all(v.dtype.kind != 'O' for v in chunk.values())
        assert chunk['obs'].shape == (1, 159)
        assert chunk['dt'].dtype == np.float64
        assert json.loads(chunk['metadata_json'].item())['schema_version'] == 1
    with pytest.raises(FileExistsError):
        write_chunk(path, [row], {})


def test_collector_stops_boundary_and_discards_invalid_next(monkeypatch, tmp_path):
    import rlsetup.assetto.driving as d
    bundle = tmp_path/'actor'
    bundle.mkdir()
    (bundle/'manifest.json').write_text('{}')
    (bundle/'config.json').write_text('{}')
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps(steering().data))
    monkeypatch.setattr(d, 'source_contract_sha256', lambda manifest: 'a'*64)
    class Actor:
        manifest = {}
        def __init__(self, _path):
            pass
        def predict(self, obs):
            return np.array([.1, -.8, -.8], np.float32)
    monkeypatch.setattr(d, 'ActorRuntime', Actor)
    g = geometry()
    class Management:
        def __init__(self, _config):
            pass
        def query(self, name):
            return {'get_static_info': dict(CarName='car', TrackName='track', TrackConfiguration='gp'),
                    'get_config': dict(ego_sampling_freq=25, vjoy_executed_by_server=True),
                    'get_track_info': dict(left_lane=g.left.tolist(), right_lane=g.right.tolist())}[name]
    monkeypatch.setattr(d, 'ManagementClient', Management)
    class FakeSession:
        invalid = False
        def __init__(self, *_args):
            self.i = 0
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def receive(self, _deadline):
            self.i += 1
            result = packet(self.i)
            result.update(actualGear=2, accStatus=0, brakeStatus=0, numberOfTyresOut=0)
            if self.invalid and self.i == 7:
                del result['velocity_x']
            return result
        def send(self, action):
            return controls(action, steering(), self.i)
    monkeypatch.setattr(d, 'Session', FakeSession)
    for invalid, failure_lap in ((False, False), (True, False), (False, True)):
        FakeSession.invalid = invalid
        if failure_lap:
            g.length = .01
            g.progress = lambda *_args: .01
            monkeypatch.setattr(d, 'Geometry', lambda *_args, **_kwargs: g)
            original_receive = FakeSession.receive
            def receive_failure(self, deadline):
                result = original_receive(self, deadline)
                if self.i >= 6:
                    result['numberOfTyresOut'] = 1
                return result
            FakeSession.receive = receive_failure
        output = tmp_path/f'chunk{invalid}{failure_lap}.npz'
        result = d.collect(CollectConfig('localhost', 2), bundle, calibration,
                           reference='centerline', output=output, execute=True)
        with np.load(output, allow_pickle=False) as chunk:
            assert len(chunk['obs']) == (1 if invalid or failure_lap else 2)
            if failure_lap:
                assert chunk['terminated'][-1] and not chunk['truncated'][-1]
            else:
                assert chunk['truncated'][-1] and not chunk['terminated'].any()
            assert np.isfinite(chunk['next_obs']).all()
            if not invalid and not failure_lap:
                np.testing.assert_array_equal(chunk['next_obs'][0], chunk['obs'][1])
            assert result['transitions'] == len(chunk['obs'])


def test_udp_drains_latest_and_rejects_gap_after_anchor():
    import time
    class BacklogSocket(FakeSocket):
        def recv(self, _size):
            if self.packets:
                return json.dumps(self.packets.pop(0)).encode()
            raise BlockingIOError()
    sess = Session(CollectConfig('localhost', 1), steering())
    sess.sock = BacklogSocket([packet(1), packet(2), packet(3)])
    assert sess.receive(time.monotonic()+1)['steps'] == 3
    sess.sock.packets = [packet(4), packet(5)]
    with pytest.raises(ValueError, match='gap'):
        sess.receive(time.monotonic()+1)
    sess.close()


def test_handshake_deadline_checked_after_reply_and_releases(monkeypatch):
    import rlsetup.assetto.driving as d
    ticks = iter([0., 0., 4.])
    monkeypatch.setattr(d.time, 'monotonic', lambda: next(ticks))
    class LateSocket(FakeSocket):
        def settimeout(self, timeout):
            pass
        def recv(self, _size):
            return b'identified'
    sock = LateSocket([])
    monkeypatch.setattr(d, '_open_socket', lambda *_args: sock)
    with pytest.raises(TimeoutError, match='handshake'):
        with Session(CollectConfig('localhost', 1), steering()):
            pytest.fail('late handshake accepted')
    assert sock.closed and sock.sent[-1] == b'disconnect'


@pytest.mark.parametrize('overall_s', [1., .1])
def test_startup_wait_allows_120ms_but_retains_active_and_overall_limits(monkeypatch, overall_s):
    import rlsetup.assetto.driving as d
    clock = [0.]
    monkeypatch.setattr(d.time, 'monotonic', lambda: clock[0])
    class DelayedSocket(FakeSocket):
        def settimeout(self, timeout):
            super().settimeout(timeout)
            self.timeout = timeout
        def recv(self, size):
            if self.nonblocking:
                raise BlockingIOError()
            delay = .12 if self.blocking_receives == 0 else .09
            clock[0] += min(delay, self.timeout)
            if delay > self.timeout:
                raise TimeoutError('mock transport timeout')
            return super().recv(size)
    sess = Session(CollectConfig('localhost', 1), steering())
    sock = DelayedSocket([packet(1), packet(2)])
    sess.sock = sock
    try:
        if overall_s == .1:
            with pytest.raises(TimeoutError):
                sess.receive(overall_s)
            assert clock[0] == pytest.approx(.1) and sess.last is None
        else:
            assert sess.receive(overall_s)['steps'] == 1
            assert clock[0] == pytest.approx(.12)
            with pytest.raises(TimeoutError):
                sess.receive(overall_s)
            assert sock.timeout == pytest.approx(.08)
            assert sess.last['steps'] == 1
    finally:
        sess.close()
    assert json.loads(sock.sent[-2])['acc'] == -1.
    assert json.loads(sock.sent[-2])['brake'] == -1.
    assert sock.sent[-1] == b'disconnect' and sock.closed
