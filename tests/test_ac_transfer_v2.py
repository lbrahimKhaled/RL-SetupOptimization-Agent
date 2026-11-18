"""OCD -> AC transfer for v2 references (source lock, R4 terms, OCD settle start, stall rule).

Synthetic circular track and a kinematic fake car; no game, network or OCD access.
"""
import hashlib
import json
import math

import numpy as np
import pytest

import rlsetup.assetto.driving as d
from rlsetup.assetto.driving import (CollectConfig, Geometry, GAMMA, Profile, RewardTerms, SteeringMap,
                                     ac_borders_sha256, controls, load_reference, reward)
from rlsetup.assetto.learn import replay_identity

N = 720
RADIUS = 100.
F1_REWARD = dict(mode='time_pbrs', a=1., b=1., shaping_k=1., penalty_const=100., gamma=.992,
                 speed_k=20., speed_sigma=.3, line_k=.0705, line_free_m=.5, line_cap=.5)
SETTLE = dict(rear_axle_m=1.632, wheelbase_m=3.6, steer_lock_rad=.3, lookahead_m=3., lookahead_s=.35,
              speed_kp=.4, speed_ki=.2, integral_limit_m=5., brake_cap=.3, traction_control=True,
              traction_slip_start=.06, traction_slip_span=.03, released_pedal_action=-.8, settle_s=2.,
              start_speed_fraction=[.3, .7], eval_speed_fraction=.5, settle_window_margin_m=50.,
              stall_speed_mps=3., stall_time_s=2., wrong_way_rad=math.pi/2)
CONFIG_SHA = 'c'*64


def track_info():
    a = np.arange(N)*2*np.pi/N
    return dict(left_lane=np.c_[90*np.cos(a), 90*np.sin(a)].tolist(),
                right_lane=np.c_[110*np.cos(a), 110*np.sin(a)].tolist())


def write_reference(path, *, v_qss=20., corner_from=N//2, config_sha=CONFIG_SHA):
    info = track_info()
    a = np.arange(N)*2*np.pi/N
    rows = np.c_[RADIUS*np.cos(a), RADIUS*np.sin(a)]
    step = 2*np.pi*RADIUS/N  # chord ~ arc; a synthetic OCD path 3 % longer than the AC line
    s_path = 1.03*step*np.arange(N)+5.
    corner = (np.arange(N) >= corner_from).astype(float)
    path.write_text(json.dumps(dict(
        schema='rlsetup.ac.ocd-reference.v2', reference='ocd_path',
        ac_borders_sha256=ac_borders_sha256(info['left_lane'], info['right_lane']), rows=rows.tolist(),
        profile=dict(s_path_m=s_path.tolist(), v_qss_mps=[v_qss]*N, v_start_mps=[v_qss]*N,
                     corner=corner.tolist(), path_length_m=1.03*step*N),
        settle=SETTLE, source=dict(source_config_sha256=config_sha))))
    return info


def steering():
    return SteeringMap(dict(schema_version=1, validated=True, angle_rad=[-.3, 0, .3], command=[.9, 0, -.9],
                            baseline_angle_rad=0., car_name='car', track_name='track', track_configuration='gp',
                            setup_id='default_user_confirmed', sampling_hz=25))


class Car:
    """Kinematic fake: commands are AC controls dicts; one 0.04 s step per received packet."""
    def __init__(self, x=RADIUS, y=0., yaw=math.pi/2, tyres_out_after=None):
        self.x, self.y, self.yaw, self.v, self.i = x, y, yaw, 0., 0
        self.steer, self.gas, self.brake = 0., 0., 0.
        self.tyres_out_after, self.packets = tyres_out_after, []

    def apply(self, command, lock_map):
        self.steer = float(np.interp(-command['steer'], [-.9, 0, .9], [-.3, 0, .3]))
        self.gas, self.brake = (command['acc']+1)/2, (command['brake']+1)/2

    def step(self):
        self.i += 1
        if self.i > 1:
            self.v = max(0., self.v+.04*(8*self.gas-12*self.brake))
            self.yaw += self.v/3.6*math.tan(self.steer)*.04
            self.x += self.v*.04*math.cos(self.yaw)
            self.y += self.v*.04*math.sin(self.yaw)
        p = dict(steps=self.i, timestamp_ac=self.i*.04, world_position_x=self.x, world_position_y=self.y,
                 velocity_z=self.v*math.cos(self.yaw), velocity_x=self.v*math.sin(self.yaw), yaw=self.yaw,
                 tyre_slip_ratio_rl=.01, tyre_slip_ratio_rr=.01, actualGear=2, accStatus=0., brakeStatus=0.,
                 numberOfTyresOut=int(self.tyres_out_after is not None and self.i > self.tyres_out_after))
        for name in ('FL', 'FR', 'RL', 'RR'):
            angle = self.yaw+(self.steer if name in ('FL', 'FR') else 0)
            p[f'tyreContactHeading_{name}_z'], p[f'tyreContactHeading_{name}_x'] = math.cos(angle), math.sin(angle)
        self.packets.append(p)
        return p


def harness(monkeypatch, tmp_path, *, actions, reward_config=F1_REWARD, car=None, reference_kwargs=None,
            config_sha=CONFIG_SHA):
    bundle = tmp_path/'actor'
    bundle.mkdir(exist_ok=True)
    (bundle/'manifest.json').write_text('{}')
    (bundle/'config.json').write_text('{}')
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps(steering().data))
    reference = tmp_path/'reference.json'
    info = write_reference(reference, **(reference_kwargs or {}))
    monkeypatch.setattr(d, 'source_contract_sha256', lambda manifest: 'a'*64)
    script = list(actions)

    class Actor:
        manifest = {'source': {'config_sha256': config_sha}}
        steer_lock_rad = .3
        source_config = {'env': {'reward': reward_config, 'stall_speed_mps': 3., 'stall_time_s': 2.}}
        seen = []
        def __init__(self, _path):
            pass
        def predict(self, obs, **_kwargs):
            Actor.seen.append(obs.copy())
            return np.asarray(script.pop(0) if len(script) > 1 else script[0], np.float32)
    monkeypatch.setattr(d, 'ActorRuntime', Actor)

    class Management:
        def __init__(self, _config):
            pass
        def query(self, name):
            return {'get_static_info': dict(CarName='car', TrackName='track', TrackConfiguration='gp'),
                    'get_config': dict(ego_sampling_freq=25, vjoy_executed_by_server=True),
                    'get_track_info': info}[name]
    monkeypatch.setattr(d, 'ManagementClient', Management)
    fake = car or Car()

    class FakeSession:
        def __init__(self, config, steering_map, lock):
            assert lock == .3
            self.steering_map, self.lock = steering_map, lock
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def receive(self, _deadline):
            self.last = fake.step()
            return self.last
        def send(self, action):
            command = controls(action, self.steering_map, self.last['steps'], saturation='clip',
                               steer_lock_rad=self.lock)
            fake.apply(command, None)
            return command
    monkeypatch.setattr(d, 'Session', FakeSession)
    return bundle, calibration, reference, info, fake, Actor


def run(bundle, calibration, reference, output, max_steps=400, **config):
    return d.collect(CollectConfig('localhost', max_steps, **config), bundle, calibration,
                     reference='ocd_path', output=output, execute=True, reference_file=reference)


def test_controls_use_the_source_lock():
    s = steering()
    assert controls([1., -1., -1.], s, 3, steer_lock_rad=.3)['steer'] == pytest.approx(s.command(.3))
    assert controls([-.5, -1., -1.], s, 3, steer_lock_rad=.3)['steer'] == pytest.approx(s.command(-.15))
    with pytest.raises(ValueError, match='calibration domain'):  # legacy 0.6 rad on a 0.3 rad car
        controls([1., -1., -1.], s, 3, steer_lock_rad=.6)
    with pytest.raises(ValueError, match='steer_lock_rad'):
        controls([0., -1., -1.], s, 3, steer_lock_rad=0.)


def test_reward_terms_follow_the_source_config():
    legacy = RewardTerms.from_source({'env': {'reward': {k: F1_REWARD[k] for k in
                                                         ('mode', 'a', 'b', 'shaping_k', 'penalty_const', 'gamma')}}})
    assert not legacy.active
    terms = RewardTerms.from_source({'env': {'reward': F1_REWARD}})
    assert terms.active and terms.speed_k == 20 and terms.line_k == .0705
    assert terms.phi_v(20., 20.) == 20 and terms.phi_v(26., 20.) == 0 and terms.phi_v(17., 20.) == pytest.approx(15)
    assert terms.line_penalty(.4, 2., True) == 0                       # inside the free band
    assert terms.line_penalty(2.5, 2., True) == pytest.approx(.0705*2*2)
    assert terms.line_penalty(50., 2., True) == pytest.approx(.5*2)    # capped per metre
    assert terms.line_penalty(2.5, 2., False) == 0 and terms.line_penalty(2.5, -1., True) == 0
    for key, bad in (('mode', 'ref_tracking'), ('b', 2.), ('penalty_const', 50.), ('gamma', .99)):
        with pytest.raises(ValueError, match=key):
            RewardTerms.from_source({'env': {'reward': dict(F1_REWARD, **{key: bad})}})


def test_v2_drive_starts_from_ocd_settle_and_rewards_r4_terms(monkeypatch, tmp_path):
    # Policy: hold ~steady throttle, then full brake to a standstill -> OCD stall truncation.
    bundle, calibration, reference, info, car, Actor = harness(
        monkeypatch, tmp_path, actions=[[0., -.4, -.8]]*60+[[0., -.8, 1.]]*200)
    result = run(bundle, calibration, reference, tmp_path/'chunk.npz')
    with np.load(tmp_path/'chunk.npz', allow_pickle=False) as chunk:
        meta = json.loads(chunk['metadata_json'].item())
        rewards, obs = chunk['rewards'].astype(float), chunk['obs']
        truncated, terminated = chunk['truncated'], chunk['terminated']
    start = meta['start']
    assert start['outcome'] == 'handover' and start['fraction'] == .5        # deterministic: eval fraction
    assert start['target_mps'] == pytest.approx(10.)                          # 0.5 x v_qss 20 m/s
    assert start['speed_mps'] >= 9. and start['runup_s'] >= 2.
    assert obs[0, 0]*80 == pytest.approx(start['speed_mps'], abs=.5)          # first stored state is moving
    assert (result['stop_reason'], truncated[-1], terminated.any()) == ('stall', True, False)
    assert meta['source_terms'] == dict(speed_k=20., speed_sigma=.3, line_k=.0705, line_free_m=.5, line_cap=.5,
                                        progress='ocd_track_path', start='ocd_settle',
                                        stall_speed_mps=3., stall_time_s=2.)
    assert replay_identity(meta)['source_terms'] == meta['source_terms']
    assert meta['steer_lock_rad'] == .3
    # Independent recomputation from the packets the policy saw (hand-over packet onward).
    track, _ = load_reference(info, 'ocd_path', reference)
    g = Geometry(track, reference='ocd_path')
    p = Profile(track['ocd_profile'], g)
    terms = RewardTerms.from_source({'env': {'reward': F1_REWARD}})
    packets = car.packets[-(len(rewards)+1):]
    speed = lambda q: math.hypot(q['velocity_z'], q['velocity_x'])
    s0, e0, _ = g.project(np.array([packets[0]['world_position_x'], packets[0]['world_position_y']]))
    phi = terms.phi_v(speed(packets[0]), 20.)
    expected = []
    for q in packets[1:]:
        s1, e1, _ = g.project(np.array([q['world_position_x'], q['world_position_y']]))
        ds = p.path_progress(s0, s1)
        assert ds == pytest.approx(1.03*g.progress(s0, s1), rel=1e-4)    # OCD Track.path metres (chord vs arc)
        phi_next = terms.phi_v(speed(q), 20.)
        expected.append(reward(ds, .04, e0, e1) + GAMMA*phi_next - phi - terms.line_penalty(abs(e1), ds, p.corner(s1)))
        s0, e0, phi = s1, e1, phi_next
    np.testing.assert_allclose(rewards, expected, rtol=1e-5, atol=1e-4)
    # The stall clock: > 2 s below 3 m/s, i.e. 51 decisions at 0.04 s once the car is slow.
    slow = [speed(q) < 3. for q in packets[1:]]
    assert slow[-51:] == [True]*51 and not all(slow[-52:])


def test_explore_drives_draw_the_ocd_start_fraction(monkeypatch, tmp_path):
    bundle, calibration, reference, *_ = harness(monkeypatch, tmp_path, actions=[[0., -.4, -.8]])
    fractions = []
    for seed in (1, 2, 3):
        output = tmp_path/f'explore{seed}.npz'
        run(bundle, calibration, reference, output, max_steps=5, exploration_std_scale=.3, exploration_seed=seed)
        with np.load(output, allow_pickle=False) as chunk:
            start = json.loads(chunk['metadata_json'].item())['start']
        assert .3 <= start['fraction'] <= .7 and start['target_mps'] == pytest.approx(20*start['fraction'])
        fractions.append(start['fraction'])
    assert len(set(fractions)) == 3
    assert fractions[0] == float(np.random.default_rng(1).uniform(.3, .7))  # reproducible per drive seed


def test_failed_runup_stores_nothing(monkeypatch, tmp_path):
    bundle, calibration, reference, *_ = harness(monkeypatch, tmp_path, actions=[[0., -.4, -.8]],
                                                 car=Car(tyres_out_after=20))
    result = run(bundle, calibration, reference, tmp_path/'off.npz')
    assert (result['stop_reason'], result['transitions']) == ('runup_off_track', 0)


def test_v2_reference_is_bound_to_the_source_run(monkeypatch, tmp_path):
    bundle, calibration, reference, *_ = harness(monkeypatch, tmp_path, actions=[[0., -.4, -.8]],
                                                 config_sha='d'*64)
    with pytest.raises(ValueError, match='different source run config'):
        run(bundle, calibration, reference, tmp_path/'x.npz')


def test_r4_source_needs_a_v2_reference(monkeypatch, tmp_path):
    bundle, calibration, reference, info, *_ = harness(monkeypatch, tmp_path, actions=[[0., -.4, -.8]])
    saved = json.loads(reference.read_text())
    legacy = tmp_path/'v1.json'
    legacy.write_text(json.dumps(dict(schema='rlsetup.ac.ocd-reference.v1', reference='ocd_path',
                                      ac_borders_sha256=saved['ac_borders_sha256'], rows=saved['rows'])))
    with pytest.raises(ValueError, match='v2 reference'):
        run(bundle, calibration, legacy, tmp_path/'y.npz')


def test_v2_profile_validation(tmp_path):
    info = write_reference(tmp_path/'ok.json')
    load_reference(info, 'ocd_path', tmp_path/'ok.json')
    for mutate in (lambda r: r['profile']['s_path_m'].reverse(), lambda r: r['profile']['corner'].__setitem__(0, .5),
                   lambda r: r['settle'].pop('brake_cap'), lambda r: r['source'].pop('source_config_sha256'),
                   lambda r: r['profile'].__setitem__('v_start_mps', [0.]*N)):
        saved = json.loads((tmp_path/'ok.json').read_text())
        mutate(saved)
        bad = tmp_path/f'bad{hashlib.sha256(json.dumps(saved).encode()).hexdigest()[:8]}.json'
        bad.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match='v2 reference'):
            load_reference(info, 'ocd_path', bad)
