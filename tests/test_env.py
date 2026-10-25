import math
import warnings
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from ocd_lap.backend import OCDBackend, StartPose, Inputs
from ocd_lap.car_setup import Setup
from ocd_lap.env import OCDRacingEnv, MAX_STEER_RAD
from rlsetup.driving.config import EnvConfig, RewardConfig, ObsConfig
from rlsetup.driving.demo_driver import DemoPolicy
from rlsetup.driving.env import DrivingEnv, body_accelerations

NO_SETTLE = dict(settle_s=0., action_repeat=1)


def test_check_env(make_env):
    env = make_env()
    with warnings.catch_warnings():
        warnings.simplefilter('error')        # finite Box: no warnings expected at all
        check_env(env, skip_render_check=True)


def test_default_setup_is_p0_and_track_is_shared(env, track):
    assert env.setup == Setup()
    assert env.track is track


@pytest.mark.parametrize('obs_gear, history, n', [(False, 3, 159), (True, 3, 160), (False, 1, 135)])
def test_observation_size_follows_flags(make_env, obs_gear, history, n):
    env = make_env(EnvConfig(obs_gear=obs_gear, obs=ObsConfig(history=history)))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == (n,) == obs.shape
    obs, *_ = env.step(np.zeros(3, np.float32))
    assert obs.shape == (n,)


def test_equivalence_with_ocd_racing_env(make_env, track):
    """R2.5: xi_ref = Track.path, start s = 0 at 5 m/s, no settle, no repeat, original pedals,
    constant P = 100, stall/wrong-way checks off (OCDRacingEnv has none) -> rewards identical
    to OCDRacingEnv, including the off-track step."""
    cfg = EnvConfig(pedal_deadzone=False, max_episode_s=600., **NO_SETTLE,
                    stall_time_s=math.inf, wrong_way_rad=math.inf,
                    reward=RewardConfig(penalty_base=100., penalty_kappa=0.))
    ours = make_env(cfg)
    base = OCDRacingEnv(setup=Setup(), max_time_s=600., backend_factory=lambda: OCDBackend(
        powertrain='engine', max_brake_torque_Nm=12000.))
    base.reset()
    ours.reset(options={'start_pose': StartPose(*track.path[0, 1:], track.yaw0, 5.)})
    rng = np.random.default_rng(0)
    n = 0
    for k in range(3000):
        if k < 150:
            a = np.array([0., .2, -1.], np.float32)       # pull away
        elif k < 300:
            a = rng.uniform(-1, 1, 3).astype(np.float32)   # random
        else:
            a = np.array([1., 1., -1.], np.float32)       # drive off the track
        _, rb, tb, trb, ib = base.step(a)
        _, rd, td, trd, idd = ours.step(a)
        assert rd == rb, k
        assert (td, trd) == (tb, trb)
        n += 1
        if tb or trb:
            break
    assert ib['off_track'] and idd['termination'] == 'off_track' and n > 150


def test_action_repeat_sums_base_tick_rewards(make_env):
    one = make_env(EnvConfig(action_repeat=1, settle_s=0.))   # settle runs at the decision rate
    two = make_env(EnvConfig(action_repeat=2, settle_s=0.))
    one.reset(seed=3), two.reset(seed=3)
    for a in ([0., .3, -1.], [.1, -.2, -1.], [-.05, 1., -1.]):
        a = np.array(a, np.float32)
        _, r1, *_ = one.step(a)
        _, r2, *_ = one.step(a)
        _, r, _, _, info = two.step(a)
        assert r == r1 + r2
        assert info['transition_dt_s'] == pytest.approx(.04)


def test_reset_determinism_and_setup_independent_start(make_env):
    env = make_env()
    a, ia = env.reset(seed=7)
    b, ib = env.reset(seed=7)
    np.testing.assert_array_equal(a, b)
    _, ic = env.reset(seed=7, options={'setup': Setup(brake_bias_front=.6, aero_level=.8)})
    assert (ic['start']['s0'], ic['start']['v0']) == (ia['start']['s0'], ia['start']['v0'])
    _, idd = env.reset(seed=8)
    assert idd['start']['s0'] != ia['start']['s0']
    lo, hi = env.cfg.start_speed_fraction
    assert lo <= ia['start']['fraction'] <= hi


def test_setup_none_option_means_default_setup(make_env):
    """N1 (D1_signoff): options={'setup': None} must not reach backend.reset (which would skip
    the translator and drive OCD's shipped car); it means the env's default setup."""
    env = make_env()
    obs_a, ia = env.reset(seed=9)
    state_a = dict(env.state)
    obs_b, ib = env.reset(seed=9, options={'setup': None})
    assert ib['setup'] == Setup() == ia['setup']
    assert env.state == state_a
    np.testing.assert_array_equal(obs_a, obs_b)


def test_observation_scales(make_env):
    """D1_signoff A5: v_y / 5 m/s, actual steer / 0.6 rad (action units), heading / 0.5 rad."""
    o = ObsConfig()
    assert o.steer_scale_rad == MAX_STEER_RAD
    env = make_env(EnvConfig(**NO_SETTLE))
    obs, info = env.reset(seed=3)
    st = env.state
    assert obs[2] == pytest.approx(st['vy'] / o.vy_scale_mps, abs=1e-6)
    assert obs[6] == pytest.approx(st['steer_actual'] / o.steer_scale_rad, abs=1e-6)
    assert obs[10] == pytest.approx(info['heading_error'] / o.heading_scale_rad, abs=1e-6)
    for _ in range(10):
        obs, *_ = env.step(np.array([.5, .2, -1.], np.float32))
    st = env.state
    assert abs(st['steer_actual']) > .05 and abs(st['vy']) > 0.
    assert obs[2] == pytest.approx(st['vy'] / 5., rel=1e-5)
    assert obs[6] == pytest.approx(st['steer_actual'] / .6, rel=1e-5)


def test_settle_hands_over_with_its_last_action(env):
    _, info = env.reset(seed=11)
    assert info['start']['settle_s'] == pytest.approx(2.)
    assert not np.array_equal(env.last_action, [0., -1., -1.])
    assert env.time_s == 0. and env.steps == 0
    assert abs(info['e_ref']) < .3
    assert info['speed'] == pytest.approx(info['start']['v0'], rel=.05)


def test_placed_on_reference_without_settle(make_env):
    env = make_env(EnvConfig(**NO_SETTLE))
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        assert abs(info['e_ref']) < .05 and abs(info['heading_error']) < .01
        assert -1. < env.track_position() < 1.
        assert np.isfinite(obs).all()


def test_reward_matches_hand_computation(make_env):
    rw = RewardConfig()
    env = make_env(EnvConfig(**NO_SETTLE))
    env.reset(seed=2)
    last = env.last_action.copy()
    d0 = env._distance()
    a = np.array([.1, .2, -1.], np.float32)
    _, r, _, _, info = env.step(a)
    dt = .02
    du = (a - last) / dt
    expected = (rw.a * (info['progress_m'] - d0) - rw.b * dt * (1 + float(np.sum(np.array(rw.w) * du ** 2)))
                - rw.c * info['e_ref'] ** 2)
    assert r == pytest.approx(expected, abs=1e-9)


def test_cost_components_sum_to_reward(env):
    env.reset(seed=4)
    policy = DemoPolicy(env, .5)
    policy.reset()
    total = 0.
    for _ in range(60):
        _, r, term, trunc, info = env.step(policy())
        c = info['step_cost']
        assert r == pytest.approx(c['progress'] - c['time'] - c['smoothness'] - c['tracking'] - c['penalty'], abs=1e-9)
        total += r
    e = info['episode_cost']
    assert total == pytest.approx(e['progress'] - e['time'] - e['smoothness'] - e['tracking'], abs=1e-6)
    assert e['time'] == pytest.approx(info['time_s'])


def test_observation_finite_and_bounded_on_demo(env):
    obs, _ = env.reset(seed=5)
    policy = DemoPolicy(env, .6)
    policy.reset()
    for _ in range(150):
        obs, _, term, trunc, info = env.step(policy())
        assert np.isfinite(obs).all() and env.observation_space.contains(obs)
        assert info['obs_clipped'] == 0
        assert not (term or trunc)


def test_failure_penalty_formula():
    rw = RewardConfig()
    assert rw.failure_penalty(0., .02) == 50.
    assert rw.failure_penalty(2., .02) == pytest.approx(50. + 30 * .1 * 4)        # 62
    assert rw.failure_penalty(13.8, .02) == pytest.approx(50. + 30 * .1 * 13.8 ** 2)  # ~621
    assert RewardConfig(c=.2).failure_penalty(2., .02) == pytest.approx(50. + 30 * .2 * 4)


def test_off_track_terminates_with_offset_penalty(make_env):
    env = make_env()
    env.reset(seed=1)
    for _ in range(500):
        _, r, term, trunc, info = env.step(np.array([1., 1., -1.], np.float32))
        if term:
            break
    assert info['termination'] == 'off_track' and term and not trunc
    assert info['step_cost']['penalty'] == pytest.approx(env.cfg.reward.failure_penalty(info['e_ref'], .02))
    assert r < -50.
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3, np.float32))


def test_stall_terminates(make_env, track):
    env = make_env(EnvConfig(settle_s=0.))
    env.reset(options={'start_s': 4900., 'start_speed': 3.5})   # main straight
    brake = np.array([0., -1., 1.], np.float32)
    for k in range(200):
        _, r, term, trunc, info = env.step(brake)
        if term or trunc:
            break
    assert info['termination'] == 'stall'
    assert info['time_s'] == pytest.approx(2.0, abs=.3)      # below 3 m/s almost immediately
    assert info['step_cost']['penalty'] == pytest.approx(env.cfg.reward.failure_penalty(info['e_ref'], .02))


def test_wrong_way_terminates(make_env, track):
    env = make_env(EnvConfig(settle_s=0.))
    s = 4900.
    x, y = track.point(s)
    yaw = env.ref.heading(s) + math.pi
    env.reset(options={'start_pose': StartPose(x, y, yaw, 5.)})
    _, r, term, _, info = env.step(np.array([0., -1., -1.], np.float32))
    assert term and info['termination'] == 'wrong_way' and r < -40.


def test_time_limit_truncates(make_env):
    env = make_env(EnvConfig(max_episode_s=.2))
    env.reset(seed=0)
    for k in range(10):
        _, _, term, trunc, info = env.step(np.array([0., -.5, -1.], np.float32))
        if term or trunc:
            break
    assert trunc and not term and info['truncation'] == 'time_limit' and k == 4


def test_training_never_completes_a_lap(make_env):
    env = make_env()
    env.reset(seed=0)
    assert env.progress < -1e6     # base lap-completion check out of reach (R7)


def test_bad_actions_rejected_before_advancing(env):
    env.reset(seed=0)
    for a in ([0., 0.], [float('nan'), 0., 0.], [[0., 0., 0.]]):
        with pytest.raises(ValueError):
            env.step(a)
    assert env.time_s == 0.


def test_flying_lap_timing_two_crossings(make_env):
    """Interpolated crossings and the per-lap split of the tick costs (synthetic ticks)."""
    env = make_env()
    env._lines, env._crossings, env._laps = [10., 110., 210.], [], []
    env._lap_cost = dict.fromkeys(('progress', 'time', 'smoothness', 'tracking', 'penalty'), 0.)
    d, t = 0., 0.
    while len(env._crossings) < 3:
        costs = dict(progress=3., time=.1, smoothness=.01, tracking=.02, penalty=0.)
        env._time_laps(d, d + 3., t, t + .1, costs)
        d, t = d + 3., t + .1
    np.testing.assert_allclose(env._crossings, [10 / 30, 110 / 30, 210 / 30])
    assert len(env._laps) == 2
    for lap in env._laps:
        assert lap['lap_time_s'] == pytest.approx(100 / 30)
        assert lap['time'] == pytest.approx(lap['lap_time_s'])      # b = 1: time term = lap time
        assert lap['progress'] == pytest.approx(100.)
        assert lap['cost'] == pytest.approx(lap['time'] + lap['smoothness'] + lap['tracking'])


def test_evaluation_reset_is_fixed(make_env, track):
    env = make_env()
    _, a = env.reset(seed=1, options={'evaluation': True})
    _, b = env.reset(seed=2, options={'evaluation': True})
    assert a['start'] == b['start'] and a['evaluation']
    assert a['start']['s0'] == pytest.approx(track.length - 200.)
    assert a['start']['fraction'] == .5
    assert env.max_time_s == env.cfg.eval_max_s
    d_line = env._lines[0]
    assert env.progress == pytest.approx(-(d_line + track.length))   # base ends after 2 laps


def test_telemetry_only_when_requested(make_env):
    plain, rec = make_env(), make_env(EnvConfig(record_telemetry=True))
    for env, has in ((plain, False), (rec, True)):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0., 0., -1.], np.float32))
        assert ('telemetry' in info) == has
    assert {'Fz_FL_N', 'Fz_RR_N', 'kappa_FL'} <= set(info['telemetry'])


def test_body_frame_accelerations_on_steady_circle():
    """R4.1: a_y = v r on a steady circle (the naive dv_y/dt would be ~0)."""
    b = OCDBackend(powertrain='engine', max_brake_torque_Nm=12000.)
    st = b.reset(Setup(), StartPose(0., 0., 0., 30.))
    integral = 0.
    for k in range(300):                     # 6 s at 30 m/s, constant steer
        e = 30. - st['vx']
        integral = float(np.clip(integral + e * .02, -5., 5.))
        u = .4 * e + .2 * integral
        prev = (st['vx'], st['vy'], st['yaw_rate'])
        st = b.step(Inputs(.02, min(1., max(0., u)), min(.3, max(0., -u))), 10)
    cur = (st['vx'], st['vy'], st['yaw_rate'])
    ax, ay = body_accelerations(prev, cur, .02)
    v = math.hypot(st['vx'], st['vy'])
    assert v * st['yaw_rate'] > 3.
    assert ay == pytest.approx(v * st['yaw_rate'], rel=.02)
    assert abs((cur[1] - prev[1]) / .02) < .1 * abs(ay)
    assert abs(ax) < .5
    # synthetic: pure rotation at constant body velocity
    assert body_accelerations((20., 1., .5), (20., 1., .5), .04) == pytest.approx((-.5, 10.))
