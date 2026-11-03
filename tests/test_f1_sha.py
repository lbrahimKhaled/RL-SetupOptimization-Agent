"""Car sf70h_f1 support: steering scaled to the car's lock, named setup p1_TUM_f1, preset f1_sha.

The sf70h_f1 tests need ocd_first_lap branch sf70h-f1 on the path
(PYTHONPATH=<ocd_first_lap-f1 worktree>); on other checkouts they are skipped and the
main-checkout behaviour (unknown car) is checked instead.
"""
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from ocd_lap import cars
from ocd_lap.env import MAX_STEER_RAD, OCDRacingEnv
from rlsetup.driving import qss_profile, setups
from rlsetup.driving.config import DemoConfig, EnvConfig
from rlsetup.driving.demo_driver import DemoDriver, action_to_inputs, inputs_to_action, steer_lock_rad
from rlsetup.driving.env import DrivingEnv
from rlsetup.driving.train_config import PRESETS, config_json, train_config_from_dict

HAS_F1 = 'sf70h_f1' in cars.CARS
needs_f1 = pytest.mark.skipif(not HAS_F1, reason='sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path')


def test_existing_cars_keep_the_0_6_rad_scale():
    for car in ('av21', 'sf70h_tum'):
        assert steer_lock_rad(car) == MAX_STEER_RAD == .6
    assert steer_lock_rad(SimpleNamespace()) == MAX_STEER_RAD          # CarSpec without the field
    for a in np.random.default_rng(1).uniform(-1, 1, (20, 3)).astype(np.float32):
        assert action_to_inputs(a.astype(float), False) == OCDRacingEnv.to_inputs(a)
        assert action_to_inputs(a.astype(float), True) == action_to_inputs(a.astype(float), True, MAX_STEER_RAD)


def test_env_steering_scale_for_existing_car(env):
    assert env.steer_max_rad == .6
    assert env.to_inputs(np.array([1., -1., -1.])).steering_rad == .6
    assert env.to_inputs(np.array([-3., -1., -1.])).steering_rad == -.6   # clipped action


def test_steering_mapping_with_another_lock():
    lock = .3
    assert action_to_inputs(np.array([1., 0., 0.]), True, lock).steering_rad == lock
    assert action_to_inputs(np.array([-.5, 0., 0.]), False, lock).steering_rad == -.15
    a = inputs_to_action(.15, .2, 0., True, steer_max_rad=lock)
    assert a[0] == pytest.approx(.5)
    assert inputs_to_action(.45, 0., 0., True, steer_max_rad=lock)[0] == 1.     # beyond the lock: clipped


def test_demo_driver_clips_to_the_car_lock():
    class Line:            # a goal far to the left: pure pursuit asks for more than any lock
        def point(self, s):
            return np.array([0., 5.])
    spec = replace(cars.get('sf70h_tum'))
    object.__setattr__(spec, 'steer_max_rad', .3)     # a CarSpec with a 0.30 rad lock on any checkout
    d = DemoDriver(Line(), DemoConfig(), pedal_deadzone=True, control_dt_s=.04, car=spec)
    state = dict(x=0., y=0., yaw=0., vx=10., slip_rl=0., slip_rr=0.)
    assert d.steering(state, 0.) == pytest.approx(.3)
    assert d.act(state, 0., 10.)[0] == pytest.approx(1.)
    wide = DemoDriver(Line(), DemoConfig(), pedal_deadzone=True, control_dt_s=.04, car='sf70h_tum')
    assert .3 < wide.steering(state, 0.) <= .6


def test_traction_control_new_window():
    """f1_sha: throttle untouched up to slip 0.06, linear cut, zero from 0.09."""
    cfg = PRESETS['f1_sha']().env.demo
    d = DemoDriver(None, cfg, pedal_deadzone=True, control_dt_s=.04)
    full = d.pedals(100., 0., 0.)[0]
    assert full == 1.
    for slip, factor in ((.06, 1.), (.075, .5), (.09, 0.), (.2, 0.)):
        d.reset()
        assert d.pedals(100., 0., slip)[0] == pytest.approx(factor)


def test_f1_sha_preset_is_p1_tum_with_the_new_car():
    c, p1 = PRESETS['f1_sha'](), PRESETS['p1_TUM']()
    assert (c.preset, c.env.car, c.env.setup) == ('f1_sha', 'sf70h_f1', 'p1_TUM_f1')
    assert (c.env.demo.traction_slip_start, c.env.demo.traction_slip_span) == (.06, .03)
    assert replace(c, preset=p1.preset, env=replace(c.env, car=p1.env.car, setup=p1.env.setup,
                                                    demo=p1.env.demo)) == p1
    assert (c.env.ers, c.env.max_brake_torque_Nm, c.env.reference, c.env.reward.mode) == (False, 15000., 'path', 'time_pbrs')
    assert (c.demos.fraction_range, c.bc.gate_required, c.max_wall_h) == ((.5, .8), False, None)
    assert c.env.obs.steer_scale_rad == .6                                # kept (config.py ObsConfig)
    assert train_config_from_dict(json.loads(config_json(c))) == c
    assert PRESETS['p1_TUM']().env.demo == DemoConfig()                   # other presets unchanged


@pytest.mark.skipif(HAS_F1, reason='checks the checkouts without sf70h_f1')
def test_p1_tum_f1_needs_the_f1_checkout():
    with pytest.raises(ValueError, match='sf70h_f1'):
        setups.get('p1_TUM_f1')


@needs_f1
def test_p1_tum_f1_setup():
    s, p1 = setups.get('p1_TUM_f1'), setups.get('p1_TUM')
    assert type(s) is cars.SF70HF1Setup
    diff = {k: v for k, v in s.to_dict().items() if v != p1.to_dict()[k]}
    assert set(diff) == {'heave_freq_front_hz', 'heave_freq_rear_hz', 'heave_damping_ratio', 'brake_bias_front'}
    assert (diff['heave_damping_ratio'], diff['brake_bias_front']) == (.3, .58)
    assert diff['heave_freq_front_hz'] == pytest.approx(12.9456, abs=1e-3)
    assert diff['heave_freq_rear_hz'] == pytest.approx(5.8451, abs=1e-3)
    assert setups.get('p1_TUM_f1') == s


@needs_f1
def test_sha_f1_best_setup():
    s, base = setups.get('sha_f1_best'), setups.get('p1_TUM_f1')
    assert type(s) is cars.SF70HF1Setup
    diff = {k: v for k, v in s.to_dict().items() if v != base.to_dict()[k]}
    assert diff == {'aero_front_share': .5384, 'roll_stiffness_front_share': .40, 'roll_stiffness_Nm_per_rad': 4.0e5}


@needs_f1
def test_f1_env_steers_to_0_30_rad(track):
    cfg = PRESETS['f1_sha']().env
    qss = qss_profile.for_env(track, cfg)
    env = DrivingEnv(cfg, track=track, qss=qss)
    assert env.steer_max_rad == .3 == env.backend.base_params['steering_actuator.angle_max_rad']
    assert type(env.setup) is cars.SF70HF1Setup and env.setup.heave_damping_ratio == .3
    assert env.to_inputs(np.array([1., -1., -1.])).steering_rad == .3
    assert env.to_inputs(np.array([-2., -1., -1.])).steering_rad == -.3
    assert env.settle_driver.steer_max_rad == .3
    obs, _ = env.reset(seed=0)
    for _ in range(50):                    # hold full left lock: actual steer obs -> 0.3 / 0.6 = 0.5
        obs, *_ , info = env.step(np.array([1., -.2, -1.], np.float32))
        if info['termination']:
            break
    assert env.state['steer_actual'] <= .3 + 1e-9 and obs[6] <= .5 + 1e-6
    assert 100. < qss.lap_time_s < 105.    # QSS reference 102.53 s (SF70H_F1_UPGRADE.md §3)
