import numpy as np
import pytest
from ocd_lap import manoeuvres
from ocd_lap.env import OCDRacingEnv
from rlsetup.driving.config import DemoConfig
from rlsetup.driving.demo_driver import (DemoDriver, action_to_inputs, inputs_to_action,
                                         pedal_to_action)
from rlsetup.driving.reference import ReferenceLine


@pytest.mark.parametrize('a, pedal', [(-1., 0.), (-.5, 0.), (-.8, 0.), (.25, .5), (1., 1.), (0., 1 / 3)])
def test_deadzone_mapping(a, pedal):
    inputs = action_to_inputs(np.array([0., a, a]), True)
    assert inputs.throttle == pytest.approx(pedal) and inputs.brake == pytest.approx(pedal)


def test_no_deadzone_mapping_equals_ocd_env():
    for a in np.random.default_rng(0).uniform(-1, 1, (50, 3)).astype(np.float32):
        assert action_to_inputs(a.astype(float), False) == OCDRacingEnv.to_inputs(a)


@pytest.mark.parametrize('deadzone', [True, False])
def test_inverse_mapping_round_trip(deadzone):
    for pedal in (0., .01, .3, .77, 1.):
        a = pedal_to_action(pedal, deadzone)
        assert action_to_inputs(np.array([0., a, a]), deadzone).throttle == pytest.approx(pedal, abs=1e-12)
    assert pedal_to_action(0., True) == -.8
    a = inputs_to_action(.3, .5, 0., True)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a, [.5, .25, -.8], atol=1e-7)


def test_speed_pi_uses_manoeuvres_gains():
    """Same PI law as ocd_lap.manoeuvres._speed_pedals (catches upstream changes)."""
    cfg = DemoConfig()
    d = DemoDriver(None, cfg, pedal_deadzone=True, control_dt_s=.04)
    for integral0, target, vx in ((0., 30., 28.), (3., 30., 31.), (-4.9, 10., 30.), (4.9, 50., 20.)):
        d.reset(integral0)
        throttle, brake = d.pedals(target, vx)
        integral = float(np.clip(integral0 + (target - vx) * .04, -5., 5.))
        t_ref, b_ref = manoeuvres._speed_pedals(target - vx, integral)
        assert throttle == pytest.approx(t_ref) and brake == pytest.approx(min(b_ref, cfg.brake_cap))


def test_traction_control_scales_throttle():
    d = DemoDriver(None, DemoConfig(), pedal_deadzone=True, control_dt_s=.04)
    full, _ = d.pedals(80., 10.)
    d.reset()
    half, _ = d.pedals(80., 10., rear_slip=.14)
    d.reset()
    none, _ = d.pedals(80., 10., rear_slip=.3)
    assert full == 1. and half == pytest.approx(.5) and none == 0.


def test_pure_pursuit_steers_toward_line(track):
    ref = ReferenceLine.from_track_path(track)
    d = DemoDriver(ref, DemoConfig(), pedal_deadzone=True, control_dt_s=.04)
    s = 500.
    p, t = ref.frame(s)
    left = np.array([-t[1], t[0]])
    yaw = np.arctan2(t[1], t[0])
    for offset, sign in ((2., -1.), (-2., 1.)):          # left of the line -> steer right
        x, y = p + offset * left
        steer = d.steering(dict(x=x, y=y, yaw=yaw, vx=20.), ref.match((x, y), s))
        assert np.sign(steer) == sign
