"""C2a telemetry lap -> style-rl input (spec S3, review R3): time from time_mapping with (0 m, 0 s)
prepended, 25 Hz uniform resample, steer x 115, lap QC codes."""
import numpy as np
import pytest
from rlsetup.critic.config import LAP_NO_SAMPLES, LAP_SPACING, LAP_TIME_CLOSURE, OK, StyleInputConfig
from rlsetup.critic.sources.base import resample_uniform
from rlsetup.critic.sources.f1laps import STEER_DEG_PER_UNIT, TelemetryLap, lap_style_input

V = 50.                             # constant speed [m/s]


def lap(spacing=3., L=1000., v=V, time_scale=1., steer=.5, lap_time=19.9, drop=None):
    d = np.arange(1., L, spacing)
    ch = {'steer': np.column_stack((d, np.full_like(d, steer))), 'throttle': np.column_stack((d, d / L)),
          'brake': np.column_stack((d, 1 - d / L))}
    if drop:
        ch.pop(drop)
    keys = np.arange(20., L, 10.)
    return TelemetryLap(game='F1 26', track='t', driver=0, lap_time_s=lap_time, controller='Wheel', assists={},
                        date_days=0, channels=ch, time_map=np.column_stack((keys, keys / v * 1000. * time_scale)))


def test_time_reconstruction_with_zero_prepended():
    si = lap_style_input(lap())
    assert si.reason == OK
    # linear in distance through (0 m, 0 s): exact for constant speed, also before the first key (20 m)
    # and after the last key (990 m, continued at the last interval's speed)
    np.testing.assert_allclose(si.distance_m, V * si.time_s, atol=1e-9)
    assert si.time_s[0] == pytest.approx(1. / V)            # first sample at 1 m, not clamped to 0.4 s
    assert si.t_end_s == pytest.approx(990. / V)
    assert si.distance_m[-1] > 990.


def test_uniform_25hz_resample():
    si = lap_style_input(lap())
    np.testing.assert_allclose(np.diff(si.time_s), 1 / 25, atol=1e-12)
    np.testing.assert_allclose(si.throttle, si.distance_m / 1000., atol=1e-9)    # linear in t -> exact
    np.testing.assert_allclose(si.throttle + si.brake, 1., atol=1e-9)
    grid, u = resample_uniform(np.array([0., 1., 2.]), {'a': np.array([0., 2., 4.])}, 4.)
    np.testing.assert_allclose(grid, np.arange(8) / 4)
    np.testing.assert_allclose(u['a'], 2 * grid)


def test_steering_scale():
    assert STEER_DEG_PER_UNIT == 115.
    si = lap_style_input(lap(steer=-.5))
    np.testing.assert_allclose(si.steer_deg, -57.5)
    np.testing.assert_allclose(si.raw_steer_deg, -57.5)


@pytest.mark.parametrize('kw, reason', [({'time_scale': 1.03}, LAP_TIME_CLOSURE),
                                        ({'spacing': 9.}, LAP_SPACING),
                                        ({'drop': 'brake'}, LAP_NO_SAMPLES),
                                        ({'lap_time': None}, LAP_NO_SAMPLES)])
def test_lap_qc(kw, reason):
    si = lap_style_input(lap(**kw))
    assert si.reason == reason and not si.ok
    assert len(si.time_s) == 0


def test_qc_thresholds_are_config():
    si = lap_style_input(lap(spacing=9.), StyleInputConfig(max_median_spacing_m=10.))
    assert si.ok and si.median_spacing_m == pytest.approx(9., abs=1e-9)
    assert lap_style_input(lap(time_scale=1.03), StyleInputConfig(max_time_closure_rel=.05)).ok


def test_spacing_recorded_when_time_map_missing():
    l = lap(spacing=12.)
    l.time_map = None
    si = lap_style_input(l)
    assert si.reason == LAP_NO_SAMPLES and si.median_spacing_m == pytest.approx(12.)


def test_digest_tracks_the_samples():
    a, b = lap(), lap(steer=.4)
    assert lap_style_input(a).digest == lap_style_input(lap()).digest != lap_style_input(b).digest


def test_distance_ties_keep_first_and_are_counted():
    """T2 (QA M5): of samples tied in distance the first is kept; the number dropped is on StyleInput."""
    from rlsetup.critic.sources.f1laps import _increasing
    a = np.array([[1, .1], [2, .2], [2, .9], [3, .3], [4, .4]])
    kept, n = _increasing(a)
    np.testing.assert_array_equal(kept[:, 0], [1, 2, 3, 4])
    assert kept[1, 1] == pytest.approx(.2) and n == 1
    lp = lap()
    d = lp.channels['steer']
    lp.channels['steer'] = np.insert(d, 2, [d[1, 0], .9], axis=0)        # one tie at the 2nd sample's distance
    si = lap_style_input(lp)
    assert si.reason == OK and si.n_dropped_ties == 1 and si.dropped_ties['steer'] == 1
    assert si.dropped_ties['throttle'] == 0 and si.dropped_ties['time_map'] == 0
    np.testing.assert_array_equal(si.raw_steer_deg, d[:, 1] * STEER_DEG_PER_UNIT)
