import warnings
import numpy as np
import pytest
from ocd_lap import qss_lap
from ocd_lap.car_setup import Setup
from rlsetup.driving import qss_profile


@pytest.fixture(scope='module')
def computed(track):
    return qss_profile.compute(track)   # measures the p0 g-g-v (~0.4 s)


def test_profile_reproduces_qss_lap_time_exactly(track, computed):
    """R5: calc_t_profile of our v(s) == qss_lap.lap_time with the same g-g-v data."""
    profile, ggv_data = computed
    assert profile.lap_time_s == qss_lap.lap_time(Setup(), track, ggv_data=ggv_data)['lap_time_s']
    assert profile.lap_time_s == pytest.approx(133.9, abs=.05)   # DECISIONS #9


def test_cached_file_matches_fresh_computation(track, qss, computed):
    profile, _ = computed
    assert qss.meta['key'] == qss_profile.cache_key(track, Setup())
    np.testing.assert_array_equal(qss.v, profile.v)
    assert qss.lap_time_s == profile.lap_time_s
    assert {'ocd_first_lap', 'rlsetup', 'ocd_lap_code_version'} <= set(qss.meta['provenance'])


def test_speed_lookup_and_window_min(qss):
    L = qss.length
    assert qss.speed(L + 10.) == pytest.approx(qss.speed(10.))
    s0 = 100.
    window = qss.speed(s0 + np.linspace(0., 300., 301))
    assert qss.min_speed(s0, 300.) == pytest.approx(window.min())
    assert 9. < qss.v.min() < 11. and 79. < qss.v.max() < 81.


def test_stale_cache_is_recomputed_not_rewritten(tmp_path, track, qss):
    path = tmp_path / 'p.npz'
    stale = qss_profile.QSSProfile(qss.s, qss.v * 0 + 1., 1., dict(qss.meta, key={'stale': True}))
    qss_profile.save(stale, path)
    before = path.read_bytes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fresh = qss_profile.load_or_build(track, path)
    assert any('cache key' in str(w.message) for w in caught)
    assert fresh.lap_time_s == pytest.approx(qss.lap_time_s)
    assert path.read_bytes() == before
