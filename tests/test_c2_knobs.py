"""C2a game -> OCD knob map (spec S4, review R1): known values, K_b >= 0, box check and stop rule,
toe signs (label-supported), car constants vs the sf70h_f1 worktree."""
import numpy as np
import pytest
from rlsetup.critic.config import CarConfig, DataConfig, KNOB_ORDER
from rlsetup.critic.knobs import bar_roll_rates, knob_bounds, knob_feasibility, normalise, denormalise, out_of_bounds, \
    setup_hash, wheel_rate
from rlsetup.critic.sources.f1_game_map import BACINGER_IDS, F1GameMapConfig, GAME_COLUMNS, TRACK_GEOMETRY, box_check, \
    game_to_knobs

BASE = {'front_wing': 20, 'rear_wing': 30, 'differential_on_throttle_pct': 50, 'differential_off_throttle_pct': 60,
        'front_camber_deg': -3.0, 'rear_camber_deg': -1.5, 'front_toe_deg': .05, 'rear_toe_deg': .2,
        'front_suspension': 21, 'rear_suspension': 1, 'front_anti_roll_bar': 21, 'rear_anti_roll_bar': 1,
        'front_ride_height': 22, 'rear_ride_height': 45, 'front_brake_bias_pct': 56}


def knobs_of(**over):
    row = {**BASE, **over}
    return dict(zip(KNOB_ORDER, game_to_knobs({k: np.array([row[k]], float) for k in GAME_COLUMNS})[0]))


def test_known_values():
    k = knobs_of()
    car = CarConfig()
    assert k['ride_freq_front_hz'] == pytest.approx(3.75) and k['ride_freq_rear_hz'] == pytest.approx(2.5)
    # K_phi = sum over axles of K_b + 1/2 K_w t^2, K_w = m_c (2 pi f)^2 (hand-written, not via roll_from_bars)
    front = 260_000. + .5 * car.corner_sprung_mass_front_kg * (2 * np.pi * 3.75) ** 2 * 1.6 ** 2
    rear = 10_000. + .5 * car.corner_sprung_mass_rear_kg * (2 * np.pi * 2.5) ** 2 * 1.6 ** 2
    assert k['roll_stiffness_Nm_per_rad'] == pytest.approx(front + rear)
    assert k['roll_stiffness_front_share'] == pytest.approx(front / (front + rear))
    assert k['ride_height_front_m'] == pytest.approx(.027) and k['rake_m'] == pytest.approx(.023)
    assert k['aero_level'] == pytest.approx(.5)
    assert k['aero_front_share'] == pytest.approx(.35 + .25 * (21 / 52 - 1 / 52) / (50 / 52))
    assert k['brake_bias_front'] == pytest.approx(.56)
    assert (k['lsd_lock_drive'], k['lsd_lock_coast']) == pytest.approx((.5, .6))
    assert (k['camber_front_deg'], k['camber_rear_deg']) == (-3.0, -1.5)


def test_toe_signs_label_supported():
    """F1 25 menu: 'Front Toe-Out' / 'Rear Toe-In'; OCD toe positive = toe-out (ocd_lap.car_setup)."""
    from ocd_lap.car_setup import KNOB
    assert 'toe-out' in KNOB['toe_front_deg'].note and 'toe-out' in KNOB['toe_rear_deg'].note
    k = knobs_of(front_toe_deg=.1, rear_toe_deg=.2)
    assert k['toe_front_deg'] == pytest.approx(.1) and k['toe_rear_deg'] == pytest.approx(-.2)


def test_click_ends_and_kb_round_trip():
    cfg = F1GameMapConfig()
    rng = np.random.default_rng(0)
    n = 500
    game = {c: np.full(n, float(BASE[c])) for c in GAME_COLUMNS}
    for c, (lo, hi) in {'front_suspension': (1, 41), 'rear_suspension': (1, 41), 'front_anti_roll_bar': (1, 21),
                        'rear_anti_roll_bar': (1, 21)}.items():
        game[c] = rng.integers(lo, hi + 1, n).astype(float)
    x = game_to_knobs(game)
    assert x[:, 0].min() >= cfg.freq_base_hz and x[:, 0].max() <= cfg.freq_base_hz + cfg.freq_span_hz
    kb_f, kb_r = bar_roll_rates(normalise(x))           # the frequency mediator inverts exactly
    np.testing.assert_allclose(kb_f, 10_000 + 12_500 * (game['front_anti_roll_bar'] - 1), rtol=1e-9)
    np.testing.assert_allclose(kb_r, 10_000 + 12_500 * (game['rear_anti_roll_bar'] - 1), rtol=1e-9)
    assert (knob_feasibility(normalise(x)) >= 10_000 * (1 - 1e-9)).all()   # K_b >= 0 by construction


def test_normalise_round_trip_and_bounds():
    lo, hi = knob_bounds()
    x = (lo + hi) / 2
    np.testing.assert_allclose(normalise(x), 0, atol=1e-12)
    np.testing.assert_allclose(denormalise(normalise(lo)), lo)
    bad = np.vstack((x, x))
    bad[1, KNOB_ORDER.index('roll_stiffness_front_share')] = hi[KNOB_ORDER.index('roll_stiffness_front_share')] + .01
    assert out_of_bounds(bad).any(axis=1).tolist() == [False, True]


def test_box_check_drops_and_stop_rule():
    lo, hi = knob_bounds()
    x = np.tile((lo + hi) / 2, (100, 1))
    x[:3, KNOB_ORDER.index('roll_stiffness_front_share')] = 2.          # 3 % of game A outside
    groups = np.array(['A'] * 100, object)
    keep, rep = box_check(x, groups, DataConfig(), log=lambda *a: None)
    assert rep['dropped'] == 3 and keep.sum() == 97 and rep['per_knob'] == {'roll_stiffness_front_share': 3}
    x[:6, KNOB_ORDER.index('rake_m')] = -1.                             # 6 % now: per-game stop rule
    with pytest.raises(ValueError, match='stop rule'):
        box_check(x, groups, DataConfig(), log=lambda *a: None)
    x[1, KNOB_ORDER.index('rake_m')] = lo[KNOB_ORDER.index('rake_m')] - 1e-12   # within tol: kept
    assert not out_of_bounds(x[1:2, :], DataConfig().bounds_tol)[0, KNOB_ORDER.index('rake_m')]


def test_stop_rule_is_per_game():
    """T3 (QA M8): game A 6 of 100 out, game B 0 of 1000 -> pooled 0.5 % but A alone is over 5 %: stop."""
    lo, hi = knob_bounds()
    x = np.tile((lo + hi) / 2, (1100, 1))
    x[:6, KNOB_ORDER.index('rake_m')] = -1.
    groups = np.array(['A'] * 100 + ['B'] * 1000, object)
    with pytest.raises(ValueError, match="'A'") as e:
        box_check(x, groups, DataConfig(), log=lambda *a: None)
    assert "'B'" not in str(e.value)


def test_setup_hash_ignores_negative_zero():
    a = np.zeros((1, 15))
    b = a.copy()
    b[0, 3] = -0.0
    assert setup_hash(a)[0] == setup_hash(b)[0]
    assert setup_hash(a + 1e-10)[0] == setup_hash(a)[0]


def test_track_maps_cover_every_game_track():
    assert len(TRACK_GEOMETRY) == 25
    assert set(BACINGER_IDS) <= set(TRACK_GEOMETRY.values())
    assert len(BACINGER_IDS) == 14


def test_car_constants_match_the_f1_worktree():
    from ocd_lap import cars
    if 'sf70h_f1' not in cars.CARS:
        pytest.skip('sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path (PYTHONPATH=<ocd_first_lap-f1>)')
    from ocd_lap.backend import OCDBackend
    from ocd_lap.translator import CarGeometry
    g = CarGeometry.from_params(OCDBackend(car='sf70h_f1').base_params)
    car = CarConfig()
    assert g.corner_sprung_mass_kg == pytest.approx((car.corner_sprung_mass_front_kg, car.corner_sprung_mass_rear_kg))
    assert (g.track_front_m, g.track_rear_m) == pytest.approx((car.track_front_m, car.track_rear_m))


def test_wheel_rate_formula():
    assert wheel_rate(1. / (2 * np.pi), 100.) == pytest.approx(100.)
