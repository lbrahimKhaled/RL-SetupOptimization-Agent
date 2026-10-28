"""Named setups (rlsetup/driving/setups.py), EnvConfig.setup / .reference and presets p1_TUM, p0_TUMref."""
import json
import pytest
from ocd_lap.car_setup import Setup
from rlsetup.driving import qss_profile, setups
from rlsetup.driving.config import EnvConfig
from rlsetup.driving.train_config import PRESETS, config_json, train_config_from_dict


def test_p1_tum_changes_exactly_the_tum_knobs():
    p0, p1 = setups.get('p0').to_dict(), setups.get('p1_TUM').to_dict()
    assert setups.get('p0') == Setup()
    assert {k: v for k, v in p1.items() if v != p0[k]} == {
        'roll_stiffness_front_share': .4533, 'aero_front_share': .4508, 'brake_bias_front': .49, 'lsd_lock_drive': 1.}
    with pytest.raises(KeyError):
        setups.get('p2')


def test_presets():
    p1, ref = PRESETS['p1_TUM'](), PRESETS['p0_TUMref']()
    assert (p1.env.car, p1.env.ers, p1.env.setup, p1.env.reference, p1.env.max_brake_torque_Nm) == \
        ('sf70h_tum', False, 'p1_TUM', 'path', 15000.)
    assert (ref.env.car, ref.env.ers, ref.env.setup, ref.env.reference, ref.env.max_brake_torque_Nm) == \
        ('sf70h_tum', False, 'p0', 'raceline', 12000.)
    for c in (p1, ref):
        assert c.env.reward.mode == 'time_pbrs' and c.demos.fraction_range == (.5, .8) and not c.bc.gate_required
        assert train_config_from_dict(json.loads(config_json(c))) == c


def test_old_config_json_means_p0_and_safe_line():
    raw = json.loads(config_json(PRESETS['full']()))
    del raw['env']['setup'], raw['env']['reference']
    env = train_config_from_dict(raw).env
    assert (env.setup, env.reference) == ('p0', 'path') == (EnvConfig().setup, EnvConfig().reference)


def test_qss_cache_unchanged_for_p0_and_12_kNm(track):
    assert qss_profile.default_path('Shanghai').name == 'qss_profile_Shanghai_p0.npz'
    assert qss_profile.default_path('Shanghai', 'p0', car='sf70h_tum').name == 'qss_profile_Shanghai_p0_sf70h_tum.npz'
    assert qss_profile.default_path('Shanghai', 'p1_TUM', car='sf70h_tum', brake_torque_Nm=15000.).name == \
        'qss_profile_Shanghai_p1_TUM_sf70h_tum_brake15000.npz'
    assert 'brake_torque_Nm' not in qss_profile.cache_key(track, Setup(), 'sf70h_tum')
    assert qss_profile.cache_key(track, Setup(), 'sf70h_tum', brake_torque_Nm=15000.)['brake_torque_Nm'] == 15000.


def test_env_setup_and_reference(make_env):
    env = make_env(EnvConfig(setup='p1_TUM'))
    assert env.setup == setups.get('p1_TUM') and env.drive_line is env.ref and env.ref.name.endswith('Track.path')
    raceline = make_env(EnvConfig(reference='raceline'))
    assert raceline.ref.name.endswith('Track.reference') and raceline.drive_line.name.endswith('Track.path')
    assert raceline.env_metadata()['reference']['name'].endswith('Track.reference')
    raceline.reset(seed=3)
    assert raceline.s_drive == pytest.approx(raceline.s % raceline.track.length)
