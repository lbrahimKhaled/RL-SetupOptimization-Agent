"""Fast checks of source identity and exact lap clipping for the offline ghost video."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'comparison' / 'ghost_video_p1.py'
spec = importlib.util.spec_from_file_location('ghost_video_p1', SCRIPT)
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


def test_exact_crossing_pose_uses_bracketing_samples_and_finishes_once():
    data = {'t_s':np.array([9.,10.,11.,12.]), 'x_m':np.array([0.,10.,20.,30.]),
            'y_m':np.array([0.,2.,4.,6.]), 'speed_mps':np.ones(4)*10,
            'distance_m':np.array([0.,10.,20.,30.]),
            'yaw_rad':np.radians([178.,179.,-179.,-178.]),
            'on_track':np.array([1.,1.,0.,1.]), 'throttle':np.array([.1,.2,.3,.4])}
    lap = video.clip_trace(data,9.5,11.5)
    np.testing.assert_array_equal(lap['t'],[0.,.5,1.5,2.])
    np.testing.assert_array_equal(lap['x_m'],[5.,10.,20.,25.])
    state = video.state_at(lap,20.)
    assert state['finished'] is True
    assert state['x_m'] == 25.
    assert np.degrees(state['yaw_rad']) == pytest.approx(181.5)
    assert video.state_at(lap,1.)['finished'] is False
    # Interval-end samples identify the actual held command/violation; never blend flags.
    assert video.state_at(lap,1.)['on_track'] == 0.
    assert video.state_at(lap,1.)['throttle'] == .3


def metadata():
    common = {'car':'sf70h_tum','setup':{'spring':4.2},'max_brake_torque_Nm':15000.,
              'footprint_m':[[2.,1.]],'physics_dt_s':.002,'ers':False,'drs':False,'recuperation':False}
    rl = {'source':'rl','episode':{'kind':'deterministic','clean':True,'termination':'lap_complete',
                                  'reproduces_training_eval':True},
          'saved_run_config':{'env':{'setup':'p1_TUM'}},'conditions':dict(common,decision_dt_s=.04)}
    pp = {'conditions':dict(common,setup_name='p1_TUM'),
          'controller':{'control_dt_s':.04,'speed_scale_f':1.,'weights_loaded':False},
          'summary':{'completed_lap2':True},'stop_rules':{'footprint_exit_terminates':False},
          'valid_fair_RL_comparison':False,'tum_input':{'sha256':'abc'}}
    tum = {'source':'tum','variant':'none','ers_used_J':0.,'drs_share_of_points':0.,'sha256':'abc'}
    return rl,pp,tum


def test_matching_original_fullspeed_diagnostic_is_accepted():
    video.validate_metadata(*metadata())


@pytest.mark.parametrize('mutate',[
    lambda rl,pp,tum: pp['controller'].update(speed_scale_f=.7),
    lambda rl,pp,tum: pp['conditions'].update(max_brake_torque_Nm=12000.),
    lambda rl,pp,tum: rl['episode'].update(clean=False),
    lambda rl,pp,tum: pp['tum_input'].update(sha256='different'),
    lambda rl,pp,tum: pp['controller'].update(weights_loaded=True),
])
def test_invalid_or_mismatched_source_fails_fast(mutate):
    inputs = copy.deepcopy(metadata()); mutate(*inputs)
    with pytest.raises(ValueError):
        video.validate_metadata(*inputs)


def test_modified_csv_rejected_before_loading(tmp_path):
    path = tmp_path/'trace.csv'; path.write_text('t_s,x_m\n0,1\n')
    path.with_suffix('.meta.json').write_text(json.dumps({'file':'trace.csv','sha256':'wrong'}))
    with pytest.raises(ValueError,match='hash'):
        video.verified_input(path)


def test_missing_crossing_bracket_fails_fast():
    with pytest.raises(ValueError,match='bracket'):
        video.clip_trace({'t_s':np.array([1.,2.]),'x_m':np.array([3.,4.])},0.,1.5)
