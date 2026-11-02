"""Geometry, target correspondence and strict evaluation orchestration checks; no full laps."""
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/comparison'))
import tune_tum_ocd as tune
from rlsetup.driving.config import DemoConfig


def square(alpha):
    tum=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    safe=np.array([[0.,0.],[20.,0.],[20.,20.],[0.,20.]])
    return tune.blend_candidate(alpha,safe,tum,np.array([0.,10.,20.,30.]),40.)


def test_blend_recomputes_true_geometry_and_preserves_spatial_speed_correspondence():
    candidate=square(.5)
    assert candidate.line.length == 60.
    np.testing.assert_array_equal(candidate.line.point(15.),[15.,0.])
    # At the same physical corresponding vertex, query TUM station10, rather than
    # passing candidate station15 directly to the original profile.
    profile=SimpleNamespace(speed=lambda station:station+100.)
    station,speed=candidate.target(15.,profile)
    assert station == 10.
    assert speed == 110.
    assert candidate.target(75.,profile) == (station,speed)
    assert candidate.line.path[-1,0] == 60.
    np.testing.assert_array_equal(candidate.line.path[-1,1:],candidate.line.path[0,1:])


def test_geometry_screen_uses_actual_footprint_and_candidate_tangent():
    candidate=square(.5)
    calls=[]
    footprint=[[2.468,1.],[-2.132,-1.]]
    def contains(x,y,yaw,received):
        calls.append((x,y,yaw,received))
        return x <= 14.
    result=tune.screen_candidate(candidate,SimpleNamespace(contains_car=contains),footprint,grid_m=1.)
    assert len(calls)==60
    assert all(call[3] is footprint for call in calls)
    assert result['planned_only'] is True
    assert result['all_sampled_footprints_inside'] is False
    assert result['violating_samples']>0
    assert calls[5][2] == 0.


def test_feasible_geometry_selection_keeps_lowest_three_and_safe_margin():
    candidates=[square(alpha) for alpha in [1.,.4,.2,.3,0.]]
    for candidate in candidates:
        candidate.planned={'all_sampled_footprints_inside':candidate.alpha>0.}
    selected=tune.selected_candidates(candidates)
    assert [candidate.alpha for candidate in selected]==[.2,.3,.4,1.]
    plan=tune.trial_plan(selected,[1.,.85],[.35],[.4],[.2])
    assert [row['alpha'] for row in plan[:4]]==[.2,.3,.4,1.]
    assert [row['factor'] for row in plan]==[1.]*4+[.85]*4


def test_repeated_projection_knots_are_removed_without_zero_length_segments():
    xy=np.array([[0.,0.],[10.,0.],[10.,0.],[10.,10.],[0.,10.]])
    candidate=tune.blend_candidate(0.,xy,xy,np.array([0.,10.,10.,20.,30.]),40.)
    assert candidate.line.length==40.
    assert len(candidate.tum_station)==5
    assert np.all(np.diff(candidate.line.path[:,0])>0.)


def test_reverse_spatial_correspondence_fails_fast():
    xy=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    with pytest.raises(ValueError,match='reverses'):
        tune.blend_candidate(.5,xy,xy,np.array([0.,10.,9.,20.]),40.)


def test_trial_retains_genuine_reset_seed_settle_integral_and_strict_dnf(monkeypatch):
    candidate=square(.5)
    original=DemoConfig()
    params={'alpha':.5,'factor':.85,'lookahead_s':.16,'speed_kp':.2,'speed_ki':.05}
    captured={}
    class Driver:
        def __init__(self,line,cfg,**kwargs):
            captured.update(line=line,config=cfg,kwargs=kwargs)
        def reset(self,value):
            self.integral=value
            captured['initial_integral']=value
        def act(self,state,station,target):
            captured['target']=target
            return np.array([0.,.2,-.8])
    class Env:
        cfg=SimpleNamespace(demo=original,pedal_deadzone=True,car='sf70h_tum')
        decision_dt=.04
        settle_driver=SimpleNamespace(integral=1.23)
        state={'x':5.,'y':0.,'vx':10.}
        rows=[{'on_track':0,'x_m':5.}]
        handover={'reset':'actual'}
        decision=1
        def reset(self,**kwargs):
            captured['reset']=kwargs
        def step(self,action):
            return None,1.,True,False,{'laps':[],'termination':'off_track','truncation':None,
                                      'time_s':.04,'progress_m':.4}
    monkeypatch.setattr(tune,'DemoDriver',Driver)
    monkeypatch.setattr(tune,'check',lambda env,report:{'lap_times_match_env':True})
    env=Env()
    rows,summary=tune.run_trial(env,candidate,SimpleNamespace(speed=lambda station:50.),params,42)
    assert captured['reset']=={'seed':42,'options':{'evaluation':True}}
    assert captured['initial_integral']==1.23
    assert captured['kwargs']['control_dt_s']==.04
    assert captured['config'].brake_cap==1.
    assert captured['target']==42.5
    assert summary['clean'] is False
    assert summary['termination']=='off_track'
    assert summary['lap2_s'] is None
    assert summary['terminal_sample']['on_track']==0
    assert asdict(env.cfg.demo)==asdict(original)
