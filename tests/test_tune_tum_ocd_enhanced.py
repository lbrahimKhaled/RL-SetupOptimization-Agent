"""Signs, closed speed-envelope constraints and global sampled brake assistance."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/comparison'))
import tune_tum_ocd_enhanced as enhanced
import tune_tum_ocd as base
from rlsetup.driving.config import DemoConfig
from rlsetup.driving.demo_driver import action_to_inputs
from rlsetup.driving.reference import ReferenceLine


def circle(radius=100.):
    theta=np.linspace(0,2*np.pi,720,endpoint=False)
    xy=np.column_stack([radius*np.cos(theta),radius*np.sin(theta)])
    line=ReferenceLine.from_xy(xy)
    return base.Candidate(1.,line,np.linspace(0,line.length,len(line.path)),{})


def test_curvature_sign_and_lateral_limit_are_periodic():
    candidate=circle()
    envelope=enhanced.SpeedEnvelope(candidate,SimpleNamespace(speed=lambda s:60.),
        [[0.,8.,10.],[100.,8.,10.]],[[0.,5.],[100.,5.]],grid_m=2.)
    assert np.median(envelope.kappa)==pytest.approx(.01,rel=.02)
    assert np.all(envelope.v<=60.)
    assert np.max(envelope.v**2*np.abs(envelope.kappa))<=10.+1e-6
    assert envelope.at(-1.)==pytest.approx(envelope.at(envelope.length-1.))
    assert abs(envelope.kappa[0]-np.median(envelope.kappa))<.001


def test_feasible_TUM_targets_remain_unchanged():
    envelope=enhanced.SpeedEnvelope(circle(),SimpleNamespace(speed=lambda s:20.),
        [[0.,8.,10.],[100.,8.,10.]],[[0.,5.],[100.,5.]],grid_m=2.)
    np.testing.assert_array_equal(envelope.v,envelope.tum)
    assert envelope.meta['fraction_reduced_from_TUM']==0.


def test_backward_braking_carries_a_low_speed_target_through_timing_line():
    candidate=circle(500.)
    profile=SimpleNamespace(speed=lambda s:10. if s<20. else 50.)
    envelope=enhanced.SpeedEnvelope(candidate,profile,[[0.,4.,100.],[100.,4.,100.]],
                                     [[0.,4.],[100.,4.]],grid_m=4.)
    # A low target immediately after the line imposes braking before the line.
    assert envelope.v[-1]<15.
    assert np.max(envelope.v)>40.
    following=np.roll(envelope.v,-1)
    assert np.all(envelope.v**2<=following**2+2*4.*envelope.ds+1e-5)


def driver(feedback):
    line=ReferenceLine.from_xy([[0.,0.],[100.,0.],[100.,100.],[0.,100.]])
    envelope=SimpleNamespace(at=lambda s:(50.,0.))
    return enhanced.FeedbackDriver(line,DemoConfig(brake_cap=1.),feedback,envelope,
                                   pedal_deadzone=True,control_dt_s=.04,car='sf70h_tum')


def state(**changes):
    value={'x':50.,'y':0.,'yaw':0.,'vx':30.,'vy':0.,'yaw_rate':0.,'slip_rl':0.,'slip_rr':0.}
    value.update(changes);return value


def telemetry(kappa=0.):
    return {f'kappa_{w}':kappa for w in ['FL','FR','RL','RR']}


def test_cross_heading_and_yaw_feedback_have_stabilizing_signs():
    controller=driver(enhanced.FeedbackConfig(understeer_gain=0.,beta_gain=0.))
    left=controller.command(state(y=1.),50.,30.,telemetry())
    heading=controller.command(state(yaw=.1),50.,30.,telemetry())
    yaw=controller.command(state(yaw_rate=.2),50.,30.,telemetry())
    assert left[0]<0.
    assert heading[0]<0.
    assert yaw[0]<0.
    centered=controller.command(state(),50.,30.,telemetry())
    assert centered[0]==0.


def test_optional_global_brake_limiter_releases_at_one_decision_rate():
    controller=driver(enhanced.FeedbackConfig(slip_limit=.12))
    raw=driver(enhanced.FeedbackConfig(slip_limit=0.))
    limited=action_to_inputs(controller.command(state(),50.,0.,telemetry(-.5)),True)
    unlimited=action_to_inputs(raw.command(state(),50.,0.,telemetry(-.5)),True)
    assert controller.brake_ceiling==pytest.approx(1.-8.*.04)
    assert limited.brake==pytest.approx(.68,abs=1e-6)
    assert unlimited.brake==pytest.approx(1.)
    assert controller.diagnostics['measured_braking_slip']==.5
    assert limited.throttle==0.


def test_controller_gains_fail_fast_outside_explicit_bounds():
    with pytest.raises(ValueError,match='yaw_gain'):
        enhanced.FeedbackConfig(yaw_gain=-1.)
