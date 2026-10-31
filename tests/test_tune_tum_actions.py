"""Position basis, physical action corrections and clean-only optimization invariants."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/comparison'))
import tune_tum_actions as actions
from rlsetup.driving.demo_driver import inputs_to_action,action_to_inputs


def test_cosine_basis_is_periodic_partition_with_compact_support():
    for station in np.linspace(-100.,200.,701):
        i,j,a,b=actions.basis(station,100.)
        assert 0<=i<64 and j==(i+1)%64
        assert 0<=a<=1 and 0<=b<=1
        assert a+b==1.
        assert actions.basis(station+100.,100.)==pytest.approx((i,j,a,b),abs=1e-11)
    profile=actions.ActionProfile.anchor();profile.steer[0]=.01
    h=100./64
    assert profile.value('steer',2*h,100.)==0.
    assert profile.value('steer',-h/2,100.)==pytest.approx(.005)
    eps=1e-5
    left=(profile.value('steer',0.,100.)-profile.value('steer',-eps,100.))/eps
    right=(profile.value('steer',eps,100.)-profile.value('steer',0.,100.))/eps
    assert abs(left)<1e-6 and abs(right)<1e-6


def test_zero_correction_preserves_original_float32_action_object_and_scalar_target():
    profile=actions.ActionProfile.anchor()
    nominal=inputs_to_action(.123456789,.456789,0.,True)
    applied,details=actions.corrected_action(nominal,profile,12.345,5388.77,True,-.8)
    assert applied is nominal
    np.testing.assert_array_equal(applied,nominal)
    assert profile.value('speed',12.345,5388.77)==.7
    assert details['applied_residual_steer_rad']==0.
    assert details['applied_residual_throttle']==0.
    assert details['applied_residual_brake']==0.


def test_corrections_use_physical_units_and_legal_clamps():
    profile=actions.ActionProfile.anchor()
    profile.steer[:]=.02;profile.throttle[:]=.3;profile.brake[:]=-.3
    nominal=inputs_to_action(.59,.9,.2,True)
    action,details=actions.corrected_action(nominal,profile,0.,100.,True,-.8)
    applied=action_to_inputs(action,True)
    assert applied.steering_rad==pytest.approx(.6,abs=1e-7)
    assert applied.throttle==1.
    assert applied.brake==0.
    assert details['residual_steer_rad']==.02
    assert details['applied_residual_steer_rad']==pytest.approx(.01,abs=1e-7)
    assert details['applied_residual_throttle']==pytest.approx(.1,abs=1e-7)


def test_one_coordinate_change_keeps_incumbent_immutable_and_obeys_caps():
    anchor=actions.ActionProfile.anchor();before=anchor.digest()
    proposal=anchor.changed('speed',[41,42,43],.5)
    assert anchor.digest()==before
    np.testing.assert_array_equal(anchor.speed,np.full(64,.7))
    np.testing.assert_array_equal(proposal.speed[41:44],np.full(3,1.05))
    np.testing.assert_array_equal(proposal.speed[:41],anchor.speed[:41])
    np.testing.assert_array_equal(proposal.steer,anchor.steer)


def test_dnf_or_tiny_gain_never_replaces_clean_incumbent():
    incumbent={'lap2_s':100.}
    candidate={'clean':True,'lap2_s':99.98,'termination':'lap_complete','truncation':None,
               'recorder_check':{'lap_times_match_env':True}}
    assert actions.accepted(candidate,incumbent)
    assert not actions.accepted({**candidate,'clean':False},incumbent)
    assert not actions.accepted({**candidate,'lap2_s':99.995},incumbent)
    assert not actions.accepted({**candidate,'termination':'off_track'},incumbent)


def test_profile_resume_round_trip_and_invalid_values_fail_fast():
    profile=actions.ActionProfile.anchor().changed('brake',[5],.03)
    restored=actions.ActionProfile.from_dict(profile.to_dict())
    assert restored.digest()==profile.digest()
    bad=profile.to_dict();bad['speed'][0]=1.1
    with pytest.raises(ValueError,match='speed'):
        actions.ActionProfile.from_dict(bad)
