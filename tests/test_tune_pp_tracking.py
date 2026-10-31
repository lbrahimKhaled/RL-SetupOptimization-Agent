"""Rear-axle kinematics, PP signs and diagnostic metric separation; no physical laps."""
from pathlib import Path
import sys
import math
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/comparison'))
import tune_pp_tracking as tracking
from rlsetup.driving.reference import ReferenceLine
from rlsetup.driving.config import DemoConfig
from rlsetup.driving.demo_driver import DemoDriver


def state(**changes):
    value={'x':50.,'y':0.,'yaw':0.,'vx':15.,'vy':0.,'yaw_rate':0.,'slip_rl':0.,'slip_rr':0.}
    value.update(changes);return value


def line():return ReferenceLine.from_xy([[0.,0.],[100.,0.],[100.,100.],[0.,100.]])


def test_rear_velocity_includes_yaw_rate_and_rotates_into_world_prediction():
    actual,predicted,yaw,beta,speed=tracking.rear_state(state(yaw=math.pi/2,vy=3.,yaw_rate=.5),2.,.08)
    np.testing.assert_allclose(actual,[50.,-2.],atol=1e-12)
    np.testing.assert_allclose(predicted,[49.84,-.8],atol=1e-12)
    assert yaw==pytest.approx(math.pi/2+.04)
    assert beta==pytest.approx(math.atan2(2.,15.))
    assert speed==pytest.approx(math.hypot(15.,2.))


def test_original_mode_retains_original_pure_pursuit_formula_exactly():
    cfg=DemoConfig()
    old=DemoDriver(line(),cfg,pedal_deadzone=True,control_dt_s=.04,car='sf70h_tum')
    wrapped=tracking.RearPursuit(line(),cfg,tracking.FormulaConfig(mode='original'),
                               pedal_deadzone=True,control_dt_s=.04,car='sf70h_tum')
    assert wrapped.steering(state(y=1.,yaw=.05),50.)==old.steering(state(y=1.,yaw=.05),50.)


def test_predicted_rear_formula_matches_rear_point_and_steers_toward_line():
    driver=tracking.RearPursuit(line(),DemoConfig(),tracking.FormulaConfig(),
                               pedal_deadzone=True,control_dt_s=.04,car='sf70h_tum')
    assert driver.steering(state(y=1.),50.)<0.
    assert driver.diagnostics['s_predicted_rear_m']<50.
    assert driver.diagnostics['lookahead_m']>=2*15.*.04
    centered=tracking.RearPursuit(line(),DemoConfig(),tracking.FormulaConfig(),
                                pedal_deadzone=True,control_dt_s=.04,car='sf70h_tum')
    assert centered.steering(state(),50.)==0.


def metric_rows():
    # Lap2 traverses9m quickly, then the final1m slowly. Time-weighted RMS would
    # substantially overemphasize the slow section; station weighting must not.
    rows=[]
    for t,station,error in [(0.,0.,0.),(.5,4.5,0.),(1.,9.,0.),(9.,9.9,1.),(10.,0.,1.)]:
        rows.append({'t_s':t,'s_raceline_m':station,'on_track':int(t<9.),
                     'e_raceline_m':error,'e_rear_m':2*error,'heading_err_raceline_rad':.1,
                     'CoG_velocity_heading_error_rad':.3,'rear_velocity_heading_error_rad':.2,
                     'speed_error_mps':-.5,'speed_mps':12.,'vx_mps':11.5,'target_speed_mps':12.})
    return rows


def test_metrics_keep_station_CoG_rear_velocity_heading_and_speed_separate():
    result=tracking.metrics(metric_rows(),10.,[-10.,0.,10.])
    assert result['full_station_coverage'] is True
    assert result['station_coverage_fraction']==pytest.approx(1.)
    assert result['CoG_error_m']['rms']==pytest.approx(math.sqrt(.055))
    assert result['rear_error_m']['rms']==pytest.approx(2*math.sqrt(.055))
    assert result['CoG_velocity_heading_error_rad']['rms']==pytest.approx(.3)
    assert result['body_heading_error_rad']['rms']==pytest.approx(.1)
    assert result['not_a_clean_lap_claim'] is True
    assert result['speed_error_mps']['mean']==pytest.approx(-.5)
    assert result['mean_actual_speed_mps']==12.
    assert result['tracking_score_m']==pytest.approx(math.sqrt(.055)+.25)


def attempt(index,error,mean_speed=20.,mode='rear_predicted',constant=None,speed_error=0.,complete=True):
    return {'trial_index':index,'completed_lap2':complete,'formula':{'mode':mode},
            'factor':.8 if constant is None else None,'constant_speed_mps':constant,
            'lap2_metrics':{'samples':10,'full_station_coverage':complete,
                'CoG_error_m':{'rms':error,'max_abs':2*error},'tracking_score_m':1.5*error,
                'mean_actual_speed_mps':mean_speed,'speed_error_mps':{'rms':speed_error}}}


def test_full_profile_crawling_cannot_win_and_does_not_require_constant_speed_RMS():
    baseline=attempt(0,.2,mode='original',speed_error=6.)
    crawler=attempt(1,.001,mean_speed=18.)
    faster_geometry=attempt(2,.1,mean_speed=19.1,speed_error=4.)
    assert tracking.select_best([baseline,crawler,faster_geometry])['trial_index']==2
    assert crawler['tracking_eligible'] is False
    assert 'tradeoff' in crawler['tracking_ineligible_reason']
    assert baseline['tracking_eligible'] is True


def test_constant_speed_fairness_rejects_very_low_geometric_error_with_large_speed_error():
    baseline=attempt(0,.2,mode='original',constant=12.)
    crawler=attempt(1,.001,constant=12.,speed_error=2.01)
    assert tracking.select_best([baseline,crawler])['trial_index']==0
    assert crawler['tracking_eligible'] is False


def test_incomplete_station_coverage_is_not_eligible():
    baseline=attempt(0,.2,mode='original')
    incomplete=attempt(1,.01)
    incomplete['lap2_metrics']['full_station_coverage']=False
    assert tracking.select_best([baseline,incomplete])['trial_index']==0


def test_optional_CoG_correction_sign_and_zero_defaults_keep_PP_base():
    cfg=DemoConfig()
    base=tracking.RearPursuit(line(),cfg,tracking.FormulaConfig(),
        pedal_deadzone=True,control_dt_s=.04,car='sf70h_tum')
    corrected=tracking.RearPursuit(line(),cfg,tracking.FormulaConfig(cross_gain=.02),
        pedal_deadzone=True,control_dt_s=.04,car='sf70h_tum')
    st=state(y=.1)
    assert corrected.steering(st,50.)==pytest.approx(base.steering(st,50.)-.002)


def test_formula_configuration_rejects_nonphysical_preview_values():
    with pytest.raises(ValueError,match='prediction_s'):
        tracking.FormulaConfig(prediction_s=-.1)
