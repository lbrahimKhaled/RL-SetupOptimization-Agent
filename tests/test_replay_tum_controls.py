"""Force reconstruction, frozen command timing and temporary parameter changes; no simulation."""
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/comparison'))
import replay_tum_controls as replay

REFERENCE=Path(__file__).resolve().parents[1]/'comparison/tum/tum_none.csv'


@pytest.fixture
def reference():
    return replay.load_reference(REFERENCE)


def test_delivered_engine_torque_reconstructs_force_to_CSV_rounding_precision(reference):
    controls,report=replay.construct_controls(reference)
    assert report['positive_force_samples']==912
    assert report['positive_force_residual_rms_N']<.001
    assert report['positive_force_residual_max_abs_N']<.005
    assert report['brake_raw_max']==pytest.approx(.8809510054607462)
    assert report['brake_clipped_rows']==0
    assert controls[0].throttle==pytest.approx(1.,abs=1e-6)
    # Requested torque is a demand, not delivered crankshaft engine torque.
    changed=copy.deepcopy(reference)
    changed['data']['requested_torque_crank_Nm']*=10.
    same,_=replay.construct_controls(changed)
    assert controls==same


def test_missing_closing_mechanical_fields_are_periodic_not_last_sample_holds(reference):
    data=reference['data']
    assert 'engine_torque_crank_Nm' in reference['closing_fields_periodically_filled']
    assert data['engine_torque_crank_Nm'][-1]==data['engine_torque_crank_Nm'][0]
    assert data['ax_mps2'][-1]==data['ax_mps2'][0]
    assert data['t_s'][-1]==reference['meta']['lap_time_s']


def test_wrong_force_or_mass_assumptions_fail_validation(reference):
    changed=copy.deepcopy(reference)
    changed['data']['Fz_FL_N']*=2.
    with pytest.raises(ValueError,match='inconsistent'):
        replay.construct_controls(changed)


def test_time_only_commands_are_frozen_held_and_stop_at_reference_horizon(reference):
    commands,_=replay.construct_controls(reference)
    assert isinstance(commands,tuple)
    with pytest.raises(FrozenInstanceError):commands[0].throttle=.5
    assert replay.command_at(commands,.039999).index==0
    assert replay.command_at(commands,.04).index==1
    assert commands[0].t_start_s==0.
    assert commands[-1].t_start_s<reference['meta']['lap_time_s']
    assert commands[-1].t_end_s==reference['meta']['lap_time_s']
    assert len(commands)==2373
    with pytest.raises(ValueError):replay.command_at(commands,-1.)


def test_static_aero_zeroes_all_ride_height_polynomial_terms_without_mutating_baseline():
    baseline=replay.build_variant_overrides('baseline')
    aligned=replay.build_variant_overrides('static_aero')
    prefix=replay.P+'aerodynamics.'
    assert baseline=={}
    assert aligned[prefix+'c0']==2.20
    assert aligned[prefix+'a0']==2.68
    assert aligned[prefix+'c_d']==1.56
    assert aligned[prefix+'A_m2']==1.
    assert aligned[prefix+'l_f']==1.968
    for family in ['a','c']:
        for suffix in ['1_F','1_R','2_F','2_R','3_F','3_R','4','5','6']:
            assert aligned[prefix+family+suffix]==0.
    assert replay.build_variant_overrides('baseline')=={}
    with pytest.raises(ValueError):replay.build_variant_overrides('silently_unknown')


def test_timeline_quantization_never_claims_exact_fractional_physics_step(reference):
    horizon=reference['meta']['lap_time_s']
    steps=int(np.ceil(horizon/replay.DT));actual=steps*replay.DT
    assert actual==pytest.approx(94.888)
    assert 0<=actual-horizon<replay.DT
    assert replay.tum_radius(60/3.6)==pytest.approx(2.073/(2*np.pi))


def test_cli_main_freezes_one_command_artifact_for_all_variants_and_repeat(tmp_path,monkeypatch):
    import json
    calls=[]
    def fake_replay(reference,controls,variant,startup_gear='native'):
        calls.append((id(controls),variant))
        return [],[],{'variant':variant,'termination':'reference_horizon_reached',
                      'state_trace_numeric_sha256':'same','transient_numeric_sha256':'same'},                      {'post_reset_parameters_sha256':'parameter_hash'}
    monkeypatch.setattr(replay,'run_replay',fake_replay)
    monkeypatch.setattr(replay,'run_provenance',lambda track:{})
    out=tmp_path/'cli_result'
    monkeypatch.setattr(sys,'argv',['replay_tum_controls.py','--reference',str(REFERENCE),
                                   '--out',str(out),'--variants','baseline','static_aero'])
    replay.main()
    assert [variant for _,variant in calls]==['baseline','static_aero','baseline']
    assert len({identity for identity,_ in calls})==1
    commands_hash=replay.sha256_file(out/'commands.csv')
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['command_csv_sha256']==commands_hash
    for name in ['baseline','static_aero']:
        assert json.loads((out/f'{name}.meta.json').read_text())['command_csv_sha256']==commands_hash
    assert json.loads((out/'baseline_repeat.json').read_text())['state_trace_bit_identical'] is True


def test_preroll_freezes_global_25Hz_prefix_and_preserves_original_postzero_commands(reference):
    original,_=replay.construct_controls(reference)
    warm,_=replay.construct_controls(reference,3.)
    assert len(warm)==len(original)+75
    assert warm[75:]==original
    assert warm[0].t_start_s==-3.
    assert warm[74].t_end_s==0.
    assert warm[75].t_start_s==0.
    assert warm[0].index==-75 and warm[75].index==0
    # At500Hz, held input changes only at the predeclared global25Hz boundaries,
    # including the seamless negative-to-positive zero boundary.
    indices=[replay.command_at(warm,-3.+i*replay.DT).index for i in range(1520)]
    changes=[i for i in range(1,len(indices)) if indices[i]!=indices[i-1]]
    assert all(index%20==0 for index in changes)
    assert indices[1499]==-1 and indices[1500]==0
    ref_negative=replay.reference_at(reference,-3.)
    ref_periodic=replay.reference_at(reference,reference['meta']['lap_time_s']-3.)
    assert ref_negative==ref_periodic


def test_reference_gear_is_a_categorical_hold_not_fractional_interpolation(reference):
    data=reference['data']
    changes=np.flatnonzero(np.diff(data['gear'])!=0.)
    index=int(changes[0])
    query=.5*(data['t_s'][index]+data['t_s'][index+1])
    assert replay.reference_at(reference,query)['gear']==int(data['gear'][index])


def test_dynamic_radius_matches_TUM_growth_and_starts_with_consistent_wheel_speed(reference):
    for initial in [reference['data']['speed_mps'][0],replay.reference_at(reference,-3.)['speed_mps']]:
        values=replay.build_variant_overrides('static_aero_radius',float(initial))
        prefix=replay.P+'tire.rolling_radius_m.'
        base=values[prefix+'rear']
        scale=np.interp(initial,values[prefix+'velocity_scaling.velocity_points_mps'],values[prefix+'velocity_scaling.factors'])
        assert base*scale==pytest.approx(replay.tum_radius(initial),abs=1e-12)
        assert scale==pytest.approx(1.,abs=1e-12)
        omega=initial/base
        assert omega*base*scale==pytest.approx(initial,abs=1e-10)
        for velocity in [0.,30.,68.,100.]:
            factor=np.interp(velocity,values[prefix+'velocity_scaling.velocity_points_mps'],values[prefix+'velocity_scaling.factors'])
            assert base*factor==pytest.approx(replay.tum_radius(velocity),abs=1e-12)
    with pytest.raises(ValueError,match='initial speed'):
        replay.build_variant_overrides('static_aero_radius')


def test_startup_reference_gear_is_set_once_without_changing_future_scheduler_or_body():
    import copy
    from types import SimpleNamespace
    class FakePowertrain:
        def __init__(self):
            self._gear=1;self.gear_assignments=0;self.since_shift_s=.0;self.scheduler_calls=0
        @property
        def gear(self):return self._gear
        @gear.setter
        def gear(self,value):self._gear=value;self.gear_assignments+=1
        def rpm(self,omega):return omega*(8-self.gear)*60/(2*np.pi)
        @property
        def engine_inertia_at_wheels_kgm2(self):return float(self.gear+1)
        def update_gear(self,omega,dt_s):
            self.scheduler_calls+=1;self.gear=self.gear+1
    pt=FakePowertrain()
    backend=SimpleNamespace(powertrain=pt,car=SimpleNamespace(rear_axle_omega=lambda:200.),
                            params={'initial_body':{'x':1.,'vx':68.},'wheel_omega':200.})
    params=copy.deepcopy(backend.params)
    scheduler=pt.update_gear.__func__
    audit=replay.apply_startup_gear(backend,6,'reference')
    assert audit['before']['gear']==2 and audit['after']['gear']==6
    assert pt.gear_assignments==1
    assert pt.since_shift_s==replay.MIN_SHIFT_INTERVAL_S
    assert backend.params==params
    assert audit['before']['rear_axle_omega_radps']==audit['after']['rear_axle_omega_radps']
    assert pt.scheduler_calls==0
    assert pt.update_gear.__func__ is scheduler
    pt.update_gear(220.,.002)
    assert pt.scheduler_calls==1 and pt.gear==6 # normal future update can move beyond startupgear


def test_native_startup_policy_is_read_only_and_rejects_unknown_modes():
    from types import SimpleNamespace
    pt=replay.full_engine_curve()
    backend=SimpleNamespace(powertrain=pt,car=SimpleNamespace(rear_axle_omega=lambda:200.))
    before=(pt.gear,pt.since_shift_s)
    audit=replay.apply_startup_gear(backend,6)
    assert (pt.gear,pt.since_shift_s)==before
    assert audit['assignment_count']==0
    with pytest.raises(ValueError):replay.apply_startup_gear(backend,6,'force_every_step')
