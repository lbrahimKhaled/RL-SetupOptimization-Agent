#!/usr/bin/env python3
"""Reconstruct and freeze TUM-derived commands, then replay open loop in temporary OCD variants.

These are reconstructed inputs, NOT actuator commands exported by TUM's QSS solver.
Filtered TUM curvature and the XY trajectory need not describe exactly the same geometry.
No feedback, pure pursuit, progress retiming, learned policy or forced lap completion is used.
"""
import argparse
import csv
from dataclasses import asdict,dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import common
import comparison_state
import ocd_trace
from rlsetup.driving import setups
from rlsetup.provenance import run_provenance,sha256_file
from ocd_lap.backend import OCDBackend,StartPose,Inputs,DT,CONFIG
from ocd_lap.powertrain import Powertrain,F1_SHANGHAI
from ocd_lap.translator import P
from ocd_lap.track import Track
from ocd_lap import cars
import ocd_native

MASS=733.
RHO=1.18
DRAG_AREA=1.56
ROLLING=.03
BRAKES=15000.
WHEELBASE=3.6
CONTROL_DT=.04
WHEELS=('FL','FR','RL','RR')


def tum_radius(speed):
    """TUM driven rolling radius [m], including speed-dependent circumference growth."""
    return 2.073/(2*math.pi)*(1.+(np.asarray(speed)*3.6-60.)*.045/200.)


def full_engine_curve():
    return Powertrain(vehicle_mass_kg=MASS,wheel_radius_m=float(tum_radius(0.)),wheel_inertia_kgm2=0.)


def load_reference(path):
    path=Path(path).resolve();meta=json.loads(path.with_suffix('.meta.json').read_text())
    if (meta['variant']!='none' or meta['sha256']!=sha256_file(path) or meta['ers_used_J']!=0 or
            meta['drs_share_of_points']!=0):
        raise ValueError('verified standard no-ERS/no-DRS TUM reference required')
    with path.open(newline='') as handle:rows=list(csv.DictReader(handle))
    if len(rows)<4 or int(rows[-1]['closing_point'])!=1:raise ValueError('TUM reference needs its closing row')
    fields=['t_s','s_m','x_m','y_m','kappa_radpm','speed_mps','ax_mps2','gear','engine_rpm',
            'engine_torque_crank_Nm','requested_torque_crank_Nm']+[f'Fz_{w}_N' for w in WHEELS]
    data={};filled=[]
    for field in fields:
        values=[]
        for i,row in enumerate(rows):
            text=row[field]
            if text=='':
                if i!=len(rows)-1:raise ValueError(f'missing nonclosing TUM {field}, row{i}')
                text=rows[0][field];filled.append(field)
            value=float(text)
            if not math.isfinite(value):raise ValueError(f'nonfinite TUM {field}, row{i}')
            values.append(value)
        data[field]=np.asarray(values)
    if not np.all(np.diff(data['t_s'])>0) or data['t_s'][0]!=0:raise ValueError('invalid TUM time axis')
    if abs(data['t_s'][-1]-meta['lap_time_s'])>1e-5:raise ValueError('TUM CSV/metadata time mismatch')
    data['t_s'][-1]=float(meta['lap_time_s'])
    xy=np.column_stack((data['x_m'][:-1],data['y_m'][:-1]))
    tangent=np.roll(xy,-1,axis=0)-np.roll(xy,1,axis=0)
    yaw=np.arctan2(tangent[:,1],tangent[:,0])
    data['yaw_rad']=np.unwrap(np.r_[yaw,yaw[0]])
    return {'path':str(path),'meta':meta,'data':data,'closing_fields_periodically_filled':sorted(set(filled))}


def reference_at(reference,time_s):
    data=reference['data'];time_s=min(max(float(time_s),0.),data['t_s'][-1])
    return {key:float(np.interp(time_s,data['t_s'],data[key])) for key in
            ('s_m','x_m','y_m','yaw_rad','speed_mps','kappa_radpm','gear','engine_torque_crank_Nm','requested_torque_crank_Nm')}


def reconstruct_knots(reference):
    """Reconstruct steering/throttle/brake at TUM nodes and validate longitudinal force units."""
    d=reference['data'];engine=full_engine_curve()
    full=np.asarray([engine.engine_torque_Nm(rpm) for rpm in d['engine_rpm']])
    if np.any(full<=0):raise ValueError('nonpositive full-load engine torque')
    delivered=d['engine_torque_crank_Nm']
    throttle_raw=delivered/full
    f_required=MASS*d['ax_mps2']+.5*RHO*DRAG_AREA*d['speed_mps']**2+ROLLING*sum(d[f'Fz_{w}_N'] for w in WHEELS)
    radius=tum_radius(d['speed_mps'])
    brake_raw=np.maximum(0.,-f_required)*radius/BRAKES
    steering=np.arctan(WHEELBASE*d['kappa_radpm'])
    gear=d['gear'].astype(int)-1
    if np.any(gear<0) or np.any(gear>=len(F1_SHANGHAI['i_trans'])) or np.any(d['gear']!=gear+1):
        raise ValueError('TUM gear must be a valid integer1..8')
    engine_force=.96*delivered/(np.asarray(F1_SHANGHAI['i_trans'])[gear]*radius*np.asarray(F1_SHANGHAI['e_i'])[gear])
    positive=(f_required[:-1]>0.)
    residual=f_required[:-1][positive]-engine_force[:-1][positive]
    if not len(residual):raise ValueError('no positive-force TUM rows to validate')
    report={'positive_force_samples':len(residual),'positive_force_residual_rms_N':float(np.sqrt(np.mean(residual**2))),
            'positive_force_residual_max_abs_N':float(np.max(np.abs(residual))),
            'positive_force_formula':'733ax+.5rho*cWA*v^2+.03sumFz vs eta_g*m_eng/(i_trans*r(v)*e_i)',
            'throttle_raw_min':float(throttle_raw.min()),'throttle_raw_max':float(throttle_raw.max()),
            'throttle_rounding_clipped_rows':int(np.sum((throttle_raw<0)|(throttle_raw>1))),
            'brake_raw_max':float(brake_raw.max()),'brake_clipped_rows':int(np.sum(brake_raw>1)),
            'steering_max_abs_rad':float(np.max(np.abs(steering))),
            'requested_torque_used_for_throttle':False,'inertia_mass_factor_divided_from_throttle':False}
    if report['positive_force_residual_max_abs_N']>.1:raise ValueError(f'longitudinal reconstruction inconsistent: {report}')
    if throttle_raw.min()<-1e-5 or throttle_raw.max()>1.+1e-5:raise ValueError('throttle reconstruction outside rounding tolerance')
    if brake_raw.max()>1. or np.max(np.abs(steering))>.6:raise ValueError('reconstructed input exceeds physical authority')
    return {'steer':steering,'throttle':np.clip(throttle_raw,0.,1.),'brake':brake_raw},report


@dataclass(frozen=True)
class Command:
    index:int
    t_start_s:float
    t_end_s:float
    steer_rad:float
    throttle:float
    brake:float


def construct_controls(reference):
    knots,validation=reconstruct_knots(reference)
    d=reference['data'];end=float(d['t_s'][-1])
    count=int(math.ceil(end/CONTROL_DT))
    controls=tuple(Command(i,i*CONTROL_DT,min((i+1)*CONTROL_DT,end),
                         *[float(np.interp(i*CONTROL_DT,d['t_s'],knots[k])) for k in ('steer','throttle','brake')])
                   for i in range(count))
    return controls,validation


def build_variant_overrides(variant):
    if variant=='baseline':return {}
    if variant!='static_aero':raise ValueError(f'unknown temporary variant {variant!r}')
    aero=P+'aerodynamics.'
    values={aero+family+suffix:0. for family in ('c','a') for suffix in
            ('0','1_F','1_R','2_F','2_R','3_F','3_R','4','5','6')}
    values.update({aero+'A_m2':1.,aero+'c_d':DRAG_AREA,aero+'c0':2.20,aero+'a0':2.68,
                   aero+'l':WHEELBASE,aero+'l_f':1.968,aero+'air_density_kgpm3':RHO})
    return values


def command_at(controls,time_s):
    """Time alone selects a frozen25Hz ZOH command, irrespective of actual state/progress."""
    if not math.isfinite(time_s) or time_s<0:raise ValueError('invalid replay time')
    index=min(int(math.floor((time_s+1e-12)/CONTROL_DT)),len(controls)-1)
    return controls[index]


def numeric_hash(data):
    return hashlib.sha256(json.dumps(common._clean(data),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def error_summary(dense,mask=None):
    array=np.asarray(dense,dtype=float)
    if mask is not None:array=array[np.asarray(mask,dtype=bool)]
    if not len(array):return {'samples':0}
    return {'samples':len(array),'time_start_s':float(array[0,0]),'time_end_s':float(array[-1,0]),
            'position_RMSE_m':float(np.sqrt(np.mean(array[:,1]**2))),
            'position_max_m':float(np.max(array[:,1])),
            'speed_RMSE_mps':float(np.sqrt(np.mean(array[:,2]**2))),
            'speed_bias_mps':float(np.mean(array[:,2])),
            'terminal_position_error_m':float(array[-1,1])}


EXTRA=['command_index','command_time_s','reference_time_s','reference_s_m','reference_x_m','reference_y_m',
       'reference_speed_mps','reference_yaw_rad','reference_kappa_radpm','reference_gear',
       'reference_delivered_engine_torque_Nm','reference_requested_torque_Nm','time_position_error_m','time_speed_error_mps']


def run_replay(reference,controls,variant):
    """Run the same immutable input tuple in a temporary backend; no state enters command selection."""
    overrides=build_variant_overrides(variant)
    track=Track('Shanghai');geo=ocd_trace.Geometry(track);setup=setups.get('p1_TUM')
    backend=OCDBackend(powertrain='engine',car='sf70h_tum',ers=False,max_brake_torque_Nm=BRAKES,overrides=overrides)
    start=reference_at(reference,0.)
    pose=StartPose(start['x_m'],start['y_m'],start['yaw_rad'],start['speed_mps'])
    state=backend.reset(setup,pose);params=backend.run_metadata()
    for key,value in overrides.items():
        if params['parameters'][key]!=value:raise RuntimeError(f'temporary override not applied: {key}')
    initial=geo.locate(state['x'],state['y'],state['yaw'])
    path=initial['s_path_m'];progress=0.
    clock=ocd_trace.LapClock(track,path,20)
    footprint=cars.get('sf70h_tum').footprint
    initial_inside=bool(track.contains_car(state['x'],state['y'],state['yaw'],footprint))
    end=float(reference['meta']['lap_time_s']);total_steps=int(math.ceil(end/DT))
    rows=[];transient=[];dense=[];cause='reference_horizon_reached';steps=0;stall=0.
    inside=initial_inside;control_previous=0.;started=time.perf_counter()
    def measure(t,current):
        ref=reference_at(reference,t);speed=math.hypot(current['vx'],current['vy'])
        error=math.hypot(current['x']-ref['x_m'],current['y']-ref['y_m'])
        dense.append((t,error,speed-ref['speed_mps'],ref['s_m']))
        return ref,error,speed-ref['speed_mps']
    measure(0.,state)
    for step in range(total_steps):
        command=command_at(controls,step*DT)
        inputs=Inputs(command.steer_rad,command.throttle,command.brake)
        state=backend.step(inputs);steps+=1;t=steps*DT
        if not all(math.isfinite(value) for value in state.values()):cause='nonfinite';break
        instant_inside=bool(track.contains_car(state['x'],state['y'],state['yaw'],footprint))
        inside &= instant_inside
        ref,pos_error,speed_error=measure(t,state)
        if t<=2.+1e-12:
            transient.append({'t_s':t,**ocd_trace.backend_sample(backend,state,inputs),
                              'reference_x_m':ref['x_m'],'reference_y_m':ref['y_m'],'reference_speed_mps':ref['speed_mps'],
                              'reference_s_m':ref['s_m'],'time_position_error_m':pos_error,'time_speed_error_mps':speed_error,
                              'on_track_at_physics_step':int(instant_inside)})
        # Full forces/slips/gear and line/progress diagnostics at command boundaries,
        # with an explicit final partial command interval. Footprint checks stay500Hz.
        if steps%20==0 or steps==total_steps:
            geometry=geo.locate(state['x'],state['y'],state['yaw'])
            previous=progress
            progress+=(geometry['s_path_m']-path+track.length/2)%track.length-track.length/2
            path=geometry['s_path_m']
            clock.update(previous,progress,control_previous,t);control_previous=t
            row=dict(t_s=t,lap=clock.lap,t_lap_s=clock.t_lap(t),progress_m=progress,**geometry,
                     **ocd_trace.backend_sample(backend,state,inputs),on_track=int(inside),target_speed_mps=ref['speed_mps'],
                     command_index=command.index,command_time_s=command.t_start_s,reference_time_s=min(t,end),
                     reference_s_m=ref['s_m'],reference_x_m=ref['x_m'],reference_y_m=ref['y_m'],
                     reference_speed_mps=ref['speed_mps'],reference_yaw_rad=ref['yaw_rad'],reference_kappa_radpm=ref['kappa_radpm'],
                     reference_gear=ref['gear'],reference_delivered_engine_torque_Nm=ref['engine_torque_crank_Nm'],
                     reference_requested_torque_Nm=ref['requested_torque_crank_Nm'],
                     time_position_error_m=pos_error,time_speed_error_mps=speed_error)
            rows.append(row);inside=True
            stall=stall+(t-(rows[-2]['t_s'] if len(rows)>1 else 0.)) if row['speed_mps']<3. else 0.
            if abs(geometry['heading_err_raceline_rad'])>math.pi/2:cause='spin';break
            if stall>2.:cause='stall';break
    dense_array=np.asarray(dense)
    line_error=np.asarray([row['e_raceline_m'] for row in rows])
    summary={'variant':variant,'termination':cause,'reference_horizon_s':end,'actual_sim_time_s':steps*DT,
             'physics_step_quantization_overshoot_s':max(0.,steps*DT-end),'reference_horizon_reached':bool(steps==total_steps and cause=='reference_horizon_reached'),
             'actual_progress_path_m':progress,'actual_line_crossings_s':clock.crossings,
             'actual_lap_times_s':clock.lap_times,'actual_completed_full_laps':len(clock.lap_times),
             'initial_pose':asdict(pose),'initial_vy_mps':0.,'initial_yaw_rate_radps':0.,
             'initial_pose_on_track':initial_inside,'no_initial_steady_turn_state':'public StartPose cannot set vy/yaw_rate',
             'trajectory_time_aligned':error_summary(dense),
             'first_2s':error_summary(dense,dense_array[:,0]<=2.),
             'first_reference_meter':error_summary(dense,dense_array[:,3]<=1.),
             'station_matched_CoG_error_m':{'RMS':float(np.sqrt(np.mean(line_error**2))) if len(line_error) else None,
                                          'max_abs':float(np.max(np.abs(line_error))) if len(line_error) else None,
                                          'sample_rate_hz':25},
             'on_track_control_interval_fraction':float(np.mean([row['on_track'] for row in rows])) if rows else None,
             'initial_pose_and_every_2ms_footprint_checked':True,'footprint_exit_terminates':False,
             'valid_clean_lap_claim':False,'wall_s':time.perf_counter()-started,
             'state_trace_numeric_sha256':numeric_hash(rows),'transient_numeric_sha256':numeric_hash(transient)}
    return rows,transient,summary,{'variant':variant,'overrides':overrides,'post_reset':params,
                                 'post_reset_parameters_sha256':numeric_hash(params['parameters']),
                                 'setup':setup.to_dict(),'setup_sha256':numeric_hash(setup.to_dict())}


def write_csv(path,rows,columns):
    with Path(path).open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(common._clean(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--variants',nargs='+',choices=['baseline','static_aero'],default=['baseline','static_aero'])
    args=parser.parse_args()
    if len(set(args.variants))!=len(args.variants):parser.error('duplicate variants not allowed')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    reference=load_reference(args.reference);controls,validation=construct_controls(reference)
    write_csv(out/'commands.csv',[asdict(command) for command in controls],list(Command.__dataclass_fields__))
    command_hash=sha256_file(out/'commands.csv')
    method={'source':'reconstructed_TUM_derived_commands_NOT_exported_solver_actuators','reference_sha256':reference['meta']['sha256'],
            'reference':reference['path'],'reference_lap_s':reference['meta']['lap_time_s'],
            'steering':'atan(3.6 * filtered_TUM_kappa); filtered curvature can differ from XY curvature',
            'throttle':'delivered engine crank torque / full TUM engine torque at exported rpm; requested torque NOT used; no e_i division',
            'brake':'max(0,-(733ax+.5*1.18*1.56*v^2+.03sumFz))*rTUM(v)/15000',
            'radius':'2.073/(2pi)*(1+(3.6v-60)*.045/200)',
            'interpolation':'reconstruct at TUM nodes, linear interpolation to25Hz times, ZOH until next command',
            'closure':'missing closing mechanical values copied periodically from first row',
            'periodic_closure_fields':reference['closing_fields_periodically_filled'],'validation':validation,
            'no_feedback':True,'time_index_only':True,'no_progress_retiming':True,'ers':False,'drs':False,
            'physics_dt_s':DT,'commands_dt_s':CONTROL_DT,'command_csv_sha256':command_hash}
    common.write_meta(out/'commands.csv',method)
    comparison_state.write_json(out/'reconstruction_validation.json',common._clean(validation))
    protocol={**method,'variants':args.variants,'temporary_overrides':{variant:build_variant_overrides(variant) for variant in args.variants},
              'code':{**run_provenance('Shanghai'),'script_sha256':sha256_file(__file__),
                      'ocd_native_sha256':sha256_file(ocd_native.__file__),'ocd_config_sha256':sha256_file(CONFIG)},
              'command':sys.argv,'initial_state_limitation':'StartPose sets XY/yaw/vx only; vy/yaw_rate start0',
              'completion':'reference time horizon is not a finish; report actual OCD line crossings/progress',
              'gear_radius_caveat':'OCD automatic gearbox/hysteresis and fixed rolling radius remain; TUM uses speed-based gear and growing radius'}
    comparison_state.write_json(out/'protocol.json',common._clean(protocol))
    summaries=[]
    for variant in args.variants:
        if sha256_file(out/'commands.csv')!=command_hash:raise RuntimeError('frozen command artifact changed')
        rows,transient,summary,audit=run_replay(reference,controls,variant)
        path=out/f'{variant}.csv';write_csv(path,rows,[c[0] for c in common.PP_COLUMNS]+EXTRA)
        common.write_meta(path,{'source':'frozen_TUM_reconstructed_open_loop','summary':summary,
                               'command_csv_sha256':command_hash,'reference_sha256':reference['meta']['sha256'],
                               'parameter_snapshot_sha256':audit['post_reset_parameters_sha256'],
                               'valid_clean_lap':False,'protocol':'protocol.json'})
        if transient:
            transient_path=out/f'{variant}_first2s.csv'
            columns=list(transient[0])
            write_csv(transient_path,transient,columns)
            common.write_meta(transient_path,{'source':'first2s500Hz_open_loop_transient','command_csv_sha256':command_hash})
        comparison_state.write_json(out/f'{variant}.parameters.json',common._clean(audit))
        comparison_state.write_json(out/f'{variant}.summary.json',common._clean(summary))
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    if 'baseline' in args.variants:
        _,_,repeat,_=run_replay(reference,controls,'baseline')
        original=next(summary for summary in summaries if summary['variant']=='baseline')
        repeat_check={'state_trace_bit_identical':original['state_trace_numeric_sha256']==repeat['state_trace_numeric_sha256'],
                      'transient_bit_identical':original['transient_numeric_sha256']==repeat['transient_numeric_sha256'],
                      'termination_matches':original['termination']==repeat['termination'],
                      'command_csv_sha256':command_hash,'repeat_summary':repeat}
        comparison_state.write_json(out/'baseline_repeat.json',common._clean(repeat_check))
        if not all(repeat_check[k] for k in ('state_trace_bit_identical','transient_bit_identical','termination_matches')):
            raise RuntimeError('baseline open-loop replay is not deterministic')
    comparison_state.write_json(out/'summary.json',{'variants':common._clean(summaries),'command_csv_sha256':command_hash,
                             'no_torch_loaded':'torch' not in sys.modules,'no_dependency_source_edits':True})


if __name__=='__main__':main()
