#!/usr/bin/env python3
"""Pure-pursuit formula accuracy on the original TUM raceline, no RL.

Diagnostic only: the exact original raceline is not footprint-feasible everywhere.
Footprint exits are checked at every 2 ms physics step and logged without terminating.
Commands remain 25 Hz. Spin/nonfinite/stall/time-limit stops remain active.
"""
import argparse
from dataclasses import asdict,dataclass,replace
import itertools
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
from tum_pure_pursuit import TumProfile
from rlsetup.driving.demo_driver import DemoDriver
from rlsetup.driving.train_config import train_config_from_dict
from rlsetup.driving import setups
from rlsetup.provenance import run_provenance,sha256_file
from ocd_lap.backend import OCDBackend,StartPose,Inputs,DT,CONFIG
from ocd_lap.env import MAX_STEER_RAD
from ocd_lap.track import Track
from ocd_lap import cars
import ocd_native

CONTROL_DT=.04
CONTROL_STEPS=20


@dataclass(frozen=True)
class FormulaConfig:
    mode:str='rear_predicted'
    base_m:float=1.5
    lookahead_s:float=.12
    prediction_s:float=0.
    beta_weight:float=0.
    maximum_lookahead_m:float=35.
    understeer_gain:float=0. # s^2/m, multiplies speed^2*kappa
    cross_gain:float=0. # rad/m, minus gain * positive-left CoG offset
    yaw_gain:float=0. # s, minus gain * (yaw_rate - speed*kappa)

    def __post_init__(self):
        if self.mode not in ('original','rear_predicted'):
            raise ValueError('unknown pure-pursuit formula mode')
        for name,lo,hi in [('base_m',.1,10.),('lookahead_s',0.,1.),('prediction_s',0.,.2),
                           ('beta_weight',0.,1.),('maximum_lookahead_m',2.,100.),
                           ('understeer_gain',0.,.01),('cross_gain',0.,.2),('yaw_gain',0.,1.)]:
            value=getattr(self,name)
            if not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'{name} must be finite in [{lo},{hi}]')
        if self.mode=='original' and any((self.understeer_gain,self.cross_gain,self.yaw_gain)):
            raise ValueError('original baseline must have zero optional corrections')


def rear_state(state,rear_axle_m,prediction_s=0.):
    """Actual/predicted rear axle position, yaw and velocity-side-slip; SI units."""
    yaw=state['yaw'];c,s=math.cos(yaw),math.sin(yaw)
    rear=np.array([state['x']-rear_axle_m*c,state['y']-rear_axle_m*s])
    vx,vy=state['vx'],state['vy']-rear_axle_m*state['yaw_rate']
    predicted=rear+prediction_s*np.array([c*vx-s*vy,s*vx+c*vy])
    return rear,predicted,yaw+state['yaw_rate']*prediction_s,math.atan2(vy,max(vx,.1)),math.hypot(vx,vy)


class RearPursuit(DemoDriver):
    """Original PP or predicted-rear, rear-velocity-heading compensated PP; same PI pedals."""
    def __init__(self,line,config,formula,**kwargs):
        super().__init__(line,config,**kwargs)
        self.formula=formula
        self.rear_hint=None
        self.diagnostics={}

    def steering(self,state,s_ref):
        config=self.formula
        if config.mode=='original':
            self.diagnostics={'lookahead_m':self.cfg.lookahead_m+self.cfg.lookahead_s*max(state['vx'],0.),
                              'beta_rear_rad':0.,'s_predicted_rear_m':float('nan')}
            return super().steering(state,s_ref)
        rear,predicted,yaw,beta,speed=rear_state(state,self.rear_axle_m,config.prediction_s)
        self.rear_hint=self.ref.match(predicted,self.rear_hint)
        minimum=max(2.,2.*speed*self.dt)
        lookahead=float(np.clip(config.base_m+config.lookahead_s*speed,minimum,config.maximum_lookahead_m))
        goal=self.ref.point(self.rear_hint+lookahead)
        delta=goal-predicted
        distance=float(np.linalg.norm(delta))
        if distance<1e-9:
            raise RuntimeError('pure-pursuit goal coincides with predicted rear axle')
        alpha=math.atan2(delta[1],delta[0])-(yaw+config.beta_weight*beta)
        steer=math.atan2(2.*self.wheelbase_m*math.sin(alpha),distance)
        # Optional small corrections; default zero. Never add a second geometric
        # atan(L*kappa) term on top of the pure-pursuit curvature command.
        e_cog,_=self.ref.errors_at(s_ref,state['x'],state['y'],state['yaw'])
        chord_a=self.ref.point(s_ref)-self.ref.point(s_ref-5.)
        chord_b=self.ref.point(s_ref+5.)-self.ref.point(s_ref)
        turn=math.atan2(chord_a[0]*chord_b[1]-chord_a[1]*chord_b[0],float(np.dot(chord_a,chord_b)))
        kappa=turn/5.
        cog_speed=math.hypot(state['vx'],state['vy'])
        steer += config.understeer_gain*cog_speed*cog_speed*kappa-config.cross_gain*e_cog-config.yaw_gain*(state['yaw_rate']-cog_speed*kappa)
        self.diagnostics={'lookahead_m':lookahead,'beta_rear_rad':beta,
                          's_predicted_rear_m':self.rear_hint}
        return float(np.clip(steer,-MAX_STEER_RAD,MAX_STEER_RAD))


def metrics(rows,raceline_length,crossings,lap=2):
    """Whole flying lap, exact clock boundaries, station-weighted lateral errors.

    Continuous metrics use sampled 25Hz states; footprint flags summarize every2ms.
    Forward TUM station increments weight lateral/heading RMS rather than dwell time.
    Speed fairness uses time-weighted speed error/means within the same speed condition.
    """
    if len(crossings)<=lap:return {'samples':0,'full_station_coverage':False}
    times=np.asarray([row['t_s'] for row in rows],dtype=float)
    start,end=crossings[lap-1:lap+1]
    if not len(times) or times[0]>start or times[-1]<end:
        return {'samples':0,'full_station_coverage':False}
    query=np.r_[start,times[(times>start)&(times<end)],end]
    stations=np.unwrap(np.asarray([row['s_raceline_m'] for row in rows]),period=raceline_length)
    unwrapped=np.interp(query,times,stations)
    ds=np.maximum(np.diff(unwrapped),0.)
    distance=float(ds.sum());span=float(unwrapped[-1]-unwrapped[0])
    dt=np.diff(query);duration=float(end-start)
    if distance<=0 or duration<=0:raise ValueError('nonpositive station coverage / flying lap duration')
    def values(key):return np.interp(query,times,np.asarray([row[key] for row in rows],dtype=float))
    def mean(array,weight,total):return float(np.sum(.5*(array[:-1]+array[1:])*weight)/total)
    result={'samples':len(query),'station_weighting':'positive TUM-raceline progress, trapezoidal integration',
            'station_distance_m':distance,'station_span_m':span,'raceline_length_m':raceline_length,
            'station_coverage_fraction':span/raceline_length,
            'full_station_coverage':bool(abs(span-raceline_length)<=max(5.,.002*raceline_length) and
                                         abs(distance-raceline_length)<=max(5.,.002*raceline_length)),
            'not_a_clean_lap_claim':True}
    for key,label in [('e_raceline_m','CoG_error_m'),('e_rear_m','rear_error_m'),
                      ('heading_err_raceline_rad','body_heading_error_rad'),
                      ('CoG_velocity_heading_error_rad','CoG_velocity_heading_error_rad'),
                      ('rear_velocity_heading_error_rad','rear_velocity_heading_error_rad')]:
        array=values(key)
        result[label]={'rms':math.sqrt(mean(array*array,ds,distance)),
                       'max_abs':float(np.max(np.abs(array))),'mean':mean(array,ds,distance)}
    speed_error=values('speed_error_mps')
    result['speed_error_mps']={'rms':math.sqrt(mean(speed_error*speed_error,dt,duration)),
                             'max_abs':float(np.max(np.abs(speed_error))),'mean':mean(speed_error,dt,duration)}
    result['mean_actual_speed_mps']=mean(values('speed_mps'),dt,duration)
    result['mean_actual_vx_mps']=mean(values('vx_mps'),dt,duration)
    result['mean_target_speed_mps']=mean(values('target_speed_mps'),dt,duration)
    # Flags describe the interval ending at each row: no interpolation of boolean state.
    indices=np.minimum(np.searchsorted(times,.5*(query[:-1]+query[1:]),side='left'),len(rows)-1)
    flags=np.asarray([row['on_track'] for row in rows],dtype=float)[indices]
    result['on_track_interval_fraction']=float(np.sum(flags*dt)/duration)
    result['tracking_score_m']=result['CoG_error_m']['rms']+.25*result['CoG_error_m']['max_abs']
    return result


EXTRA=[('lookahead_m','m','actual commanded pure-pursuit arc lookahead'),
       ('beta_rear_rad','rad','body rear-axle velocity sideslip used by the new formula'),
       ('s_predicted_rear_m','m','original TUM-line station matched to predicted rear axle'),
       ('e_rear_m','m','actual rear-axle signed lateral offset from original TUM line'),
       ('rear_velocity_heading_error_rad','rad','actual rear velocity heading minus tangent at rear match'),
       ('speed_error_mps','m/s','body vx minus the commanded speed during this control interval'),
       ('CoG_velocity_heading_error_rad','rad','CoG velocity heading minus tangent at CoG match')]


def run_trial(context,formula,factor=None,constant_speed=None):
    backend,track,geo,profile,setup,env_cfg=(context[key] for key in ('backend','track','geo','profile','setup','cfg'))
    raceline=geo.raceline
    driver_cfg=replace(env_cfg.demo,brake_cap=1.)
    driver=RearPursuit(raceline,driver_cfg,formula,pedal_deadzone=env_cfg.pedal_deadzone,
                       control_dt_s=CONTROL_DT,car=env_cfg.car)
    start_path=float((-env_cfg.eval_run_in_m)%track.length)
    start_ref=raceline.match(track.point(start_path))
    x,y=raceline.point(start_ref)
    speed=constant_speed if constant_speed is not None else factor*profile.speed(start_ref)
    state=backend.reset(setup,StartPose(float(x),float(y),raceline.heading(start_ref),speed))
    initial_on_track=bool(track.contains_car(state['x'],state['y'],state['yaw'],context['footprint']))
    initial_pose=dict(state)
    geo.reset(s_path=start_path,s_raceline=start_ref,s_path_ref=geo.path_ref.match((x,y)))
    clock=ocd_trace.LapClock(track,start_path,3)
    path,station,dist,t,steps,stall=start_path,start_ref,0.,0.,0,0.
    rows=[];cause=None;rear_hint=None;started=time.perf_counter()
    # Diagnostic cap is long enough for two 12m/s geometric-test laps.
    maximum_sim_time=1200.
    while cause is None:
        target=constant_speed if constant_speed is not None else factor*profile.speed(station+max(state['vx'],0.)*driver_cfg.speed_preview_s)
        steer=driver.steering(state,station)
        throttle,brake=driver.pedals(target,state['vx'],max(abs(state['slip_rl']),abs(state['slip_rr'])))
        inputs=Inputs(steer,throttle,brake);inside=initial_on_track if steps==0 else True
        for _ in range(CONTROL_STEPS):
            state=backend.step(inputs);steps+=1
            if not all(math.isfinite(value) for value in state.values()):
                cause='nonfinite';break
            inside &= bool(track.contains_car(state['x'],state['y'],state['yaw'],context['footprint']))
        previous,t=t,steps*DT
        if cause is not None:break
        geometry=geo.locate(state['x'],state['y'],state['yaw'])
        before=dist
        dist+=(geometry['s_path_m']-path+track.length/2)%track.length-track.length/2
        path,station=geometry['s_path_m'],geometry['s_raceline_m']
        clock.update(before,dist,previous,t)
        rear,_,_,beta,_=rear_state(state,driver.rear_axle_m)
        rear_hint=raceline.match(rear,rear_hint)
        rear_error,rear_heading=raceline.errors_at(rear_hint,float(rear[0]),float(rear[1]),state['yaw']+beta)
        row=dict(t_s=t,lap=clock.lap,t_lap_s=clock.t_lap(t),progress_m=dist,**geometry,
                 **ocd_trace.backend_sample(backend,state,inputs),on_track=int(inside),target_speed_mps=target,
                 **driver.diagnostics,e_rear_m=rear_error,rear_velocity_heading_error_rad=rear_heading,
                 speed_error_mps=state['vx']-target,
                 CoG_velocity_heading_error_rad=(geometry['heading_err_raceline_rad']+math.atan2(state['vy'],state['vx'])+math.pi)%(2*math.pi)-math.pi)
        rows.append(row)
        speed_magnitude=row['speed_mps']
        stall=stall+(t-previous) if speed_magnitude<env_cfg.stall_speed_mps else 0.
        if abs(geometry['heading_err_raceline_rad'])>env_cfg.wrong_way_rad:cause='spin'
        elif stall>env_cfg.stall_time_s:cause='stall'
        elif clock.lap==3:cause='lap2_complete'
        elif t>=maximum_sim_time:cause='time_limit'
    times=clock.lap_times
    summary={'formula':asdict(formula),'factor':factor,'constant_speed_mps':constant_speed,
             'termination':cause,'completed_lap2':cause=='lap2_complete','lap_times_s':times,
             'lap2_s':times[1] if len(times)==2 else None,'progress_m':dist,'sim_time_s':t,
             'maximum_sim_time_s':maximum_sim_time,'wall_s':time.perf_counter()-started,'lap2_metrics':metrics(rows,raceline.length,clock.crossings),
             'terminal_sample':rows[-1] if rows else None,
             'footprint_limits':'recorded every2ms, do not terminate; diagnostic NOT a clean lap',
             'recorder_crossings_s':clock.crossings,'initial_pose':initial_pose,'initial_on_track':initial_on_track,
             'flag_semantics':'initial pose checked; each25Hz interval ANDs every2ms footprint check, first interval includes initial pose'}
    return rows,summary


def select_best(attempts):
    """Rank geometry only within an equal speed condition and enforce speed fairness."""
    baselines=[a for a in attempts if a['formula']['mode']=='original' and a['completed_lap2'] and
               a['lap2_metrics'].get('full_station_coverage',False)]
    baseline_mean=baselines[0]['lap2_metrics']['mean_actual_speed_mps'] if baselines else None
    eligible=[]
    for attempt in attempts:
        measured=attempt['lap2_metrics']
        reason=None
        if not attempt['completed_lap2'] or not measured.get('full_station_coverage',False):
            reason='incomplete flying lap or incomplete station coverage'
        elif attempt['constant_speed_mps'] is not None:
            if measured['speed_error_mps']['rms']>2.:reason='constant-speed tracking RMS error exceeds2m/s'
        elif baseline_mean is None:
            reason='original PP baseline mean speed unavailable'
        elif measured['mean_actual_speed_mps']<.95*baseline_mean:
            reason='tracking-versus-speed tradeoff: mean actual speed >5% below original PP at same TUM fraction'
        attempt['tracking_eligible']=reason is None
        attempt['tracking_ineligible_reason']=reason
        attempt['original_PP_mean_speed_mps']=baseline_mean
        if reason is None:eligible.append(attempt)
    return min(eligible,key=lambda a:(a['lap2_metrics']['tracking_score_m'],
                    a['lap2_metrics']['CoG_error_m']['max_abs'],a['trial_index'])) if eligible else None


def write_trace(out,name,rows,summary,protocol):
    path=out/f'{name}.csv'
    import csv
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=[col[0] for col in common.PP_COLUMNS+EXTRA])
        writer.writeheader();writer.writerows(common._clean(rows))
    common.write_meta(path,{'source':'pure_pursuit_formula_tracking_diagnostic','protocol':'protocol.json',
                           'protocol_sha256':sha256_file(out/'protocol.json'),'summary':summary,
                           'conditions':protocol['conditions'],'lap_time_s':summary['lap2_s'],
                           'valid_clean_lap':False,'schema':common.PP_COLUMNS+EXTRA})
    return path.name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/mac/D2/p1_TUM/seed0')
    parser.add_argument('--tum-csv',default='comparison/tum/tum_none.csv')
    parser.add_argument('--out',required=True)
    parser.add_argument('--max-trials',type=int,default=40)
    parser.add_argument('--wall-seconds',type=float,default=900.)
    parser.add_argument('--constant-speed',type=float,default=12.)
    parser.add_argument('--target-mode',choices=['constant','tum'])
    parser.add_argument('--factor',type=float,help='single TUM fraction; alias for --factors')
    parser.add_argument('--factors',nargs='+',type=float,help='use TUM fraction targets instead of constant-speed stage')
    parser.add_argument('--bases',nargs='+',type=float,default=[1.5,.5,3.])
    parser.add_argument('--lookaheads',nargs='+',type=float,default=[.12,.08,.04,.2,.35])
    parser.add_argument('--predictions',nargs='+',type=float,default=[0.,.04,.08])
    parser.add_argument('--beta-weights',nargs='+',type=float,default=[0.,.5,1.])
    parser.add_argument('--understeer-gains',nargs='+',type=float,default=[0.])
    parser.add_argument('--cross-gains',nargs='+',type=float,default=[0.])
    parser.add_argument('--yaw-gains',nargs='+',type=float,default=[0.])
    args=parser.parse_args()
    if args.factor is not None:
        if args.factors:parser.error('use --factor or --factors, not both')
        args.factors=[args.factor]
    if args.target_mode=='constant' and args.factors:parser.error('constant mode cannot have TUM fractions')
    if args.target_mode=='tum' and not args.factors:args.factors=[.4,.6,.8,1.]
    if args.max_trials<1 or not math.isfinite(args.wall_seconds) or args.wall_seconds<=0:
        parser.error('positive finite budgets required')
    if not math.isfinite(args.constant_speed) or not 3.<=args.constant_speed<=80.:
        parser.error('constant speed must lie in [3,80]m/s')
    if args.factors and not all(math.isfinite(f) and 0<f<=1 for f in args.factors):
        parser.error('TUM fractions must lie in (0,1]')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg_bytes=(Path(args.run)/'config.json').read_bytes();saved_cfg=train_config_from_dict(json.loads(cfg_bytes));cfg=saved_cfg.env
    if cfg.setup!='p1_TUM' or cfg.car!='sf70h_tum' or cfg.ers or cfg.max_brake_torque_Nm!=15000.:
        raise ValueError('requires saved noERS p1_TUM/sf70h_tum/15kNm')
    (out/'config.json').write_bytes(cfg_bytes)
    track=Track(cfg.track,cfg.track_vmax_mps);geo=ocd_trace.Geometry(track)
    tum_csv=Path(args.tum_csv).resolve();tum_meta=json.loads(tum_csv.with_suffix('.meta.json').read_text())
    if tum_meta['variant']!='none' or tum_meta['sha256']!=sha256_file(tum_csv):
        raise ValueError('verified original TUM none CSV required')
    code=run_provenance(track.name)
    differences=comparison_state.source_differences(json.loads((Path(args.run)/'provenance.json').read_text()),code)
    if differences:raise RuntimeError('saved simulator source differs: '+'; '.join(differences))
    backend=OCDBackend(powertrain='engine',car=cfg.car,ers=False,max_brake_torque_Nm=cfg.max_brake_torque_Nm)
    context={'backend':backend,'track':track,'geo':geo,'profile':TumProfile(tum_csv,geo.raceline),
             'setup':setups.get(cfg.setup),'cfg':cfg,'footprint':cars.get(cfg.car).footprint,
             'TUM_lap_time_s':tum_meta['lap_time_s']}
    configs=[FormulaConfig(mode='original',base_m=cfg.demo.lookahead_m,lookahead_s=cfg.demo.lookahead_s)]
    configs += [FormulaConfig(base_m=a,lookahead_s=tau,prediction_s=prediction,beta_weight=beta,
                              understeer_gain=ku,cross_gain=ge,yaw_gain=kr)
                for a,tau,prediction,beta,ku,ge,kr in itertools.product(args.bases,args.lookaheads,args.predictions,
                    args.beta_weights,args.understeer_gains,args.cross_gains,args.yaw_gains)]
    speeds=[(factor,None) for factor in args.factors] if args.factors else [(None,args.constant_speed)]
    plan=[{'formula':asdict(formula),'factor':factor,'constant_speed_mps':speed}
          for factor,speed in speeds for formula in configs]
    protocol={'label':'Pure-pursuit formula tracking accuracy on original TUM line; diagnostic',
              'conditions':{'car':cfg.car,'setup_name':cfg.setup,'setup':context['setup'].to_dict(),
               'ers':False,'drs':False,'recuperation':False,'max_brake_torque_Nm':cfg.max_brake_torque_Nm,
               'physics_dt_s':DT,'control_dt_s':CONTROL_DT,'footprint_m':context['footprint'],
               'footprint_exit_terminates':False,'start':'original TUM line abeam path -200m; no settle (existing TUM diagnostic)',
               'not_matched_RL_reset':True,'simulation_time_limit_s':1200.},'TUM_sha256':tum_meta['sha256'],
              'TUM_lap_time_s':tum_meta['lap_time_s'],'line_digest':geo.raceline.digest,'predeclared_plan':plan,
              'formula':'predicted rear point rematch, rear velocity heading blend, standard PP curvature; no added curvature feedforward',
              'selection':'minimum station-weighted CoG RMS +.25*peak, then peak/trial ID; equal speed conditions only; constant RMSspeed<=2m/s or mean speed>=95%original PP at same fraction',
              'identity':{'config_sha256':sha256_file(out/'config.json'),'ocd_config_sha256':sha256_file(CONFIG),
                          'setup_sha256':__import__('hashlib').sha256(json.dumps(context['setup'].to_dict(),sort_keys=True).encode()).hexdigest()},
              'code':{**code,'script_sha256':sha256_file(__file__),'ocd_native_sha256':sha256_file(ocd_native.__file__)},
              'learned_weights_loaded':False,'RL_trajectory_used':False,'command':sys.argv}
    base_time=time.perf_counter();attempts=[];best_by_speed={}
    comparison_state.write_json(out/'protocol.json',common._clean(protocol))
    def save_json(path,data):
        temporary=path.with_suffix(path.suffix+'.next');comparison_state.write_json(temporary,common._clean(data));temporary.replace(path)
    for index,item in enumerate(plan):
        if index>=args.max_trials or time.perf_counter()-base_time>=args.wall_seconds:break
        rows,summary=run_trial(context,FormulaConfig(**item['formula']),item['factor'],item['constant_speed_mps'])
        summary['trial_index']=index
        key=f'f{item["factor"]}' if item['factor'] is not None else f'constant{item["constant_speed_mps"]}'
        summary['speed_condition']=key
        candidates=[a for a in attempts if a['speed_condition']==key]+[summary]
        best=select_best(candidates)
        if best is not None and best['trial_index']==index:
            summary['trace']=write_trace(out,f'best_trial{index:04d}',rows,summary,protocol)
            best_by_speed[key]=summary
        summary['speed_condition']=key;attempts.append(summary)
        save_json(out/f'trial_{index:04d}.json',summary)
        save_json(out/'summary.json',{'complete':False,'attempts':attempts,'best_by_speed':best_by_speed})
        print(f'trial{index:04d} {key} {item["formula"]} complete={summary["completed_lap2"]} '
              f'metrics={summary["lap2_metrics"]}',flush=True)
    repeats={}
    for key,best in best_by_speed.items():
        rows,repeat=run_trial(context,FormulaConfig(**best['formula']),best['factor'],best['constant_speed_mps'])
        repeat['metrics_match']=repeat['lap2_metrics']==best['lap2_metrics']
        repeat['lap_time_matches']=repeat['lap2_s']==best['lap2_s']
        repeat['verified']=repeat['completed_lap2'] and repeat['metrics_match'] and repeat['lap_time_matches']
        repeat['trace']=write_trace(out,f'repeat_{key}',rows,repeat,protocol);repeats[key]=repeat
        if not repeat['verified']:raise RuntimeError('tracking metrics/time did not repeat exactly')
    save_json(out/'summary.json',{'complete':True,'attempts':attempts,'best_by_speed':best_by_speed,
              'repeats':repeats,'elapsed_wall_s':time.perf_counter()-base_time,'no_torch_loaded':'torch' not in sys.modules})
    print(json.dumps({'trials':len(attempts),'conditions':list(best_by_speed),'no_torch':'torch' not in sys.modules}),flush=True)


if __name__=='__main__':main()
