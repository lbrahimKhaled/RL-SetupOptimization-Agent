#!/usr/bin/env python3
"""PP + position-indexed action/target corrections (non-RL), strict p1 OCD laps.

Starts by reproducing the frozen clean pure-pursuit anchor, then changes one local
zone/channel at a time. Every trial is a fresh genuine evaluation. Only a faster clean
flying lap can replace the incumbent. No learned weights or RL trajectory are used.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import tune_tum_ocd as base
import common
import comparison_state
from comparison_recording import RecordingEnv, check
from fair_tum_baseline import evaluation_config, COLUMNS
from tum_pure_pursuit import TumProfile
from rlsetup.driving import qss_profile
from rlsetup.driving.demo_driver import DemoDriver, action_to_inputs, inputs_to_action
from rlsetup.driving.reference import ReferenceLine
from rlsetup.driving.train_config import train_config_from_dict
from rlsetup.provenance import run_provenance, sha256_file
from ocd_lap.track import Track
from ocd_lap.env import MAX_STEER_RAD
import ocd_native

KNOTS=64
BOUNDS={'speed':(.5,1.05),'steer':(-.02,.02),'throttle':(-.30,.30),'brake':(-.30,.30)}
INITIAL_STEPS={'speed':.05,'steer':.002,'throttle':.03,'brake':.03}
MIN_STEPS={'speed':.025,'steer':.0005,'throttle':.0075,'brake':.0075}
MAX_STEPS={'speed':.05,'steer':.01,'throttle':.12,'brake':.12}
EXTRA=[('s_action_path_m','m','Track.path station at the start of the held command'),
       ('local_speed_factor','-','position-indexed multiplier of spatial TUM speed; not literal TUM targets unless one'),
       ('TUM_preview_speed_mps','m/s','original TUM speed at controller spatial preview'),
       ('pp_steer_rad','rad','nominal geometric PP steering before action correction'),
       ('pp_throttle','-','nominal PP/PI throttle for the locally scaled target before correction'),
       ('pp_brake','-','nominal PP/PI brake for the locally scaled target before correction'),
       ('residual_steer_rad','rad','requested position-indexed steering correction'),
       ('residual_throttle','-','requested position-indexed throttle correction'),
       ('residual_brake','-','requested position-indexed brake correction'),
       ('applied_residual_steer_rad','rad','actual steering difference after legal clamp and action encoding'),
       ('applied_residual_throttle','-','actual throttle difference after legal clamp and action encoding'),
       ('applied_residual_brake','-','actual brake difference after legal clamp and action encoding')]


def basis(station,length):
    """Periodic C1 cosine partition: at most two nonzero weights, support +/-L/64."""
    if not math.isfinite(station) or not math.isfinite(length) or length<=0:
        raise ValueError('finite station and positive finite length required')
    u=(station%length)*KNOTS/length
    i=int(math.floor(u))%KNOTS
    frac=u-math.floor(u)
    weight=.5*(1.+math.cos(math.pi*frac))
    return i,(i+1)%KNOTS,weight,1.-weight


@dataclass
class ActionProfile:
    speed: np.ndarray
    steer: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray

    @classmethod
    def anchor(cls):
        return cls(np.full(KNOTS,.7),np.zeros(KNOTS),np.zeros(KNOTS),np.zeros(KNOTS))

    @classmethod
    def from_dict(cls,data):
        result=cls(**{channel:np.asarray(data[channel],dtype=float) for channel in BOUNDS})
        result.validate();return result

    def validate(self):
        for channel,(lo,hi) in BOUNDS.items():
            values=getattr(self,channel)
            if values.shape!=(KNOTS,) or not np.isfinite(values).all() or np.any(values<lo-1e-12) or np.any(values>hi+1e-12):
                raise ValueError(f'{channel} must contain {KNOTS} finite knots in [{lo},{hi}]')

    def to_dict(self):
        return {channel:getattr(self,channel).tolist() for channel in BOUNDS}

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def value(self,channel,station,length):
        i,j,a,b=basis(station,length);values=getattr(self,channel)
        # Preserve the bit-identical scalar anchor (0.70), rather than introducing
        # roundoff from interpolating two identical floating-point values.
        return float(values[i]) if values[i]==values[j] else float(a*values[i]+b*values[j])

    def changed(self,channel,indices,delta):
        result=ActionProfile.from_dict(self.to_dict())
        values=getattr(result,channel);lo,hi=BOUNDS[channel]
        values[indices]=np.clip(values[indices]+delta,lo,hi)
        result.validate();return result


def corrected_action(nominal,profile,station,length,pedal_deadzone,released_action):
    """Physical corrections followed only by existing action encoding and legal clamps."""
    inputs=action_to_inputs(nominal,pedal_deadzone)
    residual={channel:profile.value(channel,station,length) for channel in ('steer','throttle','brake')}
    if all(value==0. for value in residual.values()):
        action=nominal # exact identity preserves the frozen pure-pursuit anchor
    else:
        action=inputs_to_action(float(np.clip(inputs.steering_rad+residual['steer'],-MAX_STEER_RAD,MAX_STEER_RAD)),
             float(np.clip(inputs.throttle+residual['throttle'],0.,1.)),
             float(np.clip(inputs.brake+residual['brake'],0.,1.)),pedal_deadzone,released_action)
    applied=action_to_inputs(action,pedal_deadzone)
    return action,{'pp_steer_rad':inputs.steering_rad,'pp_throttle':inputs.throttle,'pp_brake':inputs.brake,
                   'residual_steer_rad':residual['steer'],'residual_throttle':residual['throttle'],
                   'residual_brake':residual['brake'],
                   'applied_residual_steer_rad':applied.steering_rad-inputs.steering_rad,
                   'applied_residual_throttle':applied.throttle-inputs.throttle,
                   'applied_residual_brake':applied.brake-inputs.brake}


def run_trial(env,candidate,tum,profile,seed,fixed):
    started=time.perf_counter();env.reset(seed=int(seed),options={'evaluation':True})
    driver_cfg=replace(env.cfg.demo,brake_cap=1.,lookahead_s=fixed['lookahead_s'],
                       speed_kp=fixed['speed_kp'],speed_ki=fixed['speed_ki'])
    driver=DemoDriver(candidate.line,driver_cfg,pedal_deadzone=env.cfg.pedal_deadzone,
                      control_dt_s=env.decision_dt,car=env.cfg.car)
    driver.reset(env.settle_driver.integral)
    hint,total=None,0.
    while True:
        state=env.state;hint=candidate.line.match((state['x'],state['y']),hint)
        preview=hint+max(state['vx'],0.)*driver_cfg.speed_preview_s
        s_tum,original_speed=candidate.target(preview,tum)
        factor=profile.value('speed',env.s,env.track.length)
        target=factor*original_speed
        nominal=driver.act(state,hint,target)
        action,details=corrected_action(nominal,profile,env.s,env.track.length,
                                       env.cfg.pedal_deadzone,driver_cfg.released_pedal_action)
        env.command_context={'target_speed_mps':target,'s_controller_m':hint,
                             's_preview_controller_m':preview,'s_preview_tum_m':s_tum,'speed_scale_f':factor,
                             's_action_path_m':env.s,'local_speed_factor':factor,
                             'TUM_preview_speed_mps':original_speed,**details}
        _,reward,term,trunc,info=env.step(action);total+=reward
        if term or trunc:break
    times=[lap['lap_time_s'] for lap in info['laps']]
    recorder=check(env,SimpleNamespace(lap_times_s=times))
    if not recorder['lap_times_match_env']:raise RuntimeError('recorder and genuine environment clocks differ')
    clean=info['termination']=='lap_complete' and info['truncation'] is None and len(times)==2
    summary={'profile_sha256':profile.digest(),'clean':bool(clean),'termination':info['termination'],
             'truncation':info['truncation'],'lap_times_s':times,'lap2_s':times[1] if len(times)==2 else None,
             'recorder_check':recorder,'laps':info['laps'],'decisions':env.decision,'sim_time_s':info['time_s'],
             'progress_m':info['progress_m'],'episode_return':total,'wall_s':time.perf_counter()-started,
             'handover':env.handover,'terminal_state':dict(env.state),
             'terminal_sample':dict(env.rows[-1]) if env.rows else None,'controller_config':asdict(driver_cfg)}
    return env.rows,summary


def zones(length,width=4):
    """Main under-throttled straight first, then compact contiguous four-knot zones."""
    stations=np.arange(KNOTS)*length/KNOTS
    main=[int(i) for i in np.flatnonzero((stations>=3500.)&(stations<=4500.))]
    result=[{'name':'main_straight_3500_4500','indices':main}] if main else []
    result += [{'name':f'knots_{start:02d}_{min(start+width,KNOTS)-1:02d}',
                'indices':list(range(start,min(start+width,KNOTS)))} for start in range(0,KNOTS,width)]
    return result


def accepted(candidate,incumbent,min_improvement=.01):
    return bool(candidate['clean'] and candidate['lap2_s'] is not None and
                candidate['termination']=='lap_complete' and candidate['truncation'] is None and
                candidate['recorder_check']['lap_times_match_env'] and
                candidate['lap2_s']<=incumbent['lap2_s']-min_improvement)


def export_trace(out,name,rows,summary,protocol):
    path=out/f'{name}.csv'
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=[col[0] for col in COLUMNS+EXTRA])
        writer.writeheader();writer.writerows(common._clean(rows))
    common.write_meta(path,{'source':'PP_position_indexed_action_target_corrections_non_RL',
                           'protocol':'protocol.json','protocol_sha256':sha256_file(out/'protocol.json'),
                           'conditions':protocol['conditions'],'summary':summary,'lap_time_s':summary['lap2_s'],
                           'learned_weights_loaded':False,'schema':COLUMNS+EXTRA,
                           'nominal_PP_semantics':'PP/PI commands at actual state with current locally scaled target, before physical residuals'})
    return path.name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/mac/D2/p1_TUM/seed0')
    parser.add_argument('--baseline',default='comparison/tum_ocd_fast_20261029_122428Z/pilot_01')
    parser.add_argument('--tum-csv',default='comparison/tum/tum_none.csv')
    parser.add_argument('--out',required=True)
    parser.add_argument('--resume-state',type=Path)
    parser.add_argument('--max-trials',type=int,default=200)
    parser.add_argument('--wall-seconds',type=float,default=900.)
    parser.add_argument('--channels',nargs='+',choices=list(BOUNDS),default=list(BOUNDS))
    parser.add_argument('--zone-width-knots',type=int,default=4)
    parser.add_argument('--minimum-improvement',type=float,default=.01)
    args=parser.parse_args()
    if (args.max_trials<1 or not math.isfinite(args.wall_seconds) or args.wall_seconds<=0 or
            not 1<=args.zone_width_knots<=16 or args.minimum_improvement<.01 or
            not math.isfinite(args.minimum_improvement) or len(set(args.channels))!=len(args.channels)):
        parser.error('positive budgets, zone width1..16, unique channels, improvement>=.01 required')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    run=Path(args.run).resolve();baseline=Path(args.baseline).resolve()
    frozen_summary=json.loads((baseline/'summary.json').read_text());anchor=frozen_summary['best']
    fixed=anchor['params']
    if fixed!={'alpha':.5,'factor':.7,'lookahead_s':.22,'speed_kp':.4,'speed_ki':.2}:
        raise ValueError('requires the approved alpha.5/f.7/lookahead.22/kp.4/ki.2 pure-pursuit anchor')
    line_path=baseline/'line_alpha0.50.csv';line_data=np.loadtxt(line_path,delimiter=',',skiprows=1)
    line=ReferenceLine(line_data[:,:3],name='frozen_TUM_safe_alpha0.5')
    if line.digest!=anchor['planned_geometry']['line_digest']:
        raise ValueError('frozen controller line differs from baseline digest')
    candidate=base.Candidate(.5,line,line_data[:,3],anchor['planned_geometry'])
    anchor_csv=baseline/anchor['trace'];anchor_meta=json.loads(anchor_csv.with_suffix('.meta.json').read_text())
    if sha256_file(anchor_csv)!=anchor_meta['sha256'] or not anchor['clean']:
        raise ValueError('baseline PP recording hash/clean status mismatch')
    config_bytes=(run/'config.json').read_bytes();cfg=train_config_from_dict(json.loads(config_bytes))
    env_cfg=evaluation_config(cfg.env)
    if (env_cfg.setup!='p1_TUM' or env_cfg.car!='sf70h_tum' or env_cfg.ers or
            env_cfg.max_brake_torque_Nm!=15000. or env_cfg.reference!='path' or
            env_cfg.action_repeat!=2 or env_cfg.settle_s!=2.):
        raise ValueError('requires frozen p1_TUM15kNm/noERS/path-reset/25Hz/2s settle')
    (out/'config.json').write_bytes(config_bytes)
    track=Track(env_cfg.track,env_cfg.track_vmax_mps);code=run_provenance(track.name)
    saved=json.loads((run/'provenance.json').read_text())
    differences=comparison_state.source_differences(saved,code)
    if differences:raise RuntimeError('saved simulator source differs: '+'; '.join(differences))
    tum_path=Path(args.tum_csv).resolve();tum_meta=json.loads(tum_path.with_suffix('.meta.json').read_text())
    if (tum_meta['variant']!='none' or tum_meta['sha256']!=sha256_file(tum_path) or
            tum_meta['ers_used_J']!=0 or tum_meta['drs_share_of_points']!=0):
        raise ValueError('verified noERS/noDRS TUM profile required')
    qss=qss_profile.for_env(track,env_cfg);env=RecordingEnv(env_cfg,track=track,qss=qss)
    try:
        tum=TumProfile(tum_path,env.geo.raceline)
        identity={'config_sha256':sha256_file(out/'config.json'),'line_sha256':sha256_file(line_path),
                  'line_digest':line.digest,'TUM_sha256':tum_meta['sha256'],
                  'ocd_native_sha256':sha256_file(ocd_native.__file__),'ocd_lap_code_version':code['ocd_lap_code_version']}
        conditions={'car':env_cfg.car,'setup_name':env_cfg.setup,'setup':env.setup.to_dict(),
                    'max_brake_torque_Nm':env_cfg.max_brake_torque_Nm,'ers':False,'drs':False,'recuperation':False,
                    'decision_dt_s':env.decision_dt,'tick_dt_s':env.dt_tick,'physics_dt_s':.002,
                    'footprint_m':env.footprint,'seed':cfg.eval.seed,'settle_s':env_cfg.settle_s,
                    'run_in_m':env_cfg.eval_run_in_m,'strict_footprint_termination':True}
        zone_list=zones(track.length,args.zone_width_knots)
        coordinates=[{'channel':channel,'zone':zone} for channel in args.channels for zone in zone_list]
        protocol={'label':'PP + position-indexed action/target corrections (non-RL)',
                  'conditions':conditions,'identity':identity,'fixed_PP':fixed,
                  'anchor_csv':str(anchor_csv),'anchor_csv_sha256':anchor_meta['sha256'],
                  'anchor_lap2_s':anchor['lap2_s'],'baseline_controller_line':str(line_path),
                  'knots':KNOTS,'basis':'periodic C1 compact cosine, support +/-Track.path.length/64; partition of unity',
                  'bounds':BOUNDS,'initial_steps':INITIAL_STEPS,'minimum_steps':MIN_STEPS,'maximum_steps':MAX_STEPS,
                  'coordinates':coordinates,'selection_rule':'clean strict lap2 improves incumbent by >=minimum_improvement',
                  'minimum_improvement_s':args.minimum_improvement,'code':{**code,'script_sha256':sha256_file(__file__)},
                  'evaluation_env_config':env_cfg.to_dict(),'learned_weights_loaded':False,'RL_trajectory_used':False,
                  'local_target_label':'when local factor differs from1, targets are tuned TUM-derived, not original TUM speeds',
                  'action_semantics':'physical residual added after nominal PP/PI and traction control, then legal clamps; both pedals may be nonzero and are logged',
                  'budget_policy':'actual search wall checked between fresh full trials; anchor/final validation outside search budget',
                  'resume_state':str(args.resume_state) if args.resume_state else None,'command':sys.argv}
        base.write_json(out/'protocol.json',protocol)
        rows,baseline_report=run_trial(env,candidate,tum,ActionProfile.anchor(),cfg.eval.seed,fixed)
        baseline_report['expected_lap2_s']=anchor['lap2_s']
        baseline_report['difference_s']=baseline_report['lap2_s']-anchor['lap2_s'] if baseline_report['clean'] else None
        baseline_report['verified']=bool(baseline_report['clean'] and abs(baseline_report['difference_s'])<=1e-9)
        base.write_json(out/'anchor_reproduction.json',baseline_report)
        if not baseline_report['verified']:
            export_trace(out,'anchor_mismatch',rows,baseline_report,protocol)
            raise RuntimeError('zero-correction PP anchor failed exact reproduction; trace retained')
        best_profile=ActionProfile.anchor();best=baseline_report
        cursor={'coordinate_index':0,'direction_index':0,'step':None,'reduced_once':False}
        if args.resume_state:
            resume=json.loads(args.resume_state.read_text())
            if resume['identity']!=identity:raise ValueError('resume profile physical/input identity mismatch')
            if resume['coordinates']!=coordinates:raise ValueError('resume coordinate plan differs; preserve channels/zone width')
            best_profile=ActionProfile.from_dict(resume['best_profile'])
            rows,replayed=run_trial(env,candidate,tum,best_profile,cfg.eval.seed,fixed)
            if not replayed['clean'] or abs(replayed['lap2_s']-resume['best']['lap2_s'])>1e-9:
                raise RuntimeError('resume incumbent failed exact clean replay')
            best=replayed;cursor=resume['cursor']
        attempts=[];search_start=time.perf_counter();stop='coordinates_complete'
        def save_state():
            base.write_json(out/'state.json',{'schema':1,'identity':identity,'coordinates':coordinates,
                            'best_profile':best_profile.to_dict(),'best':best,'cursor':cursor,
                            'attempts_this_session':len(attempts),'search_wall_s':time.perf_counter()-search_start})
        save_state()
        while cursor['coordinate_index']<len(coordinates):
            if len(attempts)>=args.max_trials:stop='max_trials';break
            if time.perf_counter()-search_start>=args.wall_seconds:stop='wall_budget';break
            coordinate=coordinates[cursor['coordinate_index']];channel=coordinate['channel'];indices=coordinate['zone']['indices']
            if cursor['step'] is None:cursor['step']=INITIAL_STEPS[channel]
            directions=[1.] if channel=='speed' else [1.,-1.]
            direction=directions[cursor['direction_index']]
            delta=direction*cursor['step']
            proposed=best_profile.changed(channel,indices,delta)
            if proposed.digest()==best_profile.digest():
                cursor['coordinate_index']+=1;cursor['step']=None;cursor['direction_index']=0;cursor['reduced_once']=False
                save_state();continue
            rows,report=run_trial(env,candidate,tum,proposed,cfg.eval.seed,fixed)
            index=len(attempts);report.update(trial_index=index,coordinate=coordinate,requested_delta=delta,
                 previous_clean_lap2_s=best['lap2_s'],profile=proposed.to_dict())
            report['accepted']=accepted(report,best,args.minimum_improvement)
            if report['accepted']:
                report['improvement_s']=best['lap2_s']-report['lap2_s']
                report['trace']=export_trace(out,f'best_trial{index:04d}',rows,report,protocol)
                best,best_profile=report,proposed
                cursor['step']=min(MAX_STEPS[channel],cursor['step']*1.25)
                cursor['reduced_once']=False
            else:
                cursor['direction_index']+=1
                if cursor['direction_index']>=len(directions):
                    cursor['direction_index']=0
                    next_step=cursor['step']*.5
                    if next_step>=MIN_STEPS[channel]-1e-12 and not cursor['reduced_once']:
                        cursor['step']=max(MIN_STEPS[channel],next_step);cursor['reduced_once']=True
                    else:
                        cursor['coordinate_index']+=1;cursor['step']=None;cursor['reduced_once']=False
            attempts.append(report);base.write_json(out/f'trial_{index:04d}.json',report);save_state()
            base.write_json(out/'summary.json',{'complete':False,'attempts':attempts,'best':best,
                                             'search_wall_s':time.perf_counter()-search_start})
            print(f'trial{index:04d} {channel} {coordinate["zone"]["name"]} delta={delta:+.4f} '
                  f'clean={report["clean"]} lap2={report["lap2_s"]} accepted={report["accepted"]} '
                  f'cause={report["termination"] or report["truncation"]}',flush=True)
        search_wall_s=time.perf_counter()-search_start
        validation_started=time.perf_counter()
        rows,repeat=run_trial(env,candidate,tum,best_profile,cfg.eval.seed,fixed)
        repeat['handover_matches']=repeat['handover']==best['handover']
        repeat['difference_s']=repeat['lap2_s']-best['lap2_s'] if repeat['clean'] else None
        repeat['verified']=bool(repeat['clean'] and repeat['handover_matches'] and abs(repeat['difference_s'])<=1e-9)
        repeat['trace']=export_trace(out,'best_repeat',rows,repeat,protocol)
        base.write_json(out/'repeat.json',repeat);base.write_json(out/'best_profile.json',best_profile.to_dict())
        base.write_json(out/'summary.json',{'complete':True,'stop_reason':stop,'attempts':attempts,'best':best,
                       'repeat':repeat,'anchor':baseline_report,'trial_count':len(attempts),
                       'search_wall_s':search_wall_s,'final_validation_wall_s':time.perf_counter()-validation_started,'no_torch_loaded':'torch' not in sys.modules})
        save_state()
        if not repeat['verified']:raise RuntimeError('best clean profile failed exact final repeat')
        print(json.dumps({'trials':len(attempts),'best_lap2_s':best['lap2_s'],'repeat_verified':True,'stop':stop}),flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
