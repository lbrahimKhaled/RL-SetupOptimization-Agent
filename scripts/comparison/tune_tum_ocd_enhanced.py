#!/usr/bin/env python3
"""Strict nonlearned OCD-tuned, TUM-derived laps with feedback and local speed envelopes.

Separate from the phase-1 runner. The saved p1 reset/settle and simulator are unchanged.
The speed planner uses measured OCD g-g-v/traction bounds; these tuned targets and global
25 Hz brake-slip assistance are explicitly not the original TUM controller or actuator data.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import itertools
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
from fair_tum_baseline import evaluation_config, choose_best, COLUMNS
from tum_pure_pursuit import TumProfile
from rlsetup.driving import qss_profile
from rlsetup.driving.demo_driver import DemoDriver, inputs_to_action
from rlsetup.driving.train_config import train_config_from_dict
from rlsetup.driving.reference import track_grid
from rlsetup.provenance import run_provenance, sha256_file
from ocd_lap.track import Track
from ocd_lap.env import MAX_STEER_RAD
import ocd_native


@dataclass(frozen=True)
class FeedbackConfig:
    mode: str = 'feedback'
    heading_gain: float = 1.
    cross_gain: float = 2.
    beta_gain: float = .5
    yaw_gain: float = .15
    understeer_gain: float = .001
    soft_speed_mps: float = 5.
    curvature_preview_s: float = .1
    slip_limit: float = 0.  # zero disables the global 25 Hz brake-slip limiter
    brake_release_per_s: float = 8.
    brake_reapply_per_s: float = 2.

    def __post_init__(self):
        if self.mode not in ('feedback','pp'):
            raise ValueError('controller mode must be feedback or pp')
        limits={'heading_gain':(0.,4.),'cross_gain':(0.,10.),'beta_gain':(0.,3.),
                'yaw_gain':(0.,1.),'understeer_gain':(0.,.01),'soft_speed_mps':(.1,20.),
                'curvature_preview_s':(0.,.5),'slip_limit':(0.,.4),
                'brake_release_per_s':(.1,30.),'brake_reapply_per_s':(.1,10.)}
        for key,(lo,hi) in limits.items():
            value=getattr(self,key)
            if not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'{key} must be finite in [{lo}, {hi}]')


class SpeedEnvelope:
    """Periodic local curvature / combined braking / traction feasibility approximation.

    It is a conservative tuning aid, not a proof of dynamic feasibility. No targets exceed
    the spatially mapped original TUM speed. Measured bounds are interpolated in speed.
    """
    def __init__(self,candidate,profile,ggv,traction,lat_margin=1.,brake_margin=1.,grid_m=2.):
        if not 0<lat_margin<=1 or not 0<brake_margin<=1 or grid_m<=0:
            raise ValueError('speed envelope margins must lie in (0,1] and grid must be positive')
        self.length=candidate.line.length
        count=int(math.ceil(self.length/grid_m))
        self.s=np.arange(count)*self.length/count
        self.ds=self.length/count
        # A 10 m centered tangent window filters polyline vertex curvature. All geometry
        # remains unchanged; this smoothing affects the planning/feedback estimate only.
        ahead=candidate.line.point(self.s+2.5)-candidate.line.point(self.s-2.5)
        heading=np.unwrap(np.arctan2(ahead[:,1],ahead[:,0]))
        # Angle differences wrap, so the final/first boundary has no artificial spike.
        dh=(np.roll(heading,-1)-np.roll(heading,1)+np.pi)%(2*np.pi)-np.pi
        self.kappa=dh/(2*self.ds)
        self.tum=np.array([candidate.target(s,profile)[1] for s in self.s])
        ggv=np.asarray(ggv,dtype=float);traction=np.asarray(traction,dtype=float)
        if (ggv.ndim!=2 or ggv.shape[1]!=3 or traction.ndim!=2 or traction.shape[1]!=2 or
                not np.isfinite(ggv).all() or not np.isfinite(traction).all() or
                np.any(ggv[:,1:]<=0) or np.any(traction[:,1]<=0) or
                np.any(np.diff(ggv[:,0])<=0) or np.any(np.diff(traction[:,0])<=0)):
            raise ValueError('invalid measured OCD ggv / traction table')
        curve=np.abs(self.kappa)
        v=self.tum.copy()
        for _ in range(30):
            ay=lat_margin*np.interp(v,ggv[:,0],ggv[:,2])
            cap=np.sqrt(ay/np.maximum(curve,1e-9))
            next_v=np.minimum(v,cap)
            if np.max(np.abs(next_v-v))<1e-8:
                v=next_v;break
            v=next_v
        else:
            raise RuntimeError('lateral envelope did not converge')
        # Closed backward / forward sweeps carry braking and acceleration constraints
        # through the timing line. Combined demand uses a labeled friction-circle aid.
        for _ in range(40):
            old=v.copy()
            for i in range(count-1,-1,-1):
                j=(i+1)%count
                ay=lat_margin*np.interp(v[i],ggv[:,0],ggv[:,2])
                ratio=min(1.,v[i]**2*curve[i]/ay)
                ax=brake_margin*np.interp(v[i],ggv[:,0],ggv[:,1])*math.sqrt(max(0.,1-ratio*ratio))
                v[i]=min(v[i],math.sqrt(v[j]**2+2*ax*self.ds))
            for i in range(count):
                j=(i+1)%count
                ay=lat_margin*np.interp(v[i],ggv[:,0],ggv[:,2])
                ratio=min(1.,v[i]**2*curve[i]/ay)
                ax=np.interp(v[i],traction[:,0],traction[:,1])*math.sqrt(max(0.,1-ratio*ratio))
                v[j]=min(v[j],math.sqrt(v[i]**2+2*ax*self.ds))
            if np.max(np.abs(v-old))<1e-6:
                break
        else:
            raise RuntimeError('closed speed envelope did not converge')
        if np.any(v<=0) or not np.isfinite(v).all():
            raise ValueError('nonfinite/nonpositive speed envelope')
        self.v=v
        self.meta={'label':'OCD-tuned TUM-derived local speed envelope','grid_m':self.ds,
                   'lateral_margin':lat_margin,'braking_margin':brake_margin,
                   'combined_demand':'friction circle, measured OCD ggv/traction; heuristic bound',
                   'curvature_estimate':'5m tangent chord, centered 2-grid-step heading derivative',
                   'fraction_reduced_from_TUM':float(np.mean(v<self.tum-1e-6)),
                   'v_min_mps':float(v.min()),'v_max_mps':float(v.max())}

    def at(self,station):
        s=float(station)%self.length
        return (float(np.interp(s,self.s,self.v,period=self.length)),
                float(np.interp(s,self.s,self.kappa,period=self.length)))


class FeedbackDriver(DemoDriver):
    """One steering + global throttle/brake command per saved 25 Hz decision."""
    def __init__(self,line,config,feedback,envelope,**kwargs):
        super().__init__(line,config,**kwargs)
        self.feedback,self.envelope=feedback,envelope
        self.brake_ceiling=1.
        self.diagnostics={}

    def command(self,state,station,target,telemetry):
        c=self.feedback
        vx=max(state['vx'],0.)
        e,heading=self.ref.errors_at(station,state['x'],state['y'],state['yaw'])
        beta=math.atan2(state['vy'],max(vx,.1))
        _,kappa=self.envelope.at(station+vx*c.curvature_preview_s)
        ff=math.atan(self.wheelbase_m*kappa)+c.understeer_gain*vx*vx*kappa
        yaw_correction=c.yaw_gain*(vx*kappa-state['yaw_rate'])
        if c.mode=='feedback':
            steer=ff-c.heading_gain*heading-math.atan(c.cross_gain*e/(vx+c.soft_speed_mps))-c.beta_gain*beta+yaw_correction
        else:
            # PP already supplies positional/heading feedback; do not add a second
            # heading/cross term here. Understeer compensation and dynamic damping only.
            steer=self.steering(state,station)+c.understeer_gain*vx*vx*kappa-c.beta_gain*beta+yaw_correction
        steer=float(np.clip(steer,-MAX_STEER_RAD,MAX_STEER_RAD))
        throttle,brake=self.pedals(target,state['vx'],max(abs(state['slip_rl']),abs(state['slip_rr'])))
        slip=max(0.,-min(float(telemetry[f'kappa_{wheel}']) for wheel in ('FL','FR','RL','RR')))
        if c.slip_limit>0:
            delta=-c.brake_release_per_s if slip>c.slip_limit else c.brake_reapply_per_s
            self.brake_ceiling=float(np.clip(self.brake_ceiling+delta*self.dt,0.,1.))
            brake=min(brake,self.brake_ceiling)
        self.diagnostics={'e_controller_m':e,'heading_controller_rad':heading,'kappa_controller_radpm':kappa,
                          'brake_ceiling':self.brake_ceiling,'measured_braking_slip':slip}
        return inputs_to_action(steer,throttle,brake,self.pedal_deadzone,self.cfg.released_pedal_action)


EXTRA=[('e_controller_m','m','signed left offset to tuned controller line'),
       ('heading_controller_rad','rad','yaw minus controller-line tangent'),
       ('kappa_controller_radpm','1/m','smoothed candidate curvature used by controller'),
       ('brake_ceiling','-','global 25Hz brake-slip limiter ceiling; one means no clipping'),
       ('measured_braking_slip','-','maximum negative wheel kappa at command start')]


def run_trial(env,candidate,profile,envelope,params,seed):
    started=time.perf_counter()
    env.reset(seed=int(seed),options={'evaluation':True})
    config=replace(env.cfg.demo,brake_cap=1.,lookahead_s=params['lookahead_s'],
                   speed_kp=params['speed_kp'],speed_ki=params['speed_ki'],speed_preview_s=params['speed_preview_s'])
    feedback=FeedbackConfig(**params['feedback'])
    driver=FeedbackDriver(candidate.line,config,feedback,envelope,pedal_deadzone=env.cfg.pedal_deadzone,
                          control_dt_s=env.decision_dt,car=env.cfg.car)
    driver.reset(env.settle_driver.integral)
    hint,total=None,0.
    while True:
        state=env.state
        hint=candidate.line.match((state['x'],state['y']),hint)
        preview=hint+max(state['vx'],0.)*config.speed_preview_s
        target=params['factor']*envelope.at(preview)[0]
        # Public read-only telemetry, sampled once per decision. No wheel-level actuator
        # commands, hidden ABS or faster-than-25Hz control are introduced.
        action=driver.command(state,hint,target,env.backend.get_telemetry())
        s_tum,_=candidate.target(preview,profile)
        env.command_context={'target_speed_mps':target,'s_controller_m':hint,
                             's_preview_controller_m':preview,'s_preview_tum_m':s_tum,
                             'speed_scale_f':params['factor'],**driver.diagnostics}
        _,reward,term,trunc,info=env.step(action);total+=reward
        if term or trunc:
            break
    times=[lap['lap_time_s'] for lap in info['laps']]
    recorder=check(env,SimpleNamespace(lap_times_s=times))
    if not recorder['lap_times_match_env']:
        raise RuntimeError('recorder and environment clocks differ')
    clean=info['termination']=='lap_complete' and info['truncation'] is None and len(times)==2
    return env.rows,{'params':params,'clean':bool(clean),'termination':info['termination'],'truncation':info['truncation'],
                     'lap_times_s':times,'lap2_s':times[1] if len(times)==2 else None,'recorder_check':recorder,
                     'laps':info['laps'],'decisions':env.decision,'sim_time_s':info['time_s'],
                     'progress_m':info['progress_m'],'episode_return':total,'wall_s':time.perf_counter()-started,
                     'handover':env.handover,'terminal_state':dict(env.state),
                     'terminal_sample':dict(env.rows[-1]) if env.rows else None,'controller_config':asdict(config),
                     'feedback_config':asdict(feedback),'speed_envelope':envelope.meta,'planned_geometry':candidate.planned}


def export_trace(out,name,rows,summary,protocol):
    path=out/f'{name}.csv'
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=[column[0] for column in COLUMNS+EXTRA])
        writer.writeheader();writer.writerows(common._clean(rows))
    common.write_meta(path,{'source':'OCD_tuned_TUM_derived_feedback','protocol':'protocol.json',
                           'protocol_sha256':sha256_file(out/'protocol.json'),'conditions':protocol['conditions'],
                           'summary':summary,'lap_time_s':summary['lap2_s'],'learned_weights_loaded':False,
                           'controller_label':'nonlearned feedback/PP, optional global brake-slip aid',
                           'schema':COLUMNS+EXTRA})
    return path.name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/mac/D2/p1_TUM/seed0')
    parser.add_argument('--tum-csv',default='comparison/tum/tum_none.csv')
    parser.add_argument('--out',required=True)
    parser.add_argument('--max-trials',type=int,default=32)
    parser.add_argument('--wall-seconds',type=float,default=900.)
    arrays={'alphas':[.5,.6,.8,1.],'factors':[1.],'lookaheads':[.16,.22],
            'lat-margins':[.9,1.],'brake-margins':[.85],
            'heading-gains':[1.],'cross-gains':[2.],'beta-gains':[.5],
            'yaw-gains':[.15],'understeer-gains':[.001],'slip-limits':[0.,.12],
            'speed-previews':[.1]}
    for name,default in arrays.items():
        parser.add_argument('--'+name,type=float,nargs='+',default=default)
    parser.add_argument('--modes',nargs='+',choices=['feedback','pp'],default=['feedback','pp'])
    parser.add_argument('--kp',type=float,default=.4);parser.add_argument('--ki',type=float,default=.2)
    args=parser.parse_args()
    for name in arrays:
        values=getattr(args,name.replace('-','_'))
        if not all(math.isfinite(value) for value in values) or len(set(values))!=len(values):
            parser.error(f'{name} must be unique finite values')
    if args.max_trials<=0 or not math.isfinite(args.wall_seconds) or args.wall_seconds<=0:
        parser.error('positive trial/wall budgets required')
    if not all(0<=a<=1 for a in args.alphas) or not all(.5<=f<=1 for f in args.factors):
        parser.error('alphas in [0,1], factors in [.5,1] required')
    if not all(0<m<=1 for m in args.lat_margins+args.brake_margins):
        parser.error('envelope margins must lie in (0,1]')
    start=time.perf_counter();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    run=Path(args.run).resolve();cfg_bytes=(run/'config.json').read_bytes()
    cfg=train_config_from_dict(json.loads(cfg_bytes));env_cfg=evaluation_config(cfg.env)
    if (env_cfg.setup!='p1_TUM' or env_cfg.car!='sf70h_tum' or env_cfg.ers or
            env_cfg.max_brake_torque_Nm!=15000. or env_cfg.reference!='path' or
            env_cfg.action_repeat!=2 or env_cfg.settle_s!=2.):
        raise ValueError('requires saved p1_TUM/15kNm/25Hz/2s settle/noERS/path reset')
    (out/'config.json').write_bytes(cfg_bytes)
    track=Track(env_cfg.track,env_cfg.track_vmax_mps);code=run_provenance(track.name)
    saved=json.loads((run/'provenance.json').read_text());differences=comparison_state.source_differences(saved,code)
    if differences:
        raise RuntimeError('saved simulator sources differ: '+'; '.join(differences))
    tum_csv=Path(args.tum_csv).resolve();tum_meta=json.loads(tum_csv.with_suffix('.meta.json').read_text())
    if (tum_meta['variant']!='none' or tum_meta['sha256']!=sha256_file(tum_csv) or
            tum_meta['ers_used_J']!=0 or tum_meta['drs_share_of_points']!=0):
        raise ValueError('verified noERS/noDRS TUM CSV required')
    qss=qss_profile.for_env(track,env_cfg)
    env=RecordingEnv(env_cfg,track=track,qss=qss)
    try:
        grid=track_grid(track,env.geo.raceline,1.)
        safe_xy=np.column_stack([np.interp(grid.s,track.path[:,0],track.path[:,i]) for i in (1,2)])
        candidates=[]
        for alpha in sorted(args.alphas):
            candidate=base.blend_candidate(alpha,safe_xy,grid.ref,grid.s_ref,env.geo.raceline.length)
            base.screen_candidate(candidate,track,env.footprint)
            if candidate.planned['all_sampled_footprints_inside']:
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError('none of requested geometries passed planned footprint screen')
        profile=TumProfile(tum_csv,env.geo.raceline)
        envelopes={(c.alpha,lat,brake):SpeedEnvelope(c,profile,qss.meta['ggv'],qss.meta['traction'],lat,brake)
                   for c,lat,brake in itertools.product(candidates,args.lat_margins,args.brake_margins)}
        configs=[FeedbackConfig(mode=mode,heading_gain=heading,cross_gain=cross,beta_gain=beta,
                                yaw_gain=yaw,understeer_gain=us,slip_limit=slip)
                 for mode,heading,cross,beta,yaw,us,slip in itertools.product(args.modes,args.heading_gains,
                   args.cross_gains,args.beta_gains,args.yaw_gains,args.understeer_gains,args.slip_limits)]
        plan=[{'alpha':c.alpha,'factor':factor,'lookahead_s':lookahead,'speed_kp':args.kp,'speed_ki':args.ki,
               'speed_preview_s':preview,'lat_margin':lat,'brake_margin':brake,'feedback':asdict(feedback)}
              for feedback,lat,brake,lookahead,preview,factor,c in itertools.product(configs,args.lat_margins,
                args.brake_margins,args.lookaheads,args.speed_previews,args.factors,candidates)]
        conditions={'car':env_cfg.car,'setup_name':env_cfg.setup,'setup':env.setup.to_dict(),
                    'max_brake_torque_Nm':env_cfg.max_brake_torque_Nm,'ers':False,'drs':False,'recuperation':False,
                    'decision_dt_s':env.decision_dt,'tick_dt_s':env.dt_tick,'physics_dt_s':.002,
                    'footprint_m':env.footprint,'seed':cfg.eval.seed,'settle_s':env_cfg.settle_s,
                    'run_in_m':env_cfg.eval_run_in_m,'strict_footprint_termination':True}
        protocol={'source':'OCD_tuned_TUM_derived_nonlearned','conditions':conditions,
                  'saved_run':str(run),'config_sha256':sha256_file(out/'config.json'),
                  'evaluation_env_config':env_cfg.to_dict(),'tum_input_sha256':tum_meta['sha256'],
                  'TUM_model_caveat':'Different QSS vehicle; no superiority or achievable-time guarantee',
                  'ggv':qss.meta['ggv'],'traction':qss.meta['traction'],'ggv_cache_key':qss.meta['key'],
                  'predeclared_plan':plan,'selected_alphas':[c.alpha for c in candidates],
                  'alpha_convention':'0=TUM,1=safe; XY=(1-alpha)*TUM+alpha*safe',
                  'assistance':'optional global brake-slip limiter at25Hz only, no individual wheel actuators',
                  'learned_weights_loaded':False,'RL_trajectory_used':False,
                  'code':{**code,'ocd_native_sha256':sha256_file(ocd_native.__file__),
                          'script_sha256':sha256_file(__file__),'phase1_helpers_sha256':sha256_file(base.__file__)},
                  'command':sys.argv,'selection_rule':'minimum clean genuine lap2, same seed/handovers exact repeat',
                  'budget_policy':'wall checked between trials; final validation repeat may exceed budget by one run'}
        base.write_json(out/'protocol.json',protocol)
        base.write_json(out/'geometry_screen.json',{'planned_only':True,'candidates':[
            {'alpha':c.alpha,**c.planned} for c in candidates]})
        for index,c in enumerate(candidates):
            np.savetxt(out/f'line_{index:02d}_alpha{c.alpha:.6g}.csv',np.column_stack([c.line.path,c.tum_station]),
                       delimiter=',',header='s_candidate_m,x_m,y_m,s_tum_abeam_m',comments='')
        for index,(key,envelope) in enumerate(envelopes.items()):
            np.savetxt(out/f'envelope_{index:02d}.csv',np.column_stack([envelope.s,envelope.tum,envelope.v,envelope.kappa]),
                       delimiter=',',header='s_candidate_m,TUM_target_mps,envelope_target_mps,kappa_radpm',comments='')
        attempts=[];best=None;stop='plan_complete'
        for index,params in enumerate(plan):
            if index>=args.max_trials:
                stop='max_trials';break
            if time.perf_counter()-start>=args.wall_seconds:
                stop='wall_budget';break
            candidate=next(c for c in candidates if c.alpha==params['alpha'])
            envelope=envelopes[(params['alpha'],params['lat_margin'],params['brake_margin'])]
            rows,summary=run_trial(env,candidate,profile,envelope,params,cfg.eval.seed)
            summary['trial_index']=index;attempts.append(summary)
            new_best=choose_best(attempts)
            if new_best is summary and (best is None or summary['lap2_s']<best['lap2_s']):
                summary['trace']=export_trace(out,f'best_trial{index:04d}',rows,summary,protocol);best=summary
            base.write_json(out/f'trial_{index:04d}.json',summary)
            base.write_json(out/'summary.json',{'complete':False,'attempts':attempts,'best':best})
            print(f'trial{index:04d} alpha={params["alpha"]} {params["feedback"]} '
                  f'clean={summary["clean"]} lap2={summary["lap2_s"]} '
                  f'cause={summary["termination"] or summary["truncation"]}',flush=True)
        repeat=None
        if best is not None:
            params=best['params'];candidate=next(c for c in candidates if c.alpha==params['alpha'])
            envelope=envelopes[(params['alpha'],params['lat_margin'],params['brake_margin'])]
            rows,repeat=run_trial(env,candidate,profile,envelope,params,cfg.eval.seed)
            repeat['trace']=export_trace(out,'best_repeat',rows,repeat,protocol)
            repeat['lap2_difference_s']=repeat['lap2_s']-best['lap2_s'] if repeat['clean'] else None
            repeat['handover_matches']=repeat['handover']==best['handover']
            repeat['verified']=bool(repeat['clean'] and repeat['handover_matches'] and abs(repeat['lap2_difference_s'])<=1e-9)
            base.write_json(out/'repeat.json',repeat)
            if not repeat['verified']:
                raise RuntimeError('winner did not reproduce clean time and handover; traces retained')
        base.write_json(out/'summary.json',{'complete':True,'stop_reason':stop,'attempts':attempts,'best':best,
                   'repeat':repeat,'trial_count':len(attempts),'plan_count':len(plan),
                   'elapsed_wall_s':time.perf_counter()-start,'no_torch_loaded':'torch' not in sys.modules})
        print(json.dumps({'stop':stop,'trials':len(attempts),'best_lap2_s':best['lap2_s'] if best else None,
                          'repeat_verified':repeat['verified'] if repeat else None}),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
