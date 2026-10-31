#!/usr/bin/env python3
"""Bounded serial, nonlearned TUM-derived trajectory / pure-pursuit tuning in strict OCD.

The original TUM QSS model is a reference, not an achievable-lap guarantee. A candidate
line blends abeam TUM geometry with the shipped safe line; only geometry and PP/PI
parameters change. The saved p1 car, setup, reset, settle, physics, timing and footprint
termination stay unchanged. No RL weights or recorded RL trajectory are read.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import itertools
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import comparison_state
from comparison_recording import RecordingEnv, check
from fair_tum_baseline import evaluation_config, choose_best, write_trace
from tum_pure_pursuit import TumProfile
from rlsetup.driving import qss_profile
from rlsetup.driving.demo_driver import DemoDriver
from rlsetup.driving.reference import ReferenceLine, track_grid
from rlsetup.driving.train_config import train_config_from_dict
from rlsetup.provenance import run_provenance, sha256_file
from ocd_lap.track import Track
import ocd_native


@dataclass
class Candidate:
    alpha: float
    line: ReferenceLine
    tum_station: np.ndarray
    planned: dict

    def target(self, station, profile):
        # Interpolate matched TUM station on genuine candidate arc length. No lap
        # fraction mapping: each knot came from sequential abeam XY matching.
        s = float(station) % self.line.length
        tum_s = float(np.interp(s, self.line.path[:, 0], self.tum_station))
        return tum_s, profile.speed(tum_s)


def blend_candidate(alpha, safe_xy, tum_xy, tum_station, tum_length):
    """Blend physical XY knots; recompute arc length and retain spatial target mapping."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite in [0, 1]')
    safe_xy, tum_xy = np.asarray(safe_xy), np.asarray(tum_xy)
    if safe_xy.shape != tum_xy.shape or safe_xy.ndim != 2 or safe_xy.shape[1] != 2:
        raise ValueError('safe and TUM knots must have identical finite XY shape')
    xy = (1-alpha)*tum_xy + alpha*safe_xy
    if not np.isfinite(xy).all() or len(xy) < 3:
        raise ValueError('invalid blended geometry')
    # Closest-point projection can repeat a TUM knot; omit zero-length segments.
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    xy, station = xy[keep], np.asarray(tum_station)[keep]
    if np.linalg.norm(xy[-1]-xy[0]) <= 1e-6:
        xy, station = xy[:-1], station[:-1]
    line = ReferenceLine.from_xy(xy, name=f'TUM_safe_blend_alpha{alpha:.3f}')
    station = np.unwrap(station, period=tum_length)
    if np.any(np.diff(station) < -1e-6):
        raise ValueError('TUM correspondence reverses direction')
    mapping = np.r_[station, station[0]+tum_length]
    if len(mapping) != len(line.path):
        raise ValueError('candidate geometry and target station lengths differ')
    return Candidate(float(alpha), line, mapping, {})


def screen_candidate(candidate, track, footprint, grid_m=1.):
    """Planned tangent-aligned footprint test only; final validity comes from DrivingEnv."""
    n = int(math.ceil(candidate.line.length/grid_m))
    stations = np.arange(n)*candidate.line.length/n
    inside = []
    for station in stations:
        xy = candidate.line.point(station)
        inside.append(bool(track.contains_car(float(xy[0]), float(xy[1]),
                                              candidate.line.heading(station), footprint)))
    candidate.planned = {'planned_only': True, 'sample_count': n,
                         'max_grid_m': grid_m, 'inside_fraction': float(np.mean(inside)),
                         'all_sampled_footprints_inside': bool(all(inside)),
                         'violating_samples': int(n-sum(inside)), 'line_length_m': candidate.line.length,
                         'line_digest': candidate.line.digest}
    return candidate.planned


def selected_candidates(candidates):
    """Smallest three sampled-feasible alphas plus the safe-line endpoint, if feasible."""
    feasible = sorted([c for c in candidates if c.planned['all_sampled_footprints_inside']], key=lambda c:c.alpha)
    selected = feasible[:3]
    safe = next((c for c in feasible if c.alpha == 1.), None)
    if safe is not None and all(c.alpha != 1. for c in selected):
        selected.append(safe)
    return selected


def trial_plan(candidates, factors, lookaheads, kp_values, ki_values):
    # Round-robin geometry at each control/speed setting: every candidate receives an
    # early test before the wall budget is consumed by refining a single trajectory.
    return [{'alpha': c.alpha, 'factor':float(f), 'lookahead_s':float(lh),
             'speed_kp':float(kp), 'speed_ki':float(ki)}
            for lh,kp,ki,f,c in itertools.product(lookaheads,kp_values,ki_values,factors,candidates)]


def run_trial(env, candidate, profile, params, seed):
    """Use the genuine evaluation reset and strict environment; no backend stepping here."""
    started = time.perf_counter()
    env.reset(seed=int(seed), options={'evaluation': True})
    controller = replace(env.cfg.demo, brake_cap=1., lookahead_s=params['lookahead_s'],
                         speed_kp=params['speed_kp'], speed_ki=params['speed_ki'])
    driver = DemoDriver(candidate.line, controller, pedal_deadzone=env.cfg.pedal_deadzone,
                        control_dt_s=env.decision_dt, car=env.cfg.car)
    driver.reset(env.settle_driver.integral)
    initial_integral = driver.integral
    hint, total = None, 0.
    while True:
        state = env.state
        hint = candidate.line.match((state['x'],state['y']),hint)
        preview = hint + max(state['vx'],0.)*controller.speed_preview_s
        s_tum, speed = candidate.target(preview,profile)
        target = params['factor']*speed
        action = driver.act(state,hint,target)
        env.command_context = {'target_speed_mps':target,'s_controller_m':hint,
                               's_preview_controller_m':preview,'s_preview_tum_m':s_tum,
                               'speed_scale_f':params['factor']}
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    times = [lap['lap_time_s'] for lap in info['laps']]
    recorder = check(env,SimpleNamespace(lap_times_s=times))
    if not recorder['lap_times_match_env']:
        raise RuntimeError('recorder and genuine environment lap clock differ')
    clean = info['termination']=='lap_complete' and info['truncation'] is None and len(times)==2
    return env.rows, {'params':dict(params),'clean':bool(clean), 'termination':info['termination'],
                     'truncation':info['truncation'],'lap_times_s':times,
                     'lap2_s':times[1] if len(times)==2 else None,'recorder_check':recorder,
                     'decisions':env.decision,'sim_time_s':info['time_s'],'progress_m':info['progress_m'],
                     'laps':info['laps'],'episode_return':total,'wall_s':time.perf_counter()-started,
                     'handover':env.handover,'controller_initial_integral_m':initial_integral,
                     'terminal_state':dict(env.state), 'terminal_sample':dict(env.rows[-1]) if env.rows else None,
                     'planned_geometry':candidate.planned,'controller_config':asdict(controller)}


def write_json(path,data):
    path=Path(path)
    temporary=path.with_name(path.name+'.next')
    comparison_state.write_json(temporary,common._clean(data))
    temporary.replace(path)


def export_trace(out,name,rows,summary,protocol):
    path=out/f'{name}.csv'
    write_trace(path,rows)
    common.write_meta(path,{'source':'tuned_TUM_geometric_PP_PI','protocol':'protocol.json',
                           'protocol_sha256':sha256_file(out/'protocol.json'),
                           'conditions':protocol['conditions'],'controller':summary['controller_config'],
                           'summary':summary,'lap_time_s':summary['lap2_s'],
                           'line':summary['planned_geometry'],'learned_weights_loaded':False})
    return path.name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/mac/D2/p1_TUM/seed0')
    parser.add_argument('--tum-csv',default='comparison/tum/tum_none.csv')
    parser.add_argument('--out',required=True,help='new output directory')
    parser.add_argument('--max-trials',type=int,default=320)
    parser.add_argument('--wall-seconds',type=float,default=900.)
    parser.add_argument('--alphas',nargs='+',type=float,default=[k/20 for k in range(21)])
    parser.add_argument('--factors',nargs='+',type=float,default=[1.,.95,.85,.70])
    parser.add_argument('--lookaheads',nargs='+',type=float,default=[.35,.28,.22,.16,.10])
    parser.add_argument('--kp',nargs='+',type=float,default=[.4,.2])
    parser.add_argument('--ki',nargs='+',type=float,default=[.2,.05])
    parser.add_argument('--geometry-only',action='store_true')
    args=parser.parse_args()
    if args.max_trials<1 or not math.isfinite(args.wall_seconds) or args.wall_seconds<=0:
        parser.error('trial and wall budgets must be positive finite values')
    for name,lo,hi in [('alphas',0.,1.),('factors',.5,1.),('lookaheads',.01,1.),('kp',.001,2.),('ki',0.,2.)]:
        values=getattr(args,name)
        if len(set(values))!=len(values) or not all(math.isfinite(v) and lo<=v<=hi for v in values):
            parser.error(f'{name} must be unique finite values in [{lo}, {hi}]')
    start=time.perf_counter()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    run=Path(args.run).resolve()
    cfg_bytes=(run/'config.json').read_bytes()
    cfg=train_config_from_dict(json.loads(cfg_bytes))
    env_cfg=evaluation_config(cfg.env)
    if (env_cfg.setup!='p1_TUM' or env_cfg.car!='sf70h_tum' or env_cfg.ers or
            env_cfg.max_brake_torque_Nm!=15000. or env_cfg.reference!='path' or
            env_cfg.action_repeat!=2 or env_cfg.settle_s!=2.):
        raise ValueError('requires saved p1_TUM/sf70h_tum, 15kNm, no ERS, path reset, 25Hz, 2s settle')
    (out/'config.json').write_bytes(cfg_bytes)
    source_provenance=json.loads((run/'provenance.json').read_text())
    write_json(out/'saved_run_provenance.json',source_provenance)
    track=Track(env_cfg.track,env_cfg.track_vmax_mps)
    code=run_provenance(track.name)
    differences=comparison_state.source_differences(source_provenance,code)
    if differences:
        raise RuntimeError('saved simulator sources differ: '+'; '.join(differences))
    tum_csv=Path(args.tum_csv).resolve();tum_meta=json.loads(tum_csv.with_suffix('.meta.json').read_text())
    if (tum_meta['variant']!='none' or tum_meta['sha256']!=sha256_file(tum_csv) or
            tum_meta['ers_used_J']!=0 or tum_meta['drs_share_of_points']!=0):
        raise ValueError('requires verified standard TUM no-ERS/no-DRS CSV')
    qss=qss_profile.for_env(track,env_cfg) # saved reset/settle only, not lapping targets
    env=RecordingEnv(env_cfg,track=track,qss=qss)
    try:
        raceline=env.geo.raceline
        grid=track_grid(track,raceline,1.)
        safe_xy=np.column_stack([np.interp(grid.s,track.path[:,0],track.path[:,i]) for i in (1,2)])
        candidates=[]
        for alpha in sorted(args.alphas):
            candidate=blend_candidate(alpha,safe_xy,grid.ref,grid.s_ref,raceline.length)
            screen_candidate(candidate,track,env.footprint)
            candidates.append(candidate)
            print(f'geometry alpha={alpha:.2f}: planned inside {candidate.planned["inside_fraction"]:.6f}',flush=True)
        write_json(out/'geometry_screen.json',{'grid_m':1.,'footprint_m':env.footprint,
                   'planned_only':True,'candidates':[{'alpha':c.alpha,**c.planned} for c in candidates]})
        selected=selected_candidates(candidates)
        plan=trial_plan(selected,args.factors,args.lookaheads,args.kp,args.ki)
        for candidate in selected:
            np.savetxt(out/f'line_alpha{candidate.alpha:.2f}.csv',
                       np.column_stack([candidate.line.path,candidate.tum_station]),delimiter=',',
                       header='s_candidate_m,x_m,y_m,s_tum_abeam_m',comments='')
        profile=TumProfile(tum_csv,raceline)
        conditions={'car':env_cfg.car,'setup_name':env_cfg.setup,'setup':env.setup.to_dict(),
                    'max_brake_torque_Nm':env_cfg.max_brake_torque_Nm,'ers':False,'drs':False,'recuperation':False,
                    'decision_dt_s':env.decision_dt,'tick_dt_s':env.dt_tick,'physics_dt_s':.002,
                    'footprint_m':env.footprint,'seed':cfg.eval.seed,'settle_s':env_cfg.settle_s,
                    'run_in_m':env_cfg.eval_run_in_m,'strict_footprint_termination':True}
        protocol={'source':'nonlearned_TUM_PP_PI_search','conditions':conditions,'saved_run':str(run),
                  'config_sha256':sha256_file(out/'config.json'),'evaluation_env_config':env_cfg.to_dict(),
                  'tum_csv':str(tum_csv),'tum_sha256':tum_meta['sha256'],'tum_lap_time_s':tum_meta['lap_time_s'],
                  'TUM_model_caveat':'Different QSS vehicle model; its lap time is not a guaranteed OCD target',
                  'geometry':'1m safe-path stations; sequential abeam TUM XY; linear XY blend; true candidate arc length',
                  'selected_alphas':[c.alpha for c in selected],'predeclared_plan':plan,
                  'selection_rule':'minimum clean genuine lap2_s across all tried settings; identical-seed repeat',
                  'controller':'geometric pure pursuit + speed PI, brake_cap=1, saved traction control; no ABS/yaw feedback',
                  'learned_weights_loaded':False,'RL_trajectory_used':False,
                  'lapping_speed':'factor times TUM speed, spatial correspondence interpolated on candidate arc length',
                  'reset':'unchanged saved DrivingEnv path reset/settle; QSS used only during reset/settle',
                  'wall_seconds':args.wall_seconds,'max_trials':args.max_trials,
                  'code':{**code,'ocd_native_sha256':sha256_file(ocd_native.__file__),
                          'script_sha256':sha256_file(__file__)},'command':sys.argv}
        write_json(out/'protocol.json',protocol)
        attempts=[];best=None;stop='geometry_only' if args.geometry_only else 'plan_complete'
        for index,params in enumerate(plan if not args.geometry_only else []):
            if index>=args.max_trials:
                stop='max_trials';break
            if time.perf_counter()-start>=args.wall_seconds:
                stop='wall_budget';break
            candidate=next(c for c in selected if c.alpha==params['alpha'])
            rows,summary=run_trial(env,candidate,profile,params,cfg.eval.seed)
            summary['trial_index']=index
            attempts.append(summary)
            new_best=choose_best(attempts)
            if new_best is summary and (best is None or summary['lap2_s']<best['lap2_s']):
                summary['trace']=export_trace(out,f'best_trial{index:04d}',rows,summary,protocol)
                best=summary
            write_json(out/f'trial_{index:04d}.json',summary)
            write_json(out/'summary.json',{'complete':False,'attempts':attempts,'best':best,
                                         'elapsed_wall_s':time.perf_counter()-start})
            print(f'trial{index:04d} {params} clean={summary["clean"]} '
                  f'cause={summary["termination"] or summary["truncation"]} lap2={summary["lap2_s"]}',flush=True)
        repeat=None
        if best is not None:
            candidate=next(c for c in selected if c.alpha==best['params']['alpha'])
            rows,repeat=run_trial(env,candidate,profile,best['params'],cfg.eval.seed)
            repeat['trace']=export_trace(out,'best_repeat',rows,repeat,protocol)
            repeat['lap2_difference_s']=(repeat['lap2_s']-best['lap2_s']) if repeat['clean'] else None
            repeat['verified']=bool(repeat['clean'] and abs(repeat['lap2_difference_s'])<=1e-9)
            write_json(out/'repeat.json',repeat)
            if not repeat['verified']:
                raise RuntimeError('winning controller did not reproduce its clean lap; diagnostic trace retained')
        write_json(out/'summary.json',{'complete':True,'stop_reason':stop,'attempts':attempts,'best':best,
                   'repeat':repeat,'trial_count':len(attempts),'plan_count':len(plan),
                   'elapsed_wall_s':time.perf_counter()-start,'no_torch_loaded':'torch' not in sys.modules})
        print(json.dumps({'stop':stop,'trials':len(attempts),'best_lap2_s':best['lap2_s'] if best else None,
                          'repeat_verified':repeat['verified'] if repeat else None}),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
