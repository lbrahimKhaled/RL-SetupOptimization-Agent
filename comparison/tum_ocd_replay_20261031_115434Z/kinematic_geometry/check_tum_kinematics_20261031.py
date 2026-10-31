from pathlib import Path
import csv,hashlib,importlib.util,json,math,shutil,sys,time
import numpy as np
root=Path('/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent');tum=Path('/Users/ibrahimkhaled/Documents/tum/laptime-simulation');sys.path.insert(0,str(tum))
for alias,typ in [('int',int),('float',float),('bool',bool)]:
 if alias not in np.__dict__:setattr(np,alias,typ)
import laptimesim,trajectory_planning_helpers as tph
spec=importlib.util.spec_from_file_location('tum_runner','/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap/scripts/run_tum_laptimesim.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
base=root/'comparison/tum_ocd_replay_20261031_115434Z';out=base/'kinematic_geometry';out.mkdir(exist_ok=False)
ref=base/'reference/tum_none.csv';meta=json.loads(ref.with_suffix('.meta.json').read_text());rows=list(csv.DictReader(ref.open()));data={k:np.array([float(r[k]) for r in rows]) for k in ['t_s','s_m','x_m','y_m','speed_mps','kappa_radpm']};end=meta['lap_time_s'];data['t_s'][-1]=end
track=laptimesim.src.track.Track(pars_track=dict(runner.TRACK,curv_filt_width=None,use_drs1=False,use_drs2=False),parfilepath=str(tum/'laptimesim/input/tracks/track_pars.ini'),trackfilepath=str(tum/'laptimesim/input/tracks/racelines/Shanghai.csv'),vel_lim_glob=np.inf)
xy=np.vstack((track.raceline,track.raceline[0]));round7=lambda arr:np.array([float(f'{x:.7g}') for x in np.ravel(arr)]).reshape(arr.shape)
assert track.no_points==1069 and len(rows)==1070
assert np.array_equal(round7(xy),np.column_stack((data['x_m'],data['y_m'])))
assert np.array_equal(round7(track.dists_cl),data['s_m'])
# Recreate TUM's public spline preparation to obtain its analytical XY tangent;
# no track/model source is edited and no vehicle solver/physics is executed.
raw=np.loadtxt(tum/'laptimesim/input/tracks/racelines/Shanghai.csv',comments='#',delimiter=',');closed=np.vstack((raw,raw[0]));cx,cy=tph.calc_splines.calc_splines(path=closed,use_dist_scaling=True)[:2];lens=tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=cx,coeffs_y=cy,quickndirty=False);sxy,indices,ts,dist=tph.interp_splines.interp_splines(spline_lengths=lens,coeffs_x=cx,coeffs_y=cy,incl_last_point=False,stepsize_approx=5.);assert np.array_equal(sxy,track.raceline)
psi,kappa=tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=cx,coeffs_y=cy,ind_spls=indices,t_spls=ts);assert np.array_equal(kappa,track.kappa)
heading=np.unwrap(np.r_[psi+math.pi/2,psi[0]+math.pi/2]);tangent0=float(heading[0]);k_unfiltered=np.r_[track.kappa,track.kappa[0]];k_filtered=data['kappa_radpm'];L=3.6;lr=1.632;dt=.002
nfull=int(math.floor(end/dt));timegrid=np.r_[np.arange(nfull+1)*dt,end];assert np.all(np.diff(timegrid)>0)
mid=.5*(timegrid[:-1]+timegrid[1:]);h=np.diff(timegrid);v0=np.interp(timegrid[:-1],data['t_s'],data['speed_mps']);vmid=np.interp(mid,data['t_s'],data['speed_mps']);xref=np.interp(timegrid,data['t_s'],data['x_m']);yref=np.interp(timegrid,data['t_s'],data['y_m']);href=np.interp(timegrid,data['t_s'],heading)
results=[];started=time.perf_counter()
for stream,k_source in [('filtered10m',k_filtered),('unfiltered_spline',k_unfiltered)]:
 for formula in ('CoG_corrected','simple_atan_Lkappa'):
  def controls(query):
   k=np.interp(query,data['t_s'],k_source)
   if formula=='CoG_corrected':
    if np.any(np.abs(lr*k)>=1):raise ValueError('CG steering inversion outside domain')
    delta=np.arctan(L*k/np.sqrt(1-(lr*k)**2))
   else:delta=np.arctan(L*k)
   beta=np.arctan(lr/L*np.tan(delta));return delta,beta
  d0,b0=controls(timegrid[:-1]);dm,bm=controls(mid);de,be=controls(timegrid);r0=v0/L*np.cos(b0)*np.tan(d0);rm=vmid/L*np.cos(bm)*np.tan(dm)
  x=float(data['x_m'][0]);y=float(data['y_m'][0]);yaw=tangent0-float(be[0]);pred=np.empty((len(timegrid),3));pred[0]=x,y,yaw
  for i,step in enumerate(h):
   yaw_mid=yaw+.5*step*r0[i];x+=step*vmid[i]*math.cos(yaw_mid+bm[i]);y+=step*vmid[i]*math.sin(yaw_mid+bm[i]);yaw+=step*rm[i];pred[i+1]=x,y,yaw
  error=np.hypot(pred[:,0]-xref,pred[:,1]-yref);velheading=(pred[:,2]+be-href+math.pi)%(2*math.pi)-math.pi
  cuts={}
  for label,stop in [('first3s',3.),('first15s',15.),('full',end)]:
   select=timegrid<=stop+1e-12;e=error[select];cuts[label]={'samples':int(np.sum(select)),'end_s':float(timegrid[select][-1]),'position_RMSE_m':float(np.sqrt(np.mean(e*e))),'position_max_m':float(np.max(e)),'position_terminal_m':float(e[-1]),'velocity_heading_RMSE_rad':float(np.sqrt(np.mean(velheading[select]**2))),'velocity_heading_terminal_rad':float(velheading[select][-1])}
  name=stream+'_'+formula;result={'case':name,'curvature':stream,'steering_formula':formula,'initial_velocity_tangent_rad':tangent0,'initial_body_yaw_rad':tangent0-float(be[0]),'initial_beta_rad':float(be[0]),'max_abs_steer_rad':float(np.max(np.abs(de))),'metrics':cuts,'reference_speed_forced':True,'NOT_an_OCD_lap':True};results.append(result)
  with (out/(name+'.csv')).open('x',newline='') as f:
   w=csv.writer(f);w.writerow(['t_s','x_pred_m','y_pred_m','body_yaw_rad','beta_rad','steer_rad','x_reference_m','y_reference_m','position_error_m','velocity_heading_error_rad']);indices_out=np.r_[np.arange(0,len(timegrid)-1,20),len(timegrid)-1]
   w.writerows((timegrid[i],*pred[i],be[i],de[i],xref[i],yref[i],error[i],velheading[i]) for i in indices_out)
  print(json.dumps(result),flush=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'label':'Analytic kinematic reference/reconstruction consistency only; forced reference speed, NOT OCD driving or achieved lap','physics_steps_executed':0,'no_vehicle_model_loaded':True,'no_training':True,'reference_horizon_s':end,'integration':{'method':'explicit midpoint, nominal2ms, exact partial final step','full_2ms_steps':nfull,'last_partial_step_s':float(h[-1]),'end_s':float(timegrid[-1]),'time_overshoot_s':0.,'total_steps':len(h)},'node_identity':{'points_unclosed':track.no_points,'XY_equal_export_after7significant_digit_rounding':True,'s_equal_export_after7significant_digit_rounding':True,'XY_max_abs_rounding_difference_m':float(np.max(np.abs(xy-np.column_stack((data['x_m'],data['y_m']))))),'s_max_abs_rounding_difference_m':float(np.max(np.abs(track.dists_cl-data['s_m']))),'path_length_exact_m':float(track.dists_cl[-1]),'unfiltered_track_vs_export_same_geometry':True},'curvature':{'filter_width_m':10.,'unfiltered_width':None,'filtered_minus_unfiltered_RMS_radpm':float(np.sqrt(np.mean((k_filtered[:-1]-k_unfiltered[:-1])**2))),'filtered_minus_unfiltered_max_abs_radpm':float(np.max(np.abs(k_filtered[:-1]-k_unfiltered[:-1]))),'filter_window_nodes':3},'equations':{'CoG_corrected':'delta=atan(L*kappa/sqrt(1-(lr*kappa)^2))','simple':'delta=atan(L*kappa)','beta':'atan(lr/L*tan(delta))','yawdot':'v/L*cos(beta)*tan(delta)','xdot':'v*cos(yaw+beta)','ydot':'v*sin(yaw+beta)','initial_body_yaw':'analytical spline XY tangent minus beta0'},'interpretation_limits':['Requested CG inversion is a steady-curvature correction: when beta varies, velocity-heading derivative=yawdot+betadot. All output drift is not uniquely attributable to filtering.','Reference speed is deliberately forced only in this analytic sanity check, not in OCD or training.','Reference XY and speed are linearly interpolated in exported time; nodecurvature interpolation,7digit rounding and5m sampling can contribute reconstruction error.','TUM curvature filtering does not change its XY nodes, so geometry/filteredcurvature need not integrate to identical trajectories.'],'results':results,'wall_s':time.perf_counter()-started,'hashes':{'scratch':sha(__file__),'reference_CSV':sha(ref),'TUM_track_source':sha(tum/'laptimesim/src/track.py'),'raw_raceline_CSV':sha(tum/'laptimesim/input/tracks/racelines/Shanghai.csv')},'command':sys.argv,'no_torch_loaded':'torch' not in sys.modules}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/Path(__file__).name)
print(json.dumps({'output':str(out),'wall_s':report['wall_s'],'node_identity':report['node_identity'],'no_torch':report['no_torch_loaded']}),flush=True)
