from pathlib import Path
import csv,datetime,hashlib,json,math,re,shutil,subprocess,sys,time
root=Path('/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent');sys.path.insert(0,str(root))
from ocd_lap.backend import OCDBackend,StartPose,Inputs,DT
from ocd_lap.translator import P
from rlsetup.driving import setups
import ocd_native
out=root/'comparison/tum_ocd_replay_20261031_115434Z/physics_validation';out.mkdir(exist_ok=False)
a=P+'aerodynamics.';setup=setups.get('p1_TUM')
probe=OCDBackend(powertrain='engine',car='sf70h_tum',ers=False,max_brake_torque_Nm=15000)
poly=[key for key in probe.base_params if key.startswith(a) and re.fullmatch(r'[ca][0-6](?:_[FR])?',key[len(a):])]
assert len(poly)==20,(len(poly),poly)
overrides={key:0. for key in poly};overrides.update({a+'A_m2':1.,a+'c_d':1.56,a+'c0':2.2,a+'a0':2.68})
records=[];params={};started=time.perf_counter()
for variant,override in [('baseline',{}),('static_TUM_aero',overrides)]:
 for requested_v in (15.,30.,60.,90.):
  backend=OCDBackend(powertrain='engine',car='sf70h_tum',ers=False,max_brake_torque_Nm=15000,overrides=override)
  initial=backend.reset(setup,StartPose(0.,0.,0.,requested_v));params[variant]=backend.params
  for step in range(1,6):
   state=backend.step(Inputs(0.,0.,0.));tel=backend.get_telemetry();v=state['vx'];rho=backend.params[a+'air_density_kgpm3']
   expected_fx=-.5*rho*v*v*1.56;expected_fz=-.5*rho*v*v*(2.2+2.68)
   cfront=backend.params[a+'c0']+backend.params[a+'c1_F']*tel['ride_height_front_m']
   crear=backend.params[a+'a0']+backend.params[a+'a1_R']*tel['ride_height_rear_m']
   expected_model_fz=-.5*rho*v*v*(max(0.,min(5.,cfront))+max(0.,min(5.,crear)))
   loads={w:tel['Fz_'+w+'_N'] for w in ('FL','FR','RL','RR')}
   records.append({'variant':variant,'reset_speed_mps':requested_v,'step':step,'t_s':step*DT,'actual_vx_mps':v,'actual_vy_mps':state['vy'],'aero_Fx_N':tel['aero_Fx_N'],'aero_Fz_N':tel['aero_Fz_N'],'expected_TUM_Fx_N':expected_fx,'expected_TUM_Fz_N':expected_fz,'Fx_error_N':tel['aero_Fx_N']-expected_fx,'Fz_vs_TUM_error_N':tel['aero_Fz_N']-expected_fz,'Fz_vs_current_model_error_N':tel['aero_Fz_N']-expected_model_fz,'current_CzA_front':cfront,'current_CzA_rear':crear,'ride_height_front_m':tel['ride_height_front_m'],'ride_height_rear_m':tel['ride_height_rear_m'],'pitch_rad':tel['pitch_rad'],'heave_m':tel['heave_m'],'wheel_loads_N':loads,'wheel_load_sum_N':sum(loads.values()),'mg_plus_downforce_N':733.*9.81-tel['aero_Fz_N'],'wheel_load_sum_minus_mg_downforce_N':sum(loads.values())-(733.*9.81-tel['aero_Fz_N'])})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base,static=params['baseline'],params['static_TUM_aero'];diff={k:{'baseline':base[k],'static_TUM_aero':static[k]} for k in base if base[k]!=static[k]}
checks={'static_aero_max_Fx_error_N':max(abs(r['Fx_error_N']) for r in records if r['variant']=='static_TUM_aero'),'static_aero_max_Fz_error_N':max(abs(r['Fz_vs_TUM_error_N']) for r in records if r['variant']=='static_TUM_aero'),'both_variants_max_model_Fz_error_N':max(abs(r['Fz_vs_current_model_error_N']) for r in records),'all_finite':all(math.isfinite(v) for r in records for v in r.values() if isinstance(v,(int,float)))}
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':'Reset p1_TUM separately per speed/variant;5zero-input2ms steps; native aero/loads from public get_telemetry; compare same returned vx not reset request.','units':'SI; aero Fz negative downward, wheel Fz positive load','start':'x=y=yaw=0,no settle,0steer0throttle0brake','wall_s':time.perf_counter()-started,'physics_steps':len(records),'control_input':{'steering_rad':0.,'throttle':0.,'brake':0.},'limitations':['This validates force calculation and constructor override application only; it is not a lap or dynamic equivalence proof.','Suspension starts un-settled at reset: wheel-load sums need not equal mg+downforce during the first10ms, due to vertical acceleration.','Native integrator returned output and intermediate logger may differ at integration-stage precision; formula residuals are recorded rather than hidden.'],'overrides':overrides,'parameter_diff':diff,'checks':checks,'records':records,'source':{'scratch_path':__file__,'scratch_sha256':sha(__file__),'native_sha256':sha(ocd_native.__file__),'OCD_HEAD':subprocess.check_output(['git','-C','/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap','rev-parse','HEAD'],text=True).strip(),'OCD_status_after':subprocess.check_output(['git','-C','/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap','status','--short'],text=True)},'no_torch_loaded':'torch' not in sys.modules,'command':sys.argv}
(out/'force_checks.json').write_text(json.dumps(report,indent=2)+'\n');(out/'baseline_parameters.json').write_text(json.dumps(base,indent=2,sort_keys=True)+'\n');(out/'static_TUM_aero_parameters.json').write_text(json.dumps(static,indent=2,sort_keys=True)+'\n');shutil.copyfile(__file__,out/Path(__file__).name)
flat=[{**{k:v for k,v in r.items() if k!='wheel_loads_N'},**{'Fz_'+w+'_N':v for w,v in r['wheel_loads_N'].items()}} for r in records]
with (out/'force_checks.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps({'output':str(out),'checks':checks,'wall_s':report['wall_s'],'changed_parameters':len(diff),'no_torch':report['no_torch_loaded']}))
for r in records:
 if r['step']==5:print(json.dumps(r))
