from pathlib import Path
import ast,configparser,datetime,hashlib,json,math,shutil,subprocess,sys
root=Path('/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent');sys.path.insert(0,str(root))
ocd=Path('/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap');tum=Path('/Users/ibrahimkhaled/Documents/tum/laptime-simulation')
out=root/'comparison/tum_ocd_replay_20261031_115434Z';assert (out/'reference/tum_none.meta.json').exists()
from ocd_lap.backend import OCDBackend,StartPose,CONFIG
from ocd_lap.cars import TUM_F1
from ocd_lap.translator import P
from rlsetup.driving import setups
import ocd_native
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ini=tum/'laptimesim/input/vehicles/F1_Shanghai.ini';parser=configparser.ConfigParser();parser.read(ini);v=json.loads(parser['VEH_PARS']['veh_pars'])
s=setups.get('p1_TUM');b=OCDBackend(powertrain='engine',car='sf70h_tum',ers=False,max_brake_torque_Nm=15000);b.reset(s,StartPose(0.,0.,0.,12.));p=b.params;g=v['general'];a=P+'aerodynamics.'
comparisons=[]
def add(name,actual,target,units,definition):comparisons.append(dict(name=name,OCD=actual,TUM=target,delta=actual-target,units=units,definition=definition))
for name,key,target,unit in [('mass',P+'mass_vehicle_kg',g['m'],'kg'),('wheelbase',P+'wheelbase_m',g['lf']+g['lr'],'m'),('CG to front',P+'cog.distance_from_front_axle_m',g['lf'],'m'),('CG height',P+'cog.height_m',g['h_cog'],'m'),('front track',P+'track_width_m.front',g['sf'],'m'),('rear track',P+'track_width_m.rear',g['sr'],'m'),('rho_air',a+'air_density_kgpm3',g['rho_air'],'kg/m^3'),('rolling resistance',P+'tire.rolling_resistance_coefficient',g['f_roll'],'1')]:add(name,p[key],target,unit,key)
add('CG to rear',p[P+'wheelbase_m']-p[P+'cog.distance_from_front_axle_m'],g['lr'],'m','wheelbase minus CG-to-front distance')
add('drag CwA',p[a+'A_m2']*p[a+'c_d'],g['c_w_a'],'m^2','A*c_d, DRS off')
add('reference front CzA',p[a+'c0']+p[a+'c1_F']*.035,g['c_z_a_f'],'m^2','front polynomial at reference physical rideheight .035m, not dynamic lift')
add('reference rear CzA',p[a+'a0']+p[a+'a1_R']*.085,g['c_z_a_r'],'m^2','rear polynomial at reference physical rideheight .085m, not dynamic lift')
for axle,ref in [('front','f'),('rear','r')]:
 add(axle+' loaded radius (60km/h reference)',p[P+'tire.rolling_radius_m.'+axle],v['tires'][ref]['circ_ref']/(2*math.pi),'m','TUM radius expands with speed; OCD current velocity-scaling factors stay1')
 k=P+'z.tire_model.'+axle+'_left.';nom=p[k+'FNOMIN']*p[k+'LFZO']
 add(axle+' tyre nominal Fz',nom,v['tires'][ref]['fz_0'],'N','MF FNOMIN*LFZO')
 for axis,code in [('x','X'),('y','Y')]:
  lmu=p[k+'LMU'+code]
  add(axle+' mu'+axis+' nominal',p[k+'PD'+code+'1']*lmu,v['tires'][ref]['mu'+axis],'1','zero-camber pure-slip peak coefficient PD1*LMU; not full combined-slip force')
  add(axle+' dmu'+axis+'/dFz',p[k+'PD'+code+'2']*lmu/nom,v['tires'][ref]['dmu'+axis+'_dfz'],'1/N','PD2*LMU/(FNOMIN*LFZO)')
def git(path):return {'HEAD':subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),'status':subprocess.check_output(['git','-C',str(path),'status','--short'],text=True)}
source_files=[Path(__file__),Path('/private/tmp/export_tum_none_replay_20261031.py'),root/'scripts/comparison/export_tum_lap.py',root/'scripts/comparison/common.py',ocd/'ocd_lap/backend.py',ocd/'ocd_lap/cars.py',ocd/'ocd_lap/translator.py',ocd/'ocd_lap/aero_map.py',ocd/'ocd_lap/powertrain.py',tum/'laptimesim/src/car.py',tum/'laptimesim/src/car_hybrid.py',tum/'laptimesim/src/lap.py',ini,Path(CONFIG),Path(ocd_native.__file__),root/'runs/mac/D2/p1_TUM/seed0/config.json']
signatures={}
for fn in ['car.py','car_hybrid.py']:
 path=tum/'laptimesim/src'/fn; tree=ast.parse(path.read_text());records=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.FunctionDef) and not node.name.startswith('__'):
   records.append({'name':node.name,'signature':ast.unparse(node).split('\n',1)[0],'line':node.lineno})
 signatures[fn]=records
meta=json.loads((out/'reference/tum_none.meta.json').read_text())
structural=[
 'TUM FB+ solves speed on prescribed raceline, algebraic load transfer with per-wheel minimum30N, axle-wise exponent2 friction ellipse; it does not integrate x/y/yaw from steer/throttle/brake.',
 'OCD is double-track MF5.2 with combined-slip/shape/stiffness, wheel inertia and transient suspension; matching peak mu(Fz) does not match these omitted TUM dynamics.',
 'OCD reference-height aero is almost identical (front share rounding), but placeholder rideheight sensitivity is2 per metre and dynamic suspension changes lift; TUM fixed CzA has no rideheight dynamics.',
 'OCD inertias are scaled AV21 placeholders, Ix97.6316337149,Iy579.814453125,Iz1159.62890625kgm2; TUM QSS has no matching pitch/roll/yaw inertias.',
 'OCD steering actuator PT1 .06s, rate cap1.5rad/s, angle cap.6rad; TUM QSS does not solve steering actuator states.',
 'TUM expands tyre circumference with speed, current OCD radius scaling fixed1. Gear selection in TUM uses vehicle speed; OCD uses wheel rpm plus1000rpm hysteresis and.1s shift dwell.',
 'TUM gear torsional factors e_i act algebraically in required force/torque; OCD reflects approximate additional rotating inertia through drivetrain.',
 'TUM all-wheel braking uses tyre force potential and front weak-side FA rule; it has no original fixed15kNm hardware cap/bias. Current p1 OCD brake hardware15kNm, front bias.49.',
 'TUM reference has weather friction1; current OCD car coefficient mapping preserves nominal tyre peaks. Exact force matching additionally depends on dynamic load/camber/slip.',
 'Original TUM CSV is QSS geometry/speed/force demand, not an original three-pedal/steering command trace. Any derived open-loop steer/brake command must be labeled reconstructed.'
]
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only_source_audit':True,'physics_steps_executed':0,'reset_only':True,'training_state':'No learner/keeper/search jobs found in initial read-only process inventory; no signals sent.', 'reference':{'path':str(out/'reference/tum_none.csv'),'sha256':sha(out/'reference/tum_none.csv'),'lap_s':meta['lap_time_s'],'settings':meta['settings'],'input_hashes':meta['inputs_sha256']},'current_p1_setup':s.to_dict(),'controls':{'ERS':False,'DRS':False,'brake_cap_Nm':b.max_brake_torque_Nm,'brake_bias_front':b.brake_bias_front,'steer_PT1_s':p['steering_actuator.T_PT1'],'steer_rate_radps':p['steering_actuator.angle_rate_max_radps'],'steer_cap_rad':p['steering_actuator.angle_max_rad']},'comparison':comparisons,'structural_mismatches':structural,'powertrain':{'OCD_ratios':b.powertrain.ratios.tolist(),'OCD_data':b.powertrain.d,'TUM_engine':v['engine'],'TUM_gearbox':v['gearbox'],'TUM_radius_m_at_v':{str(speed):v['tires']['r']['circ_ref']*(1+(speed*3.6-60)*(.045/200))/(2*math.pi) for speed in (0,12,30,60,80)},'OCD_fixed_radius_m':p[P+'tire.rolling_radius_m.rear']},'TUM_public_signatures':signatures,'source_sha256':{str(f):sha(f) for f in source_files},'git':{'RL':git(root),'OCD':git(ocd),'TUM':git(tum)},'command':sys.argv}
(out/'parameter_audit.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'p1_backend_parameters.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
(out/'TUM_vehicle_parameters.json').write_text(json.dumps(v,indent=2)+'\n')
for src in [Path(__file__),Path('/private/tmp/export_tum_none_replay_20261031.py')]:shutil.copyfile(src,out/src.name)
print(json.dumps({'report':str(out/'parameter_audit.json'),'reference_lap_s':meta['lap_time_s'],'comparisons':len(comparisons),'no_torch_loaded':'torch' not in sys.modules}))
