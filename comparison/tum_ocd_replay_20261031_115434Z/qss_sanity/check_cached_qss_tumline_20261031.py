from pathlib import Path
import csv,datetime,hashlib,json,shutil,sys,time
import numpy as np
root=Path('/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent');sys.path.insert(0,str(root))
from rlsetup.driving import qss_profile,setups
from ocd_lap import qss_lap,cars
from ocd_lap.track import Track
import ocd_native
out=root/'comparison/tum_ocd_replay_20261031_115434Z/qss_sanity';out.mkdir(exist_ok=False)
cache=root/'data/qss_profile_Shanghai_p1_TUM_sf70h_tum_brake15000.npz';track=Track('Shanghai',12.);setup=setups.get('p1_TUM');cached=qss_profile.load(cache)
key=qss_profile.cache_key(track,setup,car='sf70h_tum',ers=False,brake_torque_Nm=15000.)
assert cached.meta['key']==key,'Cache key mismatch: refuse unapproved GGV remeasurement'
ggv_data=(np.array(cached.meta['ggv']),np.array(cached.meta['traction']),[])
started=time.perf_counter();result=qss_lap.lap_time(setup,track,car='sf70h_tum',ers=False,drs=False,path=track.reference,ggv_data=ggv_data);wall=time.perf_counter()-started
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
path_hash=hashlib.sha256(np.ascontiguousarray(track.reference).tobytes()).hexdigest()
# A pure lateral capacity check is independent of longitudinal force bookkeeping;
# combined-use feasibility requires explicit drag/rolling conventions and is not asserted here.
ref=root/'comparison/tum_ocd_replay_20261031_115434Z/reference/tum_none.csv';rows=[r for r in csv.DictReader(ref.open()) if r['closing_point']=='0'];v=np.array([float(r['speed_mps']) for r in rows]);ay=np.abs([float(r['ay_mps2']) for r in rows]);limit=np.interp(v,ggv_data[0][:,0],ggv_data[0][:,2]);ratio=ay/limit
scalar={k:v for k,v in result.items() if k not in ('v_profile','s_m')}
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'type':'OCD cached-limit quasi-steady-state forward/backward calculation, NOT an actual driven lap','current_baseline_only':True,'static_aero_GGV_not_measured_or_inferred':True,'GGV_remeasurement':False,'physics_steps_executed':0,'cache_path':str(cache),'cache_sha256':sha(cache),'cache_key_exact_match':True,'validated_cache_key':key,'cached_safe_path_lap_s':cached.lap_time_s,'original_TUM_geometry_result':scalar,'wall_s':wall,'path':{'type':'track.reference original TUM/racetrack-database geometry at existing track sampling','length_m':float(track.reference[-1,0]),'shape':list(track.reference.shape),'numpy_bytes_sha256':path_hash,'TUM_solver_path_length_m':json.loads(ref.with_suffix('.meta.json').read_text())['raceline_length_m'],'note':'Existing OCD helper computes numeric curvature on this line, not the separately resampled5m TUM solver grid.'},'model_limits':{'car':'sf70h_tum','setup':setup.to_dict(),'ERS':False,'DRS':False,'GGV_brake_ramp_cap_Nm':15000.,'GGV_slip_cutoff':.12,'friction_exponent':cars.get('sf70h_tum').qss_dyn_model_exp,'mass_kg':733.,'drag_coefficient_kgpm':result['drag_coeff'],'QSS_vcap_mps':100.,'machine_limit':'best of8gears,min(engine,measured traction); engine grid5m/s; ignores rotating-mass factor (optimistic)','caveats':['No yaw stability or dynamic path-following/steering feasibility.','GGV comes from short open-loop manoeuvres with settling; lateral peak/longitudinal ramps are separate tests.','Tyre longitudinal ramp limits remove aerodynamic drag, retain rolling loss implicitly.','TUM net ax, tyre demand including drag/rolling, and cached measured ax cannot be equated without converting those conventions.','No combined-force feasibility claim or static-aero reuse made.']},'TUM_required_lateral_diagnostic':{'definition':'abs(TUM ay) / interpolated current cached OCD peak ay(speed); no combined longitudinal claim','points':len(rows),'points_over_one':int(np.sum(ratio>1)),'fraction_over_one':float(np.mean(ratio>1)),'max_ratio':float(np.max(ratio))},'source_hashes':{'scratch':sha(__file__),'qss_lap':sha(qss_lap.__file__),'qss_profile':sha(qss_profile.__file__),'cars':sha(cars.__file__),'native':sha(ocd_native.__file__),'TUM_csv':sha(ref)},'no_torch_loaded':'torch' not in sys.modules,'command':sys.argv}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/Path(__file__).name)
with (out/'qss_original_TUM_line.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['s_m','speed_mps']);w.writerows(zip(result['s_m'],result['v_profile']))
print(json.dumps({'original_TUM_QSS_s':result['lap_time_s'],'cached_safe_s':cached.lap_time_s,'key_exact_match':True,'wall_s':wall,'lateral_diagnostic':report['TUM_required_lateral_diagnostic'],'no_torch':report['no_torch_loaded']}))
