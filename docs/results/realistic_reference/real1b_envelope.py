"""Test 1b: the car's real straight-line acceleration envelope a(v) and where it differs from QSS.
(1) slip-limited throttle from 12 m/s (pedal cut by a fast slip controller, 500 Hz), several slip targets;
(2) force breakdown at full pedal: drive force from the gearbox, aero drag, the rest."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ocd_lap.backend import OCDBackend, StartPose, Inputs
from ocd_lap.translator import P
from ocd_lap import qss_lap
from rlsetup.driving.setups import get as get_setup

OUT = sys.argv[1]
setup = get_setup('sha_f1_best')

def make(v0):
    b = OCDBackend(powertrain='engine', car='sf70h_f1', ers=False, max_brake_torque_Nm=15000.)
    b.reset(setup, StartPose(0., 0., 0., v0))
    return b

def tc_run(v0, slip_target, T=30., hz=500):
    b = make(v0); m = b.params[P + 'mass_vehicle_kg']; r = b.params[P + 'tire.rolling_radius_m.rear']
    pedal = 1.; out = []; v = v0; x = 0.
    n = int(T * hz)
    for k in range(n):
        st = b.step(Inputs(0., pedal, 0.), 1)
        tel = b.get_telemetry()
        slip = max(tel['kappa_RL'], tel['kappa_RR'])
        # proportional cut above the target, recover quickly below
        pedal = float(np.clip(pedal + (0.02 if slip < slip_target else -0.2 * (slip - slip_target) / slip_target - 0.02), 0., 1.))
        x += st['vx'] / hz
        if k % 10 == 9:
            cap = b.drive_torque_capacity()
            out.append((k / hz, st['vx'], x, slip, pedal, b.gear, cap * pedal / r, -tel['aero_Fx_N']))
        if st['vx'] > 85: break
    return np.array(out), m

res = {}
for tgt in (0.06, 0.08, 0.10, 0.12):
    o, m = tc_run(12., tgt)
    res[tgt] = o
    t, v = o[:, 0], o[:, 1]
    print(f'slip target {tgt:.2f}: 12->40 m/s {t[np.argmax(v>=40)]:.2f} s, 12->60 {t[np.argmax(v>=60)]:.2f} s, 12->75 {t[np.argmax(v>=75)]:.2f} s,'
          f' v at 30 s {v[-1]:.2f}, dist to 75: {o[np.argmax(v>=75),2]:.0f} m')
best = min(res, key=lambda k: res[k][np.argmax(res[k][:, 1] >= 75), 0])
o = res[best]; t, v, x, slip, ped, gear, Fd, Fdrag = o.T
a = np.gradient(v, t)
b0 = make(30.); params = b0.params
drag = .5 * params[P + 'aerodynamics.air_density_kgpm3'] * params[P + 'aerodynamics.c_d'] * params[P + 'aerodynamics.A_m2']
q = np.load(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', '') + 'qss_profile_Shanghai_sha_f1_best_sf70h_f1_brake15000.npz', allow_pickle=True)
trac = np.array(json.loads(str(q['meta']))['traction'])
vg = np.arange(15., 86., 5.)
eng = qss_lap.machine_limit(setup, params, vg)
qa = np.minimum(eng, np.interp(vg, trac[:, 0], trac[:, 1])) - drag * vg ** 2 / m
print(f'\nbest slip target {best}; breakdown per speed (forces per unit mass, m/s^2)')
print('  v | QSS: engine  traction  drag  -> a_net | SIM: drive(pedal*cap) drag  other  -> a_net  pedal gear slip | sim/QSS')
env = []
for vv, e, qq in zip(vg, eng, qa):
    i = np.argmin(np.abs(v - vv))
    if abs(v[i] - vv) > 1.: print(f' {vv:3.0f} | not reached'); continue
    sl = slice(max(0, i - 10), i + 11)
    am = float(np.mean(a[sl])); fd = float(np.mean(Fd[sl])) / m; fdr = float(np.mean(Fdrag[sl])) / m
    other = am - (fd - fdr)
    tr = float(np.interp(vv, trac[:, 0], trac[:, 1]))
    print(f' {vv:3.0f} | {e:6.2f} {tr:7.2f} {drag*vv**2/m:6.2f} -> {qq:6.2f} | {fd:6.2f} {fdr:6.2f} {other:6.2f} -> {am:6.2f}  {np.mean(ped[sl]):.2f}  {int(np.median(gear[sl]))}  {np.mean(slip[sl]):.3f} | {am/qq:5.2f}')
    env.append((vv, am, qq))
np.savez(OUT, env=np.array(env), t=t, v=v, x=x, a=a, best=best)
