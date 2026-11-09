"""Test 2: QSS lap on the w3.5 line with the car's REAL straight-line acceleration (test 1b) instead of
QSS's engine-minus-drag model. Same g-g-v (braking + lateral) and the same tph solver; only the
acceleration ("machines") limit changes. Then segment-by-segment vs the RL's best laps."""
import numpy as np, json, os, sys, math
from dataclasses import replace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ocd_lap import qss_lap
from ocd_lap.track import Track
from rlsetup.driving import qss_profile
from rlsetup.driving.corners import cumulative_time
from rlsetup.driving.reference import ReferenceLine
from rlsetup.driving.setups import get as get_setup
from rlsetup.driving.train import parse_args
from rlsetup.driving.train_config import PRESETS, apply_overrides

S = os.path.dirname(os.path.abspath(__file__)) + os.sep
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')) + os.sep
args = parse_args(['--preset', 'f1_r4', '--run-name', 'x'])
cfg = apply_overrides(PRESETS['f1_r4'](), args.overrides)
track = Track(cfg.env.track)
qss = qss_profile.for_env(track, cfg.env)
ref = ReferenceLine.for_track(track, cfg.env.reference)
setup = get_setup(cfg.env.setup)
ggv_data = (np.asarray(qss.meta['ggv'], float), np.asarray(qss.meta['traction'], float), None)
drag = float(qss.meta.get('drag_coeff', 0.92)); m = 733.

env = np.load(S + 'real1b.npz')['env']            # rows (v, a_net_sim, a_net_qss)
vs, a_sim = env[:, 0], env[:, 1]
# extend: below 15 m/s hold the 15 m/s value; above 80 m/s extrapolate the last slope (terminal 80.6)
slope = (a_sim[-1] - a_sim[-2]) / (vs[-1] - vs[-2])
def a_real(v):
    v = np.asarray(v, float)
    return np.where(v <= vs[-1], np.interp(v, vs, a_sim), a_sim[-1] + slope * (v - vs[-1]))

orig_ml = qss_lap.machine_limit
def run(label, machine_fn=None, trac_big=False):
    gd = ggv_data
    if machine_fn is not None:
        qss_lap.machine_limit = lambda setup, params, speeds, ers=None, with_power=False: machine_fn(np.asarray(speeds, float))
    if trac_big:
        tr = ggv_data[1].copy(); tr[:, 1] = 1e3; gd = (ggv_data[0], tr, None)
    try:
        r = qss_lap.lap_time(setup, track, ggv_data=gd, car=cfg.env.car, ers=cfg.env.ers, drs=False,
                             path=np.asarray(ref.path, float))
    finally:
        qss_lap.machine_limit = orig_ml
    print(f'{label:62s} lap {r["lap_time_s"]:7.3f} s   v_max {r["v_max_mps"]:.2f}')
    return r

print('QSS on the IQP w3.5 line, same g-g-v; only the straight-line acceleration model changes:')
base = run('A  QSS as used (engine - drag, no rolling res./inertia)')
# B: the real net acceleration everywhere (machine = a_sim + drag v^2/m so that tph's net = a_sim)
realB = run('B  real car accel (sim, full pedal + slip control)', lambda v: a_real(v) + drag * v ** 2 / m, trac_big=True)
# C: real high-speed losses only (QSS traction kept at low speed; ~1.1-1.5 m/s^2 extra resistance)
eng_q = None
def mC(v):
    from ocd_lap.backend import OCDBackend, StartPose
    b = OCDBackend(car=cfg.env.car); b.reset(setup, StartPose(speed=0.))
    e = orig_ml(setup, b.params, v)
    other = np.interp(v, vs, a_sim - (env[:, 2]) * 0)  # placeholder, replaced below
    return e
# extra resistance per speed from test 1b breakdown (other term), measured: ~1.1 (>=60) .. 1.5 (30-40)
other_v = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80.])
other_a = np.array([1.39, 1.28, 1.23, 1.51, 1.55, 1.52, 1.34, 1.32, 1.31, 1.16, 1.19, 1.11, 1.11, 1.10])
def mC(v):
    from ocd_lap.backend import OCDBackend, StartPose
    b = OCDBackend(car=cfg.env.car); b.reset(setup, StartPose(speed=0.))
    return orig_ml(setup, b.params, v) - np.interp(v, other_v, other_a)
realC = run('C  QSS + only the measured extra resistance (rolling res./inertia)', mC)

L = float(track.length)
def line_to_track(v_line):
    t_line = cumulative_time(ref.path, v_line)
    xy = np.asarray(ref.path, float)[:-1, 1:]
    s = np.empty(len(xy)); hint = None
    for i, p in enumerate(xy):
        hint = s[i] = float(np.asarray(track.match(p, hint)[0]))
    s = np.unwrap(s, period=L)
    return s, np.asarray(v_line), t_line[:-1]
sA, vA, tA = line_to_track(base['v_profile'])
sB, vB, tB = line_to_track(realB['v_profile'])
sC, vC, tC = line_to_track(realC['v_profile'])

ct = json.load(open(R + 'data/corner_table_6c61b28a4876.json'))
U = ct['units']
print('\nentry (QSS braking point) and exit speeds, m/s:   QSS  | real-accel QSS (B)')
for u in U:
    e, x = u['start_m'], u['exit_m']
    f = lambda sx, vx, p: float(np.interp(p, sx % L, vx, period=L))
    print(f"  {u['name']} entry {f(sA,vA,e):5.1f} | {f(sB,vB,e):5.1f}    exit {f(sA,vA,x):5.1f} | {f(sB,vB,x):5.1f}")

# RL laps: best of each run
def rl(run, step):
    d = np.load(R + f'runs/mac/D2/{run}/seed0/laps/step_{step}_det.npz')
    msk = d['lap_index'] == 2
    s, t, v = d['s'][msk], d['time_s'][msk], d['speed'][msk]
    o = np.argsort(s); s, t, v = s[o], t[o] - t[o][0], v[o]
    return s, t, v, d['actions'][msk][o], float(d['lap_times_s'][1])
laps = {'r4 6.95M': rl('f1_r4', 6950000), 'r4 5.45M': rl('f1_r4', 5450000), 'r3 0.45M': rl('f1_r3', 450000),
        'r2 6.55M': rl('f1_r2', 6550000), 'w35 9.2M': rl('f1_best_w35', 9200000)}
print('\nRL lap 2 times:', {k: round(v[4], 2) for k, v in laps.items()})

# segments: corner windows and straights between them (track s)
edges = []
for u in U: edges += [u['start_m'], u['exit_m']]
segs = []
for k, u in enumerate(U):
    segs.append((u['name'], u['start_m'], u['exit_m']))
    nxt = U[(k + 1) % len(U)]['start_m']
    segs.append((f"str {u['exit_m']:.0f}-{nxt:.0f}", u['exit_m'], nxt))
def seg_time_ref(sx, tx, a, b):
    T = tx[-1] + (tx[-1] - tx[-2])  # approx closing
    sm = sx - sx[0]
    ta = np.interp((a - sx[0]) % L, sm, tx); tb = np.interp((b - sx[0]) % L, sm, tx)
    return (tb - ta) % (tB_end if False else 1e9) if tb >= ta else tb - ta + lapT
def seg_time(sx, tx, a, b, lap):
    sm = (sx - sx[0])
    ta = np.interp((a - sx[0]) % L, sm, tx); tb = np.interp((b - sx[0]) % L, sm, tx)
    d = tb - ta
    return d if d >= 0 else d + lap
lapA, lapB = base['lap_time_s'], realB['lap_time_s']
rlk = 'r4 6.95M'
s_r, t_r, v_r, a_r, lap_r = laps[rlk]
# RL laps start at the line s=0 crossing: shift by s[0]
print(f'\nsegment times [s]: QSS | real-accel QSS (B) | RL best {rlk} | RL - B  (+ = RL slower than the realistic reference)')
tot = [0, 0, 0]
rows = []
for name, a, b in segs:
    x = seg_time(sA, tA, a, b, lapA); y = seg_time(sB, tB, a, b, lapB)
    sm = s_r - s_r[0]
    ta = np.interp((a - s_r[0]) % L, sm, t_r); tb = np.interp((b - s_r[0]) % L, sm, t_r)
    z = tb - ta if tb >= ta else tb - ta + lap_r
    rows.append((name, x, y, z)); tot[0] += x; tot[1] += y; tot[2] += z
    print(f'  {name:16s} {x:7.3f} {y:7.3f} {z:7.3f}   {z - y:+6.3f}')
print(f'  {"sum":16s} {tot[0]:7.3f} {tot[1]:7.3f} {tot[2]:7.3f}   {tot[2] - tot[1]:+6.3f}')
np.savez(S + 'real2.npz', sA=sA, vA=vA, sB=sB, vB=vB, sC=sC, vC=vC, lapA=lapA, lapB=lapB, lapC=realC['lap_time_s'])
json.dump(rows, open(S + 'real2_segs.json', 'w'))
