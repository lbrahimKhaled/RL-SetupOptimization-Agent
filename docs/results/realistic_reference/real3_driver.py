"""Test 3: does the RL driver use the car? Throttle / braking / grip on its best laps vs the measured
limits (straight-line accel from test 1b, braking + lateral g-g-v from the QSS cache)."""
import numpy as np, json, os, sys
S = os.path.dirname(os.path.abspath(__file__)) + os.sep
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')) + os.sep
env = np.load(S + 'real1b.npz')['env']; vs, a_sim = env[:, 0], env[:, 1]
q = np.load(R + 'data/qss_profile_Shanghai_sha_f1_best_sf70h_f1_brake15000.npz', allow_pickle=True)
meta = json.loads(str(q['meta'])); ggv = np.array(meta['ggv'])  # v, ax_brake, ay_max
print('g-g-v rows (v, brake decel, lateral):', np.round(ggv, 1).tolist())
r2 = np.load(S + 'real2.npz'); L = 5388.773713834052
ct = json.load(open(R + 'data/corner_table_6c61b28a4876.json')); U = ct['units']

def lap(run, step):
    d = np.load(R + f'runs/mac/D2/{run}/seed0/laps/step_{step}_det.npz')
    m = d['lap_index'] == 2
    out = {k: d[k][m] for k in ('s', 'x', 'y', 'speed', 'time_s', 'actions', 'slip_rl', 'slip_rr', 'kappa', 'Fz', 'gear', 'e', 'track_width')}
    out['lap'] = float(d['lap_times_s'][1]); return out

for run, step in [('f1_r4', 6950000), ('f1_r4', 8700000)]:
    try: d = lap(run, step)
    except FileNotFoundError: continue
    s, v, t = d['s'], d['speed'], d['time_s']
    thr = np.clip((d['actions'][:, 1] + .5) / 1.5, 0, 1); brk = np.clip((d['actions'][:, 2] + .5) / 1.5, 0, 1)
    acc = np.gradient(v, t)
    # driven-path lateral acceleration from x, y (smoothed heading rate * v)
    hd = np.unwrap(np.arctan2(np.gradient(d['y'], t), np.gradient(d['x'], t)))
    k = 7; ker = np.ones(k) / k
    yr = np.convolve(np.gradient(hd, t), ker, 'same'); ay = v * yr
    rear = np.maximum(np.abs(d['slip_rl']), np.abs(d['slip_rr']))
    print(f'\n===== {run} {step/1e6:.2f}M lap 2 {d["lap"]:.2f} s =====')
    print('pedal use over the lap: full throttle (>=0.99) {:.0%}, partial throttle {:.0%}, braking {:.0%}, coasting {:.0%}, throttle+brake overlap {:.0%}'.format(
        np.mean(thr >= .99), np.mean((thr > .02) & (thr < .99)), np.mean(brk > .02), np.mean((thr <= .02) & (brk <= .02)), np.mean((thr > .02) & (brk > .02))))
    # 1) straights: accel achieved vs the car's real full-pedal accel at the same speed
    print('\n1) ACCELERATION on straights/exits (not braking, |ay| < 8 m/s^2):')
    ok = (brk <= .02) & (np.abs(ay) < 8.) & (v > 40)
    for lo, hi in [(40, 50), (50, 60), (60, 70), (70, 76), (76, 81)]:
        mm = ok & (v >= lo) & (v < hi)
        if mm.sum() < 5: continue
        real = np.interp(v[mm], vs, a_sim)
        print(f'   v {lo}-{hi}: time share {mm.mean():5.1%}  throttle mean {thr[mm].mean():.2f}, at full {np.mean(thr[mm] >= .99):4.0%}  | accel RL {acc[mm].mean():5.2f} vs car max {real.mean():5.2f} m/s^2 ({acc[mm].mean()/real.mean():4.0%})  rear slip {rear[mm].mean():.3f}')
    # per straight: time lost vs flooring it, simulated with the measured a(v)
    print('   per straight (exit of unit k -> braking point of k+1): RL throttle use, and time if floored from the RL exit speed with the real a(v) until the RL starts braking:')
    for kk, u in enumerate(U):
        a0, b0 = u['exit_m'], U[(kk + 1) % 7]['start_m']
        seg = ((s - a0) % L) <= ((b0 - a0) % L)
        idx = np.where(seg)[0]
        if len(idx) < 5: continue
        # contiguous in lap order
        idx = idx[np.argsort((s[idx] - a0) % L)]
        # first braking sample in this segment (or its end)
        bi = next((i for i in idx if brk[i] > .05), idx[-1])
        part = idx[: list(idx).index(bi) + 1]
        ss = (s[part] - a0) % L
        tl = t[part[-1]] - t[part[0]]
        # floored: integrate dv/ds = a/v from v at segment start to the same distance
        vv = v[part[0]]; tt = 0.; grid = np.linspace(0, ss[-1], 400); ds = grid[1] - grid[0]
        for _ in grid[1:]:
            a = float(np.interp(vv, vs, a_sim)) if vv <= 80 else a_sim[-1]
            vn = np.sqrt(max(vv * vv + 2 * a * ds, 1.)); tt += 2 * ds / (vv + vn); vv = vn
        print(f'     {u["name"]}->{U[(kk+1)%7]["name"]}: {ss[-1]:5.0f} m before braking, full throttle {np.mean(thr[part] >= .99):4.0%}, '
              f'mean throttle {thr[part].mean():.2f}, v {v[part[0]]:.1f}->{v[part[-1]]:.1f} (floored: ->{vv:.1f}), time {tl:.3f} s vs floored {tt:.3f} s = {tl-tt:+.3f}')
    # 2) braking
    print('\n2) BRAKING (brake > 0.05): decel achieved vs g-g-v braking limit at that speed')
    bm = brk > .05
    for lo, hi in [(30, 45), (45, 60), (60, 70), (70, 82)]:
        mm = bm & (v >= lo) & (v < hi)
        if mm.sum() < 3: continue
        lim = np.interp(v[mm], ggv[:, 0], ggv[:, 1])
        print(f'   v {lo}-{hi}: brake pedal mean {brk[mm].mean():.2f}, max {brk[mm].max():.2f}; decel RL {-acc[mm].mean():5.2f} (peak {-acc[mm].min():5.2f}) vs limit {lim.mean():5.2f} m/s^2 ({-acc[mm].mean()/lim.mean():4.0%})')
    # braking points vs QSS / realistic QSS
    print('   braking onset (first brake > 0.05 before each unit) vs the QSS braking point:')
    for u in U:
        st = u['start_m']; ap = u['apexes_m'][0]
        w = (((s - (st - 250)) % L) <= ((ap - (st - 250)) % L))
        ii = np.where(w)[0]; ii = ii[np.argsort((s[ii] - (st - 250)) % L)]
        b = next((i for i in ii if brk[i] > .05), None)
        if b is None: continue
        vmin_i = ii[np.argmin(v[ii])]
        vb = float(np.interp(st, r2['sB'] % L, r2['vB'], period=L))
        print(f'     {u["name"]}: RL brakes at {s[b]:7.1f} m ({(s[b]-st+L/2)%L-L/2:+6.1f} m vs QSS point) at {v[b]:.1f} m/s (QSS {np.interp(st, np.array(ct["s"]), np.array(ct["v"])):.1f}, realistic {vb:.1f}); min speed {v[vmin_i]:.1f} at {s[vmin_i]:.0f} m (QSS apex {u["v_apex"][0]:.1f})')
    # 3) cornering grip
    print('\n3) CORNERING: lateral acceleration at the slowest point of each unit vs g-g-v lateral limit')
    for u in U:
        for apx, vq in zip(u['apexes_m'], u['v_apex']):
            w = ((s - apx + 60) % L) <= 120
            ii = np.where(w)[0]
            if len(ii) < 3: continue
            i = ii[np.argmax(np.abs(ay[ii]))]
            lim = float(np.interp(v[i], ggv[:, 0], ggv[:, 2]))
            print(f'     {u["name"]} apex {apx:6.0f}: RL v_min {v[ii].min():5.1f} (QSS {vq:5.1f}), peak |ay| {abs(ay[i]):5.1f} at {v[i]:.1f} m/s vs limit {lim:5.1f} ({abs(ay[i])/lim:4.0%}), rear slip {rear[i]:.3f}, throttle {thr[i]:.2f} brake {brk[i]:.2f}')
