#!/usr/bin/env python3
"""Lap comparison of the sf70h_tum car at Shanghai from the comparison/ data (no simulation is run):
RL driver (lap 2, deterministic), TUM QSS (`none`: no ERS / DRS = the RL run's conditions;
`tum_default`: ERS with recuperation + DRS), TUM's `ers` speeds driven in OCD by pure pursuit.

Writes comparison/figures/lap_comparison.png (speed and time delta vs lap distance, time lost per
corner group, g-g) and comparison/figures/segments.json (per corner group: time, minimum and entry
speed, braking point, for every source). Corner groups run from one speed peak of TUM `none` to the
next, so each holds one braking zone and its corner(s). Distance = arc length on TUM's raceline (OCD
laps projected onto it; TUM's spline rescaled to the polyline length).

    .venv/bin/python scripts/comparison/compare_laps.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ghost_video import CMP, L_RACELINE, ocd_lap2, read_csv, tum_lap, t_at_s

COLORS = {'rl': '#2a78d6', 'tum_none': '#eb6834', 'tum_default': '#1baf7a', 'pp': '#eda100'}
LABELS = {'rl': 'RL driver (OCD, ERS off)', 'tum_none': 'TUM QSS, ERS/DRS off',
          'tum_default': 'TUM QSS default (ERS+recup, DRS)', 'pp': "TUM's speeds in OCD (pure pursuit, ERS on)"}
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def ocd_accel(path):
    d = read_csv(path, ['lap', 'ax_mps2', 'ay_mps2'])
    m = d['lap'] == 2
    return d['ax_mps2'][m], d['ay_mps2'][m]


def peaks(v, s, min_gap=150., min_drop=8.):
    """Indices of speed maxima followed by a drop of >= min_drop m/s, >= min_gap m apart."""
    idx = [i for i in range(1, len(v) - 1) if v[i] >= v[i - 1] and v[i] > v[i + 1]]
    out = []
    for i in idx:
        j = i + 1
        while j < len(v) - 1 and v[j + 1] <= v[j]:
            j += 1
        if v[i] - v[j] >= min_drop and (not out or s[i] - s[out[-1]] >= min_gap):
            out.append(i)
    return out


def speed_at(lap, s):
    return np.interp(s, lap['s'], lap['speed_mps'])


def brake_point(lap, apex, prev_apex, grid=1.):
    """Braking point of the corner whose slowest point (TUM QSS) is at `apex`: where the speed peaks
    between the previous corner's slowest point and this one (wrapping round the line)."""
    ss = np.arange(prev_apex if prev_apex < apex else prev_apex - L_RACELINE, apex, grid)
    v = speed_at(lap, ss % L_RACELINE)
    return float(ss[np.argmax(v)] % L_RACELINE)


def main():
    laps = {'rl': ocd_lap2(CMP / 'rl' / 'rl_det_seed0.csv'),
            'tum_none': tum_lap(CMP / 'tum' / 'tum_none.csv'),
            'tum_default': tum_lap(CMP / 'tum' / 'tum_tum_default.csv'),
            'pp': ocd_lap2(CMP / 'tum_pure_pursuit_ocd' / 'pp_f1.00.csv')}
    ref = laps['tum_none']
    pk = peaks(ref['speed_mps'], ref['s'])
    bounds = [ref['s'][i] for i in pk]
    edges = [0.] + bounds + [L_RACELINE]
    # merge a leading/trailing sliver into its neighbour
    edges = [e for k, e in enumerate(edges) if k in (0, len(edges) - 1) or 60. < e < L_RACELINE - 60.]
    segs, apexes = [], []
    for k in range(len(edges) - 1):
        ss = np.linspace(edges[k], edges[k + 1], 400)
        apexes.append(float(ss[np.argmin(speed_at(ref, ss))]))
    for k in range(len(edges) - 1):
        s0, s1 = edges[k], edges[k + 1]
        ss = np.linspace(s0, s1, 400)
        row = {'segment': k + 1, 's0_m': round(s0, 1), 's1_m': round(s1, 1),
               'apex_s_m': round(float(ss[np.argmin(speed_at(ref, ss))]), 1)}
        for name, lap in laps.items():
            v = speed_at(lap, ss)
            row[name] = {'time_s': round(t_at_s(lap, s1) - t_at_s(lap, s0), 3),
                         'min_speed_kmh': round(float(v.min()) * 3.6, 1),
                         'max_speed_kmh': round(float(v.max()) * 3.6, 1),
                         'brake_point_m': round(brake_point(lap, apexes[k], apexes[k - 1]), 1)}
        row['rl_minus_tum_none_s'] = round(row['rl']['time_s'] - row['tum_none']['time_s'], 3)
        row['pp_minus_tum_none_s'] = round(row['pp']['time_s'] - row['tum_none']['time_s'], 3)
        segs.append(row)
    out_dir = CMP / 'figures'
    out_dir.mkdir(exist_ok=True)
    lap_times = {k: round(float(v['t'][-1]), 3) for k, v in laps.items()}
    (out_dir / 'segments.json').write_text(json.dumps({'lap_times_s': lap_times, 'segments': segs}, indent=1))

    # --- figure ---
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK})
    fig = plt.figure(figsize=(14, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], height_ratios=[1.15, 1, 1], left=.06, right=.985,
                          bottom=.06, top=.87, hspace=.42, wspace=.18)
    ax_v = fig.add_subplot(gs[0, 0]); ax_dt = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_seg = fig.add_subplot(gs[2, 0]); ax_gg = fig.add_subplot(gs[1:, 1]); ax_tab = fig.add_subplot(gs[0, 1])
    order = ['tum_default', 'tum_none', 'pp', 'rl']
    for ax in (ax_v, ax_dt):
        for k in range(1, len(edges) - 1):
            ax.axvline(edges[k] / 1000, color='#e5e4df', lw=.8, zorder=0)
        ax.set_facecolor(SURFACE); ax.grid(axis='y', color='#e5e4df', lw=.6)
    for name in order:
        lap = laps[name]
        ax_v.plot(lap['s'] / 1000, lap['speed_mps'] * 3.6, color=COLORS[name], lw=2 if name == 'rl' else 1.4,
                  label=f"{LABELS[name]}: {lap_times[name]:.2f} s", zorder=3 if name == 'rl' else 2)
    ax_v.set_ylabel('speed (km/h)')
    ax_tab.legend(*ax_v.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0, 1.08), fontsize=8,
                  frameon=False, handlelength=1.8)
    ax_v.set_title('Speed along the lap', loc='left', fontsize=10.5)
    grid = np.linspace(0, L_RACELINE, 2000)
    t_ref = np.interp(grid, ref['s'], ref['t'])
    for name in ['tum_default', 'pp', 'rl']:
        lap = laps[name]
        dt = np.interp(grid, lap['s'], lap['t']) - t_ref
        ax_dt.plot(grid / 1000, dt, color=COLORS[name], lw=2 if name == 'rl' else 1.4)
        ax_dt.annotate(f'{dt[-1]:+.2f} s', (grid[-1] / 1000, dt[-1]), xytext=(4, 0), textcoords='offset points',
                       color=INK2, fontsize=8, va='center')
    ax_dt.axhline(0, color=COLORS['tum_none'], lw=1.4)
    ax_dt.set_ylabel('time vs TUM QSS ERS/DRS off (s)'); ax_dt.set_xlabel('lap distance along the TUM raceline (km)')
    ax_dt.set_title('Cumulative time difference (positive = behind)', loc='left', fontsize=10.5)
    ax_v.set_xlim(0, L_RACELINE / 1000 * 1.035)

    x = np.arange(len(segs))
    d_rl = [r['rl_minus_tum_none_s'] for r in segs]
    ax_seg.bar(x, d_rl, width=.62, color=COLORS['rl'], zorder=2)
    for i, v in enumerate(d_rl):
        if abs(v) >= .4:
            ax_seg.annotate(f'{v:+.1f}', (i, v), xytext=(0, 3 if v > 0 else -10), textcoords='offset points',
                            ha='center', fontsize=8, color=INK2)
    ax_seg.set_xticks(x, [f"{r['apex_s_m'] / 1000:.2f}" for r in segs], fontsize=7.5)
    ax_seg.set_xlabel('corner group (slowest point of TUM QSS, km)'); ax_seg.set_ylabel('RL minus TUM QSS (s)')
    ax_seg.axhline(0, color='#c3c2b7', lw=.8); ax_seg.grid(axis='y', color='#e5e4df', lw=.6, zorder=0)
    ax_seg.set_facecolor(SURFACE)
    ax_seg.set_title('Time the RL driver loses per corner group vs TUM QSS (ERS/DRS off)', loc='left', fontsize=10.5)

    ax_rl, ay_rl = ocd_accel(CMP / 'rl' / 'rl_det_seed0.csv')
    ax_gg.scatter(ay_rl[::3] / 9.81, ax_rl[::3] / 9.81, s=4, color=COLORS['rl'], alpha=.35, lw=0,
                  label='RL driver (OCD, body frame, 50 Hz)')
    tn = read_csv(CMP / 'tum' / 'tum_none.csv', ['ax_mps2', 'ay_mps2'])
    ax_gg.scatter(tn['ay_mps2'] / 9.81, tn['ax_mps2'] / 9.81, s=9, color=COLORS['tum_none'], alpha=.7, lw=0,
                  label='TUM QSS, ERS/DRS off (per 5 m)')
    ax_gg.axhline(0, color='#c3c2b7', lw=.7); ax_gg.axvline(0, color='#c3c2b7', lw=.7)
    ax_gg.set_xlabel('lateral acceleration (g)'); ax_gg.set_ylabel('longitudinal acceleration (g)')
    ax_gg.set_aspect('equal'); ax_gg.set_facecolor(SURFACE); ax_gg.grid(color='#e5e4df', lw=.6)
    ax_gg.legend(loc='upper left', bbox_to_anchor=(0, -.13), fontsize=8, frameon=False, markerscale=2.5)
    ax_gg.set_title('g-g diagram', loc='left', fontsize=10.5)

    ax_tab.axis('off')
    rows = [['', 'lap 2 (s)', 'vs TUM QSS off']]
    for name in ['tum_default', 'tum_none', 'pp', 'rl']:
        rows.append([LABELS[name].split(' (')[0] if name != 'pp' else "TUM speeds in OCD", f'{lap_times[name]:.2f}',
                     f"{lap_times[name] - lap_times['tum_none']:+.2f}"])
    rows.append(['Real SF70H, 2017 Q (Vettel)', '91.86', f"{91.864 - lap_times['tum_none']:+.2f}"])
    tb = ax_tab.table(cellText=rows, bbox=[0, 0, 1, .56], cellLoc='left', colWidths=[.5, .2, .3])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor('#e5e4df'); cell.set_facecolor(SURFACE)
        if r == 0:
            cell.set_text_props(color=INK2)
    fig.suptitle('sf70h_tum at Shanghai: RL driver vs TUM laptime-simulation (lap 2)', x=.06, ha='left', y=.985,
                 fontsize=14, fontweight='bold', color=INK)
    fig.text(.06, .935, 'Not like-for-like: TUM = point mass on its raceline, ideal brakes, no track limits. RL and pure '
             'pursuit = OCD double-track, 12 kN·m brakes, 4.6 × 2.0 m footprint checked.\nPure pursuit was off track on '
             '29 % of its lap (TUM\'s line does not fit the footprint). Data: comparison/README.md.',
             fontsize=8.5, color=INK2, va='top')
    fig.savefig(out_dir / 'lap_comparison.png', dpi=130, facecolor=SURFACE)
    print(json.dumps(lap_times))
    for r in segs:
        print(f"seg {r['segment']:2d} {r['s0_m']:7.1f}-{r['s1_m']:7.1f} apex {r['apex_s_m']:7.1f}  "
              f"RL-TUMnone {r['rl_minus_tum_none_s']:+6.2f}  PP-TUMnone {r['pp_minus_tum_none_s']:+6.2f}  "
              f"min v RL/TUM {r['rl']['min_speed_kmh']:5.1f}/{r['tum_none']['min_speed_kmh']:5.1f}  "
              f"max v RL/TUM {r['rl']['max_speed_kmh']:5.1f}/{r['tum_none']['max_speed_kmh']:5.1f}  "
              f"brake RL/TUM {r['rl']['brake_point_m']}/{r['tum_none']['brake_point_m']}")


if __name__ == '__main__':
    main()
