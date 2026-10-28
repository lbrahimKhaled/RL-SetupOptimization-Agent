#!/usr/bin/env python3
"""Ghost-lap MP4 of the sf70h_tum car at Shanghai from the comparison/ data (no simulation is run).

ERS and DRS off everywhere (the RL run's conditions): lap 2 of the RL driver
(comparison/rl/rl_det_seed0.csv), TUM's QSS lap `none` (comparison/tum/tum_none.csv) and that lap
driven in OCD by pure pursuit with ERS off (comparison/tum_pure_pursuit_ocd_noers/pp_f1.00.csv), all
started together at the line and played in real time. Left: the circuit with every car; right top:
a close-up following the RL car (sf70h_tum footprint, TUM `none` ghost box on its raceline); right
bottom: speed vs lap distance with each car's current point. Both maps show the RL driver's
reference line xi_ref (Track.path, the safe line) and the 20 xi_ref preview stations (up to 300 m
ahead on Track.path, ObsConfig) that are in its observation at that moment.

    .venv/bin/python scripts/comparison/ghost_video.py [--out comparison/figures/ghost_lap.mp4] [--speedup 1]

Needs ocd_first_lap on the path (track edges and the car footprint) and ffmpeg.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
CMP = ROOT / 'comparison'
L_RACELINE = 5340.77      # Track.reference polyline length; TUM's spline is 5341.43 m (README caveat)


def read_csv(path, cols):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) if r[c] != '' else np.nan for r in rows]) for c in cols}


def ocd_lap2(path):
    """Lap 2 of an OCD recording: t from the line, s along the raceline (unwrapped, monotone)."""
    d = read_csv(path, ['lap', 't_lap_s', 'x_m', 'y_m', 'yaw_rad', 'speed_mps', 's_raceline_m', 's_path_m', 'e_path_m', 'on_track'])
    m = d['lap'] == 2
    out = {k: v[m] for k, v in d.items()}
    s = out['s_raceline_m'].copy()
    s[(s > L_RACELINE / 2) & (np.arange(len(s)) < len(s) // 4)] -= L_RACELINE
    out['s'] = np.maximum.accumulate(s)
    out['t'] = out['t_lap_s']
    # close the lap at the exact (interpolated) crossing time from the metadata
    T = json.loads(Path(path).with_suffix('.meta.json').read_text())['lap_time_s']
    for k in out:
        out[k] = np.r_[out[k], out[k][-1]]
    out['t'][-1], out['s'][-1] = T, max(out['s'][-2], L_RACELINE)
    return out


def tum_lap(path):
    d = read_csv(path, ['s_m', 't_s', 'x_m', 'y_m', 'speed_mps'])
    s = d['s_m'] * L_RACELINE / d['s_m'][-1]
    xy = np.column_stack([d['x_m'], d['y_m']])
    seg = np.diff(xy, axis=0)
    yaw = np.arctan2(seg[:, 1], seg[:, 0])
    return {'t': d['t_s'], 's': s, 'x_m': d['x_m'], 'y_m': d['y_m'], 'speed_mps': d['speed_mps'],
            'yaw_rad': np.r_[yaw, yaw[-1]], 'on_track': np.full(len(s), np.nan)}


def state_at(lap, t):
    """Interpolated (x, y, yaw, v, s, finished) at time t since the line."""
    T = lap['t'][-1]
    tt = min(t, T)
    f = lambda k: float(np.interp(tt, lap['t'], lap[k]))
    yaw = float(np.interp(tt, lap['t'], np.unwrap(lap['yaw_rad'])))
    return f('x_m'), f('y_m'), yaw, f('speed_mps'), f('s'), t >= T


def t_at_s(lap, s):
    return float(np.interp(s, lap['s'], lap['t']))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=CMP / 'figures' / 'ghost_lap.mp4')
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--speedup', type=float, default=1., help='playback speed (1 = real time)')
    p.add_argument('--frames', type=int, default=None, help='render only the first N frames (test)')
    a = p.parse_args()
    from ocd_lap import cars
    from ocd_lap.track import Track
    track = Track('Shanghai')
    foot = np.array(cars.get('sf70h_tum').footprint)

    cars_ = [  # name, lap, colour, legend label, HUD label
        ('rl', ocd_lap2(CMP / 'rl' / 'rl_det_seed0.csv'), '#2a78d6', 'RL driver (OCD, ERS off)', 'RL driver'),
        ('tum_none', tum_lap(CMP / 'tum' / 'tum_none.csv'), '#eb6834', 'TUM QSS, ERS/DRS off', 'TUM QSS, ERS/DRS off'),
        ('pp', ocd_lap2(CMP / 'tum_pure_pursuit_ocd_noers' / 'pp_f1.00.csv'), '#1baf7a',
         "TUM QSS lap driven in OCD (pure pursuit)", 'TUM lap in OCD'),
    ]
    from rlsetup.driving.config import ObsConfig
    oc = ObsConfig()
    preview_d = oc.preview_horizon_m * (np.arange(1, oc.n_preview + 1) / oc.n_preview) ** oc.preview_exponent
    path_s, path_x, path_y = track.path[:, 0], track.path[:, 1], track.path[:, 2]
    off_pp = 1 - float(np.nanmean(cars_[2][1]['on_track']))
    rl = cars_[0][1]
    T_end = max(c[1]['t'][-1] for c in cars_) + 3.
    n_frames = int(T_end * a.fps / a.speedup)
    if a.frames:
        n_frames = min(n_frames, a.frames)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='#fcfcfb')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.25, 1], left=.02, right=.985,
                          bottom=.11, top=.75, hspace=.52, wspace=.1)
    ax_map = fig.add_subplot(gs[:, 0]); ax_close = fig.add_subplot(gs[0, 1]); ax_v = fig.add_subplot(gs[1, 1])
    for ax in (ax_map, ax_close):
        for edge in (track.left, track.right):
            e = np.vstack([edge, edge[0]])
            ax.plot(e[:, 0], e[:, 1], color='#64748b', lw=.8)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    keys = []
    for ax, lw in ((ax_map, .7), (ax_close, 1.3)):
        keys = [ax.plot(track.reference[:, 1], track.reference[:, 2], '--', color='#f4a582', lw=lw, zorder=1,
                        label='TUM raceline (TUM QSS and pure pursuit)')[0],
                ax.plot(path_x, path_y, '-', color='#52514e', lw=lw, zorder=2,
                        label='RL reference line ξref (safe line, in the RL observation)')[0]]
    previews = [ax.plot([], [], 'o', ms=ms, mfc='none', mec='#0b0b0b', mew=.9, zorder=4,
                        label='ξref preview points the RL policy sees (20, up to 300 m)')[0]
                for ax, ms in ((ax_map, 3), (ax_close, 5))]
    fig.legend(handles=keys + previews[1:], loc='center', bbox_to_anchor=(.785, .452), ncol=1, fontsize=7.5,
               frameon=False)
    ax_map.set_title('Shanghai International Circuit, lap 2 from the line', fontsize=11, loc='left')
    ax_close.set_title('Close-up on the RL car', fontsize=10, loc='left')

    dots, trails = {}, {}
    for name, lap, col, lab, _ in cars_:
        trails[name], = ax_map.plot([], [], color=col, lw=1.6, alpha=.75)
        dots[name], = ax_map.plot([], [], 'o', color=col, ms=8, mec='white', mew=1.2, label=lab, zorder=5)
    body_rl = Polygon(np.zeros((4, 2)), closed=True, color='#2a78d6', zorder=6)
    body_tum = Polygon(np.zeros((4, 2)), closed=True, fc='#eb6834', alpha=.45, ec='#c2410c', zorder=5)
    ax_close.add_patch(body_tum); ax_close.add_patch(body_rl)
    trail_close, = ax_close.plot([], [], color='#2a78d6', lw=1.5, alpha=.6)

    for name, lap, col, *_ in cars_:
        ax_v.plot(lap['s'] / 1000, lap['speed_mps'] * 3.6, color=col, lw=1, alpha=.55)
    cursors = {name: ax_v.plot([], [], 'o', color=col, ms=6, zorder=5)[0] for name, _, col, *_ in cars_}
    ax_v.set_xlim(0, L_RACELINE / 1000); ax_v.set_ylim(50, 330)
    ax_v.set_xlabel('lap distance along the TUM raceline (km)', fontsize=9); ax_v.set_ylabel('speed (km/h)', fontsize=9)
    ax_v.grid(alpha=.3); ax_v.tick_params(labelsize=8)

    fig.suptitle('sf70h_tum at Shanghai, ERS and DRS off: RL driver vs TUM laptime-simulation (ghost lap, real time)',
                 x=.02, ha='left', y=.985, fontsize=15, fontweight='bold')
    clock = fig.text(.02, .935, '', fontsize=9.5, family='monospace', va='top')
    hud = {}
    for k, (name, _, col, lab, _) in enumerate(cars_):
        y = .905 - k * .03
        fig.text(.02, y, '\u25cf', color=col, fontsize=12, va='top')
        hud[name] = fig.text(.035, y - .003, '', fontsize=9.5, family='monospace', va='top', color='#0b0b0b')
    fig.text(.02, .015, 'TUM QSS = point mass on its raceline, no track limits. RL and pure pursuit = OCD double-track, '
             f'12 kN·m brakes. RL kept the 4.6 × 2.0 m footprint on track; pure pursuit was off track {100 * off_pp:.0f} % '
             'of its lap. Data: comparison/ (README).', fontsize=7.5, color='#475569')
    foot_closed = foot

    def box(x, y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return foot_closed @ np.array([[c, s], [-s, c]]) + [x, y]

    def update(frame):
        t = frame * a.speedup / a.fps
        clock.set_text(f't = {t:6.2f} s')
        xr, yr, yawr, vr, sr, _ = state_at(rl, t)
        for name, lap, col, lab, _ in cars_:
            x, y, yaw, v, s, done = state_at(lap, t)
            dots[name].set_data([x], [y])
            k = np.searchsorted(lap['t'], min(t, lap['t'][-1]))
            trails[name].set_data(lap['x_m'][max(0, k - 400):k + 1], lap['y_m'][max(0, k - 400):k + 1])
            cursors[name].set_data([s / 1000], [v * 3.6])
            if done:
                status = f'FINISHED {lap["t"][-1]:7.3f} s'
            else:
                gap = t_at_s(lap, sr) - t_at_s(rl, sr) if name != 'rl' else 0.
                status = f'{v * 3.6:5.0f} km/h  {s / 1000:5.2f} km' + (f'  gap at RL point {gap:+6.2f} s' if name != 'rl' else '')
            hud[name].set_text(f'{lab:<50}{status}')
        body_rl.set_xy(box(xr, yr, yawr))
        sp = float(np.interp(min(t, rl['t'][-1]), rl['t'], rl['s_path_m']))
        st = (sp + preview_d) % track.length
        for pv in previews:
            pv.set_data(np.interp(st, path_s, path_x), np.interp(st, path_s, path_y))
        xt, yt, yawt, *_ = state_at(cars_[1][1], t)
        body_tum.set_xy(box(xt, yt, yawt))
        k = np.searchsorted(rl['t'], min(t, rl['t'][-1]))
        trail_close.set_data(rl['x_m'][max(0, k - 150):k + 1], rl['y_m'][max(0, k - 150):k + 1])
        ax_close.set_xlim(xr - 45, xr + 45); ax_close.set_ylim(yr - 22, yr + 22)
        return []

    a.out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / a.fps, blit=False)
    anim.save(a.out, writer=FFMpegWriter(fps=a.fps, codec='libx264', bitrate=3000,
                                         extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart']))
    update(int(min(n_frames - 1, 40 * a.fps / a.speedup)))
    fig.savefig(a.out.with_suffix('.png'), dpi=100)
    print(a.out, n_frames, 'frames')


if __name__ == '__main__':
    main()
