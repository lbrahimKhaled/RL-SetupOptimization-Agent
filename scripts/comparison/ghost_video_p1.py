#!/usr/bin/env python3
"""Render three immutable lap traces: p1 RL, standard TUM QSS, and TUM tracked in OCD.

No training, model inference or simulation runs here. TUM exports a trajectory and speed
profile, not OCD actuator commands. PP commands are computed by geometric PP + PI.
OCD lap 2 is clipped by interpolating the complete raw recording at both exact crossings.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_input(path):
    path = Path(path).resolve()
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(meta['sha256'] == digest, f'CSV hash differs from metadata: {path}')
    require(meta['file'] == path.name, f'CSV filename differs from metadata: {path}')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f'Empty recording: {path}')
    return rows, meta, {'path': str(path), 'sha256': digest}


def numeric(rows, names):
    result = {}
    for name in names:
        values = np.array([float(row[name]) for row in rows])
        require(np.all(np.isfinite(values)), f'Nonfinite data in {name}')
        result[name] = values
    return result


def validate_metadata(rl, pp, tum):
    require(rl['source'] == 'rl' and rl['episode']['kind'] == 'deterministic',
            'RL input must be a deterministic RL export')
    require(rl['episode']['clean'] is True and rl['episode']['termination'] == 'lap_complete',
            'RL recording must have a clean completed lap')
    require(rl['episode']['reproduces_training_eval'] is True, 'RL must reproduce its frozen training evaluation')
    require(rl['saved_run_config']['env']['setup'] == 'p1_TUM', 'Expected p1_TUM RL setup')
    require(pp['conditions']['setup_name'] == 'p1_TUM', 'Expected p1_TUM PP setup')
    for key in ('car', 'setup', 'max_brake_torque_Nm', 'footprint_m', 'physics_dt_s'):
        require(rl['conditions'][key] == pp['conditions'][key], f'RL / PP {key} differs')
    for meta in (rl, pp):
        for key in ('ers', 'drs', 'recuperation'):
            require(meta['conditions'][key] is False, f'{key} must be off')
    require(np.isclose(rl['conditions']['decision_dt_s'], pp['controller']['control_dt_s'], atol=1e-12),
            'RL / PP command rates differ')
    require(pp['controller']['speed_scale_f'] == 1.0, 'PP must use full TUM target speed')
    require(pp['controller']['weights_loaded'] is False, 'PP cannot use learned weights')
    for key in ('ocd_lap_code_version',):
        if 'code' in rl and 'code' in pp:
            require(rl['code'][key] == pp['code'][key], f'RL / PP simulator {key} differs')
    require(pp['summary']['completed_lap2'] is True, 'PP lap 2 did not finish')
    require(pp['stop_rules']['footprint_exit_terminates'] is False,
            'Expected explicitly nonterminating PP diagnostic')
    require(pp['valid_fair_RL_comparison'] is False, 'PP diagnostic must retain its invalid-clean-lap status')
    require(tum['source'] == 'tum' and tum['variant'] == 'none', 'Expected standard TUM none variant')
    require(tum['ers_used_J'] == 0 and tum['drs_share_of_points'] == 0, 'TUM ERS / DRS must be off')
    require(pp['tum_input']['sha256'] == tum['sha256'], 'PP used a different TUM trajectory')


def clip_trace(data, start, finish):
    """Interpolate continuous columns at exact boundaries; preserve interval-end flags/actions.

    on_track and commands describe the interval ending at a recorded row. Thus the
    first row at/after a query supplies its discrete state (no linear interpolation).
    """
    times = data['t_s']
    require(np.all(np.diff(times) > 0), 'Raw recording times must strictly increase')
    require(times[0] <= start < finish <= times[-1], 'Recording does not bracket both lap crossings')
    selected = times[(times > start) & (times < finish)]
    query = np.r_[start, selected, finish]
    out = {'t': query - start}
    discrete = {'on_track', 'steer_cmd_rad', 'throttle', 'brake'}
    indices = np.minimum(np.searchsorted(times, query, side='left'), len(times) - 1)
    for name, values in data.items():
        if name in discrete:
            out[name] = values[indices]
        elif name == 'yaw_rad':
            out[name] = np.interp(query, times, np.unwrap(values))
        else:
            out[name] = np.interp(query, times, values)
    return out


def ocd_lap(rows, meta, role, raceline_length, path_length):
    cols = ['t_s', 't_lap_s', 'lap', 'x_m', 'y_m', 'yaw_rad', 'speed_mps',
            's_raceline_m', 's_path_m', 'on_track', 'steer_cmd_rad', 'throttle', 'brake']
    data = numeric(rows, cols)
    second = data['lap'] == 2
    require(np.any(second), f'{role}: no lap 2 samples')
    if role == 'rl':
        crossings = meta['episode']['recorder_check']['recorder_crossings_s']
        require(len(crossings) == 3, 'Expected run-in plus two completed RL laps')
        start, finish = crossings[1:3]
    else:
        # PP CSV timestamps are rounded to 7 significant digits. Use the stable
        # start clock and exact metadata duration, then cross-check lap-3 clock.
        estimates = data['t_s'][second] - data['t_lap_s'][second]
        start = float(np.median(estimates))
        finish = start + float(meta['lap_time_s'])
        third = data['lap'] == 3
        require(np.any(third), 'PP full trace needs the sample beyond its final crossing')
        end_est = data['t_s'][third] - data['t_lap_s'][third]
        require(np.max(np.abs(end_est - finish)) < 2e-5, 'PP end clock disagrees with metadata')
    require(abs(finish - start - meta['lap_time_s']) < 1e-8, f'{role}: lap clock mismatch')
    # Projected station is unwrapped BEFORE clipping; it is only a distance HUD.
    # No position geometry is stretched, shifted, normalized or reprojected.
    data['distance_m'] = np.unwrap(data['s_raceline_m'] * 2*np.pi/raceline_length) * raceline_length/(2*np.pi)
    data['s_path_unwrapped_m'] = np.unwrap(data['s_path_m'] * 2*np.pi/path_length) * path_length/(2*np.pi)
    clipped = clip_trace(data, start, finish)
    clipped['distance_m'] -= clipped['distance_m'][0]
    if role == 'rl':
        require(np.all(clipped['on_track'] == 1), 'RL clean metadata conflicts with lap-2 off-track flags')
    clipped['_start_s'] = start
    return clipped


def tum_lap(rows, meta):
    data = numeric(rows, ['t_s', 's_m', 'x_m', 'y_m', 'speed_mps'])
    require(data['t_s'][0] == 0 and np.all(np.diff(data['t_s']) > 0), 'Invalid TUM lap clock')
    # Metadata stores full precision; CSV's final time has 7 significant digits.
    require(abs(data['t_s'][-1] - meta['lap_time_s']) < 1e-5, 'TUM CSV / metadata finish mismatch')
    data['t_s'][-1] = float(meta['lap_time_s'])
    xy = np.column_stack((data['x_m'], data['y_m']))
    delta = np.roll(xy[:-1], -1, axis=0) - np.roll(xy[:-1], 1, axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    return {**data, 't': data['t_s'], 'distance_m': data['s_m'],
            'yaw_rad': np.unwrap(np.r_[yaw, yaw[0]])}


def state_at(lap, time):
    """Shared seconds clock, unwrapped heading and final pose held after finish."""
    q = min(max(float(time), 0.), lap['t'][-1])
    state = {key: float(np.interp(q, lap['t'], lap[key]))
             for key in ('x_m', 'y_m', 'yaw_rad', 'speed_mps', 'distance_m')}
    index = min(int(np.searchsorted(lap['t'], q, side='left')), len(lap['t'])-1)
    for key in ('on_track', 'steer_cmd_rad', 'throttle', 'brake'):
        if key in lap:
            state[key] = float(lap[key][index])
    state['finished'] = bool(time >= lap['t'][-1])
    return state


def load_inputs(rl_csv, tum_csv, pp_csv):
    rl_rows, rl_meta, rl_prov = verified_input(rl_csv)
    tum_rows, tum_meta, tum_prov = verified_input(tum_csv)
    pp_rows, pp_meta, pp_prov = verified_input(pp_csv)
    validate_metadata(rl_meta, pp_meta, tum_meta)
    length = pp_meta['track']['raceline_length_m']
    path_length = pp_meta['track']['path_length_m']
    laps = [ocd_lap(rl_rows, rl_meta, 'rl', length, path_length), tum_lap(tum_rows, tum_meta),
            ocd_lap(pp_rows, pp_meta, 'pp', length, path_length)]
    return laps, [rl_meta, tum_meta, pp_meta], [rl_prov, tum_prov, pp_prov]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rl-csv', 'tum-csv', 'pp-csv', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--speedup', type=float, default=1.)
    parser.add_argument('--frames', type=int, help='Render first N video frames only')
    parser.add_argument('--screenshots-only', action='store_true', help='Skip MP4, save deterministic QA frames')
    parser.add_argument('--screenshot-times', type=float, nargs='+', default=[0., 20., 60., 103.])
    args = parser.parse_args()
    require(args.fps > 0 and np.isfinite(args.speedup) and args.speedup > 0, 'Invalid FPS / speedup')
    require(args.frames is None or args.frames > 0, 'Frame count must be positive')
    require(all(np.isfinite(t) and t >= 0 for t in args.screenshot_times), 'Invalid screenshot times')
    laps, metas, provenance = load_inputs(args.rl_csv, args.tum_csv, args.pp_csv)
    rl_meta, tum_meta, pp_meta = metas
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.animation import FFMpegWriter
    from ocd_lap.track import Track
    track = Track(rl_meta['saved_run_config']['env']['track'])
    footprint = np.array(rl_meta['conditions']['footprint_m'])
    brake_knm = rl_meta['conditions']['max_brake_torque_Nm'] / 1000
    step = rl_meta['checkpoint']['num_timesteps']
    eval_date = rl_meta['episode']['training_eval']['utc']
    obs = rl_meta['saved_run_config']['env']['obs']
    distances = obs['preview_horizon_m'] * (np.arange(1, obs['n_preview']+1)/obs['n_preview']) ** obs['preview_exponent']
    labels = ['p1_TUM RL / clean', 'TUM QSS / standard', 'TUM PP+PI / OCD diagnostic']
    colors = ['#58a6ff', '#ffad66', '#45d6a5']
    bg, panel, text_col, muted = '#0c1422', '#121f30', '#e8eef5', '#91a4bb'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': text_col,
                         'axes.labelcolor': muted, 'xtick.color': muted, 'ytick.color': muted,
                         'axes.edgecolor': '#34465f', 'axes.titlecolor': text_col})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=bg)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1], height_ratios=[1.12, 1],
                          left=.035, right=.975, bottom=.18, top=.69, hspace=.4, wspace=.16)
    ax_map = fig.add_subplot(gs[:, 0]); ax_close = fig.add_subplot(gs[0, 1]); ax_speed = fig.add_subplot(gs[1, 1])
    for ax in (ax_map, ax_close):
        ax.set_facecolor(panel)
        for edge in (track.left, track.right):
            e = np.vstack((edge, edge[0]))
            ax.plot(e[:, 0], e[:, 1], color='#708099', lw=.8, zorder=0)
        ax.plot(track.reference[:,1], track.reference[:,2], '--', color=colors[1], lw=.65, alpha=.6)
        ax.plot(track.path[:,1], track.path[:,2], color='#8995ad', lw=.7, alpha=.55)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax_map.set_title('Shanghai / actual recorded trajectories', fontsize=10, loc='left')
    ax_close.set_title('Follow p1_TUM / footprint to scale', fontsize=10, loc='left')
    ax_speed.set_facecolor(panel)
    markers, tails, close_bodies, cursors = [], [], [], []
    for index, (lap, color) in enumerate(zip(laps, colors)):
        ax_map.plot(lap['x_m'], lap['y_m'], color=color, lw=.65, alpha=.23)
        tails.append(ax_map.plot([], [], color=color, lw=1.7, alpha=.9)[0])
        markers.append(ax_map.plot([], [], 'o', ms=7, color=color, mec=bg, mew=1.2, zorder=5)[0])
        if index == 1:
            # TUM is point-mass QSS: no illustrative footprint masquerading as valid geometry.
            close_bodies.append(ax_close.plot([], [], 'D', ms=7, color=color, alpha=.8, zorder=5)[0])
        else:
            body = Polygon(np.zeros((4,2)), closed=True, facecolor=color, edgecolor=color, alpha=.8, zorder=6)
            ax_close.add_patch(body); close_bodies.append(body)
        ax_speed.plot(lap['distance_m']/1000, lap['speed_mps']*3.6, color=color, lw=.9, alpha=.7)
        cursors.append(ax_speed.plot([], [], 'o', color=color, ms=5, zorder=5)[0])
    previews = [ax.plot([], [], 'o', ms=sz, mfc='none', mec='#d6e2f1', mew=.7, zorder=4)[0]
                for ax,sz in ((ax_map,2.6),(ax_close,4.5))]
    ax_speed.set_xlim(0, max(l['distance_m'][-1] for l in laps)/1000)
    ax_speed.set_ylim(40, 310); ax_speed.grid(alpha=.15)
    ax_speed.set_xlabel('Projected raceline distance since line (km)', fontsize=8)
    ax_speed.set_ylabel('Speed (km/h)', fontsize=8); ax_speed.tick_params(labelsize=8)
    fig.text(.035,.96,'p1_TUM  /  RL and the original TUM lap',fontsize=19,weight='bold')
    fig.text(.035,.923,f'ERS + DRS OFF   |   OCD setup p1_TUM   |   {brake_knm:g} kN·m brakes   |   25 Hz commands',fontsize=10,color=muted)
    fig.text(.035,.894,f'Frozen RL best: {step:,} training steps  •  evaluation {eval_date}',fontsize=9,color=muted)
    clock = fig.text(.975,.956,'',ha='right',fontsize=13,family='monospace')
    hud = []
    for i,(label,color,lap) in enumerate(zip(labels,colors,laps)):
        y=.852-i*.043
        fig.text(.035,y,'●',color=color,fontsize=11)
        fig.text(.055,y,label,fontsize=10,color=color)
        hud.append(fig.text(.40,y,'',fontsize=10,family='monospace'))
    actions = fig.text(.035,.733,'',fontsize=9,family='monospace',color=colors[2])
    pp_on = 100*pp_meta['summary']['on_track_fraction_lap2']
    fig.text(.035,.106,'Reference: dashed orange = original TUM line; grey = RL observation reference; circles = '
             f'{obs["n_preview"]} preview points up to {obs["preview_horizon_m"]:g} m.',fontsize=8,color=muted)
    fig.text(.035,.078,f'PP uses 100% of TUM speeds: {pp_on:.1f}% intervals on track. Red marks a footprint violation '
             'within the current recorded interval.',fontsize=8,color='#ffc28a')
    fig.text(.035,.050,'PP is a nonterminating diagnostic, not a clean lap. TUM QSS uses a different vehicle model; '
             'its diamond is a point marker, not a checked footprint.',fontsize=8,color=muted)
    fig.text(.035,.023,'TUM provides trajectory + speed targets. PP+PI computes steering/throttle/brake for OCD; '
             'these are not TUM solver actuator commands.',fontsize=8,color=muted)

    def update(time):
        states = [state_at(lap,time) for lap in laps]
        xr,yr = states[0]['x_m'],states[0]['y_m']
        clock.set_text(f'{time:06.2f} s')
        for i,(lap,state,color) in enumerate(zip(laps,states,colors)):
            x,y = state['x_m'],state['y_m']
            off = i == 2 and state['on_track'] == 0 and not state['finished']
            current_color = '#ff657a' if off else color
            markers[i].set_data([x],[y]); markers[i].set_color(current_color)
            k = np.searchsorted(lap['t'],min(time,lap['t'][-1]),side='right')
            start = max(0,k-150)
            tails[i].set_data(np.r_[lap['x_m'][start:k],x],np.r_[lap['y_m'][start:k],y])
            cursors[i].set_data([state['distance_m']/1000],[state['speed_mps']*3.6])
            if i == 1:
                close_bodies[i].set_data([x],[y])
            else:
                c,s = np.cos(state['yaw_rad']),np.sin(state['yaw_rad'])
                close_bodies[i].set_xy(footprint @ np.array([[c,s],[-s,c]]) + [x,y])
                close_bodies[i].set_facecolor(current_color); close_bodies[i].set_edgecolor(current_color)
            status = 'FINISHED' if state['finished'] else ('OFF TRACK' if off else 'RUNNING')
            hud[i].set_text(f'{lap["t"][-1]:7.3f} s   {state["speed_mps"]*3.6:5.0f} km/h   {status}')
            hud[i].set_color(current_color)
        pp = states[2]
        actions.set_text(f'PP computed commands: steer {np.degrees(pp["steer_cmd_rad"]):+6.2f}°   '
                         f'throttle {100*pp["throttle"]:5.1f}%   brake {100*pp["brake"]:5.1f}%')
        # Unwrap saved path stations before interpolation, then wrap to the reference.
        path_station = float(np.interp(min(time,laps[0]['t'][-1]),laps[0]['t'],
                                      laps[0]['s_path_unwrapped_m']))
        stations = (path_station+distances) % track.length
        for preview in previews:
            preview.set_data(np.interp(stations,track.path[:,0],track.path[:,1]),
                             np.interp(stations,track.path[:,0],track.path[:,2]))
        ax_close.set_xlim(xr-48,xr+48); ax_close.set_ylim(yr-27,yr+27)

    args.out.parent.mkdir(parents=True,exist_ok=True)
    duration = max(lap['t'][-1] for lap in laps)+3.
    count = int(np.ceil(duration*args.fps/args.speedup))
    if args.frames is not None:
        count = min(count,args.frames)
    if not args.screenshots_only:
        writer = FFMpegWriter(fps=args.fps,codec='libx264',bitrate=3500,
                             extra_args=['-threads','1','-pix_fmt','yuv420p','-movflags','+faststart'])
        with writer.saving(fig,str(args.out),dpi=100):
            for frame in range(count):
                update(frame*args.speedup/args.fps)
                writer.grab_frame()
                if frame % (args.fps*10) == 0:
                    print(f'frame {frame}/{count}',flush=True)
    screenshots = []
    for time in args.screenshot_times:
        update(time)
        path = args.out.with_name(args.out.stem+f'_t{time:07.2f}.png')
        fig.savefig(path,dpi=100,facecolor=bg)
        screenshots.append(str(path))
    manifest = {'inputs':provenance,'lap_times_s':[float(l['t'][-1]) for l in laps],
                'rl_snapshot_step':step,'rl_evaluation_utc':eval_date,'rl_checkpoint_sha256':rl_meta['checkpoint']['sha256'],
                'setup':rl_meta['conditions']['setup'],'max_brake_torque_Nm':brake_knm*1000,
                'pp_valid_clean_lap':False,'pp_on_track_fraction':pp_on/100,
                'time_alignment':'lap 2 timing crossings, linear continuous pose interpolation; shared seconds clock',
                'PP_clock_precision':'start recovered from rounded CSV t_s minus t_lap_s; full-precision metadata duration',
                'discrete_commands':'zero-order interval hold; command at row applies during interval ending at row',
                'geometry':'original XY retained; projected distance used only by speed plot and HUD',
                'fps':args.fps,'speedup':args.speedup,'frames':count,'full_duration_sim_s':duration,
                'screenshots_only':args.screenshots_only,'screenshots':screenshots}
    args.out.with_suffix('.video.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plt.close(fig)
    print(json.dumps({'out':str(args.out),'frames':count,'lap_times_s':manifest['lap_times_s']}),flush=True)


if __name__ == '__main__':
    main()
