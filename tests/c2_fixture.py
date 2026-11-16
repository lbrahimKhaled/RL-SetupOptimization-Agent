"""Synthetic f1laps export + track geometry for the C2a tests (no real data, no real usernames).

Tracks: 'trk_good' (geometry 'Good' matches the laps), 'trk_long' (geometry 'Long' = the same shape
scaled by 0.97, so the length test of R2 fails), 'trk_none' (no geometry file). Each lap is driven
on the full-size shape of its track; steering = -gain * kappa (game convention: + = right turn) plus
noise; the game lap starts START_OFFSET_M after the geometry's index 0, so the gate must find
shift = +START_OFFSET_M.

Layout written by `write_fixture(root)`:
  root/exp_f1_25/data/   F1 25 export folder (laps.csv without a game column, like the real F1 25 export)
  root/bundle.zip        F1_26/data/ (full F1 26 export) + F1_25/data/setups.csv (byte copy: skipped)
  root/extra_setups.csv  bare setups CSV: 2 F1 25 setup_ids already seen + 1 new one
  root/geometry/extra/   Good.csv, Long.csv (TUM format) ; root/geometry/tum/ empty TUM database
"""
from pathlib import Path
import csv
import io
import zipfile
import numpy as np
from rlsetup.critic.sources.f1_game_map import GAME_COLUMNS
from rlsetup.style import Track as StyleTrack

USERS_25 = ('drv_alpha', 'drv_beta', 'drv_gamma')
USERS_26 = ('drv_alpha', 'drv_delta', 'drv_eps')
TRACKS = ('trk_good', 'trk_long', 'trk_none')
TRACK_MAP = {'trk_good': 'Good', 'trk_long': 'Long', 'trk_none': 'NoSuchTrack'}
START_OFFSET_M = 30.
HALF_WIDTH_M = 6.
STEER_GAIN = {'drv_alpha': 35., 'drv_beta': 45., 'drv_gamma': 40., 'drv_delta': 50., 'drv_eps': 40.}


def shape_xy(scale=1., n=4000, lobes=3, phase=0.):
    """Closed star-shaped centre line [m], counter-clockwise, ~4.3 km at scale 1."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 650. * scale * (1 + .25 * np.cos(lobes * th + phase) + .08 * np.sin(5 * th))
    return np.column_stack((r * np.cos(th), r * np.sin(th)))


def resample(xy, step):
    closed = np.vstack((xy, xy[:1]))
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    n = int(round(s[-1] / step))
    g = np.arange(n) * s[-1] / n
    return np.column_stack((np.interp(g, s, closed[:, 0]), np.interp(g, s, closed[:, 1])))


def write_tum(path, xy, half_width=HALF_WIDTH_M):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('# x_m,y_m,w_tr_right_m,w_tr_left_m\n')
        for x, y in xy:
            f.write(f'{x:.4f},{y:.4f},{half_width:.4f},{half_width:.4f}\n')


def truth_track(scale=1.):
    """style-rl Track of the full-size shape (the 'real' circuit the laps are driven on)."""
    xy = resample(shape_xy(scale), 5.)
    n = np.column_stack((-np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return StyleTrack(xy, xy + HALF_WIDTH_M * n, xy - HALF_WIDTH_M * n)


def make_lap(track, user, rng, spacing=3., time_scale=1., noise=.01):
    """dict of the lap's raw tables: distance-sampled channels, time keys [ms], lap time [s]."""
    L = track.length_m
    d = np.arange(1., L - 1., spacing)
    d = np.sort(np.clip(d + rng.uniform(-.4, .4, len(d)), 0., L - 1e-3))
    s = (d + START_OFFSET_M) % L                        # progress on the geometry
    kap = np.interp(s, track.s, track.kappa, period=L)
    steer = np.clip(-STEER_GAIN[user] * kap + noise * rng.standard_normal(len(d)), -1, 1)
    # speed: grip-limited from curvature, then smoothed; time by integration
    grid = np.arange(0., L + 1., 1.)
    kg = np.abs(np.interp((grid + START_OFFSET_M) % L, track.s, track.kappa, period=L))
    v = np.clip(np.sqrt(30. / np.maximum(kg, 1e-6)), 25., 85.)
    v = np.convolve(np.r_[v[-60:], v, v[:60]], np.ones(121) / 121, mode='valid')
    t = np.r_[0., np.cumsum(1. / v[:-1])]               # [s] at grid metres
    dv = np.gradient(v)
    brake = np.clip(-dv * (25. if user != 'drv_beta' else 40.), 0, 1)
    throttle = np.clip(.4 + 15. * dv, 0, 1) * (brake < .05)
    keys = np.arange(20., L - 5., 10.)
    return {'d': d, 'steer': steer, 'throttle': np.interp(d, grid, throttle), 'brake': np.interp(d, grid, brake),
            'keys': keys, 'key_ms': np.interp(keys, grid, t) * 1000. * time_scale, 'lap_s': float(np.interp(L, grid, t))}


def fmt_lap(sec):
    return f'{int(sec // 60)}:{sec % 60:06.3f}'


def _csv(rows, header):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode('utf-8-sig')


def build_export(game, users, rng, lap_tracks):
    """{file name: bytes} of one game's export. `lap_tracks`: {track: style Track} for telemetry laps."""
    setups, laps, smap, tmap = [], [], [], []
    graphs = {c: [] for c in ('throttle', 'brake', 'steer')}
    base = {'trk_good': 80., 'trk_long': 82., 'trk_none': 75.}
    tag = game[-2:]
    for t in TRACKS:
        for ui, u in enumerate(users):
            for k in range(4):
                sid = f'{tag}-{t}-{u}-{k}'
                vals = {'front_wing': 20 + k, 'rear_wing': 18 + ui, 'differential_on_throttle_pct': 50 + 5 * k,
                        'differential_off_throttle_pct': 55, 'front_camber_deg': -3.0, 'rear_camber_deg': -1.5,
                        'front_toe_deg': .05, 'rear_toe_deg': .2, 'front_suspension': 20 + k, 'rear_suspension': 18,
                        'front_anti_roll_bar': 10 + ui, 'rear_anti_roll_bar': 9, 'front_ride_height': 22,
                        'rear_ride_height': 45 + k, 'front_brake_bias_pct': 56}
                lap = base[t] * (1.05 if game == 'F1 26' else 1.) * (1 + .004 * k + .01 * ui) \
                    * (1.10 if k == 3 and u == users[0] else 1.)      # F1 26 car 5 % slower: gap is per game
                row = {'game': game, 'track': t, 'team': 'Ferrari', 'setup_id': sid, 'user': u,
                       'session': ['Race', 'Time Trial', 'Qualifying 3', 'Time Trial'][k],
                       'session_group': ['Race', 'Time Trial', 'Qualifying', 'Time Trial'][k], 'conditions': 'Dry',
                       'date_key': f'2026-0{1 + k}-1{ui}', 'controller': ['Wheel', '', 'Controller'][ui % 3],
                       'lap_time': fmt_lap(lap), **{c: str(vals[c]) for c in GAME_COLUMNS}}
                if game == 'F1 26':
                    row.update(traction_control=['Off', 'Full', 'Medium'][ui % 3], anti_lock_brakes=['Off', 'On', '151'][ui % 3])
                setups.append(row)
            if t in lap_tracks:
                lid = f'lap-{tag}-{t}-{u}'
                bad = {'drv_gamma': {'spacing': 12.}, 'drv_eps': {'time_scale': 1.05}}.get(u, {})
                if bad and t != 'trk_good':
                    continue
                lp = make_lap(lap_tracks[t], u, rng, **bad)
                lr = {'lap_id': lid, 'track': t, 'user': u, 'team': 'Ferrari', 'session': 'Time Trial',
                      'conditions': 'Dry', 'lap_time': fmt_lap(lp['lap_s']), 'controller': ['Wheel', '', 'Controller'][ui % 3],
                      'date_key': '2026-05-01'}
                if game == 'F1 26':
                    lr.update(game=game, traction_control='Off', anti_lock_brakes='Off')
                laps.append(lr)
                smap.append({'setup_id': f'{tag}-{t}-{u}-1', 'lap_id': lid, 'has_telemetry': 'true'})
                for i, dist in enumerate(lp['d']):
                    for c, unit in (('throttle', 'fraction'), ('brake', 'fraction'), ('steer', 'source_normalized')):
                        graphs[c].append({'lap_id': lid, 'sample_index': i, 'distance_m': f'{dist:.1f}',
                                          'source_value': f'{lp[c][i]:.4f}', 'unit': unit})
                tmap += [{'lap_id': lid, 'source_distance_key': f'{k:.0f}', 'source_time_value': f'{ms:.1f}',
                          'time_unit': 'not_declared_by_endpoint'} for k, ms in zip(lp['keys'], lp['key_ms'])]
    setup_cols = list(setups[0]) if game == 'F1 25' else list(dict.fromkeys(k for r in setups for k in r))
    lap_cols = list(dict.fromkeys(k for r in laps for k in r))
    out = {'setups.csv': _csv(setups, setup_cols), 'laps.csv': _csv(laps, lap_cols),
           'setup_lap_map.csv': _csv(smap, ['setup_id', 'lap_id', 'has_telemetry']),
           'time_mapping.csv': _csv(tmap, ['lap_id', 'source_distance_key', 'source_time_value', 'time_unit'])}
    for c, rows in graphs.items():
        out[f'graph_{c}.csv'] = _csv(rows, ['lap_id', 'sample_index', 'distance_m', 'source_value', 'unit'])
    return out


def write_fixture(root: Path, seed: int = 0) -> dict:
    """Write the fixture under `root`; returns paths and facts the tests use."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    full = truth_track(1.)
    lap_tracks = {t: full for t in TRACKS}              # every lap is driven on the full-size shape
    write_tum(root / 'geometry/extra/Good.csv', resample(shape_xy(1.), 5.))
    write_tum(root / 'geometry/extra/Long.csv', resample(shape_xy(.97), 5.))
    (root / 'geometry/tum/tracks').mkdir(parents=True, exist_ok=True)
    (root / 'geometry/tum/racelines').mkdir(parents=True, exist_ok=True)
    e25 = build_export('F1 25', USERS_25, rng, lap_tracks)
    e26 = build_export('F1 26', USERS_26, rng, lap_tracks)
    folder = root / 'exp_f1_25/data'
    folder.mkdir(parents=True)
    for name, data in e25.items():
        (folder / name).write_bytes(data)
    with zipfile.ZipFile(root / 'bundle.zip', 'w') as z:
        for name, data in e26.items():
            z.writestr(f'F1_26/data/{name}', data)
        z.writestr('F1_25/data/setups.csv', e25['setups.csv'])
    rows = list(csv.DictReader(io.StringIO(e25['setups.csv'].decode('utf-8-sig'))))
    extra = rows[:2] + [dict(rows[0], setup_id='25-extra-new', lap_time=rows[0]['lap_time'])]
    (root / 'extra_setups.csv').write_bytes(_csv(extra, list(rows[0])))
    return {'root': root, 'sources': [root / 'exp_f1_25', root / 'bundle.zip', root / 'extra_setups.csv'],
            'n_setups_25': len(rows) + 1, 'n_setups_26': len(list(csv.DictReader(io.StringIO(e26['setups.csv'].decode('utf-8-sig'))))),
            'extra_dir': root / 'geometry/extra', 'tum_dir': root / 'geometry/tum', 'users': set(USERS_25) | set(USERS_26)}
