#!/usr/bin/env python
"""Export the OCD training line (ocd_lap Track.path) as an AC reference file.

The AC border midpoint is rigidly aligned to the OCD border midpoint (rotation + translation,
no scale/mirror; scripts/ac_track_compare.py). The inverse map brings Track.path into AC
coordinates (rows of (AC z, AC x), as the collector uses them). For every AC border row i the
reference point is where that line crosses the cross-section left_i -> right_i, so reference,
left and right rows correspond exactly as driving.Geometry requires. The file is bound to the
exact AC borders by `ac_borders_sha256` (driving.ac_borders_sha256); the collector refuses it
for any other track data. Output is JSON only (numbers and strings).

--source-run <OCD seed dir> (schema v2): the line is that run's own xi_ref (env.reference: Track.path
or a 'csv:' line), and each row also carries the OCD quantities the AC worker needs without OCD:
s_path_m (Track.path arc length of the row, unwrapped), v_qss_mps (R4 corner-table speed),
v_start_mps (the env QSS profile that sets OCD's start speed), corner (1 inside an R4 corner
window), plus the run's settle/stall settings and car geometry. The QSS profile and corner table
are the training's own files in data/, found and verified by the keys in the run's provenance.json.
The file is bound to the run's config.json by SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from ac_track_compare import Polyline, align, open_loop, resample  # noqa: E402
from rlsetup.assetto.driving import Geometry, ac_borders_sha256  # noqa: E402


def read_track(path):
    data = Path(path).read_bytes()
    text = data.decode('utf-16') if data[:2] in (b'\xff\xfe', b'\xfe\xff') else data.decode('utf-8-sig')
    raw = json.loads(text)
    return raw['queries']['get_track_info']['data'] if 'queries' in raw else raw


def crossings(a, b, path, *, extended=False):
    """Parameters u in [0,1] along a->b where the closed polyline `path` crosses it
    (extended=True: crossings of the infinite line through a, b, any u)."""
    p, q = path, np.roll(path, -1, axis=0)
    r, s = b-a, q-p
    den = r[0]*s[:, 1]-r[1]*s[:, 0]
    ok = np.abs(den) > 1e-12
    w = p-a
    u = np.where(ok, (w[:, 0]*s[:, 1]-w[:, 1]*s[:, 0])/np.where(ok, den, 1), np.nan)
    v = np.where(ok, (w[:, 0]*r[1]-w[:, 1]*r[0])/np.where(ok, den, 1), np.nan)
    hit = ok & (v >= 0) & (v < 1) & (np.isfinite(u) if extended else (u >= 0) & (u <= 1))
    return u[hit]


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_profiles(run, env):
    """The run's own QSS profile and corner table from data/, verified against provenance.json."""
    from rlsetup.driving import corners, qss_profile
    provenance = json.loads((run/'provenance.json').read_text())
    want = provenance['qss']['cache_key_sha256']
    qss = None
    for path in sorted((ROOT/'data').glob('qss_profile_*.npz')):
        candidate = qss_profile.load(path)
        if hashlib.sha256(json.dumps(candidate.meta['key'], sort_keys=True).encode()).hexdigest() == want:
            qss, qss_file = candidate, path
            break
    if qss is None:
        raise SystemExit(f'no data/qss_profile_*.npz with the training cache key {want[:12]}')
    key = (provenance.get('r4') or {}).get('corner_table_key') or \
        ((provenance.get('r2') or {}).get('corner_table') or {}).get('key')
    table = table_file = None
    if key:
        table_file = ROOT/'data'/f'corner_table_{key[:12]}.json'
        table = corners.load(table_file)
        if table.meta.get('key') != key:
            raise SystemExit(f'{table_file.name} does not carry the training corner-table key')
    reward = env['reward']
    if table is None and (reward.get('speed_k', 0) > 0 or reward.get('line_k', 0) > 0):
        raise SystemExit('the run uses R4 terms but its provenance names no corner table')
    return qss, qss_file, table, table_file


def settle_block(env):
    """Settle / stall settings of the source env (rlsetup/driving/config.py EnvConfig, DemoConfig)."""
    from ocd_lap import cars
    from rlsetup.driving.demo_driver import steer_lock_rad
    spec = cars.get(env['car'])
    demo = env['demo']
    block = dict(rear_axle_m=float(spec.rear_axle_m), wheelbase_m=float(spec.wheelbase_m),
                 steer_lock_rad=steer_lock_rad(spec),
                 **{k: demo[k] for k in ('lookahead_m', 'lookahead_s', 'speed_kp', 'speed_ki', 'integral_limit_m',
                                         'brake_cap', 'traction_control', 'traction_slip_start',
                                         'traction_slip_span', 'released_pedal_action')},
                 **{k: env[k] for k in ('settle_s', 'eval_speed_fraction', 'settle_window_margin_m',
                                        'stall_speed_mps', 'stall_time_s', 'wrong_way_rad')},
                 start_speed_fraction=list(env['start_speed_fraction']))
    if block['settle_s'] <= 0:
        raise SystemExit('source run has no settle phase (settle_s = 0)')
    return block


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ac_json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--track', default='Catalunya')
    parser.add_argument('--margin', type=float, default=.5, help='minimum distance to an AC border, m')
    parser.add_argument('--max-outside', type=float, default=.5,
                        help='largest distance the line may pass outside an AC border before it is pulled in, m')
    parser.add_argument('--source-run', type=Path,
                        help='OCD seed dir (config.json, provenance.json) of the source actor: v2 file')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output must be a new file')
    info = read_track(args.ac_json)
    left, right = np.asarray(info['left_lane'], float), np.asarray(info['right_lane'], float)

    from ocd_lap.track import Track
    env = None
    if args.source_run is not None:
        config_bytes = (args.source_run/'config.json').read_bytes()
        env = json.loads(config_bytes)['env']
        if env['track'] != args.track:
            parser.error(f"source run track {env['track']} != --track {args.track}")
    track = Track(args.track)
    ocd_mid = (open_loop(track.left)+open_loop(track.right))/2
    ac_mid, _ = resample((left+right)/2, 2.)
    rot, _, trans, rms = align(ac_mid, Polyline(ocd_mid))
    if env is None:
        ocd_path = open_loop(track.path[:, 1:3])
        line_name = 'ocd_lap.track.Track.path (training reference)'
    else:
        from rlsetup.driving.reference import ReferenceLine
        line = ReferenceLine.for_track(track, env['reference'])
        ocd_path = line.point(np.arange(0., line.length, 1.))
        line_name = f"source run xi_ref {env['reference']}"
    path_ac = (rot.T@(ocd_path-trans).T).T  # inverse rigid map: OCD -> AC

    rows, fraction, outside = [], [], []
    for a, b in zip(left, right):
        u = crossings(a, b, path_ac)
        width = np.linalg.norm(b-a)
        if len(u) == 0:
            # Where AC and OCD edges differ, the line can pass just outside the AC edge: take its
            # crossing of the extended cross-section and pull it inside (margin), if it is close.
            u = crossings(a, b, path_ac, extended=True)
            if len(u) == 0 or np.min(np.maximum(-u, u-1))*width > args.max_outside:
                raise ValueError('OCD training line leaves the AC track at a border row; widen or inspect')
            outside.append(float(np.min(np.maximum(-u, u-1))*width))
        u = u[np.argmin(np.abs(u-.5))] if len(u) > 1 else u[0]
        u = float(np.clip(u, args.margin/width, 1-args.margin/width))
        rows.append(a+u*(b-a))
        fraction.append(u)
    rows = np.asarray(rows)
    fraction = np.asarray(fraction)
    clipped = int(np.sum((fraction <= args.margin/np.linalg.norm(right-left, axis=1)+1e-12)
                         | (fraction >= 1-args.margin/np.linalg.norm(right-left, axis=1)-1e-12)))
    track_info = dict(left_lane=left.tolist(), right_lane=right.tolist(), ocd_path=rows.tolist())
    geometry = Geometry(track_info, reference='ocd_path')  # same validation as the collector
    _, _, e, _ = Polyline(path_ac).project(geometry.ref)
    width = np.linalg.norm(geometry.left-geometry.right, axis=1)
    near = np.minimum(np.linalg.norm(geometry.ref-geometry.left, axis=1),
                      np.linalg.norm(geometry.ref-geometry.right, axis=1))
    extra = {}
    if env is not None:
        qss, qss_file, table, table_file = training_profiles(args.source_run, env)
        ocd_rows = rows@rot.T+trans  # AC -> OCD
        s_raw, hint = np.empty(len(rows)), None
        for i, xy in enumerate(ocd_rows):
            hint = s_raw[i] = float(track.match(xy, hint)[0])
        s_path = np.unwrap(s_raw, period=float(track.length))
        if np.any(np.diff(s_path) <= 0):
            raise ValueError('AC rows do not map to increasing OCD Track.path stations')
        speed = table.speed if table is not None else qss.speed
        corner = (np.array([table.unit_at(s) is not None for s in s_path], dtype=float) if table is not None
                  else np.zeros(len(rows)))
        extra = dict(
            profile=dict(s_path_m=s_path.tolist(), v_qss_mps=np.asarray(speed(s_path), float).tolist(),
                         v_start_mps=np.asarray(qss.speed(s_path), float).tolist(), corner=corner.tolist(),
                         path_length_m=float(track.length),
                         v_qss_source=('R4 corner table (QSS on xi_ref, Track.path stations)' if table is not None
                                       else 'env QSS profile on Track.path'),
                         corner_share=float(corner.mean())),
            settle=settle_block(env))
        source_binding = dict(run=str(args.source_run), source_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                              reference=env['reference'], car=env['car'], setup=env['setup'],
                              qss_file=qss_file.name, qss_file_sha256=_sha_file(qss_file),
                              corner_table_file=table_file.name if table_file else None,
                              corner_table_sha256=_sha_file(table_file) if table_file else None)
    report = dict(
        schema='rlsetup.ac.ocd-reference.v2' if env is not None else 'rlsetup.ac.ocd-reference.v1',
        reference='ocd_path',
        ac_borders_sha256=ac_borders_sha256(left, right), rows=rows.tolist(), **extra,
        source=dict(ocd_track=args.track, path=line_name, **(source_binding if env is not None else {}),
                    ocd_path_sha256=hashlib.sha256(np.ascontiguousarray(ocd_path).tobytes()).hexdigest(),
                    ocd_path_length_m=float(Polyline(ocd_path).length),
                    ac_track_file_sha256=hashlib.sha256(args.ac_json.read_bytes()).hexdigest()),
        alignment=dict(method='rigid ICP of AC border midpoint onto OCD border midpoint',
                       rotation_ac_to_ocd=rot.tolist(), translation_ac_to_ocd_m=trans.tolist(),
                       midpoint_rms_m=rms),
        diagnostics=dict(reference_length_m=geometry.length,
                         rows_vs_transformed_line_max_m=float(np.abs(e).max()),
                         min_border_distance_m=float(near.min()), clipped_rows=clipped,
                         rows_line_outside_ac=len(outside),
                         max_line_outside_ac_m=max(outside) if outside else 0.,
                         lateral_fraction_min=float(fraction.min()), lateral_fraction_max=float(fraction.max()),
                         median_ac_width_m=float(np.median(width))))
    args.output.write_text(json.dumps(report, allow_nan=False))
    summary = {k: report[k] for k in ('ac_borders_sha256', 'source', 'alignment', 'diagnostics')}
    if env is not None:
        p = report['profile']
        summary.update(settle=report['settle'], profile=dict(
            corner_share=p['corner_share'], v_qss_range_mps=[min(p['v_qss_mps']), max(p['v_qss_mps'])],
            s_path_span_m=p['s_path_m'][-1]-p['s_path_m'][0], path_length_m=p['path_length_m']))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
