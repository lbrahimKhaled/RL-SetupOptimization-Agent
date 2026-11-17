#!/usr/bin/env python
"""Export the OCD training line (ocd_lap Track.path) as an AC reference file.

The AC border midpoint is rigidly aligned to the OCD border midpoint (rotation + translation,
no scale/mirror; scripts/ac_track_compare.py). The inverse map brings Track.path into AC
coordinates (rows of (AC z, AC x), as the collector uses them). For every AC border row i the
reference point is where that line crosses the cross-section left_i -> right_i, so reference,
left and right rows correspond exactly as driving.Geometry requires. The file is bound to the
exact AC borders by `ac_borders_sha256` (driving.ac_borders_sha256); the collector refuses it
for any other track data. Output is JSON only (numbers and strings).
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


def crossings(a, b, path):
    """Parameters u in [0,1] along a->b where the closed polyline `path` crosses it."""
    p, q = path, np.roll(path, -1, axis=0)
    r, s = b-a, q-p
    den = r[0]*s[:, 1]-r[1]*s[:, 0]
    ok = np.abs(den) > 1e-12
    w = p-a
    u = np.where(ok, (w[:, 0]*s[:, 1]-w[:, 1]*s[:, 0])/np.where(ok, den, 1), np.nan)
    v = np.where(ok, (w[:, 0]*r[1]-w[:, 1]*r[0])/np.where(ok, den, 1), np.nan)
    hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (v < 1)
    return u[hit]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ac_json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--track', default='Catalunya')
    parser.add_argument('--margin', type=float, default=.5, help='minimum distance to an AC border, m')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output must be a new file')
    info = read_track(args.ac_json)
    left, right = np.asarray(info['left_lane'], float), np.asarray(info['right_lane'], float)

    from ocd_lap.track import Track
    track = Track(args.track)
    ocd_mid = (open_loop(track.left)+open_loop(track.right))/2
    ac_mid, _ = resample((left+right)/2, 2.)
    rot, _, trans, rms = align(ac_mid, Polyline(ocd_mid))
    ocd_path = open_loop(track.path[:, 1:3])
    path_ac = (rot.T@(ocd_path-trans).T).T  # inverse rigid map: OCD -> AC

    rows, fraction = [], []
    for a, b in zip(left, right):
        u = crossings(a, b, path_ac)
        if len(u) == 0:
            raise ValueError('OCD training line leaves the AC track at a border row; widen or inspect')
        u = u[np.argmin(np.abs(u-.5))] if len(u) > 1 else u[0]
        width = np.linalg.norm(b-a)
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
    report = dict(
        schema='rlsetup.ac.ocd-reference.v1', reference='ocd_path',
        ac_borders_sha256=ac_borders_sha256(left, right), rows=rows.tolist(),
        source=dict(ocd_track=args.track, path='ocd_lap.track.Track.path (training reference)',
                    ocd_path_sha256=hashlib.sha256(np.ascontiguousarray(ocd_path).tobytes()).hexdigest(),
                    ocd_path_length_m=float(Polyline(ocd_path).length),
                    ac_track_file_sha256=hashlib.sha256(args.ac_json.read_bytes()).hexdigest()),
        alignment=dict(method='rigid ICP of AC border midpoint onto OCD border midpoint',
                       rotation_ac_to_ocd=rot.tolist(), translation_ac_to_ocd_m=trans.tolist(),
                       midpoint_rms_m=rms),
        diagnostics=dict(reference_length_m=geometry.length,
                         rows_vs_transformed_line_max_m=float(np.abs(e).max()),
                         min_border_distance_m=float(near.min()), clipped_rows=clipped,
                         lateral_fraction_min=float(fraction.min()), lateral_fraction_max=float(fraction.max()),
                         median_ac_width_m=float(np.median(width))))
    args.output.write_text(json.dumps(report, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('ac_borders_sha256', 'source', 'alignment', 'diagnostics')}, indent=2))


if __name__ == '__main__':
    main()
