#!/usr/bin/env python
"""Compare AC ks_barcelona/layout_gp geometry with the OCD training track (TUM Catalunya).

Input: the JSON written by `python -m rlsetup.assetto inspect --include-track` (or a bare
get_track_info dictionary). AC arrays are used exactly as the collector uses them
(rows of (AC z, AC x), driving.Geometry). OCD geometry comes from ocd_lap.track.Track.

Method (all metres / radians / 1/m):
 1. Closed polylines resampled every `--step` m of arc length.
 2. AC border midpoint is rigidly aligned (rotation + translation, no scale, no mirror) to the
    OCD border midpoint by point-to-polyline ICP, started from a coarse grid of rotations and
    arc-length offsets. The same fit with a mirror image and with a free scale are reported
    only as diagnostics (a better mirror fit would mean a handedness error; scale shows a
    length mismatch).
 3. After alignment every AC sample is projected onto the OCD centreline: lateral offset,
    station, total width (AC |left-right| vs OCD w_left+w_right) and curvature are compared
    as functions of OCD station. Curvature: heading derivative of the resampled line,
    smoothed with a centred `--smooth` m window.
 4. The same projection is applied to the AC fast_lane vs OCD Track.path (training reference)
    and Track.reference (TUM raceline).
Outputs: <out>/report.json, overlay.png, profiles.png. Nothing is sent anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def open_loop(xy):
    xy = np.asarray(xy, float)
    if np.linalg.norm(xy[0]-xy[-1]) < 1e-6:
        xy = xy[:-1]
    return xy


def resample(xy, step):
    xy = open_loop(xy)
    closed = np.vstack([xy, xy[:1]])
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    n = max(16, int(round(s[-1]/step)))
    t = np.linspace(0., s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])]), float(s[-1])


def heading(xy):
    d = np.roll(xy, -1, axis=0)-np.roll(xy, 1, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def curvature(xy, smooth_m, step):
    psi = np.unwrap(heading(xy))
    ds = np.linalg.norm(np.roll(xy, -1, axis=0)-np.roll(xy, 1, axis=0), axis=1)
    dpsi = np.roll(psi, -1)-np.roll(psi, 1)
    dpsi = (dpsi+np.pi) % (2*np.pi)-np.pi
    kappa = dpsi/ds
    w = max(1, int(round(smooth_m/step)))
    kernel = np.ones(2*w+1)/(2*w+1)
    padded = np.r_[kappa[-w:], kappa, kappa[:w]]
    return np.convolve(padded, kernel, mode='valid')


class Polyline:
    """Closed polyline with exact point-to-segment projection (station, signed lateral, index)."""
    def __init__(self, xy):
        self.a = open_loop(xy)
        self.b = np.roll(self.a, -1, axis=0)
        self.d = self.b-self.a
        self.l2 = (self.d**2).sum(1)
        self.s0 = np.r_[0., np.cumsum(np.sqrt(self.l2))[:-1]]
        self.length = float(np.sqrt(self.l2).sum())

    def project(self, p, chunk=512):
        out_q, out_s, out_e, out_i = [], [], [], []
        for k in range(0, len(p), chunk):
            q = p[k:k+chunk, None, :]
            t = np.clip(((q-self.a)*self.d).sum(-1)/self.l2, 0, 1)
            c = self.a+t[..., None]*self.d
            dist2 = ((q-c)**2).sum(-1)
            i = dist2.argmin(1)
            r = np.arange(len(i))
            cc = c[r, i]
            cross = self.d[i, 0]*(q[r, 0, 1]-self.a[i, 1])-self.d[i, 1]*(q[r, 0, 0]-self.a[i, 0])
            out_q.append(cc)
            out_s.append(self.s0[i]+t[r, i]*np.sqrt(self.l2[i]))
            out_e.append(np.sign(cross)*np.sqrt(dist2[r, i]))  # + = left of OCD direction
            out_i.append(i)
        return np.vstack(out_q), np.concatenate(out_s), np.concatenate(out_e), np.concatenate(out_i)


def fit_rigid(src, dst, scale=False):
    """Least-squares similarity/rigid map src->dst (Umeyama), proper rotation only."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src-mu_s, dst-mu_d
    u, sig, vt = np.linalg.svd(b.T@a/len(src))
    sgn = np.diag([1., np.sign(np.linalg.det(u@vt))])
    rot = u@sgn@vt
    c = float((sig*np.diag(sgn)).sum()/((a**2).sum()/len(src))) if scale else 1.
    return rot, c, mu_d-c*rot@mu_s


def icp(src, target, rot, c, trans, scale=False, iterations=60):
    for _ in range(iterations):
        moved = (c*(rot@src.T)).T+trans
        nearest, *_ = target.project(moved)
        rot, c, trans = fit_rigid(src, nearest, scale)
    moved = (c*(rot@src.T)).T+trans
    _, _, e, _ = target.project(moved)
    return rot, c, trans, float(np.sqrt((e**2).mean()))


def align(src, target, scale=False):
    """Coarse search over rotation and centroid shift, then ICP; best RMS wins."""
    best = None
    tc = target.a.mean(0)
    for angle in np.deg2rad(np.arange(0, 360, 15)):
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        trans = tc-rot@src.mean(0)
        result = icp(src[::4], target, rot, 1., trans, False, 25)
        if best is None or result[3] < best[3]:
            best = result
    return icp(src, target, *best[:3], scale=scale)


def wrap(a):
    return (a+np.pi) % (2*np.pi)-np.pi


def compare_lines(ac_xy, ocd_xy, transform, step):
    rot, c, trans = transform
    moved = (c*(rot@ac_xy.T)).T+trans
    target = Polyline(ocd_xy)
    _, s, e, _ = target.project(moved)
    return dict(ac_length_m=float(Polyline(moved).length), ocd_length_m=target.length,
                lateral_rms_m=float(np.sqrt((e**2).mean())), lateral_p95_m=float(np.percentile(abs(e), 95)),
                lateral_max_m=float(abs(e).max()), max_at_ocd_station_m=float(s[abs(e).argmax()])), (s, e)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ac_json', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--track', default='Catalunya')
    parser.add_argument('--step', type=float, default=2.)
    parser.add_argument('--smooth', type=float, default=10.)
    args = parser.parse_args(argv)
    data = args.ac_json.read_bytes()
    # PowerShell 5 '>' writes UTF-16 with a BOM; cmd.exe writes plain bytes.
    text = data.decode('utf-16') if data[:2] in (b'\xff\xfe', b'\xfe\xff') else data.decode('utf-8-sig')
    raw = json.loads(text)
    info = raw['queries']['get_track_info']['data'] if 'queries' in raw else raw
    left, right = np.asarray(info['left_lane'], float), np.asarray(info['right_lane'], float)
    fast = np.asarray(info['fast_lane'], float) if info.get('fast_lane') is not None else None
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 2:
        raise ValueError('need corresponding Nx2 left/right lanes')
    extra_columns = left.shape[1] > 2
    left2, right2 = left[:, :2], right[:, :2]

    from ocd_lap.track import Track
    track = Track(args.track)
    # Like-for-like with AC: midpoint of the OCD borders (TUM centre is off-centre where w_l != w_r).
    ocd_center = (open_loop(track.left)+open_loop(track.right))/2
    center_line = Polyline(ocd_center)
    ocd_width = np.linalg.norm(open_loop(track.left)-open_loop(track.right), axis=1)

    ac_mid, ac_len = resample((left2+right2)/2, args.step)
    ocd_res, _ = resample(ocd_center, args.step)
    # Same driving direction is required; reversed AC order is reported, not silently fixed.
    rot, c, trans, rms = align(ac_mid, center_line)
    moved = (rot@ac_mid.T).T+trans
    _, s_mid, e_mid, _ = center_line.project(moved)
    order = np.diff(np.unwrap(s_mid/center_line.length*2*np.pi))
    same_direction = bool(np.median(order) > 0)
    mirror = ac_mid*np.array([1., -1.])
    *_, rms_mirror = align(mirror, center_line)
    _, c_scale, _, rms_scale = align(ac_mid, center_line, scale=True)

    # Widths: AC border distance at each AC row, mapped to OCD station; OCD width at its station.
    ac_width_rows = np.linalg.norm(left2-right2, axis=1)
    ac_mid_rows = (rot@((left2+right2)/2).T).T+trans
    _, s_rows, _, _ = center_line.project(ac_mid_rows)
    ocd_s_rows = center_line.s0
    ocd_width_at = np.interp(s_rows, np.r_[ocd_s_rows, center_line.length],
                             np.r_[ocd_width, ocd_width[:1]], period=center_line.length)
    dw = ac_width_rows-ocd_width_at

    # Curvature profiles on a common OCD-station grid (sign: + = left turn).
    k_ac = curvature(ac_mid, args.smooth, args.step)
    _, s_ac, _, _ = center_line.project(moved)
    k_ocd = curvature(ocd_res, args.smooth, args.step)
    _, s_ocd, _, _ = center_line.project(ocd_res)
    grid = np.arange(0, center_line.length, args.step)
    def on_grid(s, k):
        o = np.argsort(s)
        return np.interp(grid, s[o], k[o], period=center_line.length)
    kg_ac, kg_ocd = on_grid(s_ac, k_ac), on_grid(s_ocd, k_ocd)
    corr = float(np.corrcoef(kg_ac, kg_ocd)[0, 1])
    # Corners: OCD |kappa| maxima tighter than 200 m radius, at least 80 m apart.
    corners = []
    mag = abs(kg_ocd)
    for i in np.argsort(-mag):
        if mag[i] < 1/200:
            break
        if all(abs(wrap((grid[i]-g['ocd_station_m'])/center_line.length*2*np.pi))*center_line.length/(2*np.pi) > 80
               for g in corners):
            w = max(1, int(round(30/args.step)))
            idx = np.arange(i-w, i+w+1) % len(grid)
            j = idx[np.argmax(abs(kg_ac[idx]))]
            corners.append(dict(ocd_station_m=float(grid[i]), ocd_min_radius_m=float(1/mag[i]),
                                ac_min_radius_m=float(1/max(abs(kg_ac[j]), 1e-9)),
                                ac_peak_offset_m=float(grid[j]-grid[i]),
                                same_turn_direction=bool(np.sign(kg_ac[j]) == np.sign(kg_ocd[i]))))
    corners.sort(key=lambda g: g['ocd_station_m'])

    report = dict(
        inputs=dict(ac_file=str(args.ac_json), ocd_track=args.track, step_m=args.step, smooth_m=args.smooth,
                    ac_rows=int(len(left)), ac_extra_columns=extra_columns,
                    ac_has_fast_lane=fast is not None, ac_info_keys=sorted(info)),
        alignment=dict(rotation_deg=float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
                       translation_m=trans.tolist(), same_driving_direction=same_direction,
                       centerline_rms_m=rms, mirrored_rms_m=float(rms_mirror),
                       free_scale=float(c_scale), free_scale_rms_m=float(rms_scale)),
        centerline=dict(ac_midpoint_length_m=ac_len, ocd_centerline_length_m=center_line.length,
                        length_difference_m=ac_len-center_line.length,
                        lateral_rms_m=float(np.sqrt((e_mid**2).mean())),
                        lateral_p95_m=float(np.percentile(abs(e_mid), 95)),
                        lateral_max_m=float(abs(e_mid).max()),
                        lateral_max_at_ocd_station_m=float(s_mid[abs(e_mid).argmax()])),
        width=dict(ac_median_m=float(np.median(ac_width_rows)), ocd_median_m=float(np.median(ocd_width)),
                   ac_minus_ocd_median_m=float(np.median(dw)), ac_minus_ocd_p5_m=float(np.percentile(dw, 5)),
                   ac_minus_ocd_p95_m=float(np.percentile(dw, 95)),
                   ac_narrower_by_2m_fraction=float((dw < -2).mean())),
        curvature=dict(profile_correlation=corr, corners=corners),
    )
    transform = (rot, 1., trans)
    if fast is not None:
        fast2 = open_loop(fast[:, :2])
        report['racing_lines'] = {
            'ac_fast_lane_vs_ocd_training_path': compare_lines(fast2, open_loop(track.path[:, 1:3]), transform, args.step)[0],
            'ac_fast_lane_vs_tum_raceline': compare_lines(fast2, open_loop(track.reference[:, 1:3]), transform, args.step)[0]}
    if extra_columns:
        z = (left[:, 2]+right[:, 2])/2
        report['ac_third_column_range'] = float(z.max()-z.min())

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.plot(*np.vstack([ocd_center, ocd_center[:1]]).T, label='OCD/TUM border midpoint', lw=1.2)
    ax.plot(*np.vstack([moved, moved[:1]]).T, label='AC border midpoint (aligned)', lw=1.2)
    for b in (open_loop(track.left), open_loop(track.right)):
        ax.plot(*np.vstack([b, b[:1]]).T, color='C0', lw=.5, alpha=.6)
    for b in (left2, right2):
        bb = (rot@open_loop(b).T).T+trans
        ax.plot(*np.vstack([bb, bb[:1]]).T, color='C1', lw=.5, alpha=.6)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.legend()
    fig.tight_layout(); fig.savefig(args.out/'overlay.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    o = np.argsort(s_mid)
    axes[0].plot(s_mid[o], e_mid[o]); axes[0].set_ylabel('AC mid - OCD centre\nlateral [m]')
    o = np.argsort(s_rows)
    axes[1].plot(s_rows[o], ac_width_rows[o], label='AC'); axes[1].plot(s_rows[o], ocd_width_at[o], label='OCD')
    axes[1].set_ylabel('total width [m]'); axes[1].legend()
    axes[2].plot(grid, kg_ocd, label='OCD'); axes[2].plot(grid, kg_ac, label='AC', alpha=.8)
    axes[2].set_ylabel('curvature [1/m]'); axes[2].set_xlabel('OCD station [m]'); axes[2].legend()
    fig.tight_layout(); fig.savefig(args.out/'profiles.png', dpi=150); plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'curvature'}, indent=2))
    print(f'curvature correlation {corr:.3f}; {len(corners)} corners in report.json')


if __name__ == '__main__':
    main()
