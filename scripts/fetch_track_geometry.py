#!/usr/bin/env python
"""C2a: centre lines from bacinger/f1-circuits (MIT) -> TUM-format track files (data/tracks_extra/).

Recipe of docs/research/c2_track_geometry.md section 5, for the 9 circuits TUM lacks and the 5 TUM
layouts that fail the R2 gate (`sources.f1_game_map.BACINGER_IDS`), at the pinned commit
`ExtraGeometryConfig.commit`:
 1. download each GeoJSON into its own directory data/tracks_extra/raw/bacinger/<id>/ (gitignored;
    untrusted: size-capped, parsed with `json` only, sha256 recorded; nothing in it is executed);
 2. lon/lat (WGS84) -> local metric frame: east/north of the tangent plane (ENU, h = 0) at the
    vertex centroid, exact ellipsoid formulas (no pyproj);
 3. close the loop (drop a repeated closing vertex);
 4. periodic smoothing spline (scipy splprep, per=1) with s = n * spline_rms_m^2, resampled at
    ~resample_m of arc length; RMS / max vertex deviation recorded;
 5. driving direction: both orientations go through the R2 gate on the game laps of that track;
    the one with the larger median |corr(steer, kappa)| is kept (the gate also checks the sign the
    F1 adapter declares, so a mirror image would show as a sign failure in the table);
 6. start: the loop is rotated so that index 0 is game lap distance 0 (the fitted gate shift);
 7. width: ONE PLACEHOLDER constant per side, `ExtraGeometryConfig.assumed_half_width_m` (half the
    median total width of the TUM tracks; recomputed and checked here), `width_assumed = true`;
 8. write <stem>.csv (`# x_m,y_m,w_tr_right_m,w_tr_left_m`) + <stem>.json provenance sidecar (no
    timestamp: re-runs on the same inputs are byte-identical) and the licence notice
    LICENSE-bacinger-f1-circuits.md (verbatim copy of the upstream LICENSE.md, QA M14).
A raw file whose sha256 differs from the `source.sha256` of the existing sidecar stops the script
(QA M13) unless --update is given. Then every candidate (converted and TUM) of these tracks goes
through the same gate and the table is printed. No usernames are read into the output (laps are only
used for the gate).

Usage: .venv/bin/python scripts/fetch_track_geometry.py [--sources setup_data/*] [--only Monaco ...]
       [--offline]   (re-convert from the raw files already on disk)   [--update] (accept new raw files)
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import urllib.request
import numpy as np
from scipy.interpolate import splev, splprep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlsetup.critic.config import Config, REPO_ROOT                               # noqa: E402
from rlsetup.critic.geometry import Geometry, gate, gate_track, polyline_length, tum_median_total_width  # noqa: E402
from rlsetup.critic.sources.f1_game_map import BACINGER_IDS, TRACK_GEOMETRY      # noqa: E402
from rlsetup.critic.sources.f1laps import STEER_KAPPA_SIGN, expand_sources, lap_style_input, load_exports  # noqa: E402

WGS84_A, WGS84_F = 6378137.0, 1 / 298.257223563
# official lengths for the report only (docs/research/c2_track_geometry.md section 2), metres
OFFICIAL_M = {'Madrid': 5414., 'Lusail': 5419., 'MexicoCity': 4304.}
LICENSE_COPY = 'LICENSE-bacinger-f1-circuits.md'


def download(url: str, dest: Path, max_bytes: int) -> bytes:
    """GET `url` into `dest` (own directory), refusing more than `max_bytes`. Generic client name."""
    req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib (track-geometry fetch)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f'{url}: more than {max_bytes} bytes, refused')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    tmp.write_bytes(data)
    tmp.replace(dest)
    return data


def parse_geojson(data: bytes) -> np.ndarray:
    """(n, 2) [lon, lat] degrees of the single LineString of a bacinger circuit file (JSON only)."""
    obj = json.loads(data.decode('utf-8'))
    feats = obj['features'] if obj.get('type') == 'FeatureCollection' else [obj]
    lines = [f['geometry']['coordinates'] for f in feats if f.get('geometry', {}).get('type') == 'LineString']
    if len(lines) != 1:
        raise ValueError(f'expected one LineString, found {len(lines)}')
    a = np.array([[float(p[0]), float(p[1])] for p in lines[0]], float)
    if len(a) < 10 or not np.isfinite(a).all() or np.abs(a[:, 0]).max() > 180 or np.abs(a[:, 1]).max() > 90:
        raise ValueError('implausible coordinates')
    return a


def lonlat_to_enu(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """WGS84 lon/lat [deg] (h = 0) -> east/north [m] in the tangent plane at (lon0, lat0)."""
    e2 = WGS84_F * (2 - WGS84_F)

    def ecef(lon, lat):
        lon, lat = np.radians(lon), np.radians(lat)
        n = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        return np.stack([n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon), n * (1 - e2) * np.sin(lat)], -1)
    d = ecef(lonlat[:, 0], lonlat[:, 1]) - ecef(np.array(lon0), np.array(lat0))
    lo, la = math.radians(lon0), math.radians(lat0)
    east = -math.sin(lo) * d[:, 0] + math.cos(lo) * d[:, 1]
    north = -math.sin(la) * math.cos(lo) * d[:, 0] - math.sin(la) * math.sin(lo) * d[:, 1] + math.cos(la) * d[:, 2]
    return np.column_stack((east, north))


def close_loop(xy: np.ndarray, tol_m: float = .5):
    """(open loop, closing gap [m]): a repeated closing vertex (within tol) is dropped."""
    gap = float(np.linalg.norm(xy[0] - xy[-1]))
    return (xy[:-1] if gap < tol_m else xy), gap


def smooth_resample(xy: np.ndarray, rms_m: float, step_m: float):
    """Periodic smoothing spline through the open loop `xy`, resampled at ~step_m of arc length.
    Returns (points (m, 2), info dict with s, RMS / max vertex deviation and the length)."""
    pts = np.vstack((xy, xy[:1]))                       # splprep(per=1) wants the first point repeated
    u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = len(xy) * rms_m ** 2
    tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u / u[-1], s=s, per=1, k=3)
    dense = np.column_stack(splev(np.linspace(0, 1, 40 * int(u[-1] / step_m) + 1), tck))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    n = int(round(arc[-1] / step_m))
    grid = np.arange(n) * arc[-1] / n
    out = np.column_stack((np.interp(grid, arc, dense[:, 0]), np.interp(grid, arc, dense[:, 1])))
    dev = np.min(np.linalg.norm(xy[:, None, :] - dense[None, :-1, :], axis=2), axis=1)
    return out, {'spline_s': s, 'vertex_dev_rms_m': float(np.sqrt(np.mean(dev ** 2))),
                 'vertex_dev_max_m': float(dev.max()), 'length_m': float(arc[-1]), 'spacing_m': float(arc[-1] / n)}


def track_laps(data, cfg, stem):
    """(game tracks, raw (distance, steer) of the laps that pass R3) of the game tracks mapped to `stem`."""
    names = sorted(t for t, s in TRACK_GEOMETRY.items() if s == stem)
    laps = []
    for key in sorted(data.laps):
        lap = data.laps[key]
        if lap.track in names and lap.has_samples:
            si = lap_style_input(lap, cfg.style_input)
            if si.ok:
                laps.append((si.raw_distance_m, si.raw_steer_deg))
    return names, laps


def convert(stem, gid, raw: bytes, sha: str, laps, cfg: Config, out_dir: Path, url: str, licence: dict):
    xcfg = cfg.extra_geometry
    lonlat = parse_geojson(raw)
    lon0, lat0 = float(lonlat[:, 0].mean()), float(lonlat[:, 1].mean())
    xy, gap = close_loop(lonlat_to_enu(lonlat, lon0, lat0))
    pts, info = smooth_resample(xy, xcfg.spline_rms_m, xcfg.resample_m)
    hw = np.full(len(pts), xcfg.assumed_half_width_m)
    orient = {}
    for name, p in (('as_file', pts), ('reversed', pts[::-1].copy())):
        g = Geometry(stem, 'extra', Path(''), '', p, hw, hw, width_assumed=True)
        orient[name] = (p, gate(g, laps, cfg.geometry, STEER_KAPPA_SIGN))
    if laps:
        chosen = max(orient, key=lambda k: orient[k][1].median_abs_corr)
        p, res = orient[chosen]
        k = int(round(res.shift_m / info['spacing_m'])) % len(p)
        p = np.roll(p, -k, axis=0)                      # index 0 = game lap distance 0
        direction_checked = True
    else:
        chosen, p, k, direction_checked = 'as_file', pts, 0, False
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{stem}.csv'
    with open(csv_path, 'w') as f:
        f.write('# x_m,y_m,w_tr_right_m,w_tr_left_m\n')
        for (x, y), w in zip(p, hw):
            f.write(f'{x:.4f},{y:.4f},{w:.4f},{w:.4f}\n')
    side = {'stem': stem, 'source': {'repo': xcfg.repo, 'commit': xcfg.commit, 'id': gid, 'url': url, 'sha256': sha,
                                     'licence': licence},
            'crs': f'local ENU tangent plane (east, north) at lon {lon0:.7f}, lat {lat0:.7f}, WGS84 ellipsoid, h = 0',
            'vertices': int(len(lonlat)), 'closing_gap_m': gap, **info,
            'direction': chosen, 'direction_checked': direction_checked,
            'direction_gate': {n: {'median_abs_corr': r.median_abs_corr, 'scale': r.scale, 'shift_m': r.shift_m}
                               for n, (_, r) in orient.items()},
            'start': 'index 0 = game lap distance 0 (fitted gate shift)' if direction_checked else 'as file',
            'start_roll_samples': k, 'gate_laps': len(laps),
            'width_assumed': True, 'w_tr_right_m': xcfg.assumed_half_width_m, 'w_tr_left_m': xcfg.assumed_half_width_m,
            'width_method': 'PLACEHOLDER (provisional pending Ibrahim): half the median over the vendored TUM tracks '
                            'of each track median total width (W2 of c2_track_geometry.md 4.1)',
            'csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    csv_path.with_suffix('.json').write_text(json.dumps(side, indent=1) + '\n')
    return side


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sources', nargs='+', default=[str(REPO_ROOT / 'setup_data' / '*')],
                    help='f1laps exports (laps for the gate)')
    ap.add_argument('--only', nargs='*', default=None, help='geometry stems (default: all of BACINGER_IDS)')
    ap.add_argument('--offline', action='store_true', help='use the raw files on disk, no download')
    ap.add_argument('--update', action='store_true', help='accept raw files whose sha256 differs from the sidecar')
    args = ap.parse_args(argv)
    cfg = Config()
    xcfg = cfg.extra_geometry
    median_w = tum_median_total_width(cfg.geometry)
    print(f'TUM median total width {median_w:.4f} m -> half {median_w / 2:.4f} m; '
          f'config assumed_half_width_m = {xcfg.assumed_half_width_m}')
    if abs(median_w / 2 - xcfg.assumed_half_width_m) > 5e-4:
        raise SystemExit('assumed_half_width_m no longer equals half the TUM median width: update the config')
    raw_root, out_dir = Path(xcfg.raw_dir), Path(cfg.geometry.extra_dir)
    lic_path = raw_root / 'LICENSE' / 'LICENSE.md'
    lic_url = xcfg.licence_url.format(repo=xcfg.repo, commit=xcfg.commit)
    lic = lic_path.read_bytes() if args.offline else download(lic_url, lic_path, xcfg.max_download_bytes)
    licence = {'name': 'MIT', 'url': lic_url, 'sha256': hashlib.sha256(lic).hexdigest(), 'copy': LICENSE_COPY,
               'copyright': next((l.strip() for l in lic.decode('utf-8', 'replace').splitlines() if 'Copyright' in l), '')}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LICENSE_COPY).write_bytes(lic)          # verbatim MIT notice next to the derived files (QA M14)
    data = load_exports(expand_sources(args.sources), log=lambda *a: None)
    stems = args.only or list(BACINGER_IDS)
    for stem in stems:
        gid = BACINGER_IDS[stem]
        url = xcfg.url.format(repo=xcfg.repo, commit=xcfg.commit, id=gid)
        raw_path = raw_root / gid / f'{gid}.geojson'
        raw = raw_path.read_bytes() if args.offline else download(url, raw_path, xcfg.max_download_bytes)
        sha = hashlib.sha256(raw).hexdigest()
        side_path = out_dir / f'{stem}.json'
        if side_path.is_file() and not args.update:     # QA M13: a changed raw file is never converted silently
            old = json.loads(side_path.read_text()).get('source', {}).get('sha256')
            if old != sha:
                raise SystemExit(f'{stem}: raw {gid}.geojson sha256 {sha[:12]} != sidecar {str(old)[:12]}; '
                                 f'check the file, then re-run with --update')
        names, laps = track_laps(data, cfg, stem)
        side = convert(stem, gid, raw, sha, laps, cfg, out_dir, url, licence)
        print(f"{stem:11s} {gid}: {side['vertices']} vertices, sha256 {sha[:12]}, spline RMS "
              f"{side['vertex_dev_rms_m']:.2f} m (max {side['vertex_dev_max_m']:.1f}), length {side['length_m']:.0f} m, "
              f"direction {side['direction']} ({side['direction_gate']['as_file']['median_abs_corr']:.3f} as file / "
              f"{side['direction_gate']['reversed']['median_abs_corr']:.3f} reversed), {len(laps)} laps of {names}")
    print('\nR2 gate on every candidate (extra = bacinger conversion, tum = vendored TUM file):')
    print(f"{'track':12s} {'cand':5s} {'laps':>4s} {'L_game':>7s} {'L_geo':>7s} {'dL %':>7s} {'scale':>7s} "
          f"{'shift m':>7s} {'closure':>7s} {'|corr|':>6s} {'@s=1':>6s} {'signed':>7s} result")
    for stem in stems:
        _, laps = track_laps(data, cfg, stem)
        for g, r in gate_track(stem, laps, cfg.geometry, STEER_KAPPA_SIGN).results:
            print(f'{stem:12s} {g.kind:5s} {r.n_laps:4d} {r.L_game:7.0f} {r.L_geo:7.0f} {100 * r.length_rel:+7.3f} '
                  f"{r.scale:7.4f} {r.shift_m:+7.0f} {r.closure_gap_m:+7.1f} {r.median_abs_corr:6.3f} "
                  f"{r.shift_only.get('median_abs_corr', np.nan):6.3f} {r.median_signed_corr:+7.3f} "
                  f"{'PASS' if r.passed else 'FAIL ' + r.reason}")
        if stem in OFFICIAL_M:
            L = polyline_length(np.loadtxt(out_dir / f'{stem}.csv', delimiter=',', comments='#')[:, :2])
            print(f'{"":12s} official {OFFICIAL_M[stem]:.0f} m; bacinger measured {L:.0f} m')


if __name__ == '__main__':
    main()
