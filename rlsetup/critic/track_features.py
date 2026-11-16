"""Track summary z (5 numbers, S6) from a closed centre line. Carried over from C1 unchanged.

Smoothing and the corner rule are the C1 review's [ARCH] table; the radius bands are PDF Table VIII,
PLACEHOLDER (PROJECT_CONTEXT 11.7; C2_review "Needs Ibrahim 5"). `n_corners` is coarse (C1 review:
mean error 2.5 corners). In C2 the centre line must come from a geometry that passed the R2 gate
(`geometry.gate_track`); tracks without one get NaN (`style_data.track_feature_table`).
"""
import numpy as np
from ocd_lap.track import closed_path
from .config import TRACK_FEATURES, TrackFeatureConfig


def resample_closed(xy: np.ndarray, spacing_m: float) -> np.ndarray:
    """Closed polyline (first point not repeated) resampled to ~spacing_m by arc length."""
    path = closed_path(np.asarray(xy, dtype=float))
    s, x, y = path[:, 0], path[:, 1], path[:, 2]
    n = max(int(round(s[-1] / spacing_m)), 3)
    grid = np.arange(n) * s[-1] / n
    return np.column_stack((np.interp(grid, s, x), np.interp(grid, s, y)))


def _smooth(a: np.ndarray, width: int) -> np.ndarray:
    """Circular moving average over `width` (odd) samples."""
    half = width // 2
    wrapped = np.r_[a[-half:], a, a[:half]]
    return np.convolve(wrapped, np.ones(width) / width, mode='valid')


def curvature(xy: np.ndarray):
    """(signed curvature [1/m], step length ds [m]) of a closed sampled curve, by central differences."""
    x, y = xy[:, 0], xy[:, 1]
    dx, dy = (np.roll(x, -1) - np.roll(x, 1)) / 2, (np.roll(y, -1) - np.roll(y, 1)) / 2
    ddx, ddy = np.roll(x, -1) - 2 * x + np.roll(x, 1), np.roll(y, -1) - 2 * y + np.roll(y, 1)
    return (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-12), np.hypot(dx, dy)


def _runs(mask: np.ndarray):
    """[(start, stop)] of True runs in a linear (non-wrapping) boolean array."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def corner_summary(kappa: np.ndarray, ds: np.ndarray, cfg: TrackFeatureConfig = TrackFeatureConfig()):
    """(n_corners, longest straight [m]) of a closed curve given curvature and step lengths. A corner
    is a run with radius < corner_radius_m; two runs merge only with the same curvature sign and a gap
    < merge_gap_m; runs shorter than min_corner_m are dropped. Runs are found on the lap rotated to
    start at its longest raw straight, so nothing wraps the array end."""
    straight = 1 / np.maximum(np.abs(kappa), 1e-9) >= cfg.corner_radius_m
    if straight.all():
        return 0., float(ds.sum())
    if not straight.any():
        return 1., 0.
    n = len(straight)
    start, stop = max(_runs(np.r_[straight, straight]), key=lambda r: min(r[1] - r[0], n))
    roll = start % n
    kappa, ds, corner = np.roll(kappa, -roll), np.roll(ds, -roll), np.roll(~straight, -roll)
    merged = []
    for a, b in _runs(corner):
        if merged:
            pa, pb = merged[-1]
            same = np.sign(kappa[pa:pb].mean()) == np.sign(kappa[a:b].mean())
            if same and ds[pb:a].sum() < cfg.merge_gap_m:
                merged[-1][1] = b
                continue
        merged.append([a, b])
    kept = [(a, b) for a, b in merged if ds[a:b].sum() >= cfg.min_corner_m]
    in_corner = np.zeros(n, dtype=bool)
    for a, b in kept:
        in_corner[a:b] = True
    stretches = _runs(~in_corner)
    lens = [float(ds[a:b].sum()) for a, b in stretches]
    if len(stretches) > 1 and stretches[0][0] == 0 and stretches[-1][1] == n:
        lens[0] += lens.pop()
    return float(len(kept)), max(lens) if lens else 0.


def track_features(xy: np.ndarray, cfg: TrackFeatureConfig = TrackFeatureConfig()) -> np.ndarray:
    """z = [n_corners, longest_straight_m, straight_share, slow_corner_share, fast_corner_share] for a
    closed centerline `xy` (N, 2) [m]: 1 m resample, 40 m moving average twice, then shares by length
    of the radius bands (straight > 500 m, slow < 50 m, fast 150-500 m; R15)."""
    pts = resample_closed(xy, cfg.resample_m)
    width = 2 * int(round(cfg.smooth_window_m / cfg.resample_m / 2)) + 1
    for _ in range(cfg.smooth_passes):
        pts = np.column_stack((_smooth(pts[:, 0], width), _smooth(pts[:, 1], width)))
    kappa, ds = curvature(pts)
    n_corners, longest = corner_summary(kappa, ds, cfg)
    radius = 1 / np.maximum(np.abs(kappa), 1e-9)
    share = lambda mask: float(ds[mask].sum() / ds.sum())
    return np.array([n_corners, longest, share(radius >= cfg.corner_radius_m), share(radius < cfg.slow_radius_m),
                     share((radius >= cfg.fast_radius_m) & (radius < cfg.corner_radius_m))])


class TrackScaler:
    """Standardises z on the training tracks that have z (unweighted per track) and clips to +-z_clip
    (C1-R15). Tracks without geometry keep NaN after `transform`."""

    def __init__(self, z_clip: float = TrackFeatureConfig.z_clip):
        self.mean, self.std, self.z_clip = None, None, z_clip

    def fit(self, table: np.ndarray, train_tracks) -> 'TrackScaler':
        z = table[np.unique(train_tracks)]
        z = z[np.isfinite(z).all(axis=1)]
        if not len(z):
            raise ValueError('no training track has track features')
        self.mean, self.std = z.mean(axis=0), z.std(axis=0)
        self.std = np.where(self.std > 0, self.std, 1.)
        return self

    def transform(self, table: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        return np.clip((table[tracks] - self.mean) / self.std, -self.z_clip, self.z_clip)
