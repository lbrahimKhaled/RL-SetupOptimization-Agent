"""xi_ref geometry and the per-track precomputed preview grid.

`ReferenceLine` is the driver's reference line xi_ref: geometry only (DECISIONS #2), a closed
x, y polyline [m] in the track's frame. Arc length s_ref [m] starts at its first point. The
lateral offset e is positive to the LEFT of the line (ocd_first_lap convention); the heading
error is yaw - line heading, wrapped to [-pi, pi). Matching and the local frame use exactly the
same TUM helpers and +-0.5 m tangent as `ocd_lap.env.OCDRacingEnv`, so a ReferenceLine built
from `Track.path` reproduces that env's offsets bit for bit.

`TrackGrid` precomputes, on a regular grid of `Track.path` arc length (the driver-independent
progress coordinate), the xi_ref point abeam each grid station and the left/right track edges
(review R4.2), so the observation only interpolates at run time.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
import trajectory_planning_helpers as tph
from ocd_lap.track import closed_path

MATCH_RANGE_M = 15.       # = Track.match: search window around the hint
MATCH_INTERP = 101        # = Track.match
TANGENT_HALF_M = .5       # = OCDRacingEnv._frame
CLOSE_TOL_M = 1e-6        # = closed_path: first/last point treated as the same point
MIN_SEGMENT_M = 1e-6      # shorter segments count as duplicate points


class ReferenceLine:
    """Closed reference polyline with arc length. `path` is [s, x, y] with the first point
    repeated at the end (as `Track.path`)."""

    def __init__(self, path, name='xi_ref'):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 4 or not np.isfinite(path).all():
            raise ValueError('path must be a finite [s, x, y] array with at least 4 rows')
        # QA MINOR 10: a duplicate point gives a zero tangent and NaN offsets downstream
        if np.linalg.norm(path[-1, 1:] - path[0, 1:]) > CLOSE_TOL_M:
            raise ValueError('path must be closed (last point = first point)')
        if path[0, 0] != 0. or not (np.diff(path[:, 0]) > 0.).all():
            raise ValueError('arc length s must start at 0 and increase strictly')
        if not (np.linalg.norm(np.diff(path[:, 1:], axis=0), axis=1) > MIN_SEGMENT_M).all():
            raise ValueError(f'path has a segment shorter than {MIN_SEGMENT_M} m (duplicate point?)')
        self.path = path
        self.length = float(path[-1, 0])
        self.name = name
        self.digest = hashlib.sha256(np.ascontiguousarray(path).tobytes()).hexdigest()[:16]

    @classmethod
    def from_xy(cls, xy, name='xi_ref'):
        return cls(closed_path(np.asarray(xy, dtype=float)[:, :2]), name)

    @classmethod
    def from_track_path(cls, track):
        """The safe minimum-curvature line of ocd_first_lap (xi_ref placeholder, DECISIONS #1)."""
        return cls(track.path, f'{track.name}:Track.path')

    @classmethod
    def from_csv(cls, path):
        """x, y [m] columns in the track's frame ('#' comments allowed), e.g. Jad's laps."""
        xy = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
        return cls.from_xy(xy[:, :2], Path(path).name)

    def point(self, s):
        """Point(s) at arc length s [m] (any real s; wraps). Scalar s -> (2,), array -> (n, 2)."""
        s = np.asarray(s, dtype=float) % self.length
        out = np.stack([np.interp(s, self.path[:, 0], self.path[:, i]) for i in (1, 2)], axis=-1)
        return out

    def frame(self, s):
        """Point and unit tangent at s (tangent from points +-0.5 m, as OCDRacingEnv._frame)."""
        p = np.array([np.interp(s % self.length, self.path[:, 0], self.path[:, i]) for i in (1, 2)])
        ahead = np.array([np.interp((s + TANGENT_HALF_M) % self.length, self.path[:, 0], self.path[:, i])
                          for i in (1, 2)])
        behind = np.array([np.interp((s - TANGENT_HALF_M) % self.length, self.path[:, 0], self.path[:, i])
                           for i in (1, 2)])
        t = ahead - behind
        return p, t / np.linalg.norm(t)

    def heading(self, s):
        _, t = self.frame(s)
        return math.atan2(t[1], t[0])

    def match(self, xy, hint=None):
        """Arc length [0, L) of the closest point; `hint` restricts the search to +-15 m."""
        s, _ = tph.path_matching_global.path_matching_global(
            self.path, np.asarray(xy, dtype=float), s_expected=hint, s_range=MATCH_RANGE_M,
            no_interp_values=MATCH_INTERP)
        return float(s)

    def errors_at(self, s, x, y, yaw):
        """(e [m], heading error [rad]) of a pose relative to the line at arc length s."""
        p, t = self.frame(s)
        e = float(t[0] * (y - p[1]) - t[1] * (x - p[0]))   # positive = left of the line
        heading = (yaw - math.atan2(t[1], t[0]) + math.pi) % (2 * math.pi) - math.pi
        return e, heading


def _closed_like(points, n_closed):
    """Append the first point if `points` is one row short of a closed array of n_closed rows."""
    if len(points) == n_closed - 1:
        return np.vstack((points, points[:1]))
    if len(points) == n_closed:
        return points
    raise ValueError('edge arrays do not match the centre line')


class TrackGrid:
    """Per-station geometry on a regular grid of Track.path arc length s (step `grid_m`).

    Arrays (n = number of grid stations, station i at s = i * step, step <= grid_m):
      s_ref[i]   xi_ref arc length matched to the path point at station i (unwrapped, increasing)
      ref[i]     xi_ref point at s_ref[i]
      left[i], right[i]  track-edge points of the centre-line cross-section closest to station i
    Interpolation is linear in s with wrap-around at the lap length.
    """

    def __init__(self, track, ref, grid_m=1.):
        self.length = float(track.length)
        n = int(math.ceil(self.length / grid_m))
        self.step = self.length / n
        self.s = np.arange(n) * self.step
        xy = np.column_stack([np.interp(self.s, track.path[:, 0], track.path[:, i]) for i in (1, 2)])
        # xi_ref arc length abeam each station: sequential hinted matching (robust where other
        # parts of the line come close, e.g. hairpins).
        s_ref = np.empty(n)
        hint = ref.match(xy[0])
        for i in range(n):
            hint = s_ref[i] = ref.match(xy[i], hint)
        self.s_ref = np.unwrap(s_ref, period=ref.length)
        if self.s_ref[-1] < self.s_ref[0]:
            raise ValueError('reference line runs against Track.path direction')
        self.ref_length = ref.length
        self.ref = ref.point(self.s_ref)
        # Edges: the cross-section through the nearest centre-line point (Track.left/right are
        # aligned with the centre-line points).
        center = track.center
        left = _closed_like(track.left, len(center))
        right = _closed_like(track.right, len(center))
        s_c = np.empty(n)
        hint = None
        for i in range(n):
            hint = s_c[i] = tph.path_matching_global.path_matching_global(
                center, xy[i], s_expected=hint, s_range=MATCH_RANGE_M, no_interp_values=MATCH_INTERP)[0]
        self.left = np.column_stack([np.interp(s_c, center[:, 0], left[:, i]) for i in (0, 1)])
        self.right = np.column_stack([np.interp(s_c, center[:, 0], right[:, i]) for i in (0, 1)])

    def _weights(self, s):
        u = (np.asarray(s, dtype=float) % self.length) / self.step
        i0 = np.floor(u).astype(int) % len(self.s)
        return i0, (i0 + 1) % len(self.s), (u - np.floor(u))[..., None]

    def at(self, s):
        """(ref, left, right) points at path arc length(s) s, each (..., 2)."""
        i0, i1, w = self._weights(s)
        lerp = lambda a: a[i0] * (1 - w) + a[i1] * w
        return lerp(self.ref), lerp(self.left), lerp(self.right)

    def s_ref_at(self, s):
        """xi_ref arc length [0, L_ref) abeam path arc length s."""
        i0, i1, w = self._weights(s)
        w = w[..., 0]
        b = self.s_ref[i1] + np.where(i1 == 0, self.ref_length, 0.)  # one lap on past the end
        return (self.s_ref[i0] * (1 - w) + b * w) % self.ref_length


_GRID_CACHE = {}


def track_grid(track, ref, grid_m=1.):
    """Cached TrackGrid (one per process, track, reference line and grid)."""
    key = (track.name, float(track.length), ref.digest, float(grid_m))
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = TrackGrid(track, ref, grid_m)
    return _GRID_CACHE[key]
