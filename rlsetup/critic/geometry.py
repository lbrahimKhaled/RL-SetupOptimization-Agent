"""Track geometry for C2: TUM-format files -> style-rl `Track`, the placeholder line and the R2 gate.

A geometry file is TUM format (`# x_m,y_m,w_tr_right_m,w_tr_left_m`, closed loop, driving direction,
metres). Edges (S3): left = centre + w_left n, right = centre - w_right n, with n the left unit
normal of the centre line (tangent rotated +90 deg). Candidates of a track: `extra_dir/<stem>.csv`
(converted bacinger centre lines, `scripts/fetch_track_geometry.py`), then the vendored TUM file.

R2 gate: progress on the geometry = (shift + scale * game distance) mod L_geo. Per track (laps of all
games pooled) the shift and scale that maximise the median over laps of |corr(steer, kappa)| are
fitted: shift over the whole lap on the geometry's ~2 m kappa grid (exact circular cross-correlation
by FFT), scale on a grid around L_geo / L_game (`GeometryConfig.scale_rel_grid`). Accept only if the
median |corr| >= 0.85 and |L_game / L_geo - 1| <= 1 % and the median signed corr has the sign the
source declares (`SourceAdapter.steer_kappa_sign`; a mirror-image geometry has the other sign, QA M2).
kappa is style-rl's `Track.kappa` (+ = left turn, 1/m, 12 m Gaussian smoothing); steering is the raw
game samples (+ = right turn in F1 25/26), so an aligned lap has corr < 0. L_game = median over the track's laps of the last steering-sample
distance [m].

Placeholder line (Decision 6, C2_review [ARCH]): TUM raceline where the chosen geometry is a TUM file,
else the centre line (lead's instruction for C2a; the review's minimum-curvature line is not built).
Every line is a placeholder (`is_placeholder` = True); `line_kind` says which.
"""
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import hashlib
import json
import numpy as np
from ..style import Track as StyleTrack
from .config import GATE_LENGTH, GATE_LOW_CORR, GATE_NO_LAPS, GATE_SIGN, GeometryConfig, NO_GEOMETRY, OK

TUM_LINE = 'tum_raceline'
CENTRE_LINE = 'centre_line'


def tum_dir(cfg: GeometryConfig = GeometryConfig()) -> Path:
    """The TUM racetrack-database directory (`tracks/`, `racelines/`)."""
    if cfg.tum_dir:
        return Path(cfg.tum_dir)
    import ocd_lap.track
    return ocd_lap.track.ROOT / 'vendor/racetrack-database'


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tum_csv(path):
    """(xy (n, 2), w_right (n,), w_left (n,)) [m] of a TUM track file; a repeated closing point is dropped."""
    a = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    if a.shape[1] < 4:
        raise ValueError(f'{path}: expected x_m,y_m,w_tr_right_m,w_tr_left_m')
    if np.linalg.norm(a[0, :2] - a[-1, :2]) < 1e-4:
        a = a[:-1]
    return a[:, :2], a[:, 2], a[:, 3]


def left_normals(xy: np.ndarray) -> np.ndarray:
    """Unit left normals (tangent rotated +90 deg) of a closed polyline, central differences."""
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack((-t[:, 1], t[:, 0]))


def edges(xy, w_right, w_left):
    """(left, right) edge vertices [m]: centre +- width along the left normal (S3)."""
    n = left_normals(np.asarray(xy, float))
    return xy + np.asarray(w_left)[:, None] * n, xy - np.asarray(w_right)[:, None] * n


def polyline_length(xy) -> float:
    """Length [m] of the closed polyline (last point joined to the first)."""
    return float(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1).sum())


def tum_median_total_width(cfg: GeometryConfig = GeometryConfig()) -> float:
    """Median over the vendored TUM tracks of each track's median total width w_right + w_left [m]
    (the basis of `ExtraGeometryConfig.assumed_half_width_m`)."""
    per_track = [np.median(w_r + w_l) for w_r, w_l in
                 (read_tum_csv(p)[1:] for p in sorted((tum_dir(cfg) / 'tracks').glob('*.csv')))]
    return float(np.median(per_track))


@dataclass
class Geometry:
    """One geometry candidate of a track. `line_xy` None = the centre line is the placeholder line."""
    stem: str
    kind: str                       # 'extra' (data/tracks_extra) or 'tum'
    path: Path
    sha256: str
    xy: np.ndarray                  # (n, 2) centre line [m]
    w_right: np.ndarray             # (n,) [m]
    w_left: np.ndarray              # (n,) [m]
    line_xy: np.ndarray = None      # (m, 2) placeholder line [m] (TUM raceline) or None
    line_sha256: str = ''
    width_assumed: bool = False     # True when the widths are the PLACEHOLDER constant (sidecar)
    sidecar: dict = field(default_factory=dict)

    @property
    def line_kind(self) -> str:
        return CENTRE_LINE if self.line_xy is None else TUM_LINE

    @cached_property
    def track(self) -> StyleTrack:
        left, right = edges(self.xy, self.w_right, self.w_left)
        return StyleTrack(self.xy, left, right, name=self.stem)

    @property
    def length_m(self) -> float:
        return self.track.length_m

    @cached_property
    def corners(self):
        return self.track.find_corners()

    @cached_property
    def line(self) -> 'PlaceholderLine':
        return PlaceholderLine(self.track, self.line_xy)

    def digest(self) -> str:
        """sha256 of the geometry and line files (cache key part)."""
        return hashlib.sha256(f'{self.sha256}:{self.line_sha256}:{self.width_assumed}'.encode()).hexdigest()


def load_geometry(path, kind: str, raceline=None) -> Geometry:
    """Geometry from a TUM-format file; `raceline` = optional TUM raceline CSV (`# x_m,y_m`). A JSON
    sidecar `<stem>.json` next to the file (written by fetch_track_geometry.py) sets `width_assumed`."""
    path = Path(path)
    xy, w_r, w_l = read_tum_csv(path)
    side = path.with_suffix('.json')
    sidecar = json.loads(side.read_text()) if side.is_file() else {}
    line_xy, line_sha = None, ''
    if raceline is not None and Path(raceline).is_file():
        line_xy = np.loadtxt(raceline, delimiter=',', comments='#', ndmin=2)[:, :2]
        line_sha = sha256_file(raceline)
    return Geometry(stem=path.stem, kind=kind, path=path, sha256=sha256_file(path), xy=xy, w_right=w_r, w_left=w_l,
                    line_xy=line_xy, line_sha256=line_sha, width_assumed=bool(sidecar.get('width_assumed', False)),
                    sidecar=sidecar)


def candidates(stem: str, cfg: GeometryConfig = GeometryConfig()) -> list:
    """Geometry candidates of a track stem, in preference order (extra first, then TUM)."""
    out = []
    extra = Path(cfg.extra_dir) / f'{stem}.csv'
    if extra.is_file():
        out.append(load_geometry(extra, 'extra'))
    tum = tum_dir(cfg)
    if (tum / 'tracks' / f'{stem}.csv').is_file():
        out.append(load_geometry(tum / 'tracks' / f'{stem}.csv', 'tum', tum / 'racelines' / f'{stem}.csv'))
    return out


class PlaceholderLine:
    """The placeholder line as a function of progress on a style `Track`: `q(progress)` gives the
    dict style-rl's `style_vector(q=...)` takes (progress_m, lateral_m (+ = left of the centre line),
    left_width_m, right_width_m), linear in progress, periodic. Built by projecting the line's
    vertices with `Track.project` (s_hint from the line's own arc length, anchored at the projection
    of its first vertex, so no vertex is matched to a nearby other part of the circuit)."""

    def __init__(self, track: StyleTrack, line_xy=None):
        L = track.length_m
        if line_xy is None:
            xy, hint = track.center, track.s_vertices[:-1]
        else:
            xy = np.asarray(line_xy, float)
            if np.linalg.norm(xy[0] - xy[-1]) < 1e-4:
                xy = xy[:-1]
            s_line = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            s0 = track.project(xy[:1])['progress_m'][0]
            hint = (s0 + s_line * L / polyline_length(xy)) % L
        q = track.project(xy, s_hint=hint)
        order = np.argsort(q['progress_m'], kind='stable')
        s = q['progress_m'][order]
        keep = np.r_[True, np.diff(s) > 1e-9]
        self.length_m = L
        self.s = s[keep]
        self.values = {k: q[k][order][keep] for k in ('lateral_m', 'left_width_m', 'right_width_m')}

    def q(self, progress_m) -> dict:
        p = np.asarray(progress_m, float) % self.length_m
        out = {k: np.interp(p, self.s, v, period=self.length_m) for k, v in self.values.items()}
        return {'progress_m': p, **out}


# ---------------------------------------------------------------------------------------------------
# R2 gate
# ---------------------------------------------------------------------------------------------------
def corr_over_shifts(dist, value, kappa, ds: float, scale: float) -> np.ndarray:
    """Pearson corr(value(d), kappa(shift + scale d)) for every shift = k ds, k = 0..n-1 (n = len(kappa)),
    over the lap's samples d_j = j ds / scale <= dist[-1] (value linear in d). Laps longer than the
    loop after scaling wrap around; exact, by circular cross-correlation (FFT)."""
    kappa = np.asarray(kappa, float)
    n = len(kappa)
    m = int(np.floor(dist[-1] * scale / ds)) + 1
    u = np.interp(np.arange(m) * ds / scale, dist, value)
    k = np.arange(m) % n
    fold_u = np.bincount(k, u, n)
    fold_c = np.bincount(k, None, n).astype(float)

    def xcorr(a, b):                                    # sum_k a_k b_{(k + s) mod n}
        return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n)
    cross, sk, skk = xcorr(fold_u, kappa), xcorr(fold_c, kappa), xcorr(fold_c, kappa ** 2)
    su, suu = u.sum(), (u * u).sum()
    var = (suu - su * su / m) * (skk - sk * sk / m)
    return np.where(var > 0, (cross - su * sk / m) / np.sqrt(np.maximum(var, 1e-300)), 0.)


@dataclass
class GateResult:
    """R2 outcome of one geometry candidate. progress(d) = (shift_m + scale d) mod L_geo.
    Per-lap arrays (`lap_corr`, `lap_index`) follow the order of the laps passed to `gate`;
    `lap_index[i]` is the position in that input list of the lap whose corr is `lap_corr[i]` (QA M3)."""
    stem: str
    kind: str
    sha256: str
    n_laps: int
    L_game: float                   # median last steering-sample distance of the laps [m]
    L_geo: float                    # centre-line length of the geometry [m]
    line_sha256: str = ''           # sha256 of the placeholder-line file ('' = centre line), QA M11
    scale: float = 1.
    shift_m: float = 0.
    median_abs_corr: float = float('nan')
    lap_corr: np.ndarray = None     # signed corr per lap at the fitted alignment (steering sign check)
    lap_index: np.ndarray = None    # input position of each lap in lap_corr
    group_median_abs_corr: dict = field(default_factory=dict)   # per group (game) at the fitted alignment (QA I2)
    shift_only: dict = field(default_factory=dict)  # best shift at scale = 1 exactly (report only)
    expected_sign: int = 0          # declared sign of corr(steer, kappa) of the source; 0 = not checked
    passed: bool = False
    reason: str = GATE_NO_LAPS
    corr_ok: bool = False           # median |corr| >= min_abs_corr
    length_ok: bool = False         # |L_game / L_geo - 1| <= max_length_rel
    sign_ok: bool = False           # median signed corr has the expected sign (QA M2)

    @property
    def length_rel(self) -> float:
        return self.L_game / self.L_geo - 1.

    @property
    def closure_gap_m(self) -> float:
        """scale L_game - L_geo [m]: where the lap end lands relative to the start (QA I1); < 0 = short."""
        return self.scale * self.L_game - self.L_geo

    @property
    def median_signed_corr(self) -> float:
        return float(np.median(self.lap_corr)) if self.lap_corr is not None and len(self.lap_corr) else float('nan')

    def progress(self, distance_m) -> np.ndarray:
        return (self.shift_m + self.scale * np.asarray(distance_m, float)) % self.L_geo

    def summary(self) -> dict:
        corr = np.asarray(self.lap_corr if self.lap_corr is not None else [], float)
        return {'stem': self.stem, 'kind': self.kind, 'sha256': self.sha256, 'line_sha256': self.line_sha256,
                'n_laps': self.n_laps, 'L_game_m': self.L_game, 'L_geo_m': self.L_geo, 'length_rel': self.length_rel,
                'scale': self.scale, 'shift_m': self.shift_m, 'closure_gap_m': self.closure_gap_m,
                'median_abs_corr': self.median_abs_corr, 'median_signed_corr': self.median_signed_corr,
                'group_median_abs_corr': self.group_median_abs_corr,
                'laps_corr_negative': int((corr < 0).sum()), 'shift_only': self.shift_only,
                'expected_sign': self.expected_sign, 'corr_ok': self.corr_ok, 'length_ok': self.length_ok,
                'sign_ok': self.sign_ok, 'passed': self.passed, 'reason': self.reason}


def _wrap(x, L):
    return float((x + L / 2) % L - L / 2)


def gate(geom: Geometry, laps, cfg: GeometryConfig = GeometryConfig(), expected_sign: int = 0,
         groups=None) -> GateResult:
    """R2 for one candidate. `laps`: list of (distance_m, steer) raw arrays of the laps that passed
    the lap QC (R3), distance increasing; laps with < 4 samples are left out (`lap_index` says which
    were used). `expected_sign` (-1 / +1): the source's declared sign of corr(steer, kappa); a median
    signed corr of the other sign means a mirror-image geometry and fails with GATE_SIGN (QA M2).
    0 = not checked (only for report-only uses). `groups`: optional label per input lap (the game)
    for `group_median_abs_corr`."""
    track = geom.track
    L_geo = track.length_m
    idx = np.array([i for i, (d, _) in enumerate(laps) if len(d) >= 4], int)
    used = [(np.asarray(laps[i][0], float), np.asarray(laps[i][1], float)) for i in idx]
    if len(used) < cfg.min_gate_laps:
        return GateResult(geom.stem, geom.kind, geom.sha256, len(used), float('nan'), L_geo, line_sha256=geom.line_sha256,
                          lap_index=idx, lap_corr=np.zeros(0), expected_sign=expected_sign)
    L_game = float(np.median([d[-1] for d, _ in used]))
    base = L_geo / L_game

    def fit(scale):
        curves = np.stack([corr_over_shifts(d, v, track.kappa, track.ds, scale) for d, v in used])
        med = np.median(np.abs(curves), axis=0)
        k = int(np.argmax(med))
        return float(med[k]), k, curves[:, k]

    best = None
    for rel in cfg.scale_rel_grid:
        scale = base * (1. + rel)
        med, k, corr = fit(scale)
        if best is None or med > best[0]:
            best = (med, k, corr, scale)
    med, k, corr, scale = best
    med1, k1, _ = fit(1.)
    res = GateResult(geom.stem, geom.kind, geom.sha256, len(used), L_game, L_geo, line_sha256=geom.line_sha256,
                     scale=scale, shift_m=_wrap(k * track.ds, L_geo), median_abs_corr=med, lap_corr=corr, lap_index=idx,
                     shift_only={'scale': 1., 'shift_m': _wrap(k1 * track.ds, L_geo), 'median_abs_corr': med1},
                     expected_sign=int(expected_sign))
    if groups is not None:
        g = np.asarray(groups, object)[idx]
        res.group_median_abs_corr = {str(x): float(np.median(np.abs(corr[g == x]))) for x in sorted(set(g))}
    res.corr_ok, res.length_ok = med >= cfg.min_abs_corr, abs(res.length_rel) <= cfg.max_length_rel
    res.sign_ok = expected_sign == 0 or np.sign(res.median_signed_corr) == np.sign(expected_sign)
    res.passed = bool(res.corr_ok and res.length_ok and res.sign_ok)
    res.reason = (OK if res.passed else GATE_LOW_CORR if not res.corr_ok else GATE_SIGN if not res.sign_ok
                  else GATE_LENGTH)
    return res


def scale_scan(geom: Geometry, laps, scales) -> list:
    """Report only (never used by the gate): [(scale, best shift [m], median |corr|)] for each scale,
    e.g. to see whether a track's best correlation sits at a uniform distance scale."""
    track = geom.track
    laps = [(np.asarray(d, float), np.asarray(v, float)) for d, v in laps if len(d) >= 4]
    out = []
    for scale in scales:
        med = np.median(np.abs(np.stack([corr_over_shifts(d, v, track.kappa, track.ds, scale) for d, v in laps])), axis=0)
        k = int(np.argmax(med))
        out.append((float(scale), _wrap(k * track.ds, track.length_m), float(med[k])))
    return out


@dataclass
class TrackGeometry:
    """The gate outcome of one track: every candidate's result, and the chosen (first passing) one."""
    stem: str
    results: list                   # [(Geometry, GateResult)] in candidate order
    chosen: tuple = None            # (Geometry, GateResult) or None

    @property
    def reason(self) -> str:
        """OK, NO_GEOMETRY, or the gate reason of the preferred (first) candidate."""
        if self.chosen is not None:
            return OK
        return self.results[0][1].reason if self.results else NO_GEOMETRY


def gate_track(stem: str, laps, cfg: GeometryConfig = GeometryConfig(), expected_sign: int = 0,
               groups=None) -> TrackGeometry:
    """Gate every candidate of `stem` with `laps` (see `gate`); the first passing candidate is chosen."""
    results = [(g, gate(g, laps, cfg, expected_sign, groups)) for g in candidates(stem, cfg)]
    chosen = next(((g, r) for g, r in results if r.passed), None)
    return TrackGeometry(stem, results, chosen)
