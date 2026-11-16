"""Telemetry laps -> the 12 style numbers per lap and per cell, the R4 input transform, track z (C2a).

Flow (S3 + review R2-R4):
1. `lap_style_input` per lap (adapter): time base, R3 QC, 25 Hz, steer x 115.
2. Per track, the R2 gate on the laps that passed R3 (`geometry.gate_track`), all games pooled.
3. Per lap, style-rl's `style_vector` (numpy port) with `q=` from the placeholder line at
   progress = (shift + scale * game distance) mod L_geo. Tracks without a gated geometry keep their
   laps: B/T numbers and the two steering-rate numbers are computed on a stub track (length = the
   track's median game lap distance, no corners, no position), so S_corner_peak_angle and the three
   L_* are NaN and `geometry_reason` says why. Laps rejected by R3 get no numbers (`lap_reason`).
4. Cell (game, driver, track) = median over its laps with numbers (style-rl's rule).
Nothing is dropped. Each lap's numbers are cached in `StyleInputConfig.cache_dir/<cache_env>/` (JSON,
numbers only): the directory is the code digest + numpy / scipy versions + style-rl commit, the file
the lap's raw-sample digest, lap time, geometry digest + alignment and the config; stale directories
are pruned (QA M4).
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import inspect
import json
import os
import tempfile
import warnings
import numpy as np
from ..style import KEYS, STYLE_RL_COMMIT, style_vector
from .config import Config, LAP_NO_GRAPH, NO_GEOMETRY, OK, REPO_ROOT, StyleTransformConfig, TRACK_FEATURES
from .geometry import TrackGeometry, gate_track
from .track_features import track_features

GEOMETRY_KEYS = ('S_corner_peak_angle', 'L_border_mean_m', 'L_inside_dist_apex_m', 'L_outside_dist_exit_m')
# sources whose content defines the cached numbers (plus the adapter's own module, see code_digest)
CODE_FILES = ('rlsetup/style/__init__.py', 'rlsetup/style/features.py', 'rlsetup/style/track.py',
              'rlsetup/critic/style_data.py', 'rlsetup/critic/geometry.py', 'rlsetup/critic/sources/base.py')


def code_digest(adapter=None) -> str:
    """sha256 of CODE_FILES and, when given, the source file of the adapter's class (its
    `style_inputs` defines the time base and resampling)."""
    files = [REPO_ROOT / f for f in CODE_FILES]
    if adapter is not None:
        files.append(Path(inspect.getsourcefile(type(adapter))))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


class _StubTrack:
    """A track without geometry for `style_vector`: length only (per-km numbers), no corners."""

    def __init__(self, length_m: float):
        self.length_m = float(length_m)

    @staticmethod
    def find_corners():
        return []


def lap_numbers(si, tg: TrackGeometry, fallback_length_m: float):
    """(12 numbers in KEYS order, geometry_reason) of one StyleInput that passed R3."""
    lap = {'time_s': si.time_s, 'throttle': si.throttle, 'brake': si.brake, 'steer_deg': si.steer_deg}
    if tg is not None and tg.chosen is not None:
        geom, res = tg.chosen
        q = geom.line.q(res.progress(si.distance_m))
        v = style_vector(lap, geom.track, corners=geom.corners, q=q)
        return np.array([v[k] for k in KEYS], float), OK
    nan = np.full(len(si.time_s), np.nan)
    q = {'progress_m': nan, 'lateral_m': nan, 'left_width_m': nan, 'right_width_m': nan}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = style_vector(lap, _StubTrack(fallback_length_m), corners=[], q=q)
    out = np.array([v[k] for k in KEYS], float)
    out[[KEYS.index(k) for k in GEOMETRY_KEYS]] = np.nan
    return out, NO_GEOMETRY if tg is None else tg.reason


@dataclass
class LapStyles:
    """Per telemetry lap (all laps of the data, in lap-key order)."""
    keys: np.ndarray                # lap key
    game: np.ndarray                # game name (object)
    track: np.ndarray               # game track name (object)
    driver: np.ndarray              # integer id
    values: np.ndarray              # (n, 12) KEYS order; NaN where not computed
    lap_reason: np.ndarray          # OK or the R3 / missing-samples code (object)
    geometry_reason: np.ndarray     # OK, NO_GEOMETRY or a gate code; '' when the lap has no numbers
    median_spacing_m: np.ndarray
    time_closure_rel: np.ndarray    # t_end / lap_time - 1
    steer_kappa_corr: np.ndarray    # signed corr at the fitted alignment (< 0: steer + = right); NaN ungated
    is_placeholder: np.ndarray      # True where the L_* come from a placeholder line (always, today)
    line_kind: np.ndarray           # geometry.TUM_LINE / CENTRE_LINE / ''
    n_dropped_ties: np.ndarray = None   # steering samples dropped as distance ties (QA M5)
    cache_hits: int = 0

    @property
    def ok(self) -> np.ndarray:
        return self.lap_reason == OK


def cache_env(adapter) -> str:
    """16-hex digest of everything the cached numbers depend on besides the lap and its geometry: the
    code (`code_digest`), the numpy / scipy versions (gaussian_filter1d, cKDTree, interp; QA M4) and the
    style-rl commit. Each value is a sub-directory of the cache; other sub-directories are stale."""
    import scipy
    h = hashlib.sha256(json.dumps({'code': code_digest(adapter), 'numpy': np.__version__, 'scipy': scipy.__version__,
                                   'style_rl': STYLE_RL_COMMIT}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def prune_cache(cache_dir: Path, keep_env: str) -> int:
    """Delete stale cache entries (QA M4): sub-directories named like another `cache_env` digest and
    loose `<sha256>.json` files of the earlier flat layout. Only files with those name patterns are
    touched. Returns the number of files removed."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return 0
    hexchars = set('0123456789abcdef')
    removed = 0
    for entry in cache_dir.iterdir():
        if entry.is_dir() and len(entry.name) == 16 and set(entry.name) <= hexchars and entry.name != keep_env:
            files = [f for f in entry.iterdir() if f.suffix in ('.json', '.tmp') and set(f.stem) <= hexchars | set('_')]
            for f in files:
                f.unlink()
            removed += len(files)
            if not any(entry.iterdir()):
                entry.rmdir()
        elif entry.is_file() and entry.suffix == '.json' and len(entry.stem) == 64 and set(entry.stem) <= hexchars:
            entry.unlink()
            removed += 1
    return removed


def lap_styles(data, track_map: dict, cfg: Config = Config(), *, adapter, use_cache: bool = True,
               prune: bool = True, log=print):
    """(LapStyles, {source track: TrackGeometry}) for every lap of `data` (a `dataset.LapSource`).
    `track_map`: source track name -> geometry stem. `adapter` (required; QA M12) provides
    `style_inputs(lap, cfg)` and `steer_kappa_sign` (the gate's sign check, QA M2). With `use_cache`,
    numbers are read from / written to `cache_dir/<cache_env>/`; `prune` deletes stale entries first.
    `use_cache=False` neither reads nor writes the cache."""
    keys = sorted(data.laps)
    inputs = {k: adapter.style_inputs(data.laps[k], cfg.style_input) if data.laps[k].has_samples else None
              for k in keys}
    tracks = sorted({data.laps[k].track for k in keys})
    geoms, gated_keys = {}, {}
    for t in tracks:
        lk = [k for k in keys if data.laps[k].track == t and inputs[k] is not None and inputs[k].ok]
        laps = [(inputs[k].raw_distance_m, inputs[k].raw_steer_deg) for k in lk]
        gated_keys[t] = lk
        geoms[t] = gate_track(track_map[t], laps, cfg.geometry, adapter.steer_kappa_sign,
                              groups=[data.laps[k].game for k in lk]) if t in track_map else TrackGeometry(t, [])
    fallback = {t: float(np.median([inputs[k].lap_length_m for k in gated_keys[t]] or [np.nan])) for t in tracks}
    env = cache_env(adapter)
    cache_dir = Path(cfg.style_input.cache_dir)
    if use_cache and prune:
        n_pruned = prune_cache(cache_dir, env)
        if n_pruned:
            log(f'style cache: {n_pruned} stale entries removed')
    cache_dir = cache_dir / env
    n = len(keys)
    out = LapStyles(keys=np.array(keys, int), game=np.array([data.laps[k].game for k in keys], object),
                    track=np.array([data.laps[k].track for k in keys], object),
                    driver=np.array([data.laps[k].driver for k in keys], int), values=np.full((n, len(KEYS)), np.nan),
                    lap_reason=np.full(n, LAP_NO_GRAPH, object), geometry_reason=np.full(n, '', object),
                    median_spacing_m=np.full(n, np.nan), time_closure_rel=np.full(n, np.nan),
                    steer_kappa_corr=np.full(n, np.nan), is_placeholder=np.zeros(n, bool),
                    line_kind=np.full(n, '', object), n_dropped_ties=np.zeros(n, int))
    corr_of = {}
    for t, tg in geoms.items():
        if tg.chosen is not None:
            r = tg.chosen[1]
            if len(r.lap_index) != len(r.lap_corr) or len(r.lap_index) != len(gated_keys[t]):
                raise AssertionError(f'{t}: gate used {len(r.lap_index)} of {len(gated_keys[t])} laps')   # QA M3
            corr_of.update({gated_keys[t][j]: c for j, c in zip(r.lap_index, r.lap_corr)})
    for i, k in enumerate(keys):
        si = inputs[k]
        if si is None:
            continue
        out.lap_reason[i] = si.reason
        out.median_spacing_m[i] = si.median_spacing_m
        out.n_dropped_ties[i] = si.n_dropped_ties
        out.time_closure_rel[i] = si.t_end_s / si.lap_time_s - 1 if si.lap_time_s else np.nan
        if not si.ok:
            continue
        tg = geoms[data.laps[k].track]
        geo = {'reason': tg.reason}
        if tg.chosen is not None:
            g, r = tg.chosen
            geo.update(digest=g.digest(), shift=repr(r.shift_m), scale=repr(r.scale), line=g.line_kind)
            out.is_placeholder[i], out.line_kind[i] = True, g.line_kind
            out.steer_kappa_corr[i] = corr_of[k]
        else:
            geo['fallback_length_m'] = repr(fallback[data.laps[k].track])
        key = hashlib.sha256(json.dumps({'lap': si.digest, 'lap_time': repr(si.lap_time_s), 'geo': geo,
                                         'cfg': asdict(cfg.style_input)}, sort_keys=True).encode()).hexdigest()
        path = cache_dir / f'{key}.json'
        cached = _read_cache(path) if use_cache else None
        if cached is not None:
            values, reason = cached
            out.cache_hits += 1
        else:
            values, reason = lap_numbers(si, tg, fallback[data.laps[k].track])
            if use_cache:
                _write_cache(path, values, reason)
        out.values[i], out.geometry_reason[i] = values, reason
    log(f'style: {int(out.ok.sum())} of {n} laps with numbers ({out.cache_hits} from cache); '
        f'rejected: { {r: int((out.lap_reason == r).sum()) for r in sorted(set(out.lap_reason) - {OK})} }')
    return out, geoms


def _read_cache(path: Path):
    try:
        d = json.loads(path.read_text())
        values = np.array([np.nan if v is None else v for v in d['values']], float)
        return (values, d['geometry_reason']) if len(values) == len(KEYS) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, values, reason):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({'values': [None if not np.isfinite(v) else float(v) for v in values],
                       'geometry_reason': reason, 'keys': list(KEYS)})
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


@dataclass
class CellStyles:
    """Per cell of `Frame.cells` (game index, driver id, track index): median over its laps with numbers."""
    values: np.ndarray              # (n_cells, 12), NaN where no lap of the cell has numbers
    n_laps: np.ndarray              # laps with numbers per cell
    reason: np.ndarray              # OK, or the lap_reason of the cell's laps when none has numbers
    geometry_reason: np.ndarray     # geometry reason of the cell's track ('' when no numbers)

    @property
    def has_style(self) -> np.ndarray:
        return self.n_laps > 0


def cell_styles(frame, laps: LapStyles) -> CellStyles:
    """S3 cells: median over the cell's laps (style-rl's per (driver, track) rule)."""
    row = {k: i for i, k in enumerate(laps.keys)}
    nc = len(frame.cells)
    vals, n_laps = np.full((nc, len(KEYS)), np.nan), np.zeros(nc, int)
    reason, geo = np.full(nc, OK, object), np.full(nc, '', object)
    for c, lap_keys in enumerate(frame.cell_laps):
        idx = [row[k] for k in lap_keys]
        good = [i for i in idx if laps.ok[i]]
        n_laps[c] = len(good)
        if good:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)      # all-NaN geometry columns stay NaN
                vals[c] = np.nanmedian(laps.values[good], axis=0)
            geo[c] = laps.geometry_reason[good[0]]
        else:
            reason[c] = ','.join(sorted({laps.lap_reason[i] for i in idx}))
    return CellStyles(vals, n_laps, reason, geo)


def style_summary(values: np.ndarray) -> dict:
    """{key: {'n', 'median', 'p1', 'p99'}} over the finite rows of each column (KEYS order)."""
    out = {}
    for j, k in enumerate(KEYS):
        x = values[:, j][np.isfinite(values[:, j])]
        out[k] = {'n': int(len(x)), 'median': float(np.median(x)) if len(x) else float('nan'),
                  'p1': float(np.percentile(x, 1)) if len(x) else float('nan'),
                  'p99': float(np.percentile(x, 99)) if len(x) else float('nan')}
    return out


class StyleTransform:
    """R4 input transform of the style numbers, fitted on training cells only.

    The 9 B/T/S numbers enter: log of the six rate numbers (`log_keys`, floored at `log_floor`), then
    robust z = (x - median) / (IQR / iqr_to_sd) with median and IQR of the training cells (finite
    values only), clipped to +-z_clip. The numbers themselves are style-rl's (Decision 2). L_* are held
    out (C2_review [ARCH]): their slot is kept, `mask` is False there and `transform` writes 0. A NaN
    input (e.g. S_corner_peak_angle on an ungated track) stays NaN; C2b decides the imputation."""

    def __init__(self, cfg: StyleTransformConfig = StyleTransformConfig()):
        self.cfg = cfg
        self.mask = np.array([k not in cfg.masked_keys for k in KEYS])
        self.log_cols = np.array([k in cfg.log_keys for k in KEYS])
        self.center = self.scale = None

    def _pre(self, values):
        x = np.array(values, float, copy=True).reshape(-1, len(KEYS))
        x[:, self.log_cols] = np.log(np.maximum(x[:, self.log_cols], self.cfg.log_floor))
        return x

    def fit(self, values: np.ndarray, train_idx) -> 'StyleTransform':
        x = self._pre(np.asarray(values)[np.asarray(train_idx)])
        self.center, self.scale = np.zeros(len(KEYS)), np.ones(len(KEYS))
        for j in np.flatnonzero(self.mask):
            col = x[:, j][np.isfinite(x[:, j])]
            if not len(col):
                raise ValueError(f'no training cell has a finite {KEYS[j]}')
            q1, med, q3 = np.percentile(col, [25, 50, 75])
            self.center[j] = med
            self.scale[j] = (q3 - q1) / self.cfg.iqr_to_sd if q3 > q1 else 1.
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.center is None:
            raise RuntimeError('StyleTransform.fit first')
        z = np.clip((self._pre(values) - self.center) / self.scale, -self.cfg.z_clip, self.cfg.z_clip)
        z[:, ~self.mask] = 0.
        return z

    def state(self) -> dict:
        return {'keys': list(KEYS), 'mask': self.mask.tolist(), 'log': self.log_cols.tolist(),
                'center': self.center.tolist(), 'scale': self.scale.tolist(), 'cfg': asdict(self.cfg)}


def track_feature_table(track_names, geoms: dict, cfg: Config = Config()) -> np.ndarray:
    """(n_tracks, 5) raw z (TRACK_FEATURES) from the gated centre line of each track; NaN rows for
    tracks without a gated geometry (R2 applies the gate to z as well, C1-R15)."""
    rows = []
    for t in track_names:
        tg = geoms.get(t)
        rows.append(track_features(tg.chosen[0].xy, cfg.track) if tg is not None and tg.chosen is not None
                    else np.full(len(TRACK_FEATURES), np.nan))
    return np.array(rows, float).reshape(-1, len(TRACK_FEATURES))


def track_z_table(track_names, geoms: dict, cfg: Config = Config()):
    """C2b_addendum B3: (z (n_tracks, 5) raw TRACK_FEATURES, provisional (n_tracks,) bool, source
    [str per track]). A track with a gated geometry uses it (`track_feature_table`). Otherwise z only
    needs the layout, not the alignment: the first candidate that passes the length test and has
    median |corr| >= `GeometryConfig.z_min_abs_corr` gives z, flagged provisional. Else NaN. The style
    geometry numbers keep the full R2 gate (this does not change them)."""
    z = track_feature_table(track_names, geoms, cfg)
    provisional = np.zeros(len(track_names), bool)
    source = []
    for i, t in enumerate(track_names):
        tg = geoms.get(t)
        if tg is not None and tg.chosen is not None:
            source.append(f'{tg.chosen[0].kind}:gated')
            continue
        cand = next(((g, r) for g, r in (tg.results if tg is not None else [])
                     if r.length_ok and r.median_abs_corr >= cfg.geometry.z_min_abs_corr), None)
        if cand is None:
            source.append('none')
            continue
        z[i] = track_features(cand[0].xy, cfg.track)
        provisional[i] = True
        source.append(f'{cand[0].kind}:provisional')
    return z, provisional, source


__all__ = ['CellStyles', 'GEOMETRY_KEYS', 'LapStyles', 'StyleTransform', 'cell_styles', 'code_digest', 'lap_numbers',
           'cache_env', 'lap_styles', 'prune_cache', 'style_summary', 'track_feature_table', 'track_z_table']
