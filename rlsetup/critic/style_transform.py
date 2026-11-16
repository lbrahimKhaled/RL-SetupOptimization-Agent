"""C2b_addendum B4: the style input of critics 1 and 2 (transform only; the numbers stay style-rl's).

12 raw style-rl numbers (KEYS order) -> s~ (9, the B/T/S numbers; the three L_* are held out):
1. log of the six rate numbers (floored at `log_floor`);
2. track centring: subtract the median of that track's training cells (both games pooled); a track
   with fewer than `min_track_cells` training cells (or unknown at query) uses the global training
   median and is flagged 'global';
3. robust z with the median and IQR / 1.349 of the centred training cells, clipped to +-3;
4. NaN -> 0 (the training median; QA M9). No missingness flag (B4).
Everything is fitted on training cells only. Tracks are canonical keys (strings).
"""
from dataclasses import asdict
import warnings
import numpy as np
from ..style import KEYS
from .config import StyleTransformConfig

MIN_TRACK_CELLS = 5                     # B4


class TrackStyleTransform:
    """B4 transform, fitted on training cells; `transform` -> (s~ (n, 9), centre per row 'track'/'global')."""

    def __init__(self, cfg: StyleTransformConfig = StyleTransformConfig(), min_track_cells: int = MIN_TRACK_CELLS):
        self.cfg, self.min_track_cells = cfg, int(min_track_cells)
        self.keys = [k for k in KEYS if k not in cfg.masked_keys]
        self.cols = np.array([KEYS.index(k) for k in self.keys])
        self.log_cols = np.array([k in cfg.log_keys for k in self.keys])
        self.track_median, self.global_median, self.center, self.scale = {}, None, None, None
        self.n_train_cells = {}

    def _pre(self, values) -> np.ndarray:
        x = np.array(values, float, copy=True).reshape(-1, len(KEYS))[:, self.cols]
        x[:, self.log_cols] = np.log(np.maximum(x[:, self.log_cols], self.cfg.log_floor))
        return x

    @staticmethod
    def _nanmedian(x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN columns stay NaN
            return np.nanmedian(x, axis=0)

    def _centre(self, x, tracks):
        tracks = np.asarray(tracks, object)
        out, where = np.empty_like(x), np.empty(len(x), object)
        for i, t in enumerate(tracks):
            med = self.track_median.get(t)
            where[i] = 'track' if med is not None else 'global'
            out[i] = x[i] - (med if med is not None else self.global_median)
        return out, where

    def fit(self, values, tracks, train_idx) -> 'TrackStyleTransform':
        """`values` (n, 12) raw, `tracks` (n,) canonical keys; only rows `train_idx` are used."""
        idx = np.asarray(train_idx)
        x, tr = self._pre(np.asarray(values)[idx]), np.asarray(tracks, object)[idx]
        if not len(x):
            raise ValueError('no training cells')
        self.global_median = self._nanmedian(x)
        self.track_median, self.n_train_cells = {}, {}
        for t in sorted(set(tr)):
            m = tr == t
            self.n_train_cells[t] = int(m.sum())
            if m.sum() >= self.min_track_cells:
                self.track_median[t] = self._nanmedian(x[m])
        c, _ = self._centre(x, tr)
        self.center, self.scale = np.zeros(len(self.keys)), np.ones(len(self.keys))
        for j in range(len(self.keys)):
            col = c[:, j][np.isfinite(c[:, j])]
            if not len(col):
                raise ValueError(f'no training cell has a finite {self.keys[j]}')
            q1, med, q3 = np.percentile(col, [25, 50, 75])
            self.center[j] = med
            self.scale[j] = (q3 - q1) / self.cfg.iqr_to_sd if q3 > q1 else 1.
        return self

    def transform(self, values, tracks):
        """(s~ (n, 9) finite, centre (n,) 'track' or 'global')."""
        if self.center is None:
            raise RuntimeError('TrackStyleTransform.fit first')
        c, where = self._centre(self._pre(values), tracks)
        z = np.clip((c - self.center) / self.scale, -self.cfg.z_clip, self.cfg.z_clip)
        return np.where(np.isfinite(z), z, 0.), where

    def state(self) -> dict:
        return {'keys': self.keys, 'all_keys': list(KEYS), 'cfg': asdict(self.cfg), 'min_track_cells': self.min_track_cells,
                'track_median': {t: v.tolist() for t, v in self.track_median.items()},
                'global_median': self.global_median.tolist(), 'center': self.center.tolist(),
                'scale': self.scale.tolist(), 'n_train_cells': self.n_train_cells}

    @classmethod
    def from_state(cls, st: dict) -> 'TrackStyleTransform':
        cfg = StyleTransformConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in st['cfg'].items()})
        t = cls(cfg, st['min_track_cells'])
        nan = lambda v: np.array([np.nan if x is None else x for x in v], float)
        t.track_median = {k: nan(v) for k, v in st['track_median'].items()}
        t.global_median, t.center, t.scale = nan(st['global_median']), nan(st['center']), nan(st['scale'])
        t.n_train_cells = dict(st['n_train_cells'])
        return t


__all__ = ['MIN_TRACK_CELLS', 'TrackStyleTransform']
