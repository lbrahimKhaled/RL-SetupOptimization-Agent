"""C2b_addendum B14 evaluation on stored out-of-fold predictions.

Predictions are (R, n) arrays: one row per repeat, NaN where a row was not a test row of that repeat.
Every metric is a ratio of per-item sums (rows or pairs), averaged over the repeats per item first and
then bootstrapped over persons (`metrics.ClusterBoot`, one universe per evaluated subset; paired
comparisons share the resamples). Subsets: Barcelona first, then pooled, then per game (B18).
"""
import numpy as np
from .c2_config import BARCELONA
from .metrics import ClusterBoot, pair_outcomes, pairs_of, ratio_metric, within_spearman


def _avg(a):
    a = np.asarray(a, float)
    with np.errstate(invalid='ignore'):
        cnt = np.isfinite(a).sum(0)
        return np.where(cnt > 0, np.nansum(a, 0) / np.maximum(cnt, 1), np.nan)


class Evaluator:
    def __init__(self, d, n_boot: int = 2000, seed: int = 0, mde_factor: float = 2.8):
        self.d, self.n_boot, self.seed, self.mde = d, n_boot, seed, mde_factor
        self._boots = {}
        self._pairs = None

    # ---- subsets ------------------------------------------------------------------------------------
    def subsets(self, rows_mask) -> dict:
        d = self.d
        m = np.asarray(rows_mask, bool)
        out = {'barcelona': m & (d.track_key == BARCELONA), 'pooled': m}
        for gi, g in enumerate(d.games):
            out[g] = m & (d.game == gi)
        for si, sn in enumerate(d.sessions):                 # B22.2: per-session rows after the per-game rows
            out[f'session:{sn}'] = m & (d.session == si)
        return out

    def boot(self, persons) -> ClusterBoot:
        u = np.unique(persons)
        key = u.tobytes()
        if key not in self._boots:
            self._boots[key] = ClusterBoot(u, self.n_boot, self.seed)
        return self._boots[key]

    def pairs(self, rows_mask=None) -> np.ndarray:
        d = self.d
        if self._pairs is None:
            self._pairs = pairs_of(d.game, d.driver, d.track, d.session, d.setup_hash, d.gap)
        if rows_mask is None:
            return self._pairs
        m = np.asarray(rows_mask, bool)
        return self._pairs[m[self._pairs[:, 0]] & m[self._pairs[:, 1]]]

    # ---- per-item numbers ---------------------------------------------------------------------------
    def pair_items(self, setup, pairs) -> np.ndarray:
        """(n_pairs,) outcome averaged over the repeats where both rows have a prediction."""
        setup = np.atleast_2d(setup)
        out = []
        for s in setup:
            ok = np.isfinite(s[pairs[:, 0]]) & np.isfinite(s[pairs[:, 1]])
            o = np.full(len(pairs), np.nan)
            o[ok] = pair_outcomes(pairs[ok], s, self.d.gap)
            out.append(o)
        return _avg(out)

    def sq(self, pred):
        return (self.d.gap[None, :] - np.atleast_2d(pred)) ** 2

    # ---- metrics ------------------------------------------------------------------------------------
    def ratio(self, num, den, persons):
        ok = np.isfinite(num) & np.isfinite(den)
        if not ok.any():
            return {'point': None, 'ci': [None, None], 'se': None, 'mde': None, 'n': 0}
        b = self.boot(persons[ok])
        return ratio_metric(b, num[ok], den[ok], persons[ok], self.mde)

    def pair_acc(self, setup, rows_mask) -> dict:
        P = self.pairs(rows_mask)
        o = self.pair_items(setup, P)
        r = self.ratio(o, np.ones(len(o)), self.d.person[P[:, 0]])
        return r | {'n_pairs': int(np.isfinite(o).sum()), 'n_persons': int(len(np.unique(self.d.person[P[:, 0]][np.isfinite(o)])))}

    def pair_diff(self, setup_a, setup_b, rows_mask) -> dict:
        P = self.pairs(rows_mask)
        o = self.pair_items(setup_a, P) - self.pair_items(setup_b, P)
        return self.ratio(o, np.ones(len(o)), self.d.person[P[:, 0]]) | {'n_pairs': int(np.isfinite(o).sum())}

    def r2_gain(self, pred, nuis, sst, rows_mask) -> dict:
        m = np.asarray(rows_mask, bool)
        num = _avg(self.sq(nuis) - self.sq(pred))[m]
        den = _avg(sst)[m]
        return self.ratio(num, den, self.d.person[m])

    def r2_diff(self, pred_a, pred_b, sst, rows_mask) -> dict:
        """(SSE_b - SSE_a) / SST_w: > 0 when a predicts better."""
        m = np.asarray(rows_mask, bool)
        return self.ratio(_avg(self.sq(pred_b) - self.sq(pred_a))[m], _avg(sst)[m], self.d.person[m])

    def mae(self, pred, rows_mask, seconds: bool = False) -> dict:
        m = np.asarray(rows_mask, bool)
        a = _avg(np.abs(self.d.gap[None, :] - np.atleast_2d(pred)))
        if seconds:
            a = a * self.d.label_median_s / 100.
        return self.ratio(a[m], np.ones(m.sum()), self.d.person[m])

    def mae_diff(self, pred_a, pred_b, rows_mask) -> dict:
        m = np.asarray(rows_mask, bool)
        a = _avg(np.abs(self.d.gap[None, :] - np.atleast_2d(pred_a)) - np.abs(self.d.gap[None, :] - np.atleast_2d(pred_b)))
        return self.ratio(a[m], np.ones(m.sum()), self.d.person[m])

    def spearman(self, pred, rows_mask) -> float:
        m = np.asarray(rows_mask, bool) & np.isfinite(_avg(pred))
        gt = (self.d.game * len(self.d.track_names) + self.d.track)[m]
        return within_spearman(_avg(pred)[m], self.d.gap[m], gt)

    def block(self, pred, setup, nuis, sst, rows_mask, pairs: bool = True) -> dict:
        """The standard metric set per subset (Barcelona, pooled, per game)."""
        out = {}
        for name, m in self.subsets(rows_mask).items():
            if not m.any():
                continue
            r = {'n_rows': int(m.sum()), 'r2_gain': self.r2_gain(pred, nuis, sst, m), 'mae_pct': self.mae(pred, m),
                 'mae_s': self.mae(pred, m, True), 'mae_nuis_pct': self.mae(nuis, m), 'spearman_within': self.spearman(pred, m)}
            if pairs:
                r['pair_acc'] = self.pair_acc(setup, m)
            out[name] = r
        return out


def stack_preds(n, items) -> np.ndarray:
    """items: list per repeat of list of (rows, values) -> (R, n) with NaN elsewhere."""
    out = np.full((len(items), n), np.nan)
    for r, parts in enumerate(items):
        for rows, vals in parts:
            out[r, rows] = vals
    return out


__all__ = ['Evaluator', 'stack_preds']
