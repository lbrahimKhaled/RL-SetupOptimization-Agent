"""C2b_addendum B14: metrics, person-cluster bootstrap, ranking pairs and the stratified style shuffle.

- R2 gain = (SSE_nuis - SSE_model) / SST_w, SST_w = sum_test (y - mean_train(y | game, track, session))^2.
- MAE in % and in seconds (% x the row's B1 group median lap / 100).
- Within-(game, track) Spearman of prediction vs label, mean over groups weighted by n (point estimate).
- Pairs (rows A): two rows of one (game, driver, track, session) with different setup hash and different
  gap; correct when the predictions order them like the labels; a tie in prediction counts 0.5. Pairs
  never cross sessions (R6).
- CIs: person-cluster bootstrap, B = 2,000 (multinomial resampling of persons), paired when two models
  are compared (same resamples). A metric is first averaged over the repeats per row / pair, then the
  ratio of cluster sums is bootstrapped. MDE = mde_factor x bootstrap SE.
"""
import numpy as np


def pairs_of(game, driver, track, session, setup_hash, y, rows=None) -> np.ndarray:
    """(m, 2) row indices of the ranking pairs among `rows` (default all)."""
    rows = np.arange(len(y)) if rows is None else np.asarray(rows)
    key = np.stack([game[rows], driver[rows], track[rows], session[rows]], 1)
    order = np.lexsort(key.T[::-1])
    r, k = rows[order], key[order]
    brk = np.flatnonzero(np.any(np.diff(k, axis=0) != 0, axis=1)) + 1
    out = []
    for grp in np.split(r, brk):
        if len(grp) < 2:
            continue
        a, b = np.triu_indices(len(grp), 1)
        ia, ib = grp[a], grp[b]
        ok = (setup_hash[ia] != setup_hash[ib]) & (y[ia] != y[ib])
        if ok.any():
            out.append(np.stack([ia[ok], ib[ok]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def pair_outcomes(pairs: np.ndarray, pred, y) -> np.ndarray:
    """1 = same order as the label, 0 = reversed, 0.5 = tie in prediction."""
    dp = pred[pairs[:, 0]] - pred[pairs[:, 1]]
    dy = y[pairs[:, 0]] - y[pairs[:, 1]]
    return np.where(dp == 0, .5, (np.sign(dp) == np.sign(dy)).astype(float))


def sst_w(y, group, train_mask, test_rows):
    """Per test row (y - mean of y over the training rows of its group)^2; a group without training
    rows uses the overall training mean."""
    y = np.asarray(y, float)
    tr = np.flatnonzero(train_mask)
    means = {g: y[tr][group[tr] == g].mean() for g in np.unique(group[tr])}
    overall = y[tr].mean()
    m = np.array([means.get(g, overall) for g in group[test_rows]])
    return (y[test_rows] - m) ** 2


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(float)
    rb = np.argsort(np.argsort(b, kind='stable'), kind='stable').astype(float)
    # average ranks for ties
    for x, r in ((a, ra), (b, rb)):
        u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            s = np.bincount(inv, r)
            r[:] = (s / cnt)[inv]
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def within_spearman(pred, y, group) -> float:
    """Mean over groups of the Spearman of (pred, y), weighted by group size (groups with >= 3 rows)."""
    num = den = 0.
    for g in np.unique(group):
        m = group == g
        if m.sum() >= 3:
            r = spearman(pred[m], y[m])
            if np.isfinite(r):
                num, den = num + m.sum() * r, den + m.sum()
    return num / den if den else float('nan')


class ClusterBoot:
    """Shared person-cluster resamples: `weights` (B, n_clusters) multinomial counts."""

    def __init__(self, clusters, n_boot: int = 2000, seed: int = 0):
        self.u, self.inv = np.unique(np.asarray(clusters), return_inverse=True)
        rng = np.random.default_rng(seed)
        k = len(self.u)
        self.weights = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)]).astype(float)

    def index(self, clusters) -> np.ndarray:
        pos = np.searchsorted(self.u, clusters)
        if np.any(self.u[np.minimum(pos, len(self.u) - 1)] != clusters):
            raise ValueError('cluster not in the bootstrap universe')
        return pos

    def ratio(self, num, den, clusters) -> np.ndarray:
        """(B,) bootstrap distribution of sum(num) / sum(den) over resampled clusters."""
        ci = self.index(np.asarray(clusters))
        sn = np.bincount(ci, np.asarray(num, float), len(self.u))
        sd = np.bincount(ci, np.asarray(den, float), len(self.u))
        with np.errstate(invalid='ignore', divide='ignore'):
            return (self.weights @ sn) / (self.weights @ sd)


def summarise(point, dist, mde_factor: float = 2.8) -> dict:
    d = np.asarray(dist)[np.isfinite(dist)]
    if not len(d):
        return {'point': float(point), 'ci': [None, None], 'se': None, 'mde': None, 'n_boot': 0}
    lo, hi = np.percentile(d, [2.5, 97.5])
    se = float(d.std(ddof=1)) if len(d) > 1 else 0.
    return {'point': float(point), 'ci': [float(lo), float(hi)], 'se': se, 'mde': mde_factor * se, 'n_boot': int(len(d))}


def ratio_metric(boot: ClusterBoot, num, den, clusters, mde_factor=2.8) -> dict:
    num, den = np.asarray(num, float), np.asarray(den, float)
    point = num.sum() / den.sum() if den.sum() else float('nan')
    return summarise(point, boot.ratio(num, den, clusters), mde_factor) | {'n': int(len(num))}


def strata_shuffle(values: np.ndarray, strata, rows, seed: int) -> np.ndarray:
    """Copy of `values` with the rows `rows` permuted within their stratum (`strata` per row, any
    hashable codes); rows outside `rows` and singleton strata are unchanged. Seed-deterministic."""
    out = np.array(values, copy=True)
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    st = np.asarray(strata)[rows]
    _, inv = np.unique(st, return_inverse=True, axis=0) if st.ndim > 1 else np.unique(st, return_inverse=True)
    inv = inv.ravel()
    for k in range(inv.max() + 1 if len(inv) else 0):
        r = rows[inv == k]
        if len(r) > 1:
            out[r] = values[rng.permutation(r)]
    return out


def singleton_rows(strata, rows) -> int:
    st = np.asarray(strata)[np.asarray(rows)]
    _, inv, cnt = (np.unique(st, return_inverse=True, return_counts=True, axis=0) if st.ndim > 1
                   else np.unique(st, return_inverse=True, return_counts=True))
    return int((cnt[inv.ravel()] == 1).sum())


__all__ = ['ClusterBoot', 'pair_outcomes', 'pairs_of', 'ratio_metric', 'singleton_rows', 'spearman', 'sst_w',
           'strata_shuffle', 'summarise', 'within_spearman']
