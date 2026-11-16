"""C2b_addendum B13 (C1 S4, C1-R12): knob groups of the group critics.

PCA groups: on the training rows A deduplicated by (track, setup hash), knobs standardised in-fold,
the top-k eigenvectors of the correlation matrix rotated by varimax; each knob joins the component with
its largest |rotated loading|. Components that receive no knob are dropped. Physics groups: fixed
(c2_config.PHYSICS_GROUPS). `adjusted_rand` compares two partitions (fold-to-fold stability).
"""
import numpy as np
from .c2_config import PHYSICS_GROUPS
from .config import KNOB_ORDER


def varimax(loadings: np.ndarray, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Kaiser varimax rotation of (p, k) loadings (no normalisation), deterministic."""
    p, k = loadings.shape
    R = np.eye(k)
    d = 0.
    for _ in range(iters):
        L = loadings @ R
        u, s, vt = np.linalg.svd(loadings.T @ (L ** 3 - L @ np.diag((L ** 2).sum(axis=0)) / p))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return loadings @ R


def pca_groups(p_train: np.ndarray, dedup_keys=None, k: int = 4) -> tuple:
    """Tuple of knob-index tuples (sorted; groups ordered by their smallest knob index)."""
    x = np.asarray(p_train, float)
    if dedup_keys is not None:
        _, first = np.unique(np.asarray(dedup_keys), return_index=True)
        x = x[np.sort(first)]
    sd = x.std(axis=0)
    xs = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.)
    corr = xs.T @ xs / len(xs)
    val, vec = np.linalg.eigh(corr)
    order = np.argsort(val)[::-1][:k]
    load = vec[:, order] * np.sqrt(np.maximum(val[order], 0))
    rot = varimax(load)
    lab = np.argmax(np.abs(rot), axis=1)
    groups = [tuple(int(j) for j in np.flatnonzero(lab == c)) for c in range(k)]
    return tuple(sorted((g for g in groups if g), key=lambda g: g[0]))


def physics_groups() -> tuple:
    return tuple(tuple(KNOB_ORDER.index(n) for n in g) for g in PHYSICS_GROUPS)


def labels_of(groups, n: int = len(KNOB_ORDER)) -> np.ndarray:
    lab = np.full(n, -1)
    for c, g in enumerate(groups):
        lab[list(g)] = c
    return lab


def adjusted_rand(a, b) -> float:
    """Adjusted Rand index of two labelings (Hubert & Arabie)."""
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2.
    s_ij, s_a, s_b = comb(table).sum(), comb(table.sum(1)).sum(), comb(table.sum(0)).sum()
    expected = s_a * s_b / comb(len(a))
    mx = (s_a + s_b) / 2.
    return 1. if mx == expected else float((s_ij - expected) / (mx - expected))


def group_names(groups) -> list:
    return [[KNOB_ORDER[j] for j in g] for g in groups]


__all__ = ['adjusted_rand', 'group_names', 'labels_of', 'pca_groups', 'physics_groups', 'varimax']
