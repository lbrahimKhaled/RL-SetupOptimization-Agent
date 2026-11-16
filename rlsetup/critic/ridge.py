"""C2b_addendum B15: closed-form baselines (training rows only, `nuisance.ridge_backfit`).

- nuisance-only: the B7 terms (A: no style; exact rows: + m(s~)), no p;
- ridge on A: nuisance + driver + p + p x z^ (alpha 1 on the p blocks);
- ridge on exact rows: nuisance + m(s~) + p + s~ x PC4(p) (alpha 10), PCs of p fitted in-fold;
- Barcelona variants (E4): see `ridge_A` arguments (`delta_alpha` adds p x 1[Barcelona]).
Each returns {'full': prediction of the test rows, 'setup': the p-dependent part (for pairs)}.
"""
import numpy as np
from .c2_config import BARCELONA, C2bConfig
from .nuisance import NuisanceEncoder, ridge_backfit
from .style_transform import TrackStyleTransform
from .train import context, fit_track_scaler


def _fit_predict(d, train, test, enc, extra_tr, extra_te, alpha, names, s_tr=None, s_te=None, cfg=C2bConfig(),
                 setup_names=()):
    x_tr = enc.encode(d, train, s_tilde=s_tr, extra=extra_tr, extra_alpha=alpha, extra_names=names)
    x_te = enc.encode(d, test, s_tilde=s_te, extra=extra_te, extra_alpha=alpha, extra_names=names)
    fit = ridge_backfit(x_tr, d.gap[train], iters=max(cfg.nuisance.backfit_iters, 300), tol=cfg.nuisance.backfit_tol)
    mask = np.array([any(n.startswith(s) for s in setup_names) for n in x_te.dense_names]) if setup_names else None
    full = fit.predict(x_te)
    setup = x_te.dense @ np.where(mask, fit.beta, 0.) if mask is not None else np.zeros(len(test))
    return {'full': full, 'setup': setup, 'iters': fit.iters}


def _not_estimable(test) -> dict:
    """No training rows (e.g. the Barcelona-only variant on data without Barcelona): NaN predictions,
    which every metric treats as untested rows."""
    n = len(np.asarray(test))
    return {'full': np.full(n, np.nan), 'setup': np.full(n, np.nan), 'iters': 0}


def nuisance_only(d, train, test, cfg: C2bConfig = C2bConfig(), style: bool = False) -> dict:
    train, test = np.asarray(train), np.asarray(test)
    if not len(train):
        return _not_estimable(test)
    enc = NuisanceEncoder(cfg.nuisance, style=style).fit(d, train)
    s_tr = s_te = None
    if style:
        tf = TrackStyleTransform().fit(d.style_raw, d.track_key, train)
        s_tr, s_te = tf.transform(d.style_raw[train], d.track_key[train])[0], tf.transform(d.style_raw[test], d.track_key[test])[0]
    return _fit_predict(d, train, test, enc, None, None, None, None, s_tr, s_te, cfg)


def _p_blocks(d, rows, scaler, delta: bool):
    z, _ = context(d, rows, scaler)
    p = d.p[rows]
    blocks = [p, (p[:, :, None] * z[:, None, :]).reshape(len(rows), p.shape[1] * z.shape[1])]
    names = [f'p:{j}' for j in range(15)] + [f'pz:{j}' for j in range(75)]
    if delta:
        blocks.append(p * np.asarray(d.track_key[rows] == BARCELONA, float)[:, None])
        names += [f'pd:{j}' for j in range(15)]
    return np.column_stack(blocks), names


def ridge_A(d, train, test, cfg: C2bConfig = C2bConfig(), alpha: float = None, with_pz: bool = True,
            delta_alpha: float = None) -> dict:
    """B15 ridge on A (nuisance + driver + p + p x z^, alpha 1). `with_pz=False`: p only (the
    Barcelona-only variant). `delta_alpha`: adds the Barcelona deviation p x 1[spain] with that alpha."""
    train, test = np.asarray(train), np.asarray(test)
    if not len(train):
        return _not_estimable(test)
    a = cfg.eval.ridge_alpha_a if alpha is None else alpha
    enc = NuisanceEncoder(cfg.nuisance).fit(d, train)
    scaler = fit_track_scaler(d, train)
    xtr, names = _p_blocks(d, train, scaler, delta_alpha is not None)
    xte, _ = _p_blocks(d, test, scaler, delta_alpha is not None)
    alphas = np.full(len(names), a)
    if delta_alpha is not None:
        alphas[[i for i, n in enumerate(names) if n.startswith('pd:')]] = delta_alpha
    keep = [i for i, n in enumerate(names) if with_pz or not n.startswith('pz:')]
    return _fit_predict(d, train, test, enc, xtr[:, keep], xte[:, keep], alphas[keep], [names[i] for i in keep],
                        cfg=cfg, setup_names=('p:', 'pz:', 'pd:'))


def pca_fit(x, k):
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 0, sd, 1.)
    xs = (x - mu) / sd
    val, vec = np.linalg.eigh(xs.T @ xs / max(len(xs) - 1, 1))
    V = vec[:, np.argsort(val)[::-1][:k]]
    return lambda y: ((y - mu) / sd) @ V


def ridge_exact(d, train, test, cfg: C2bConfig = C2bConfig()) -> dict:
    """B15 ridge on exact rows: nuisance + m(s~) + p + s~ x PC4(p), alpha 10 (PCs of the training p)."""
    train, test = np.asarray(train), np.asarray(test)
    if not len(train):
        return _not_estimable(test)
    a = cfg.eval.ridge_alpha_exact
    enc = NuisanceEncoder(cfg.nuisance, style=True).fit(d, train)
    tf = TrackStyleTransform().fit(d.style_raw, d.track_key, train)
    s_tr, s_te = tf.transform(d.style_raw[train], d.track_key[train])[0], tf.transform(d.style_raw[test], d.track_key[test])[0]
    pcs = pca_fit(d.p[train], cfg.eval.ridge_pcs)

    def blocks(rows, s):
        pc = pcs(d.p[rows])
        return np.column_stack([d.p[rows], (s[:, :, None] * pc[:, None, :]).reshape(len(rows), s.shape[1] * pc.shape[1])])
    names = [f'p:{j}' for j in range(15)] + [f'sp:{j}' for j in range(s_tr.shape[1] * cfg.eval.ridge_pcs)]
    return _fit_predict(d, train, test, enc, blocks(train, s_tr), blocks(test, s_te), np.full(len(names), a), names,
                        s_tr, s_te, cfg, setup_names=('p:', 'sp:'))


__all__ = ['nuisance_only', 'pca_fit', 'ridge_A', 'ridge_exact']
