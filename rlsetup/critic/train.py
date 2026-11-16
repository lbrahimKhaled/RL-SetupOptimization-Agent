"""C2b training (C2b_addendum B7-B12): pretrain on rows A (critic 3, critic 2), fine-tune on exact rows
(critic 1, critic 2), the Barcelona residual head, twins by bootstrap, the paired p guard.

Everything fitted here (track scaler, nuisance levels, closed-form init, support box, style transform,
the PCA initialisation of W) sees training rows only. CPU, float32, deterministic: every fit takes an
explicit integer seed (`seed_of`) for its initialisation, batch order, guard draws and bootstrap.
"""
from dataclasses import dataclass, field, replace
import copy
import hashlib
import numpy as np
import torch
from torch import nn
from .c2_config import BARCELONA, BcnHeadConfig, C2bConfig, FineTuneConfig, GuardConfig, PretrainConfig
from .folds import inner_split
from .meta_prior import MetaPrior
from .model import BarcelonaHead, FullCritic, GroupCritic
from .nuisance import NuisanceEncoder, NuisanceModule, design_tensors, ridge_backfit
from .style_transform import TrackStyleTransform
from .track_features import TrackScaler


def seed_of(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def set_determinism(threads: int = 1):
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------
def fit_track_scaler(d, train_rows) -> TrackScaler:
    """B3: standardise z on the training tracks with a gated (non-provisional) z; clip +-3."""
    tracks = np.unique(d.track[train_rows])
    gated = tracks[~d.z_provisional[tracks] & np.isfinite(d.z_raw[tracks]).all(axis=1)]
    return TrackScaler().fit(d.z_raw, gated)


def context(d, rows, scaler: TrackScaler):
    """(z^ (n, 5), g^ (n, n_games)) of rows; z^ of a track without z is 0 (none in C2b after B2/B3)."""
    rows = np.asarray(rows)
    z = scaler.transform(d.z_raw, d.track[rows])
    g = np.zeros((len(rows), len(d.games)))
    g[np.arange(len(rows)), d.game[rows]] = 1.
    return np.where(np.isfinite(z), z, 0.), g


def boot_weights(keys, seed: int) -> np.ndarray:
    """Bootstrap by key (setup hash for critic 3, cell for critic 1, B12): draw the unique keys with
    replacement; a row's weight = how often its key was drawn."""
    keys = np.asarray(keys)
    u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(np.random.default_rng(seed).integers(0, len(u), len(u)), minlength=len(u))
    return cnt[inv.ravel()].astype(float)


def style_pca_init(s_train: np.ndarray, n_e: int = 2) -> np.ndarray:
    """B9: W = the top-n_e PCA loadings (unit eigenvectors of the covariance) of the training cells' s~;
    sign fixed so the largest |entry| of each row is positive."""
    x = s_train - s_train.mean(axis=0)
    val, vec = np.linalg.eigh(x.T @ x / max(len(x) - 1, 1))
    W = vec[:, np.argsort(val)[::-1][:n_e]].T.copy()
    for r in W:
        if r[np.argmax(np.abs(r))] < 0:
            r *= -1
    return W


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


@dataclass
class Fitted:
    """One fitted critic (or nuisance-only comparator) with everything needed to predict."""
    kind: str                       # critic3 | critic1 | critic2_pre | critic2 | c3_exact | c2pre_exact
    model: nn.Module
    nuis: NuisanceModule
    encoder: NuisanceEncoder
    scaler: TrackScaler
    style_tf: TrackStyleTransform = None
    support: tuple = None           # (low, high) per knob, training rows (C1-R9)
    bcn: BarcelonaHead = None
    info: dict = field(default_factory=dict)
    prior: MetaPrior = None         # B21: lambda Pi added to J (frozen; inherited by critic 1)


def refit_nuisance(f: 'Fitted', d, train_rows, cfg: C2bConfig = C2bConfig()) -> 'Fitted':
    """Final step of every fit (coder's reading, see the C2b report): with the critic frozen, the nuisance
    is refitted in closed form (B7 terms and penalties) on ALL training rows, unweighted. The jointly
    trained nuisance saw only the inner-fit rows (and, for twins, a bootstrap of them), so its R2 / MAE
    would not be comparable with the closed-form nuisance-only baseline on the same training rows; after
    the refit the R2 gain measures what the critic adds. q is unchanged (the nuisance is never in q)."""
    train_rows = np.asarray(train_rows)
    style = f.style_tf is not None
    sdev = getattr(f.encoder, 'session_dev', None) if f.encoder is not None else None
    enc = NuisanceEncoder(cfg.nuisance, style=style, session_dev=sdev).fit(d, train_rows)
    s = style_inputs(d, train_rows, f.style_tf)[0] if style else None
    x = enc.encode(d, train_rows, s_tilde=s)
    J = predict(f, d, train_rows, nuisance=False)['setup']
    fit = ridge_backfit(x, d.gap[train_rows], offset=J)
    f.nuis, f.encoder = NuisanceModule(x, fit), enc
    for q in f.nuis.parameters():
        q.requires_grad_(False)
    f.info['nuisance_refit_rows'] = int(len(train_rows))
    return f


# ---------------------------------------------------------------------------------------------------
# paired guard (B12 = C1 amendment item 1)
# ---------------------------------------------------------------------------------------------------
def guard_loss(model, p, z, g, low, high, cfg: GuardConfig, gen, e_pool=None):
    """0.1 mean((delta - 1)^2), delta = sum h(p_corrupt) - sum h(p_source) on the corrupted rows outside
    the training support; gradients through both terms. None when no row leaves the support."""
    n = p.shape[0]
    if n == 0:
        return None
    idx = torch.randint(0, n, (cfg.rows,), generator=gen)
    ps, zs, gs = p[idx], z[idx], g[idx]
    mask = torch.rand(ps.shape, generator=gen) < cfg.prob
    pc = torch.where(mask, torch.rand(ps.shape, generator=gen) * 2 - 1, ps)
    out = ((pc < low) | (pc > high)).any(dim=1)
    if not out.any():
        return None
    e = None
    if e_pool is not None:                       # critic 1: s sampled from training cells (B12)
        e = e_pool[torch.randint(0, e_pool.shape[0], (int(out.sum()),), generator=gen)]
    delta = model.setup_sum(pc[out], zs[out], gs[out], e) - model.setup_sum(ps[out], zs[out], gs[out], e)
    return cfg.weight * ((delta - cfg.target_pct) ** 2).mean()


# ---------------------------------------------------------------------------------------------------
# pretrain on rows A (critic 3, critic 2)
# ---------------------------------------------------------------------------------------------------
class ExtraLoss:
    """Generic hook for an additional pretrain loss term that depends on (p, track, game). (B21 itself is
    an additive prior on J, `pretrain(prior=...)`, not a loss term; this hook is kept for later terms.)
    Subclasses implement:

    - `spec() -> dict`: everything that defines the term (enters the resumable fit hash);
    - `bind(d, fit_rows, scaler, support) -> callable | None`: built per fit from the in-fold fit rows
      only (never the held-out or inner-validation rows); the callable gets
      `(model, batch: dict, gen: torch.Generator)` with batch keys p, z, g (tensors of the minibatch),
      track, game (int64 tensors), rows (global row indices) and returns a scalar tensor (percent^2
      scale, already weighted) or None.

    The term has its own torch generator (seeded from the fit seed), so the main batch order, the guard
    draws and the initialisation are bit-identical with and without it under the same folds and seeds.
    """

    def spec(self) -> dict:
        raise NotImplementedError

    def bind(self, d, fit_rows, scaler, support):
        raise NotImplementedError


def pretrain(d, train_rows, kind: str, groups=None, cfg: C2bConfig = C2bConfig(), seed: int = 0,
             weights=None, guard: bool = True, smoke: bool = False, salt='', extra: ExtraLoss | None = None,
             prior: MetaPrior = None, wd_grid=None, session_dev: bool = False) -> Fitted:
    """B8 (critic 3) / B10 pretrain (critic 2, e2 absent): Adam, batch 256, <= 500 epochs, patience 30,
    inner validation on 10 % of training cells, weight decay of the heads from the grid (lowest inner
    validation MSE). Nuisance (B7) trained jointly, initialised by the closed-form ridge. `weights`:
    bootstrap weights of `train_rows` (twins) or None. The paired guard (critic 3 only). `extra`: an
    optional `ExtraLoss` (a generic hook; None = the B8 objective exactly). `prior`: the B21 meta prior with
    its lambda (critic 3; critic 2 inherits the delivered critic 3's Pi frozen, B21.5; lambda Pi is added to
    J in training, validation and prediction; lambda = 0 or None = the plain fit bit for bit). `wd_grid`: overrides the weight-decay grid (B21.3: the c = 0 fit's
    choice). `session_dev`: B22 S22b p x 1[Race / Quali] nuisance block (training only, never in q)."""
    pc: PretrainConfig = cfg.pretrain
    train_rows = np.asarray(train_rows)
    w_all = np.ones(len(train_rows)) if weights is None else np.asarray(weights, float)
    val = inner_split(d.cell_key[train_rows], pc.inner_frac, d.data_hash, f'pre|{salt}')
    fit_rows, val_rows = train_rows[~val & (w_all > 0)], train_rows[val]
    if not len(fit_rows):
        raise ValueError(f'pretrain: no fit rows left of {len(train_rows)} training rows (data too small)')
    w_fit = w_all[~val & (w_all > 0)]
    scaler = fit_track_scaler(d, train_rows)
    sdev = (cfg.session.s22b_alpha, cfg.session.s22b_sessions) if session_dev else None
    enc = NuisanceEncoder(cfg.nuisance, session_dev=sdev).fit(d, fit_rows)
    x_fit, x_val = enc.encode(d, fit_rows), enc.encode(d, val_rows)
    use_prior = prior is not None and prior.lam != 0.
    off_fit = _t(prior.offset(d, fit_rows)) if use_prior else None
    off_val = _t(prior.offset(d, val_rows)) if use_prior else None
    y_fit, y_val = d.gap[fit_rows], d.gap[val_rows]
    b0 = float(np.sum(w_fit * y_fit) / np.sum(w_fit))
    init = ridge_backfit(x_fit, y_fit, w_fit, offset=np.full(len(y_fit), b0), iters=cfg.nuisance.backfit_iters,
                         tol=cfg.nuisance.backfit_tol)
    z_fit, g_fit = context(d, fit_rows, scaler)
    z_val, g_val = context(d, val_rows, scaler)
    low, high = d.p[train_rows].min(axis=0), d.p[train_rows].max(axis=0)
    T = dict(p=_t(d.p[fit_rows]), z=_t(z_fit), g=_t(g_fit), y=_t(y_fit), w=_t(w_fit))
    Tk = dict(track=torch.as_tensor(d.track[fit_rows], dtype=torch.int64),
              game=torch.as_tensor(d.game[fit_rows], dtype=torch.int64), rows=torch.as_tensor(fit_rows, dtype=torch.int64))
    extra_fn = extra.bind(d, fit_rows, scaler, (low, high)) if extra is not None else None
    V = dict(p=_t(d.p[val_rows]), z=_t(z_val), g=_t(g_val), y=_t(y_val))
    cat_f, dense_f = design_tensors(x_fit)
    cat_v, dense_v = design_tensors(x_val)
    max_epochs, patience = (pc.smoke_epochs, pc.smoke_patience) if smoke else (pc.max_epochs, pc.patience)
    best = None
    for wd in (pc.weight_decay_grid if wd_grid is None else tuple(wd_grid)):
        if kind == 'critic3':
            model = GroupCritic(groups, pc.hidden, n_e=0, n_g=len(d.games), final_std=pc.final_init_std, seed=seed)
        else:
            model = FullCritic(pc.hidden_full, n_e=cfg.finetune.n_e, n_g=len(d.games), final_std=pc.final_init_std, seed=seed)
        model.b.fill_(b0)
        nuis = NuisanceModule(x_fit, init)
        heads = [q for q in model.parameters() if q.requires_grad and q.numel()]
        for q in (model.W1e, model.W) if kind == 'critic3' else (model.We, model.W):
            q.requires_grad_(False)
        heads = [q for q in heads if q.requires_grad]
        opt = torch.optim.Adam([{'params': heads, 'weight_decay': wd}, {'params': list(nuis.parameters()), 'weight_decay': 0.}],
                               lr=pc.lr)
        gen = torch.Generator().manual_seed(seed)
        gen_extra = torch.Generator().manual_seed(seed_of('extra', seed))
        extra_sum, extra_n = 0., 0
        lo_t, hi_t = _t(low), _t(high)
        n, W = len(fit_rows), float(w_fit.sum())
        hist = {'val': [], 'best_epoch': 0}
        best_val, best_state, wait = np.inf, None, 0
        for epoch in range(1, max_epochs + 1):
            perm = torch.randperm(n, generator=gen)
            for k in range(0, n, pc.batch):
                bi = perm[k:k + pc.batch]
                pred = model(T['p'][bi], T['z'][bi], T['g'][bi]) + nuis({c: v[bi] for c, v in cat_f.items()}, dense_f[bi])
                if use_prior:
                    pred = pred + off_fit[bi]
                wb = T['w'][bi]
                loss = (wb * (T['y'][bi] - pred) ** 2).sum() / wb.sum() + nuis.penalty() / W
                if guard and kind == 'critic3':
                    gl = guard_loss(model, T['p'], T['z'], T['g'], lo_t, hi_t, cfg.guard, gen)
                    if gl is not None:
                        loss = loss + gl
                if extra_fn is not None:
                    el = extra_fn(model, {'p': T['p'][bi], 'z': T['z'][bi], 'g': T['g'][bi], 'track': Tk['track'][bi],
                                          'game': Tk['game'][bi], 'rows': Tk['rows'][bi]}, gen_extra)
                    if el is not None:
                        loss = loss + el
                        extra_sum, extra_n = extra_sum + float(el.detach()), extra_n + 1
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pv = model(V['p'], V['z'], V['g']) + nuis(cat_v, dense_v)
                if use_prior:
                    pv = pv + off_val
                v = float(((V['y'] - pv) ** 2).mean()) if len(val_rows) else 0.
            hist['val'].append(v)
            if v < best_val - 1e-12:
                best_val, wait, hist['best_epoch'] = v, 0, epoch
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(nuis.state_dict()))
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state[0])
        nuis.load_state_dict(best_state[1])
        if best is None or best_val < best[0]:
            best = (best_val, wd, model, nuis, epoch, hist['best_epoch'], extra_sum / max(extra_n, 1))
    best_val, wd, model, nuis, epochs, best_epoch, extra_mean = best
    with torch.no_grad():
        if kind == 'critic3':
            model.centre_heads(T['p'], T['z'], T['g'], w=T['w'])
        hsum = model.setup_sum(T['p'], T['z'], T['g'])
        head_sd = float(hsum.std())
    info = {'weight_decay': wd, 'inner_val_mse': best_val, 'epochs_run': epochs, 'best_epoch': best_epoch,
            'head_sum_sd_pct': head_sd, 'stalled': bool(best_epoch <= 5 and head_sd < .05), 'n_fit': int(len(fit_rows)),
            'n_inner_val': int(len(val_rows)), 'guard': bool(guard and kind == 'critic3'), 'smoke': smoke,
            'nuisance_init_iters': init.iters,
            'extra_loss': None if extra is None else {'spec': extra.spec(), 'mean_term': extra_mean},
            'meta_prior': None if prior is None else {'source': prior.source, 'c': prior.c, 'lam': prior.lam},
            'session_dev': bool(session_dev)}
    for q in model.parameters():
        q.requires_grad_(False)
    out = Fitted(kind='critic3' if kind == 'critic3' else 'critic2_pre', model=model, nuis=nuis, encoder=enc,
                 scaler=scaler, support=(low, high), info=info, prior=prior)
    return refit_nuisance(out, d, train_rows, cfg)


# ---------------------------------------------------------------------------------------------------
# fine-tune on exact rows (critic 1, critic 2) and the nuisance-only comparators
# ---------------------------------------------------------------------------------------------------
def style_inputs(d, rows, style_tf: TrackStyleTransform):
    s, where = style_tf.transform(d.style_raw[rows], d.track_key[rows])
    return s, where


def finetune(d, train_rows, base: Fitted, mode: str, cfg: C2bConfig = C2bConfig(), seed: int = 0, weights=None,
             guard: bool = True, smoke: bool = False, salt='', s_head=None, l2=None, level=None) -> Fitted:
    """B9 / B10 fine-tune on the exact training rows `train_rows` (all with style).
    mode 'critic1' (base = critic-3 twin), 'critic2' (base = A-pretrained critic 2): trainable W, the
    e-columns (W1e / We) and a new exact-row nuisance incl. m(s~); everything inherited frozen. Adam
    lr 3e-3, full batch, <= 2,000 steps, inner validation on 20 % of training persons, patience 100;
    L2 on (W, W1e) from the grid unless `l2` is given (lowest inner-validation MSE).
    mode 'c3_exact' / 'c2pre_exact': the comparator (base heads, e = 0, the exact-row nuisance in closed
    form on all training rows: the convex optimum of 'only the nuisance trained').
    `s_head` (len(train_rows), 9): the s~ fed to the heads (the shuffle ablation); m(s~) always gets
    the real s~. `weights`: twin bootstrap weights by cell.
    `level` (B23 rung 2): (name, gamma, l2, alpha of m(s~)): e' = gamma e in the heads (`e_gate`, kept in the
    model and the bundle), the fixed L2 on (W, W1e) and the m(s~) penalty; the B9 initialisation is unchanged.
    The B21 prior of `base` is inherited frozen (lambda Pi added to J; it never sees s or e)."""
    fc: FineTuneConfig = cfg.finetune
    if level is not None:
        _, gamma, l2, a_style = level
        cfg = replace(cfg, nuisance=replace(cfg.nuisance, alpha_style=float(a_style)))
    train_rows = np.asarray(train_rows)
    style_tf = TrackStyleTransform(min_track_cells=5).fit(d.style_raw, d.track_key, train_rows)
    s_real, _ = style_inputs(d, train_rows, style_tf)
    s_h = s_real if s_head is None else np.asarray(s_head, float)
    comparator = mode in ('c3_exact', 'c2pre_exact')
    if comparator:          # base critic + the exact-row nuisance (incl. m(s~)) in closed form on all training rows
        out = Fitted(kind=mode, model=base.model, nuis=None, encoder=None, scaler=base.scaler, style_tf=style_tf,
                     support=base.support, bcn=base.bcn, info={'mode': mode, 'smoke': smoke}, prior=base.prior)
        return refit_nuisance(out, d, train_rows, cfg)
    w_all = np.ones(len(train_rows)) if weights is None else np.asarray(weights, float)
    val = inner_split(d.person[train_rows], fc.inner_frac, d.data_hash, f'ft|{salt}')
    fit_m = ~val & (w_all > 0)
    fit_rows, val_rows = train_rows[fit_m], train_rows[val]
    if not len(fit_rows):
        raise ValueError(f'finetune: no fit rows left of {len(train_rows)} training rows after the inner split and '
                         'the bootstrap (data too small)')
    w_fit = w_all[fit_m]
    enc = NuisanceEncoder(cfg.nuisance, style=True).fit(d, fit_rows)
    x_fit = enc.encode(d, fit_rows, s_tilde=s_real[fit_m])
    x_val = enc.encode(d, val_rows, s_tilde=s_real[val])
    scaler = base.scaler
    z_fit, g_fit = context(d, fit_rows, scaler)
    z_val, g_val = context(d, val_rows, scaler)
    T = dict(p=_t(d.p[fit_rows]), z=_t(z_fit), g=_t(g_fit), y=_t(d.gap[fit_rows]), w=_t(w_fit), s=_t(s_h[fit_m]))
    V = dict(p=_t(d.p[val_rows]), z=_t(z_val), g=_t(g_val), y=_t(d.gap[val_rows]), s=_t(s_h[val]))
    prior = base.prior if base.prior is not None and base.prior.lam != 0. else None
    off_fit = _t(prior.offset(d, fit_rows)) if prior is not None else torch.zeros(len(fit_rows))
    off_val = _t(prior.offset(d, val_rows)) if prior is not None else torch.zeros(len(val_rows))
    with torch.no_grad():
        j0 = (base.model(T['p'], T['z'], T['g']) + off_fit).double().numpy()
    init = ridge_backfit(x_fit, d.gap[fit_rows], w_fit, offset=j0, iters=cfg.nuisance.backfit_iters, tol=cfg.nuisance.backfit_tol)
    cat_f, dense_f = design_tensors(x_fit)
    cat_v, dense_v = design_tensors(x_val)
    W_init = style_pca_init(s_h, fc.n_e)
    grid = (l2,) if l2 is not None else fc.l2_grid
    if level is not None:
        grid = (float(level[2]),)
    max_steps = fc.smoke_steps if smoke else fc.max_steps
    low, high = base.support
    lo_t, hi_t = _t(low), _t(high)
    W_n = float(w_fit.sum())
    best = None
    for lam in grid:
        if base.kind == 'critic3':
            model = GroupCritic.from_critic3(base.model, W_init, fc.n_e)
            e_params = [model.W, model.W1e]
            if level is not None:
                model.e_gate.fill_(float(level[1]))
        else:
            model = copy.deepcopy(base.model)
            with torch.no_grad():
                model.W.copy_(_t(W_init))
                model.We.zero_()
            model.use_e = True
            e_params = [model.W, model.We]
            if level is not None:
                model.e_gate.fill_(float(level[1]))
        for q in model.parameters():
            q.requires_grad_(False)
        if not comparator:
            for q in e_params:
                q.requires_grad_(True)
        nuis = NuisanceModule(x_fit, init)
        params = list(nuis.parameters()) + ([] if comparator else e_params)
        opt = torch.optim.Adam(params, lr=fc.lr)
        gen = torch.Generator().manual_seed(seed)
        use_e = not comparator
        best_val, best_state, wait, best_step, step = np.inf, None, 0, 0, 0
        for step in range(1, max_steps + 1):
            e = model.encode(T['s']) if use_e else None
            pred = model(T['p'], T['z'], T['g'], e=e) + nuis(cat_f, dense_f) + off_fit
            loss = (T['w'] * (T['y'] - pred) ** 2).sum() / T['w'].sum() + nuis.penalty() / W_n
            if not comparator:
                loss = loss + lam * sum((q ** 2).sum() for q in e_params)
                if guard and base.kind == 'critic3':
                    gl = guard_loss(model, T['p'], T['z'], T['g'], lo_t, hi_t, cfg.guard, gen, e_pool=model.encode(T['s']))
                    if gl is not None:
                        loss = loss + gl
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ev = model.encode(V['s']) if use_e else None
                v = float(((V['y'] - model(V['p'], V['z'], V['g'], e=ev) - nuis(cat_v, dense_v) - off_val) ** 2).mean()) \
                    if len(val_rows) else 0.
            if v < best_val - 1e-12:
                best_val, wait, best_step = v, 0, step
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(nuis.state_dict()))
            else:
                wait += 1
                if wait >= fc.patience:
                    break
        model.load_state_dict(best_state[0])
        nuis.load_state_dict(best_state[1])
        if best is None or best_val < best[0]:
            best = (best_val, lam, model, nuis, step, best_step)
    best_val, lam, model, nuis, steps, best_step = best
    for q in model.parameters():
        q.requires_grad_(False)
    if not comparator:                          # R10: e range of the training rows (clip at evaluation / query)
        with torch.no_grad():
            e_tr = model.encode(_t(s_h))
            model.e_low.copy_(e_tr.min(0).values)
            model.e_high.copy_(e_tr.max(0).values)
    info = {'mode': mode, 'l2': lam, 'inner_val_mse': best_val, 'steps_run': steps, 'best_step': best_step,
            'n_fit': int(len(fit_rows)), 'n_inner_val': int(len(val_rows)), 'smoke': smoke,
            'guard': bool(guard and base.kind == 'critic3' and not comparator), 'shuffled': s_head is not None,
            'level': None if level is None else {'name': level[0], 'gamma': float(level[1]), 'l2': float(level[2]),
                                                 'alpha_m_s': float(level[3])},
            'e_gate': float(model.e_gate)}
    if not comparator:
        info['W_norm'] = float(model.W.norm())
        info['W1e_norm'] = float((model.W1e if base.kind == 'critic3' else model.We).norm())
        info['m_style_coef'] = nuis.dense.detach().numpy()[[i for i, nm in enumerate(x_fit.dense_names)
                                                            if nm.startswith('style:')]].tolist()
    out = Fitted(kind=mode, model=model, nuis=nuis, encoder=enc, scaler=scaler, style_tf=style_tf, support=base.support,
                 bcn=base.bcn, info=info, prior=base.prior)
    return refit_nuisance_ft(out, d, train_rows, s_h, cfg)


# ---------------------------------------------------------------------------------------------------
# Barcelona residual head (B11)
# ---------------------------------------------------------------------------------------------------
def fit_bcn_head(d, train_rows, base: Fitted, cfg: BcnHeadConfig = BcnHeadConfig(), seed: int = 0, smoke: bool = False,
                 smoke_epochs: int = 20, salt='') -> tuple:
    """Head on the Barcelona rows among `train_rows` with `base` (critic + its nuisance) frozen; L2 from
    the grid on 20 % of the Barcelona training cells. Returns (head, info)."""
    train_rows = np.asarray(train_rows)
    rows = train_rows[np.asarray(d.track_key[train_rows] == BARCELONA)]
    pred = predict(base, d, rows, with_bcn=False)
    r = d.gap[rows] - pred['full']
    val = inner_split(d.cell_key[rows], cfg.inner_frac, d.data_hash, f'bcn|{salt}')
    P, R = _t(d.p[rows]), _t(r)
    fi, vi = torch.as_tensor(np.flatnonzero(~val)), torch.as_tensor(np.flatnonzero(val))
    max_epochs = smoke_epochs if smoke else cfg.max_epochs
    best = None
    for lam in cfg.l2_grid:
        head = BarcelonaHead(cfg.hidden, seed=seed)
        opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
        gen = torch.Generator().manual_seed(seed)
        best_val, best_state, wait, best_epoch, epoch = np.inf, copy.deepcopy(head.state_dict()), 0, 0, 0
        for epoch in range(1, max_epochs + 1):
            perm = fi[torch.randperm(len(fi), generator=gen)]
            for k in range(0, len(perm), cfg.batch):
                bi = perm[k:k + cfg.batch]
                loss = ((R[bi] - head(P[bi])) ** 2).mean() + lam * sum((q ** 2).sum() for q in head.parameters())
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = float(((R[vi] - head(P[vi])) ** 2).mean()) if len(vi) else 0.
            if v < best_val - 1e-12:
                best_val, wait, best_epoch = v, 0, epoch
                best_state = copy.deepcopy(head.state_dict())
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        head.load_state_dict(best_state)
        if best is None or best_val < best[0]:
            best = (best_val, lam, head, epoch, best_epoch)
    best_val, lam, head, epochs, best_epoch = best
    for q in head.parameters():
        q.requires_grad_(False)
    with torch.no_grad():
        out = head(P).numpy()
    return head, {'l2': lam, 'inner_val_mse': best_val, 'epochs_run': epochs, 'best_epoch': best_epoch,
                  'n_rows': int(len(rows)), 'n_inner_val': int(val.sum()), 'head_sd_pct': float(out.std()) if len(out) else 0.,
                  'head_mean_pct': float(out.mean()) if len(out) else 0.}


# ---------------------------------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------------------------------
def refit_nuisance_ft(f: Fitted, d, train_rows, s_head, cfg: C2bConfig = C2bConfig()) -> Fitted:
    """`refit_nuisance` for a fine-tuned critic whose heads were fed `s_head` (the shuffle ablation)."""
    train_rows = np.asarray(train_rows)
    enc = NuisanceEncoder(cfg.nuisance, style=True).fit(d, train_rows)
    x = enc.encode(d, train_rows, s_tilde=style_inputs(d, train_rows, f.style_tf)[0])
    J = predict(f, d, train_rows, s_head=s_head, nuisance=False)['setup']
    fit = ridge_backfit(x, d.gap[train_rows], offset=J)
    f.nuis, f.encoder = NuisanceModule(x, fit), enc
    for q in f.nuis.parameters():
        q.requires_grad_(False)
    f.info['nuisance_refit_rows'] = int(len(train_rows))
    return f


def predict(f: Fitted, d, rows, s_head=None, with_bcn: bool = True, e_zero: bool = False, nuisance: bool = True) -> dict:
    """{'full': J + head + nuisance, 'setup': J + head (the q part, incl. lambda Pi), 'nuis', 'clipped_frac',
    'prior' (lambda Pi), 'session_dev' (the S22b p x session part of the nuisance; 0 without)} for rows.
    Style critics: s~ from the fitted B4 transform (real s~ for m(s~)); `s_head` overrides the s~ fed
    to the heads (shuffle ablation); e is clipped to the training range (R10). `e_zero`: heads at e = 0."""
    rows = np.asarray(rows)
    z, g = context(d, rows, f.scaler)
    P, Z, G = _t(d.p[rows]), _t(z), _t(g)
    s_real = None
    if f.style_tf is not None:
        s_real, _ = style_inputs(d, rows, f.style_tf)
    clipped = 0.
    with torch.no_grad():
        e = None
        if f.kind in ('critic1', 'critic2'):
            s_h = s_real if s_head is None else np.asarray(s_head, float)
            e_raw = f.model.encode(_t(s_h))
            e = f.model.encode(_t(s_h), clip=True)
            clipped = float((e_raw != e).any(dim=1).float().mean()) if len(rows) else 0.
            if e_zero:
                e = torch.zeros_like(e)
        J = f.model(P, Z, G, e=e).double().numpy()
        pri = f.prior.offset(d, rows) if f.prior is not None else np.zeros(len(rows))
        J = J + pri
        sdev = np.zeros(len(rows))
        if nuisance:
            x = f.encoder.encode(d, rows, s_tilde=s_real)
            cat, dense = design_tensors(x)
            nu = f.nuis(cat, dense).double().numpy()
            sel = np.array([nm.startswith('sdev:') for nm in x.dense_names])
            if sel.any():
                sdev = x.dense[:, sel] @ f.nuis.dense.detach().double().numpy()[sel]
        else:
            nu = np.zeros(len(rows))
        if with_bcn and f.bcn is not None:
            m = np.asarray(d.track_key[rows] == BARCELONA)
            if m.any():
                J = J + np.where(m, f.bcn(P).double().numpy(), 0.)
    return {'full': J + nu, 'setup': J, 'nuis': nu, 'clipped_frac': clipped, 'prior': pri, 'session_dev': sdev}


__all__ = ['Fitted', 'boot_weights', 'context', 'finetune', 'fit_bcn_head', 'fit_track_scaler', 'guard_loss', 'predict',
           'pretrain', 'refit_nuisance', 'seed_of', 'set_determinism', 'style_inputs', 'style_pca_init']
