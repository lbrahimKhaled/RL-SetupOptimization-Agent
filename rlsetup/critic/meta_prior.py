"""C2b_addendum B21: the meta-setup prior of critic 3 (source-independent; inputs are C2Data and p).

    J3(p, c) = scale (b + sum_g h_g(p_g, z, g)) + lambda Pi(p; t, g),   Pi = sum_k w_tk (p_k - m_tk)^2

- p box-normalised (KNOB_ORDER); m_t = the g-weighted mean of the games' m at track t (a training row has a
  one-hot g, so its own game's m); a track without m (unseen in training) has Pi = 0 (`meta_prior='none'`).
- m^in (B21.1, the primary): per (game, track) the knob-wise median of the training rows whose B15
  nuisance-only residual (fit on the fold's training rows) is in the lowest 20 % of that (game, track);
  fewer than 20 training rows -> the game's pooled fast set (the union of its per-(game, track) fast sets),
  flagged. Only tracks with training rows get an m.
- Published meta (sensitivity only, never delivered): m from the published per-(game, track) setups.
- w (B21.2): cross-fitted. Training cells are split in two by sha256(data_hash | 'meta' | cell) mod 2; m is
  computed from one half and scores the within-(game, driver, track, session) pairs of the other half on
  nuisance residuals (win = the setup closer to m on knob k has the lower residual; a pair counts on k if
  the two distances differ by more than 1e-9); swap; pool games and halves per track: W_tk, N_tk.
  r_bar_k = sum_t W / sum_t N, r_hat_tk = (W + kappa r_bar) / (N + kappa), w = max(0, 2 r_hat - 1).
  An unseen track uses max(0, 2 r_bar - 1) (it has no m, so Pi = 0 there anyway).
- lambda = c SD_train(setup part of the c = 0 fit) / SD_train(Pi), c in {0, 0.03, 0.1, 0.3} (B21.3).
Pi has no trainable parameter and never takes s or e (B21.6). Everything is fitted on the fold's
training rows only (not per bootstrap twin).
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np
import torch
from .c2_config import C2bConfig, MetaPriorConfig
from .metrics import pairs_of

SOURCES = ('in_fold', 'published')


def _half(data_hash: str, cell_keys) -> np.ndarray:
    cache, out = {}, np.empty(len(cell_keys), int)
    for i, k in enumerate(cell_keys):
        if k not in cache:
            cache[k] = int(hashlib.sha256(f'{data_hash}|meta|{int(k)}'.encode()).hexdigest()[:16], 16) % 2
        out[i] = cache[k]
    return out


def fast_set_m(d, rows, resid, n_games: int, n_tracks: int, cfg: MetaPriorConfig = MetaPriorConfig()):
    """(m (n_games, n_tracks, 15) NaN where none, pooled flag (n_games, n_tracks) bool) from `rows`.
    Tracks without any of `rows` get no m; a (game, track) with fewer than `min_rows` rows uses the game's
    pooled fast set (flagged)."""
    rows = np.asarray(rows)
    m = np.full((n_games, n_tracks, d.p.shape[1]), np.nan)
    pooled = np.zeros((n_games, n_tracks), bool)
    fast = np.zeros(d.n, bool)
    gt = d.game[rows] * n_tracks + d.track[rows]
    for key in np.unique(gt):
        r = rows[gt == key]
        f = r[resid[r] <= np.quantile(resid[r], cfg.fast_frac)]
        fast[f] = True
        if len(r) >= cfg.min_rows:
            m[key // n_tracks, key % n_tracks] = np.median(d.p[f], axis=0)
    seen = np.unique(d.track[rows])
    for g in range(n_games):
        fg = rows[fast[rows] & (d.game[rows] == g)]
        game_m = np.median(d.p[fg], axis=0) if len(fg) else None
        for t in seen:
            if not np.isfinite(m[g, t]).all():
                if game_m is not None:
                    m[g, t], pooled[g, t] = game_m, True
    return m, pooled


def cross_fit_pairs(d, rows, resid, m_of_half: dict, half: np.ndarray, n_tracks: int, eps: float = 1e-9):
    """Score the within-session pairs of each half against the other half's m.
    Returns (W (n_tracks, 15), N (n_tracks, 15), log) with log = {'pairs', 'pair_half', 'm_half'} for tests."""
    rows = np.asarray(rows)
    K = d.p.shape[1]
    W, N = np.zeros((n_tracks, K)), np.zeros((n_tracks, K))
    log = {'pairs': [], 'pair_half': [], 'm_half': []}
    for h in (0, 1):
        rh = rows[half == h]
        if not len(rh):
            continue
        pr = pairs_of(d.game, d.driver, d.track, d.session, d.setup_hash, d.gap, rows=rh)
        if not len(pr):
            continue
        m = m_of_half[1 - h]
        i, j = pr[:, 0], pr[:, 1]
        M = m[d.game[i], d.track[i]]
        dk = np.abs(d.p[i] - M) - np.abs(d.p[j] - M)
        first_faster = resid[i] < resid[j]
        ok = np.isfinite(dk) & (np.abs(dk) > eps)
        win = (dk < 0) == first_faster[:, None]
        for k in range(K):
            np.add.at(W[:, k], d.track[i][ok[:, k]], win[ok[:, k], k])
            np.add.at(N[:, k], d.track[i][ok[:, k]], 1.)
        log['pairs'].append(pr)
        log['pair_half'].append(np.full(len(pr), h))
        log['m_half'].append(np.full(len(pr), 1 - h))
    log = {k: (np.concatenate(v) if v else np.zeros((0, 2) if k == 'pairs' else 0, int)) for k, v in log.items()}
    return W, N, log


@dataclass
class MetaPrior:
    m: np.ndarray                   # (n_games, n_tracks, 15) in p units, NaN = none
    w: np.ndarray                   # (n_tracks, 15)
    r_bar: np.ndarray               # (15,)
    source: str                     # 'in_fold' | 'published'
    games: tuple
    tracks: tuple
    lam: float = 0.
    c: float = 0.
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f'meta prior source {self.source!r}')

    # ---- per row ------------------------------------------------------------------------------------
    def m_rows(self, game, track) -> np.ndarray:
        """(n, 15) the row's own game's m at its track (NaN = no m)."""
        return self.m[np.asarray(game), np.asarray(track)]

    def pi_rows(self, p, game, track) -> np.ndarray:
        """Pi per row (numpy); 0 where the track has no m."""
        M = self.m_rows(game, track)
        w = self.w[np.asarray(track)]
        out = np.sum(w * (np.asarray(p, float) - M) ** 2, axis=-1)
        return np.where(np.isfinite(out), out, 0.)

    def offset(self, d, rows) -> np.ndarray:
        """lambda Pi of rows (the additive term of J3); zeros when lambda = 0."""
        rows = np.asarray(rows)
        if self.lam == 0.:
            return np.zeros(len(rows))
        return self.lam * self.pi_rows(d.p[rows], d.game[rows], d.track[rows])

    # ---- query (bundle) -----------------------------------------------------------------------------
    def m_query(self, track: str, g: np.ndarray):
        """(m (15,) or None, flag). g-weighted mean over the games that have an m at the track."""
        if track not in self.tracks:
            return None, 'none'
        t = self.tracks.index(track)
        have = np.isfinite(self.m[:, t]).all(axis=1)
        if not have.any():
            return None, 'none'
        g = np.asarray(g, float) * have
        if g.sum() <= 0:
            g = have.astype(float)
        tp = self.info.get('track_pairs')
        flag = 'pooled_w' if tp is not None and tp[t] == 0 else 'track'
        return (g[:, None] * np.nan_to_num(self.m[:, t])).sum(0) / g.sum(), flag

    @staticmethod
    def pi_torch(p: torch.Tensor, m: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Pi(p) for one context (m, w (15,)): sum_k w_k (p_k - m_k)^2; differentiable in p."""
        return (w * (p - m) ** 2).sum(-1)

    # ---- lambda, state ------------------------------------------------------------------------------
    def with_lambda(self, c: float, setup_sd: float, pi_sd: float) -> 'MetaPrior':
        lam = 0. if c == 0. or pi_sd <= 0 else float(c) * float(setup_sd) / float(pi_sd)
        return MetaPrior(self.m, self.w, self.r_bar, self.source, self.games, self.tracks, lam=lam, c=float(c),
                         info=dict(self.info, setup_sd=float(setup_sd), pi_sd=float(pi_sd)))

    def state(self) -> dict:
        return {'m': np.where(np.isfinite(self.m), self.m, np.nan).tolist(), 'w': self.w.tolist(),
                'r_bar': self.r_bar.tolist(), 'source': self.source, 'games': list(self.games),
                'tracks': list(self.tracks), 'lam': self.lam, 'c': self.c,
                'info': {k: v for k, v in self.info.items() if k in ('pooled_m', 'n_pairs_scored', 'setup_sd', 'pi_sd',
                                                                    'kappa', 'n_train_rows', 'track_pairs')}}

    @classmethod
    def from_state(cls, st: dict) -> 'MetaPrior':
        nan = lambda v: np.array(v, dtype=float)
        return cls(m=nan(st['m']), w=nan(st['w']), r_bar=nan(st['r_bar']), source=st['source'], games=tuple(st['games']),
                   tracks=tuple(st['tracks']), lam=float(st['lam']), c=float(st['c']), info=st.get('info', {}))


def nuisance_residual(d, train_rows, cfg: C2bConfig = C2bConfig()) -> np.ndarray:
    """(n,) gap minus the B15 nuisance-only prediction (fit on `train_rows`), NaN off the training rows."""
    from .ridge import nuisance_only
    train_rows = np.asarray(train_rows)
    res = np.full(d.n, np.nan)
    res[train_rows] = d.gap[train_rows] - nuisance_only(d, train_rows, train_rows, cfg)['full']
    return res


def fit_meta_prior(d, train_rows, cfg: C2bConfig = C2bConfig(), published: dict = None) -> MetaPrior:
    """B21.1-2 on the fold's training rows: m (in-fold fast set, or `published` {(game, track): p (15,)}),
    cross-fitted w and r_bar. lambda = 0 (set with `with_lambda`)."""
    mc = cfg.meta
    train_rows = np.asarray(train_rows)
    nG, nT = len(d.games), len(d.track_names)
    resid = nuisance_residual(d, train_rows, cfg)
    half = _half(d.data_hash, d.cell_key[train_rows])
    if published is None:
        m, pooled = fast_set_m(d, train_rows, resid, nG, nT, mc)
        halves = {h: fast_set_m(d, train_rows[half == h], resid, nG, nT, mc)[0] for h in (0, 1)}
        source = 'in_fold'
    else:
        m = np.full((nG, nT, d.p.shape[1]), np.nan)
        for (g, t), v in published.items():
            if g in d.games and t in d.track_names:
                m[d.games.index(g), d.track_names.index(t)] = v
        pooled = np.zeros((nG, nT), bool)
        halves = {0: m, 1: m}
        source = 'published'
    W, N, log = cross_fit_pairs(d, train_rows, resid, halves, half, nT, mc.dist_eps)
    r_bar = np.where(N.sum(0) > 0, W.sum(0) / np.maximum(N.sum(0), 1), .5)
    r_hat = (W + mc.kappa * r_bar) / (N + mc.kappa)
    w = np.maximum(0., 2. * r_hat - 1.)
    info = {'pooled_m': [[d.games[g], d.track_names[t]] for g, t in zip(*np.nonzero(pooled))],
            'track_pairs': N.max(1).astype(int).tolist(),
            'n_pairs_scored': int(len(log['pairs'])), 'kappa': mc.kappa, 'n_train_rows': int(len(train_rows)),
            'wins': W, 'counts': N, 'cross_fit_log': log, 'half': half}
    return MetaPrior(m=m, w=w, r_bar=r_bar, source=source, games=tuple(d.games), tracks=tuple(d.track_names), info=info)


def closeness(m_a: np.ndarray, m_b: np.ndarray, sd: np.ndarray) -> dict:
    """E21.3 numbers: mean |m_a - m_b| / SD per knob over (game, track) where both exist, and the share of
    exactly equal values."""
    ok = np.isfinite(m_a).all(-1) & np.isfinite(m_b).all(-1)
    a, b = m_a[ok], m_b[ok]
    if not len(a):
        return {'n_game_tracks': 0}
    dist = np.abs(a - b) / np.where(sd > 0, sd, 1.)
    return {'n_game_tracks': int(len(a)), 'mean_distance_per_sd': float(dist.mean()),
            'per_knob': dist.mean(0).tolist(), 'share_equal': float(np.mean(np.abs(a - b) < 1e-9))}


__all__ = ['MetaPrior', 'SOURCES', 'closeness', 'cross_fit_pairs', 'fast_set_m', 'fit_meta_prior', 'nuisance_residual']
