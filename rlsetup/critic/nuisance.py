"""C2b_addendum B7: the nuisance terms (training only, never in q), their closed-form ridge fit and the
torch module trained jointly with the heads.

Terms (additive; ridge penalty alpha on the sum of squares = alpha / n_train on the mean loss):
- 'gts'      (game, track, session) fixed effect, alpha 1e-2;
- 'driver'   offset by (game, user), alpha 1; drivers with < 2 training rows share level 0 ('rare');
- 'ctrl'     controller x game; 'assist:<name>' each assist x game; 'team' team x game; alpha 1;
- 'month'    game x calendar month (unknown date = its own level), alpha 1;
- 'trend'    dense, per game: months from the driver's mean date (training rows of that driver; a driver
             without training rows uses the mean of its own rows), 0 when the date is unknown, alpha 1;
- 'style'    dense m(s~), 9 -> 1, alpha 10 (critics 1 and 2 only);
- 'sdev'     (B22 S22b only) p x 1[session] for the listed sessions (Race, Quali; TT is the reference),
             dense, alpha 10;
- extra dense blocks (ridge baselines B15: p, p x z, s x PC4(p)) with their own alpha.
Level 0 of every categorical term is 'rare / unseen': training levels below the minimum count and every
level unseen in training map there (C1-R4). Encoders are fitted on training rows only.
"""
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
from .c2_config import NuisanceConfig

BIG = 1_000_000_000


@dataclass
class Design:
    """Encoded nuisance (+ extra dense) inputs of a set of rows."""
    cat: dict                       # name -> (n,) int level (0 = rare / unseen)
    dense: np.ndarray               # (n, d)
    n_levels: dict                  # name -> number of levels (incl. 0)
    alpha_cat: dict                 # name -> alpha
    alpha_dense: np.ndarray         # (d,)
    dense_names: list = field(default_factory=list)

    def __len__(self):
        return len(self.dense)


class NuisanceEncoder:
    """Level maps of the B7 terms, fitted on training rows (`fit(data, idx)`)."""

    def __init__(self, cfg: NuisanceConfig = NuisanceConfig(), style: bool = False, terms=None, session_dev=None):
        """`session_dev`: None or (alpha, session names) for the B22 S22b p x 1[session] block."""
        self.cfg, self.style = cfg, style
        self.session_dev = None if session_dev is None else (float(session_dev[0]), tuple(session_dev[1]))
        self.terms = terms                          # None = all categorical terms of B7
        self.maps, self.driver_mean_month = {}, {}

    def _keys(self, d, idx) -> dict:
        idx = np.asarray(idx)
        g = d.game[idx].astype(np.int64)
        month = np.where(d.date_days[idx] >= 0, d.date_days[idx] // 1, -1)
        cal = np.where(month >= 0, (np.asarray(month, 'datetime64[D]').astype('datetime64[M]').astype(np.int64)), -1)
        out = {'gts': d.group_key()[idx].astype(np.int64), 'driver': g * BIG + d.driver[idx],
               'ctrl': g * BIG + d.controller[idx], 'team': g * BIG + d.team[idx], 'month': g * BIG + cal}
        for a, v in d.assists.items():
            out[f'assist:{a}'] = g * BIG + v[idx]
        if self.terms is not None:
            out = {k: v for k, v in out.items() if k in self.terms}
        return out

    def _alpha(self, name):
        c = self.cfg
        return c.alpha_gts if name == 'gts' else c.alpha_driver if name == 'driver' else c.alpha_other

    def fit(self, d, idx) -> 'NuisanceEncoder':
        idx = np.asarray(idx)
        self.maps = {}
        for name, keys in self._keys(d, idx).items():
            u, cnt = np.unique(keys, return_counts=True)
            keep = u[cnt >= (self.cfg.min_driver_rows if name == 'driver' else 1)]
            self.maps[name] = {int(k): i + 1 for i, k in enumerate(keep)}
        months = d.date_days[idx] / self.cfg.days_per_month
        dk = d.game[idx].astype(np.int64) * BIG + d.driver[idx]
        known = d.date_days[idx] >= 0
        self.driver_mean_month = {}
        for k in np.unique(dk[known]):
            self.driver_mean_month[int(k)] = float(months[known & (dk == k)].mean())
        return self

    def encode(self, d, idx, s_tilde=None, extra=None, extra_alpha=None, extra_names=None) -> Design:
        """`s_tilde` (n, 9) for the style term (required when the encoder has `style`); `extra` (n, k)
        dense columns with penalties `extra_alpha` (k,) (ridge baselines)."""
        idx = np.asarray(idx)
        cat = {}
        for name, keys in self._keys(d, idx).items():
            m = self.maps[name]
            cat[name] = np.fromiter((m.get(int(k), 0) for k in keys), int, len(keys))
        n_games = len(d.games)
        months = d.date_days[idx] / self.cfg.days_per_month
        known = d.date_days[idx] >= 0
        dk = d.game[idx].astype(np.int64) * BIG + d.driver[idx]
        own = {}
        for k in np.unique(dk[known]):
            if int(k) not in self.driver_mean_month:
                own[int(k)] = float(months[known & (dk == k)].mean())
        mean = np.array([self.driver_mean_month.get(int(k), own.get(int(k), 0.)) for k in dk])
        dev = np.where(known, months - mean, 0.)
        trend = np.zeros((len(idx), n_games))
        trend[np.arange(len(idx)), d.game[idx]] = dev
        cols, alphas, names = [trend], [np.full(n_games, self.cfg.alpha_other)], [f'trend:{g}' for g in d.games]
        if self.style:
            if s_tilde is None:
                raise ValueError('style nuisance needs s_tilde')
            cols.append(np.asarray(s_tilde, float))
            alphas.append(np.full(s_tilde.shape[1], self.cfg.alpha_style))
            names += [f'style:{j}' for j in range(s_tilde.shape[1])]
        if self.session_dev is not None:
            a, names_s = self.session_dev
            for sn in names_s:
                ind = (d.session[idx] == list(d.sessions).index(sn)).astype(float) if sn in d.sessions else np.zeros(len(idx))
                cols.append(d.p[idx] * ind[:, None])
                alphas.append(np.full(d.p.shape[1], a))
                names += [f'sdev:{sn}:{j}' for j in range(d.p.shape[1])]
        if extra is not None:
            cols.append(np.asarray(extra, float))
            alphas.append(np.asarray(extra_alpha, float))
            names += list(extra_names or [f'x{j}' for j in range(extra.shape[1])])
        return Design(cat=cat, dense=np.column_stack(cols), n_levels={k: len(m) + 1 for k, m in self.maps.items()},
                      alpha_cat={k: self._alpha(k) for k in self.maps}, alpha_dense=np.concatenate(alphas),
                      dense_names=names)


@dataclass
class RidgeFit:
    """Closed-form weighted ridge (coordinate descent over the blocks)."""
    theta: dict                     # name -> (n_levels,)
    beta: np.ndarray                # (d,)
    iters: int = 1

    def predict(self, x: Design, dense_mask=None) -> np.ndarray:
        """Sum of the terms; `dense_mask` (d,) bool restricts the dense part (e.g. the setup part)."""
        beta = self.beta if dense_mask is None else np.where(dense_mask, self.beta, 0.)
        out = x.dense @ beta
        for k, v in x.cat.items():
            out = out + self.theta[k][v]
        return out


def ridge_backfit(x: Design, y, w=None, offset=None, iters: int = 0, tol: float = 0., init: RidgeFit = None) -> RidgeFit:
    """Minimise sum w (y - offset - f)^2 + sum_t alpha_t |theta_t|^2 + sum_j alpha_j beta_j^2 (B7 penalty
    on the sum of squares) exactly: one sparse direct solve of the normal equations (scipy SuperLU) over
    all one-hot blocks and the dense block. (Block coordinate descent converged too slowly on the
    real data: 3,000 sweeps were still 0.4 above the optimum of the objective; measured in C2b.)
    `iters`, `tol`, `init` are accepted for API compatibility and unused."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    r0 = y - (0. if offset is None else np.asarray(offset, float))
    names = sorted(x.cat)
    n = len(y)
    blocks = [sp.csr_matrix((np.ones(n), (np.arange(n), x.cat[k])), shape=(n, x.n_levels[k])) for k in names]
    A = sp.hstack(blocks + [sp.csr_matrix(x.dense)]).tocsr()
    alpha = np.concatenate([np.full(x.n_levels[k], x.alpha_cat[k]) for k in names] + [x.alpha_dense])
    AtW = A.T.multiply(w[None, :]).tocsr()
    lhs = (AtW @ A + sp.diags(alpha)).tocsc()
    sol = np.atleast_1d(sla.spsolve(lhs, AtW @ r0)) if lhs.shape[0] else np.zeros(0)
    theta, pos = {}, 0
    for k in names:
        theta[k] = sol[pos:pos + x.n_levels[k]].copy()
        pos += x.n_levels[k]
    return RidgeFit(theta=theta, beta=sol[pos:].copy(), iters=1)


class NuisanceModule(nn.Module):
    """The B7 terms as a torch module (embeddings + a linear dense part), initialised from a RidgeFit."""

    def __init__(self, x: Design, init: RidgeFit = None):
        super().__init__()
        self.names = sorted(x.n_levels)
        self.emb = nn.ParameterDict({self._k(k): nn.Parameter(torch.zeros(x.n_levels[k])) for k in self.names})
        self.dense = nn.Parameter(torch.zeros(x.dense.shape[1]))
        self.register_buffer('alpha_cat', torch.tensor([x.alpha_cat[k] for k in self.names], dtype=torch.float32))
        self.register_buffer('alpha_dense', torch.as_tensor(x.alpha_dense, dtype=torch.float32))
        if init is not None:
            with torch.no_grad():
                for k in self.names:
                    self.emb[self._k(k)].copy_(torch.as_tensor(init.theta[k]))
                self.dense.copy_(torch.as_tensor(init.beta))

    @staticmethod
    def _k(name):
        return name.replace(':', '__')

    def forward(self, cat: dict, dense: torch.Tensor) -> torch.Tensor:
        out = dense @ self.dense
        for k in self.names:
            out = out + self.emb[self._k(k)][cat[k]]
        return out

    def penalty(self) -> torch.Tensor:
        """sum_t alpha_t |theta_t|^2 + sum_j alpha_j beta_j^2 (divide by n_train for the mean loss)."""
        cat = torch.stack([(self.emb[self._k(k)] ** 2).sum() for k in self.names]) if self.names else torch.zeros(1)
        return (self.alpha_cat * cat).sum() + (self.alpha_dense * self.dense ** 2).sum()

    def to_ridge(self) -> RidgeFit:
        return RidgeFit(theta={k: self.emb[self._k(k)].detach().double().numpy().copy() for k in self.names},
                        beta=self.dense.detach().double().numpy().copy())


def design_tensors(x: Design, rows=None):
    """(cat {name: LongTensor}, dense FloatTensor) of a Design, optionally a subset of rows."""
    sel = slice(None) if rows is None else np.asarray(rows)
    return ({k: torch.as_tensor(v[sel], dtype=torch.long) for k, v in x.cat.items()},
            torch.as_tensor(x.dense[sel], dtype=torch.float32))


__all__ = ['Design', 'NuisanceEncoder', 'NuisanceModule', 'RidgeFit', 'design_tensors', 'ridge_backfit']
