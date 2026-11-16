"""C2b critics (C2b_addendum B8-B11). Pure torch, CPU. Output J = % lap-time gap (lower = better);
q = -J. Inputs: p (..., 15) normalised knobs in KNOB_ORDER, z^ (..., 5) standardised track features,
g^ (..., 2) source one-hot (uniform at an OCD / AC query, R8), s~ (..., 9) B4 style.

- GroupCritic: J = scale (b + sum_g h_g), h_g = w2_g SiLU(W1p_g p_g + W1z_g z + W1g_g g + W1e_g e + b1_g) + b2_g,
  e = W s~ (W: n_e x 9, no bias, shared by every head). Critic 3 has n_e = 0 (B8); critic 1 is a copy of
  critic 3 with W1e = 0 and W from PCA (B9), so it equals critic 3 at initialisation for any s.
  The heads are stored stacked: block g of the first layer is rows g*H .. (g+1)*H; W1p is masked so each
  head sees only its knobs.
- FullCritic (critic 2, B10): MLP(p 15, e2 2, z 5, g 2 -> 32 -> 32 -> 1), SiLU; e2 = W2 s~.
- BarcelonaHead (B11): r(p) = Linear(15 -> 16) -> SiLU -> Linear(16 -> 1), last layer zero-initialised.
`scale` (= 1) and `b` are buffers / parameters frozen for the later OCD fine-tune (C1-R8).
`e_gate` (buffer, scalar, = 1): multiplies e wherever it enters the heads (critics 1 and 2). A hook for
the scalar gate / shrinkage of the style path of the remedy ladder (spec section 12.21, B23, not yet
specified); 1 = B9 / B10 exactly, 0 = the style-free heads (critic 1 then equals its critic 3). Not
trained in C2b.
"""
import math
import numpy as np
import torch
from torch import nn

N_P, N_Z, N_G, N_S = 15, 5, 2, 9


def _uniform(t: torch.Tensor, bound: float, gen):
    with torch.no_grad():
        t.uniform_(-bound, bound, generator=gen)


class GroupCritic(nn.Module):
    def __init__(self, groups, hidden: int = 16, n_e: int = 0, n_z: int = N_Z, n_g: int = N_G, n_s: int = N_S,
                 final_std: float = 1e-3, seed: int = 0):
        super().__init__()
        self.groups = tuple(tuple(int(j) for j in g) for g in groups)
        G, H = len(self.groups), hidden
        self.G, self.H, self.n_e, self.n_s = G, H, n_e, n_s
        mask = torch.zeros(G * H, N_P)
        for gi, g in enumerate(self.groups):
            mask[gi * H:(gi + 1) * H, list(g)] = 1.
        self.register_buffer('mask', mask)
        self.W1p = nn.Parameter(torch.zeros(G * H, N_P))
        self.W1z = nn.Parameter(torch.zeros(G * H, n_z))
        self.W1g = nn.Parameter(torch.zeros(G * H, n_g))
        self.b1 = nn.Parameter(torch.zeros(G * H))
        self.w2 = nn.Parameter(torch.zeros(G, H))
        self.b2 = nn.Parameter(torch.zeros(G))
        self.W1e = nn.Parameter(torch.zeros(G * H, n_e))
        self.W = nn.Parameter(torch.zeros(n_e, n_s))
        self.register_buffer('b', torch.zeros(()))
        self.register_buffer('scale', torch.ones(()))
        self.register_buffer('e_low', torch.full((n_e,), -math.inf))
        self.register_buffer('e_high', torch.full((n_e,), math.inf))
        self.register_buffer('e_gate', torch.ones(()))
        gen = torch.Generator().manual_seed(int(seed))
        for gi, g in enumerate(self.groups):          # nn.Linear(|g| + 5 + 2 -> H) default bounds per head
            bound = 1. / math.sqrt(len(g) + n_z + n_g)
            rows = slice(gi * H, (gi + 1) * H)
            for t in (self.W1p, self.W1z, self.W1g, self.b1):
                _uniform(t.data[rows], bound, gen)
        with torch.no_grad():
            self.W1p.mul_(self.mask)
            self.w2.normal_(0., final_std, generator=gen)

    def encode(self, s: torch.Tensor, clip: bool = False) -> torch.Tensor:
        e = s @ self.W.T
        return torch.maximum(torch.minimum(e, self.e_high), self.e_low) if clip else e

    def heads(self, p, z, g, e=None) -> torch.Tensor:
        """(..., G) head outputs h_g."""
        pre = p @ (self.W1p * self.mask).T + z @ self.W1z.T + g @ self.W1g.T + self.b1
        if self.n_e and e is not None:
            pre = pre + (self.e_gate * e) @ self.W1e.T
        hid = nn.functional.silu(pre).reshape(*pre.shape[:-1], self.G, self.H)
        return (hid * self.w2).sum(-1) + self.b2

    def forward(self, p, z, g, s=None, clip_e: bool = False, e=None) -> torch.Tensor:
        if e is None and self.n_e and s is not None:
            e = self.encode(s, clip_e)
        return self.scale * (self.b + self.heads(p, z, g, e).sum(-1))

    def setup_sum(self, p, z, g, e=None) -> torch.Tensor:
        """sum_g h_g (the term the paired guard acts on; b and scale excluded)."""
        return self.heads(p, z, g, e).sum(-1)

    @classmethod
    def from_critic3(cls, c3: 'GroupCritic', W_init, n_e: int = 2) -> 'GroupCritic':
        """Critic 1 initialisation (B9): every critic-3 weight copied, W1e = 0, W = W_init (n_e x 9)."""
        c1 = cls(c3.groups, c3.H, n_e=n_e, n_z=c3.W1z.shape[1], n_g=c3.W1g.shape[1], n_s=np.asarray(W_init).shape[1])
        with torch.no_grad():
            for name in ('W1p', 'W1z', 'W1g', 'b1', 'w2', 'b2'):
                getattr(c1, name).copy_(getattr(c3, name))
            c1.b.copy_(c3.b)
            c1.scale.copy_(c3.scale)
            c1.W1e.zero_()
            c1.W.copy_(torch.as_tensor(np.asarray(W_init), dtype=c1.W.dtype))
        return c1

    def inherited(self):
        return [self.W1p, self.W1z, self.W1g, self.b1, self.w2, self.b2]

    def centre_heads(self, p, z, g, e=None, w=None):
        """C1-R5: shift each head so its (weighted) mean over the given rows is 0; b absorbs it (J unchanged)."""
        with torch.no_grad():
            h = self.heads(p, z, g, e)
            m = h.mean(0) if w is None else (h * w[:, None]).sum(0) / w.sum()
            self.b2.sub_(m)
            self.b.add_(m.sum())


class FullCritic(nn.Module):
    """Critic 2 (B10). `use_e` False during the pretrain on A (the e2 columns are absent)."""

    def __init__(self, hidden: int = 32, n_e: int = 2, n_z: int = N_Z, n_g: int = N_G, n_s: int = N_S,
                 final_std: float = 1e-3, seed: int = 0):
        super().__init__()
        H = hidden
        self.n_e, self.n_s = n_e, n_s
        self.Wp, self.We = nn.Parameter(torch.zeros(H, N_P)), nn.Parameter(torch.zeros(H, n_e))
        self.Wz, self.Wg, self.b1 = nn.Parameter(torch.zeros(H, n_z)), nn.Parameter(torch.zeros(H, n_g)), nn.Parameter(torch.zeros(H))
        self.l2, self.l3 = nn.Linear(H, H), nn.Linear(H, 1)
        self.W = nn.Parameter(torch.zeros(n_e, n_s))
        self.register_buffer('b', torch.zeros(()))
        self.register_buffer('scale', torch.ones(()))
        self.register_buffer('e_low', torch.full((n_e,), -math.inf))
        self.register_buffer('e_high', torch.full((n_e,), math.inf))
        self.register_buffer('e_gate', torch.ones(()))
        self.use_e = False
        gen = torch.Generator().manual_seed(int(seed))
        bound = 1. / math.sqrt(N_P + n_e + n_z + n_g)          # nn.Linear(24 -> 32) default bounds
        for t in (self.Wp, self.Wz, self.Wg, self.b1):
            _uniform(t.data, bound, gen)
        _uniform(self.l2.weight.data, 1. / math.sqrt(H), gen)
        _uniform(self.l2.bias.data, 1. / math.sqrt(H), gen)
        with torch.no_grad():
            self.l3.weight.normal_(0., final_std, generator=gen)
            self.l3.bias.zero_()

    def encode(self, s, clip: bool = False):
        e = s @ self.W.T
        return torch.maximum(torch.minimum(e, self.e_high), self.e_low) if clip else e

    def setup_sum(self, p, z, g, e=None):
        pre = p @ self.Wp.T + z @ self.Wz.T + g @ self.Wg.T + self.b1
        if self.use_e and e is not None:
            pre = pre + (self.e_gate * e) @ self.We.T
        h = nn.functional.silu(self.l2(nn.functional.silu(pre)))
        return self.l3(h).squeeze(-1)

    def forward(self, p, z, g, s=None, clip_e: bool = False, e=None):
        if e is None and self.use_e and s is not None:
            e = self.encode(s, clip_e)
        return self.scale * (self.b + self.setup_sum(p, z, g, e))

    def inherited(self):
        return [self.Wp, self.Wz, self.Wg, self.b1, *self.l2.parameters(), *self.l3.parameters()]


class BarcelonaHead(nn.Module):
    """B11 residual head r_BCN(p); added to J iff the context track is Barcelona."""

    def __init__(self, hidden: int = 16, seed: int = 0):
        super().__init__()
        self.l1, self.l2 = nn.Linear(N_P, hidden), nn.Linear(hidden, 1)
        gen = torch.Generator().manual_seed(int(seed))
        bound = 1. / math.sqrt(N_P)
        _uniform(self.l1.weight.data, bound, gen)
        _uniform(self.l1.bias.data, bound, gen)
        with torch.no_grad():
            self.l2.weight.zero_()
            self.l2.bias.zero_()

    def forward(self, p):
        return self.l2(nn.functional.silu(self.l1(p))).squeeze(-1)


__all__ = ['BarcelonaHead', 'FullCritic', 'GroupCritic', 'N_G', 'N_P', 'N_S', 'N_Z']
