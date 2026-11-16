import pytest, torch
from rlsetup.critic.ensemble import EnsembleCritic


class _M:
    def __init__(self, kind, k):
        self.kind, self.k = kind, k

    def context(self, **kw):
        return None

    def J(self, p, c, prior=True):
        return self.k * p.sum(-1)

    def _J_twins(self, p, c, prior=True):
        return torch.stack([self.k * p.sum(-1), self.k * p.sum(-1) + 1], 0)


def test_needs_style_aware_critic1():
    with pytest.raises(ValueError):
        EnsembleCritic([_M('critic3', 1.)])
    with pytest.raises(ValueError):
        EnsembleCritic([_M('critic1', 1.), _M('critic3', 1.)])


def test_mean_and_q_min():
    e = EnsembleCritic([_M('critic1', 1.), _M('critic2', 3.)])
    p = torch.ones(4, 15)
    c = e.context()
    assert torch.allclose(e.q(p, c), -2 * p.sum(-1))
    assert (e.q_min(p, c) <= e.q(p, c)).all()
