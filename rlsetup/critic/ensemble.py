"""Ensemble TD3 Q (lead, 16 Nov 2026; post-hoc, see docs/results/critic_c2.md "Post-hoc ensemble check").

Q = -mean over members of J_member(p, ctx). Members are style-aware critic bundles (critic 1 = grouped heads,
critic 2 = full-interaction MLP), so the ensemble keeps the driver-style path (spec 12.21) and is smooth in p.
Offline check on the saved CV predictions: critic 3 + critic-2-pretrain 50/50 ranks unseen drivers' setups
+0.016 [+0.005, +0.028] better than critic 3 alone (F_person, 3 repeats).

Every member builds its own context (each has its own track/style scalers). Support box and feasibility come
from the critic-1 member (TD3's Q of record). q_min = -max over every twin of every member (pessimistic).
"""
import json
from pathlib import Path

import torch

from rlsetup.critic.bundle import CriticBundle

MANIFEST_FORMAT = 'c2-ensemble-td3-q-1'


class EnsembleCritic:
    def __init__(self, members, weights=None):
        if not members:
            raise ValueError('ensemble needs members')
        kinds = [m.kind for m in members]
        if 'critic1' not in kinds:
            raise ValueError('an ensemble TD3 Q must contain the style-aware critic 1 (spec 12.21)')
        if any(k not in ('critic1', 'critic2') for k in kinds):
            raise ValueError(f'every member must be style-aware (critic 1 or critic 2), got {kinds}')
        w = torch.ones(len(members)) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        self.members, self.weights = list(members), w / w.sum()
        self.primary = members[kinds.index('critic1')]
        self.role = 'td3_q'

    @classmethod
    def load(cls, manifest_path) -> 'EnsembleCritic':
        mp = Path(manifest_path)
        man = json.loads(mp.read_text())
        if man.get('format') != MANIFEST_FORMAT:
            raise ValueError(f'{mp}: not a {MANIFEST_FORMAT} manifest')
        members = [CriticBundle.load(mp.parent / p) for p in man['members']]
        return cls(members, man.get('weights'))

    def context(self, track=None, geometry=None, style=None, source=None):
        return [m.context(track=track, geometry=geometry, style=style, source=source) for m in self.members]

    def _stack(self, p, ctxs, prior=True):
        return torch.stack([m.J(p, c, prior) for m, c in zip(self.members, ctxs)], 0)

    def J(self, p, ctxs, prior: bool = True) -> torch.Tensor:
        Js = self._stack(p, ctxs, prior)
        return (self.weights.view(-1, *[1] * (Js.dim() - 1)) * Js).sum(0)

    def q(self, p, ctxs, prior: bool = True) -> torch.Tensor:
        return -self.J(p, ctxs, prior)

    def q_min(self, p, ctxs, prior: bool = True) -> torch.Tensor:
        tw = torch.cat([m._J_twins(p, c, prior) for m, c in zip(self.members, ctxs)], 0)
        return -tw.max(0).values

    def knob_feasibility(self, p):
        return self.primary.knob_feasibility(p)

    def support_box(self):
        return self.primary.support_box()

    def off_support(self, p):
        return self.primary.off_support(p)

    def style_report(self, style, track=None):
        return {m.kind: m.style_report(style, track) for m in self.members}


def write_manifest(path, members, weights=None, note=''):
    man = {'format': MANIFEST_FORMAT, 'role': 'td3_q', 'members': list(members), 'weights': weights, 'note': note}
    Path(path).write_text(json.dumps(man, indent=1))


__all__ = ['EnsembleCritic', 'write_manifest', 'MANIFEST_FORMAT']
