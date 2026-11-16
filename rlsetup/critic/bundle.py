"""C2b_addendum B17: the saved critic and its query interface (pure torch + numpy, CPU, no pandas).

    b = CriticBundle.load(path)                 # kind in {'critic1', 'critic2', 'critic3'}
    ctx = b.context(track=None, geometry=None, style=None, source=None)
    b.J(p, ctx, prior=True) -> % gap (lower better, mean over twins); b.q = -J; b.q_min = -max over twins
    b.prior(p, ctx) -> lambda Pi (B21 meta prior; 0 without one); ctx.flags['meta_prior'] in {track, pooled_w, none}
    b.knob_feasibility(p) -> min K_b [N m/rad]; b.support_box(); b.off_support(p) -> bool
    b.style_report(style, track) -> {'clipped', 'mahalanobis', 'centre'}
    b.groups, b.style_scaler, b.track_scaler, b.provenance

p: (..., 15) normalised knobs in KNOB_ORDER (critic/knobs.normalise of the adapter's physical knobs),
differentiable. Context:
- track: canonical key (e.g. 'spain'); its saved raw z is standardised with the saved TrackScaler.
  An unknown track needs `geometry` ((N, 2) closed centre line [m]) -> z by `track_features`.
- style: the 12 raw style-rl numbers (KEYS order). None -> s~ = 0 (the training median), e = 0. Centred
  with the saved per-track medians; an unknown track uses the global median and is flagged. e is
  clipped to the training range (R10).
- source: None -> uniform 1 / n_sources (R8); a known source -> one-hot; an unseen source (e.g. an
  Assetto Corsa adapter's name) -> uniform, flagged.
Roles (spec section 12.21): TD3's Q is always a critic-1 bundle (`role='td3_q'`; `save_bundle` refuses
any other kind); critic 3 is saved alongside as `role='style_free_reference'` and is never TD3's Q;
critic 2 is `role='comparison'`. `CriticBundle.load(path, require_td3_q=True)` raises unless the file is
the delivered TD3 Q. The style path can be gated by the models' `e_gate` buffer (B23 hook, = 1).
The Barcelona head is added iff the context track is Barcelona. A data-source adapter never imports
this module's model code; it only maps its own track ids to canonical keys and its setups to knobs.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import torch
from .c2_config import BARCELONA
from .config import CarConfig, KNOB_ORDER, TrackFeatureConfig
from .knobs import knob_feasibility as _knob_feasibility
from .meta_prior import MetaPrior
from .model import BarcelonaHead, FullCritic, GroupCritic
from .style_transform import TrackStyleTransform
from .track_features import TrackScaler, track_features

FORMAT = 'c2b-critic-bundle-1'
ROLES = {'critic1': 'td3_q', 'critic3': 'style_free_reference', 'critic2': 'comparison'}


@dataclass
class Context:
    z: torch.Tensor                 # (5,) standardised track features
    g: torch.Tensor                 # (n_sources,)
    s: torch.Tensor = None          # (9,) s~ or None (e = 0)
    barcelona: bool = False
    flags: dict = field(default_factory=dict)
    meta_m: torch.Tensor = None     # (15,) B21 m at the context (g-weighted), None = no prior here
    meta_w: torch.Tensor = None     # (15,) B21 w at the context track


def _state_to_lists(st):
    return {k: v for k, v in st.items()}


def save_bundle(path, kind: str, twins: list, *, sources, track_z_raw: dict, z_provisional: dict, style_train=None,
                provenance: dict = None, car: CarConfig = CarConfig(), role: str = None) -> Path:
    """Write `<path>` (.pt) and `<path minus .pt>.json`. `twins`: train.Fitted objects of one kind
    (critic 3 twins / critic 1 twins / one critic 2); `style_train` (n, 9) the training s~ (style
    report of critics 1 and 2). `role`: default ROLES[kind]; 'td3_q' only for kind 'critic1' (12.21)."""
    role = ROLES[kind] if role is None else role
    if role == 'td3_q' and kind != 'critic1':
        raise ValueError(f"TD3's Q is always critic 1 (spec 12.21); a {kind} bundle cannot have role 'td3_q'")
    path = Path(path)
    f0 = twins[0]
    prior = f0.prior if f0.prior is not None and f0.prior.lam != 0. else None
    if any((f.prior is not None and f.prior.source != 'in_fold') for f in twins):
        raise ValueError('the published meta prior is sensitivity only and is never written into a bundle (B21.1)')
    if prior is not None and any(f.prior is None or f.prior.lam != prior.lam or not np.array_equal(f.prior.w, prior.w, equal_nan=True)
                                 for f in twins):
        raise ValueError('twins must share lambda, m and w (B21.3)')
    arch = {'kind': kind, 'n_g': len(sources)}
    if kind in ('critic3', 'critic1'):
        arch.update(groups=[list(g) for g in f0.model.groups], hidden=f0.model.H, n_e=f0.model.n_e, n_s=f0.model.n_s)
    else:
        arch.update(hidden=f0.model.l2.in_features, n_e=f0.model.n_e, n_s=f0.model.n_s)
    meta = {'format': FORMAT, 'kind': kind, 'role': role, 'arch': arch, 'knob_order': list(KNOB_ORDER), 'sources': list(sources),
            'track_z_raw': {t: [float(x) for x in v] for t, v in track_z_raw.items()},
            'z_provisional': {t: bool(v) for t, v in z_provisional.items()},
            'track_scaler': {'mean': f0.scaler.mean.tolist(), 'std': f0.scaler.std.tolist(), 'z_clip': f0.scaler.z_clip},
            'style_transform': f0.style_tf.state() if f0.style_tf is not None else None,
            'support_box': [np.asarray(f0.support[0]).tolist(), np.asarray(f0.support[1]).tolist()],
            'car': {k: float(v) for k, v in vars(car).items()}, 'barcelona': BARCELONA,
            'twin_info': [f.info for f in twins], 'provenance': provenance or {},
            'meta_prior': prior.state() if prior is not None else None,
            'e_gate': [float(f.model.e_gate) for f in twins]}
    if style_train is not None:
        s = np.asarray(style_train, float)
        cov = np.cov(s.T) + 1e-9 * np.eye(s.shape[1])
        meta['style_train'] = {'mean': s.mean(0).tolist(), 'cov_inv': np.linalg.inv(cov).tolist(), 'n': int(len(s))}
    blob = {'meta_json': json.dumps(meta, default=float),
            'twins': [{'model': {k: v.detach().clone() for k, v in f.model.state_dict().items()},
                       'bcn': ({k: v.detach().clone() for k, v in f.bcn.state_dict().items()} if f.bcn is not None else {})}
                      for f in twins]}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    return path


class CriticBundle:
    def __init__(self, meta: dict, twins: list):
        self.meta, self.kind, self.role = meta, meta['kind'], meta.get('role')
        self.provenance = meta['provenance']
        self.sources = tuple(meta['sources'])
        a = meta['arch']
        self.groups = tuple(tuple(g) for g in a.get('groups', ())) or None
        self.models, self.heads = [], []
        for t in twins:
            if self.kind in ('critic3', 'critic1'):
                m = GroupCritic(a['groups'], a['hidden'], n_e=a['n_e'], n_g=a['n_g'], n_s=a['n_s'])
            else:
                m = FullCritic(a['hidden'], n_e=a['n_e'], n_g=a['n_g'], n_s=a['n_s'])
                m.use_e = True
            m.load_state_dict(t['model'])
            m.eval()
            for q in m.parameters():
                q.requires_grad_(False)
            self.models.append(m)
            h = None
            if t['bcn']:
                h = BarcelonaHead(t['bcn']['l1.weight'].shape[0])
                h.load_state_dict(t['bcn'])
                for q in h.parameters():
                    q.requires_grad_(False)
            self.heads.append(h)
        ts = meta['track_scaler']
        self.track_scaler = TrackScaler(ts['z_clip'])
        self.track_scaler.mean, self.track_scaler.std = np.array(ts['mean']), np.array(ts['std'])
        st = meta['style_transform']
        self.style_scaler = TrackStyleTransform.from_state(st) if st is not None else None
        self.track_z_raw = {t: np.array(v, float) for t, v in meta['track_z_raw'].items()}
        self._low, self._high = (np.array(v, float) for v in meta['support_box'])
        self.car = CarConfig(**meta['car'])
        mp = meta.get('meta_prior')
        self.meta_prior = MetaPrior.from_state(mp) if mp is not None else None

    @classmethod
    def load(cls, path, require_td3_q: bool = False) -> 'CriticBundle':
        blob = torch.load(Path(path), map_location='cpu', weights_only=True)
        meta = json.loads(blob['meta_json'])
        if meta.get('format') != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        if require_td3_q and (meta.get('role') != 'td3_q' or meta['kind'] != 'critic1'):
            raise ValueError(f"{path}: role {meta.get('role')!r} ({meta['kind']}) is not TD3's Q (critic 1, spec 12.21)")
        return cls(meta, blob['twins'])

    # ---- context ------------------------------------------------------------------------------------
    def _z(self, track, geometry):
        if track is not None and track in self.track_z_raw:
            raw = self.track_z_raw[track]
        elif geometry is not None:
            raw = track_features(np.asarray(geometry, float), TrackFeatureConfig())
        else:
            raise ValueError(f'unknown track {track!r}: pass its centre line as `geometry`')
        z = np.clip((raw - self.track_scaler.mean) / self.track_scaler.std, -self.track_scaler.z_clip, self.track_scaler.z_clip)
        return np.where(np.isfinite(z), z, 0.)

    def context(self, track=None, geometry=None, style=None, source=None) -> Context:
        flags = {'source': 'uniform' if source is None else ('known' if source in self.sources else 'unseen_uniform')}
        g = np.full(len(self.sources), 1. / len(self.sources))
        if source in self.sources:
            g = np.eye(len(self.sources))[self.sources.index(source)]
        z = self._z(track, geometry)
        s = None
        if style is not None and self.style_scaler is not None:
            st, where = self.style_scaler.transform(np.asarray(style, float).reshape(1, -1), [track])
            s = torch.as_tensor(st[0], dtype=torch.float32)
            flags['style_centre'] = str(where[0])
        flags['track_known'] = track in self.track_z_raw
        mm = mw = None
        flags['meta_prior'] = 'none'
        if self.meta_prior is not None:
            m, flag = self.meta_prior.m_query(track, g)
            flags['meta_prior'] = flag
            if m is not None:
                t = self.meta_prior.tracks.index(track)
                mm, mw = (torch.as_tensor(v, dtype=torch.float32) for v in (m, self.meta_prior.w[t]))
        return Context(z=torch.as_tensor(z, dtype=torch.float32), g=torch.as_tensor(g, dtype=torch.float32), s=s,
                       barcelona=track == BARCELONA, flags=flags, meta_m=mm, meta_w=mw)

    # ---- outputs ------------------------------------------------------------------------------------
    def prior(self, p, ctx: Context) -> torch.Tensor:
        """lambda Pi(p) at the context (B21; 0 without a prior or where the track has no m)."""
        p = torch.as_tensor(p, dtype=torch.float32) if not torch.is_tensor(p) else p
        if self.meta_prior is None or ctx.meta_m is None or self.meta_prior.lam == 0.:
            return torch.zeros(p.shape[:-1])
        return self.meta_prior.lam * MetaPrior.pi_torch(p, ctx.meta_m, ctx.meta_w)

    def _J_twins(self, p, ctx: Context, prior: bool = True) -> torch.Tensor:
        p = torch.as_tensor(p, dtype=torch.float32) if not torch.is_tensor(p) else p
        z = ctx.z.expand(*p.shape[:-1], ctx.z.shape[-1])
        g = ctx.g.expand(*p.shape[:-1], ctx.g.shape[-1])
        out = []
        for m, h in zip(self.models, self.heads):
            e = None
            if self.kind in ('critic1', 'critic2'):
                n_e = m.W.shape[0]
                e = (m.encode(ctx.s, clip=True) if ctx.s is not None else torch.zeros(n_e)).expand(*p.shape[:-1], n_e)
            J = m(p, z, g, e=e)
            if ctx.barcelona and h is not None:
                J = J + h(p)
            out.append(J)
        J = torch.stack(out, 0)
        return J + self.prior(p, ctx) if prior else J

    def J(self, p, ctx: Context, prior: bool = True) -> torch.Tensor:
        return self._J_twins(p, ctx, prior).mean(0)

    def q(self, p, ctx: Context, prior: bool = True) -> torch.Tensor:
        return -self.J(p, ctx, prior)

    def q_min(self, p, ctx: Context, prior: bool = True) -> torch.Tensor:
        return -self._J_twins(p, ctx, prior).max(0).values

    # ---- feasibility and support --------------------------------------------------------------------
    def knob_feasibility(self, p):
        return _knob_feasibility(p, self.car)

    def support_box(self):
        return self._low.copy(), self._high.copy()

    def off_support(self, p):
        if torch.is_tensor(p):
            return ((p < p.new_tensor(self._low)) | (p > p.new_tensor(self._high))).any(-1)
        p = np.asarray(p)
        return ((p < self._low) | (p > self._high)).any(-1)

    def style_report(self, style, track=None) -> dict:
        """{'clipped': per e-dim bool (per twin), 'mahalanobis': of s~ to the training s~, 'centre'}."""
        if self.style_scaler is None:
            return {'clipped': None, 'mahalanobis': None, 'centre': None}
        st, where = self.style_scaler.transform(np.asarray(style, float).reshape(1, -1), [track])
        s = torch.as_tensor(st[0], dtype=torch.float32)
        clipped = []
        for m in self.models:
            e = m.encode(s)
            clipped.append(((e < m.e_low) | (e > m.e_high)).tolist())
        mt = self.meta.get('style_train')
        maha = None
        if mt is not None:
            dlt = st[0] - np.array(mt['mean'])
            maha = float(np.sqrt(max(dlt @ np.array(mt['cov_inv']) @ dlt, 0.)))
        return {'clipped': clipped, 'mahalanobis': maha, 'centre': str(where[0])}


__all__ = ['CriticBundle', 'Context', 'FORMAT', 'ROLES', 'save_bundle']
