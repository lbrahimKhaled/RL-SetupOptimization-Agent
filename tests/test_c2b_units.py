"""C2b unit tests (C2b_addendum B19) on the synthetic data build (tests/c2b_fixture.py): folds, leakage,
the B1 label, q's independence of the nuisance, critic-1 initialisation and frozen weights, gradients,
e clipping, the strata shuffle, twins, ranking pairs, the Barcelona head, the bundle and a non-F1
adapter. Tiny training budgets (the defaults are the addendum's)."""
from dataclasses import replace
import inspect
import subprocess
import sys
import numpy as np
import pytest
import torch
from c2b_fixture import write_c2b_dataset
import c2b_dummy_adapter as dummy
from rlsetup.critic import train as T
from rlsetup.critic.bundle import CriticBundle, save_bundle
from rlsetup.critic.c2_config import BARCELONA, C2bConfig
from rlsetup.critic.c2data import load_c2data
from rlsetup.critic.config import REPO_ROOT
from rlsetup.critic.dataset import robust_label
from rlsetup.critic.folds import data_hash_of, load_folds, make_folds, rows_digest, write_folds
from rlsetup.critic.groups import pca_groups, physics_groups
from rlsetup.critic.metrics import pair_outcomes, pairs_of, singleton_rows, strata_shuffle
from rlsetup.critic.model import BarcelonaHead, GroupCritic

C = C2bConfig()
TINY = replace(C, pretrain=replace(C.pretrain, max_epochs=4, patience=3, weight_decay_grid=(0., 1e-4)),
               finetune=replace(C.finetune, max_steps=30, patience=10, l2_grid=(1e-3, 1e-1)),
               bcn=replace(C.bcn, max_epochs=4, patience=3))


@pytest.fixture(scope='module')
def d(tmp_path_factory):
    return load_c2data(write_c2b_dataset(tmp_path_factory.mktemp('c2b_units') / 'build'))


@pytest.fixture(scope='module')
def folds(d):
    return make_folds(d.data_hash, np.array(d.games)[d.game], d.driver, np.array(d.track_names)[d.track], d.person,
                      d.setup_hash)


@pytest.fixture(scope='module')
def split(d, folds):
    te = np.flatnonzero(folds['F_cell'][0] == 0)
    tr = np.flatnonzero(folds['F_cell'][0] != 0)
    etr = tr[d.exact[tr] & d.has_style[tr]]
    ete = te[d.exact[te] & d.has_style[te]]
    return tr, te, etr, ete


@pytest.fixture(scope='module')
def fits(d, split):
    tr, te, etr, ete = split
    torch.set_num_threads(1)
    f3 = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u')
    f1 = T.finetune(d, etr, f3, 'critic1', TINY, seed=5, salt='u')
    return f3, f1


def _c(d, **arrays):
    return replace(d, **arrays)


# ---- folds (B6) -------------------------------------------------------------------------------------
def test_folds_deterministic_and_keyed_to_data_hash(d, folds, tmp_path):
    again = make_folds(d.data_hash, np.array(d.games)[d.game], d.driver, np.array(d.track_names)[d.track], d.person,
                       d.setup_hash)
    assert all(np.array_equal(folds[k], again[k]) for k in folds)
    other = make_folds('0' * 64, np.array(d.games)[d.game], d.driver, np.array(d.track_names)[d.track], d.person,
                       d.setup_hash)
    assert not np.array_equal(folds['F_cell'], other['F_cell'])
    write_folds(tmp_path / 'folds.json', folds, d.data_hash, d.rows_sha256)
    got, sha = load_folds(tmp_path / 'folds.json', d.data_hash, d.rows_sha256)
    assert np.array_equal(got['F_person'], folds['F_person']) and len(sha) == 64
    with pytest.raises(ValueError, match='data hash'):
        load_folds(tmp_path / 'folds.json', data_hash_of({'x': 'y'}))
    with pytest.raises(ValueError, match='row digest'):
        load_folds(tmp_path / 'folds.json', d.data_hash, rows_digest(d.setup_hash[::-1], d.lap_s[::-1]))


def test_fold_units(d, folds):
    """F_cell: every row of a cell (exact rows included) in one fold; F_person: a person in one fold
    across both games; F_hash: one setup in one fold."""
    for name, key in (('F_cell', d.cell_key), ('F_person', d.person), ('F_hash', d.setup_hash)):
        for r in range(folds[name].shape[0]):
            f = folds[name][r]
            for k in np.unique(key):
                assert len(np.unique(f[key == k])) == 1
    both = [p for p in np.unique(d.person) if len(np.unique(d.game[d.person == p])) == 2]
    assert both                                     # the fixture has cross-game persons
    assert set(np.unique(folds['F_cell'][0])) == set(range(5)) and folds['F_cell'].shape[0] == 3


# ---- B1 label ---------------------------------------------------------------------------------------
def _ref_label(lap, group, z_max=3.5, passes=2):
    keep = np.ones(len(lap), bool)
    for _ in range(passes):
        new = keep.copy()
        for g in np.unique(group):
            r = np.flatnonzero(group == g)
            k = r[keep[r]]
            m = np.median(lap[k])
            gap = 100 * (lap[r] / m - 1)
            dd = 1.4826 * np.median(np.abs(100 * (lap[k] / m - 1)))
            if dd > 0:
                new[r] = keep[r] & (np.abs(gap / dd) <= z_max)
        keep = new
    return keep


def test_robust_label_two_passes_exact_counts():
    """Hand case: 11 laps around 100 s; 104 s and 103.5 s (slow) and 97 s (fast) are dropped in pass 1;
    100.9 s passes pass 1 (|z| < 3.5) and is dropped in pass 2, once m and d are recomputed on the kept rows."""
    core = 100 + np.array([-.3, -.2, -.15, -.1, -.05, 0., .05, .1, .15, .2, .25])
    lap = np.r_[core, 104., 100.9, 97., 103.5]
    keep, gap, med, rep = robust_label(lap, np.zeros(len(lap), int))
    assert rep['passes'] == [{'kept': 12, 'dropped_low': 1, 'dropped_high': 2},
                             {'kept': 11, 'dropped_low': 0, 'dropped_high': 1}]
    assert keep.sum() == 11 and not keep[11:].any()
    m = np.median(lap[np.r_[np.arange(11), 12]])            # the pass-2 median (rows kept after pass 1)
    np.testing.assert_allclose(gap, 100 * (lap / m - 1))
    np.testing.assert_allclose(med, m)
    rng = np.random.default_rng(1)
    lap = 90 * (1 + rng.standard_t(2, 600) / 100)
    group = rng.integers(0, 12, 600)
    keep, gap, med, rep = robust_label(lap, group)
    assert np.array_equal(keep, _ref_label(lap, group)) and len(rep['passes']) == 2
    assert rep['passes'][1]['kept'] == keep.sum() <= rep['passes'][0]['kept']


def test_mexico_dropped_after_person_ids(tmp_path):
    from c2_fixture import TRACK_MAP, write_fixture
    from rlsetup.critic.config import Config, DataConfig
    from rlsetup.critic.sources.f1laps import build_frame, load_exports
    assert DataConfig().drop_tracks == ('mexico',)
    fx = write_fixture(tmp_path)
    full = load_exports(fx['sources'], log=lambda *a: None)
    victim = sorted({s['track'] for s in full.setups})[0]
    cut = load_exports(fx['sources'], log=lambda *a: None, drop_tracks=(victim,))
    assert victim not in {s['track'] for s in cut.setups} and cut.n_persons == full.n_persons
    cfg = Config()
    fr = build_frame(cut, replace(cfg, data=replace(cfg.data, drop_tracks=(victim,))), log=lambda *a: None)
    assert victim not in fr.track_names or not (np.array(fr.track_names)[fr.track] == victim).any()


# ---- leakage ----------------------------------------------------------------------------------------
def test_no_leakage_from_held_out_rows(d, split, fits):
    """Held-out rows x 1000 (label, knobs, lap, style): the track scaler, nuisance, support box, PCA
    groups, critic weights, style transform (track medians) and the W initialisation are unchanged."""
    tr, te, etr, ete = split
    f3, f1 = fits
    bad = lambda a: np.where(np.isin(np.arange(d.n), te)[(...,) + (None,) * (a.ndim - 1)], a * 1000., a)
    d2 = _c(d, gap=bad(d.gap), p=bad(d.p), lap_s=bad(d.lap_s), style_raw=bad(d.style_raw))
    g3 = T.pretrain(d2, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u')
    g1 = T.finetune(d2, etr, g3, 'critic1', TINY, seed=5, salt='u')
    for a, b in ((f3, g3), (f1, g1)):
        sa, sb = a.model.state_dict(), b.model.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
        na, nb = a.nuis.state_dict(), b.nuis.state_dict()
        assert all(torch.equal(na[k], nb[k]) for k in na)
        assert np.array_equal(a.support[0], b.support[0]) and np.array_equal(a.support[1], b.support[1])
        assert np.array_equal(a.scaler.mean, b.scaler.mean) and np.array_equal(a.scaler.std, b.scaler.std)
    st1, st2 = f1.style_tf.state(), g1.style_tf.state()
    assert st1['track_median'].keys() == st2['track_median'].keys() and st1['n_train_cells'] == st2['n_train_cells']
    for t in st1['track_median']:
        np.testing.assert_array_equal(np.array(st1['track_median'][t], float), np.array(st2['track_median'][t], float))
    for k in ('global_median', 'center', 'scale'):
        np.testing.assert_array_equal(np.array(st1[k], float), np.array(st2[k], float))
    s1 = T.style_inputs(d, etr, f1.style_tf)[0]
    assert np.array_equal(T.style_pca_init(s1), T.style_pca_init(T.style_inputs(d2, etr, g1.style_tf)[0]))
    keys = lambda x: [f'{t}|{h}' for t, h in zip(x.track[tr], x.setup_hash[tr])]
    assert pca_groups(d.p[tr], keys(d)) == pca_groups(d2.p[tr], keys(d2))


def test_track_scaler_uses_gated_training_tracks_only(d, split):
    tr = split[0]
    sc = T.fit_track_scaler(d, tr)
    z2 = d.z_raw.copy()
    z2[d.z_provisional] *= 1000.                    # provisional z (B3) never enters the scaler fit
    sc2 = T.fit_track_scaler(_c(d, z_raw=z2), tr)
    assert np.array_equal(sc.mean, sc2.mean) and np.array_equal(sc.std, sc2.std)


# ---- q, critic 1 ------------------------------------------------------------------------------------
def test_q_independent_of_nuisance_descriptors_and_m_style(d, split, fits):
    tr, te, etr, ete = split
    f3, f1 = fits
    rng = np.random.default_rng(0)
    perm = lambda a: a[rng.permutation(len(a))]
    d2 = _c(d, driver=perm(d.driver), team=perm(d.team), session=perm(d.session), controller=perm(d.controller),
            date_days=perm(d.date_days), person=perm(d.person), assists={k: perm(v) for k, v in d.assists.items()})
    for f, rows in ((f3, te), (f1, ete)):
        assert np.array_equal(T.predict(f, d, rows)['setup'], T.predict(f, d2, rows)['setup'])
    before = T.predict(f1, d, ete)
    saved = f1.nuis.dense.detach().clone()
    with torch.no_grad():
        f1.nuis.dense.add_(torch.randn(saved.shape, generator=torch.Generator().manual_seed(0)))
    after = T.predict(f1, d, ete)
    with torch.no_grad():
        f1.nuis.dense.copy_(saved)
    assert np.array_equal(before['setup'], after['setup']) and not np.allclose(before['full'], after['full'])


def test_critic1_initialisation_and_frozen_inherited_weights(d, split, fits):
    tr, te, etr, ete = split
    f3, f1 = fits
    W0 = T.style_pca_init(T.style_inputs(d, etr, f1.style_tf)[0])
    c1 = GroupCritic.from_critic3(f3.model, W0, 2)
    assert float(c1.W.detach().abs().sum()) > 0 and float(c1.W1e.detach().abs().sum()) == 0
    gen = torch.Generator().manual_seed(1)
    p, z, g = torch.rand(64, 15, generator=gen) * 2 - 1, torch.randn(64, 5, generator=gen), torch.eye(2)[torch.arange(64) % 2]
    for scale in (0., 1., 100.):
        s = scale * torch.randn(64, 9, generator=gen)
        assert torch.allclose(c1(p, z, g, s=s), f3.model(p, z, g), atol=1e-7, rtol=0)
    for a, b in zip(f1.model.inherited(), f3.model.inherited()):
        assert torch.equal(a, b)
    assert torch.equal(f1.model.b, f3.model.b) and float(f1.model.W1e.abs().sum()) > 0


def test_gradients_finite(d, split, fits, tmp_path):
    tr, te, etr, ete = split
    f3, f1 = fits
    path = save_bundle(tmp_path / 'c1.pt', 'critic1', [f1], sources=d.games, track_z_raw={t: d.z_raw[i] for i, t in enumerate(d.track_names)},
                       z_provisional={t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)},
                       style_train=T.style_inputs(d, etr, f1.style_tf)[0])
    b = CriticBundle.load(path)
    style = d.style_raw[ete[0]].copy()
    style[3] = np.nan
    ctx = b.context(track='monza', style=style, source='F1 25')
    p = torch.zeros(8, 15, requires_grad=True)
    b.q(p, ctx).sum().backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
    s = ctx.s.clone().requires_grad_(True)
    m = b.models[0]
    out = m(p.detach(), ctx.z.expand(8, 5), ctx.g.expand(8, 2), s=s.expand(8, 9))
    out.sum().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(b.q(p.detach(), ctx)).all()


def test_e_clipped_and_logged(d, split, fits):
    tr, te, etr, ete = split
    f1 = fits[1]
    m = f1.model
    assert torch.isfinite(m.e_low).all() and (m.e_low < m.e_high).all()
    style = d.style_raw.copy()
    style[ete] = style[ete] ** 4                       # far outside the training style
    pr = T.predict(f1, _c(d, style_raw=style), ete)
    assert 0 < pr['clipped_frac'] <= 1
    s, _ = T.style_inputs(d, ete, f1.style_tf)
    e = m.encode(torch.as_tensor(s * 50, dtype=torch.float32), clip=True)
    assert ((e >= m.e_low - 1e-7) & (e <= m.e_high + 1e-7)).all()
    src = inspect.getsource(sys.modules['critic_c2'].run_R5) if 'critic_c2' in sys.modules else \
        (REPO_ROOT / 'scripts/critic_c2.py').read_text()
    assert "clipped_frac_test" in src


# ---- ablation, twins, pairs -------------------------------------------------------------------------
def test_strata_shuffle():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 3))
    strata = np.r_[np.repeat([0, 1, 2], 12), [7, 8, 9, 9]]
    rows = np.r_[np.arange(30), [36, 37, 38]]
    a, b = strata_shuffle(values, strata, rows, 4), strata_shuffle(values, strata, rows, 4)
    assert np.array_equal(a, b) and not np.array_equal(a, strata_shuffle(values, strata, rows, 5))
    out = np.setdiff1d(np.arange(40), rows)
    assert np.array_equal(a[out], values[out])
    assert np.array_equal(a[[36, 37]], values[[36, 37]])          # singleton strata among the rows
    assert singleton_rows(strata, rows) == 3                      # strata 2 has 6 rows in `rows`, 7 and 8 singletons, 9 one
    for k in (0, 1, 2):
        r = rows[strata[rows] == k]
        assert sorted(map(tuple, a[r])) == sorted(map(tuple, values[r]))
    assert not np.array_equal(a[:12], values[:12])


def test_twins_bootstrap_units_and_q_min(d, split, fits, tmp_path):
    tr = split[0]
    w = T.boot_weights(d.setup_hash[tr], 1)
    for h in np.unique(d.setup_hash[tr])[:50]:
        assert len(np.unique(w[d.setup_hash[tr] == h])) == 1
    wc = T.boot_weights(d.cell_key[tr], 1)
    for k in np.unique(d.cell_key[tr]):
        assert len(np.unique(wc[d.cell_key[tr] == k])) == 1
    assert (w == 0).any() and (w > 1).any()
    src = (REPO_ROOT / 'scripts/critic_c2.py').read_text()
    assert 'boot_weights(d.setup_hash[train]' in src and "boot_weights(d.cell_key[etr], T.seed_of('boot_cell'" in src
    f3 = fits[0]
    f3b = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=11, weights=w, salt='u2')
    zr = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    b = CriticBundle.load(save_bundle(tmp_path / 'c3.pt', 'critic3', [f3, f3b], sources=d.games, track_z_raw=zr, z_provisional=zp))
    ctx = b.context(track='monza', source='F1 26')
    p = torch.as_tensor(d.p[:32], dtype=torch.float32)
    Jt = b._J_twins(p, ctx)
    assert torch.equal(b.q_min(p, ctx), -Jt.max(0).values) and torch.allclose(b.q(p, ctx), -Jt.mean(0))


def test_pairs_within_session_and_ties():
    game = np.zeros(6, int)
    driver = np.zeros(6, int)
    track = np.zeros(6, int)
    session = np.array([0, 0, 0, 1, 1, 2])
    h = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
    y = np.array([1., 2., 3., 1., 1.5, 0.])
    pairs = pairs_of(game, driver, track, session, h, y)
    assert len(pairs) == 4 and (session[pairs[:, 0]] == session[pairs[:, 1]]).all()
    pred = np.array([0., 0., 5., 2., 1., 0.])
    out = dict(zip(map(tuple, pairs.tolist()), pair_outcomes(pairs, pred, y)))
    assert out[(0, 1)] == .5 and out[(0, 2)] == 1. and out[(3, 4)] == 0.


# ---- Barcelona head, bundle -------------------------------------------------------------------------
def test_barcelona_head(d, split, fits, tmp_path):
    tr = split[0]
    f3 = fits[0]
    p = torch.rand(16, 15) * 2 - 1
    assert float(BarcelonaHead(seed=2)(p).detach().abs().max()) == 0.
    head, info = T.fit_bcn_head(d, tr, f3, TINY.bcn, seed=1)
    with torch.no_grad():
        head.l2.weight.add_(.1)                         # make it visibly non-zero for the test
    f = replace(f3, bcn=head)
    zr = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    b = CriticBundle.load(save_bundle(tmp_path / 'b.pt', 'critic3', [f], sources=d.games, track_z_raw=zr, z_provisional=zp))
    for track in d.track_names:
        ctx = b.context(track=track, source='F1 25')
        base = f3.model(p, ctx.z.expand(16, 5), ctx.g.expand(16, 2))
        want = base + head(p) if track == BARCELONA else base
        assert torch.allclose(b.J(p, ctx), want, atol=1e-6) and ctx.barcelona == (track == BARCELONA)
    rows = np.flatnonzero(d.track_key == 'monza')
    pr = T.predict(f, d, rows)
    rows_b = np.flatnonzero(d.track_key == BARCELONA)
    assert np.allclose(T.predict(f, d, rows_b)['setup'] - T.predict(f, d, rows_b, with_bcn=False)['setup'],
                       head(torch.as_tensor(d.p[rows_b], dtype=torch.float32)).numpy(), atol=1e-5)
    assert np.array_equal(pr['setup'], T.predict(f, d, rows, with_bcn=False)['setup'])


def test_bundle_round_trip_matches_predict(d, split, fits, tmp_path):
    tr, te, etr, ete = split
    f3, f1 = fits
    zr = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    b3 = CriticBundle.load(save_bundle(tmp_path / 'c3.pt', 'critic3', [f3], sources=d.games, track_z_raw=zr, z_provisional=zp))
    b1 = CriticBundle.load(save_bundle(tmp_path / 'c1.pt', 'critic1', [f1], sources=d.games, track_z_raw=zr, z_provisional=zp,
                                       style_train=T.style_inputs(d, etr, f1.style_tf)[0]))
    for b, f, rows in ((b3, f3, te[:40]), (b1, f1, ete)):
        want = T.predict(f, d, rows)['setup']
        got = [float(b.J(torch.as_tensor(d.p[r:r + 1], dtype=torch.float32),
                         b.context(track=d.track_key[r], style=d.style_raw[r] if f is f1 else None,
                                   source=d.games[d.game[r]]))[0]) for r in rows]
        np.testing.assert_allclose(got, want, atol=1e-5)
    assert b1.style_report(d.style_raw[ete[0]], d.track_key[ete[0]])['mahalanobis'] >= 0


def test_dummy_adapter_needs_nothing_from_f1_or_model_code(d, split, fits, tmp_path):
    """B17: a non-F1 adapter imports neither the F1 module nor torch / model / train / bundle code; its
    outputs (unseen source, unseen track with a centre line, style with NaN) give a valid query."""
    code = ('import sys; sys.path.insert(0, %r); import c2b_dummy_adapter; '
            'bad = [m for m in ("torch", "rlsetup.critic.model", "rlsetup.critic.train", "rlsetup.critic.bundle", '
            '"rlsetup.critic.sources.f1laps", "rlsetup.critic.sources.f1_game_map", "pandas") if m in sys.modules]; '
            'print(bad); assert not bad' % str(REPO_ROOT / 'tests'))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    tr, te, etr, ete = split
    f1 = fits[1]
    zr = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    b = CriticBundle.load(save_bundle(tmp_path / 'c1.pt', 'critic1', [f1], sources=d.games, track_z_raw=zr, z_provisional=zp,
                                      style_train=T.style_inputs(d, etr, f1.style_tf)[0]))
    p = torch.as_tensor(dummy.to_p(dummy.baseline_physical(4)), dtype=torch.float32)
    assert torch.allclose(p, torch.zeros_like(p), atol=1e-6)
    ctx = b.context(track=dummy.canonical_track('dummy_ring'), geometry=dummy.centre_line('dummy_ring'),
                    style=dummy.style_numbers(), source=dummy.SOURCE)
    assert ctx.flags['source'] == 'unseen_uniform' and torch.allclose(ctx.g, torch.full((2,), .5))
    assert ctx.flags['track_known'] is False and ctx.flags['style_centre'] == 'global' and not ctx.barcelona
    assert torch.isfinite(b.q(p, ctx)).all() and torch.isfinite(ctx.z).all()
    ctx2 = b.context(track=dummy.canonical_track('dummy_barcelona'), style=dummy.style_numbers(), source=None)
    assert ctx2.barcelona and ctx2.flags['source'] == 'uniform' and torch.isfinite(b.q_min(p, ctx2)).all()
    with pytest.raises(ValueError, match='unknown track'):
        b.context(track=None, source=dummy.SOURCE)


# ---- B21 hook (extra loss; the term itself is not implemented) ---------------------------------------
class _ZeroExtra(T.ExtraLoss):
    def spec(self):
        return {'name': 'zero'}

    def bind(self, d, fit_rows, scaler, support):
        self.rows = np.asarray(fit_rows)
        return lambda model, batch, gen: 0. * model.setup_sum(batch['p'], batch['z'], batch['g']).sum()


class _PullExtra(T.ExtraLoss):
    def spec(self):
        return {'name': 'pull'}

    def bind(self, d, fit_rows, scaler, support):
        return lambda model, batch, gen: 10. * (model.setup_sum(batch['p'], batch['z'], batch['g']) ** 2).mean()


def test_extra_loss_hook_in_fold_and_identical_without(d, split, fits):
    """A zero extra term reproduces critic 3 bit for bit (same folds, seeds, batch order, guard draws);
    the hook sees the in-fold fit rows only; a non-zero term changes the fit."""
    tr, te = split[:2]
    f3 = fits[0]
    z = _ZeroExtra()
    g = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u', extra=z)
    sa, sb = f3.model.state_dict(), g.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not np.isin(z.rows, te).any() and np.isin(z.rows, tr).all() and len(z.rows) < len(tr)
    assert g.info['extra_loss']['spec'] == {'name': 'zero'} and f3.info['extra_loss'] is None
    h = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u', extra=_PullExtra())
    assert not torch.equal(h.model.w2, f3.model.w2)


# ---- spec 12.21: TD3's Q is always critic 1; scalar gate hook on e ------------------------------------
def test_td3_q_is_always_critic1_and_e_gate_hook(d, split, fits, tmp_path):
    tr, te, etr, ete = split
    f3, f1 = fits
    zr = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    with pytest.raises(ValueError, match='always critic 1'):
        save_bundle(tmp_path / 'x.pt', 'critic3', [f3], sources=d.games, track_z_raw=zr, z_provisional=zp, role='td3_q')
    p3 = save_bundle(tmp_path / 'c3.pt', 'critic3', [f3], sources=d.games, track_z_raw=zr, z_provisional=zp)
    p1 = save_bundle(tmp_path / 'c1.pt', 'critic1', [f1], sources=d.games, track_z_raw=zr, z_provisional=zp,
                     style_train=T.style_inputs(d, etr, f1.style_tf)[0])
    assert CriticBundle.load(p3).role == 'style_free_reference'
    with pytest.raises(ValueError, match="not TD3's Q"):
        CriticBundle.load(p3, require_td3_q=True)
    b1 = CriticBundle.load(p1, require_td3_q=True)
    assert b1.role == 'td3_q' and float(b1.models[0].e_gate) == 1.
    src = (REPO_ROOT / 'scripts/critic_c2.py').read_text()
    assert "'td3_q': {'bundle': str(paths['critic1']" in src
    # e_gate = 0: critic 1 is exactly its critic 3 for any style; = 1 the B9 model (default)
    m = f1.model
    gen = torch.Generator().manual_seed(2)
    p, z, g = torch.rand(32, 15, generator=gen) * 2 - 1, torch.randn(32, 5, generator=gen), torch.eye(2)[torch.arange(32) % 2]
    s = torch.randn(32, 9, generator=gen)
    j1 = m(p, z, g, s=s)
    m.e_gate.fill_(0.)
    try:
        assert torch.allclose(m(p, z, g, s=s), f3.model(p, z, g), atol=1e-6, rtol=0)
    finally:
        m.e_gate.fill_(1.)
    assert torch.equal(m(p, z, g, s=s), j1)


# ---- B21 meta prior ----------------------------------------------------------------------------------
from rlsetup.critic.meta_prior import MetaPrior, fit_meta_prior     # noqa: E402


@pytest.fixture(scope='module')
def prior(d, split):
    return fit_meta_prior(d, split[0], TINY)


def test_b21_c0_is_critic3_bit_for_bit(d, split, fits, prior):
    tr = split[0]
    f3 = fits[0]
    g = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u', prior=prior.with_lambda(0., 1., 1.))
    assert all(torch.equal(a, b) for a, b in zip(f3.model.state_dict().values(), g.model.state_dict().values()))
    assert np.array_equal(T.predict(f3, d, split[1])['setup'], T.predict(g, d, split[1])['setup'])
    h = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u', prior=prior.with_lambda(.3, 1., 1.),
                   wd_grid=(f3.info['weight_decay'],))
    assert h.prior.lam == .3 and not torch.equal(h.model.w2, f3.model.w2)
    pr = T.predict(h, d, split[1])
    assert np.allclose(pr['prior'], h.prior.offset(d, split[1])) and (pr['prior'] >= 0).all() and pr['prior'].max() > 0


def test_b21_prior_has_no_leak_from_held_out_rows(d, split, prior):
    tr, te = split[:2]
    bad = lambda a: np.where(np.isin(np.arange(d.n), te)[(...,) + (None,) * (a.ndim - 1)], a * 1000., a)
    d2 = _c(d, gap=bad(d.gap), p=bad(d.p), lap_s=bad(d.lap_s))
    p2 = fit_meta_prior(d2, tr, TINY)
    np.testing.assert_array_equal(prior.m, p2.m)
    np.testing.assert_array_equal(prior.w, p2.w)
    np.testing.assert_array_equal(prior.r_bar, p2.r_bar)


def test_b21_cross_fitting_never_scores_against_own_half(d, split, prior):
    log, half = prior.info['cross_fit_log'], prior.info['half']
    tr = split[0]
    half_of_row = np.full(d.n, -1)
    half_of_row[tr] = half
    assert len(log['pairs']) > 0
    assert (log['pair_half'] != log['m_half']).all()
    assert (half_of_row[log['pairs'][:, 0]] == log['pair_half']).all() and (half_of_row[log['pairs'][:, 1]] == log['pair_half']).all()
    assert (d.cell_key[log['pairs'][:, 0]] == d.cell_key[log['pairs'][:, 1]]).all()       # pairs inside one cell
    assert ((prior.w >= 0) & (prior.w <= 1)).all() and prior.r_bar.shape == (15,)


def test_b21_gradient_unseen_track_and_style_independence(d, split, fits, prior, tmp_path):
    tr, te, etr, ete = split
    f3 = fits[0]
    pk = prior.with_lambda(.3, 1., 1.)
    g3 = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u', prior=pk, wd_grid=(f3.info['weight_decay'],))
    g1 = T.finetune(d, etr, g3, 'critic1', TINY, seed=5, salt='u')
    assert g1.prior is g3.prior                                        # inherited, frozen
    assert np.array_equal(T.predict(g1, d, te)['prior'], T.predict(g3, d, te)['prior'])
    style = d.style_raw.copy()
    style[ete] = style[ete] * 3.
    assert np.array_equal(T.predict(g1, _c(d, style_raw=style), ete)['prior'], T.predict(g1, d, ete)['prior'])
    assert list(inspect.signature(MetaPrior.pi_rows).parameters) == ['self', 'p', 'game', 'track']
    zr = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    b = CriticBundle.load(save_bundle(tmp_path / 'c3.pt', 'critic3', [g3], sources=d.games, track_z_raw=zr, z_provisional=zp))
    ctx = b.context(track='monza', source='F1 25')
    p = torch.zeros(4, 15, requires_grad=True)
    b.J(p, ctx).sum().backward()
    assert torch.isfinite(p.grad).all() and ctx.flags['meta_prior'] in ('track', 'pooled_w')
    pp = torch.rand(4, 15) * 2 - 1
    assert torch.allclose(b.J(pp, ctx) - b.J(pp, ctx, prior=False), b.prior(pp, ctx)) and float(b.prior(pp, ctx).abs().sum()) > 0
    rows = te[d.track_key[te] == 'monza']
    if len(rows):
        got = b.prior(torch.as_tensor(d.p[rows], dtype=torch.float32), b.context(track='monza', source=d.games[d.game[rows[0]]]))
        same = d.game[rows] == d.game[rows[0]]
        np.testing.assert_allclose(got.numpy()[same], g3.prior.offset(d, rows)[same], rtol=1e-5, atol=1e-6)
    import c2b_dummy_adapter as dm
    cu = b.context(track=None, geometry=dm.centre_line('dummy_ring'), source=dm.SOURCE)
    assert cu.flags['meta_prior'] == 'none' and float(b.prior(pp, cu).abs().sum()) == 0.


def test_b21_published_meta_never_in_a_bundle(d, split, fits, tmp_path):
    tr = split[0]
    pub = {(g, t): np.zeros(15) for g in d.games for t in d.track_names}
    pp = fit_meta_prior(d, tr, TINY, published=pub).with_lambda(.1, 1., 1.)
    assert pp.source == 'published'
    g3 = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u', prior=pp)
    zr = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    with pytest.raises(ValueError, match='never written'):
        save_bundle(tmp_path / 'x.pt', 'critic3', [g3], sources=d.games, track_z_raw=zr, z_provisional=zp)


# ---- B22 sessions ------------------------------------------------------------------------------------
def test_s22a_filters_training_rows_only(d, split):
    import importlib.util
    spec = importlib.util.spec_from_file_location('critic_c2', REPO_ROOT / 'scripts/critic_c2.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tr, te = split[:2]
    q = list(d.sessions).index('Quali')
    tr2 = mod.s22a_train(d, tr)
    assert not (d.session[tr2] == q).any() and np.array_equal(tr2, tr[d.session[tr] != q])
    src = inspect.getsource(mod.run_R3d)
    assert "s22a_train(d, tr, c.cfg.session.s22a_drop), te," in src            # evaluation rows unchanged
    from rlsetup.critic.nuisance import NuisanceEncoder
    enc = NuisanceEncoder(TINY.nuisance, session_dev=(10., ('Race', 'Quali'))).fit(d, tr)
    x = enc.encode(d, te)
    cols = [i for i, n in enumerate(x.dense_names) if n.startswith('sdev:')]
    assert len(cols) == 30 and (x.alpha_dense[cols] == 10.).all()
    tt = d.session[te] == list(d.sessions).index('TT')
    assert (x.dense[tt][:, cols] == 0).all()                                   # TT is the reference level
    g = T.pretrain(d, tr, 'critic3', physics_groups(), TINY, seed=3, salt='u', session_dev=True)
    pr = T.predict(g, d, te)
    assert (pr['session_dev'][tt] == 0).all() and np.abs(pr['session_dev'][~tt]).max() > 0


# ---- B23 ladder --------------------------------------------------------------------------------------
def test_b23_gamma_train_and_query_and_saved(d, split, fits, tmp_path):
    tr, te, etr, ete = split
    f3 = fits[0]
    lv = TINY.ladder.levels[1]                                                 # L2: gamma 0.1
    f1 = T.finetune(d, etr, f3, 'critic1', TINY, seed=5, salt='u', level=lv)
    assert float(f1.model.e_gate) == pytest.approx(.1) and f1.info['level']['name'] == 'L2' and f1.info['l2'] == 10.
    assert f1.info['m_style_coef'] is not None
    b = CriticBundle.load(save_bundle(tmp_path / 'c1.pt', 'critic1', [f1], sources=d.games,
                                      track_z_raw={t: d.z_raw[i] for i, t in enumerate(d.track_names)},
                                      z_provisional={t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)},
                                      style_train=T.style_inputs(d, etr, f1.style_tf)[0]), require_td3_q=True)
    assert float(b.models[0].e_gate) == pytest.approx(.1) and b.meta['e_gate'] == [pytest.approx(.1)]
    r = ete[0]
    ctx = b.context(track=d.track_key[r], style=d.style_raw[r], source=d.games[d.game[r]])
    np.testing.assert_allclose(float(b.J(torch.as_tensor(d.p[r:r + 1], dtype=torch.float32), ctx)[0]),
                               T.predict(f1, d, ete[:1])['setup'][0], atol=1e-5)
    m = f1.model                                                               # gamma enters the heads: e' = gamma e
    s = torch.as_tensor(T.style_inputs(d, ete, f1.style_tf)[0], dtype=torch.float32)
    z, gg = (torch.as_tensor(a, dtype=torch.float32) for a in T.context(d, ete, f1.scaler))
    P = torch.as_tensor(d.p[ete], dtype=torch.float32)
    e = m.encode(s, clip=True)
    j_gamma = m(P, z, gg, e=e)
    m.e_gate.fill_(1.)
    try:
        assert torch.allclose(j_gamma, m(P, z, gg, e=.1 * e), atol=1e-6)
    finally:
        m.e_gate.fill_(.1)


def _ladder_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location('critic_c2', REPO_ROOT / 'scripts/critic_c2.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _m5(dr2_lo, e0):
    blk = {'pooled': {'delta_r2_c1_minus_c3': {'point': 0., 'ci': [dr2_lo, .1]}}, 'e0_pair_acc_A_minus_critic3': {'point': e0}}
    return {'critic1_F_cell': blk, 'critic1_F_person': blk, 'fallback_inputs': {
        'F_cell': {'i_delta_r2_point': dr2_lo, 'iii_e0_pair_diff': e0}, 'F_person': {'i_delta_r2_point': dr2_lo, 'iii_e0_pair_diff': e0}}}


def test_b23_ladder_order_is_deterministic_and_q_is_always_critic1():
    mod = _ladder_mod()
    ok, bad = _m5(-.01, 0.), _m5(-.05, 0.)
    r7 = {'fallback_inputs': {'F_cell': {'ii_delta_r2_point': 0.}, 'F_person': {'ii_delta_r2_point': 0.}}}
    r5b = lambda sel, l3: {'needed': True, 'bars_selected': {'evaluated': True, 'passes': sel},
                           'bars_L3': {'evaluated': True, 'passes': l3}}
    L = mod.remedy_ladder
    assert L({'R5': ok, 'R7': r7})['rung'] == 'rung0_B9_exact'
    assert L({'R5': bad, 'R7': r7})['rung'] == 'pending'                       # trigger (i) fired, rows S not run
    assert L({'R5': bad, 'R7': r7, 'R5_S': ok})['rung'] == 'rung1_rows_S'
    assert L({'R5': bad, 'R7': r7, 'R5_S': bad, 'R5b': r5b(True, True)})['rung'] == 'rung2_selected_level'
    assert L({'R5': bad, 'R7': r7, 'R5_S': bad, 'R5b': r5b(False, True)})['rung'] == 'rung2_L3'
    out = L({'R5': bad, 'R7': r7, 'R5_S': bad, 'R5b': r5b(False, False)})
    assert out['rung'] == 'implementation_fault' and out['td3_q'] == 'critic1'
    no_trig = _m5(-.03, 0.)              # CI low -0.03 misses the bar, but the B16 trigger (point estimates) does not fire
    no_trig['fallback_inputs'] = {fs: {'i_delta_r2_point': 0., 'iii_e0_pair_diff': 0.} for fs in ('F_cell', 'F_person')}
    assert L({'R5': no_trig, 'R7': r7, 'R5b': r5b(True, True)})['rung'] == 'rung2_selected_level'   # rung 1 skipped
    for m in ({'R5': ok, 'R7': r7}, {'R5': bad, 'R7': r7, 'R5_S': bad, 'R5b': r5b(False, True)}):
        assert L(m) == L(m) and L(m)['td3_q'] == 'critic1'
    assert list(TINY.ladder.rung_order) == ['rung0_B9_exact', 'rung1_rows_S', 'rung2_selected_level', 'rung2_L3']



# ---- fix round: QA T1 (M2) ---------------------------------------------------------------------------
def test_e0_bar_compares_like_with_like(d, split, fits):
    """Critic 1 at e = 0 and the critic-3 comparator the script stores ('setupA_c3' = critic 3 with the same
    delivered Barcelona head) are identical on every row A, Barcelona included (inherited weights frozen)."""
    tr, te, etr, ete = split
    f3 = fits[0]
    head, _ = T.fit_bcn_head(d, tr, f3, TINY.bcn, seed=1)
    with torch.no_grad():
        head.l2.weight.add_(.2)
    g3 = replace(f3, bcn=head)
    f1 = T.finetune(d, etr, g3, 'critic1', TINY, seed=5, salt='u')
    e0 = T.predict(f1, d, te, e_zero=True, nuisance=False)['setup']
    c3 = T.predict(g3, d, te, nuisance=False)['setup']
    assert (d.track_key[te] == BARCELONA).any()
    np.testing.assert_array_equal(e0, c3)
    src = (REPO_ROOT / 'scripts/critic_c2.py').read_text()
    assert "'setupA_c3': p3['setup']" in src and "'setupA_c3', rows_key='rowsA'" in src
