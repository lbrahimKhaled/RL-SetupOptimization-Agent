"""C2b run script (scripts/critic_c2.py) end to end on the synthetic data build (tests/c2b_fixture.py):
every run R0-R9 with a tiny training budget injected by the test (the script's defaults are the
addendum's), bars.json written read-only before any fit, two seeded runs give identical metrics,
resume skips finished fits, the bundles load, and no fixture username reaches any output (B19)."""
from dataclasses import replace
import importlib.util
import copy
import json
import os
from pathlib import Path
import stat
import numpy as np
import pytest
from c2b_fixture import write_c2b_dataset
from rlsetup.critic.c2_config import BARS, C2bConfig
from rlsetup.critic.config import REPO_ROOT

C = C2bConfig()
ALL_RUNS = 'R1,R2,R3,R3b,R3c,R3d,R4,R5,R5b,R6,R7,R8,R9'
TINY = replace(C, pretrain=replace(C.pretrain, max_epochs=3, patience=2, weight_decay_grid=(0., 1e-4), smoke_epochs=2,
                                   smoke_patience=1),
               finetune=replace(C.finetune, max_steps=8, patience=3, l2_grid=(1e-3, 1e-1), smoke_steps=4),
               bcn=replace(C.bcn, max_epochs=3, patience=2), eval=replace(C.eval, n_boot=40, shuffle_seeds=(0, 1), r8_setups=50))


def script():
    spec = importlib.util.spec_from_file_location('critic_c2', REPO_ROOT / 'scripts/critic_c2.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    return write_c2b_dataset(tmp_path_factory.mktemp('c2b_synth') / 'build')


def run(monkeypatch, argv):
    if '--data' in argv and '--meta-sources' not in argv:
        argv = argv + ['--meta-sources', str(Path(argv[argv.index('--data') + 1]) / 'meta_export')]
    mod = script()
    monkeypatch.setattr(mod, 'C2bConfig', lambda: TINY)
    monkeypatch.setattr('builtins.print', lambda *a, **k: None)
    return mod, mod.main(argv)


@pytest.fixture(scope='module')
def full_run(synth, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        out = tmp_path_factory.mktemp('c2b_runs')
        _, a = run(mp, ['--runs', ALL_RUNS, '--data', str(synth), '--out', str(out / 'a'),
                        '--repeats', '2'])
        _, b = run(mp, ['--runs', ALL_RUNS, '--data', str(synth), '--out', str(out / 'b'),
                        '--repeats', '2'])
    finally:
        mp.undo()
    return a, b


def _strip(m):
    return {k: v for k, v in m.items() if k != 'R0'}


def test_all_runs_complete_and_bundles_load(full_run):
    a, _ = full_run
    m = json.loads((a / 'metrics.json').read_text())
    for r in ALL_RUNS.split(','):
        assert r in m, r
    assert m['R2']['choice']['groups'] in ('pca', 'physics')
    assert m['fallback_trigger']['evaluated'] is True
    pa = m['R3']['critic3_F_cell']['pooled']['pair_acc']
    assert 0 <= pa['point'] <= 1 and pa['ci'][0] <= pa['point'] <= pa['ci'][1]
    assert m['R3']['critic3_F_cell']['barcelona']['n_rows'] > 0
    assert list(m['R3']['critic3_F_cell'])[:2] == ['barcelona', 'pooled']          # B18: Barcelona row first
    from rlsetup.critic.bundle import CriticBundle
    for kind in ('critic1', 'critic2', 'critic3'):
        b = CriticBundle.load(a / 'bundles' / f'{kind}_final.pt')
        assert b.kind == kind
        assert (a / 'bundles' / f'{kind}_final.json').exists()
        prov = b.provenance
        assert prov['data']['data_hash'] and prov['folds_sha256'] and prov['style_rl_commit']
        assert 'run_provenance' in prov


def test_two_seeded_runs_give_identical_metrics(full_run):
    a, b = full_run
    ma, mb = (json.loads((x / 'metrics.json').read_text()) for x in (a, b))
    assert _strip(ma) == _strip(mb)
    assert (a / 'folds.json').read_text() == (b / 'folds.json').read_text()


def test_bars_written_first_and_read_only(full_run):
    a, _ = full_run
    bars = a / 'bars.json'
    assert json.loads(bars.read_text())['bars'] == json.loads(json.dumps(BARS))
    assert not (os.stat(bars).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    first_fit = min(p.stat().st_mtime_ns for p in (a / 'fits').rglob('*.json'))
    assert bars.stat().st_mtime_ns <= first_fit
    with pytest.raises(PermissionError):
        bars.write_text('{}')


def test_resume_skips_finished_fits(full_run, synth, monkeypatch):
    a, _ = full_run
    before = {p: p.stat().st_mtime_ns for p in (a / 'fits').rglob('*.npz')}
    mod, _ = run(monkeypatch, ['--runs', 'R1,R3', '--data', str(synth), '--resume', str(a), '--repeats', '2'])
    after = {p: p.stat().st_mtime_ns for p in (a / 'fits').rglob('*.npz')}
    assert after == before


def test_resume_rejects_other_data(full_run, tmp_path, monkeypatch):
    a, _ = full_run
    other = write_c2b_dataset(tmp_path / 'other', seed=3)
    with pytest.raises(ValueError, match='data hash'):
        run(monkeypatch, ['--runs', 'R1', '--data', str(other), '--resume', str(a)])


def test_smoke_run_r0(synth, tmp_path, monkeypatch):
    _, out = run(monkeypatch, ['--runs', 'R0', '--smoke', '--data', str(synth), '--out', str(tmp_path)])
    m = json.loads((out / 'metrics' / 'R0.json').read_text())
    assert m['smoke'] is True and m['projection']['upper']['total_h'] > 0
    for v in m['bundle_roundtrip'].values():
        assert v['abs_diff'] < 1e-6


def test_outputs_hold_no_person_or_driver_ids(full_run):
    """B6: folds.json maps row positions to folds; no output stores the person / driver id arrays."""
    a, _ = full_run
    doc = json.loads((a / 'folds.json').read_text())
    assert set(doc) == {'data_hash', 'rows_sha256', 'build_sha256', 'n_folds', 'n_rows', 'keys', 'sets'}
    for p in (a / 'fits').rglob('*.npz'):
        z = np.load(p)
        assert not {'person', 'driver'} & set(z.files)


def test_no_username_from_real_export_format_through_critic_runs(tmp_path, monkeypatch):
    """T8 extended (B19): scripts/critic_c2_data.py on the C2a export fixture (real F1Laps file format with
    usernames), then critic_c2 R1 + R2 (ridge baselines, critic-3 fits on A) on that build: no fixture
    username in stdout or any written file. R0 needs critic-1 fits, which the 12 exact rows of 5 persons
    of that fixture cannot support (finetune raises 'no fit rows'); R0-R9 outputs are scanned on the
    synthetic C2b build above (test_outputs_hold_no_ids)."""
    from c2_fixture import TRACK_MAP, write_fixture
    from rlsetup.critic import config as cconfig
    fx = write_fixture(tmp_path / 'fx')
    spec = importlib.util.spec_from_file_location('critic_c2_data', REPO_ROOT / 'scripts/critic_c2_data.py')
    dmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dmod)
    geo = replace(cconfig.GeometryConfig(), extra_dir=str(fx['extra_dir']), tum_dir=str(fx['tum_dir']))
    sty = replace(cconfig.StyleInputConfig(), cache_dir=str(tmp_path / 'cache'))
    monkeypatch.setattr(dmod, 'Config', lambda: replace(cconfig.Config(), geometry=geo, style_input=sty))
    monkeypatch.setattr(dmod, 'TRACK_GEOMETRY', TRACK_MAP)
    monkeypatch.setattr(dmod, 'run_provenance', lambda: {'rlsetup': 'stub'})
    lines = []
    monkeypatch.setattr('builtins.print', lambda *a, **k: lines.append(' '.join(map(str, a))))
    data_dir = dmod.main(['--sources', *map(str, fx['sources']), '--out', str(tmp_path / 'data')])
    # the fixture's gate-failed tracks have no z; in C2b every track has one (B2 / B3): give them a z
    arrays = dict(np.load(data_dir / 'dataset.npz'))
    z = arrays['track_features']
    z[~np.isfinite(z).all(axis=1)] = z[np.isfinite(z).all(axis=1)][0] * np.array([1.1, .9, 1., 1., 1.])
    np.savez_compressed(data_dir / 'dataset.npz', **arrays)
    mod = script()
    monkeypatch.setattr(mod, 'C2bConfig', lambda: TINY)
    out = tmp_path / 'runs'
    with pytest.raises(ValueError, match='no fit rows'):
        mod.main(['--runs', 'R0', '--smoke', '--data', str(data_dir), '--out', str(tmp_path / 'r0')])
    mod.main(['--runs', 'R1,R2', '--data', str(data_dir), '--out', str(out), '--repeats', '1'])
    text = '\n'.join(lines)
    files = [f for f in out.rglob('*') if f.is_file()]
    assert any(f.name == 'R1.json' for f in files) and any(f.name == 'R2.json' for f in files)
    for f in files:
        if f.suffix in ('.json', '.txt', '.log'):
            text += f.read_text()
        elif f.suffix == '.npz':
            z = np.load(f)
            text += ' '.join(' '.join(map(str, z[k].ravel())) for k in z.files if z[k].dtype.kind in 'US')
        else:
            text += f.read_bytes().decode('latin-1')
    for u in fx['users']:
        assert u not in text


def test_forced_b21_delivery_and_b23_rung2(synth, tmp_path, monkeypatch):
    """B21 adoption forced (adoption bars relaxed in the test only) -> R3c runs and the delivered critic 3 carries
    the in-fold prior; critic-1 bars made impossible -> rung 0 fails, R5b fits every level; with the ladder
    answering rung 2 (selected level), R9 delivers a critic-1 TD3 Q with that level's gamma saved in the bundle,
    and critic 3 is never TD3's Q. With the real ladder (L3 fails too) R9 refuses to deliver."""
    mod = script()
    bars = copy.deepcopy(mod.BARS)
    bars['b21_adoption'].update(F_person_delta_pair_ci_low_gt=-1., F_cell_delta_pair_point_ge=-1., barcelona_max_drop_point=1.,
                                F_hash_r2_gain_max_drop=1.)
    bars['critic1_exact']['delta_r2_ci_low_gt'] = 10.
    monkeypatch.setattr(mod, 'BARS', bars)
    monkeypatch.setattr(mod, 'C2bConfig', lambda: TINY)
    monkeypatch.setattr('builtins.print', lambda *a, **k: None)
    out = tmp_path / 'forced'
    argv = ['--data', str(synth), '--out', str(out), '--repeats', '1', '--meta-sources', str(synth / 'meta_export')]
    run_dir = mod.main(['--runs', 'R1,R2,R3,R3b,R3c,R3d,R4,R5,R5b', *argv])
    m = json.loads((run_dir / 'metrics.json').read_text())
    assert m['R3b']['adoption']['adopted'] and m['R3c']['needed'] and m['R3c']['delivered_critic3'] == 'B21'
    assert any(f.name.startswith('c3B21_') for f in (run_dir / 'fits' / 'R3c').glob('*.json'))
    assert m['R5b']['needed'] and sum(m['R5b']['selected_counts'].values()) == 2 * 2 * 5
    assert m['remedy_ladder']['rung'] == 'implementation_fault' and m['remedy_ladder']['td3_q'] == 'critic1'
    resume = ['--data', str(synth), '--resume', str(run_dir), '--repeats', '1', '--meta-sources', str(synth / 'meta_export')]
    with pytest.raises(RuntimeError, match='implementation_fault'):
        mod.main(['--runs', 'R9', *resume])
    real = mod.remedy_ladder
    monkeypatch.setattr(mod, 'remedy_ladder', lambda mm: real(mm) | {'rung': 'rung2_selected_level'})
    mod.main(['--runs', 'R6,R7,R8,R9', *resume])
    m = json.loads((run_dir / 'metrics.json').read_text())
    q = m['R9']['td3_q']
    assert q['kind'] == 'critic1' and q['rung'] == 'rung2_selected_level' and q['level'] in ('B9', 'L1', 'L2', 'L3')
    gamma = {'B9': 1., 'L1': .3, 'L2': .1, 'L3': .03}[q['level']]
    from rlsetup.critic.bundle import CriticBundle
    b1 = CriticBundle.load(run_dir / 'bundles' / 'critic1_final.pt', require_td3_q=True)
    assert all(abs(float(mdl.e_gate) - gamma) < 1e-7 for mdl in b1.models)
    b3 = CriticBundle.load(run_dir / 'bundles' / 'critic3_final.pt')
    assert b3.role == 'style_free_reference' and b3.meta_prior is not None and b3.meta_prior.source == 'in_fold'
    assert b3.meta_prior.lam > 0 and m['R9']['critic3_delivered']['variant'] == 'B21'
    assert b1.meta_prior is not None and b1.meta_prior.lam == b3.meta_prior.lam      # critic 1 inherits Pi frozen
    with pytest.raises(ValueError):
        CriticBundle.load(run_dir / 'bundles' / 'critic3_final.pt', require_td3_q=True)


def test_resume_refuses_when_build_content_changes(synth, tmp_path, monkeypatch):
    """QA T2 / M1: a build whose style and z change (data hash and row digest unchanged) is refused on resume;
    the build hash, versions and threads are in every fit key."""
    import shutil
    _, out = run(monkeypatch, ['--runs', 'R1', '--data', str(synth), '--out', str(tmp_path / 'r'), '--repeats', '1'])
    fit = json.loads(next((out / 'fits' / 'R1').glob('nuisE_*.json')).read_text())
    assert fit['hash']
    other = tmp_path / 'build2'
    shutil.copytree(synth, other)
    arr = dict(np.load(other / 'dataset.npz'))
    arr['cell_style'] = arr['cell_style'] * 1.7
    arr['track_features'] = arr['track_features'][::-1].copy()
    np.savez_compressed(other / 'dataset.npz', **arr)
    with pytest.raises(ValueError, match='build hash'):
        run(monkeypatch, ['--runs', 'R1', '--data', str(other), '--resume', str(out), '--repeats', '1'])
    mod = script()
    assert 'rlsetup/style/__init__.py' in mod.CODE_FILES or any(f.startswith('rlsetup/style/') for f in mod.CODE_FILES)
    assert any(f.startswith('rlsetup/critic/sources/') for f in mod.CODE_FILES)
    assert 'scripts/critic_c2_report.py' not in mod.CODE_FILES


def test_rung2_matched_comparator_and_selection_set(synth, tmp_path, monkeypatch):
    """QA T4 / architect must-change 1-2: every rung-2 level has a comparator fitted at that level's alpha of
    m(s~), metrics_R5b judges against it, and B9 is never a rung-2 candidate."""
    mod = script()
    bars = copy.deepcopy(mod.BARS)
    bars['critic1_exact']['delta_r2_ci_low_gt'] = 10.                           # rung 0 fails -> R5b runs
    monkeypatch.setattr(mod, 'BARS', bars)
    monkeypatch.setattr(mod, 'C2bConfig', lambda: TINY)
    monkeypatch.setattr('builtins.print', lambda *a, **k: None)
    out = mod.main(['--runs', 'R1,R2,R3,R3b,R3c,R4,R5,R5b', '--data', str(synth), '--out', str(tmp_path / 'f'), '--repeats', '1',
                    '--meta-sources', str(synth / 'meta_export')])
    m = json.loads((out / 'metrics' / 'R5b.json').read_text())
    assert m['selection_set'] == ['L1', 'L2', 'L3'] and 'B9' not in m['selected_counts']
    alphas = {lv[0]: lv[3] for lv in TINY.ladder.levels}
    for name, a in alphas.items():
        fits = sorted((out / 'fits' / 'R5b').glob(f'c3E{name}_*.json'))
        assert len(fits) == 2 * 5 * 2
        for f in fits:
            doc = json.loads(f.read_text())
            assert doc['info']['alpha_m_s'] == a and doc['params']['alpha_m_s'] == a
    assert set(m['variants']) == {'selected', 'L1', 'L2', 'L3'} and 'report_only_vs_alpha10_comparator' in m
    sel = mod.r5b_selection(_ctx_like(mod, out))
    assert set(sel.values()) <= {'L1', 'L2', 'L3'}
    assert mod.BARS['b23']['selection_set'] == ['L1', 'L2', 'L3'] and 'comparator' in mod.BARS['b23']


def _ctx_like(mod, out):
    """A minimal stand-in for Ctx: what r5b_selection reads (store paths, folds, repeats)."""
    class C:
        pass
    c = C()
    c.store = mod.Store(out, {})
    c.smoke, c.repeats = False, 1
    folds, _ = mod.load_folds(out / 'folds.json', json.loads((out / 'folds.json').read_text())['data_hash'])
    c.folds = folds
    c.n_folds = lambda: 5
    c.fold_sets = lambda sets=('F_cell', 'F_person'), repeats=None: mod.Ctx.fold_sets(c, sets, repeats)
    return c
