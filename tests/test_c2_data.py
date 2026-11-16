"""C2a data loading (spec S2, review R13): several exports (folder, zip read in place, bare CSV),
dedup, usernames -> integer ids, per-game gap, rows A / S, privacy of the outputs."""
from dataclasses import replace
import zipfile
import numpy as np
import pytest
from c2_fixture import TRACK_MAP, USERS_26, build_export, truth_track, write_fixture
from rlsetup.critic.config import Config
from rlsetup.critic.dataset import Frame, ranking_stats
from rlsetup.critic.sources.base import SourceAdapter
from rlsetup.critic.sources.f1_game_map import GAME_COLUMNS, game_to_knobs
from rlsetup.critic.sources.f1laps import (META_COLUMNS, SESSIONS, F1LapsAdapter, build_frame, discover, expand_sources,
                                           lap_time_s, load_exports, s_rows_before_filters, session_of)


@pytest.fixture(scope='module')
def fx(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp('c2data'))


@pytest.fixture(scope='module')
def data(fx):
    return load_exports(fx['sources'], log=lambda *a: None)


@pytest.fixture(scope='module')
def frame(data):
    return build_frame(data, Config(), log=lambda *a: None)


def test_discover_folder_zip_and_bare_csv(fx):
    root = fx['root']
    assert [e.label for e in discover(root / 'exp_f1_25')] == ['exp_f1_25/data']
    assert [e.label for e in discover(root / 'bundle.zip')] == ['bundle.zip:F1_25/data', 'bundle.zip:F1_26/data']
    assert discover(root / 'extra_setups.csv')[0].names == {'setups.csv'}
    assert expand_sources([root / '*.zip']) == [root / 'bundle.zip']
    with pytest.raises(FileNotFoundError):
        expand_sources([root / 'nothing*'])


def test_zip_is_read_in_place(fx, data):
    with zipfile.ZipFile(fx['root'] / 'bundle.zip') as z:
        names = z.namelist()
    assert not any((fx['root'] / n).exists() for n in names if n.startswith('F1_26'))
    assert data.provenance.files['bundle.zip:F1_26/data/setups.csv']


def test_dedup_by_setup_id_and_bytes(fx, data):
    assert len(data.setups) == fx['n_setups_25'] + fx['n_setups_26']           # 36 + 1 new + 36
    assert any('bundle.zip:F1_25/data/setups.csv ==' in s for s in data.provenance.skipped_duplicates)
    counts = dict(data.counts)
    assert counts['setup rows read (all exports)'] == 36 + 36 + 3             # identical bytes not re-read
    assert counts['unique setup_id'] == 73
    assert len(data.laps) == 14 and len({(l.game, l.driver, l.track) for l in data.laps.values()}) == 14


def test_usernames_become_integer_ids_per_game(data, frame):
    assert all(isinstance(s['driver'], int) for s in data.setups)
    assert not any('user' in s or '_user' in s for s in data.setups)
    # the same name in two games is two drivers (C2: ids keyed by (game, user))
    assert data.n_drivers == 6
    assert frame.driver.dtype.kind == 'i'


def test_gap_is_per_game_and_filtered(frame):
    """B1: gap = 100 (lap / median lap of the kept rows of (game, track, session) - 1) after the two-pass
    robust filter. The F1 26 laps are 5 % slower: a pooled median would shift them. The +10 % setups
    (6 rows, all Time Trial) are dropped as |z| > 3.5 in pass 1; pass 2 drops nothing more."""
    for g in range(len(frame.games)):
        for t in range(len(frame.track_names)):
            for se in range(len(frame.sessions)):
                m = (frame.game == g) & (frame.track == t) & (frame.session == se)
                if not m.any():
                    continue
                np.testing.assert_allclose(frame.gap[m], 100 * (frame.lap_s[m] / np.median(frame.lap_s[m]) - 1), atol=1e-12)
                assert abs(np.median(frame.gap[m])) < 1e-9
    assert frame.gap.max() <= 8.                    # the +10 % setups are gone
    counts = dict(frame.counts)
    assert counts['5. |robust z| <= 3.5 per (game, track, session), 2 passes (B1; rows A)'] == 67
    lab = frame.box_report['label']
    assert [q['dropped_high'] for q in lab['passes']] == [6, 0] and lab['dropped_by_session']['TT']['high'] == 6


def test_rows_s_and_ranking(data, frame):
    assert s_rows_before_filters(data) == {'F1 25': 29, 'F1 26': 28}   # before the gap filter
    assert frame.in_s.sum() == 51
    assert (frame.exact_lap >= 0).sum() == 14
    r = ranking_stats(frame)
    assert r['pairs'] > 0 and r['groups'] > 0
    assert set(np.unique(frame.session)) <= set(range(len(SESSIONS))) and frame.sessions == SESSIONS


def test_session_and_time_parsing():
    assert session_of({'session_group': 'Qualifying', 'session': 'Qualifying 3'}) == 'Quali'
    assert session_of({'session': 'Time Trial'}) == 'TT'
    assert lap_time_s('1:17.323') == pytest.approx(77.323) and lap_time_s('') is None


def test_garbled_assist_is_unknown(frame):
    abs_ = frame.assists['anti_lock_brakes']
    assert (abs_ == 0).any() and (abs_ == 1).any()  # '151' -> unknown (0); F1 25 rows unknown too


def test_no_username_in_any_output(fx, tmp_path, monkeypatch):
    """scripts/critic_c2_data.py on the fixture: no fixture username in stdout or written files."""
    import importlib.util
    import sys
    from rlsetup.critic.sources import f1_game_map
    from rlsetup.critic import config as cconfig
    spec = importlib.util.spec_from_file_location('critic_c2_data', cconfig.REPO_ROOT / 'scripts/critic_c2_data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    geo = replace(cconfig.GeometryConfig(), extra_dir=str(fx['extra_dir']), tum_dir=str(fx['tum_dir']))
    sty = replace(cconfig.StyleInputConfig(), cache_dir=str(tmp_path / 'cache'))
    monkeypatch.setattr(mod, 'Config', lambda: replace(Config(), geometry=geo, style_input=sty))
    monkeypatch.setattr(mod, 'TRACK_GEOMETRY', TRACK_MAP)
    monkeypatch.setattr(mod, 'run_provenance', lambda: {'rlsetup': 'stub'})
    lines = []
    monkeypatch.setattr('builtins.print', lambda *a, **k: lines.append(' '.join(map(str, a))))
    out = mod.main(['--sources', *map(str, fx['sources']), '--out', str(tmp_path / 'runs')])
    text = '\n'.join(lines) + (out / 'summary.json').read_text()
    cache = sorted((tmp_path / 'cache').rglob('*.json'))          # T8: the style cache too
    assert cache
    text += ''.join(f.read_text() for f in cache)
    npz = np.load(out / 'dataset.npz')
    text += ' '.join(' '.join(map(str, npz[k].ravel())) for k in npz.files if npz[k].dtype.kind in 'US')
    assert 'rows A' in text and 'R2 geometry gate' in text
    for u in fx['users']:
        assert u not in text
    assert f1_game_map.TRACK_GEOMETRY['mexico'] == 'MexicoCity'       # module constant untouched


def test_lap_samples_from_a_later_export(tmp_path):
    """T1 (QA M1): export A lists lap L without graph / time files, export B (read after A) lists L
    with them: the loaded lap has samples and a time map."""
    full = build_export('F1 26', USERS_26, np.random.default_rng(5), {'trk_good': truth_track(1.)})
    for name, files in (('a_first', ('setups.csv', 'laps.csv', 'setup_lap_map.csv')), ('b_second', tuple(full))):
        d = tmp_path / name
        d.mkdir()
        for f in files:
            (d / f).write_bytes(full[f])
    data = load_exports([tmp_path / 'a_first', tmp_path / 'b_second'], log=lambda *a: None)
    assert len(data.laps) == 3
    assert all(l.has_samples and l.time_map is not None for l in data.laps.values())
    assert dict(data.counts)['laps given telemetry by a later export'] == 3
    only_a = load_exports([tmp_path / 'a_first'], log=lambda *a: None)
    assert not any(l.has_samples for l in only_a.laps.values())


def _frame(session, setup_hash, gap):
    n = len(gap)
    z = np.zeros(n, int)
    return Frame(knobs=np.zeros((n, 15)), p=np.zeros((n, 15)), gap=np.asarray(gap, float), lap_s=np.ones(n), game=z,
                 track=z, driver=z, team=z, session=np.asarray(session), controller=z, assists={}, date_days=z,
                 setup_hash=np.asarray(setup_hash, object), cell=z, exact_lap=z - 1, source_row=np.arange(n),
                 games=('G',), track_names=('t',), team_names=('x',), sessions=SESSIONS, controllers=('c',),
                 assist_levels={}, cells=((0, 0, 0),), cell_laps=((0,),))


def test_ranking_pairs_within_session_only():
    """T4 (QA M8, R6): one cell, 2 Race rows + 2 TT rows -> 2 pairs (not 6); equal-hash or equal-gap
    pairs are excluded."""
    f = _frame([0, 0, 2, 2], ['a', 'b', 'c', 'd'], [0., 1., 2., 3.])
    r = ranking_stats(f)
    assert r == {'groups': 2, 'rows': 4, 'pairs': 2, 'drivers': 1}
    assert ranking_stats(_frame([0, 0, 2, 2], ['a', 'a', 'c', 'd'], [0., 1., 2., 3.]))['pairs'] == 1   # same hash
    assert ranking_stats(_frame([0, 0, 2, 2], ['a', 'b', 'c', 'd'], [0., 0., 2., 3.]))['pairs'] == 1   # same gap
    assert ranking_stats(f, np.array([True, True, True, False]))['pairs'] == 1


def test_adapter_contract(data, frame):
    """T7 (QA M7): the adapter satisfies the Protocol; `meta` gives the same nuisance codes as
    build_frame for the same rows; `to_knobs` equals `game_to_knobs`."""
    ad = F1LapsAdapter()
    assert isinstance(ad, SourceAdapter)
    rows = [data.setups[i] for i in frame.source_row[[0, 20, 40]]]
    meta = ad.meta({c: np.array([r['meta_raw'][c] for r in rows], object) for c in META_COLUMNS})
    assert list(meta['source']) == [frame.games[g] for g in frame.game[[0, 20, 40]]]
    assert list(meta['session']) == [frame.sessions[i] for i in frame.session[[0, 20, 40]]]
    assert list(meta['controller']) == [frame.controllers[i] for i in frame.controller[[0, 20, 40]]]
    for a, levels in frame.assist_levels.items():
        assert list(meta['assists'][a]) == [levels[i] for i in frame.assists[a][[0, 20, 40]]]
    table = {c: np.array([r['values'][c] for r in rows], float) for c in GAME_COLUMNS}
    np.testing.assert_array_equal(ad.to_knobs(table), game_to_knobs(table))
    np.testing.assert_allclose(frame.knobs[[0, 20, 40]], game_to_knobs(table))


def test_explicit_pairs_file_and_game_filter(fx, tmp_path):
    """16 Nov files: a bare CSV with `matched_lap_id` is a pairs map (precedence over setup_lap_map.csv,
    telemetry from exports already read); setups of games without an S4 map are skipped when `games` is set."""
    import csv as _csv
    import io as _io
    from c2_fixture import USERS_25, _csv as to_csv
    rows = list(_csv.DictReader(_io.StringIO((fx['root'] / 'extra_setups.csv').read_text(encoding='utf-8-sig'))))
    u, t = USERS_25[0], 'trk_good'
    old_game = dict(rows[0], game='F1_23', setup_id='23-old-game')
    (tmp_path / 'more_setups.csv').write_bytes(to_csv([old_game], list(rows[0])))
    pairs = [{'setup_id': f'25-{t}-{u}-2', 'matched_lap_id': f'lap-25-{t}-{u}', 'has_telemetry': 'true'},
             {'setup_id': '23-old-game', 'matched_lap_id': f'lap-25-{t}-{u}', 'has_telemetry': 'true'},
             {'setup_id': f'25-{t}-{u}-0', 'matched_lap_id': 'lap-not-downloaded', 'has_telemetry': 'true'}]
    (tmp_path / 'pairs.csv').write_bytes(to_csv(pairs, ['setup_id', 'matched_lap_id', 'has_telemetry']))
    assert discover(tmp_path / 'pairs.csv')[0].names == {'setup_lap_pairs.csv'}
    srcs = fx['sources'] + [tmp_path / 'more_setups.csv', tmp_path / 'pairs.csv']
    base = load_exports(fx['sources'], log=lambda *a: None)
    d = load_exports(srcs, log=lambda *a: None, games=('F1_25', 'F1_26'))
    c = dict(d.counts)
    assert c['setup rows skipped: game F1 23 (no S4 map)'] == 1 and len(d.setups) == len(base.setups)
    assert c['explicit setup-lap pairs read (pairs files)'] == 3
    assert c['explicit pairs with the setup kept and the lap telemetry available'] == 1
    assert len(d.exact) == len(base.exact) + 1            # the k=2 setup gains the lap; k=1 keeps it too
    assert set(base.exact.items()) <= set(d.exact.items())
    assert len(load_exports(srcs, log=lambda *a: None).setups) == len(base.setups) + 1   # no filter: kept
    from rlsetup.critic.sources.f1laps import game_label
    assert game_label('F1_25') == game_label('F1 25') == 'F1 25'
