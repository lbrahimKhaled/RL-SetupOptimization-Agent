"""C2a style numbers (spec S3, review R2-R4): the numpy port equals style-rl at the pinned commit,
per-lap style through the gate (placeholder flag, ungated tracks keep their rows with NaN geometry
numbers), cells, the R4 transform fitted on training cells only, track z, determinism and cache."""
from dataclasses import replace
from pathlib import Path
import hashlib
import importlib.util
import os
import shutil
import subprocess
import numpy as np
import pytest
from c2_fixture import TRACK_MAP, write_fixture
from rlsetup.critic.config import (Config, GATE_LENGTH, LAP_SPACING, LAP_TIME_CLOSURE, NO_GEOMETRY, OK,
                                   StyleTransformConfig, TRACK_FEATURES)
from rlsetup.critic.geometry import CENTRE_LINE, edges, read_tum_csv
from rlsetup.critic.geometry import corr_over_shifts
from rlsetup.critic.sources.f1laps import F1LapsAdapter, build_frame, lap_style_input, load_exports
from rlsetup.critic.style_data import (GEOMETRY_KEYS, StyleTransform, cache_env, cell_styles, lap_styles, prune_cache,
                                       style_summary, track_feature_table)
from rlsetup.critic.track_features import TrackScaler
from rlsetup.style import KEYS, STYLE_RL_COMMIT, STYLE_RL_SOURCES_SHA256, Track, style_vector

STYLE_RL_REPO = Path(os.environ.get('STYLE_RL_REPO', Path.home() / 'Documents/style-rl'))
GEO_IDX = [KEYS.index(k) for k in GEOMETRY_KEYS]
BT_IDX = [i for i in range(len(KEYS)) if i not in GEO_IDX]


# --- port equality with the pinned original ----------------------------------------------------------
@pytest.fixture(scope='module')
def original(tmp_path_factory):
    """style-rl's dstyle features.py / track.py at STYLE_RL_COMMIT, loaded from `git show` output."""
    pytest.importorskip('pandas', reason='pandas (dev/test only) is needed to run the original DataFrame code')
    if shutil.which('git') is None or not (STYLE_RL_REPO / '.git').exists():
        pytest.skip(f'style-rl repository not found at {STYLE_RL_REPO} (set STYLE_RL_REPO)')
    d = tmp_path_factory.mktemp('style_rl_pinned')
    mods = {}
    for path, sha in STYLE_RL_SOURCES_SHA256.items():
        r = subprocess.run(['git', '-C', str(STYLE_RL_REPO), 'show', f'{STYLE_RL_COMMIT}:{path}'], capture_output=True)
        if r.returncode:
            pytest.skip(f'style-rl commit {STYLE_RL_COMMIT[:10]} not available: {r.stderr.decode().strip()}')
        assert hashlib.sha256(r.stdout).hexdigest() == sha, f'{path} at the pinned commit changed'
        f = d / Path(path).name
        f.write_bytes(r.stdout)
        spec = importlib.util.spec_from_file_location(f'pinned_{f.stem}', f)
        mods[f.stem] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[f.stem])
    return mods


@pytest.fixture(scope='module')
def fixture_lap(tmp_path_factory):
    """A 2,000-sample lap on a TUM track (Monza) with positions, and the track as a style-rl CSV."""
    from rlsetup.critic.geometry import tum_dir
    xy, w_r, w_l = read_tum_csv(tum_dir() / 'tracks/Monza.csv')
    left, right = edges(xy, w_r, w_l)
    csv = tmp_path_factory.mktemp('trk') / 'Monza_geometry.csv'
    np.savetxt(csv, np.column_stack((xy, left, right)), delimiter=',', comments='',
               header='x_center,y_center,x_left,y_left,x_right,y_right')
    port_track = Track.from_csv(csv)
    rng = np.random.default_rng(3)
    n = 2000
    s = np.linspace(0, port_track.length_m, n, endpoint=False)
    c = np.column_stack([np.interp(s, port_track.s, port_track.center_grid[:, j]) for j in range(2)])
    lap = {'time_s': np.arange(n) * .04 + 3., 'throttle': np.clip(.5 + .5 * np.sin(s / 300), 0, 1),
           'brake': np.clip(np.cos(s / 210) - .4, 0, 1), 'steer_deg': 60 * np.sin(s / 90) + rng.normal(0, 2, n),
           'x': c[:, 0] + rng.normal(0, 1.5, n), 'y': c[:, 1] + rng.normal(0, 1.5, n)}
    return csv, port_track, lap, s


def test_pinned_sources_hash(original):
    assert set(original) == {'features', 'track'}


def test_port_equals_original(original, fixture_lap):
    import pandas as pd
    csv, port_track, lap, s = fixture_lap
    orig_track = original['track'].Track(csv)
    np.testing.assert_array_equal(orig_track.kappa, port_track.kappa)
    oc, pc = orig_track.find_corners(), port_track.find_corners()
    assert len(oc) == len(pc) > 5
    for col in ('s_start', 's_apex', 's_end', 'length_m', 'angle_deg', 'r_min_m', 'direction'):
        np.testing.assert_array_equal(oc[col].to_numpy(float), [getattr(c, col) for c in pc])
    df = pd.DataFrame(lap)
    for kw in ({}, {'lap_dist': s}):
        a = original['features'].style_vector(df, orig_track, **kw)
        b = style_vector(lap, port_track, **kw)
        assert list(a) == list(b) == KEYS
        np.testing.assert_array_equal([a[k] for k in KEYS], [b[k] for k in KEYS])
    q = orig_track.project(np.column_stack((lap['x'], lap['y'])), s_hint=s)
    a = original['features'].style_vector(df.drop(columns=['x', 'y']), orig_track, q=q)
    b = style_vector({k: v for k, v in lap.items() if k not in 'xy'}, port_track, q=q)
    np.testing.assert_array_equal([a[k] for k in KEYS], [b[k] for k in KEYS])
    tbl = original['features'].style_table([a, a])
    from rlsetup.style import style_table
    np.testing.assert_array_equal(tbl.to_numpy(float), style_table([b, b]))


# --- per-lap style on the synthetic export ------------------------------------------------------------
@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp('c2style')
    fx = write_fixture(root)
    cfg = Config()
    cfg = replace(cfg, geometry=replace(cfg.geometry, extra_dir=str(fx['extra_dir']), tum_dir=str(fx['tum_dir'])),
                  style_input=replace(cfg.style_input, cache_dir=str(root / 'cache')))
    data = load_exports(fx['sources'], log=lambda *a: None)
    frame = build_frame(data, cfg, log=lambda *a: None)
    adapter = F1LapsAdapter(cfg=cfg)
    laps, geoms = lap_styles(data, TRACK_MAP, cfg, adapter=adapter, use_cache=False, log=lambda *a: None)
    return {'cfg': cfg, 'data': data, 'frame': frame, 'laps': laps, 'geoms': geoms, 'adapter': adapter}


def test_nothing_is_dropped_and_reasons(built):
    laps = built['laps']
    assert len(laps.keys) == len(built['data'].laps) == 14
    assert sorted(laps.lap_reason[~laps.ok]) == [LAP_SPACING, LAP_TIME_CLOSURE]
    for t, reason in (('trk_good', OK), ('trk_long', GATE_LENGTH), ('trk_none', NO_GEOMETRY)):
        m = laps.ok & (laps.track == t)
        assert m.sum() == 4 and set(laps.geometry_reason[m]) == {reason}
        assert np.isfinite(laps.values[m][:, BT_IDX]).all()          # B/T/S rate numbers always computed
        assert np.isfinite(laps.values[m][:, GEO_IDX]).all() == (reason == OK)
        if reason != OK:
            assert np.isnan(laps.values[m][:, GEO_IDX]).all()
    assert np.isnan(laps.values[~laps.ok]).all()


def test_placeholder_flag(built):
    laps = built['laps']
    gated = laps.ok & (laps.geometry_reason == OK)
    assert laps.is_placeholder[gated].all() and set(laps.line_kind[gated]) == {CENTRE_LINE}
    assert not laps.is_placeholder[~gated].any()
    assert (laps.steer_kappa_corr[gated] < -.9).all() and np.isnan(laps.steer_kappa_corr[~gated]).all()
    # centre line on a 6 m half-width road: the line numbers are the half width (an artefact, hence placeholder)
    np.testing.assert_allclose(laps.values[gated][:, KEYS.index('L_border_mean_m')], 6., atol=.05)


def test_cells_are_medians_of_their_laps(built):
    frame, laps = built['frame'], built['laps']
    cells = cell_styles(frame, laps)
    assert len(cells.values) == len(frame.cells) == 14 and cells.has_style.sum() == 12
    row = {k: i for i, k in enumerate(laps.keys)}
    for c, keys in enumerate(frame.cell_laps):
        i = row[keys[0]]
        if laps.ok[i]:
            np.testing.assert_array_equal(cells.values[c], laps.values[i])      # one lap per cell
        else:
            assert np.isnan(cells.values[c]).all() and cells.reason[c] == laps.lap_reason[i]
    s = style_summary(laps.values[laps.ok])
    assert s['B_mean_active']['n'] == 12 and s['L_border_mean_m']['n'] == 4


def test_track_features_only_for_gated_tracks(built):
    frame = built['frame']
    z = track_feature_table(frame.track_names, built['geoms'], built['cfg'])
    assert z.shape == (3, len(TRACK_FEATURES))
    good = frame.track_names.index('trk_good')
    assert np.isfinite(z[good]).all() and z[good, 0] >= 2
    assert np.isnan(np.delete(z, good, axis=0)).all()


def test_determinism_and_cache(built):
    cfg, data, ad = built['cfg'], built['data'], built['adapter']
    assert not Path(cfg.style_input.cache_dir).exists()               # use_cache=False wrote nothing
    again, _ = lap_styles(data, TRACK_MAP, cfg, adapter=ad, use_cache=False, log=lambda *a: None)
    np.testing.assert_array_equal(again.values, built['laps'].values)
    first, _ = lap_styles(data, TRACK_MAP, cfg, adapter=ad, use_cache=True, log=lambda *a: None)
    cached, _ = lap_styles(data, TRACK_MAP, cfg, adapter=ad, use_cache=True, log=lambda *a: None)
    assert first.cache_hits == 0 and cached.cache_hits == int(built['laps'].ok.sum())
    np.testing.assert_array_equal(cached.values, built['laps'].values)
    env_dir = Path(cfg.style_input.cache_dir) / cache_env(ad)
    assert len(list(env_dir.glob('*.json'))) == int(built['laps'].ok.sum())


def test_cache_prune_removes_stale_entries_only(tmp_path):
    """QA M4: other cache_env directories and old flat files go; the current env and foreign files stay."""
    keep, stale = 'a' * 16, 'b' * 16
    for d in (keep, stale):
        (tmp_path / d).mkdir()
        (tmp_path / d / ('c' * 64 + '.json')).write_text('{}')
    (tmp_path / ('d' * 64 + '.json')).write_text('{}')
    (tmp_path / 'notes.txt').write_text('x')
    assert prune_cache(tmp_path, keep) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [keep, 'notes.txt']
    import scipy
    assert len(cache_env(F1LapsAdapter())) == 16 and np.__version__ and scipy.__version__


def test_lap_corr_alignment(built):
    """T9 (QA M3): per gated track every lap has its corr, and `steer_kappa_corr` equals a per-lap
    recomputation at the fitted shift and scale."""
    laps, data = built['laps'], built['data']
    for t, tg in built['geoms'].items():
        if tg.chosen is None:
            continue
        g, r = tg.chosen
        keys = laps.keys[(laps.track == t) & laps.ok]
        assert len(keys) == len(r.lap_corr) == len(r.lap_index)
        k = int(round(r.shift_m / g.track.ds)) % len(g.track.kappa)
        for key in keys:
            si = lap_style_input(data.laps[int(key)])
            want = corr_over_shifts(si.raw_distance_m, si.raw_steer_deg, g.track.kappa, g.track.ds, r.scale)[k]
            assert laps.steer_kappa_corr[laps.keys == key][0] == pytest.approx(want, abs=1e-12)


def test_track_scaler_uses_training_tracks_only():
    """T6 (QA M6): mean / std from the training tracks only; NaN rows stay NaN; clip."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=(8, len(TRACK_FEATURES)))
    z[6] = np.nan
    train = np.array([0, 1, 2, 3, 4, 6])
    s1 = TrackScaler().fit(z, train)
    w = z.copy()
    w[[5, 7]] *= 1000.
    s2 = TrackScaler().fit(w, train)
    np.testing.assert_array_equal(s1.mean, s2.mean)
    np.testing.assert_array_equal(s1.std, s2.std)
    np.testing.assert_allclose(s1.mean, z[:5].mean(axis=0))
    out = s2.transform(w, np.arange(8))
    assert np.isnan(out[6]).all() and np.isfinite(np.delete(out, 6, axis=0)).all()
    assert np.abs(out[[5, 7]]).max() <= 3.
    with pytest.raises(ValueError):
        TrackScaler().fit(z, [6])


# --- R4 transform -------------------------------------------------------------------------------------
def cell_table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    v = np.abs(rng.lognormal(0, .5, (n, len(KEYS))))
    v[0, KEYS.index('T_apply_rate')] = 0.                              # a lap with no throttle event
    v[1, KEYS.index('S_corner_peak_angle')] = np.nan                    # an ungated track
    return v


def test_r4_transform_definition():
    cfg = StyleTransformConfig()
    v = cell_table()
    train = np.arange(30)
    t = StyleTransform(cfg).fit(v, train)
    z = t.transform(v)
    assert z.shape == v.shape
    assert (z[:, ~t.mask] == 0).all() and [k for k, m in zip(KEYS, t.mask) if not m] == list(cfg.masked_keys)
    assert t.mask.sum() == 9 and np.abs(z[np.isfinite(z)]).max() <= cfg.z_clip
    j = KEYS.index('B_apply_rate')                                      # a rate: log first
    x = np.log(v[train, j])
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    np.testing.assert_allclose(z[:, j], np.clip((np.log(v[:, j]) - med) / ((q3 - q1) / cfg.iqr_to_sd), -3, 3))
    j = KEYS.index('B_mean_active')                                     # not a rate: no log
    q1, med, q3 = np.percentile(v[train, j], [25, 50, 75])
    np.testing.assert_allclose(z[:, j], np.clip((v[:, j] - med) / ((q3 - q1) / cfg.iqr_to_sd), -3, 3))
    assert np.isfinite(z[0, KEYS.index('T_apply_rate')])               # log floor
    assert np.isnan(z[1, KEYS.index('S_corner_peak_angle')])           # NaN stays NaN
    assert set(cfg.log_keys) == {k for k in KEYS if 'rate' in k or 'reversals' in k}


def test_r4_transform_uses_training_cells_only():
    v = cell_table()
    train = np.arange(30)
    t1 = StyleTransform().fit(v, train)
    w = v.copy()
    w[30:] *= 1000.                                                     # change only the held-out cells
    t2 = StyleTransform().fit(w, train)
    np.testing.assert_array_equal(t1.center, t2.center)
    np.testing.assert_array_equal(t1.scale, t2.scale)
    t3 = StyleTransform().fit(w, np.arange(40))
    assert not np.allclose(t1.center, t3.center)
    with pytest.raises(RuntimeError):
        StyleTransform().transform(v)
    assert StyleTransform().fit(v, train).state()['keys'] == KEYS
