"""C2a geometry (spec S3, review R2): TUM edges, the R2 gate (accept / reject on corr / length / no
laps), the shift + scale fit, candidate order, the placeholder line, the width placeholder, the
bacinger conversion helpers (projection, JSON-only parsing, periodic spline resample)."""
from dataclasses import replace
import importlib.util
import json
import numpy as np
import pytest
from c2_fixture import START_OFFSET_M, make_lap, resample, shape_xy, truth_track, write_tum
from rlsetup.critic.config import (ExtraGeometryConfig, GATE_LENGTH, GATE_LOW_CORR, GATE_NO_LAPS, GATE_SIGN, GeometryConfig,
                                   NO_GEOMETRY, OK, REPO_ROOT)
from rlsetup.critic.geometry import (CENTRE_LINE, TUM_LINE, Geometry, candidates, corr_over_shifts, edges, gate,
                                     gate_track, load_geometry, polyline_length, read_tum_csv, tum_dir,
                                     tum_median_total_width)
from rlsetup.style import Track as StyleTrack


def geom_from(xy, half=6.):
    w = np.full(len(xy), half)
    return Geometry('x', 'extra', None, '', xy, w, w)


@pytest.fixture(scope='module')
def laps():
    rng = np.random.default_rng(1)
    tr = truth_track(1.)
    out = []
    for u in ('drv_alpha', 'drv_beta', 'drv_delta'):
        lp = make_lap(tr, u, rng)
        out.append((lp['d'], lp['steer'] * 115.))
    return out


def test_edges_left_is_inside_on_a_ccw_circle():
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    xy = 100 * np.column_stack((np.cos(th), np.sin(th)))
    left, right = edges(xy, np.full(400, 2.), np.full(400, 3.))
    np.testing.assert_allclose(np.linalg.norm(left, axis=1), 97., atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(right, axis=1), 102., atol=1e-6)
    StyleTrack(xy, left, right)                         # style-rl's border-order check passes


def test_corr_over_shifts_equals_brute_force():
    rng = np.random.default_rng(0)
    kappa = rng.standard_normal(200)
    dist = np.sort(rng.uniform(0, 398, 150))
    val = rng.standard_normal(150)
    for scale in (1., .97, 1.03):                       # 1.03: the lap wraps past the loop end
        got = corr_over_shifts(dist, val, kappa, 2., scale)
        m = int(np.floor(dist[-1] * scale / 2.)) + 1
        u = np.interp(np.arange(m) * 2. / scale, dist, val)
        want = [np.corrcoef(u, kappa[(np.arange(m) + s) % 200])[0, 1] for s in range(200)]
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_gate_accepts_matching_geometry_and_finds_shift(laps):
    g = geom_from(resample(shape_xy(1.), 5.))
    r = gate(g, laps)
    assert r.passed and r.reason == OK and r.corr_ok and r.length_ok
    assert r.median_abs_corr > .95 and (r.lap_corr < 0).all()       # game steer + = right, kappa + = left
    assert abs(r.shift_m - START_OFFSET_M) <= 4.                       # ~2 m kappa grid
    assert r.scale == pytest.approx(1., abs=.005)
    d = laps[0][0]
    np.testing.assert_allclose(r.progress(d), (r.shift_m + r.scale * d) % r.L_geo)


def test_gate_rejects_on_length(laps):
    r = gate(geom_from(resample(shape_xy(.97), 5.)), laps)
    assert not r.passed and r.reason == GATE_LENGTH and r.corr_ok and not r.length_ok
    assert r.length_rel == pytest.approx(1 / .97 - 1, abs=.002)
    assert r.scale == pytest.approx(.97, abs=.005)                     # the fitted scale absorbs the size


def test_gate_rejects_on_correlation(laps):
    other = resample(shape_xy(1., lobes=2, phase=1.), 5.)               # a different layout
    r = gate(geom_from(other), laps)
    assert not r.passed and r.reason == GATE_LOW_CORR and not r.corr_ok


def test_gate_rejects_mirror_image(laps):
    """T5 (QA M2): a mirror image has the same |corr| at the other sign; with the source's declared
    sign (-1: steer + = right turn) it fails with GATE_SIGN, the true layout passes."""
    r = gate(geom_from(resample(shape_xy(1.), 5.) * [-1, 1]), laps, expected_sign=-1)
    assert not r.passed and r.reason == GATE_SIGN and r.corr_ok and not r.sign_ok
    assert r.median_signed_corr > 0
    ok = gate(geom_from(resample(shape_xy(1.), 5.)), laps, expected_sign=-1)
    assert ok.passed and ok.sign_ok and ok.median_signed_corr < 0


def test_gate_reports_lap_index_closure_and_groups(laps):
    """QA M3 / I1 / I2: lap_index says which input laps were used; closure gap; per-group |corr|."""
    short = (np.array([1., 2.]), np.array([0., 1.]))                   # < 4 samples: left out
    r = gate(geom_from(resample(shape_xy(1.), 5.)), [laps[0], short, *laps[1:]], expected_sign=-1,
             groups=['G1', 'G1', 'G2', 'G2'])
    np.testing.assert_array_equal(r.lap_index, [0, 2, 3])
    assert len(r.lap_corr) == 3 and r.n_laps == 3
    assert r.closure_gap_m == pytest.approx(r.scale * r.L_game - r.L_geo)
    assert set(r.group_median_abs_corr) == {'G1', 'G2'}
    assert r.group_median_abs_corr['G1'] == pytest.approx(abs(r.lap_corr[0]))


def test_gate_bars_are_config(laps):
    g = geom_from(resample(shape_xy(.97), 5.))
    assert gate(g, laps, GeometryConfig(max_length_rel=.05)).passed
    assert gate(geom_from(resample(shape_xy(1.), 5.)), laps, GeometryConfig(min_abs_corr=.999)).reason == GATE_LOW_CORR


def test_gate_without_laps():
    r = gate(geom_from(resample(shape_xy(1.), 5.)), [])
    assert not r.passed and r.reason == GATE_NO_LAPS


def test_candidates_order_and_track_reason(tmp_path, laps):
    cfg = GeometryConfig(extra_dir=str(tmp_path / 'extra'), tum_dir=str(tmp_path / 'tum'))
    write_tum(tmp_path / 'extra/T.csv', resample(shape_xy(.97), 5.))   # extra fails (length) ...
    write_tum(tmp_path / 'tum/tracks/T.csv', resample(shape_xy(1.), 5.))  # ... TUM passes
    (tmp_path / 'extra/T.json').write_text(json.dumps({'width_assumed': True}))
    c = candidates('T', cfg)
    assert [g.kind for g in c] == ['extra', 'tum'] and c[0].width_assumed and not c[1].width_assumed
    tg = gate_track('T', laps, cfg)
    assert tg.chosen[0].kind == 'tum' and tg.reason == OK
    assert gate_track('Nothing', laps, cfg).reason == NO_GEOMETRY
    (tmp_path / 'tum/tracks/T.csv').unlink()
    assert gate_track('T', laps, cfg).reason == GATE_LENGTH


def test_placeholder_line_centre_and_raceline(tmp_path):
    xy = resample(shape_xy(1.), 5.)
    write_tum(tmp_path / 'tracks/R.csv', xy)
    left, _ = edges(xy, np.full(len(xy), 6.), np.full(len(xy), 6.))
    line = xy + 2. / 6. * (left - xy)                                   # 2 m left of the centre
    (tmp_path / 'racelines').mkdir()
    np.savetxt(tmp_path / 'racelines/R.csv', np.roll(line, 37, axis=0), delimiter=',', header='x_m,y_m')  # other start
    g = load_geometry(tmp_path / 'tracks/R.csv', 'tum', tmp_path / 'racelines/R.csv')
    assert g.line_kind == TUM_LINE and g.line_sha256
    q = g.line.q(np.linspace(0, g.length_m, 500, endpoint=False))
    np.testing.assert_allclose(q['lateral_m'], 2., atol=.05)
    np.testing.assert_allclose(q['left_width_m'], 6., atol=.05)
    np.testing.assert_allclose(q['right_width_m'], 6., atol=.05)
    c = load_geometry(tmp_path / 'tracks/R.csv', 'extra')
    assert c.line_kind == CENTRE_LINE
    np.testing.assert_allclose(c.line.q(np.arange(0., 4000., 7.))['lateral_m'], 0., atol=1e-6)


def test_tum_files_build_style_tracks():
    """Every vendored TUM track gives a valid style-rl Track (border order matches the direction)."""
    for p in sorted((tum_dir() / 'tracks').glob('*.csv')):
        g = load_geometry(p, 'tum', tum_dir() / 'racelines' / p.name)
        assert abs(g.length_m - polyline_length(g.xy)) < 1e-6 and len(g.corners) >= 2      # IMS: 2
        q = g.line.q(np.arange(0., g.length_m, 5.))                # TUM raceline inside its own edges
        inside = (q['lateral_m'] < q['left_width_m']) & (q['lateral_m'] > -q['right_width_m'])
        assert g.line_kind == TUM_LINE and inside.mean() > .99, p.stem


def test_width_placeholder_is_half_the_tum_median():
    assert ExtraGeometryConfig().assumed_half_width_m == pytest.approx(tum_median_total_width() / 2, abs=5e-4)


# --- scripts/fetch_track_geometry.py helpers --------------------------------------------------------
@pytest.fixture(scope='module')
def fetch():
    spec = importlib.util.spec_from_file_location('fetch_track_geometry', REPO_ROOT / 'scripts/fetch_track_geometry.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_enu_projection_distances(fetch):
    lon0, lat0 = 2.26, 41.57                           # Catalunya area
    # 0.01 deg of latitude: WGS84 meridian arc at 41.57 deg = 1110.6 m (M = a(1-e2)/(1-e2 sin^2)^1.5)
    e2 = fetch.WGS84_F * (2 - fetch.WGS84_F)
    m = fetch.WGS84_A * (1 - e2) / (1 - e2 * np.sin(np.radians(lat0)) ** 2) ** 1.5
    p = fetch.lonlat_to_enu(np.array([[lon0, lat0 - .005], [lon0, lat0 + .005]]), lon0, lat0)
    assert np.linalg.norm(p[1] - p[0]) == pytest.approx(m * np.radians(.01), abs=.01)
    n = fetch.WGS84_A / np.sqrt(1 - e2 * np.sin(np.radians(lat0)) ** 2)
    q = fetch.lonlat_to_enu(np.array([[lon0 + .01, lat0]]), lon0, lat0)[0]
    assert q[0] == pytest.approx(n * np.cos(np.radians(lat0)) * np.radians(.01), rel=1e-6) and abs(q[1]) < 1.


def test_geojson_parsing_is_json_only_and_validated(fetch):
    coords = [[2.0 + .001 * np.cos(t), 41.0 + .001 * np.sin(t)] for t in np.linspace(0, 6.2, 30)]
    good = json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'LineString', 'coordinates': coords}}]}).encode()
    assert fetch.parse_geojson(good).shape == (30, 2)
    with pytest.raises(ValueError):
        fetch.parse_geojson(json.dumps({'type': 'FeatureCollection', 'features': []}).encode())
    with pytest.raises(ValueError):
        fetch.parse_geojson(good.replace(b'41.0', b'410.0', 1))
    with pytest.raises(ValueError):
        fetch.parse_geojson(b'__import__("os")')


def test_periodic_spline_resample(fetch):
    th = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    xy = 800 * np.column_stack((np.cos(th), np.sin(th)))
    pts, info = fetch.smooth_resample(xy, 2., 5.)
    assert info['length_m'] == pytest.approx(2 * np.pi * 800, rel=2e-3)
    assert info['spacing_m'] == pytest.approx(5., abs=.05) and info['vertex_dev_rms_m'] < 2.5
    step = np.linalg.norm(np.diff(np.vstack((pts, pts[:1])), axis=0), axis=1)
    assert step.max() - step.min() < .05                # uniform, and closed (last -> first is one step)
    open_xy, gap = fetch.close_loop(np.vstack((xy, xy[:1])))
    assert len(open_xy) == 100 and gap == 0.
