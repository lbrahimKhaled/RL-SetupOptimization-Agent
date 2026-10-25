import math
import numpy as np
import pytest
from rlsetup.driving.reference import ReferenceLine, TrackGrid, track_grid


def circle(radius=50., n=2000):
    a = np.linspace(0., 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])  # counter-clockwise


def test_circle_arc_length():
    n, r = 2000, 50.
    ref = ReferenceLine.from_xy(circle(r, n))
    assert ref.length == pytest.approx(2 * n * r * math.sin(math.pi / n), rel=1e-12)
    assert ref.length == pytest.approx(2 * math.pi * r, rel=1e-5)


def test_signed_offset_and_heading_error():
    ref = ReferenceLine.from_xy(circle(50.))
    s = ref.length / 8                                   # 45 deg on a CCW circle: left = inside
    p, t = ref.frame(s)
    left = np.array([-t[1], t[0]])
    for d in (-2., 1.5):
        x, y = p + d * left
        s_m = ref.match((x, y), s)
        e, heading = ref.errors_at(s_m, x, y, math.atan2(t[1], t[0]) + .1)
        assert e == pytest.approx(d, abs=2e-3)
        assert heading == pytest.approx(.1, abs=2e-3)


def test_match_wraps_across_line():
    ref = ReferenceLine.from_xy(circle(50.))
    L = ref.length
    near_end = ref.point(L - .5)
    assert ref.match(near_end, hint=1.) == pytest.approx(L - .5, abs=1e-3)
    assert ref.match(ref.point(.5), hint=L - 1.) == pytest.approx(.5, abs=1e-3)
    assert 0. <= ref.match(ref.point(L - 1e-9)) < L


def test_from_csv_round_trip(tmp_path, track):
    f = tmp_path / 'line.csv'
    np.savetxt(f, track.path[:-1, 1:], delimiter=',', header='x_m,y_m')
    ref = ReferenceLine.from_csv(f)
    np.testing.assert_allclose(ref.path, track.path, atol=1e-9)


def test_track_path_reference_matches_ocd_env_frame(track):
    """Built from Track.path, frame/match are those of OCDRacingEnv (same helpers)."""
    ref = ReferenceLine.from_track_path(track)
    xy = track.point(1234.5) + [.3, -.2]
    assert ref.match(xy, 1230.) == track.match(xy, 1230.)[0]


def test_grid_edges_sides_and_ref(track):
    ref = ReferenceLine.from_track_path(track)
    g = track_grid(track, ref, 1.)
    assert track_grid(track, ref, 1.) is g                        # cached per process
    s = np.arange(0., track.length, 37.)
    r, left, right = g.at(s)
    np.testing.assert_allclose(r, ref.point(s), atol=.05)          # xi_ref = path here
    tangents = np.array([ref.frame(x)[1] for x in s])
    cross = lambda d: tangents[:, 0] * d[:, 1] - tangents[:, 1] * d[:, 0]
    assert (cross(left - r) > 0).all() and (cross(right - r) < 0).all()   # + = left
    width = np.linalg.norm(left - right, axis=1)
    assert width.min() > 8. and width.max() < 25.
    # s_ref continuity across the lap end
    assert g.s_ref_at(track.length - .25) == pytest.approx(track.length - .25, abs=.05)
    assert g.s_ref_at(.25) == pytest.approx(.25, abs=.05)


def test_grid_rejects_reversed_reference(track):
    reversed_ref = ReferenceLine.from_xy(track.path[-2::-1, 1:])
    with pytest.raises(ValueError):
        TrackGrid(track, reversed_ref, 5.)


def test_rejects_bad_paths(track):
    """QA MINOR 10: open path, non-increasing s, duplicate point (zero-length segment)."""
    good = track.path.copy()
    ReferenceLine(good)
    open_path = good[:-1]
    s_bad = good.copy()
    s_bad[5, 0] = s_bad[4, 0]
    dup = np.insert(good, 5, good[5], axis=0)
    dup[6:, 0] += 1e-3        # s still strictly increasing, but points 5 and 6 coincide
    shifted = good.copy()
    shifted[:, 0] += 1.
    for bad in (open_path, s_bad, dup, shifted):
        with pytest.raises(ValueError):
            ReferenceLine(bad)
    xy = track.path[:-1, 1:]
    with pytest.raises(ValueError):           # duplicate CSV row through from_xy
        ReferenceLine.from_xy(np.insert(xy, 10, xy[10], axis=0))
