"""xi_ref from a CSV ('csv:<repo-relative path>'), the drive line that follows it, preset f1_sha_w35.

The sf70h_f1 tests need ocd_first_lap branch sf70h-f1 on the path (PYTHONPATH=<ocd_first_lap-f1>).
Full along-the-line check (every 100 m, 3 start fractions) and the demo dataset clean rate:
scripts/check_line_starts.py.
"""
import hashlib
import json
from dataclasses import replace
import numpy as np
import pytest
from ocd_lap import cars
from rlsetup.driving import qss_profile
from rlsetup.driving.config import EnvConfig
from rlsetup.driving.demos import run_episode
from rlsetup.driving.env import DrivingEnv
from rlsetup.driving.reference import ReferenceLine, csv_reference_path, reference_source
from rlsetup.driving.train import reference_metadata
from rlsetup.driving.train_config import LINE_IQP_W35, PRESETS, config_json, train_config_from_dict
from rlsetup.provenance import REPO_ROOT

LINE_DIGEST = 'eaf1792a00be2da0'      # docs/results/shanghai_line_setup (IQP, width allowance 3.5 m)
needs_f1 = pytest.mark.skipif('sf70h_f1' not in cars.CARS,
                              reason='sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path')


def test_csv_reference_loads_from_the_repo_root_and_digests(track, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                          # resolved against the repo, not the cwd
    ref = ReferenceLine.for_track(track, LINE_IQP_W35)
    assert (ref.digest, ref.name) == (LINE_DIGEST, LINE_IQP_W35)
    assert ref.length == pytest.approx(5364.67, abs=.01)
    assert np.array_equal(ref.path[0, 1:], ref.path[-1, 1:])                      # closed
    src = reference_source(LINE_IQP_W35)
    rel = LINE_IQP_W35.split(':', 1)[1]
    assert src['csv_sha256'] == hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
    assert reference_source('path') == {'spec': 'path', 'csv_path': None, 'csv_sha256': None}
    assert csv_reference_path('raceline') is None
    for bad in ('csv:', 'line.csv', 'Track.path'):
        with pytest.raises(ValueError):
            ReferenceLine.for_track(track, bad)


def test_f1_sha_w35_is_f1_sha_with_only_the_line_changed():
    c, base = PRESETS['f1_sha_w35'](), PRESETS['f1_sha']()
    assert (c.preset, c.env.reference, base.env.reference) == ('f1_sha_w35', LINE_IQP_W35, 'path')
    assert replace(c, preset=base.preset, env=replace(c.env, reference=base.env.reference)) == base
    assert train_config_from_dict(json.loads(config_json(c))) == c
    assert json.loads(config_json(c))['env']['reference'] == LINE_IQP_W35     # a string: reaches the workers


def test_drive_line_follows_the_csv_reference(make_env, track):
    """Placement, settle and evaluation start on the CSV line; progress stays on Track.path."""
    env = make_env(EnvConfig(reference=LINE_IQP_W35))
    assert env.drive_line is env.ref and env.ref.digest == LINE_DIGEST
    assert make_env(EnvConfig()).drive_line.name.endswith('Track.path')            # unchanged
    assert make_env(EnvConfig(reference='raceline')).drive_line.name.endswith('Track.path')
    placed = make_env(EnvConfig(reference=LINE_IQP_W35, settle_s=0.))
    for s0 in (300., 2100., 4700.):
        _, info = placed.reset(seed=0, options={'start_s': s0, 'start_speed': 20.})
        s_ref0 = float(placed.grid.s_ref_at(s0))
        assert [placed.state['x'], placed.state['y']] == pytest.approx(list(env.ref.point(s_ref0)), abs=1e-6)
        assert abs(info['e_ref']) < 1e-3 and placed.s_drive == placed.s_ref
        assert info['s_track'] == pytest.approx(track.match([placed.state['x'], placed.state['y']], s0)[0])
    _, info = env.reset(seed=1)
    assert env.s_drive == env.s_ref and abs(info['e_ref']) < 1.5      # settled on the new line
    _, info = env.reset(seed=1, options={'evaluation': True})
    assert abs(info['e_ref']) < 1.5


def test_line_lap_time_reproduces_the_profile_on_track_path(track, qss):
    assert qss_profile.line_lap_time(qss, track, track.path, EnvConfig()) == pytest.approx(qss.lap_time_s, rel=1e-9)


def test_reference_metadata_only_for_csv_lines(make_env, track, qss):
    assert reference_metadata(EnvConfig(), make_env(EnvConfig()), track, qss) is None
    assert reference_metadata(EnvConfig(reference='raceline'), make_env(EnvConfig(reference='raceline')), track, qss) is None
    cfg = EnvConfig(reference=LINE_IQP_W35)
    m = reference_metadata(cfg, make_env(cfg), track, qss)
    assert (m['spec'], m['digest'], m['drive_line_is_reference']) == (LINE_IQP_W35, LINE_DIGEST, True)
    assert m['qss_lap_on_line_s'] < m['qss_lap_on_track_path_s'] == qss.lap_time_s


@needs_f1
def test_f1_sha_w35_settle_demos_and_qss(track):
    """sf70h_f1 on the w3.5 line: settles every 500 m at f 0.3 / 0.7, demo episodes clean,
    QSS lap on the line = docs/results/shanghai_line_setup 96.01 s (102.53 s on Track.path)."""
    from scripts.check_line_starts import settle_check
    cfg = PRESETS['f1_sha_w35']()
    q = qss_profile.for_env(track, cfg.env)
    env = DrivingEnv(cfg.env, track=track, qss=q)
    rows = settle_check(env, 500., (.3, .7))
    assert all(r['ok'] for r in rows) and max(abs(r['e_ref']) for r in rows) < 1.
    causes = [run_episode(env, cfg.demos, ep)['cause'] for ep in range(2)]
    print(f'demo episodes on the w3.5 line: {causes}')
    assert all(c in ('none', 'time_limit') for c in causes)
    m = reference_metadata(cfg.env, env, track, q)
    assert m['qss_lap_on_line_s'] == pytest.approx(96.0096, abs=1e-3)
    assert m['qss_lap_on_track_path_s'] == pytest.approx(102.5315, abs=1e-3)


def test_f1_best_w35_differs_from_f1_sha_w35_only_in_setup():
    from dataclasses import replace
    from rlsetup.driving.train_config import PRESETS
    a, b = PRESETS['f1_sha_w35'](), PRESETS['f1_best_w35']()
    assert b.env.setup == 'sha_f1_best'
    assert replace(b, preset=a.preset, env=replace(b.env, setup=a.env.setup)) == a
