"""Run-matched comparison protocol: no network, training, or full physical lap."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/comparison'))
import comparison_state as state
import fair_tum_baseline as fair
from comparison_recording import RecordingEnv, check
from rlsetup.driving.config import EnvConfig, DemoConfig
from rlsetup.driving.env import DrivingEnv
from rlsetup.driving.reference import ReferenceLine


def checkpoint(path, step):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps({'num_timesteps': step}))


def record(step, lap=105.):
    return {'step': step, 'det_clean': True, 'det_termination': 'lap_complete',
            'det_truncation': None, 'lap2_s': lap, 'new_best': True}


def source_run(tmp_path, step):
    run = tmp_path / f'run{step}'
    (run / 'checkpoints').mkdir(parents=True)
    config = json.dumps({'env': {'setup': 'p1_TUM'}, 'device': 'cpu'}).encode()
    (run / 'config.json').write_bytes(config)
    (run / 'provenance.json').write_text(json.dumps({'config_sha256': hashlib.sha256(config).hexdigest()}))
    (run / 'evals.jsonl').write_text(json.dumps(record(step)) + '\n')
    checkpoint(run / 'checkpoints/best.zip', step)
    return run


def test_snapshot_uses_requested_run_and_survives_live_checkpoint_replace(tmp_path):
    for step in (650000, 2100000):
        run = source_run(tmp_path, step)
        out = tmp_path / f'out{step}'
        snapshot = state.prepare_snapshot(run, out)
        assert snapshot['checkpoint']['num_timesteps'] == step
        assert snapshot['selected_eval'] == record(step)
        original = (out / 'checkpoint.zip').read_bytes()
        checkpoint(run / 'checkpoints/new.zip', step + 50000)
        (run / 'checkpoints/new.zip').replace(run / 'checkpoints/best.zip')
        assert (out / 'checkpoint.zip').read_bytes() == original
        assert state.checkpoint_step(out / 'checkpoint.zip') == step
        assert state.digest(original) == snapshot['checkpoint']['sha256']
        with pytest.raises(FileExistsError):
            state.prepare_snapshot(run, out)


def test_freeze_copies_one_inode_during_atomic_source_replacement(tmp_path, monkeypatch):
    source, destination = tmp_path / 'live.zip', tmp_path / 'frozen.zip'
    replacement = tmp_path / 'new.zip'
    checkpoint(source, 10)
    checkpoint(replacement, 20)
    original_fstat = state.os.fstat
    calls = 0
    def replacing_fstat(fd):
        nonlocal calls
        value = original_fstat(fd)
        if calls == 0:
            replacement.replace(source)
        calls += 1
        return value
    monkeypatch.setattr(state.os, 'fstat', replacing_fstat)
    frozen = state.freeze_checkpoint(source, destination)
    assert frozen['num_timesteps'] == 10
    assert state.checkpoint_step(source) == 20


def test_freeze_rejects_in_place_mutation(tmp_path, monkeypatch):
    source, destination = tmp_path / 'live.zip', tmp_path / 'frozen.zip'
    checkpoint(source, 10)
    original_fstat = state.os.fstat
    calls = 0
    def changed_fstat(fd):
        nonlocal calls
        value = original_fstat(fd)
        calls += 1
        if calls == 2:
            return SimpleNamespace(st_size=value.st_size + 1, st_mtime_ns=value.st_mtime_ns)
        return value
    monkeypatch.setattr(state.os, 'fstat', changed_fstat)
    with pytest.raises(RuntimeError, match='changed in place'):
        state.freeze_checkpoint(source, destination)


def test_expected_eval_is_clean_specific_and_unambiguous():
    assert state.select_eval([record(12), record(15, 104.)], 15)['lap2_s'] == 104.
    with pytest.raises(ValueError, match='no clean'):
        state.select_eval([dict(record(15), det_clean=False)], 15)
    with pytest.raises(ValueError, match='no clean'):
        state.select_eval([record(12)], 15)
    with pytest.raises(ValueError, match='conflicting'):
        state.select_eval([record(15, 103.), record(15, 104.)], 15)


def test_live_eval_reader_only_ignores_incomplete_last_line():
    data = json.dumps(record(1)).encode() + b'\n'
    assert state.read_evals(data + b'{"step":') == [record(1)]
    with pytest.raises(json.JSONDecodeError):
        state.read_evals(b'not-json\n' + data)


def test_snapshot_rejects_changed_saved_config(tmp_path):
    run = source_run(tmp_path, 10)
    (run / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='config hash'):
        state.prepare_snapshot(run, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_spatial_speed_mapping_is_xy_matching_not_arc_fraction(track):
    raceline = ReferenceLine.from_track_raceline(track)
    profile = SimpleNamespace(speed=lambda s: float(s % raceline.length))
    mapping = fair.SpatialTumTarget(track, raceline, profile, grid_m=1.)
    differences = []
    for station in np.linspace(100., track.length - 100., 15):
        preview, tum_station, speed = mapping.preview(station - 20., 40., .5)
        exact = raceline.match(track.point(preview))
        offset = (tum_station - exact + raceline.length / 2) % raceline.length - raceline.length / 2
        assert abs(offset) < .3  # 1 m grid interpolation of spatially matched positions
        assert speed == pytest.approx(tum_station % raceline.length, abs=1e-10)
        arc_fraction = preview / track.length * raceline.length
        differences.append(abs((tum_station - arc_fraction + raceline.length / 2) % raceline.length - raceline.length / 2))
    assert max(differences) > 3., 'must detect the incorrect arc-fraction mapping'


def test_evaluation_config_preserves_fixed_start_settle_and_brakes():
    cfg = EnvConfig(setup='p1_TUM', reference='raceline', max_brake_torque_Nm=15000.,
                    demo=replace(DemoConfig(), brake_cap=.17))
    evaluated = fair.evaluation_config(cfg)
    before, after = cfg.to_dict(), evaluated.to_dict()
    assert {key for key in before if before[key] != after[key]} == {'evaluation', 'record_telemetry'}
    assert evaluated.demo is cfg.demo
    assert evaluated.demo.brake_cap == .17
    assert cfg.evaluation is False
    assert evaluated.action_repeat == 2 and evaluated.settle_s == 2.


def test_recorder_and_controller_keep_genuine_env_settle_and_step(track, qss):
    cfg = fair.evaluation_config(replace(EnvConfig(), eval_max_s=.04))
    plain, recorded = DrivingEnv(cfg, track=track, qss=qss), RecordingEnv(cfg, track=track, qss=qss)
    try:
        obs_plain, start_plain = plain.reset(seed=5)
        obs_recorded, start_recorded = recorded.reset(seed=5)
        np.testing.assert_array_equal(obs_plain, obs_recorded)
        assert plain.state == recorded.state and start_plain['start'] == start_recorded['start']
        integral = recorded.settle_driver.integral
        line = recorded.geo.path_ref
        driver = fair.DemoDriver(line, replace(cfg.demo, brake_cap=1.), pedal_deadzone=cfg.pedal_deadzone,
                                 control_dt_s=recorded.decision_dt, car=cfg.car)
        driver.reset(integral)
        assert driver.integral == integral
        assert recorded.settle_driver.cfg.brake_cap == .3
        assert driver.cfg.brake_cap == 1. and driver.dt == .04
        action = driver.act(recorded.state, recorded.s, 20.)
        out_plain, out_recorded = plain.step(action), recorded.step(action)
        np.testing.assert_array_equal(out_plain[0], out_recorded[0])
        assert out_plain[1:4] == out_recorded[1:4]
        assert plain.state == recorded.state
        assert len(recorded.rows) == 2
        assert [row['decision_end'] for row in recorded.rows] == [0, 1]
        assert recorded.rows[-1]['t_s'] == .04
    finally:
        plain.close()
        recorded.close()


@pytest.mark.parametrize('reference,column', [('path', 'e_path_m'), ('raceline', 'e_raceline_m')])
def test_recorder_checks_configured_reference(reference, column):
    row = {'decision_end': 1, 'e_path_m': 9., 'e_raceline_m': 4., 'e_ref_env_m': 4. if reference == 'raceline' else 9.}
    env = SimpleNamespace(cfg=SimpleNamespace(reference=reference), rows=[row],
                          clock=SimpleNamespace(lap_times=[100., 101.], crossings=[0., 100., 201.]))
    result = check(env, SimpleNamespace(lap_times_s=[100., 101.]))
    assert result['lap_times_match_env']
    assert result['max_abs_configured_ref_minus_env_e_ref_m'] == 0.


def test_fastest_clean_lap_wins_even_if_factor_is_lower():
    def attempt(factor, lap, **kw):
        return dict(factor=factor, clean=True, termination='lap_complete', truncation=None,
                    lap_times_s=[lap + 1., lap], lap2_s=lap, recorder_check={'lap_times_match_env': True}, **kw)
    fast_valid = attempt(.86, 105.)
    attempts = [attempt(1., 109.), fast_valid, dict(attempt(.98, 99.), clean=False, termination='off_track'),
                dict(attempt(.96, 98.), truncation='time_limit'), dict(attempt(.94, 97.), lap_times_s=[97.]),
                dict(attempt(.92, 96.), lap2_s=float('nan'))]
    assert fair.choose_best(attempts) is fast_valid
    assert fair.choose_best(attempts[2:]) is None


def test_reproduction_has_explicit_tolerance_and_device_difference():
    within = state.reproduction(105.0001, 105., True, .001, 'mps')
    assert within['within_tolerance'] and within['device_changed']
    assert within['difference_s'] == pytest.approx(.0001)
    assert not state.reproduction(None, 105., False, 1., 'cpu')['within_tolerance']
    assert not state.reproduction(106., 105., True, .001, 'cpu')['within_tolerance']


def test_pp_entry_point_does_not_import_torch(tmp_path):
    import os
    import subprocess
    script = ROOT / 'scripts/comparison'
    code = f"import sys; sys.path.insert(0, {str(script)!r}); import fair_tum_baseline; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
