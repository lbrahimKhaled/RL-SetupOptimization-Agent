"""Dashboard rates and timings must distinguish resets and teleport starts."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('ac_dashboard',
                                             Path(__file__).parents[1]/'scripts/ac_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def test_finish_rate_and_timed_laps_exclude_empty_and_teleport_times():
    rows = [dict(cycle=1, transitions=100, stop_reason='lap_complete', seconds=90.,
                 mode='deterministic', start_mode='ocd_settle', updates=25),
            dict(cycle=2, transitions=100, stop_reason='lap_complete', seconds=60.,
                 mode='explore', start_mode='teleport'),
            dict(cycle=3, transitions=50, stop_reason='wheel_off_track', seconds=10.,
                 mode='deterministic', progress_m=300),
            dict(cycle=4, transitions=0, stop_reason='teleport_timeout', seconds=0.)]
    result = dashboard.summarize(rows)
    assert result['finish_rate'] == pytest.approx(200/3)
    assert result['eval_finish_rate'] == 50
    assert result['best_lap_s'] == result['last_lap_s'] == 90
    assert result['timed_completions'] == result['teleport_completions'] == 1
    assert result['attempts'] == 4 and result['drives'] == 3
    assert result['actor_updates'] == 25 and result['transitions'] == 250


def test_unavailable_lap_duration_and_empty_run():
    assert dashboard.summarize([])['finish_rate'] is None
    result = dashboard.summarize([dict(cycle=1, transitions=1, stop_reason='lap_complete',
                                      seconds=None)])
    assert result['finish_rate'] == 100
    assert result['best_lap_s'] is None


def test_recent_window_and_partial_progress_line(tmp_path):
    rows = [dict(cycle=i, transitions=1,
                 stop_reason='lap_complete' if i < 5 else 'wheel_off_track') for i in range(25)]
    path = tmp_path/'progress.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n{"cycle":', encoding='utf-8')
    result = dashboard.summarize(dashboard.rows_from(path))
    assert result['finish_rate'] == 20
    assert result['recent_finish_rate'] == 0


def test_duration_uses_recorded_simulation_dt_and_closed_track(tmp_path):
    import numpy as np
    learner, worker = tmp_path/'learner', tmp_path/'worker'
    learner.mkdir()
    worker.mkdir()
    np.savez(worker/'chunk-00000001.npz', dt=np.array([.04, .045, .04]))
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps({'rows': [[0, 0], [3, 0], [3, 4]]}))
    view = dashboard.Dashboard(learner, worker, reference, 'Test')
    assert view.length == 12
    assert view.duration(1) == pytest.approx(.125)
    assert view.duration(2) is None
