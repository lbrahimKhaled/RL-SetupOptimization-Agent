"""Slow checks (deselect with -m 'not slow'): settle statistics and a full demo flying lap."""
import numpy as np
import pytest
from rlsetup.driving.demo_driver import DemoPolicy

pytestmark = pytest.mark.slow


def test_settle_statistics_200_resets(env):
    """R9: >= 95 % of 200 random resets settle on the first attempt, and the summed tyre load
    at hand-over is within 5 % of its mean over the last 0.5 s of the settle."""
    first_try = within = 0
    worst = 0.
    n_window = round(.5 / env.decision_dt)
    for seed in range(200):
        _, info = env.reset(seed=seed, options={'settle_trace': True})
        first_try += info['start']['attempts'] == 1
        fz = np.array(info['settle_sum_fz_N'])
        dev = abs(fz[-1] / fz[-n_window:].mean() - 1)
        worst = max(worst, dev)
        within += dev <= .05
    print(f'first-attempt settles {first_try}/200, Fz within 5 %: {within}/200, worst {worst:.3%}')
    assert first_try >= 190
    assert within >= 190


def test_demo_flying_lap(make_env, env):
    """Demo driver at f = 0.5 completes run-in + 2 timed laps; clip fraction < 1 %."""
    env.reset(options={'evaluation': True})
    clipped0, values0 = env.obs_clipped_total, env.obs_values_total
    policy = DemoPolicy(env, .5)
    policy.reset()
    while True:
        _, _, term, trunc, info = env.step(policy())
        if term or trunc:
            break
    assert info['termination'] == 'lap_complete'
    lap1, lap2 = info['laps']
    assert len(info['line_crossings_s']) == 3
    assert abs(lap1['lap_time_s'] - lap2['lap_time_s']) < .5
    assert lap2['lap_time_s'] == pytest.approx(2 * env.qss.lap_time_s, rel=.02)   # f = 0.5
    assert lap2['time'] == pytest.approx(lap2['lap_time_s'])
    assert lap2['progress'] == pytest.approx(env.track.length)
    clip_fraction = (env.obs_clipped_total - clipped0) / (env.obs_values_total - values0)
    assert clip_fraction < .01
