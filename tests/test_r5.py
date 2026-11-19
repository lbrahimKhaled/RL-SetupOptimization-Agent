"""R5 (docs/specs/R5_apex_split.md): apex-split line penalty, one-sided exit-speed term, crash emphasis."""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rlsetup.driving.buffers import BonusNStepReplayBuffer
from rlsetup.driving.callbacks import CrashEmphasisCallback
from rlsetup.driving.config import CornerBonusConfig, EnvConfig, RewardConfig
from rlsetup.driving.corners import CornerTable
from rlsetup.driving.env import DrivingEnv, post_apex_weight
from rlsetup.driving.train_config import (BCN_LINE_K, BCN_R5_EXIT_K, PRESETS, CrashEmphasisConfig, config_json,
                                          train_config_from_dict)
import speed_truth as ST
from sb3_helpers import TinyEnv, OBS

E3 = CornerBonusConfig(enabled=True, mode='entry_exit')
SPLIT = dict(mode='time_pbrs', line_k=.07, line_free_m=.5, line_cap=.5, apex_split=True, line_k_pre=.14,
             line_free_pre_m=.2, exit_k=.2, exit_sigma=.3, apex_blend_m=20.)


def table(units, L=1000., v=50.):
    return CornerTable(length=L, s=np.arange(0., L, 1.), v=np.full(int(L), v),
                       units=[dict(name=f'U{i + 1}', start_m=a, apexes_m=list(p), exit_m=b)
                              for i, (a, p, b) in enumerate(units)], lap_time_s=20.)


def fake(units, s, speed=50., **reward):
    return SimpleNamespace(corners=table(units), s=s, state={'vx': speed, 'vy': 0.},
                           cfg=SimpleNamespace(reward=RewardConfig(**{**SPLIT, **reward})))


# --- blend weight and terms ------------------------------------------------------------------------

def test_post_apex_weight_single_multi_apex_and_wrapping_windows():
    t = table([(100., [200.], 350.), (400., [450., 550.], 700.), (900., [50.], 150.)])
    w = lambda s, k: post_apex_weight(t, s, k, 20.)
    assert (w(100., 0), w(190., 0), w(200., 0), w(205., 0), w(210., 0), w(350., 0)) == (0., 0., .5, .75, 1., 1.)
    assert (w(450., 1), w(540., 1), w(550., 1), w(560., 1)) == (0., 0., .5, 1.)   # the LAST apex splits
    assert (w(950., 2), w(30., 2), w(40., 2), w(50., 2), w(60., 2), w(150., 2)) == (0., 0., 0., .5, 1., 1.)


def test_line_penalty_is_strict_before_the_apex_and_loose_after_it():
    pen, u = DrivingEnv._line_penalty, [(100., [200.], 350.)]
    assert pen(fake(u, 150.), 1.2, 2.) == pytest.approx(.14 * (1.2 - .2) * 2.)      # before: k_pre, 0.2 m band
    assert pen(fake(u, 300.), 1.2, 2.) == pytest.approx(.07 * (1.2 - .5) * 2.)      # after: k, 0.5 m band
    assert pen(fake(u, 200.), 1.2, 2.) == pytest.approx(.5 * .14 * 1. * 2. + .5 * .07 * .7 * 2.)   # blend
    assert pen(fake(u, 150.), .4, 2.) == pytest.approx(.14 * .2 * 2.) and pen(fake(u, 300.), .4, 2.) == 0.
    assert pen(fake(u, 150.), 50., 2.) == pytest.approx(.5 * 2.)                     # same cap
    assert pen(fake(u, 400.), 5., 2.) == 0. and pen(fake(u, 150.), 5., -1.) == 0.    # outside / backwards
    off = fake(u, 150., apex_split=False, exit_k=0.)
    assert pen(off, 1.2, 2.) == pytest.approx(.07 * .7 * 2.)                         # off: uniform, as before


@pytest.mark.parametrize('speed,h', [(50., 1.), (60., 1.), (42.5, .75), (35., 0.), (30., 0.)])
def test_exit_term_is_one_sided_and_paid_per_metre(speed, h):
    x, u = DrivingEnv._exit_term, [(100., [200.], 350.)]
    assert x(fake(u, 300., speed), 3.) == pytest.approx(.2 * h * 3.)
    assert x(fake(u, 205., speed), 3.) == pytest.approx(.2 * .75 * h * 3.)          # blend weight
    assert x(fake(u, 150., speed), 3.) == 0. and x(fake(u, 400., speed), 3.) == 0.   # before the apex / outside
    assert x(fake(u, 300., speed), 0.) == 0. and x(fake(u, 300., speed), -1.) == 0.
    # per metre: the same 3 m at h = 1 pay the same whatever the time it took
    assert x(fake(u, 300., 50.), 3.) == x(fake(u, 300., 55.), 3.)


def test_config_validation_and_omission():
    with pytest.raises(ValueError, match='needs apex_split'):
        RewardConfig(mode='time_pbrs', line_k=.07, exit_k=.1)
    with pytest.raises(ValueError, match='line penalty on'):
        RewardConfig(mode='time_pbrs', apex_split=True)
    with pytest.raises(ValueError):
        RewardConfig(**{**SPLIT, 'exit_sigma': 0.})
    plain = EnvConfig(reward=RewardConfig(mode='time_pbrs', line_k=.07)).to_dict()['reward']
    assert not {'apex_split', 'line_k_pre', 'exit_k'} & set(plain)
    assert EnvConfig(reward=RewardConfig(**SPLIT)).to_dict()['reward']['exit_k'] == .2
    with pytest.raises(ValueError):
        CrashEmphasisConfig(enabled=True, factor=.5)


# --- environment: off is bit-identical, on adds exactly the reported terms --------------------------

def test_env_split_off_identical_and_exit_term_added_exactly(make_env, track, qss):
    t = ST.table_for(track, qss)
    base = EnvConfig(max_episode_s=20., settle_s=0., corner_bonus=E3,
                     reward=RewardConfig(mode='time_pbrs', speed_k=20., line_k=.2, line_free_m=.3))
    ref = make_env(base, corners=t)
    same = make_env(dataclasses.replace(base, reward=dataclasses.replace(
        base.reward, apex_split=True, line_k_pre=.2, line_free_pre_m=.3)), corners=t)   # split with equal halves
    on = make_env(dataclasses.replace(base, reward=dataclasses.replace(
        base.reward, apex_split=True, line_k_pre=.2, line_free_pre_m=.3, exit_k=.15)), corners=t)
    paid = 0.
    for name, seed, opts, pol in ST.scenarios():
        ref.reset(seed=seed, options=opts)
        same.reset(seed=seed, options=opts)
        on.reset(seed=seed, options=opts)
        ep = 0.
        for k in range(1000):
            a = np.asarray(pol(on, t, k), np.float32)
            o0, r0, te, tr, i0 = ref.step(a)
            o1, r1, *_ = same.step(a)
            o2, r2, te2, tr2, i2 = on.step(a)
            assert o1.tobytes() == o0.tobytes() == o2.tobytes() and (te2, tr2) == (te, tr)
            assert r1 == r0 and 'exit_speed_reward' not in i0
            assert i2['exit_speed_reward'] >= 0. and r2 == pytest.approx(r0 + i2['exit_speed_reward'], abs=1e-9)
            ep += i2['exit_speed_reward']
            if te or tr:
                break
        assert i2['episode_exit_speed_reward'] == pytest.approx(ep, abs=1e-9)
        paid += ep
    assert paid > 0.


# --- crash emphasis ---------------------------------------------------------------------------------

def _empty(n=64, n_envs=2, horizon=5):
    import gymnasium as gym
    rb = BonusNStepReplayBuffer(n, gym.spaces.Box(-10., 10., (OBS,), np.float32), TinyEnv.action_space,
                                n_envs=n_envs, n_steps=3, gamma=.99)
    rb.crash_horizon = horizon
    return rb


def _add(rb, k, done=(False, False), trunc=(False, False)):
    n = rb.n_envs
    rb.add(np.full((n, OBS), k, np.float32), np.full((n, OBS), k, np.float32), np.zeros((n, 3), np.float32),
           np.full(n, float(k), np.float32), np.asarray(done), [{'TimeLimit.truncated': t} for t in trunc])


def test_crash_marking_horizon_episode_boundary_truncation_and_overwrite():
    rb = _empty()                                    # 32 rows x 2 columns
    for k in range(25):
        _add(rb, k, done=(k in (3, 10, 13), k == 20), trunc=(False, k == 20))
    col0 = set(np.flatnonzero(rb.crash_near[:, 0]))
    assert col0 == {0, 1, 2, 3} | {6, 7, 8, 9, 10} | {11, 12, 13}   # horizon 5; stops at the episode end at 10
    assert not rb.crash_near[:, 1].any()                             # a time-limit truncation is not a crash
    for k in range(25, 32 + 8):                                      # wrap: rows 0..7 are overwritten
        _add(rb, k)
    assert set(np.flatnonzero(rb.crash_near[:, 0])) == {8, 9, 10, 11, 12, 13}


def test_sampling_gives_marked_cells_factor_times_the_weight():
    rb = _empty(n=2000, n_envs=2)
    for k in range(1000):
        _add(rb, k, done=(k % 100 == 99, False))
    marked = rb.crash_near
    assert rb.full and marked.mean() == pytest.approx(.025)   # 5 of every 100 rows, column 0 only
    rb.crash_factor = 3.
    np.random.seed(0)
    counts = np.zeros(marked.shape)
    for _ in range(200):
        b = np.random.randint(0, rb.buffer_size, size=1024)
        e = np.random.randint(0, rb.n_envs, size=1024)
        b, e = rb._emphasize_crashes(b, e)
        np.add.at(counts, (b, e), 1)
    ratio = counts[marked].mean() / counts[~marked].mean()
    assert ratio == pytest.approx(3., rel=.05)


def test_factor_one_or_no_crash_is_bit_identical():
    def batch(factor, marks):
        rb = _empty(n=200)
        for k in range(100):
            _add(rb, k, done=(k == 50, False))
        rb.crash_factor = factor
        if not marks:
            rb.crash_near[:] = False
        np.random.seed(3)
        return rb.sample(64)
    plain = batch(1., True)
    for other in (batch(1., False), batch(3., False)):
        for x, y in zip(plain, other):
            assert x is None and y is None or np.array_equal(x.cpu().numpy(), y.cpu().numpy())
    on = batch(3., True)
    assert not np.array_equal(on.observations.cpu().numpy(), plain.observations.cpu().numpy())


def test_callback_sets_horizon_and_waits_for_start_step():
    rb = _empty()
    venv = SimpleNamespace(num_envs=2, get_attr=lambda name: [.04])
    model = SimpleNamespace(replay_buffer=rb, get_env=lambda: venv, logger=None, num_timesteps=0)
    cb = CrashEmphasisCallback(CrashEmphasisConfig(enabled=True, factor=3., horizon_s=2., start_step=500))
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert rb.crash_horizon == 50 and rb.crash_factor == 1.
    model.num_timesteps = 500
    cb.on_step()
    assert rb.crash_factor == 3.
    assert cb.eval_extra(model, 0)['crash_emphasis']['factor'] == 3.


# --- preset ---------------------------------------------------------------------------------------

def test_preset_is_bcn_f1_r4_retry_with_the_r5_reward_and_crash_emphasis():
    src, c = PRESETS['bcn_f1_r4_retry'](), PRESETS['bcn_f1_r5']()
    rw = c.env.reward
    assert (rw.apex_split, rw.line_k_pre, rw.line_free_pre_m, rw.line_k, rw.line_free_m, rw.line_cap) == \
        (True, 2 * BCN_LINE_K, .2, BCN_LINE_K, .5, .5)
    assert (rw.exit_k, rw.exit_sigma, rw.apex_blend_m) == (BCN_R5_EXIT_K, .3, 20.) and 150 * BCN_R5_EXIT_K == pytest.approx(20.)
    assert (c.env.corner_bonus.enabled, c.env.corner_bonus.bonus_in, c.env.corner_bonus.bonus_out) == (True, 0., 0.)
    assert c.crash_emphasis == CrashEmphasisConfig(True, 3., 2., 500_000)
    back = dataclasses.replace(c, preset=src.preset, crash_emphasis=CrashEmphasisConfig(),
                               env=dataclasses.replace(c.env, corner_bonus=src.env.corner_bonus,
                                                       reward=dataclasses.replace(
                                                           rw, apex_split=False, line_k_pre=.141, line_free_pre_m=.2,
                                                           exit_k=0., exit_sigma=.3, apex_blend_m=20.)))
    assert back == src
    assert train_config_from_dict(json.loads(config_json(c))) == c
    assert 'crash_emphasis' not in json.loads(config_json(src))


def test_crash_draw_comes_after_the_warmup_redraw_and_skips_runup_cells():
    import torch  # noqa: F401  (sample() returns tensors)
    rb = _empty(n=2000, n_envs=2)
    for k in range(1000):
        n = rb.n_envs
        rb.add(np.full((n, OBS), k, np.float32), np.full((n, OBS), k, np.float32), np.zeros((n, 3), np.float32),
               np.full(n, float(k), np.float32), np.asarray((k % 100 == 99, False)),
               [{'TimeLimit.truncated': False, 'warmup': k % 100 in (96, 97) or k % 100 < 20}, {'warmup': k % 10 == 0}])
    assert (rb.crash_near & rb.warmup).any()          # some marked rows are retry run-up
    rb.crash_factor = 3.
    np.random.seed(1)
    counts = np.zeros(rb.crash_near.shape)
    for _ in range(300):
        b = np.random.randint(0, rb.buffer_size, size=1024)
        e = np.random.randint(0, rb.n_envs, size=1024)
        b, e = rb._drop_warmup(b, e)
        b, e = rb._emphasize_crashes(b, e)
        np.add.at(counts, (b, e), 1)
    assert counts[rb.warmup].sum() == 0               # never a run-up row
    marked, other = rb.crash_near & ~rb.warmup, ~rb.crash_near & ~rb.warmup
    assert counts[marked].mean() / counts[other].mean() == pytest.approx(3., rel=.05)


def test_nothing_is_marked_while_the_horizon_is_zero():
    """The demo prefill (demos.load_into_buffer) runs before learn(): crash_horizon is still 0 there."""
    rb = _empty(horizon=0)
    for k in range(20):
        _add(rb, k, done=(k % 5 == 4, k % 7 == 6))
    assert not rb.crash_near.any()
