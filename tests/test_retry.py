"""Crash retries (EnvConfig.retry; Ibrahim, 19 Nov 2026; DECISIONS #31): after a terminal failure the next
training reset retries it from the previous corner exit with probability prob; the first warmup_s of the
policy's run-up are flagged info['warmup'] and never sampled; off = unchanged starts / samples."""
import dataclasses
import json
from types import SimpleNamespace
import numpy as np
import pytest
from rlsetup.driving.buffers import BonusNStepReplayBuffer
from rlsetup.driving.callbacks import RetryCallback
from rlsetup.driving.config import EnvConfig, RetryConfig, RewardConfig
from rlsetup.driving.train_config import (PRESETS, LrDecayConfig, config_json, train_config_from_dict)
import speed_truth as ST
from sb3_helpers import TinyEnv, OBS

ON = RetryConfig(enabled=True, prob=1., warmup_s=3.)


def test_config_validation_omission_and_round_trip():
    for bad in (dict(prob=1.5), dict(warmup_s=-1.), dict(speed_fraction=(.8, .5)), dict(fallback_lead_m=0.),
                dict(start_step=-1)):
        with pytest.raises(ValueError):
            RetryConfig(enabled=True, **bad)
    assert 'retry' not in EnvConfig().to_dict()
    c = PRESETS['bcn_f1_r4_retry']()
    j = json.loads(config_json(c))
    assert j['env']['retry'] == {'enabled': True, 'prob': .75, 'warmup_s': 3., 'speed_fraction': [.5, .8],
                                 'fallback_lead_m': 400., 'start_step': 500_000, 'learn_before_m': 150.}
    assert train_config_from_dict(j) == c
    for name, make in PRESETS.items():
        if name not in ('bcn_f1_r4_retry', 'bcn_f1_r5', 'bcn_f1_r6', 'bcn_f1_r6_smoke'):   # R5, R6 inherit the retries
            assert '"retry"' not in config_json(make()), name


def test_preset_is_bcn_f1_r4_plus_retries_from_500k_and_the_3m_to_6p2m_decay():
    a, c = PRESETS['bcn_f1_r4'](), PRESETS['bcn_f1_r4_retry']()
    assert c.env.retry == RetryConfig(enabled=True, prob=.75, warmup_s=3., start_step=500_000, learn_before_m=150.)
    assert c.lr_decay == LrDecayConfig(enabled=True, start_step=3_000_000, end_step=6_200_000, floor=.01)
    assert dataclasses.replace(c, preset=a.preset, env=dataclasses.replace(c.env, retry=RetryConfig()),
                               lr_decay=LrDecayConfig()) == a


@pytest.fixture
def retry_env(make_env, track, qss):
    table = ST.table_for(track, qss)
    cfg = EnvConfig(max_episode_s=20., retry=ON, reward=RewardConfig(mode='time_pbrs'))
    return make_env(cfg, corners=table), table


def test_handover_is_the_previous_corner_exit(retry_env):
    env, _ = retry_env                       # exits 4700, 450, 760, 1200, 30
    assert env.retry_handover(1100.) == 760.     # inside U4 [1000, 1200]: the exit before it
    assert env.retry_handover(1250.) == 1200.    # on the straight after U4
    assert env.retry_handover(450.5) == 30.      # an exit less than 1 m behind does not count
    assert env.retry_handover(100.) == 30.


def test_retry_reset_and_warmup_flags(retry_env):
    env, table = retry_env
    _, info = env.reset(seed=3)
    assert info['start']['retry'] is False                  # no failure yet
    env._last_fail_s = 1100.
    _, info = env.reset(seed=3)
    assert info['start']['retry'] is False                  # retries not switched on: consumed, no retry
    env.set_retry(True)
    env._last_fail_s = 1100.
    _, info = env.reset(seed=3)
    st = info['start']
    assert st['retry'] is True and st['retry_target'] == 1100. and st['retry_handover_s'] == 760.
    assert abs((env.s - 760. + table.length / 2) % table.length - table.length / 2) < 15.   # handed over near 760 m
    flags, times = [], []
    for _ in range(200):
        a = env.settle_driver.act(env.state, env.s_drive, .8 * float(table.speed(env.s + 15.)))
        t0 = env.time_s
        *_, te, tr, i = env.step(np.asarray(a, np.float32))
        flags.append(i['warmup'])
        times.append(t0)
        if te or tr:
            break
    flags, times = np.array(flags), np.array(times)
    assert flags.any() and not flags.all()
    assert np.array_equal(flags, times < 3.)                 # exactly the decisions started in the first 3 s


def test_failure_is_recorded_and_retried(retry_env):
    env, _ = retry_env
    env.set_retry(True)
    env.reset(seed=5)
    for _ in range(500):                                    # full throttle, steer hard left: leaves the track
        *_, te, tr, i = env.step(np.array([1., 1., -1.], np.float32))
        if te or tr:
            break
    assert te and i['termination'] == 'off_track'
    target = env._last_fail_s
    assert target is not None and abs(target - env.s) < 1e-9
    _, info = env.reset(seed=6)
    assert info['start']['retry'] and info['start']['retry_target'] == target and env._last_fail_s is None


def test_off_leaves_the_start_stream_unchanged(make_env, track, qss):
    table = ST.table_for(track, qss)
    base = EnvConfig(max_episode_s=20., reward=RewardConfig(mode='time_pbrs'))
    off, on = make_env(base, corners=table), make_env(dataclasses.replace(base, retry=ON), corners=table)
    for seed in (1, 2, 3):
        _, a = off.reset(seed=seed)
        _, b = on.reset(seed=seed)                           # enabled, but never switched on / no failure
        assert (a['start']['s0'], a['start']['v0']) == (b['start']['s0'], b['start']['v0'])
        assert 'retry' not in a['start'] and b['start']['retry'] is False


def _buffer(n=64, n_envs=2):
    import gymnasium as gym
    rb = BonusNStepReplayBuffer(n, gym.spaces.Box(-10., 10., (OBS,), np.float32), TinyEnv.action_space,
                                n_envs=n_envs, n_steps=3, gamma=.99)
    rng = np.random.default_rng(0)
    for k in range(rb.buffer_size):           # SB3 stores buffer_size // n_envs rows
        rb.add(rng.uniform(-1, 1, (n_envs, OBS)).astype(np.float32), rng.uniform(-1, 1, (n_envs, OBS)).astype(np.float32),
               rng.uniform(-1, 1, (n_envs, 3)).astype(np.float32), np.full(n_envs, float(k), np.float32),
               np.zeros(n_envs, bool), [{'warmup': k % 4 == 0 and e == 0} for e in range(n_envs)])
    return rb


def test_buffer_never_samples_warmup_cells_and_is_bit_identical_without_them():
    rb = _buffer()
    assert rb.full and rb.warmup.sum() == rb.buffer_size // 4 and rb.warmup[0, 0] and not rb.warmup[0, 1]
    for seed in range(20):
        np.random.seed(seed)
        b = np.random.randint(0, rb.pos if not rb.full else rb.buffer_size, size=256)
        e = np.random.randint(0, rb.n_envs, size=256)
        b2, e2 = rb._drop_warmup(b, e)
        assert not rb.warmup[b2, e2].any() and np.array_equal(b2[~rb.warmup[b, e]], b[~rb.warmup[b, e]])
    clean = _buffer()
    clean.warmup[:] = False
    np.random.seed(7)
    s1 = clean.sample(128)
    np.random.seed(7)
    ref = _buffer()
    ref.warmup = None                                      # unmasked: the pre-retry path
    s2 = ref.sample(128)
    for x, y in zip(s1, s2):
        assert x is None and y is None or np.array_equal(x.cpu().numpy(), y.cpu().numpy())


def test_callback_switches_retries_on_and_counts():
    calls = []
    venv = SimpleNamespace(num_envs=2, env_method=lambda *a: calls.append(a))
    model = SimpleNamespace(replay_buffer=_buffer(), get_env=lambda: venv, logger=None, num_timesteps=0)
    cb = RetryCallback()
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert calls == [('set_retry', True)]
    for infos, dones in (([{'warmup': False}, {'warmup': False}], [True, False]),
                         ([{'warmup': True}, {'warmup': False}], [False, True]),    # env 0 starts a retry
                         ([{'warmup': True}, {'warmup': False}], [False, False])):  # env 1 a normal episode
        cb.update_locals({'infos': infos, 'dones': np.array(dones)})
        cb.on_step()
    out = cb.eval_extra(None, 0)['retry']
    assert out == {'episodes': 2, 'retry_episodes': 1, 'retry_share': .5, 'warmup_share': pytest.approx(2 / 6)}
    assert cb.episodes == 0
    bad = SimpleNamespace(replay_buffer=object(), get_env=lambda: venv, logger=None, num_timesteps=0)
    cb2 = RetryCallback()
    cb2.init_callback(bad)
    with pytest.raises(TypeError):
        cb2.on_training_start({}, {})


def test_callback_waits_for_start_step():
    calls = []
    venv = SimpleNamespace(num_envs=2, env_method=lambda *a: calls.append(a))
    model = SimpleNamespace(replay_buffer=_buffer(), get_env=lambda: venv, logger=None, num_timesteps=0)
    cb = RetryCallback(start_step=100)
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert calls == [] and not cb.on
    infos = {'infos': [{'warmup': False}] * 2, 'dones': np.zeros(2, bool)}
    model.num_timesteps = 99
    cb.update_locals(infos)
    cb.on_step()
    assert calls == []
    model.num_timesteps = 100
    cb.on_step()
    cb.on_step()
    assert calls == [('set_retry', True)] and cb.on            # once, at the first step >= start_step


@pytest.fixture
def dist_env(make_env, track, qss):
    table = ST.table_for(track, qss)
    cfg = EnvConfig(max_episode_s=20., retry=RetryConfig(enabled=True, prob=1., learn_before_m=150.),
                    reward=RewardConfig(mode='time_pbrs'))
    return make_env(cfg, corners=table), table


def _drive(env, table, n=1500):
    flags, dists = [], []
    for _ in range(n):
        a = env.settle_driver.act(env.state, env.s_drive, float(table.speed(env.s + 15.)))
        d0 = env._dist
        *_, te, tr, i = env.step(np.asarray(a, np.float32))
        flags.append(i['warmup'])
        dists.append(d0)
        if te or tr:
            break
    return np.array(flags), np.array(dists)


def test_learning_switches_on_learn_before_m_before_the_braking_point(dist_env):
    env, table = dist_env
    env.set_retry(True)
    assert env.retry_brake_point(1100.) == 1000. and env.retry_brake_point(1250.) == 1250.
    env._last_fail_s = 1100.                         # in U4 [1000, 1200]: brake 1000, hand-over at 760
    env.reset(seed=3)
    want = ((1000. - env.s) % table.length) - 150.  # ~90 m of run-up
    assert 60. < want < 120. and env._warmup_m == pytest.approx(want)
    flags, dists = _drive(env, table)
    assert flags.any() and not flags.all() and np.array_equal(flags, dists < want)
    env._last_fail_s = 1250.                         # straight after U4: brake = target, hand-over 1200 (< 150 m)
    env.reset(seed=4)
    assert env._warmup_m == 0.
    flags, _ = _drive(env, table, 20)
    assert not flags.any()


def test_learn_before_m_may_be_set_on_resume():
    from rlsetup.driving.train import RESUME_OVERRIDABLE
    assert 'env.retry.learn_before_m' in RESUME_OVERRIDABLE
