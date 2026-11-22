"""R6 top-end throttle remap (docs/specs/R6_throttle_remap.md §4): bit identity with the remap off, the map, the exact
buffer relabel (pure, obs slots, real transitions, synthetic bonus buffer), config / warm start / AC refusal,
lifetime schedules, evaluation metrics.

needs_f1: ocd_first_lap branch sf70h-f1 on PYTHONPATH (bcn_f1_r5 env: car sf70h_f1, Catalunya)."""
import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from ocd_lap import cars
from stable_baselines3.common.vec_env import DummyVecEnv

from rlsetup.driving import bc, evaluate as E
from rlsetup.driving.buffers import BonusNStepReplayBuffer
from rlsetup.driving.callbacks import CrashEmphasisCallback, LrDecayCallback, RetryCallback
from rlsetup.driving.config import EnvConfig, RewardConfig, check_throttle_remap
from rlsetup.driving.demo_driver import (action_to_inputs, inputs_to_action, pedal_to_action, throttle_action,
                                         throttle_pedal)
from rlsetup.driving.env import throttle_obs_indices
from rlsetup.driving.relabel import relabel_throttle, relabel_throttle_values, throttle_relabel_error
from rlsetup.driving.train import offline_lr_scale, schedule_offset, throttle_gate_passed, throttle_gate_value
from rlsetup.driving.train_config import (PRESETS, R6_THROTTLE_REMAP, config_json, train_config_from_dict)
from sb3_helpers import TinyEnv, OBS, tiny_cfg

HAS_F1 = 'sf70h_f1' in cars.CARS
needs_f1 = pytest.mark.skipif(not HAS_F1, reason='sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path')
R6 = R6_THROTTLE_REMAP
PBRS = RewardConfig(mode='time_pbrs')
# r6_golden.stream(bcn_r5_env()) on the pre-R6 code (HEAD 7bf3cd0 via git archive, ocd_first_lap-f1)
GOLDEN_BCN_R5 = 'd0ffef43da9006eca554ebd14cd1c5c6de4862864b54cc63d1b09e684b97c4c1'


def grid():
    rng = np.random.default_rng(6)
    edges = np.array([-1., -.8, -.5, -.49, 0., .3, .7, .85, .99, 1.])
    f32 = np.r_[edges.astype(np.float32), np.nextafter(np.float32(.7), np.float32(1.)),
                rng.uniform(-1, 1, 2000).astype(np.float32)]
    return np.r_[edges, f32.astype(np.float64)]


# --- pre-R6 code, verbatim (HEAD 7bf3cd0) --------------------------------------------------------------------
def head_action_to_inputs(a, pedal_deadzone, steer_max_rad=.6):
    if pedal_deadzone:
        pedal = lambda x: float(min(1., max(0., (x + .5) / 1.5)))
        return (float(a[0]) * steer_max_rad, pedal(a[1]), pedal(a[2]))
    return (float(a[0]) * steer_max_rad, (a[1] + 1) / 2, (a[2] + 1) / 2)


def head_pedal_to_action(pedal, pedal_deadzone, released_action=-.8):
    if pedal_deadzone:
        return -.5 + 1.5 * pedal if pedal > 0. else released_action
    return 2. * pedal - 1.


def head_pedals(actions, pedal_deadzone):
    a = np.asarray(actions, dtype=float).reshape(-1, 3)
    if pedal_deadzone:
        return np.clip((a[:, 1] + .5) / 1.5, 0., 1.), np.clip((a[:, 2] + .5) / 1.5, 0., 1.)
    return (a[:, 1] + 1.) / 2., (a[:, 2] + 1.) / 2.


# --- 1. off = bit-exact -------------------------------------------------------------------------------------
@pytest.mark.parametrize('deadzone', [True, False])
def test_remap_off_equals_head_bitwise(deadzone):
    g = grid()
    for x in g:
        a = np.array([x / 2, x, -x])
        i = action_to_inputs(a, deadzone)
        assert (i.steering_rad, i.throttle, i.brake) == head_action_to_inputs(a, deadzone)
        p = min(1., max(0., (x + .5) / 2.))
        assert pedal_to_action(p, deadzone) == head_pedal_to_action(p, deadzone)
        got = inputs_to_action(.1, p, p / 2, deadzone)
        ref = np.array([.1 / .6, head_pedal_to_action(p, deadzone), head_pedal_to_action(p / 2, deadzone)], np.float32)
        assert got.tobytes() == ref.tobytes()
    acts = np.stack([g / 3, g, g[::-1]], 1)
    for x, y in zip(E._pedals(acts, deadzone), head_pedals(acts, deadzone)):
        assert x.tobytes() == y.tobytes()
    head_bc = np.clip((g + .5) / 1.5, 0., 1.) if deadzone else (g + 1.) / 2.
    assert bc.pedal(g, deadzone).tobytes() == head_bc.tobytes()


def test_presets_and_env_dicts_without_the_remap_are_unchanged():
    for name, make in PRESETS.items():
        c = make()
        assert ('throttle_remap' in c.to_dict()['env']) == name.startswith('bcn_f1_r6'), name
        warm_r6 = name in ('bcn_f1_r6', 'bcn_f1_r6_smoke')                          # the scratch presets: no warm start
        assert ('lifetime_schedules' in c.to_dict()) == warm_r6, name
        assert ('throttle_relabel' in c.to_dict()['warm']) == warm_r6, name
    assert 'throttle_remap' not in EnvConfig().to_dict()
    assert EnvConfig().to_dict() == dataclasses.replace(EnvConfig(), throttle_remap=None).to_dict()


def test_demo_dataset_key_env_part_unchanged_and_remap_enters_it(qss):
    from rlsetup.driving import demos as D
    from rlsetup.driving.train_config import DemoDataConfig
    prov = {'ocd_lap_code_version': 'x'}
    k0, d0 = D.dataset_key(EnvConfig(), DemoDataConfig(), qss, prov)
    assert 'throttle_remap' not in d0['env'] and d0['env'] == EnvConfig().to_dict()
    k1, d1 = D.dataset_key(EnvConfig(reward=PBRS, throttle_remap=R6), DemoDataConfig(), qss, prov)
    assert d1['env']['throttle_remap'] == R6 and k1 != k0


@needs_f1
def test_bcn_f1_r5_env_stream_equals_head():
    from r6_golden import bcn_r5_env, stream
    assert stream(bcn_r5_env()) == GOLDEN_BCN_R5
    assert stream(bcn_r5_env(throttle_remap=R6)) != GOLDEN_BCN_R5          # the stream does exercise the remap


# --- 2. the map ---------------------------------------------------------------------------------------------
def test_map_shape_continuity_saturation_and_monotone():
    k, s = R6
    pk = (k + .5) / 1.5
    assert pk == pytest.approx(.8)
    assert throttle_pedal(k, R6) == pytest.approx(pk, abs=1e-15)
    assert throttle_pedal(np.nextafter(k, 2.), R6) == pytest.approx(pk, abs=1e-12)
    assert throttle_pedal(s, R6) == 1. == throttle_pedal(1., R6) == throttle_pedal(.9, R6)
    x = np.linspace(-1., 1., 200_001)
    p = throttle_pedal(x, R6)
    assert np.all(np.diff(p) >= 0.) and p[0] == 0. and p[-1] == 1.
    lo = x <= k
    assert p[lo].tobytes() == np.clip((x[lo] + .5) / 1.5, 0., 1.).tobytes()          # bitwise R3 below the knee
    assert (throttle_pedal(.85, R6) - throttle_pedal(.7, R6)) / .15 == pytest.approx(4 / 3)   # 2 x the lower slope
    for xi in x[::997]:                                                               # scalar == vector path
        assert throttle_pedal(float(xi), R6) == p[np.searchsorted(x, xi)]


def test_round_trips_and_released_pedal():
    p = np.linspace(0., 1., 100_001)[1:]
    a = throttle_action(p, R6)
    assert np.max(np.abs(throttle_pedal(a, R6) - p)) <= 1e-12
    a32 = a.astype(np.float32).astype(np.float64)
    assert np.max(np.abs(throttle_pedal(a32, R6) - p)) <= 2e-7
    assert throttle_action(1., R6) == pytest.approx(.85, abs=1e-15)
    x = np.linspace(-.5, .85, 50_001)[1:]
    assert np.max(np.abs(throttle_action(throttle_pedal(x, R6), R6) - x)) <= 1e-12
    assert throttle_action(0., R6) == -.8 and throttle_action(0., R6, released=-.9) == -.9
    assert pedal_to_action(0., True, -.8, R6) == -.8
    assert inputs_to_action(0., .9, .3, True, throttle_remap=R6)[1] == np.float32(throttle_action(.9, R6))


def test_brake_and_steering_untouched_by_the_remap():
    for x in grid():
        a = np.array([x / 2, x, x])
        on, off = action_to_inputs(a, True, .3, R6), action_to_inputs(a, True, .3)
        assert (on.steering_rad, on.brake) == (off.steering_rad, off.brake)
        assert on.throttle == throttle_pedal(x, R6)
        on_a, off_a = inputs_to_action(.1, .5, .4, True, throttle_remap=R6), inputs_to_action(.1, .5, .4, True)
        assert on_a[[0, 2]].tobytes() == off_a[[0, 2]].tobytes()
    acts = np.stack([grid()] * 3, 1)
    assert E._pedals(acts, True, R6)[1].tobytes() == E._pedals(acts, True)[1].tobytes()


@pytest.mark.parametrize('bad', [(.7, .7), (.9, .8), (-.5, .8), (.7, 1.01), (.7,), (float('nan'), .9), 'x'])
def test_invalid_remaps_raise(bad):
    with pytest.raises(ValueError):
        check_throttle_remap(bad)
    with pytest.raises(ValueError):
        EnvConfig(reward=PBRS, throttle_remap=bad)


def test_remap_needs_deadzone_and_time_pbrs():
    with pytest.raises(ValueError, match='deadzone'):
        EnvConfig(reward=PBRS, throttle_remap=R6, pedal_deadzone=False)
    with pytest.raises(ValueError, match='time_pbrs'):
        EnvConfig(throttle_remap=R6)                                   # default ref_tracking
    with pytest.raises(ValueError):
        action_to_inputs(np.zeros(3), False, throttle_remap=R6)
    with pytest.raises(ValueError):
        E._pedals(np.zeros((1, 3)), False, R6)
    with pytest.raises(ValueError):
        bc.pedal(np.zeros(3), False, R6)
    EnvConfig(reward=PBRS, throttle_remap=(.7, 1.))                    # s = 1 allowed


# --- 3. relabel ---------------------------------------------------------------------------------------------
def test_relabel_values_exact_against_the_old_pedal():
    rng = np.random.default_rng(0)
    v = np.r_[rng.uniform(-1, 1, 1_000_000), [.7, .985, 1., -1., -.5, .85]].astype(np.float32)
    v = np.r_[v, np.nextafter(np.float32(.7), np.float32(1.))]
    new, changed = relabel_throttle_values(v, None, R6)
    assert new.dtype == np.float32 and np.array_equal(changed, v.astype(np.float64) > .7)
    assert new[~changed].tobytes() == v[~changed].tobytes()                          # identity below the knee
    old_p = np.clip((v.astype(np.float64) + .5) / 1.5, 0., 1.)
    assert np.max(np.abs(throttle_pedal(new.astype(np.float64), R6) - old_p)) <= 2e-7
    assert throttle_relabel_error(v, new, None, R6) <= 2e-7
    hi = v.astype(np.float64) > .7
    assert np.allclose(new[hi], .7 + .75 * (old_p[hi] - .8), atol=1e-7)
    assert new[v == 1.][0] == np.float32(.85) and new[v == .7][0] == np.float32(.7)
    same, ch = relabel_throttle_values(v, R6, R6)
    assert same.tobytes() == v.tobytes() and not ch.any()
    back, _ = relabel_throttle_values(new, R6, None)                                 # and back to R3
    assert np.max(np.abs(back.astype(np.float64) - v.astype(np.float64))) <= 3e-7


@pytest.mark.parametrize('gear', [False, True])
def test_throttle_obs_indices_against_a_real_env(make_env, gear):
    cfg = EnvConfig(obs_gear=gear, reward=PBRS, max_episode_s=4.)
    env = make_env(cfg)
    idx = throttle_obs_indices(cfg)
    assert idx == ([133, 145, 157] if not gear else [133, 146, 158])
    env.reset(seed=3)
    seen = []
    for t in (.31, .52, .73):
        obs, *_ = env.step(np.array([0., t, -1.], np.float32))
        seen.append(t)
        want = [np.float32(x) for x in seen[::-1]][:len(idx)]
        assert [obs[i] for i in idx[:len(want)]] == want
    assert len(obs) == env.observation_space.shape[0] and idx[-1] < len(obs)
    vals = {np.float32(x) for x in seen}                 # no other action / history slot holds these values
    assert not any(obs[j] in vals for j in range(132, len(obs)) if j not in idx)


def _roll(env, seed, opts, actions):
    """Reset with (seed, opts), then step actions[k] (an array, or a callable(env, k) -> action, recorded)."""
    obs, _ = env.reset(seed=seed, options=opts)
    o, r, d, taken = [obs], [], [], []
    for k in range(100):
        a = actions(env, k) if callable(actions) else actions[k]
        obs, rew, te, tr, _ = env.step(a)
        o.append(obs)
        r.append(rew)
        d.append(te or tr)
        taken.append(a)
        if te or tr:
            break
    return np.asarray(o), np.asarray(r), d, np.asarray(taken, np.float32)


def _driver(seed, f=.6):
    """Pure-pursuit steering on xi_ref (the env's settle driver geometry); below f v_qss a random throttle above
    the knee (0.72-1.0), above it a part throttle below the knee, brake (pedal 0.33) above (f + 0.1) v_qss."""
    rng = np.random.default_rng(seed)
    hi, lo = rng.uniform(.72, 1., 100), rng.uniform(-.6, .6, 100)

    def act(env, k):
        e = env.unwrapped
        steer = e.settle_driver.steering(e.state, e.s_drive) / e.steer_max_rad
        v, target = math.hypot(e.state['vx'], e.state['vy']), f * float(e.qss.speed(e.s + 30.))
        thr, brk = (hi[k], -1.) if v < target else (lo[k], -1. if v < target / f * (f + .1) else 0.)
        return np.array([steer, thr, brk], np.float32)
    return act


def _transition_check(make, seed, opts, f=.6):
    """Roll 100 decisions under the R3 map, relabel (obs, next_obs, actions), replay the relabelled actions under
    R6 from the same reset: everything matches the relabelled records to 1e-5."""
    old_env, new_env = make(None), make(R6)
    o, r, d, acts = _roll(old_env, seed, opts, _driver(seed, f))
    n = len(r)
    assert n == 100 and not any(d)
    cfg = old_env.cfg
    idx = throttle_obs_indices(cfg)
    obs, nxt, a = o[:-1].copy(), o[1:].copy(), acts.copy()
    obs[:, idx] = relabel_throttle_values(obs[:, idx], None, R6)[0]
    nxt[:, idx] = relabel_throttle_values(nxt[:, idx], None, R6)[0]
    a[:, 1] = relabel_throttle_values(a[:, 1], None, R6)[0]
    assert (a[:, 1] != acts[:, 1]).mean() > .2                                       # the relabel did change many
    o2, r2, d2, _ = _roll(new_env, seed, opts, a)
    assert len(r2) == n and not any(d2)
    assert np.max(np.abs(o2[:-1] - obs)) <= 1e-5 and np.max(np.abs(o2[1:] - nxt)) <= 1e-5
    assert np.max(np.abs(r2 - r)) <= 1e-5
    slots = o2[:, idx]
    assert np.max(np.abs(slots[:-1] - obs[:, idx])) <= 1e-6                          # settle actions: rounding only
    for t in range(1, n + 1):                                                        # policy actions: bitwise
        assert slots[t, 0] == a[t - 1, 1]


def test_relabelled_transitions_replay_under_the_new_map_shanghai(make_env):
    base = EnvConfig(reward=PBRS, max_episode_s=8.)
    make = lambda remap: make_env(dataclasses.replace(base, throttle_remap=remap))
    _transition_check(make, 11, {'start_s': 4500.})                                 # main straight


@needs_f1
def test_relabelled_transitions_replay_under_the_new_map_bcn_f1_r5():
    from r6_golden import bcn_r5_env
    env_cfg = PRESETS['bcn_f1_r5']().env
    built = {}

    def make(remap):
        if remap not in built:
            built[remap] = bcn_r5_env(dataclasses.replace(env_cfg, throttle_remap=remap))
        return built[remap]
    # 60 m/s on the main straight into T1: below ~50 m/s the sf70h_f1 rollout is chaotic in the R3 map itself (a 1-ulp
    # throttle change moves the obs by ~1 within 100 decisions, rear slip), so no map could be checked to 1e-5 there;
    # here a 1-ulp change moves it by ~2e-7 and 90 of the 100 throttle actions are above the knee
    _transition_check(make, 5, {'start_s': 0., 'start_speed': 60.}, f=.85)                                   # main straight -> T1


def _bonus_buffer(n=300, n_envs=2):
    import gymnasium as gym
    rb = BonusNStepReplayBuffer(n * n_envs, gym.spaces.Box(-10., 10., (OBS,), np.float32), TinyEnv.action_space,
                                n_envs=n_envs, n_steps=3, gamma=.99)
    rb.crash_horizon = 4
    rng = np.random.default_rng(1)
    for k in range(n - 20):
        rb.add(rng.uniform(-1, 1, (n_envs, OBS)).astype(np.float32), rng.uniform(-1, 1, (n_envs, OBS)).astype(np.float32),
               rng.uniform(-1, 1, (n_envs, 3)).astype(np.float32), rng.normal(size=n_envs).astype(np.float32),
               rng.uniform(size=n_envs) < .05,
               [{'corner_bonus': float(rng.uniform()), 'warmup': bool(rng.uniform() < .2)} for _ in range(n_envs)])
    return rb


def test_synthetic_bonus_buffer_changes_only_throttle_cells():
    rb = _bonus_buffer()
    before = {k: getattr(rb, k).copy() for k in ('observations', 'next_observations', 'actions', 'rewards', 'dones',
                                                 'timeouts', 'bonuses', 'labelled', 'warmup', 'crash_near')}
    assert before['crash_near'].any() and before['warmup'].any() and before['bonuses'].any()
    idx = throttle_obs_indices(EnvConfig())
    st = relabel_throttle(rb, None, R6, idx)
    assert not st['aborted'] and st['max_abs_dpedal'] <= 2e-7 and st['rows'] == rb.pos
    assert st['action_cells'] > 0 and st['obs_cells'] > 0 and st['next_obs_cells'] > 0
    for k, v in before.items():
        now = getattr(rb, k)
        if k == 'actions':
            assert np.array_equal(now[..., [0, 2]], v[..., [0, 2]])
            assert st['action_cells'] == int((v[:rb.pos, :, 1] > .7).sum()) == int((now[:rb.pos, :, 1] != v[:rb.pos, :, 1]).sum())
        elif k in ('observations', 'next_observations'):
            keep = [j for j in range(OBS) if j not in idx]
            assert np.array_equal(now[..., keep], v[..., keep])
            moved = now[..., idx] != v[..., idx]
            assert np.array_equal(moved[:rb.pos], v[:rb.pos][..., idx] > .7) and not moved[rb.pos:].any()
        else:
            assert np.array_equal(now, v), k


def test_relabel_aborts_and_leaves_the_buffer_untouched(monkeypatch):
    import rlsetup.driving.relabel as R
    rb = _bonus_buffer()
    a0 = rb.actions.copy()
    monkeypatch.setattr(R, 'THROTTLE_RELABEL_TOL', 1e-12)
    st = R.relabel_throttle(rb, None, R6, throttle_obs_indices(EnvConfig()))
    assert st['aborted'] and np.array_equal(rb.actions, a0)


# --- 4. config / warm start / AC ------------------------------------------------------------------------------
def test_bcn_f1_r6_preset_and_config_json():
    c = PRESETS['bcn_f1_r6']()
    j = json.loads(config_json(c))
    assert j['env']['throttle_remap'] == [.7, .85] and j['lifetime_schedules'] is True
    assert train_config_from_dict(j) == c
    r5 = PRESETS['bcn_f1_r5']()
    assert dataclasses.replace(c.env, throttle_remap=None) == r5.env                 # only the map changes
    assert (c.env.reward, c.env.corner_bonus, c.lr_decay, c.crash_emphasis, c.guard, c.eval) == \
           (r5.env.reward, r5.env.corner_bonus, r5.lr_decay, r5.crash_emphasis, r5.guard, r5.eval)
    assert dataclasses.replace(c.agent, init_mode='bc') == r5.agent and c.agent.init_mode == 'warm'
    w = c.warm
    # offline warm-up removed (Ibrahim, 21 Nov 2026): gate checked once before learn(), online phase at the schedule
    assert (w.throttle_relabel, w.relabel, w.speed_relabel, w.offline_critic_grad_steps, w.offline_critic_lr_scale,
            w.critic_warmup_steps, w.actor_lr, w.critic_check_tol, w.throttle_gate_tol) == \
           (True, False, False, 0, None, 20_000, 1e-4, .08, .01)
    assert c.agent.learning_rate * c.lr_decay.scale(7_000_000) == pytest.approx(3e-6)
    assert c.init_from.endswith('runs/mac/D2/bcn_f1_r5/seed0/checkpoints/step_7000000.zip')
    assert w.init_buffer.endswith('runs/mac/D2/bcn_f1_r5/seed0/replay_buffer.pkl')
    assert c.total_timesteps == 4_000_000 and not c.stop.plateau and c.strict_provenance
    s = PRESETS['bcn_f1_r6_smoke']()
    assert (s.total_timesteps, s.warm.offline_critic_grad_steps, s.warm.critic_warmup_steps, s.strict_provenance) == \
           (150_000, 0, 20_000, False)


def _src(tmp_path, remap=None):
    from test_r1 import _save_source
    c = tiny_cfg()
    src_cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, reward=PBRS, throttle_remap=remap))
    return _save_source(tmp_path, 'src', src_cfg, buffer=True)


def _r6_cfg(path, **warm):
    c = tiny_cfg()
    return dataclasses.replace(c, env=dataclasses.replace(c.env, reward=PBRS, throttle_remap=R6),
                               agent=dataclasses.replace(c.agent, init_mode='warm'),
                               warm=dataclasses.replace(c.warm, relabel=False, **warm), init_from=str(path))


def test_warm_start_refuses_a_new_map_without_relabel_and_passes_with_it(tmp_path):
    from rlsetup.driving.agent import build_model, warm_start, InitMismatch
    _, path = _src(tmp_path)
    off = _r6_cfg(path)
    with pytest.raises(InitMismatch, match='throttle_relabel'):
        warm_start(build_model(off, DummyVecEnv([TinyEnv])), path, off)
    on = _r6_cfg(path, throttle_relabel=True)
    warm_start(build_model(on, DummyVecEnv([TinyEnv])), path, on)
    other = dataclasses.replace(on, env=dataclasses.replace(on.env, max_episode_s=30.))
    with pytest.raises(InitMismatch, match='env config'):                             # other fields still checked
        warm_start(build_model(other, DummyVecEnv([TinyEnv])), path, other)


def test_preload_relabels_exactly_and_records_stats(tmp_path):
    from rlsetup.driving.agent import build_model, preload_buffer, default_init_buffer, InitMismatch
    src, path = _src(tmp_path)
    cfg = _r6_cfg(path, throttle_relabel=True, throttle_gate_batch=50)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    aux = {}
    info = preload_buffer(model, default_init_buffer(path), path, cfg, aux=aux)
    st = info['throttle_relabel']
    old, rb = src.replay_buffer, model.replay_buffer
    hi = old.actions[:old.pos, :, 1] > .7
    assert st['action_cells'] == int(hi.sum()) and st['src_remap'] is None and st['dst_remap'] == [.7, .85]
    assert st['obs_indices'] == [133, 145, 157] and st['max_abs_dpedal'] <= 2e-7 and not st['aborted']
    assert np.array_equal(rb.rewards, old.rewards) and np.array_equal(rb.actions[..., [0, 2]], old.actions[..., [0, 2]])
    r, c, o_old, a_old = aux['throttle_probe']
    assert len(r) == 50 and np.all(a_old[:, 1] > .7) and np.array_equal(o_old, old.observations[r, c])
    assert np.all(rb.actions[r, c, 1] < a_old[:, 1])                                  # relabelled down
    same = _r6_cfg(path)                                                            # no relabel flag
    same = dataclasses.replace(same, env=dataclasses.replace(same.env, throttle_remap=None))
    assert preload_buffer(build_model(same, DummyVecEnv([TinyEnv])), default_init_buffer(path), path,
                          same)['throttle_relabel'] is None
    none = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, throttle_remap=None))
    with pytest.raises(InitMismatch, match='throttle_remap'):
        preload_buffer(build_model(none, DummyVecEnv([TinyEnv])), default_init_buffer(path), path, none)


def test_ac_refuses_a_remapped_source_config():
    from rlsetup.assetto.actor import _config
    env = json.loads(json.dumps(PRESETS['bcn_f1_r5']().env.to_dict()))
    cfg = {'env': env, 'seed': 0, 'preset': 'bcn_f1_r5', 'agent': {'net_arch': [256, 256, 256]}}
    _config(cfg)                                                                    # R5: accepted, unchanged
    with pytest.raises(ValueError, match='throttle_remap'):
        _config(dict(cfg, env=dict(env, throttle_remap=[.7, .85])))
    _config(dict(cfg, env=dict(env, throttle_remap=None)))


# --- 5. lifetime schedules ------------------------------------------------------------------------------------
def test_lifetime_schedules_start_at_the_source_step():
    c = PRESETS['bcn_f1_r6']()
    status = {'lifetime_steps_offset': 7_000_000}
    off = schedule_offset(c, status)
    assert off == 7_000_000 and schedule_offset(dataclasses.replace(c, lifetime_schedules=False), status) == 0
    assert schedule_offset(c, {}) == 0
    over = dataclasses.replace(c, warm=dataclasses.replace(c.warm, offline_critic_lr_scale=.1))
    assert offline_lr_scale(over, off) == .1                                         # an offline override
    assert offline_lr_scale(c, off) == pytest.approx(.01)                            # the preset: none
    assert offline_lr_scale(PRESETS['bcn_f1_r5'](), 0) is None                       # pre-R6 runs: unscaled
    # lr decay
    model = SimpleNamespace(num_timesteps=0, lr_scale=None, get_env=lambda: None, logger=None)
    for offset, want in ((off, .01), (0, 1.)):
        cb = LrDecayCallback(c.lr_decay, offset)
        cb.init_callback(model)
        cb.on_training_start({}, {})
        assert model.lr_scale == pytest.approx(want)
    # retries
    calls = []
    venv = SimpleNamespace(num_envs=2, env_method=lambda *a: calls.append(a))
    rb = _bonus_buffer()
    m = SimpleNamespace(replay_buffer=rb, get_env=lambda: venv, logger=None, num_timesteps=0)
    cb = RetryCallback(c.env.retry.start_step, off)
    cb.init_callback(m)
    cb.on_training_start({}, {})
    assert cb.on and calls == [('set_retry', True)]
    cb0 = RetryCallback(c.env.retry.start_step, 0)
    cb0.init_callback(m)
    cb0.on_training_start({}, {})
    assert not cb0.on and len(calls) == 1
    # crash emphasis
    m.get_env = lambda: SimpleNamespace(num_envs=2, get_attr=lambda name: [.04])
    for offset, want in ((off, 3.), (0, 1.)):
        cb = CrashEmphasisCallback(c.crash_emphasis, offset)
        cb.init_callback(m)
        cb.on_training_start({}, {})
        assert rb.crash_factor == want


# --- 6. evaluation --------------------------------------------------------------------------------------------
def test_pedals_with_the_remap():
    a = np.array([[0., .7, -1.], [0., .775, -.5], [0., .85, .1], [0., 1., 1.]])
    thr, brk = E._pedals(a, True, R6)
    assert thr == pytest.approx([.8, .9, 1., 1.]) and brk == pytest.approx([0., 0., .4, 1.])
    assert E._pedals(a, True)[0] == pytest.approx([.8, .85, .9, 1.])


def _table(units, L=1000., v=50.):
    from rlsetup.driving.corners import CornerTable
    return CornerTable(length=L, s=np.arange(0., L, 1.), v=np.full(int(L), v),
                       units=[dict(name=f'U{i + 1}', start_m=a, apexes_m=[(a + b) / 2], exit_m=b)
                              for i, (a, b) in enumerate(units)], lap_time_s=20.)


def test_straight_times_full_throttle_99_and_sat_share():
    table = _table(((100., 200.), (500., 600.)), v=50.)                               # QSS 50 m/s everywhere
    s = np.r_[np.arange(0., 1000., 10.), np.arange(0., 1000., 10.)]                  # lap 1, lap 2; 10 m / decision
    T = len(s)
    lap = np.r_[np.ones(100, int), np.full(100, 2)]
    t = np.arange(1, T + 1) * .25                                                    # 40 m/s -> 0.25 s per 10 m
    acts = np.tile(np.array([0., .8, -1.], np.float32), (T, 1))
    acts[s >= 300., 1] = 1.
    tr = {k: np.zeros(T) for k in E.TRACE_KEYS}
    tr.update(s=s, lap_index=lap, speed=np.full(T, 40.), time_s=t, actions=acts, kappa=np.zeros((T, 4)),
              Fz=np.zeros((T, 4)))
    rep = E.LapReport(True, [25., 25.], [], 'lap_complete', None, T, 0., tr, None, None, [])
    from rlsetup.driving.train_config import EvalConfig
    _, out = E.lap_r4(rep, table, EvalConfig(), pedal_deadzone=True, entry_exit=False, throttle_remap=R6)
    s1, s2 = out['straights']['S1'], out['straights']['S2']                          # 200 -> 500, 600 -> 100 (wraps)
    # S1: decisions ending at 200..500 m, timed from the end of the one before (190 m): 310 m
    assert s1['time_s'] == pytest.approx(310 / 40) and s1['qss_time_s'] == pytest.approx(310 / 50)
    assert s1['time_lost_s'] == pytest.approx(310 / 40 - 310 / 50)
    # S2 in lap 2: 990 -> 100 m (lap start, from the last lap-1 decision) + 590 -> 990 m = 510 m
    assert s2['time_s'] == pytest.approx(510 / 40) and s2['qss_time_s'] == pytest.approx(510 / 50)
    assert out['straights_time_lost_s'] == pytest.approx(820 / 40 - 820 / 50)
    assert out['straights_time_s'] == pytest.approx(820 / 40)
    # pedals: a = 0.8 -> 0.9333 (R6), a = 1 -> 1.0; S1 decisions 200..500: 200-290 at 0.8 (10 of 31), rest 1.0
    assert s1['full_throttle_share_99'] == pytest.approx(21 / 31) and s1['full_throttle_share'] == pytest.approx(21 / 31)
    assert out['throttle_sat_action'] == .85 and out['throttle_action_share_above_sat'] == pytest.approx(.7)
    _, r3 = E.lap_r4(rep, table, EvalConfig(), pedal_deadzone=True, entry_exit=False)
    assert r3['straights']['S1']['full_throttle_share_99'] == pytest.approx(21 / 31) and r3['throttle_sat_action'] == .85
    _, r6_80 = E.lap_r4(rep, table, EvalConfig(), pedal_deadzone=True, entry_exit=False, throttle_remap=(.5, .8))
    assert r6_80['straights']['S1']['full_throttle_share_99'] == 1. and r6_80['throttle_sat_action'] == .8
    assert E.span_qss_time(table, 990., 10.) == pytest.approx(20 / 50)              # wraps through 0


def test_throttle_gate():
    q = np.array([10., -20., 30.])
    assert throttle_gate_value(q, q) == 0.
    assert throttle_gate_value(q + .2, q) == pytest.approx(.2 / 20)
    assert throttle_gate_passed([.05, .02, .009, .008, .007, .006], .01, 5)[0]
    assert not throttle_gate_passed([.05, .02, .011], .01, 5)[0]                     # above tol
    ok, why = throttle_gate_passed([.004, .003, .005, .006, .009], .01, 5)
    assert not ok and 'rising' in why
    assert not throttle_gate_passed([], .01, 5)[0] and not throttle_gate_passed([float('nan')], .01, 5)[0]


def test_preload_accepts_a_buffer_with_the_same_entry_exit_terms_only(tmp_path):
    """R5 -> R6: the source buffer already holds the run's own R4 entry / exit terms (B_in = B_out = 0), so no speed
    relabel is needed; an exit-bonus buffer or different entry / exit terms are still refused."""
    from test_r1 import _save_source
    from rlsetup.driving.agent import build_model, preload_buffer, default_init_buffer, InitMismatch
    from rlsetup.driving.config import CornerBonusConfig
    ee = CornerBonusConfig(enabled=True, mode='entry_exit', bonus_in=0., bonus_out=0.)
    c = tiny_cfg()
    for i, (src_cb, dst_cb, ok) in enumerate(((ee, ee, True), (CornerBonusConfig(enabled=True), ee, False),
                                              (ee, dataclasses.replace(ee, bonus_in=10.), False))):
        src_cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, reward=PBRS, corner_bonus=src_cb))
        _, path = _save_source(tmp_path / str(i), 'src', src_cfg, buffer=True)
        cfg = _r6_cfg(path, throttle_relabel=True)
        cfg = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, corner_bonus=dst_cb))
        model = build_model(cfg, DummyVecEnv([TinyEnv]))
        if ok:
            assert not preload_buffer(model, default_init_buffer(path), path, cfg)['throttle_relabel']['aborted']
        else:
            with pytest.raises(InitMismatch, match='exit bonuses'):
                preload_buffer(model, default_init_buffer(path), path, cfg)


# --- 21 Nov 2026 follow-up: offline phase removed, QA m1-m3 ----------------------------------------------------
def test_float32_action_below_the_knee_keeps_the_r3_dtype_bitwise():
    """QA m1: with the remap, a float32 action below the knee gives bitwise the R3 result of the same action."""
    rng = np.random.default_rng(2)
    for x in np.r_[rng.uniform(-1., .7, 5000), .7].astype(np.float32):
        a = np.array([0., x, x], np.float32)
        assert action_to_inputs(a, True, .3, R6).throttle == action_to_inputs(a, True, .3).throttle
    a = np.array([0., .8, -1.], np.float32)
    assert action_to_inputs(a, True, .3, R6).throttle == throttle_pedal(float(a[1]), R6)


def _r5_r6_cfgs(path):
    c = tiny_cfg()
    r5, r6 = PRESETS['bcn_f1_r5']().env, PRESETS['bcn_f1_r6']().env
    src = dataclasses.replace(c, env=r5)
    dst = dataclasses.replace(c, env=r6, agent=dataclasses.replace(c.agent, init_mode='warm'),
                              warm=dataclasses.replace(c.warm, relabel=False, throttle_relabel=True),
                              init_from=None if path is None else str(path))
    return src, dst


def test_preload_pass_through_needs_the_same_reward_block(tmp_path):
    """QA m2: R5 -> R6 (same corner terms, same reward) accepted; a changed reward block refused."""
    from test_r1 import _save_source
    from rlsetup.driving.agent import build_model, preload_buffer, default_init_buffer, InitMismatch
    src_cfg, _ = _r5_r6_cfgs(None)
    _, path = _save_source(tmp_path, 'src', src_cfg, buffer=True)
    _, cfg = _r5_r6_cfgs(path)
    info = preload_buffer(build_model(cfg, DummyVecEnv([TinyEnv])), default_init_buffer(path), path, cfg)
    assert not info['throttle_relabel']['aborted'] and info['throttle_relabel']['action_cells'] > 0
    for change in (dict(exit_k=.2), dict(line_k=.08), dict(speed_k=10.)):
        bad = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, reward=dataclasses.replace(cfg.env.reward,
                                                                                                    **change)))
        with pytest.raises(InitMismatch, match='reward'):
            preload_buffer(build_model(bad, DummyVecEnv([TinyEnv])), default_init_buffer(path), path, bad)


class _RetryTinyEnv(TinyEnv):
    decision_dt = .04

    def set_retry(self, on):
        self.retry_on = bool(on)


@pytest.mark.parametrize('offline', [0, 200])
def test_tiny_warm_run_schedules_gate_and_freeze(tmp_path, offline):
    """QA m3: throttle_relabel + lifetime_schedules through train.warm_offline_phase and learn() with the run's
    schedule callbacks. offline 0 (the R6 presets): one pre-training gate row, no lr override, no grad step, then
    critic lr 3e-6 from the first train() call. offline 200 at scale 0.1: critic lr 3e-5 offline, then 3e-6 from the
    first train() in learn(). Both: retries on and crash factor 3 at step 0, actor frozen (unchanged) until the
    configured unfreeze step, then trained."""
    from stable_baselines3.common.logger import configure
    from test_r1 import _save_source
    from rlsetup import runs
    from rlsetup.driving.agent import build_model, warm_start, preload_buffer, param_digest
    from rlsetup.driving.callbacks import PhaseCallback
    from rlsetup.driving.config import RetryConfig
    from rlsetup.driving.train import Run, warm_offline_phase
    from rlsetup.driving.train_config import CrashEmphasisConfig, LrDecayConfig
    c = tiny_cfg()
    env = dataclasses.replace(c.env, reward=PBRS, retry=RetryConfig(enabled=True, start_step=500_000))
    sched = dict(lr_decay=LrDecayConfig(enabled=True, start_step=1_000_000, end_step=2_000_000, floor=.01),
                 crash_emphasis=CrashEmphasisConfig(enabled=True, start_step=500_000))
    _, path = _save_source(tmp_path, 'src', dataclasses.replace(c, env=env, **sched), buffer=True)   # 2.6M steps
    cfg = dataclasses.replace(c, env=dataclasses.replace(env, throttle_remap=R6), lifetime_schedules=True,
                              agent=dataclasses.replace(c.agent, init_mode='warm'), init_from=str(path),
                              warm=dataclasses.replace(c.warm, relabel=False, throttle_relabel=True,
                                                       offline_critic_grad_steps=offline, offline_block=100,
                                                       offline_critic_lr_scale=.1 if offline else None,
                                                       throttle_gate_tol=1., throttle_gate_blocks=1,
                                                       critic_warmup_steps=30), **sched)
    model = build_model(cfg, DummyVecEnv([_RetryTinyEnv]))
    model.set_logger(configure(None, []))
    init = warm_start(model, path, cfg)
    aux = {}
    preload_buffer(model, cfg.warm.init_buffer or str(path.parents[1] / 'replay_buffer.pkl'), path, cfg, aux=aux)
    run = Run(cfg, tmp_path / 'run', lambda *a: None)
    run.dir.mkdir()
    run.status(lifetime_steps_offset=init['lifetime_steps_offset'])
    assert init['lifetime_steps_offset'] == 2_600_000
    lrs, frozen = [], []
    orig = model.train

    def train(*a, **k):
        orig(*a, **k)
        lrs.append(model.critic.optimizer.param_groups[0]['lr'])
        frozen.append((model.num_timesteps, model.freeze_actor))
    model.train = train
    actor0 = param_digest(model.actor)
    assert warm_offline_phase(run, model, cfg, init['lifetime_steps_offset'], aux, None) is None
    rows = runs.read_jsonl(run.dir / 'warmup.jsonl')
    gate_rows = [r for r in rows if 'throttle_gate' in r]
    st = runs.read_json(run.dir / 'status.json')
    assert gate_rows[0]['grad_steps'] == 0 and gate_rows[0]['block'] is None and st['throttle_gate']['passed']
    assert len(gate_rows) == 1 + offline // 100 and st['throttle_gate']['offline_grad_steps'] == offline
    assert model._n_updates == offline and param_digest(model.actor) == actor0
    if offline:
        assert model.lr_scale == .1 and lrs == pytest.approx([3e-5] * 2)
        assert st['phases']['offline_warmup']['lr_scale'] == .1
    else:
        assert model.lr_scale is None and lrs == [] and 'lr_scale' not in st['phases']['offline_warmup']
    n_off = len(lrs)
    from rlsetup.driving.train import schedule_offset
    off = schedule_offset(cfg, runs.read_json(run.dir / 'status.json'))
    retry = RetryCallback(cfg.env.retry.start_step, off)
    cbs = [LrDecayCallback(cfg.lr_decay, off), retry, CrashEmphasisCallback(cfg.crash_emphasis, off),
           PhaseCallback(cfg.warm.critic_warmup_steps)]
    model.learn(60, callback=cbs)
    assert lrs[n_off:] == pytest.approx([3e-6] * (len(lrs) - n_off)) and len(lrs) > n_off
    assert retry.on and model.get_env().envs[0].unwrapped.retry_on and model.replay_buffer.crash_factor == 3.
    assert [f for n, f in frozen[n_off:] if n < 30] and all(f for n, f in frozen[n_off:] if n < 30)
    assert not any(f for n, f in frozen[n_off:] if n >= 30)
    assert param_digest(model.actor) != actor0                                     # trained after the unfreeze


# --- 22 Nov 2026: from-scratch R6 (bcn_f1_r6_scratch; control bcn_f1_r5) ----------------------------------------
def test_scratch_presets_are_bcn_f1_r5_plus_the_remap_only():
    r5 = PRESETS['bcn_f1_r5']()
    c = PRESETS['bcn_f1_r6_scratch']()
    assert c.env.throttle_remap == R6 and c.env.reward.mode == 'time_pbrs'
    assert dataclasses.replace(c, preset='bcn_f1_r5', env=dataclasses.replace(c.env, throttle_remap=None)) == r5
    assert c.agent.init_mode == 'bc' and c.init_from is None and not c.lifetime_schedules
    assert (c.env.retry.start_step, c.crash_emphasis.start_step, c.lr_decay.start_step, c.lr_decay.end_step,
            c.lr_decay.floor, c.guard.threshold, c.stop.plateau) == (500_000, 500_000, 3_000_000, 6_200_000, .01, .4, False)
    assert train_config_from_dict(json.loads(config_json(c))) == c
    s = PRESETS['bcn_f1_r6_scratch_smoke']()
    assert (s.total_timesteps, s.max_wall_h, s.checkpoint_interval, s.strict_provenance, s.eval.interval,
            s.eval.stochastic_laps, s.guard.interval) == (150_000, 1., 25_000, False, 25_000, 3, 5_000)
    assert dataclasses.replace(s, preset=c.preset, total_timesteps=c.total_timesteps, max_wall_h=c.max_wall_h,
                               checkpoint_interval=c.checkpoint_interval, strict_provenance=True, eval=c.eval,
                               guard=c.guard) == c


def test_scratch_demo_key_differs_from_r5():
    from rlsetup.driving import demos as D
    qss = SimpleNamespace(meta={'key': {'x': 1}})
    prov = {'ocd_lap_code_version': 'x'}
    r5, c = PRESETS['bcn_f1_r5'](), PRESETS['bcn_f1_r6_scratch']()
    k5, d5 = D.dataset_key(r5.env, r5.demos, qss, prov)
    k6, d6 = D.dataset_key(c.env, c.demos, qss, prov)
    assert k5 != k6 and list(d6['env']['throttle_remap']) == [.7, .85] and 'throttle_remap' not in d5['env']
    assert {k: v for k, v in d6['env'].items() if k != 'throttle_remap'} == d5['env'] and d5['demos'] == d6['demos']


def test_demo_driver_acts_through_the_r6_inverse(make_env):
    """Same state, same demo controller: the R6 driver's action gives the same pedals (<= 2e-7) as the R3 driver's;
    pedal 1.0 -> a = 0.85; below the knee the actions are bitwise the R3 ones; brake / steering unchanged."""
    from rlsetup.driving.demo_driver import DemoDriver
    from rlsetup.driving.config import DemoConfig
    env = make_env(EnvConfig(reward=PBRS, throttle_remap=R6))
    from rlsetup.driving.demo_driver import DemoPolicy
    assert DemoPolicy(env, .5).driver.throttle_remap == R6 and env.settle_driver.throttle_remap == R6
    mk = lambda remap: DemoDriver(env.drive_line, DemoConfig(), pedal_deadzone=True, control_dt_s=.04,
                                  throttle_remap=remap)
    old, new = mk(None), mk(R6)
    env.reset(seed=4)
    st = dict(env.state, slip_rl=0., slip_rr=0.)
    seen = set()
    for target in np.linspace(0., 90., 61):
        old.reset(), new.reset()
        a0, a1 = old.act(st, env.s_drive, target), new.act(st, env.s_drive, target)
        p0, p1 = action_to_inputs(a0.astype(float), True), env.to_inputs(a1)
        assert abs(p0.throttle - p1.throttle) <= 2e-7 and p0.brake == p1.brake and a0[[0, 2]].tobytes() == a1[[0, 2]].tobytes()
        if a0[1] <= .7:
            assert a1[1] == a0[1]
            seen.add('below')
        if p0.throttle == 1.:
            assert a1[1] == np.float32(.85) and a0[1] == np.float32(1.)
            seen.add('full')
    assert seen == {'below', 'full'}


def test_demo_episode_runs_on_a_remapped_env(make_env):
    from rlsetup.driving import demos as D
    from rlsetup.driving.train_config import DemoDataConfig
    env = make_env(EnvConfig(reward=PBRS, throttle_remap=R6, max_episode_s=8.))
    ep = D.run_episode(env, DemoDataConfig(fraction_range=(.6, .6), noise_std=(0., 0., 0.)), 0)
    thr = ep['actions_clean'][:, 1]
    assert len(thr) == 200 and ep['cause'] == 'time_limit' and thr.max() <= np.float32(.85)
    assert (thr > .7).any()                                                          # the upper segment is used


def test_bc_runs_with_the_remap_and_reports_executed_pedals():
    """BC fits actions (map-free loss: same history with and without the remap); the throttle-vs-slip report maps
    throttle actions through the run's map (R6 pedal >= R3 pedal, > above the knee)."""
    import torch as th
    import gymnasium as gym
    from stable_baselines3.sac.policies import SACPolicy
    from rlsetup.driving import demos as D
    from rlsetup.driving.bc import train_bc
    from rlsetup.driving.train_config import BCConfig
    rng = np.random.default_rng(0)
    E, T, dim = 20, 100, 12
    x = rng.uniform(-1, 1, (E * (T + 1), dim)).astype(np.float32)
    x[:, 7:9] = rng.uniform(0., .5, (len(x), 2))
    starts = np.arange(E) * T
    d = D.with_index({'obs': x, 'ep_start_tr': starts, 'ep_start_obs': starts + np.arange(E),
                      'terminated': np.zeros(E * T, bool), 'truncated': np.zeros(E * T, bool),
                      'actions': np.zeros((E * T, 3), np.float32), 'episode': np.repeat(np.arange(E), T)})
    a = np.stack([np.zeros(E * T), rng.uniform(.5, 1., E * T), np.full(E * T, -.8)], 1).astype(np.float32)
    d['actions'], d['actions_clean'] = a, a
    reps = []
    for remap in (None, R6):
        th.manual_seed(0)
        pol = SACPolicy(gym.spaces.Box(-10, 10, (dim,), np.float32), gym.spaces.Box(-1, 1, (3,), np.float32),
                        lambda _: 3e-4, net_arch=[32, 32])
        reps.append(train_bc(pol.actor, d, BCConfig(max_epochs=2, batch_size=256), seed=0, log=lambda *_: None,
                             throttle_remap=remap))
    r3, r6 = reps
    assert [h['val_nll'] for h in r3['history']] == [h['val_nll'] for h in r6['history']]
    rows = [(x3, x6) for x3, x6 in zip(r3['throttle_vs_rear_slip'], r6['throttle_vs_rear_slip']) if x3['n']]
    assert rows and all(x6['executed_throttle_pedal'] > x3['executed_throttle_pedal'] for x3, x6 in rows)
    assert all(x6['executed_throttle_pedal'] <= 1. for _, x6 in rows)


def test_init_output_bias_pedal_unchanged_by_the_remap():
    import torch as th
    from rlsetup.driving.agent import build_model, init_output_bias
    c = PRESETS['bcn_f1_r6_scratch']()
    model = build_model(dataclasses.replace(tiny_cfg(), agent=c.agent), DummyVecEnv([TinyEnv]))
    init_output_bias(model.actor, pedal_deadzone=True, action=c.agent.init_action, log_std=c.agent.init_log_std,
                     weight_scale=c.agent.init_weight_scale)
    with th.no_grad():
        a = th.tanh(model.actor.mu.bias).numpy()
    assert c.agent.init_action == (0., -.275, -.8) and a[1] == pytest.approx(-.275)
    assert throttle_pedal(float(a[1]), R6) == throttle_pedal(float(a[1]), None) == pytest.approx(.15)
    assert E._pedals(a[None], True, R6)[1][0] == 0.                                  # brake released
