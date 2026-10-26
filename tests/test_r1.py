"""R1 (time_pbrs reward, warm start, plateau stop): docs/specs/R1_time_optimal.md §7 tests 1-11."""
import dataclasses
import json
import math
import types
import gymnasium as gym
import numpy as np
import pytest
import torch as th
from ocd_lap.backend import StartPose
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup import runs
from rlsetup.driving.agent import build_model, warm_start, InitMismatch, param_digest
from rlsetup.driving.callbacks import PlateauStop, plateau_reached, plateau_entry, critic_check_value
from rlsetup.driving.config import EnvConfig, RewardConfig
from rlsetup.driving.demo_driver import DemoPolicy
from rlsetup.driving.train_config import (PRESETS, TrainConfig, StopConfig, config_json, train_config_from_dict)
from r1_golden import legacy_rollout
from sb3_helpers import TinyEnv, tiny_cfg, make_model, OBS

R1 = lambda **kw: RewardConfig(mode='time_pbrs', **kw)
GAMMA = RewardConfig().gamma
BRAKE = np.array([0., -1., 1.], np.float32)
CRASH_LEFT = np.array([1., 1., -1.], np.float32)


# --- 1: legacy bit identity --------------------------------------------------------------------
GOLDEN_LEGACY = '0007e2813d2d28971223460790b45852513768865f08d6d0b3685f5868a1c4c2'


def test_legacy_rollout_bit_identical(make_env):
    """ref_tracking is bit-identical to the code that trained full1. GOLDEN_LEGACY was generated
    by running `r1_golden.legacy_rollout` (300 decisions: steering sweep, pedal steps, a forced
    off-track, a stall, a time-limit truncation, a wrong-way end) on the unmodified tree at
    47f1d8d, before env.py was edited for R1; 47f1d8d is code-identical to aeab814 for rlsetup/
    (`git diff aeab814 47f1d8d -- rlsetup` is empty), same machine (macOS arm64, M4)."""
    digest, ends = legacy_rollout(make_env)
    assert ends == ['off_track', 'stall', 'time_limit', 'wrong_way', 'off_track', 'stall']
    assert digest == GOLDEN_LEGACY
    assert RewardConfig().mode == 'ref_tracking'


# --- 2: shaping exactness ----------------------------------------------------------------------
def _pair_rollout(make_env, env_kw, reset_kw, action):
    """Same actions on time_pbrs envs with shaping_k = 1 and 0. Checks per decision: obs, flags
    and state bitwise equal; r(k=1) - r(k=0) = shaping; Phi = -|e| for live states. Returns
    (last info of k=1, Phi(s_0), shaping list)."""
    envs = [make_env(EnvConfig(reward=R1(shaping_k=k), **env_kw)) for k in (1., 0.)]
    (o1, i1), (o0, i0) = [e.reset(**reset_kw) for e in envs]
    assert o1.tobytes() == o0.tobytes() and i0['phi'] == 0. and i1['reward_mode'] == 'time_pbrs'
    assert i1['phi'] == -abs(i1['e_ref']) == pytest.approx(-abs(o1[9] * 5.), abs=1e-5)
    fs = []
    for t in range(2000):
        a = action(t)
        o1, r1, te1, tr1, f1 = envs[0].step(a)
        o0, r0, te0, tr0, f0 = envs[1].step(a)
        assert o1.tobytes() == o0.tobytes() and (te1, tr1) == (te0, tr0) and envs[0].state == envs[1].state
        assert f0['shaping'] == 0. and f1['step_cost'] == f0['step_cost']
        assert r1 - r0 == pytest.approx(f1['shaping'], abs=1e-12)
        fs.append(f1['shaping'])
        if not te1:
            assert f1['phi'] == -abs(f1['e_ref'])
        if te1 or tr1:
            break
    T = len(fs)
    lhs = sum(GAMMA ** t * f for t, f in enumerate(fs))
    assert lhs == pytest.approx(GAMMA ** T * f1['phi'] - i1['phi'], rel=1e-12, abs=1e-12)
    assert f1['episode_shaping'] == pytest.approx(sum(fs), abs=1e-12)
    return f1, i1['phi'], fs


def test_shaping_exact_and_telescopes_off_track(make_env):
    info, phi0, fs = _pair_rollout(make_env, {}, dict(seed=1), lambda t: CRASH_LEFT)
    assert info['termination'] == 'off_track' and info['phi'] == 0. and len(fs) > 3


def test_shaping_exact_and_telescopes_time_limit(make_env):
    info, phi0, fs = _pair_rollout(make_env, dict(max_episode_s=.4), dict(seed=0),
                                   lambda t: np.array([.02, -.5, -1.], np.float32))
    assert info['truncation'] == 'time_limit' and len(fs) == 10
    assert info['phi'] == -abs(info['e_ref']) != 0.


def test_shaping_exact_and_telescopes_stall(make_env):
    info, phi0, fs = _pair_rollout(make_env, dict(settle_s=0.),
                                   dict(options={'start_s': 4900., 'start_speed': 3.5}), lambda t: BRAKE)
    assert info['truncation'] == 'stall' and info['termination'] is None
    assert info['phi'] == -abs(info['e_ref']) != 0.


# --- 3: n-step telescoping in SB3's buffer ------------------------------------------------------
def test_nstep_shaping_telescopes_in_sb3_buffer():
    """Rewards = shaping only (F = gamma Phi(s') - Phi(s), Phi(s') = 0 at terminations, real at
    timeouts and at the open row pos - 1), obs[0] = Phi(s), next_obs[0] = Phi(s'): every SB3
    2.9.0 n-step sample satisfies n_step_return = discount Phi(next_obs) (1 - final_done) - Phi(obs)."""
    from stable_baselines3.common.buffers import NStepReplayBuffer
    space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
    act = gym.spaces.Box(-1., 1., (1,), np.float32)
    n_envs, rows = 2, 150
    rb = NStepReplayBuffer(400, space, act, device='cpu', n_envs=n_envs, n_steps=3, gamma=GAMMA,
                           handle_timeout_termination=True)
    rng = np.random.default_rng(0)
    phi, left, kind = [0.] * n_envs, [0] * n_envs, [None] * n_envs
    for row in range(rows):
        obs, nxt, rew, done, infos = (np.zeros((n_envs, 1), np.float32), np.zeros((n_envs, 1), np.float32),
                                      np.zeros(n_envs, np.float32), np.zeros(n_envs, bool), [])
        for i in range(n_envs):
            if left[i] == 0:
                phi[i], left[i] = -float(rng.uniform(0., 5.)), int(rng.integers(1, 8))
                kind[i] = 'term' if rng.uniform() < .5 else 'timeout'
            phi_next = -float(rng.uniform(0., 5.))
            left[i] -= 1
            end = left[i] == 0 and not (row == rows - 1 and i == 0)     # env 0: last row stays open (cut)
            term, timeout = end and kind[i] == 'term', end and kind[i] == 'timeout'
            obs[i, 0], nxt[i, 0], done[i] = phi[i], phi_next, end
            rew[i] = GAMMA * (0. if term else phi_next) - phi[i]
            infos.append({'TimeLimit.truncated': timeout})
            phi[i] = phi_next
        rb.add(obs, nxt, np.zeros((n_envs, 1), np.float32), rew, done, infos)
    assert not rb.dones[rows - 1, 0] and rb.timeouts[:rows].any() and (rb.dones[:rows] * (1 - rb.timeouts[:rows])).any()
    np.random.seed(0)
    s = rb._get_samples(np.tile(np.arange(rows), 8))
    o, nx = s.observations[:, 0].numpy().astype(float), s.next_observations[:, 0].numpy().astype(float)
    ret, disc, fd = (x[:, 0].numpy().astype(float) for x in (s.rewards, s.discounts, s.dones))
    np.testing.assert_allclose(ret, disc * nx * (1. - fd) - o, atol=1e-4)
    full = np.abs(disc - GAMMA ** 3) < 1e-6                         # 3-step windows vs windows cut by an end
    assert (fd == 1).any() and (fd == 0).any() and full.any() and (~full).any()


# --- 4: one gamma -------------------------------------------------------------------------------
def test_gamma_is_one_object(make_env):
    cfg = PRESETS['r1_warm']()
    cfg = dataclasses.replace(cfg, n_envs=1, agent=dataclasses.replace(cfg.agent, buffer_size=1_000))
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    env = make_env(cfg.env)
    assert model.gamma == cfg.env.reward.gamma == env.cfg.reward.gamma == GAMMA
    _, info = env.reset(seed=3)
    _, _, _, _, step = env.step(np.array([0., -.275, -.8], np.float32))
    assert step['shaping'] == model.gamma * step['phi'] - info['phi']


# --- 5: stall ------------------------------------------------------------------------------------
def _stall_vec(make_env, reward):
    venv = DummyVecEnv([lambda: make_env(EnvConfig(settle_s=0., reward=reward))])
    venv.set_options({'start_s': 4900., 'start_speed': 3.5})
    venv.reset()
    for _ in range(200):
        _, _, dones, infos = venv.step(BRAKE[None])
        if dones[0]:
            return infos[0]
    raise AssertionError('no stall')


def test_stall_is_truncation_in_r1_and_terminal_in_legacy(make_env):
    env = make_env(EnvConfig(settle_s=0., reward=R1()))
    env.reset(options={'start_s': 4900., 'start_speed': 3.5})
    for _ in range(200):
        _, r, term, trunc, info = env.step(BRAKE)
        if term or trunc:
            break
    assert trunc and not term and info['truncation'] == 'stall' and info['termination'] is None
    assert info['step_cost']['penalty'] == 0. and info['time_s'] == pytest.approx(2., abs=.3)
    c = info['step_cost']
    assert r == pytest.approx(c['progress'] - c['time'] + info['shaping'], abs=1e-12)
    v = _stall_vec(make_env, R1())
    assert v['TimeLimit.truncated'] is True and 'terminal_observation' in v and v['truncation'] == 'stall'
    legacy = _stall_vec(make_env, RewardConfig())
    assert legacy['TimeLimit.truncated'] is False and legacy['termination'] == 'stall'
    assert legacy['truncation'] is None
    assert legacy['step_cost']['penalty'] == pytest.approx(RewardConfig().failure_penalty(legacy['e_ref'], .02))


# --- 6: lap cost independent of the reward ------------------------------------------------------
def test_lap_costs_independent_of_mode_and_k(make_env):
    """Same actions -> identical `laps` (time, smoothness, tracking, cost, cost_r1) and crossings
    under ref_tracking, time_pbrs k = 0 and k = 1 (lines shortened to 5 m apart for speed)."""
    envs = [make_env(EnvConfig(reward=r)) for r in (RewardConfig(), R1(shaping_k=0.), R1())]
    for e in envs:
        e.reset(options={'evaluation': True})
        e._lines = [1., 6., 11.]
    policy = DemoPolicy(envs[0], .5)
    policy.reset()
    for _ in range(100):
        a = policy()
        outs = [e.step(a) for e in envs]
        if outs[0][2] or outs[0][3]:
            break
    infos = [o[4] for o in outs]
    assert infos[0]['termination'] == 'lap_complete' and len(infos[0]['laps']) == 2
    assert infos[0]['laps'] == infos[1]['laps'] == infos[2]['laps']
    assert infos[0]['line_crossings_s'] == infos[1]['line_crossings_s'] == infos[2]['line_crossings_s']
    for lap in infos[2]['laps']:
        assert lap['cost'] == lap['time'] + lap['smoothness'] + lap['tracking'] and lap['cost_r1'] == lap['time']
    assert infos[2]['phi'] == -abs(infos[2]['e_ref'])        # lap_complete: real Phi(s')
    assert infos[2]['shaping'] != 0. and infos[0]['shaping'] == 0.


# --- 7: crash accounting -------------------------------------------------------------------------
def _crash(make_env, reset_kw, action, **env_kw):
    env = make_env(EnvConfig(reward=R1(), **env_kw))
    _, info = env.reset(**reset_kw)
    phi_prev = info['phi']
    for _ in range(500):
        _, r, term, trunc, info = env.step(action)
        if term or trunc:
            return r, info, phi_prev
        phi_prev = info['phi']
    raise AssertionError('no crash')


def test_crash_penalty_constant_and_last_shaping(make_env, track):
    seen_e = []
    for reset_kw, action in ((dict(seed=1), CRASH_LEFT), (dict(seed=5), np.array([-1., 1., -1.], np.float32))):
        r, info, phi_prev = _crash(make_env, reset_kw, action)
        assert info['termination'] == 'off_track' and info['step_cost']['penalty'] == 100.
        assert info['phi'] == 0. and info['shaping'] == -phi_prev and phi_prev != 0.
        c = info['step_cost']
        assert r == pytest.approx(c['progress'] - c['time'] - 100. - phi_prev, abs=1e-12)
        assert info['episode_cost']['penalty'] == 100.
        seen_e.append(abs(info['e_ref']))
    assert abs(seen_e[0] - seen_e[1]) > .5                     # P does not depend on e
    s = 4900.
    x, y = track.point(s)
    pose = StartPose(x, y, make_env().ref.heading(s) + math.pi, 5.)
    r, info, phi_prev = _crash(make_env, dict(options={'start_pose': pose}), np.array([0., -1., -1.], np.float32),
                               settle_s=0.)
    assert info['termination'] == 'wrong_way' and info['step_cost']['penalty'] == 100. and info['phi'] == 0.


# --- 8 / 9: warm start ---------------------------------------------------------------------------
class TinyEnv160(TinyEnv):
    observation_space = gym.spaces.Box(-10., 10., (OBS + 1,), np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        return np.zeros(OBS + 1, np.float32), {}


def _save_source(tmp_path, name, cfg, env_cls=TinyEnv, write_config=True, train=0, n_envs=1, buffer=False):
    """A trained-looking source run folder: <name>/seed0/{config.json, provenance.json,
    checkpoints/best.zip[, replay_buffer.pkl]}; `train` gradient steps on a random buffer (Adam
    states), then parameters and log alpha perturbed away from a fresh init."""
    src = build_model(cfg, DummyVecEnv([env_cls] * n_envs))
    if train or buffer:
        rng = np.random.default_rng(0)
        obs_n = src.observation_space.shape[0]
        for _ in range(600):
            src.replay_buffer.add(rng.uniform(-1, 1, (n_envs, obs_n)).astype(np.float32),
                                  rng.uniform(-1, 1, (n_envs, obs_n)).astype(np.float32),
                                  rng.uniform(-1, 1, (n_envs, 3)).astype(np.float32),
                                  rng.normal(size=n_envs).astype(np.float32), rng.uniform(size=n_envs) < .02,
                                  [{}] * n_envs)
    if train:
        from stable_baselines3.common.logger import configure
        src.set_logger(configure(None, []))
        src.train(train, 64)
    with th.no_grad():
        for p in src.policy.parameters():
            p.add_(.01)
        src.log_ent_coef.fill_(-3.1)
    src.num_timesteps = 2_600_000
    d = tmp_path / name / 'seed0'
    (d / 'checkpoints').mkdir(parents=True)
    src.save(d / 'checkpoints' / 'best.zip')
    if write_config:
        (d / 'config.json').write_text(config_json(cfg))
    if buffer:
        src.save_replay_buffer(d / 'replay_buffer.pkl')
    runs.write_json(d / 'provenance.json', {'rlsetup': {'commit': 'f' * 40}})
    return src, d / 'checkpoints' / 'best.zip'


def _warm_cfg(src_path=None, **warm):
    c = tiny_cfg()
    return dataclasses.replace(c, env=dataclasses.replace(c.env, reward=R1()),
                               agent=dataclasses.replace(c.agent, init_mode='warm'),
                               warm=dataclasses.replace(c.warm, **warm),
                               init_from=None if src_path is None else str(src_path))


def test_warm_start_copies_policy_and_alpha(tmp_path):
    src, path = _save_source(tmp_path, 'src', tiny_cfg(), train=3)
    cfg = _warm_cfg(path, load_optimizers=False)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    info = warm_start(model, path, cfg)
    a, b = src.policy.state_dict(), model.policy.state_dict()
    assert set(a) == set(b) and all(th.equal(a[k], b[k]) for k in a)
    assert {k.split('.')[0] for k in a} == {'actor', 'critic', 'critic_target'}
    assert th.equal(model.log_ent_coef, src.log_ent_coef) and model.alpha == pytest.approx(math.exp(-3.1))
    assert model.num_timesteps == 0 and model._n_updates == 0 and model.replay_buffer.size() == 0
    assert model.actor.optimizer.state_dict()['state'] == {} and model.critic.optimizer.state_dict()['state'] == {}
    assert model.ent_coef_optimizer.state_dict()['state'] == {}
    assert model.ent_coef_optimizer.param_groups[0]['params'][0] is model.log_ent_coef   # copied in place
    assert model.freeze_actor and model.freeze_ent_coef
    assert not any(p.requires_grad for p in model.actor.parameters())
    assert model.min_train_transitions == 512 * 10
    assert info['src_num_timesteps'] == 2_600_000 and info['src_alpha'] == pytest.approx(math.exp(-3.1))
    assert info['src_rlsetup_commit'] == 'f' * 40 and info['path'] == str(path.resolve())
    import hashlib
    assert info['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info['actor_sha256'] == param_digest(model.actor) == param_digest(src.actor)
    assert info['log_ent_coef_sha256'] == param_digest(model.log_ent_coef)


def test_warm_start_without_critics(tmp_path):
    src, path = _save_source(tmp_path, 'src', tiny_cfg())
    cfg = _warm_cfg(path, load_critics=False, load_alpha=False)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    critic0 = {k: v.clone() for k, v in model.critic.state_dict().items()}
    alpha0 = model.log_ent_coef.detach().clone()
    warm_start(model, path, cfg)
    assert all(th.equal(v, src.actor.state_dict()[k]) for k, v in model.actor.state_dict().items())
    assert all(th.equal(v, critic0[k]) for k, v in model.critic.state_dict().items())
    assert all(th.equal(v, critic0[k]) for k, v in model.critic_target.state_dict().items())
    assert th.equal(model.log_ent_coef, alpha0)


def test_warm_start_refuses_mismatches(tmp_path):
    _, other_obs = _save_source(tmp_path, 'obs', tiny_cfg(), env_cls=TinyEnv160)
    c = tiny_cfg()
    _, other_env = _save_source(tmp_path, 'env', dataclasses.replace(c, env=dataclasses.replace(c.env, max_episode_s=30.)))
    _, other_gamma = _save_source(tmp_path, 'gamma', dataclasses.replace(
        c, env=dataclasses.replace(c.env, reward=dataclasses.replace(c.env.reward, gamma=.99))))
    _, other_arch = _save_source(tmp_path, 'arch', dataclasses.replace(c, agent=dataclasses.replace(c.agent, net_arch=(64, 64))))
    _, no_config = _save_source(tmp_path, 'nocfg', tiny_cfg(), write_config=False)
    for path, msg in ((other_obs, 'observation space'), (other_env, 'env config'), (other_gamma, 'gamma'),
                      (other_arch, 'net_arch'), (no_config, 'config.json missing')):
        cfg = _warm_cfg(path)
        model = build_model(cfg, DummyVecEnv([TinyEnv]))
        with pytest.raises(InitMismatch, match=msg):
            warm_start(model, path, cfg)


def test_frozen_phase_trains_critic_only(tmp_path):
    """Spec test 9: after warm_start, train() keeps actor and log alpha bit-identical while the
    critic learns; no gradient step below min_train_transitions."""
    _, path = _save_source(tmp_path, 'src', tiny_cfg())
    cfg = _warm_cfg(path, min_buffer_batches=2)
    model = make_model(cfg, fill=0)
    warm_start(model, path, cfg)
    actor0, alpha0, critic0 = param_digest(model.actor), param_digest(model.log_ent_coef), param_digest(model.critic)
    rng = np.random.default_rng(1)

    def fill(n):
        for _ in range(n):
            model.replay_buffer.add(rng.uniform(-1, 1, (1, OBS)).astype(np.float32),
                                    rng.uniform(-1, 1, (1, OBS)).astype(np.float32),
                                    rng.uniform(-1, 1, (1, 3)).astype(np.float32), rng.normal(size=1).astype(np.float32),
                                    np.array([False]), [{}])
    fill(1_000)                                                     # < 512 x 2
    model.train(5, cfg.agent.batch_size)
    assert model._n_updates == 0 and param_digest(model.critic) == critic0
    fill(100)
    model.train(5, cfg.agent.batch_size)
    assert model._n_updates == 5
    assert param_digest(model.actor) == actor0 and param_digest(model.log_ent_coef) == alpha0
    assert param_digest(model.critic) != critic0


# --- 10: plateau stop ----------------------------------------------------------------------------
STOP = StopConfig(plateau=True)          # min 1M, window 500k, 0.1 s


def _hist(det, sto, step=50_000):
    return [(k * step, d, s) for k, (d, s) in enumerate(zip(det, sto))]


def _first_stop(history, cfg=STOP):
    for i, h in enumerate(history):
        if plateau_reached(history[:i + 1], h[0], cfg):
            return h[0]
    return None


def test_plateau_stops_exactly_when_both_bests_stall():
    n = 41                                                         # 0 .. 2.0M every 50k
    det = [130. - .5 * min(k, 12) for k in range(n)]               # last gain at 600k
    sto = [135. - .5 * min(k, 14) for k in range(n)]               # last gain at 700k
    assert _first_stop(_hist(det, sto)) == 1_200_000               # 700k + 500k
    det_only = [130. - .5 * min(k, 30) for k in range(n)]          # det improves until 1.5M
    assert _first_stop(_hist(det_only, sto)) == 2_000_000
    flat = [125.] * n
    assert _first_stop(_hist(flat, flat)) == 1_000_000             # not before min_steps


def test_plateau_counts_cumulative_and_threshold_gains():
    n = 41
    creep = [130. - .03 * k for k in range(n)]                     # 0.3 s per 500k: improving
    flat = [125.] * n
    assert _first_stop(_hist(creep, flat)) is None
    tiny = [130. - .009 * k for k in range(n)]                     # 0.09 s per 500k: plateau
    assert _first_stop(_hist(tiny, flat)) == 1_000_000
    exact = [123.3] * 15 + [123.2] * 26                            # exactly 0.1 s at 700k (float)
    h = _hist(exact, flat)
    assert not plateau_reached(h[:25], 1_200_000, STOP) and plateau_reached(h[:26], 1_250_000, STOP)
    unclean = _hist([None] * 30 + [120.] * 11, flat)               # first clean lap inside the window
    assert not plateau_reached(unclean[:31], 1_500_000, STOP)
    assert plateau_reached(_hist([None] * n, [None] * n), 1_000_000, STOP)   # never a value: plateau


def test_plateau_callback_and_entries():
    evalcb = types.SimpleNamespace(records=[])
    events = []
    cb = PlateauStop(STOP, evalcb, history=[(0, 125., 126.)], on_event=lambda k, **d: events.append((k, d)))
    model = types.SimpleNamespace(num_timesteps=0)
    cb.model = model
    for k in range(1, 41):
        model.num_timesteps = k * 50_000
        evalcb.records.append({'step': k * 50_000, 'det_clean': k % 2 == 0, 'lap2_s': 120.,
                               'sto_lap_mean_s': 126.})
        go = cb.on_step()
        if not go:
            break
    # det 120 first clean at 100k; sto flat: both stalled from 100k -> stop at max(1M, 600k)
    assert model.num_timesteps == 1_000_000 and cb.triggered and events[0][0] == 'plateau'
    assert cb.on_step()                                            # no new record: no decision
    assert plateau_entry({'step': 5, 'det_clean': False, 'lap2_s': 1., 'sto_lap_mean_s': None}) == (5, None, None)
    off = PlateauStop(StopConfig(plateau=False), types.SimpleNamespace(records=[{'step': 2_000_000, 'det_clean': True,
                                                                                  'lap2_s': 1., 'sto_lap_mean_s': 1.}]))
    off.model = model
    assert off.on_step() and not off.triggered and off.history == [(2_000_000, 1., 1.)]


def test_critic_check_value():
    assert critic_check_value({'q_min_minus_mc_nstep': -9., 'mc_nstep_mean': 180., 'q_min_minus_mc': 1.,
                               'mc_mean': 1.}) == pytest.approx(.05)
    assert critic_check_value({'q_min_minus_mc': 20., 'mc_mean': -200.}) == pytest.approx(.1)
    assert critic_check_value({'n': 0}) is None and critic_check_value(None) is None


def test_eval_callback_critic_check_once(tmp_path):
    from rlsetup.driving.callbacks import DrivingEvalCallback
    model = make_model(fill=0)
    q = {'q_min_minus_mc_nstep': 20., 'mc_nstep_mean': 190.}
    events = []
    cb = DrivingEvalCallback(lambda m, n: {'det_clean': True, 'lap2_s': 123., 'q_vs_mc': q}, 50_000,
                             tmp_path / 'evals.jsonl', tmp_path, on_event=lambda k, **d: events.append(k),
                             critic_check_at=100_000, critic_check_tol=.08)
    cb.init_callback(model)
    model.num_timesteps = 0
    cb.on_training_start({}, {})
    for n in (50_000, 100_000, 150_000):
        model.num_timesteps = n
        cb.on_step()
    recs = runs.read_jsonl(tmp_path / 'evals.jsonl')
    assert [('critic_check' in r) for r in recs] == [False, False, True, False]
    assert recs[2]['critic_check'] == pytest.approx(20 / 190) and recs[2]['critic_check_tol'] == .08
    assert events == ['critic_not_converged']


# --- evaluation additions (R1 §2.3) ------------------------------------------------------------
def test_progress_jumps_and_lap_metrics():
    from rlsetup.driving import evaluate as ev
    from rlsetup.driving.train_config import EvalConfig
    p = [1., 2., 6., 7., 5.5, 6.]                 # decision 2: +4 m at 1 m/s over 1 s; 4: -1.5 m
    t = [1., 2., 3., 4., 5., 6.]
    v = [1., 1., 1., 1., 1., 1.]
    assert list(ev.progress_jumps(p, t, v)) == [False, False, True, False, True, False]
    assert list(ev.progress_jumps([30.], [.04], [math.nan])) == [False]
    n = 6
    tr = {'lap_index': np.array([1, 2, 2, 2, 2, 3]), 'e': np.array([0., .5, -1.5, 3.5, 0., 0.]),
          'track_position': np.zeros(n), 'track_width': np.full(n, 12.), 'kappa': np.zeros((n, 4)),
          'gear': np.full(n, 3), 'slip_rl': np.zeros(n), 'slip_rr': np.zeros(n), 'shaping': np.arange(n, dtype=float),
          'progress_m': np.array(p), 'time_s': np.array(t), 'speed': np.array(v)}
    lap = dict(lap_time_s=90., time=90., smoothness=1., tracking=2., cost=93., cost_r1=90.)
    rep = ev.LapReport(True, [90., 90.], [lap, lap], 'lap_complete', None, n, 0., tr)
    m = ev.lap_metrics(rep, EvalConfig())
    assert (m['cost'], m['cost_r1'], m['decisions']) == (93., 90., 4)
    assert m['share_abs_e_gt_1m'] == .5 and m['share_abs_e_gt_3m'] == .25
    assert m['shaping_sum'] == 1. + 2. + 3. + 4. and m['progress_jumps'] == 2


# --- 11: configs, presets, dirty check -----------------------------------------------------------
def _pre_r1_dict():
    d = TrainConfig().to_dict()
    for k in ('mode', 'shaping_k', 'penalty_const'):
        del d['env']['reward'][k]
    for k in ('warm', 'stop', 'init_from', 'strict_provenance'):
        del d[k]
    return json.loads(json.dumps(d))


def test_old_config_loads_as_ref_tracking():
    cfg = train_config_from_dict(_pre_r1_dict())
    assert cfg.env.reward.mode == 'ref_tracking' and cfg.env == EnvConfig() and cfg.init_from is None
    assert cfg.stop.plateau is False and cfg.strict_provenance
    from rlsetup.provenance import REPO_ROOT
    full1 = REPO_ROOT / 'runs' / 'D2' / 'full1' / 'seed0' / 'config.json'
    if full1.exists():                                              # the real full1 config (not in git)
        old = train_config_from_dict(json.loads(full1.read_text()))
        assert old.env.reward.mode == 'ref_tracking' and old.env == EnvConfig()


@pytest.mark.parametrize('kw', [dict(mode='time'), dict(shaping_k=-1.), dict(penalty_const=math.inf),
                                dict(gamma=0.), dict(gamma=1.01)])
def test_reward_config_validation(kw):
    with pytest.raises(ValueError):
        RewardConfig(**kw)


def test_r1_presets():
    full, warm, smoke = PRESETS['full'](), PRESETS['r1_warm'](), PRESETS['r1_warm_smoke']()
    assert full == TrainConfig(preset='full') and full.strict_provenance and PRESETS['smoke']().strict_provenance is False
    assert warm.env == dataclasses.replace(full.env, reward=R1()) and warm.agent.init_mode == 'warm'
    assert (warm.total_timesteps, warm.max_wall_h, warm.guard.threshold, warm.stop.plateau) == (3_000_000, 4., .65, True)
    w = warm.warm                                                  # Amendment 1 §B
    assert (w.critic_warmup_steps, w.offline_critic_grad_steps, w.load_optimizers, w.relabel, w.init_buffer,
            w.actor_lr) == (20_000, 25_000, True, True, None, None)
    assert warm.strict_provenance and warm.agent == dataclasses.replace(full.agent, init_mode='warm')
    assert (smoke.max_wall_h, smoke.warm, smoke.eval.interval, smoke.eval.stochastic_laps,          # Amendment 1 §D
            smoke.guard.interval, smoke.checkpoint_interval, smoke.stop.plateau, smoke.strict_provenance) == \
        (.5, warm.warm, 25_000, 3, 5_000, 25_000, False, False)
    assert smoke.env == warm.env and smoke.guard.threshold == .65 and smoke.total_timesteps == 150_000


def test_dirty_tree_refused_by_strict_presets(monkeypatch, tmp_path):
    from rlsetup import provenance
    from rlsetup.driving import train as T

    class Reached(Exception):
        pass

    def make_run_dir(*a, **k):
        raise Reached
    monkeypatch.setattr(provenance, 'git_state', lambda path: {'dirty': True})
    monkeypatch.setattr(runs, 'make_run_dir', make_run_dir)
    ck = tmp_path / 'src' / 'seed0' / 'checkpoints'
    ck.mkdir(parents=True)
    for f in (ck / 'x.zip', ck.parent / 'config.json', ck.parent / 'replay_buffer.pkl'):
        f.write_text('')
    init = dict(init_from=str(ck / 'x.zip'))
    assert T.run_pipeline(dataclasses.replace(PRESETS['r1_warm'](), **init), 'r') == T.EXIT_DIRTY
    assert T.run_pipeline(PRESETS['full'](), 'r') == T.EXIT_DIRTY
    with pytest.raises(Reached):
        T.run_pipeline(dataclasses.replace(PRESETS['r1_warm_smoke'](), **init), 'r')
    with pytest.raises(Reached):
        T.run_pipeline(dataclasses.replace(PRESETS['r1_warm'](), **init), 'r', allow_dirty=True)
    with pytest.raises(ValueError, match='init_from'):
        T.run_pipeline(PRESETS['r1_warm'](), 'r')                  # warm without init_from
    with pytest.raises(ValueError, match='init_from'):
        T.run_pipeline(dataclasses.replace(PRESETS['full'](), **init), 'r')


def test_cli_init_from_absolute(monkeypatch, tmp_path):
    from rlsetup.driving import train as T
    seen = {}
    monkeypatch.setattr(T, 'run_pipeline', lambda cfg, name, **k: seen.update(cfg=cfg) or 0)
    monkeypatch.chdir(tmp_path)
    assert T.main(['--preset', 'r1_warm_smoke', '--run-name', 'r', '--init-from', 'a/b.zip']) == 0
    assert seen['cfg'].init_from == str((tmp_path / 'a' / 'b.zip').resolve())
    with pytest.raises(SystemExit):
        T.main(['--preset', 'r1_warm_smoke', '--run-name', 'r'])
    with pytest.raises(SystemExit):
        T.main(['--preset', 'smoke', '--run-name', 'r', '--init-from', 'a/b.zip'])
    assert seen['cfg'].warm.init_buffer is None                     # default resolved in run_pipeline
    T.main(['--preset', 'r1_warm_smoke', '--run-name', 'r', '--init-from', 'a/b.zip', '--init-buffer', 'c/rb.pkl'])
    assert seen['cfg'].warm.init_buffer == str((tmp_path / 'c' / 'rb.pkl').resolve())
    with pytest.raises(SystemExit):
        T.main(['--preset', 'r1_warm_smoke', '--run-name', 'r', '--init-buffer', 'c/rb.pkl'])


def test_resume_warm_frozen_phase_and_strict_rule(tmp_path, monkeypatch):
    """Resume of a warm run in its critic-only phase stays frozen until critic_warmup_steps,
    keeps the critic check and the plateau history; resume never re-applies init_from. A config
    written before R1 (no strict_provenance) keeps the preset rule: smoke resumes on a dirty tree."""
    from test_callbacks import resume_fixture, CODE
    from rlsetup.driving import train as T
    cfg = dataclasses.replace(_warm_cfg(tmp_path / 'src.zip'), stop=StopConfig(plateau=True))
    d, model, seen = resume_fixture(tmp_path, monkeypatch, cfg=cfg,
                                    status=dict(phase='online_warmup', state='running', bc_baseline_m=50., rollbacks=0))
    model.set_frozen(True, True)
    model.save(d / 'checkpoints' / 'last.zip')
    runs.append_jsonl(d / 'evals.jsonl', {'step': 0, 'det_clean': True, 'lap2_s': 122.8, 'sto_lap_mean_s': 124.7})
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_OK
    k = seen['callbacks']
    assert k['unfreeze_at'] == 20_000 and k['critic_check_at'] == 20_000 and k['critic_check_done'] is False
    assert k['plateau_history'] == [(0, 122.8, 124.7)]
    assert seen['model'].freeze_actor and seen['model'].freeze_ent_coef
    with pytest.raises(ValueError, match='init_from'):
        T.resume_pipeline(d, [], argv=['x'], init_from=str(tmp_path / 'other.zip'))
    assert T.resume_pipeline(d, [], argv=['x'], init_from=str(tmp_path / 'src.zip')) == T.EXIT_OK
    # strict rule: pre-R1 smoke config (no key) resumes on a dirty tree, a strict config refuses
    dirty = {'rlsetup': dict(CODE['rlsetup'], dirty=True), 'ocd_lap_code_version': CODE['ocd_lap_code_version']}
    monkeypatch.setattr('rlsetup.provenance.code_state', lambda track=None: dirty)
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_DIRTY
    raw = json.loads((d / 'config.json').read_text())
    del raw['strict_provenance']
    raw['preset'] = 'smoke'
    (d / 'config.json').write_text(json.dumps(raw))
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_OK


# --- Amendment 1: optimizer states (C4), buffer preload (C3) ---------------------------------------
def _opt_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    if sa['param_groups'] != sb['param_groups'] or set(sa['state']) != set(sb['state']):
        return False
    return all(th.equal(sa['state'][i][k], sb['state'][i][k]) for i in sa['state'] for k in sa['state'][i])


def test_warm_start_copies_optimizer_states(tmp_path):
    """C4: with load_optimizers (default) the actor / critic / ent_coef Adam states equal the source's."""
    src, path = _save_source(tmp_path, 'src', tiny_cfg(), train=3)
    cfg = _warm_cfg(path, actor_lr=1e-4)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    warm_start(model, path, cfg)
    assert src.actor.optimizer.state_dict()['state']                # the source really has Adam moments
    for name in ('actor', 'critic'):
        assert _opt_equal(getattr(model, name).optimizer, getattr(src, name).optimizer), name
    assert _opt_equal(model.ent_coef_optimizer, src.ent_coef_optimizer)
    assert model.ent_coef_optimizer.param_groups[0]['params'][0] is model.log_ent_coef
    assert model.actor_lr == 1e-4 and model.freeze_actor


def test_preload_buffer_relabels_and_cuts(tmp_path):
    from rlsetup.driving.agent import preload_buffer, default_init_buffer
    src, path = _save_source(tmp_path, 'src', tiny_cfg(), buffer=True)
    assert default_init_buffer(path) == str(path.parents[1].resolve() / 'replay_buffer.pkl')
    cfg = _warm_cfg(path)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    model.min_train_transitions = 5_120
    r_old = src.replay_buffer.rewards[:600].copy()
    info = preload_buffer(model, default_init_buffer(path), path, cfg)
    rb = model.replay_buffer
    assert rb.pos == 600 and info['transitions'] == 600 and info['cut_row'] == 599 and rb.dones[599, 0] == 1.
    assert info['relabel']['rows'] == 600 and model.min_train_transitions == 0 and len(info['sha256']) == 64
    assert not np.array_equal(rb.rewards[:600], r_old)
    norelabel = build_model(_warm_cfg(path, relabel=False), DummyVecEnv([TinyEnv]))
    info = preload_buffer(norelabel, default_init_buffer(path), path, _warm_cfg(path, relabel=False))
    assert info['relabel'] is None and np.array_equal(norelabel.replay_buffer.rewards[:599], r_old[:599])


def test_preload_buffer_refuses_mismatches(tmp_path):
    """C3: buffer_size, n_envs, gamma, observation shape; missing file."""
    from rlsetup.driving.agent import preload_buffer
    c = tiny_cfg()
    variants = {'buffer_size': dict(cfg=dataclasses.replace(c, agent=dataclasses.replace(c.agent, buffer_size=4_000))),
                'n_envs': dict(cfg=c, n_envs=2),
                'gamma': dict(cfg=dataclasses.replace(c, env=dataclasses.replace(
                    c.env, reward=dataclasses.replace(c.env.reward, gamma=.99)))),
                'obs_shape': dict(cfg=c, env_cls=TinyEnv160)}
    for key, kw in variants.items():
        _, path = _save_source(tmp_path, key, kw.pop('cfg'), buffer=True, **kw)
        cfg = _warm_cfg(path)
        model = build_model(cfg, DummyVecEnv([TinyEnv]))
        with pytest.raises(InitMismatch, match=key):
            preload_buffer(model, path.parents[1] / 'replay_buffer.pkl', path, cfg)
    with pytest.raises(InitMismatch, match='missing'):
        preload_buffer(model, tmp_path / 'nope.pkl', path, cfg)


# --- QA review of 475bbe2 (docs/handoff/R1_qa_review.md) ------------------------------------------
def test_missing_init_files_refused_before_run_dir(monkeypatch, tmp_path):
    """QA m2: a missing init_from / config.json / init_buffer exits 6 before the run folder exists."""
    from rlsetup import provenance
    from rlsetup.driving import train as T
    monkeypatch.setattr(provenance, 'git_state', lambda path: {'dirty': False})
    monkeypatch.setattr(runs, 'make_run_dir', lambda *a, **k: pytest.fail('run dir created'))
    ck = tmp_path / 'src' / 'seed0' / 'checkpoints'
    ck.mkdir(parents=True)
    zip_, conf, buf = ck / 'x.zip', ck.parent / 'config.json', ck.parent / 'replay_buffer.pkl'
    cfg = dataclasses.replace(PRESETS['r1_warm_smoke'](), init_from=str(zip_))
    for present in ((), (zip_,), (zip_, conf)):
        for f in present:
            f.write_text('')
        assert T.run_pipeline(cfg, 'r') == T.EXIT_INIT_MISMATCH
    buf.write_text('')
    other = dataclasses.replace(cfg, warm=dataclasses.replace(cfg.warm, init_buffer=str(tmp_path / 'none.pkl')))
    assert T.run_pipeline(other, 'r') == T.EXIT_INIT_MISMATCH


def test_critic_check_eval_scheduled_at_unfreeze(tmp_path):
    """QA M1: interval 50k, critic_check_at 20k -> an extra eval record at exactly 20,000 with the
    check; afterwards the 50k grid continues; no second check."""
    from rlsetup.driving.callbacks import DrivingEvalCallback
    model = make_model(fill=0)
    q = {'q_min_minus_mc_nstep': 2., 'mc_nstep_mean': 200.}
    cb = DrivingEvalCallback(lambda m, n: {'det_clean': True, 'lap2_s': 123., 'q_vs_mc': q}, 50_000,
                             tmp_path / 'evals.jsonl', tmp_path, critic_check_at=20_000, critic_check_tol=.08)
    cb.init_callback(model)
    model.num_timesteps = 0
    cb.on_training_start({}, {})
    for n in range(4, 120_001, 4):
        model.num_timesteps = n
        cb.on_step()
    recs = runs.read_jsonl(tmp_path / 'evals.jsonl')
    assert [r['step'] for r in recs] == [0, 20_000, 50_000, 100_000]
    assert [('critic_check' in r) for r in recs] == [False, True, False, False]
    assert recs[1]['critic_check'] == pytest.approx(.01)


def test_nan_critic_check_is_not_converged(tmp_path):
    """QA m1: a non-finite critic check is None and raises critic_not_converged."""
    from rlsetup.driving.callbacks import DrivingEvalCallback
    assert critic_check_value({'q_min_minus_mc_nstep': math.nan, 'mc_nstep_mean': 200.}) is None
    assert critic_check_value({'q_min_minus_mc_nstep': 1., 'mc_nstep_mean': math.inf}) is None
    model = make_model(fill=0)
    events = []
    cb = DrivingEvalCallback(lambda m, n: {'q_vs_mc': {'q_min_minus_mc_nstep': math.nan, 'mc_nstep_mean': 1.}},
                             50_000, tmp_path / 'evals.jsonl', tmp_path, critic_check_at=0, critic_check_tol=.08,
                             on_event=lambda k, **d: events.append((k, d)))
    cb.init_callback(model)
    model.num_timesteps = 0
    cb.on_training_start({}, {})
    assert events == [('critic_not_converged', {'step': 0, 'critic_check': None, 'tol': .08})]


@pytest.mark.parametrize('nan_at', [5, 15])        # physics step: tick 1 / tick 2
def test_nonfinite_under_time_pbrs(make_env, monkeypatch, nan_at):
    """QA m5a: terminated 'nonfinite', P = 100, Phi' = 0, F = -Phi(s), r = progress - time - P + F,
    obs all zero."""
    env = make_env(EnvConfig(settle_s=0., reward=R1()))
    _, info0 = env.reset(options={'start_s': 4900., 'start_speed': 20.})
    phi_prev = info0['phi']
    calls, orig = [0], env.backend.step

    def step(inputs, steps=1):
        st = orig(inputs, steps)
        calls[0] += 1
        return dict(st, vx=float('nan')) if calls[0] == nan_at else st
    monkeypatch.setattr(env.backend, 'step', step)
    obs, r, term, trunc, info = env.step(np.array([0., .3, -1.], np.float32))
    assert term and not trunc and info['termination'] == 'nonfinite' and info['truncation'] is None
    c = info['step_cost']
    assert c['penalty'] == 100. and info['phi'] == 0. and info['shaping'] == -phi_prev and phi_prev != 0.
    assert r == c['progress'] - c['time'] - 100. + info['shaping']
    assert not obs.any()


class TinyEnvA2(TinyEnv):
    action_space = gym.spaces.Box(-1., 1., (2,), np.float32)


def test_warm_start_refuses_more_mismatches(tmp_path):
    """QA m5b / m7: action space, n_critics, critic LayerNorm / policy class, n_steps, target_entropy."""
    c = tiny_cfg()
    ag = lambda **kw: dataclasses.replace(c, agent=dataclasses.replace(c.agent, **kw))
    cases = {'action space': dict(cfg=c, env_cls=TinyEnvA2), 'n_critics': dict(cfg=ag(n_critics=3)),
             'critic LayerNorm': dict(cfg=ag(critic_layernorm=False)), 'n_steps': dict(cfg=ag(n_steps=1)),
             'target_entropy': dict(cfg=ag(target_entropy=-2.))}
    for msg, kw in cases.items():
        _, path = _save_source(tmp_path, msg.replace(' ', '_'), kw.pop('cfg'), **kw)
        cfg = _warm_cfg(path)
        model = build_model(cfg, DummyVecEnv([TinyEnv]))
        with pytest.raises(InitMismatch, match=msg):
            warm_start(model, path, cfg)
