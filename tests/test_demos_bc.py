"""D2 §10: vec env factory, demo collection, buffer prefill, BC."""
import dataclasses
import subprocess
import sys
import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.sac.policies import SACPolicy
from rlsetup.driving import demos as D
from rlsetup.driving.bc import train_bc
from rlsetup.driving.config import EnvConfig
from rlsetup.driving.train_config import DemoDataConfig, BCConfig
from rlsetup.driving.vec import make_env, track_sha256
from rlsetup.provenance import REPO_ROOT

SHORT = EnvConfig(max_episode_s=2.)
TINY = DemoDataConfig(n_transitions=100, workers=1)   # 2 episodes of 50 decisions


# --- vec.py ---------------------------------------------------------------------------------
def test_vec_module_does_not_import_torch():
    code = (f'import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); import rlsetup.driving.vec; '
            'assert "torch" not in sys.modules, "torch imported"; '
            'assert "stable_baselines3" not in sys.modules')
    r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_worker_with_wrong_track_sha_raises(track, qss):
    env = make_env(SHORT.to_dict(), 0, track_sha256(track), qss)
    assert env.cfg == SHORT
    with pytest.raises(RuntimeError, match='sha256'):
        make_env(SHORT.to_dict(), 1, '0' * 64, qss)


# --- demos ----------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def tiny_demos(tmp_path_factory, track, qss):
    out = tmp_path_factory.mktemp('demos')
    path, meta = D.collect_demos(SHORT, TINY, qss, track_sha256(track), out, log=lambda *_: None)
    return path, meta, D.load_demos(path)


def test_demo_schema(tiny_demos):
    path, meta, d = tiny_demos
    assert path.name == f'demos_{meta["key"][:12]}.npz'
    assert set(D.ARRAYS) <= set(d)
    E, N = len(d['ep_start_tr']), len(d['actions'])
    assert E == 2 and N == 100 and d['obs'].shape == (N + E, 159) and d['obs'].dtype == np.float32
    assert d['actions'].shape == d['actions_clean'].shape == (N, 3) and d['actions'].dtype == np.float32
    assert np.all(np.abs(d['actions']) <= 1)
    np.testing.assert_array_equal(d['ep_start_tr'], [0, 50])
    np.testing.assert_array_equal(d['ep_start_obs'], [0, 51])
    ends = d['ep_start_tr'] + d['ep_len'] - 1
    done = d['terminated'] | d['truncated']
    assert done[ends].all() and done.sum() == E                   # done only at episode ends
    assert list(d['ep_cause']) == ['time_limit', 'time_limit']
    assert np.all((.35 <= d['f_demo']) & (d['f_demo'] <= .65))
    s = meta['summary']
    assert s['episodes'] == 2 and s['transitions'] == 100 and s['clean_fraction'] == 1.
    assert 0 < s['mean_speed_fraction'] < 1.5 and meta['gate_passed']
    assert meta['key_inputs']['demos'] == {k: v for k, v in dataclasses.asdict(TINY).items() if k != 'workers'}
    # noise really is ~0.1 per dim
    assert 0.05 < np.std(d['actions'][:, 0] - d['actions_clean'][:, 0]) < 0.2


def test_demo_replay_matches_env(tiny_demos, make_env):
    """Replaying the recorded actions reproduces every observation (incl. the true terminal
    observation) and every reward exactly."""
    _, _, d = tiny_demos
    env = make_env(SHORT)
    for e in range(2):
        obs, _ = env.reset(seed=TINY.seed + e)
        o0, t0 = d['ep_start_obs'][e], d['ep_start_tr'][e]
        np.testing.assert_array_equal(obs, d['obs'][o0])
        for t in range(d['ep_len'][e]):
            obs, r, term, trunc, _ = env.step(d['actions'][t0 + t])
            np.testing.assert_array_equal(obs, d['obs'][o0 + t + 1])
            assert r == d['rewards'][t0 + t]
            assert (term, trunc) == (d['terminated'][t0 + t], d['truncated'][t0 + t])
        assert term or trunc


def test_demo_collection_is_byte_identical(tmp_path, tiny_demos, track, qss):
    path2, meta2 = D.collect_demos(SHORT, TINY, qss, track_sha256(track), tmp_path, log=lambda *_: None)
    _, meta, d = tiny_demos
    d2 = D.load_demos(path2)
    for k in D.ARRAYS:
        assert d[k].tobytes() == d2[k].tobytes(), k
    assert meta2['key'] == meta['key'] and meta2['npz_sha256'] == meta['npz_sha256']
    # cache hit: no recollection, same file
    path3, meta3 = D.collect_demos(SHORT, TINY, qss, track_sha256(track), tmp_path, log=lambda *_: None)
    assert path3 == path2 and meta3['created_utc'] == meta2['created_utc']


def test_demo_noise_default_and_fallback(monkeypatch):
    """Lead decision 26 Oct: steering noise 0.05 by default; the fallback (0.025) runs only if
    0.05 fails the gate; a failing fallback raises."""
    c = DemoDataConfig()
    assert c.noise_std == (.05, .1, .1) and c.fallback_steer_noise == .025
    clean = {}
    calls = []

    def fake_collect(env_cfg, demo_cfg, *a, **k):
        calls.append(demo_cfg.noise_std)
        return 'p', {'summary': {'clean_fraction': clean[demo_cfg.noise_std[0]]}}
    monkeypatch.setattr(D, 'collect_demos', fake_collect)
    run = lambda: D.ensure_demos(None, c, None, None, None, log=lambda *_: None)
    clean.update({.05: .86, .025: .9})
    assert run()[2] == c and calls == [(.05, .1, .1)]                 # passes: fallback is a no-op
    calls.clear()
    clean[.05] = .5
    _, meta, used = run()
    assert used.noise_std == (.025, .1, .1) and calls == [(.05, .1, .1), (.025, .1, .1)]
    assert meta['fallback_from']['clean_fraction'] == .5
    clean[.025] = .7
    with pytest.raises(D.DemoGateFailed):
        run()
    with pytest.raises(D.DemoGateFailed):                              # already at the fallback noise
        D.ensure_demos(None, dataclasses.replace(c, noise_std=(.025, .1, .1)), None, None, None,
                       log=lambda *_: None)


def test_dataset_key_ignores_workers_only(qss):
    prov = {'rlsetup': {'commit': 'a'}, 'ocd_lap_code_version': 'b'}
    k = D.dataset_key(SHORT, TINY, qss, prov)[0]
    assert D.dataset_key(SHORT, dataclasses.replace(TINY, workers=8), qss, prov)[0] == k
    assert D.dataset_key(SHORT, dataclasses.replace(TINY, seed=1), qss, prov)[0] != k
    assert D.dataset_key(EnvConfig(), TINY, qss, prov)[0] != k


def test_dataset_key_uses_demo_source_hashes_not_commit(qss, monkeypatch):
    """Lead decision 26 Oct: the key hashes the demo-relevant rlsetup sources, not the commit."""
    prov = {'rlsetup': {'commit': 'a'}, 'ocd_lap_code_version': 'b'}
    k, d = D.dataset_key(SHORT, TINY, qss, prov)
    assert 'rlsetup_commit' not in d
    assert set(d['rlsetup_demo_sources_sha256']) == {f'rlsetup/driving/{f}' for f in (
        'env.py', 'config.py', 'demo_driver.py', 'reference.py', 'qss_profile.py', 'demos.py')}
    assert all((REPO_ROOT / f).is_file() for f in D.DEMO_SOURCES)
    assert D.dataset_key(SHORT, TINY, qss, {'rlsetup': {'commit': 'z'}, 'ocd_lap_code_version': 'b'})[0] == k
    assert D.dataset_key(SHORT, TINY, qss, {'rlsetup': {'commit': 'a'}, 'ocd_lap_code_version': 'c'})[0] != k
    real = D.demo_sources_sha256()
    monkeypatch.setattr(D, 'demo_sources_sha256', lambda: dict(real, **{'rlsetup/driving/env.py': '0' * 64}))
    assert D.dataset_key(SHORT, TINY, qss, prov)[0] != k


# --- prefill ----------------------------------------------------------------------------------
def synthetic_demos(lengths, terminal):
    """obs[:, 0] = episode id, obs[:, 1] = step index within the episode."""
    obs, act, rew, term, trunc = [], [], [], [], []
    for e, (T, is_term) in enumerate(zip(lengths, terminal)):
        obs += [[e, t] + [0.] * 3 for t in range(T + 1)]
        act += [[e / 100, t / 1000, 0.] for t in range(T)]
        rew += [float(t) for t in range(T)]
        term += [False] * (T - 1) + [is_term]
        trunc += [False] * (T - 1) + [not is_term]
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    return D.with_index({'obs': np.array(obs, np.float32), 'actions': np.array(act, np.float32),
                         'rewards': np.array(rew), 'terminated': np.array(term), 'truncated': np.array(trunc),
                         'ep_start_tr': starts, 'ep_start_obs': starts + np.arange(len(lengths))})


def test_prefill_never_mixes_episodes_and_flags_tails():
    lengths = [40, 7, 25, 60, 13, 33, 52, 9, 18, 30]
    terminal = [True, False, True, True, False, False, True, False, True, True]
    d = synthetic_demos(lengths, terminal)
    obs_space = gym.spaces.Box(-1e3, 1e3, (5,), np.float32)
    rb = NStepReplayBuffer(1000, obs_space, gym.spaces.Box(-1, 1, (3,), np.float32), device='cpu',
                           n_envs=3, n_steps=3, gamma=.99)
    info = D.prefill_replay_buffer(rb, d)
    # columns: eps {0,3,6,9} = 182, {1,4,7} = 29, {2,5,8} = 76 -> T = 29
    assert info['rows'] == 29 and info['transitions'] == 87 and info['dropped'] == sum(lengths) - 87
    assert rb.pos == 29
    # tails: column 0 cut inside ep 0, column 1 ends exactly on ep 7 (truncated), column 2 cut in ep 5
    np.testing.assert_array_equal(rb.dones[28], [1, 1, 1])
    np.testing.assert_array_equal(rb.timeouts[28], [1, 1, 1])
    # true terminations in the middle are not timeouts
    k = np.flatnonzero((rb.observations[:29, 2, 0] == 2) & (rb.observations[:29, 2, 1] == 24))[0]
    assert rb.dones[k, 2] == 1 and rb.timeouts[k, 2] == 0
    np.testing.assert_allclose(rb.actions[:29, :, 0] * 100, rb.observations[:29, :, 0], atol=1e-4)
    # add some online rows (episode 99) after the demos
    for t in range(5):
        rb.add(np.full((3, 5), 99, np.float32), np.full((3, 5), 99, np.float32), np.zeros((3, 3), np.float32),
               np.zeros(3, np.float32), np.zeros(3, np.float32), [{}] * 3)
    np.random.seed(0)
    s = rb.sample(1000)
    o, n = s.observations.numpy(), s.next_observations.numpy()
    assert np.all(o[:, 0] == n[:, 0])                          # never across episodes / into online data
    demo = o[:, 0] != 99
    steps = n[demo, 1] - o[demo, 1]
    assert np.all((steps >= 1) & (steps <= 3))
    # n-step reward = sum gamma^j r_{t+j} with r = step index, until the episode/column end
    exp = [sum(.99 ** j * (t + j) for j in range(int(m))) for t, m in zip(o[demo, 1], steps)]
    np.testing.assert_allclose(s.rewards.numpy()[demo, 0], exp, rtol=1e-5)


# --- BC -----------------------------------------------------------------------------------------
def test_bc_fits_synthetic_tanh_policy():
    rng = np.random.default_rng(0)
    dim, E, T = 8, 40, 250
    W = rng.normal(0, .3, (3, dim))
    x = rng.uniform(-1, 1, (E * (T + 1), dim)).astype(np.float32)
    starts = np.arange(E) * T
    d = D.with_index({'obs': x, 'ep_start_tr': starts, 'ep_start_obs': starts + np.arange(E),
                      'terminated': np.zeros(E * T, bool), 'truncated': np.zeros(E * T, bool)} |
                     {'actions': np.zeros((E * T, 3), np.float32)})
    clean = np.tanh(x[d['obs_index']] @ W.T)
    d['actions'] = np.clip(clean + rng.normal(0, .1, clean.shape), -1, 1).astype(np.float32)
    d['actions_clean'] = clean.astype(np.float32)
    th.manual_seed(0)
    policy = SACPolicy(gym.spaces.Box(-10, 10, (dim,), np.float32), gym.spaces.Box(-1, 1, (3,), np.float32),
                       lambda _: 3e-4, net_arch=[64, 64])
    rep = train_bc(policy.actor, d, BCConfig(learning_rate=1e-3, batch_size=256, max_epochs=30),
                   seed=0, log=lambda *_: None)
    assert max(rep['val_mse_clean_per_dim']) < .05
    assert all(.05 < s < .3 for s in rep['mean_sigma_per_dim'])
    assert rep['history'][rep['best_epoch']]['val_nll'] == rep['best_val_nll'] < rep['history'][0]['val_nll']
    assert set(rep['val_episodes']).isdisjoint(rep['train_episodes'])
    assert len(rep['val_episodes']) == 4


def synthetic_bc_data(E, T, dim, rng):
    x = rng.uniform(-1, 1, (E * (T + 1), dim)).astype(np.float32)
    starts = np.arange(E) * T
    return D.with_index({'obs': x, 'ep_start_tr': starts, 'ep_start_obs': starts + np.arange(E),
                         'terminated': np.zeros(E * T, bool), 'truncated': np.zeros(E * T, bool),
                         'actions': np.zeros((E * T, 3), np.float32), 'episode': np.repeat(np.arange(E), T)})


def small_policy(dim):
    th.manual_seed(0)
    return SACPolicy(gym.spaces.Box(-10, 10, (dim,), np.float32), gym.spaces.Box(-1, 1, (3,), np.float32),
                     lambda _: 3e-4, net_arch=[64, 64])


def test_bc_keeps_best_epoch_and_stops_after_patience():
    """QA mutants B1 (best weights not restored) and B3 (no early stop): validation actions are
    the opposite of the training actions, so every epoch makes the validation NLL worse; the best
    epoch is 0, training stops after `patience` epochs and the actor is the initial one."""
    from rlsetup.driving.bc import _split
    rng = np.random.default_rng(0)
    E, T, dim = 20, 200, 8
    cfg = BCConfig(learning_rate=1e-3, batch_size=256, max_epochs=30, patience=3)
    d = synthetic_bc_data(E, T, dim, rng)
    _, val_eps = _split(E, cfg)
    d['actions'][:] = np.where(np.isin(d['episode'], val_eps), -.5, .5)[:, None]
    policy = small_policy(dim)
    init = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    rep = train_bc(policy.actor, d, cfg, seed=0, log=lambda *_: None)
    vals = [h['val_nll'] for h in rep['history']]
    assert rep['best_epoch'] == 0 and rep['epochs_run'] == cfg.patience and min(vals[1:]) > vals[0]
    assert all(th.equal(init[k], v) for k, v in policy.actor.state_dict().items())


def test_bc_loss_finite_at_action_limits():
    """QA mutant B2: actions at exactly +-1 are clipped to +-action_clip before the log-likelihood
    (same loss as at +-0.995; finite training history)."""
    from rlsetup.driving.bc import _nll
    rng = np.random.default_rng(1)
    E, T, dim = 10, 100, 8
    d = synthetic_bc_data(E, T, dim, rng)
    d['actions'][:] = np.where(rng.uniform(size=(E * T, 3)) < .5, -1., 1.)
    policy = small_policy(dim)
    obs = th.as_tensor(d['obs'][d['obs_index']])
    act = th.as_tensor(d['actions'])
    with th.no_grad():
        at_limit, clipped = _nll(policy.actor, obs, act, .995), _nll(policy.actor, obs, act * .995, .995)
    assert th.isfinite(at_limit) and float(at_limit) == float(clipped)
    rep = train_bc(policy.actor, d, BCConfig(max_epochs=2, batch_size=256), seed=0, log=lambda *_: None)
    assert all(np.isfinite(h['val_nll']) for h in rep['history'])
