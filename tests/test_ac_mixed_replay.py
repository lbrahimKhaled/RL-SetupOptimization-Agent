from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.buffers import ReplayBuffer

from rlsetup.assetto.learn import mixed_replay_metadata
from rlsetup.assetto.mixed_replay import mix_replay
from test_ac_learn import bundle, chunk


def buffers():
    obs = gym.spaces.Box(-10., 10., (2,), dtype=np.float32)
    act = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
    ac = ReplayBuffer(6, obs, act, n_envs=1)
    for value in (8., 9.):
        ac.add(np.array([[value, 0.]]), np.array([[value, 1.]]), np.array([[0.]]),
               np.array([value]), np.array([0.]), [{}])
    source = SimpleNamespace(observation_space=obs, action_space=act, optimize_memory_usage=False,
                             gamma=.992, handle_timeout_termination=True, buffer_size=4, pos=0,
                             full=True, n_envs=2, n_steps=3, bonus_weight=1.,
                             warmup=np.array([[True, False], [False, False], [False, False], [False, False]]),
                             bonuses=np.zeros((4, 2), np.float32))
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    source.observations = np.stack((values, values), axis=-1)
    source.next_observations = source.observations + 1
    source.actions = np.zeros((4, 2, 1), np.float32)
    source.rewards = values.copy()
    source.dones = np.zeros((4, 2), np.float32)
    source.timeouts = np.zeros((4, 2), np.float32)
    source.dones[2, 0] = source.timeouts[2, 0] = 1
    return ac, source


def test_mixed_ring_preserves_ac_and_overwrites_ocd_first():
    ac, source = buffers()
    mixed, details = mix_replay(ac, source, seed=42)
    assert details['initial_ocd_count'] == 4 and details['initial_ac_count'] == 2
    np.testing.assert_array_equal(mixed.rewards[-2:, 0], [8, 9])
    assert 0 not in mixed.rewards[:4, 0]  # excluded warmup cell
    assert mixed.pos == 0 and mixed.full
    assert np.all(mixed.timeouts <= mixed.dones)
    timeout_index = np.flatnonzero(mixed.timeouts[:, 0])[0]
    assert mixed.rewards[timeout_index, 0] == 4. and mixed.dones[timeout_index, 0] == 1
    assert mixed._get_samples(np.array([timeout_index])).dones.item() == 0
    assert mixed_replay_metadata(mixed, {})['ocd_count'] == 4
    mixed.domain[mixed.pos] = 2  # ACLearner.import_chunk's domain write
    mixed.add(np.array([[1., 0.]]), np.array([[1., 1.]]), np.array([[0.]]),
              np.array([1.]), np.array([0.]), [{}])
    assert mixed_replay_metadata(mixed, {})['ac_count'] == 3
    np.testing.assert_array_equal(mixed.rewards[-2:, 0], [8, 9])


def test_full_wrapped_ac_ring_is_chronological():
    ac, source = buffers()
    for value in (1., 2., 3., 4., 5., 6.):
        ac.add(np.array([[value, 0.]]), np.array([[value, 1.]]), np.array([[0.]]),
               np.array([value]), np.array([0.]), [{}])
    mixed, details = mix_replay(ac, source)
    assert details['initial_ocd_count'] == 0
    np.testing.assert_array_equal(mixed.rewards[:, 0], [1, 2, 3, 4, 5, 6])
    assert type(mixed) is ReplayBuffer


def test_mixed_rejects_bonus_timeout_and_warmup_shortage():
    ac, source = buffers()
    source.bonuses[:] = 1
    with pytest.raises(ValueError, match='bonus'):
        mix_replay(ac, source)
    source.bonuses[:] = 0
    source.timeouts[:] = 1
    with pytest.raises(ValueError, match='timeout without done'):
        mix_replay(ac, source)
    source.timeouts[:] = 0
    source.warmup[:] = True
    with pytest.raises(ValueError, match='eligible'):
        mix_replay(ac, source)


def test_fork_genesis_resume_and_next_cycle(bundle, tmp_path):
    import json
    import pickle
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    from rlsetup.assetto import distributed as d
    from rlsetup.assetto.actor import _encoded, _manifest, _sha
    from rlsetup.assetto.learn import ACLearner
    from rlsetup.assetto.mixed_replay import make_fork

    config = json.loads((bundle/'config.json').read_text())
    config['env']['corner_bonus'] = dict(mode='entry_exit', bonus_in=0., bonus_out=0.)
    config['n_envs'] = 4
    config['agent'].update(n_steps=3, buffer_size=32)
    cfg = _encoded(config)
    (bundle/'config.json').write_bytes(cfg)
    (bundle/'manifest.json').write_bytes(_encoded(_manifest(
        config, {'checkpoint_name':'best.zip','checkpoint_sha256':'a'*64,'num_timesteps':100},
        _sha(cfg), _sha((bundle/'actor.pt').read_bytes()))))
    args = SimpleNamespace(work_dir=tmp_path/'parent', actor_bundle=bundle, batch_size=4,
                           buffer_size=32, critic_updates=1, updates=1, max_cycles=None, max_no_update=20)
    server = d.LearnerServer(args)
    server.process(dict(op='chunk', cycle=1), chunk(tmp_path/'parent.npz', bundle).read_bytes())
    parent = args.work_dir/'cycle-00000001'/'learner'
    src = SimpleNamespace(observation_space=server._learner[1].model.observation_space,
                          action_space=server._learner[1].model.action_space, optimize_memory_usage=False,
                          gamma=.992, handle_timeout_termination=True, buffer_size=8, pos=3, full=True,
                          n_envs=4, n_steps=3, bonus_weight=1., warmup=np.zeros((8,4), bool),
                          bonuses=np.zeros((8,4), np.float32), observations=np.zeros((8,4,159), np.float32),
                          next_observations=np.zeros((8,4,159), np.float32), actions=np.zeros((8,4,3), np.float32),
                          rewards=np.arange(32, dtype=np.float32).reshape(8,4), dones=np.zeros((8,4), np.float32),
                          timeouts=np.zeros((8,4), np.float32))
    src.warmup[1, 0] = True
    source = tmp_path/'source.pkl'
    src.n_steps = 2
    source.write_bytes(pickle.dumps(src))
    with pytest.raises(ValueError, match='training configuration'):
        make_fork(args.work_dir, tmp_path/'bad-fork', source, bundle/'config.json')
    assert not (tmp_path/'bad-fork').exists()
    src.n_steps = 3
    source.write_bytes(pickle.dumps(src))
    result = make_fork(args.work_dir, tmp_path/'fork', source, bundle/'config.json')
    assert result['cycle'] == 1 and result['initial_ocd_count'] == 24 and result['initial_ac_count'] == 8
    fork = tmp_path/'fork'
    original = ACLearner.resume(parent, parent/'actor', server.config)
    converted = ACLearner.resume(fork/'bootstrap'/'learner', fork/'bootstrap'/'learner'/'actor', server.config)
    for key, value in original.model.policy.state_dict().items():
        assert torch.equal(value, converted.model.policy.state_dict()[key])
    assert torch.equal(original.model.log_ent_coef, converted.model.log_ent_coef)
    assert original.model._n_updates == converted.model._n_updates
    def equal_state(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: equal_state(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b): equal_state(x, y)
        elif torch.is_tensor(a):
            assert torch.equal(a, b)
        else:
            assert a == b
    for attr in ('actor', 'critic'):
        equal_state(getattr(original.model, attr).optimizer.state_dict(),
                    getattr(converted.model, attr).optimizer.state_dict())
    equal_state(original.model.ent_coef_optimizer.state_dict(), converted.model.ent_coef_optimizer.state_dict())
    for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
        np.testing.assert_array_equal(getattr(original.model.replay_buffer, name)[:8],
                                      getattr(converted.model.replay_buffer, name)[-8:])
    assert json.loads((parent/'rng.json').read_text()) == json.loads((fork/'bootstrap'/'learner'/'rng.json').read_text())
    assert torch.equal(torch.load(parent/'rng.pt', weights_only=True), torch.load(fork/'bootstrap'/'learner'/'rng.pt', weights_only=True))
    assert converted.imported == original.imported
    args.work_dir = fork
    resumed = d.LearnerServer(args)
    hello, _ = resumed.process(dict(op='hello'), b'')
    assert hello['cycle'] == 1
    data = chunk(tmp_path/'next.npz', fork/'bootstrap'/'learner'/'actor').read_bytes()
    response, actor = resumed.process(dict(op='chunk', cycle=2), data)
    assert response['replay_ac'] == 16 and response['replay_ocd'] == 16
    assert resumed.process(dict(op='chunk', cycle=2), data) == (response, actor)
    restarted = d.LearnerServer(args)
    assert restarted.process(dict(op='chunk', cycle=2), data) == (response, actor)
    assert restarted.state['cycle'] == 2
    fresh = chunk(tmp_path/'empty-src.npz', fork/'cycle-00000002'/'learner'/'actor')
    with np.load(fresh, allow_pickle=False) as archive:
        payload = {key: (archive[key] if key == 'metadata_json' else archive[key][:0]) for key in archive.files}
    empty = tmp_path/'empty.npz'
    np.savez(empty, **payload)
    after_restart, _ = restarted.process(dict(op='chunk', cycle=3), empty.read_bytes())
    assert not after_restart['trained']
    assert after_restart['replay_ac'] == 16 and after_restart['replay_ocd'] == 16
    saved_manifest = fork/'bootstrap'/'learner'/'manifest.json'
    original_manifest = saved_manifest.read_text()
    incomplete = json.loads(original_manifest)
    del incomplete['mixed_replay']
    saved_manifest.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match='lack provenance'):
        ACLearner.resume(fork/'bootstrap'/'learner', fork/'bootstrap'/'learner'/'actor', server.config)
    saved_manifest.write_text(original_manifest)
    genesis = fork/'genesis.json'
    document = json.loads(genesis.read_text())
    document['state']['carry'] = 99
    genesis.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='genesis'):
        d.LearnerServer(args)
