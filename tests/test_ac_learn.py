"""Offline learner contract and real small SAC update/export/resume checks."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from rlsetup.assetto.actor import (ActorRuntime, MANIFEST_FILE, SOURCE_OBS, _Actor, _encoded,
                                  _manifest, _sha, source_contract_sha256)
from rlsetup.assetto.learn import ACLearner, LearnerConfig, read_chunk, replay_identity


@pytest.fixture
def bundle(tmp_path):
    config = {'env': {'car': 'sf70h_tum', 'track': 'Catalunya', 'setup': 'bcn_best',
                      'reference': 'path', 'obs_gear': False, 'pedal_deadzone': True,
                      'action_repeat': 2, 'obs': asdict(SOURCE_OBS)},
              'agent': {'net_arch': [256]*3}, 'seed': 0, 'preset': 'bcn'}
    directory = tmp_path / 'source'
    directory.mkdir()
    cfg = _encoded(config)
    (directory / 'config.json').write_bytes(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        actor = _Actor()
    torch.save(actor.state_dict(), directory / 'actor.pt')
    weights = (directory / 'actor.pt').read_bytes()
    manifest = _manifest(config, {'checkpoint_name': 'best.zip', 'checkpoint_sha256': 'a'*64,
                                 'num_timesteps': 100}, _sha(cfg), _sha(weights))
    (directory / MANIFEST_FILE).write_bytes(_encoded(manifest))
    return directory


def chunk(path, bundle, **changes):
    runtime = ActorRuntime(bundle)
    n = 8
    observations = np.random.default_rng(12).normal(0, .1, (n+1, 159)).astype(np.float32)
    arrays = {'obs': observations[:-1], 'next_obs': observations[1:],
              'actions': np.zeros((n, 3), np.float32), 'rewards': np.ones(n, np.float32),
              'terminated': np.zeros(n, bool), 'truncated': np.zeros(n, bool),
              'dt': np.full(n, .04, np.float64), 'episode_id': np.zeros(n, np.int64),
              'seq': np.arange(n, dtype=np.int64)}
    arrays['truncated'][-1] = True
    metadata = {'schema_version': 1, 'actor_manifest_sha256': _sha((bundle / MANIFEST_FILE).read_bytes()),
                'source_contract_sha256': source_contract_sha256(runtime.manifest),
                'source_config_sha256': runtime.manifest['source']['config_sha256'],
                'target': {'car_name': 'ks_ferrari_sf70h', 'track_name': 'ks_barcelona',
                           'track_configuration': 'layout_gp', 'setup_id': 'fixed'},
                'reference': 'centerline', 'reward_mode': 'time_pbrs', 'gamma': .992,
                'n_steps': 1, 'actor_fixed': True, 'min_dt_s': .02, 'max_dt_s': .08,
                'stop_reason': 'completed_lap'}
    for key, value in changes.items():
        if key in arrays:
            arrays[key] = value
        else:
            metadata[key] = value
    np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
    return path


def test_real_updates_actor_only_export_and_resume(bundle, tmp_path):
    config = LearnerConfig(batch_size=4, buffer_size=32)
    learner = ACLearner(bundle, config)
    runtime = ActorRuntime(bundle)
    assert all(torch.equal(v, learner.model.actor.state_dict()[k]) for k, v in runtime._actor.state_dict().items())
    source_critic = {k: v.clone() for k, v in learner.model.critic.state_dict().items()}
    other = ACLearner(bundle, LearnerConfig(batch_size=4, buffer_size=32, seed=4))
    assert any(not torch.equal(v, other.model.critic.state_dict()[k]) for k, v in source_critic.items())
    path = chunk(tmp_path / 'chunk.npz', bundle)
    assert learner.import_chunk(path)
    assert not learner.import_chunk(path)
    before = {k: v.clone() for k, v in learner.model.actor.state_dict().items()}
    alpha = learner.model.log_ent_coef.clone()
    learner.train(2, 0)
    assert all(torch.equal(v, learner.model.actor.state_dict()[k]) for k, v in before.items())
    assert torch.equal(alpha, learner.model.log_ent_coef)
    learner.train(0, 2)
    assert any(not torch.equal(v, learner.model.actor.state_dict()[k]) for k, v in before.items())
    saved = learner.save(tmp_path / 'state')
    updated = ActorRuntime(saved / 'actor')
    obs = np.zeros(159, np.float32)
    np.testing.assert_allclose(updated.predict(obs), learner.model.predict(obs, deterministic=True)[0], atol=1e-7)
    assert len(updated._actor.state_dict()) == 10
    assert updated.manifest['training']['file'] == 'training.json'
    resumed = ACLearner.resume(saved, saved / 'actor', config)
    assert not resumed.import_chunk(path)
    assert resumed.model.buffer_transitions() == 8
    resumed.import_chunk(chunk(tmp_path / 'new.npz', saved / 'actor'))
    assert resumed.model.buffer_transitions() == 16
    resumed.train(1, 1)
    resumed.save(tmp_path / 'next_state')
    with pytest.raises(FileExistsError):
        learner.save(saved)


def test_terminated_truncated_bootstrap(bundle, tmp_path):
    learner = ACLearner(bundle, LearnerConfig(batch_size=2, buffer_size=32))
    done = np.zeros(8, bool)
    done[3] = True
    episode = np.array([0]*4+[1]*4, np.int64)
    learner.import_chunk(chunk(tmp_path / 'chunk.npz', bundle, terminated=done, episode_id=episode))
    rb = learner.model.replay_buffer
    assert rb.dones[3, 0] == 1 and rb.timeouts[3, 0] == 0
    assert rb.dones[7, 0] == 1 and rb.timeouts[7, 0] == 1
    sample = rb._get_samples(np.array([3, 7]))
    np.testing.assert_array_equal(sample.dones.cpu().numpy().ravel(), [1, 0])


@pytest.mark.parametrize('change', [
    {'gamma': .99}, {'reward_mode': 'ref_tracking'}, {'actor_fixed': False},
    {'actions': np.ones((8, 3), np.float64)}, {'dt': np.full(8, .1, np.float64)},
    {'obs': np.full((8, 159), np.nan, np.float32)}, {'seq': np.arange(8, dtype=np.int64)*2},
    {'episode_id': np.arange(8, dtype=np.int64)}, {'truncated': np.zeros(8, bool)},
    {'source_contract_sha256': '0'*64}, {'source_config_sha256': '0'*64},
])
def test_bad_chunks_rejected_without_replay_mutation(bundle, tmp_path, change):
    learner = ACLearner(bundle, LearnerConfig(batch_size=2, buffer_size=32))
    with pytest.raises(ValueError):
        learner.import_chunk(chunk(tmp_path / 'bad.npz', bundle, **change))
    assert learner.model.buffer_transitions() == 0


def test_resume_validates_hash_before_pickle(bundle, tmp_path, monkeypatch):
    learner = ACLearner(bundle, LearnerConfig(batch_size=2, buffer_size=32))
    learner.import_chunk(chunk(tmp_path / 'chunk.npz', bundle))
    saved = learner.save(tmp_path / 'state')
    (saved / 'replay.pkl').write_bytes(b'corrupt')
    monkeypatch.setattr('rlsetup.assetto.learn.StagedSAC.load', lambda *a, **k: pytest.fail('pickle loader reached'))
    with pytest.raises(ValueError, match='corrupted'):
        ACLearner.resume(saved, bundle, learner.config)


def test_driver_chunk_writer_to_learner(bundle, tmp_path):
    from rlsetup.assetto.driving import write_chunk
    raw = chunk(tmp_path / 'raw.npz', bundle)
    with np.load(raw, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        rows = [{key: data[key][i] for key in data.files if key != 'metadata_json'} for i in range(8)]
    for i, row in enumerate(rows):
        row.update(timestamp_ac=i*.04, next_timestamp_ac=(i+1)*.04, command=[0., -1., -1.],
                   desired_road_angle=0., executed_road_angle=0.)
    actual = tmp_path / 'driver.npz'
    write_chunk(actual, rows, metadata)
    learner = ACLearner(bundle, LearnerConfig(batch_size=4, buffer_size=32))
    assert learner.import_chunk(actual)
    learner.train(1, 1)
    assert learner.model.buffer_transitions() == 8


def test_fast_lane_chunks_require_geometry_and_cannot_mix_reference_or_geometry(bundle, tmp_path):
    config = LearnerConfig(batch_size=4, buffer_size=32)
    learner = ACLearner(bundle, config)
    fast = dict(reference='fast_lane', reference_geometry_sha256='a'*64, reference_length_m=4500.)
    with pytest.raises(ValueError, match='provenance'):
        learner.import_chunk(chunk(tmp_path/'missing.npz', bundle, reference='fast_lane'))
    assert learner.import_chunk(chunk(tmp_path/'fast.npz', bundle, **fast))
    for name, change in [('center', dict(reference='centerline')),
                         ('changed', dict(fast, reference_geometry_sha256='b'*64))]:
        with pytest.raises(ValueError, match='differs'):
            learner.import_chunk(chunk(tmp_path/f'{name}.npz', bundle, **change))
    assert learner.model.buffer_transitions() == 8
    saved = learner.save(tmp_path/'fast_state')
    resumed = ACLearner.resume(saved, saved/'actor', config)
    with pytest.raises(ValueError, match='differs'):
        resumed.import_chunk(chunk(tmp_path/'center_after_resume.npz', saved/'actor'))
    assert resumed.model.buffer_transitions() == 8


def test_ocd_path_chunks_need_geometry_provenance_and_do_not_mix(bundle, tmp_path):
    runtime = ActorRuntime(bundle)
    args = (_sha((bundle / MANIFEST_FILE).read_bytes()), source_contract_sha256(runtime.manifest))
    with pytest.raises(ValueError, match='ocd_path reference geometry provenance'):
        read_chunk(chunk(tmp_path / 'a.npz', bundle, reference='ocd_path'), *args)
    good = chunk(tmp_path / 'b.npz', bundle, reference='ocd_path',
                 reference_geometry_sha256='b'*64, reference_length_m=4602.)
    _, metadata, _ = read_chunk(good, *args)
    assert replay_identity(metadata)['reference_geometry_sha256'] == 'b'*64
    learner = ACLearner(bundle, LearnerConfig(batch_size=4, buffer_size=32))
    learner.import_chunk(good)
    other = chunk(tmp_path / 'c.npz', bundle, reference='fast_lane',
                  reference_geometry_sha256='b'*64, reference_length_m=4602.)
    with pytest.raises(ValueError, match='differs'):
        learner.import_chunk(other)


def _checkpoint_bundle(bundle, tmp_path, *, actor_changed=False):
    """Fake SB3 checkpoint holding the bundle actor + distinct critics; a bundle that names it."""
    import io, zipfile
    from rlsetup.assetto.learn import build_learner
    model = build_learner(bundle, LearnerConfig(batch_size=4, buffer_size=32))
    state = model.policy.state_dict()
    with torch.no_grad():
        for key in state:
            if key.startswith(('critic.', 'critic_target.')) or (actor_changed and key.startswith('actor.')):
                state[key] = state[key] + .5
    def member(value):
        buffer = io.BytesIO(); torch.save(value, buffer); return buffer.getvalue()
    checkpoint = tmp_path / 'best.zip'
    with zipfile.ZipFile(checkpoint, 'w') as archive:
        archive.writestr('policy.pth', member(state))
        archive.writestr('pytorch_variables.pth', member({'log_ent_coef': torch.tensor([-2.7698581])}))
    named = tmp_path / 'named'
    named.mkdir()
    for name in ('config.json', 'actor.pt'):
        (named / name).write_bytes((bundle / name).read_bytes())
    config = json.loads((bundle / 'config.json').read_text())
    manifest = _manifest(config, {'checkpoint_name': 'best.zip', 'checkpoint_sha256': _sha(checkpoint.read_bytes()),
                                  'num_timesteps': 100}, _sha((bundle / 'config.json').read_bytes()),
                         _sha((bundle / 'actor.pt').read_bytes()))
    (named / MANIFEST_FILE).write_bytes(_encoded(manifest))
    return checkpoint, named, state


def test_critic_init_copies_source_critics_and_alpha_only_from_actor_checkpoint(bundle, tmp_path):
    from rlsetup.assetto.learn import build_learner
    checkpoint, named, state = _checkpoint_bundle(bundle, tmp_path)
    model = build_learner(named, LearnerConfig(batch_size=4, buffer_size=32, critic_init=str(checkpoint)))
    loaded = model.policy.state_dict()
    for key, value in state.items():
        assert torch.equal(loaded[key], value), key  # critics copied, actor unchanged
    assert float(model.log_ent_coef.exp()) == pytest.approx(.0627, abs=1e-4)
    fresh = build_learner(named, LearnerConfig(batch_size=4, buffer_size=32))
    assert float(fresh.log_ent_coef.exp()) == pytest.approx(.1)
    with pytest.raises(ValueError, match='not the actor bundle source'):
        build_learner(bundle, LearnerConfig(batch_size=4, buffer_size=32, critic_init=str(checkpoint)))
    other = tmp_path / 'other'
    other.mkdir()
    checkpoint2, named2, _ = _checkpoint_bundle(bundle, other, actor_changed=True)
    with pytest.raises(ValueError, match='actor differs'):
        build_learner(named2, LearnerConfig(batch_size=4, buffer_size=32, critic_init=str(checkpoint2)))
    # Resume keeps the trained critics; the saved config carries the critic_init path. After actor
    # updates the saved actor no longer equals the source checkpoint's, and resume must still work.
    learner = ACLearner(named, LearnerConfig(batch_size=4, buffer_size=32, critic_init=str(checkpoint)))
    learner.import_chunk(chunk(tmp_path / 'c.npz', named))
    learner.train(1, 2)
    assert any(not torch.equal(learner.model.policy.state_dict()[k], v) for k, v in state.items()
               if k.startswith('actor.'))
    learner.save(tmp_path / 'saved')
    resumed = ACLearner.resume(tmp_path / 'saved', tmp_path / 'saved' / 'actor',
                               LearnerConfig(batch_size=4, buffer_size=32, critic_init=str(checkpoint)))
    for key, value in learner.model.policy.state_dict().items():
        assert torch.equal(resumed.model.policy.state_dict()[key], value), key
