"""AC3 actor-only export/runtime: real SB3 Actor parity and strict input contracts."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.sac.policies import Actor

from rlsetup.assetto.actor import ActorRuntime, SOURCE_OBS, export_actor, main


@pytest.fixture(scope='module', autouse=True)
def one_torch_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def source(tmp_path):
    obs = gym.spaces.Box(-10., 10., (159,), dtype=np.float32)
    action = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(712)
        actor = Actor(obs, action, [256, 256, 256], FlattenExtractor(obs), 159).eval()
    metadata = {
        'policy_kwargs': {':type:': "<class 'dict'>", ':serialized:': 'DO NOT DECODE THIS',
                          'net_arch': {'pi': [256] * 3, 'qf': [256] * 3},
                          'activation_fn': "<class 'torch.nn.modules.activation.ReLU'>",
                          'use_sde': False, 'n_critics': 2, 'share_features_extractor': False},
        'use_sde': False, 'num_timesteps': 5050000,
    }
    for key, space in [('observation_space', obs), ('action_space', action)]:
        metadata[key] = {':type:': "<class 'gymnasium.spaces.box.Box'>", ':serialized:': 'NOT PICKLE',
                         '_shape': list(space.shape), 'dtype': str(space.dtype),
                         'low': str(space.low), 'high': str(space.high)}
    config = {'env': {'car': 'sf70h_tum', 'track': 'Catalunya', 'setup': 'bcn_best', 'reference': 'path',
                      'obs_gear': False, 'pedal_deadzone': True, 'action_repeat': 2,
                      'obs': asdict(SOURCE_OBS)},
              'agent': {'net_arch': [256] * 3}, 'seed': 0, 'preset': 'bcn'}
    checkpoint = tmp_path / 'best.zip'
    configuration = tmp_path / 'config.json'

    def save():
        state = {f'actor.{key}': value for key, value in actor.state_dict().items()}
        state['critic.qf0.0.weight'] = torch.ones((2, 2))
        buffer = io.BytesIO()
        torch.save(state, buffer)
        with zipfile.ZipFile(checkpoint, 'w') as archive:
            archive.writestr('data', json.dumps(metadata))
            archive.writestr('policy.pth', buffer.getvalue())
            # Neither optimizers nor arbitrary encoded objects are read by the exporter.
            archive.writestr('actor.optimizer.pth', b'not a torch archive')
        configuration.write_text(json.dumps(config))

    save()
    return actor, metadata, config, checkpoint, configuration, save


def bundle(source, tmp_path):
    return export_actor(source[3], source[4], tmp_path / 'bundle')


@pytest.mark.parametrize('extreme', [False, True])
def test_parity_against_sb3_actor(source, tmp_path, extreme):
    actor = source[0]
    if extreme:
        with torch.no_grad():
            actor.mu.weight.zero_()
            actor.mu.bias.copy_(torch.tensor([-100., 0., 100.]))
            actor.log_std.weight.zero_()
            actor.log_std.bias.copy_(torch.tensor([-100., 0., 100.]))
        source[-1]()
    runtime = ActorRuntime(bundle(source, tmp_path))
    observations = [np.zeros(159, dtype=np.float32), np.full(159, -10., dtype=np.float32),
                    np.full(159, 10., dtype=np.float32),
                    np.random.default_rng(123).uniform(-10., 10., 159).astype(np.float32)]
    for obs in observations:
        tensor = torch.from_numpy(obs).unsqueeze(0)
        with torch.inference_mode():
            mean, log_std, _ = actor.get_action_dist_params(tensor)
            expected = actor(tensor, deterministic=True)[0].numpy()
        actual_mean, actual_std = runtime.mean_log_std(obs)
        np.testing.assert_allclose(actual_mean, mean[0].numpy(), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_std, log_std[0].numpy(), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(runtime.predict(obs), expected, atol=1e-6, rtol=1e-6)
        for seed in (0, 73, 999):
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.manual_seed(seed)
                expected = actor(tensor, deterministic=False)[0].numpy()
            actual = runtime.predict(obs, deterministic=False, generator=torch.Generator().manual_seed(seed))
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
            assert actual.shape == (3,) and actual.dtype == np.float32
            assert np.isfinite(actual).all() and np.max(np.abs(actual)) <= 1.
        if extreme:
            np.testing.assert_array_equal(actual_std, [-20., 0., 2.])
            np.testing.assert_array_equal(runtime.predict(obs)[[0, 2]], [-1., 1.])
    assert not runtime._actor.training
    assert all(not p.requires_grad and p.device.type == 'cpu' for p in runtime._actor.parameters())


def test_export_source_identity_and_exclusive_output(source, tmp_path):
    originals = [p.read_bytes() for p in source[3:5]]
    output = bundle(source, tmp_path)
    manifest = json.loads((output / 'manifest.json').read_bytes())
    assert manifest['source']['checkpoint_sha256'] == hashlib.sha256(originals[0]).hexdigest()
    assert manifest['source']['config_sha256'] == hashlib.sha256(originals[1]).hexdigest()
    assert manifest['source']['num_timesteps'] == 5050000
    assert manifest['observation']['scales'] == asdict(SOURCE_OBS)
    assert len(manifest['observation']['columns']) == 159
    assert manifest['observation']['columns'][12:18] == [
        'preview_1.xi_ref.x', 'preview_1.xi_ref.y', 'preview_1.left_edge.x',
        'preview_1.left_edge.y', 'preview_1.right_edge.x', 'preview_1.right_edge.y']
    assert manifest['observation']['columns'][135] == 'previous_1.speed'
    assert manifest['source_action']['steering_lock_rad'] == .6
    weights = torch.load(output / 'actor.pt', weights_only=True, map_location='cpu')
    assert len(weights) == 10 and all('critic' not in key for key in weights)
    assert (output / 'config.json').read_bytes() == originals[1]
    with pytest.raises(FileExistsError):
        export_actor(source[3], source[4], output)
    assert [p.read_bytes() for p in source[3:5]] == originals


@pytest.mark.parametrize('bad', [np.zeros(159), np.zeros(158, np.float32), np.zeros((1, 159), np.float32),
                                    [0.] * 159, np.full(159, np.nan, np.float32),
                                    np.full(159, np.inf, np.float32), np.full(159, 10.001, np.float32)])
def test_invalid_observation(source, tmp_path, bad):
    runtime = ActorRuntime(bundle(source, tmp_path))
    with pytest.raises(ValueError, match='observation'):
        runtime.predict(bad)


def test_stochastic_generator_and_global_rng_unchanged(source, tmp_path):
    output = bundle(source, tmp_path)
    before = torch.get_rng_state().clone()
    runtime = ActorRuntime(output)
    obs = np.zeros(159, np.float32)
    runtime.predict(obs)
    runtime.predict(obs, deterministic=False, generator=torch.Generator().manual_seed(4))
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match='Generator'):
        runtime.predict(obs, deterministic=False)
    with pytest.raises(ValueError, match='bool'):
        runtime.predict(obs, deterministic=1)


@pytest.mark.parametrize('phase', ['during_construction', 'after_construction'])
@pytest.mark.parametrize('defaults', ['float64', 'meta', 'float64_meta'])
def test_runtime_isolates_caller_torch_defaults(source, tmp_path, phase, defaults):
    output = bundle(source, tmp_path)
    obs = np.zeros(159, dtype=np.float32)
    baseline = ActorRuntime(output)
    expected_mean, expected_log_std = baseline.mean_log_std(obs)
    expected_det = baseline.predict(obs)
    expected_sample = baseline.predict(obs, deterministic=False,
                                      generator=torch.Generator(device='cpu').manual_seed(37))
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    runtime = ActorRuntime(output) if phase == 'after_construction' else None
    try:
        if 'float64' in defaults:
            torch.set_default_dtype(torch.float64)
        if 'meta' in defaults:
            torch.set_default_device('meta')
        configured_dtype, configured_device = torch.get_default_dtype(), torch.get_default_device()
        if runtime is None:
            runtime = ActorRuntime(output)
        assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in runtime._actor.parameters())
        mean, log_std = runtime.mean_log_std(obs)
        deterministic = runtime.predict(obs)
        stochastic = runtime.predict(obs, deterministic=False,
                                     generator=torch.Generator(device='cpu').manual_seed(37))
        for actual, expected in ((mean, expected_mean), (log_std, expected_log_std),
                                 (deterministic, expected_det), (stochastic, expected_sample)):
            assert actual.dtype == np.float32 and actual.shape == (3,)
            np.testing.assert_array_equal(actual, expected)
        assert torch.get_default_dtype() == configured_dtype
        assert torch.get_default_device() == configured_device
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize('mutation', ['version', 'dim', 'hash', 'extra', 'bool_version', 'columns'])
def test_invalid_manifest(source, tmp_path, mutation):
    output = bundle(source, tmp_path)
    path = output / 'manifest.json'
    manifest = json.loads(path.read_bytes())
    if mutation == 'version':
        manifest['version'] = 2
    elif mutation == 'dim':
        manifest['network']['observation_dim'] = 158
    elif mutation == 'hash':
        manifest['weights']['sha256'] = '0' * 64
    elif mutation == 'extra':
        manifest['extra'] = 'unsupported'
    elif mutation == 'bool_version':
        manifest['version'] = True
    else:
        manifest['observation']['columns'].reverse()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        ActorRuntime(output)


@pytest.mark.parametrize('file', ['actor.pt', 'config.json', 'manifest.json'])
def test_incomplete_bundle(source, tmp_path, file):
    output = bundle(source, tmp_path)
    (output / file).unlink()
    with pytest.raises(ValueError, match='incomplete'):
        ActorRuntime(output)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'shape', 'dtype', 'nan', 'inf'])
def test_invalid_tensors_even_with_updated_hash(source, tmp_path, mutation):
    output = bundle(source, tmp_path)
    weights_path = output / 'actor.pt'
    state = torch.load(weights_path, weights_only=True)
    if mutation == 'missing':
        state.pop('mu.bias')
    elif mutation == 'extra':
        state['critic.bad'] = torch.zeros(1)
    elif mutation == 'shape':
        state['mu.bias'] = torch.zeros(4)
    elif mutation == 'dtype':
        state['mu.bias'] = state['mu.bias'].double()
    else:
        state['mu.bias'][0] = float(mutation)
    torch.save(state, weights_path)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['weights']['sha256'] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='tensor'):
        ActorRuntime(output)


@pytest.mark.parametrize('mutation', ['activation', 'gSDE', 'arch', 'features', 'space', 'bounds',
                                     'obs_missing', 'obs_scale', 'obs_bool', 'gear', 'car', 'steps'])
def test_reject_unsupported_source(source, tmp_path, mutation):
    _, metadata, config, checkpoint, configuration, save = source
    if mutation == 'activation':
        metadata['policy_kwargs']['activation_fn'] = "<class 'torch.nn.modules.activation.Tanh'>"
    elif mutation == 'gSDE':
        metadata['use_sde'] = True
    elif mutation == 'arch':
        metadata['policy_kwargs']['net_arch']['pi'] = [256, 256]
    elif mutation == 'features':
        metadata['policy_kwargs']['features_extractor_class'] = 'custom'
    elif mutation == 'space':
        metadata['observation_space']['_shape'] = [158]
    elif mutation == 'bounds':
        metadata['action_space']['low'] = '[-2. -2. -2.]'
    elif mutation == 'obs_missing':
        config['env']['obs'].pop('gear_scale')
    elif mutation == 'obs_scale':
        config['env']['obs']['speed_scale_mps'] = 70.
    elif mutation == 'obs_bool':
        config['env']['obs']['grid_m'] = True
    elif mutation == 'gear':
        config['env']['obs_gear'] = True
    elif mutation == 'car':
        config['env']['car'] = 'sf70h_f1'
    else:
        metadata['num_timesteps'] = None
    save()
    with pytest.raises(ValueError):
        export_actor(checkpoint, configuration, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()


def test_malformed_json_and_file_hashes(source, tmp_path):
    output = bundle(source, tmp_path)
    manifest_path = output / 'manifest.json'
    original = manifest_path.read_bytes()
    for data in (b'{', b'[]', b'{"version":1,"version":1}', b'{"version":NaN}'):
        manifest_path.write_bytes(data)
        with pytest.raises(ValueError):
            ActorRuntime(output)
    manifest_path.write_bytes(original)
    config_path = output / 'config.json'
    config_path.write_bytes(config_path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash'):
        ActorRuntime(output)


def test_cli_and_runtime_without_simulator_or_sb3(source, tmp_path, capsys):
    output = tmp_path / 'bundle'
    main(['export', '--checkpoint', str(source[3]), '--config', str(source[4]), '--output', str(output)])
    assert capsys.readouterr().out.strip() == str(output)
    code = '''
import importlib.abc,sys
class NoSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ocd_lap','ocd_native','stable_baselines3','sb3_contrib'}:
            raise AssertionError('forbidden runtime import: '+fullname)
sys.meta_path.insert(0,NoSimulation())
import numpy as np
from rlsetup.assetto.actor import ActorRuntime
runtime=ActorRuntime(sys.argv[1])
assert runtime.predict(np.zeros(159,np.float32)).shape == (3,)
assert not any(x.split('.')[0] in {'ocd_lap','ocd_native','stable_baselines3','sb3_contrib'} for x in sys.modules)
print('actor-only runtime PASS')
'''
    completed = subprocess.run([sys.executable, '-c', code, str(output)], capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parents[1], timeout=15)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == 'actor-only runtime PASS'
