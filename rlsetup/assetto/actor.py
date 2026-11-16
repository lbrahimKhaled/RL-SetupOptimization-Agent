"""AC3 source actor bundle: CPU inference on already normalized OCD observations.

Outputs are normalized policy actions, never Assetto Corsa controls. This module
does not construct an environment, critic, replay buffer or SB3 policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class NetworkSpec:
    observation_dim: int = 159
    action_dim: int = 3
    hidden: tuple[int, ...] = (256, 256, 256)
    activation: str = 'ReLU'
    log_std_min: float = -20.
    log_std_max: float = 2.


@dataclass(frozen=True)
class SourceObservation:
    """Frozen source ObsConfig; scales have the units given in their names."""
    n_preview: int = 20
    preview_horizon_m: float = 300.
    preview_exponent: float = 1.5
    preview_min_scale_m: float = 10.
    grid_m: float = 1.
    history: int = 3
    speed_scale_mps: float = 80.
    vy_scale_mps: float = 5.
    accel_scale_mps2: float = 30.
    yaw_rate_scale_radps: float = 1.
    steer_scale_rad: float = .6
    heading_scale_rad: float = .5
    slip_gain: float = 5.
    slip_clip: float = 5.
    e_scale_m: float = 5.
    gear_scale: float = 8.
    clip: float = 10.


NETWORK = NetworkSpec()
SOURCE_OBS = SourceObservation()
SCHEMA = 'rlsetup.ac3.source-actor'
VERSION = 1
WEIGHTS_FILE = 'actor.pt'
CONFIG_FILE = 'config.json'
MANIFEST_FILE = 'manifest.json'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'nonfinite JSON number: {value}')

    try:
        result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid JSON') from exc
    if not isinstance(result, dict):
        raise ValueError('JSON root must be an object')
    return result


def _encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _config(config: dict) -> dict:
    """Validate only the supported source contract; never fill omitted ObsConfig fields."""
    env = config.get('env')
    if not isinstance(env, dict):
        raise ValueError('source config must contain env')
    expected = {'car': 'sf70h_tum', 'track': 'Catalunya', 'obs_gear': False,
                'pedal_deadzone': True, 'reference': 'path', 'action_repeat': 2}
    for key, value in expected.items():
        if key not in env or type(env[key]) is not type(value) or env[key] != value:
            raise ValueError(f'unsupported source env.{key}')
    if env.get('obs') != asdict(SOURCE_OBS):
        raise ValueError('unsupported or incomplete source observation scales/layout')
    # bool is an int in Python; reject it explicitly in numeric configuration fields.
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           for v in env['obs'].values()):
        raise ValueError('source observation fields must be numeric')
    if not isinstance(env.get('setup'), str) or not env['setup']:
        raise ValueError('source setup identity is required')
    if type(config.get('seed')) is not int or not isinstance(config.get('preset'), str):
        raise ValueError('source seed and preset identity are required')
    agent = config.get('agent', {})
    if not isinstance(agent, dict) or agent.get('net_arch') != list(NETWORK.hidden):
        raise ValueError('unsupported source config network')
    return env


def _columns() -> list[str]:
    ego = ['speed', 'v_x', 'v_y', 'yaw_rate', 'a_x', 'a_y', 'steer_actual', 'slip_rl', 'slip_rr']
    actions = ['steering_action', 'throttle_action', 'brake_action']
    columns = ego + ['e_ref', 'heading_error_ref', 'track_position']
    columns += [f'preview_{k}.{point}.{axis}' for k in range(1, SOURCE_OBS.n_preview + 1)
                for point in ('xi_ref', 'left_edge', 'right_edge') for axis in ('x', 'y')]
    columns += actions
    columns += [f'previous_{k}.{name}' for k in range(1, SOURCE_OBS.history) for name in ego + actions]
    return columns


def _manifest(config: dict, source: dict, config_sha: str, weights_sha: str) -> dict:
    env = _config(config)
    if (not isinstance(source, dict) or set(source) != {'checkpoint_name', 'checkpoint_sha256', 'num_timesteps'}
            or not isinstance(source['checkpoint_name'], str) or not source['checkpoint_name']
            or type(source['num_timesteps']) is not int or source['num_timesteps'] < 0):
        raise ValueError('invalid checkpoint identity')
    for digest in (source['checkpoint_sha256'], config_sha, weights_sha):
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('invalid SHA256')
    network = asdict(NETWORK)
    network['hidden'] = list(NETWORK.hidden)
    return {
        'schema': SCHEMA, 'version': VERSION, 'network': network,
        'weights': {'file': WEIGHTS_FILE, 'sha256': weights_sha, 'dtype': 'float32', 'device': 'cpu'},
        'source': {**source, 'config_file': CONFIG_FILE, 'config_sha256': config_sha,
                   'car': env['car'], 'track': env['track'], 'setup': env['setup'],
                   'seed': config['seed'], 'preset': config['preset']},
        'observation': {
            'dtype': 'float32', 'shape': [NETWORK.observation_dim], 'obs_gear': False,
            'scales': env['obs'], 'columns': _columns(), 'final_clip': [-SOURCE_OBS.clip, SOURCE_OBS.clip],
            'frame': 'x forward, y left; yaw and steering positive left',
            'acceleration': 'a_x=dv_x/dt-r*v_y; a_y=dv_y/dt+r*v_x; midpoint r,v over last decision',
            'lateral': 'e_ref positive left; heading error relative to xi_ref; track_position=-1 right,+1 left',
            'preview': 'Track.path arc stations d_k=300*(k/20)^1.5 m; xi_ref,left,right xy / max(d_k,10 m)',
            'history': 'previous 2 decision ego+last_action blocks, newest first; placement fill uses ego and [0,-1,-1]',
            'normalization': 'fixed source scales; rear slip clip(5*slip,-5,5); final clip; no running normalization',
            'construction': 'caller must construct all 159 values; no padding, defaults or AC telemetry mapping',
        },
        'source_action': {
            'names': ['steering', 'throttle', 'brake'], 'bounds': [-1., 1.],
            'steering_lock_rad': .6, 'steering_positive': 'left',
            'pedal_formula': 'clip((a+0.5)/1.5,0,1)', 'decision_hz': 25.,
            'meaning': 'normalized source policy actions; AC controls require separately calibrated mapping',
        },
    }


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = (NETWORK.observation_dim, *NETWORK.hidden)
        self.latent_pi = nn.Sequential(*[layer for a, b in zip(sizes, sizes[1:])
                                         for layer in (nn.Linear(a, b, device='cpu', dtype=torch.float32), nn.ReLU())])
        self.mu = nn.Linear(sizes[-1], NETWORK.action_dim, device='cpu', dtype=torch.float32)
        self.log_std = nn.Linear(sizes[-1], NETWORK.action_dim, device='cpu', dtype=torch.float32)

    def forward(self, observation):
        latent = self.latent_pi(observation)
        return self.mu(latent), self.log_std(latent).clamp(NETWORK.log_std_min, NETWORK.log_std_max)


def _weights(data: bytes) -> dict[str, torch.Tensor]:
    try:
        weights = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('cannot read tensor-only weights') from exc
    if not isinstance(weights, dict):
        raise ValueError('weights must be a tensor dictionary')
    return weights


def _validate_weights(weights: dict[str, torch.Tensor]) -> None:
    sizes = (NETWORK.observation_dim, *NETWORK.hidden)
    shapes = {}
    for index, (a, b) in enumerate(zip(sizes, sizes[1:])):
        shapes[f'latent_pi.{2 * index}.weight'] = (b, a)
        shapes[f'latent_pi.{2 * index}.bias'] = (b,)
    for head in ('mu', 'log_std'):
        shapes[f'{head}.weight'] = (NETWORK.action_dim, sizes[-1])
        shapes[f'{head}.bias'] = (NETWORK.action_dim,)
    if set(weights) != set(shapes):
        raise ValueError('actor must contain exactly the ten supported tensors')
    for name, value in weights.items():
        if (type(value) is not torch.Tensor or value.device.type != 'cpu' or value.dtype != torch.float32
                or value.layout != torch.strided or tuple(value.shape) != shapes[name]
                or not torch.isfinite(value).all()):
            raise ValueError(f'invalid actor tensor: {name}')


def _checkpoint_metadata(data: dict) -> None:
    """Read JSON literals only. SB3's :serialized: fields are deliberately ignored."""
    policy = data.get('policy_kwargs')
    allowed = {':type:', ':serialized:', 'net_arch', 'activation_fn', 'n_critics',
               'share_features_extractor', 'use_sde'}
    if (not isinstance(policy, dict) or not set(policy) <= allowed
            or policy.get('net_arch') != {'pi': list(NETWORK.hidden), 'qf': list(NETWORK.hidden)}
            or policy.get('activation_fn') != "<class 'torch.nn.modules.activation.ReLU'>"
            or policy.get('use_sde') is not False or data.get('use_sde') is not False):
        raise ValueError('unsupported policy architecture, activation, features or gSDE')
    for key, dim, bound in (('observation_space', NETWORK.observation_dim, SOURCE_OBS.clip),
                            ('action_space', NETWORK.action_dim, 1.)):
        space = data.get(key)
        if (not isinstance(space, dict) or space.get('_shape') != [dim]
                or space.get('dtype') != 'float32'
                or space.get(':type:') != "<class 'gymnasium.spaces.box.Box'>"):
            raise ValueError(f'unsupported {key}')
        for side, sign in (('low', -1), ('high', 1)):
            literal = space.get(side)
            if not isinstance(literal, str) or not literal.startswith('[') or not literal.endswith(']'):
                raise ValueError(f'missing literal {key} bounds')
            tokens = literal[1:-1].split()
            try:
                valid = len(tokens) == dim and all(float(x) == sign * bound for x in tokens)
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f'unsupported {key} bounds')


def export_actor(checkpoint: str | Path, config: str | Path, output: str | Path) -> Path:
    """Export verified source actor tensors to an exclusively created directory.

    The manifest is written last as the completion marker. No source file is changed.
    Encoded SB3 metadata is never decoded or unpickled.
    """
    checkpoint, config, output = Path(checkpoint), Path(config), Path(output)
    checkpoint_data, config_data = checkpoint.read_bytes(), config.read_bytes()
    configuration = _json(config_data)
    _config(configuration)
    with zipfile.ZipFile(io.BytesIO(checkpoint_data)) as archive:
        if archive.namelist().count('data') != 1 or archive.namelist().count('policy.pth') != 1:
            raise ValueError('checkpoint must have one data and one policy.pth member')
        metadata = _json(archive.read('data'))
        _checkpoint_metadata(metadata)
        state = _weights(archive.read('policy.pth'))
    if any(not isinstance(key, str) for key in state):
        raise ValueError('policy tensor keys must be strings')
    weights = {key.removeprefix('actor.'): value for key, value in state.items() if key.startswith('actor.')}
    _validate_weights(weights)
    buffer = io.BytesIO()
    torch.save({key: value.detach().clone() for key, value in weights.items()}, buffer)
    weights_data = buffer.getvalue()
    source = {'checkpoint_name': checkpoint.name, 'checkpoint_sha256': _sha(checkpoint_data),
              'num_timesteps': metadata.get('num_timesteps')}
    manifest = _manifest(configuration, source, _sha(config_data), _sha(weights_data))
    output.mkdir(parents=False, exist_ok=False)
    (output / WEIGHTS_FILE).write_bytes(weights_data)
    (output / CONFIG_FILE).write_bytes(config_data)
    (output / MANIFEST_FILE).write_bytes(_encoded(manifest))
    return output


class ActorRuntime:
    """Frozen CPU actor. Observation construction and AC control mapping are external."""
    def __init__(self, bundle: str | Path):
        bundle = Path(bundle)
        try:
            manifest = _json((bundle / MANIFEST_FILE).read_bytes())
            config_data = (bundle / CONFIG_FILE).read_bytes()
            weights_data = (bundle / WEIGHTS_FILE).read_bytes()
        except OSError as exc:
            raise ValueError('incomplete actor bundle') from exc
        source = manifest.get('source')
        if not isinstance(source, dict):
            raise ValueError('invalid manifest source')
        identity = {key: source.get(key) for key in ('checkpoint_name', 'checkpoint_sha256', 'num_timesteps')}
        expected = _manifest(_json(config_data), identity, _sha(config_data), _sha(weights_data))
        # Compare canonical JSON as well: bool/int substitutions are not schema-compatible.
        if _encoded(manifest) != _encoded(expected):
            raise ValueError('unsupported manifest schema/contract or file hash mismatch')
        weights = _weights(weights_data)
        _validate_weights(weights)
        # Keep module initialization from consuming the caller's stochastic rollout RNG.
        with torch.random.fork_rng(devices=[]):
            self._actor = _Actor().cpu().eval().requires_grad_(False)
        self._actor.load_state_dict(weights, strict=True)
        self.manifest = manifest

    def _observation(self, observation: np.ndarray) -> torch.Tensor:
        if (not isinstance(observation, np.ndarray) or observation.dtype != np.float32
                or observation.shape != (NETWORK.observation_dim,) or not np.isfinite(observation).all()
                or np.any(np.abs(observation) > SOURCE_OBS.clip)):
            raise ValueError('observation must be finite normalized np.float32[159] within [-10,10]')
        return torch.from_numpy(observation.copy()).unsqueeze(0)

    @torch.inference_mode()
    def mean_log_std(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pre-squash mean and clamped log standard deviation, each float32[3]."""
        mean, log_std = self._actor(self._observation(observation))
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise ValueError('nonfinite actor distribution')
        return mean[0].numpy().copy(), log_std[0].numpy().copy()

    @torch.inference_mode()
    def predict(self, observation: np.ndarray, *, deterministic: bool = True,
                generator: torch.Generator | None = None) -> np.ndarray:
        """Return normalized [steering, throttle, brake]; stochastic calls need a seeded CPU generator."""
        if type(deterministic) is not bool:
            raise ValueError('deterministic must be bool')
        mean, log_std = self.mean_log_std(observation)
        mean, log_std = torch.from_numpy(mean), torch.from_numpy(log_std)
        if not deterministic:
            if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
                raise ValueError('stochastic prediction requires an explicit seeded CPU torch.Generator')
            mean = mean + log_std.exp() * torch.randn(mean.shape, generator=generator,
                                                     device='cpu', dtype=torch.float32)
        action = mean.tanh().numpy().copy()
        if not np.isfinite(action).all():
            raise ValueError('nonfinite policy action')
        return action


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='export actor-only bundle into a new directory')
    for name in ('checkpoint', 'config', 'output'):
        export.add_argument(f'--{name}', required=True, type=Path)
    args = parser.parse_args(argv)
    print(export_actor(args.checkpoint, args.config, args.output))


if __name__ == '__main__':
    main()
