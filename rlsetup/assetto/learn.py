"""Offline AC SAC updates on validated collector chunks, with actor-only transfer.

No simulator is imported or stepped. Inputs use the source normalized 159-vector
and normalized left-positive steering/throttle/brake actions in [-1, 1]. Resume
files are trusted local outputs of this program, not arbitrary SB3 checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import re
import subprocess

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import configure

from .actor import (ActorRuntime, MANIFEST_FILE, _encoded, _json, _sha,
                    export_updated_actor, source_contract_sha256)
from ..driving.agent import StagedSAC, check_sb3_version
from ..driving.policy import LNSACPolicy


@dataclass(frozen=True)
class LearnerConfig:
    """R1 transfer: fresh AC one-step replay/critics/alpha; no source critic state."""
    seed: int = 0
    device: str = 'cpu'
    batch_size: int = 512
    buffer_size: int = 1_000_000
    learning_rate: float = 3e-4
    actor_lr: float = 1e-4
    gamma: float = .992
    n_steps: int = 1
    tau: float = .005
    ent_coef: str = 'auto_0.1'
    torch_threads: int = 1
    critic_init: str | None = None  # source SB3 checkpoint: initial critics, targets and alpha

    def validate(self):
        for name in ('batch_size', 'buffer_size', 'torch_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (type(self.seed) is not int or self.seed < 0 or self.device not in ('cpu', 'mps')
                or self.gamma != .992 or self.n_steps != 1 or self.buffer_size < self.batch_size
                or self.learning_rate <= 0 or self.actor_lr <= 0 or not 0 < self.tau <= 1
                or self.ent_coef != 'auto_0.1'
                or not (self.critic_init is None or (isinstance(self.critic_init, str) and self.critic_init))):
            raise ValueError('unsupported AC learner configuration')


class _SpacesOnly(gym.Env):
    observation_space = gym.spaces.Box(-10., 10., (159,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError('offline learner must never reset an environment')

    def step(self, action):
        raise RuntimeError('offline learner must never step an environment')


def build_learner(bundle: str | Path, config: LearnerConfig) -> StagedSAC:
    """Initialize actor exactly from portable tensors; all other trainable state is fresh."""
    config.validate()
    check_sb3_version()
    runtime = ActorRuntime(bundle)
    torch.set_num_threads(config.torch_threads)
    model = StagedSAC(LNSACPolicy, _SpacesOnly(), learning_rate=config.learning_rate,
                      buffer_size=config.buffer_size, learning_starts=0,
                      batch_size=config.batch_size, tau=config.tau, gamma=config.gamma,
                      n_steps=1, ent_coef=config.ent_coef, target_entropy='auto',
                      policy_kwargs={'net_arch': {'pi': [256]*3, 'qf': [256]*3},
                                     'activation_fn': torch.nn.ReLU, 'n_critics': 2,
                                     'share_features_extractor': False},
                      seed=config.seed, device=config.device, verbose=0)
    model.actor.load_state_dict(runtime._actor.state_dict(), strict=True)
    if config.critic_init is not None:
        load_source_critics(model, config.critic_init, runtime)
    model.actor_lr = config.actor_lr
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


CRITIC_PREFIXES = ('critic.', 'critic_target.')


def load_source_critics(model: StagedSAC, checkpoint: str | Path, runtime: ActorRuntime) -> None:
    """Copy critics, target critics and log alpha from the actor's own OCD checkpoint.

    Only the exact checkpoint the actor bundle was exported from is accepted (SHA-256 in the
    bundle manifest), and its actor tensors must equal the bundle's. Members are read with
    torch.load(weights_only=True); optimizer states and replay are not used. Alpha keeps the
    OCD-trained value (0.063 for D2 bcn seed0): the source critics were fit with it.
    """
    import io
    import zipfile
    data = Path(checkpoint).read_bytes()
    if _sha(data) != runtime.manifest['source']['checkpoint_sha256']:
        raise ValueError('critic checkpoint is not the actor bundle source checkpoint')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        weights = torch.load(io.BytesIO(archive.read('policy.pth')), weights_only=True, map_location='cpu')
        variables = torch.load(io.BytesIO(archive.read('pytorch_variables.pth')), weights_only=True,
                               map_location='cpu')
    own = model.policy.state_dict()
    critic = {k: v for k, v in weights.items() if k.startswith(CRITIC_PREFIXES)}
    if (set(critic) != {k for k in own if k.startswith(CRITIC_PREFIXES)}
            or any(own[k].shape != v.shape or not torch.isfinite(v).all() for k, v in critic.items())):
        raise ValueError('source checkpoint critics do not match the AC learner networks')
    actor = {k: v for k, v in weights.items() if k.startswith('actor.')}
    if set(actor) != {k for k in own if k.startswith('actor.')} or any(
            not torch.equal(v, own[k].cpu()) for k, v in actor.items()):
        raise ValueError('source checkpoint actor differs from the actor bundle')
    log_alpha = variables.get('log_ent_coef')
    if (not isinstance(log_alpha, torch.Tensor) or log_alpha.shape != model.log_ent_coef.shape
            or not torch.isfinite(log_alpha).all()):
        raise ValueError('source checkpoint has no usable entropy coefficient')
    own.update(critic)
    model.policy.load_state_dict(own, strict=True)
    with torch.no_grad():
        model.log_ent_coef.copy_(log_alpha.to(model.log_ent_coef.device))


_AUDIT = {'timestamp_ac': (), 'next_timestamp_ac': (), 'command': (3,),
          'desired_road_angle': (), 'executed_road_angle': ()}

_ARRAYS = {'obs': (np.dtype('float32'), (159,)), 'next_obs': (np.dtype('float32'), (159,)),
           'actions': (np.dtype('float32'), (3,)), 'rewards': (np.dtype('float32'), ()),
           'terminated': (np.dtype('bool'), ()), 'truncated': (np.dtype('bool'), ()),
           'dt': (np.dtype('float64'), ()), 'episode_id': (np.dtype('int64'), ()),
           'seq': (np.dtype('int64'), ())}


def read_chunk(path: str | Path, actor_manifest_sha256: str, contract_sha256: str,
               gamma: float = .992, config_sha256: str | None = None, *,
               allow_empty: bool = False) -> tuple[dict, dict, str]:
    """Strict numeric-only NPZ; reject time/episode discontinuities before replay mutation."""
    data = Path(path).read_bytes()
    import io
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if (not {*_ARRAYS, 'metadata_json'} <= set(archive.files)
                    or not set(archive.files) <= {*_ARRAYS, *_AUDIT, 'metadata_json'}
                    or len(set(archive.files)) != len(archive.files)):
                raise ValueError('unexpected chunk arrays')
            arrays = {key: archive[key] for key in _ARRAYS}
            audits = {key: archive[key] for key in _AUDIT if key in archive.files}
            encoded = archive['metadata_json']
            if encoded.shape != () or encoded.dtype.kind != 'U':
                raise ValueError('metadata_json must be a Unicode JSON scalar')
            metadata = _json(str(encoded.item()).encode())
    except (OSError, TypeError) as exc:
        raise ValueError('invalid numeric NPZ chunk') from exc
    if arrays['obs'].ndim != 2:
        raise ValueError('obs must have shape N x 159')
    n = len(arrays['obs'])
    if n == 0 and not allow_empty:
        raise ValueError('empty chunk')
    for key, (dtype, shape) in _ARRAYS.items():
        value = arrays[key]
        if value.dtype != dtype or value.shape != (n, *shape) or not np.isfinite(value).all():
            raise ValueError(f'invalid chunk array {key}')
    for key, value in audits.items():
        if value.dtype != np.float64 or value.shape != (n, *_AUDIT[key]) or not np.isfinite(value).all():
            raise ValueError(f'invalid audit array {key}')
    if (np.any(np.abs(arrays['obs']) > 10) or np.any(np.abs(arrays['next_obs']) > 10)
            or np.any(np.abs(arrays['actions']) > 1) or np.any(arrays['dt'] <= 0)
            or np.any(arrays['episode_id'] < 0) or np.any(arrays['seq'] < 0)
            or np.any(arrays['terminated'] & arrays['truncated'])):
        raise ValueError('chunk values outside contract')
    required = {'schema_version', 'actor_manifest_sha256', 'source_contract_sha256', 'target',
                'reference', 'gamma', 'min_dt_s', 'max_dt_s', 'stop_reason', 'reward_mode',
                'n_steps', 'actor_fixed', 'source_config_sha256'}
    if not required <= set(metadata):
        raise ValueError('incomplete chunk provenance')
    if (type(metadata['schema_version']) is not int or metadata['schema_version'] != 1
            or metadata['actor_manifest_sha256'] != actor_manifest_sha256
            or metadata['source_contract_sha256'] != contract_sha256
            or type(metadata['gamma']) not in (int, float) or metadata['gamma'] != gamma
            or metadata['reward_mode'] != 'time_pbrs'
            or type(metadata['n_steps']) is not int or metadata['n_steps'] != 1
            or metadata['actor_fixed'] is not True
            or (config_sha256 is not None and metadata['source_config_sha256'] != config_sha256)):
        raise ValueError('chunk actor, reward or source contract mismatch')
    target = metadata['target']
    if (not isinstance(target, dict) or set(target) != {'car_name', 'track_name', 'track_configuration', 'setup_id'}
            or any(not isinstance(v, str) for v in target.values())
            or any(not target[k] for k in ('car_name', 'track_name', 'setup_id'))):
        raise ValueError('invalid chunk target')
    if metadata['reference'] not in ('centerline', 'fast_lane', 'ocd_path'):
        raise ValueError('reference must be centerline, fast_lane or ocd_path')
    if metadata['reference'] != 'centerline':
        digest, length = metadata.get('reference_geometry_sha256'), metadata.get('reference_length_m')
        if (not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
                or type(length) not in (int, float) or not np.isfinite(length) or length <= 0):
            raise ValueError(f"{metadata['reference']} reference geometry provenance required")
    terms = metadata.get('source_terms')  # v2 reference: R4 terms, OCD progress/start/stall
    if terms is not None and (not isinstance(terms, dict) or not terms or any(
            not (isinstance(v, str) or (type(v) in (int, float) and np.isfinite(v))) for v in terms.values())):
        raise ValueError('invalid chunk source_terms')
    lo, hi = metadata['min_dt_s'], metadata['max_dt_s']
    if (type(lo) not in (int, float) or type(hi) not in (int, float)
            or not 0 < lo <= hi or np.any(arrays['dt'] < lo) or np.any(arrays['dt'] > hi)
            or not isinstance(metadata['stop_reason'], str) or not metadata['stop_reason']):
        raise ValueError('invalid chunk timing/stop provenance')
    done = arrays['terminated'] | arrays['truncated']
    for i in range(1, n):
        if done[i-1]:
            if arrays['episode_id'][i] == arrays['episode_id'][i-1]:
                raise ValueError('episode must change after boundary')
        elif (arrays['episode_id'][i] != arrays['episode_id'][i-1]
              or arrays['seq'][i] != arrays['seq'][i-1]+1
              or not np.array_equal(arrays['obs'][i], arrays['next_obs'][i-1])):
            raise ValueError('discontinuity without episode boundary')
    if n and not done[-1]:
        raise ValueError('chunk must end at a terminated/truncated boundary')
    return arrays, metadata, _sha(data)


def replay_identity(metadata: dict) -> dict:
    """Shared collector/learner identity; legacy centerline transactions stay unchanged."""
    identity = {key: metadata[key] for key in ('target', 'reference', 'gamma', 'min_dt_s', 'max_dt_s', 'reward_mode')}
    if metadata['reference'] != 'centerline':
        identity.update({key: metadata[key] for key in ('reference_geometry_sha256', 'reference_length_m')})
    if 'source_terms' in metadata:  # v2 reference chunks never mix with R1-only replay
        identity['source_terms'] = metadata['source_terms']
    return identity


class ACLearner:
    def __init__(self, bundle: str | Path, config: LearnerConfig):
        self.bundle, self.config = Path(bundle), config
        runtime = ActorRuntime(bundle)
        self.actor_hash = _sha((self.bundle / MANIFEST_FILE).read_bytes())
        self.contract_hash = source_contract_sha256(runtime.manifest)
        self.config_hash = runtime.manifest['source']['config_sha256']
        self.model = build_learner(bundle, config)
        self.imported: list[str] = []
        self.provenance: dict | None = None
        self.critic_updates = self.updates = 0

    def import_chunk(self, path: str | Path) -> bool:
        # Resumed data may come from the previous actor in the same AC lineage.
        digest = _sha(Path(path).read_bytes())
        if digest in self.imported:
            return False
        arrays, metadata, digest = read_chunk(path, self.actor_hash, self.contract_hash, self.config.gamma, self.config_hash)
        identity = replay_identity(metadata)
        if self.provenance is not None and identity != self.provenance:
            raise ValueError('chunk target/reference/timing differs from existing AC replay')
        self.provenance = identity
        for i in range(len(arrays['obs'])):
            terminated, truncated = bool(arrays['terminated'][i]), bool(arrays['truncated'][i])
            self.model.replay_buffer.add(arrays['obs'][i:i+1], arrays['next_obs'][i:i+1],
                                        arrays['actions'][i:i+1], arrays['rewards'][i:i+1],
                                        np.array([terminated or truncated]),
                                        [{'TimeLimit.truncated': truncated and not terminated}])
        self.imported.append(digest)
        return True

    def train(self, critic_updates: int, updates: int):
        for count in (critic_updates, updates):
            if type(count) is not int or count < 0:
                raise ValueError('update budgets must be explicit nonnegative integers')
        if self.model.buffer_transitions() < self.config.batch_size:
            raise ValueError('not enough AC replay transitions for batch_size')
        self.model.set_frozen(True, True)
        if critic_updates:
            self.model.train(critic_updates, self.config.batch_size)
        self.model.set_frozen(False, False)
        if updates:
            self.model.train(updates, self.config.batch_size)
        self.critic_updates += critic_updates
        self.updates += updates
        if any(not torch.isfinite(p).all() for p in self.model.policy.parameters()):
            raise ValueError('nonfinite learner parameters')

    def save(self, output: str | Path) -> Path:
        """New transaction directory; completion manifest last. No existing state is overwritten."""
        if self.provenance is None:
            raise ValueError('cannot save learner without validated AC provenance')
        output = Path(output)
        output.mkdir(parents=False, exist_ok=False)
        self.model.save(output / 'model.zip')
        self.model.save_replay_buffer(output / 'replay.pkl')
        training = {'schema': 'rlsetup.ac3.training.v1', 'backend': 'assetto_corsa',
                    'parent_actor_manifest_sha256': self.actor_hash,
                    'source_contract_sha256': self.contract_hash, 'target': self.provenance['target'],
                    'chunk_sha256': self.imported, 'critic_updates': self.critic_updates,
                    'updates': self.updates, 'gamma': self.config.gamma, 'n_steps': 1}
        export_updated_actor(self.model.actor.state_dict(), self.bundle, output / 'actor', training)
        np_state = np.random.get_state()
        rng = {'python': random.getstate(), 'numpy': [np_state[0], np_state[1].tolist(),
                                                     np_state[2], np_state[3], np_state[4]]}
        (output / 'rng.json').write_bytes(_encoded(rng))
        torch.save(torch.get_rng_state(), output / 'rng.pt')
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            git_hash = None
        manifest = {'schema': 'rlsetup.ac3.learner.v1', 'config': asdict(self.config),
                    'actor_manifest_sha256': self.actor_hash, 'source_contract_sha256': self.contract_hash,
                    'provenance': self.provenance, 'imported': self.imported,
                    'critic_updates': self.critic_updates, 'updates': self.updates,
                    'git_hash': git_hash, 'source_actor': ActorRuntime(self.bundle).manifest,
                    'dependency': 'no OCD runtime used; original source config in actor/config.json',
                    'files': {name: _sha((output / name).read_bytes()) for name in
                              ('model.zip', 'replay.pkl', 'rng.json', 'rng.pt')},
                    'exported_actor_manifest_sha256': _sha((output / 'actor' / MANIFEST_FILE).read_bytes())}
        (output / MANIFEST_FILE).write_bytes(_encoded(manifest))
        return output

    @classmethod
    def resume(cls, saved: str | Path, bundle: str | Path, config: LearnerConfig) -> ACLearner:
        """Validate own completion manifest and hashes BEFORE trusted local pickle loaders."""
        saved = Path(saved)
        manifest = _json((saved / MANIFEST_FILE).read_bytes())
        actor_hash = _sha((Path(bundle) / MANIFEST_FILE).read_bytes())
        ActorRuntime(bundle)
        if (manifest.get('schema') != 'rlsetup.ac3.learner.v1'
                or manifest.get('config') != asdict(config)
                or actor_hash not in (manifest.get('actor_manifest_sha256'),
                                      manifest.get('exported_actor_manifest_sha256'))
                or manifest.get('source_contract_sha256') != source_contract_sha256(ActorRuntime(bundle).manifest)
                or set(manifest.get('files', {})) != {'model.zip', 'replay.pkl', 'rng.json', 'rng.pt'}):
            raise ValueError('resume is not a compatible local AC learner transaction')
        for name, digest in manifest['files'].items():
            if _sha((saved / name).read_bytes()) != digest:
                raise ValueError(f'incomplete or corrupted learner file: {name}')
        ActorRuntime(saved / 'actor')
        if _sha((saved / 'actor' / MANIFEST_FILE).read_bytes()) != manifest['exported_actor_manifest_sha256']:
            raise ValueError('resume actor mismatch')
        if (not isinstance(manifest.get('imported'), list)
                or any(not isinstance(x, str) or re.fullmatch('[0-9a-f]{64}', x) is None for x in manifest['imported'])
                or len(set(manifest['imported'])) != len(manifest['imported'])
                or not isinstance(manifest.get('provenance'), dict)):
            raise ValueError('invalid learner replay provenance')
        # critic_init seeds a NEW learner only: the saved model below carries the trained critics, and
        # after actor updates the bundle no longer equals the source checkpoint's actor.
        obj = cls(bundle, replace(config, critic_init=None))
        obj.config = config
        obj.model = StagedSAC.load(saved / 'model.zip', device=config.device)
        obj.model.load_replay_buffer(saved / 'replay.pkl', truncate_last_traj=False)
        obj.model.set_logger(configure(folder=None, format_strings=[]))
        if (obj.model.n_steps != 1 or obj.model.gamma != config.gamma
                or obj.model.observation_space != _SpacesOnly.observation_space
                or obj.model.action_space != _SpacesOnly.action_space):
            raise ValueError('saved learner model contract mismatch')
        obj.imported, obj.provenance = manifest['imported'], manifest['provenance']
        obj.critic_updates, obj.updates = manifest['critic_updates'], manifest['updates']
        rng = _json((saved / 'rng.json').read_bytes())
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        random.setstate(tuples(rng['python']))
        state = rng['numpy']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(torch.load(saved / 'rng.pt', weights_only=True, map_location='cpu'))
        return obj


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-bundle', required=True, type=Path)
    parser.add_argument('--chunk', required=True, action='append', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--critic-updates', required=True, type=int)
    parser.add_argument('--updates', required=True, type=int)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--critic-init', type=Path, help="the actor's source SB3 checkpoint (best.zip)")
    args = parser.parse_args(argv)
    config = LearnerConfig(seed=args.seed, device=args.device, batch_size=args.batch_size,
                           buffer_size=args.buffer_size,
                           critic_init=str(args.critic_init.resolve()) if args.critic_init else None)
    if args.output.exists():
        parser.error('--output must be a new transaction directory')
    learner = (ACLearner.resume(args.resume, args.actor_bundle, config) if args.resume
               else ACLearner(args.actor_bundle, config))
    for chunk in args.chunk:
        learner.import_chunk(chunk)
    learner.train(args.critic_updates, args.updates)
    print(learner.save(args.output))


if __name__ == '__main__':
    main()
