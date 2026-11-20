"""Create an isolated AC learner fork seeded with native one-step OCD transitions.

Inputs are trusted local training artifacts: replay files are Python pickles. No
simulator is stepped. AC and OCD rewards remain in their original domains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

from .learn import ACLearner, LearnerConfig


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mix_replay(ac, source, *, seed=0):
    """Pack sampled OCD then chronological AC; next writes evict OCD first.

    Source's n-step buffer stores one-step rewards and next observations. Retry
    warmup rows are excluded, but raw rewards and timeout flags are kept intact.
    """
    if ac.n_envs != 1 or ac.optimize_memory_usage or hasattr(ac, 'domain'):
        raise ValueError('expected unmodified single-environment AC replay')
    n_ac = ac.buffer_size if ac.full else ac.pos
    if n_ac > ac.buffer_size:
        raise ValueError('AC data exceed capacity')
    if (source.observation_space != ac.observation_space or source.action_space != ac.action_space
            or source.optimize_memory_usage or source.gamma != .992
            or not hasattr(source, 'warmup') or not hasattr(source, 'bonuses')
            or not source.handle_timeout_termination or type(source.full) is not bool
            or type(source.pos) is not int or not 0 <= source.pos < source.buffer_size):
        raise ValueError('incompatible OCD replay')
    expected_shapes = {'observations': (source.buffer_size, source.n_envs, *ac.obs_shape),
                       'next_observations': (source.buffer_size, source.n_envs, *ac.obs_shape),
                       'actions': (source.buffer_size, source.n_envs, ac.action_dim),
                       'rewards': (source.buffer_size, source.n_envs), 'dones': (source.buffer_size, source.n_envs),
                       'timeouts': (source.buffer_size, source.n_envs)}
    if any(getattr(source, name).shape != shape for name, shape in expected_shapes.items()):
        raise ValueError('invalid source replay array shapes')
    n_rows = source.buffer_size if source.full else source.pos
    if (source.warmup.shape != (source.buffer_size, source.n_envs) or source.warmup.dtype != np.bool_
            or source.bonuses.shape != source.warmup.shape):
        raise ValueError('invalid source warmup or bonus layout')
    eligible = np.flatnonzero(~source.warmup[:n_rows].ravel())
    n_ocd = ac.buffer_size - n_ac
    if len(eligible) < n_ocd:
        raise ValueError('not enough eligible OCD rows')
    # Independent RNG: never consume the learner's persisted training RNG.
    chosen = np.random.default_rng(seed).choice(eligible, n_ocd, replace=False)
    rows, cols = np.divmod(chosen, source.n_envs)
    bonus = source.bonuses[:n_rows].ravel()[eligible]
    if (not np.isfinite(source.bonus_weight) or not np.isfinite(bonus).all()
            or np.any(bonus * source.bonus_weight != 0)):
        raise ValueError('source effective bonus is nonzero')
    ac_order = np.r_[np.arange(ac.pos, ac.buffer_size), np.arange(ac.pos)] if ac.full else np.arange(ac.pos)
    result = ReplayBuffer(ac.buffer_size, ac.observation_space, ac.action_space, device=ac.device,
                          n_envs=1, optimize_memory_usage=False, handle_timeout_termination=True)
    for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
        values = np.asarray(getattr(source, name)[rows, cols])
        old = np.asarray(getattr(ac, name)[ac_order, 0])
        if not np.isfinite(values).all() or not np.isfinite(old).all():
            raise ValueError(f'nonfinite replay {name}')
        if name in ('dones', 'timeouts') and (not np.isin(values, [0, 1]).all() or not np.isin(old, [0, 1]).all()):
            raise ValueError(f'invalid binary {name}')
        if name == 'actions' and (np.any(np.abs(values) > 1) or np.any(np.abs(old) > 1)):
            raise ValueError('out-of-range replay actions')
        if name in ('observations', 'next_observations') and (np.any(np.abs(values) > 10) or np.any(np.abs(old) > 10)):
            raise ValueError('out-of-range replay observations')
        target = getattr(result, name)
        target[:n_ocd, 0] = values
        target[n_ocd:, 0] = old
    if np.any(result.timeouts > result.dones):
        raise ValueError('timeout without done')
    result.domain = np.r_[np.ones(n_ocd, np.uint8), np.full(n_ac, 2, np.uint8)]
    result.full, result.pos = True, 0
    return result, dict(initial_ocd_count=n_ocd, initial_ac_count=n_ac, eligible_ocd_count=len(eligible),
                        selection_seed=seed, selection='uniform without replacement over non-warmup slots',
                        source_bonus_weight=float(source.bonus_weight), source_n_steps=int(source.n_steps))


def make_fork(parent, destination, source_replay, source_config, *, seed=0):
    parent, destination = Path(parent).resolve(), Path(destination).resolve()
    source_replay, source_config = Path(source_replay).resolve(), Path(source_config).resolve()
    if destination.exists() or destination == parent or parent in destination.parents:
        raise ValueError('fork must be a new path outside the parent run')
    receipts = sorted(parent.glob('cycle-*/receipt.json'))
    if not receipts:
        raise ValueError('parent has no committed cycles')
    receipt_path = receipts[-1]
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    state = dict(receipt['state'])
    if state['cycle'] != receipt['cycle'] or state.get('pending') or not state.get('learner'):
        raise ValueError('parent cut must have a learner and no pending chunks')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        shutil.copytree(parent/'source', tmp/'source')
        shutil.copy2(parent/'server.json', tmp/'server.json')
        (tmp/'bootstrap').mkdir()
        shutil.copytree(parent/state['learner'], tmp/'parent_learner')
        # A removed/pruned parent transaction fails here or its completion hashes fail below.
        if receipt_path.read_bytes() != receipt_bytes:
            raise ValueError('parent receipt changed during snapshot')
        manifest = json.loads((tmp/'parent_learner'/'manifest.json').read_text())
        cfg = LearnerConfig(**manifest['config'])
        learner = ACLearner.resume(tmp/'parent_learner', tmp/'parent_learner'/'actor', cfg)
        if learner.provenance != state.get('replay_identity') or learner.mixed_replay is not None:
            raise ValueError('parent replay identity or domain incompatible')
        config_hash = file_sha(source_config)
        if config_hash != learner.config_hash:
            raise ValueError('OCD run configuration does not match actor source')
        source_cfg = json.loads(source_config.read_text())
        bonus = source_cfg['env']['corner_bonus']
        if bonus['mode'] != 'entry_exit' or bonus['bonus_in'] != 0 or bonus['bonus_out'] != 0:
            raise ValueError('source corner bonus configuration is not zero')
        source_hash = file_sha(source_replay)
        status_path = source_replay.parent/'status.json'
        status_bytes = status_path.read_bytes() if status_path.is_file() else None
        source_step = json.loads(status_bytes).get('replay_buffer_step') if status_bytes else None
        with source_replay.open('rb') as stream:
            source = pickle.load(stream)  # trusted, hash-recorded local artifact
        if file_sha(source_replay) != source_hash:
            raise ValueError('source replay changed during snapshot')
        if (source.n_envs != source_cfg['n_envs'] or source.n_steps != source_cfg['agent']['n_steps']
                or source.buffer_size != max(source_cfg['agent']['buffer_size'] // source.n_envs, 1)):
            raise ValueError('source replay does not match OCD training configuration')
        learner.model.replay_buffer, details = mix_replay(learner.model.replay_buffer, source, seed=seed)
        del source
        learner.mixed_replay = dict(schema='rlsetup.ac.mixed-replay.v1', **details,
                                    source_replay_sha256=source_hash, source_config_sha256=config_hash,
                                    parent_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                                    parent_cycle=state['cycle'], parent_learner_manifest_sha256=file_sha(tmp/'parent_learner'/'manifest.json'),
                                    source_replay_path=str(source_replay), source_config_path=str(source_config),
                                    source_replay_step=source_step,
                                    source_status_sha256=hashlib.sha256(status_bytes).hexdigest() if status_bytes else None,
                                    reward_semantics='native OCD and AC one-step rewards, not relabelled')
        learner.save(tmp/'bootstrap'/'learner')
        shutil.rmtree(tmp/'parent_learner')
        state.update(learner='bootstrap/learner', actor='bootstrap/learner/actor')
        genesis = dict(schema='rlsetup.ac.mixed-genesis.v1', state=state,
                       learner_manifest_sha256=file_sha(tmp/'bootstrap'/'learner'/'manifest.json'),
                       parent_receipt_text=receipt_bytes.decode('utf-8'),
                       parent_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest())
        (tmp/'genesis.json').write_text(json.dumps(genesis, sort_keys=True, allow_nan=False)+'\n')
        os.rename(tmp, destination)
        return dict(path=str(destination), cycle=state['cycle'], **details)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-replay', required=True)
    parser.add_argument('--source-config', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(make_fork(args.parent, args.output, args.source_replay, args.source_config, seed=args.seed), sort_keys=True))


if __name__ == '__main__':
    main()
