"""Vector env for D2 training (spec §2). This module must NOT import torch (tested): worker
factories are unpickled in fresh processes.

The main process builds `Track(name)` once (it validates / rewrites the safe-line cache single
process, D1_signoff A3), loads the p0 QSS profile, and passes sha256(Track.path) and the
(picklable) `QSSProfile` to every worker; `Track` itself is not picklable, so each worker's
`DrivingEnv` loads the track by name and asserts that its path hash matches.
Seeding: SB3 `set_random_seed(seed)` seeds env i with seed + i at the first reset.
"""
from functools import partial
import hashlib
import numpy as np
from .config import EnvConfig
from .train_config import config_from_dict


def track_sha256(track):
    """sha256 of Track.path (= DrivingEnv.env_metadata()['track']['path_sha256'])."""
    return hashlib.sha256(np.ascontiguousarray(track.path).tobytes()).hexdigest()


def make_env(env_cfg_dict, rank, track_sha, qss):
    """DrivingEnv for worker `rank` (module level, so `partial(make_env, ...)` pickles by
    reference). Raises if the worker's Track.path differs from the main process's."""
    from .env import DrivingEnv
    env = DrivingEnv(config_from_dict(EnvConfig, env_cfg_dict), qss=qss)
    sha = track_sha256(env.track)
    if sha != track_sha:
        raise RuntimeError(f'worker {rank}: Track.path sha256 {sha[:12]} != main process {str(track_sha)[:12]}')
    return env


def env_fns(env_cfg, n_envs, track, qss):
    sha = track_sha256(track)
    return [partial(make_env, env_cfg.to_dict(), i, sha, qss) for i in range(n_envs)]


def build_vec_env(env_cfg, n_envs, track, qss, *, start_method='forkserver', subprocess=True):
    """VecMonitor(SubprocVecEnv) of `n_envs` DrivingEnvs (DummyVecEnv if not `subprocess`).
    Call from inside `if __name__ == '__main__':` (forkserver re-imports the main module)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    fns = env_fns(env_cfg, n_envs, track, qss)
    venv = SubprocVecEnv(fns, start_method=start_method) if subprocess else DummyVecEnv(fns)
    return VecMonitor(venv)


def base_venv(vec_env):
    """Innermost VecEnv (VecEnvWrapper keeps its own empty `reset_infos`)."""
    while hasattr(vec_env, 'venv'):
        vec_env = vec_env.venv
    return vec_env
