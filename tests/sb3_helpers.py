"""Shared test helpers: a numpy-only env with the driving env's spaces and a small StagedSAC."""
import dataclasses
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup.driving.agent import build_model
from rlsetup.driving.train_config import TrainConfig

OBS = 159


class TinyEnv(gym.Env):
    """Numpy-only stand-in with the driving env's spaces."""
    observation_space = gym.spaces.Box(-10., 10., (OBS,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (3,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.np_random.uniform(-1, 1, OBS).astype(np.float32), {}

    def step(self, action):
        obs = self.np_random.uniform(-1, 1, OBS).astype(np.float32)
        return obs, float(self.np_random.normal()), False, False, {}


def tiny_cfg(**kw):
    c = TrainConfig()
    return dataclasses.replace(c, n_envs=1, agent=dataclasses.replace(c.agent, buffer_size=5_000, **kw))


def make_model(cfg=None, fill=2_000):
    cfg = cfg or tiny_cfg()
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    rng = np.random.default_rng(0)
    rb = model.replay_buffer
    for _ in range(fill):
        rb.add(rng.uniform(-1, 1, (1, OBS)).astype(np.float32), rng.uniform(-1, 1, (1, OBS)).astype(np.float32),
               rng.uniform(-1, 1, (1, 3)).astype(np.float32), rng.normal(size=1).astype(np.float32),
               rng.uniform(size=1) < .01, [{}])
    model.set_logger(configure(None, []))
    return model
