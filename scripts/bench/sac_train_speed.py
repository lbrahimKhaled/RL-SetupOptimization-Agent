"""Throwaway measurement (sim-runner): SB3 SAC train() gradient steps/s, 3x256 MLPs, batch 512,
obs 159, act 3, at given torch thread counts on --device cpu / cuda / mps (dummy env, random
replay data)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch


class Dummy(gym.Env):
    observation_space = gym.spaces.Box(-10., 10., (159,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (3,), np.float32)
    def reset(self, *, seed=None, options=None):
        return np.zeros(159, np.float32), {}
    def step(self, a):
        return np.zeros(159, np.float32), 0., False, False, {}


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--json', type=Path)
    a = p.parse_args()
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure
    res = []
    for th in a.threads:
        torch.set_num_threads(th)
        torch.manual_seed(0)
        model = SAC('MlpPolicy', Dummy(), buffer_size=100_000, batch_size=a.batch, learning_starts=0,
                    policy_kwargs=dict(net_arch=[256, 256, 256]), device=a.device, seed=0, verbose=0)
        model.set_logger(configure(None, []))
        rng = np.random.default_rng(0)
        for _ in range(20_000):
            model.replay_buffer.add(rng.standard_normal((1, 159)).astype(np.float32),
                                    rng.standard_normal((1, 159)).astype(np.float32),
                                    rng.uniform(-1, 1, (1, 3)).astype(np.float32),
                                    rng.standard_normal(1).astype(np.float32), np.zeros(1), [{}])
        model.train(gradient_steps=a.warmup, batch_size=a.batch)
        sync(a.device)
        t = time.perf_counter()
        model.train(gradient_steps=a.steps, batch_size=a.batch)
        sync(a.device)
        wall = time.perf_counter() - t
        n_params = sum(p.numel() for p in model.policy.parameters())
        r = dict(device=a.device, threads=th, torch_threads=torch.get_num_threads(), batch=a.batch,
                 grad_steps=a.steps, wall_s=wall, grad_steps_per_s=a.steps / wall, policy_params=n_params)
        res.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        a.json.write_text(json.dumps(dict(torch=torch.__version__, results=res), indent=2))


if __name__ == '__main__':
    main()
