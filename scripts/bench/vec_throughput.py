"""Throwaway measurement (sim-runner): SubprocVecEnv throughput of DrivingEnv, random actions
(forkserver; spawn on Windows)."""
import argparse, json, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
import numpy as np


DEMO = '--demo' in sys.argv
import gymnasium as gym


class DemoActions(gym.Wrapper):
    """Replaces the incoming action by the demo driver's (f = 0.5): long episodes."""
    def reset(self, **kw):
        from rlsetup.driving.demo_driver import DemoPolicy
        out = self.env.reset(**kw)
        self.pol = DemoPolicy(self.env, .5); self.pol.reset()
        return out
    def step(self, action):
        return self.env.step(self.pol())


def make_env(rank):
    def _f():
        from rlsetup.driving.env import DrivingEnv
        from rlsetup.driving.config import EnvConfig
        env = DrivingEnv(EnvConfig())
        if DEMO:
            env = DemoActions(env)
        env.reset(seed=1000 + rank)
        return env
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 6, 8, 10])
    p.add_argument('--steps', type=int, default=2000, help='vec steps per measurement')
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--json', type=Path)
    p.add_argument('--demo', action='store_true')
    p.add_argument('--start-method', default='spawn' if sys.platform == 'win32' else 'forkserver')
    a = p.parse_args()
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    track = Track('Shanghai'); qss_profile.load_or_build(track)   # main process first (R1)
    out = dict(machine=platform.platform(), start_method=a.start_method, results=[])
    for n in a.workers:
        for kind in (['dummy'] if n == 1 else []) + ['subproc']:
            t0 = time.perf_counter()
            venv = DummyVecEnv([make_env(i) for i in range(n)]) if kind == 'dummy' else \
                SubprocVecEnv([make_env(i) for i in range(n)], start_method=a.start_method)
            venv.seed(0); venv.reset()
            t_start = time.perf_counter() - t0
            rng = np.random.default_rng(0)
            for _ in range(a.warmup):
                venv.step(rng.uniform(-1, 1, (n, 3)).astype(np.float32))
            dones = 0
            t = time.perf_counter()
            for _ in range(a.steps):
                _, _, d, _ = venv.step(rng.uniform(-1, 1, (n, 3)).astype(np.float32))
                dones += int(d.sum())
            wall = time.perf_counter() - t
            venv.close()
            r = dict(kind=kind, workers=n, vec_steps=a.steps, decisions=n * a.steps, wall_s=wall,
                     decisions_per_s=n * a.steps / wall, per_worker=a.steps / wall,
                     episodes_ended=dones, mean_ep_decisions=n * a.steps / max(dones, 1), startup_s=t_start)
            out['results'].append(r)
            print(json.dumps(r), flush=True)
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
