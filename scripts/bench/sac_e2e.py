"""Throwaway (sim-runner): end-to-end SB3 SAC.learn() on SubprocVecEnv(forkserver) DrivingEnv,
3x256 MLPs, batch 512, torch threads 1, train_freq 1 vec step, gradient_steps g -> UTD = g / N.
Untrained policy (episodes short, like random). Reports env decisions/s and grad steps/s."""
import argparse, json, sys, time
sys.path.insert(0, '/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent')


def make_env(rank):
    def _f():
        from rlsetup.driving.env import DrivingEnv
        from rlsetup.driving.config import EnvConfig
        return DrivingEnv(EnvConfig())
    return _f


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--grad', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--env-steps', type=int, default=8000)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    qss_profile.load_or_build(Track('Shanghai'))
    venv = SubprocVecEnv([make_env(i) for i in range(a.workers)], start_method='forkserver')
    venv.seed(0)
    for g in a.grad:
        model = SAC('MlpPolicy', venv, batch_size=512, learning_starts=1024, train_freq=1,
                    gradient_steps=g, gamma=.992, policy_kwargs=dict(net_arch=[256, 256, 256]),
                    device='cpu', seed=0, verbose=0)
        model.learn(1024)          # fill learning_starts (no training yet)
        t = time.perf_counter()
        model.learn(a.env_steps, reset_num_timesteps=False)
        wall = time.perf_counter() - t
        n_env = a.env_steps
        r = dict(workers=a.workers, gradient_steps_per_vec_step=g, utd=g / a.workers,
                 env_decisions=n_env, wall_s=wall, env_decisions_per_s=n_env / wall,
                 grad_steps_per_s=(n_env / a.workers) * g / wall)
        print(json.dumps(r), flush=True)
    venv.close()
