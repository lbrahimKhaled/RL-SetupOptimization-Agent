"""Ground truth for the relabel tests (R1 Amendment 1 §C1/C2): identical seeds, starts and
actions under a ref_tracking and a time_pbrs DrivingEnv, stored like SB3 stores them
(DummyVecEnv, terminal_observation as next_obs, TimeLimit.truncated as timeout) into two
NStepReplayBuffers."""
import math
import numpy as np
from ocd_lap.backend import StartPose
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup.driving.config import EnvConfig, RewardConfig

GAMMA = RewardConfig().gamma


def _sweep(k, amp=.3):
    s = amp * math.sin(2 * math.pi * k / 25.)
    return np.array([s, .6, -1.], np.float32) if (k // 20) % 2 == 0 else np.array([s, -.5, .2], np.float32)


def scenarios(track, ref):
    """(name, seed, options, action(k), max decisions). Off-track crashes at several speeds and
    both steering signs, stalls from 3.5 and 2.9 m/s, a time-limit episode, wrong-way."""
    x, y = track.point(4900.)
    back = StartPose(float(x), float(y), ref.heading(4900.) + math.pi, 5.)
    out = []
    for i, seed in enumerate((1, 2, 3, 5, 8, 13)):
        sign = 1. if i % 2 == 0 else -1.
        out.append((f'off_track_{seed}', seed, None,
                    lambda k, sign=sign, t=10 + 7 * i: _sweep(k) if k < t else np.array([sign, 1., -1.], np.float32), 400))
    for speed in (3.5, 2.9, 3.2):
        out.append((f'stall_{speed}', 22, {'start_s': 4900., 'start_speed': speed},
                    lambda k: np.array([0., -1., 1.], np.float32), 400))
    out.append(('time_limit', 23, {'start_s': 4500., 'start_speed': 10.}, lambda k: _sweep(k, .01), 400))
    out.append(('wrong_way', 24, {'start_pose': back}, lambda k: np.array([0., -1., -1.], np.float32), 400))
    return out


def collect(make_env, track, ref, *, max_episode_s=4., settle_s=0.):
    """Returns (legacy buffer, time_pbrs buffer, per-row episode names, env configs (legacy, R1))."""
    cfgs = [EnvConfig(max_episode_s=max_episode_s, settle_s=settle_s, reward=r)
            for r in (RewardConfig(), RewardConfig(mode='time_pbrs'))]
    venvs = [DummyVecEnv([lambda c=c: make_env(c)]) for c in cfgs]
    sp = venvs[0].observation_space
    bufs = [NStepReplayBuffer(4000, sp, venvs[0].action_space, device='cpu', n_envs=1, n_steps=3, gamma=GAMMA,
                              handle_timeout_termination=True) for _ in cfgs]
    names = []
    for name, seed, options, action, max_k in scenarios(track, ref):
        cur = []
        for v in venvs:
            v.seed(seed)
            if options:
                v.set_options(options)
            cur.append(v.reset())
        for k in range(max_k):
            a = action(k)[None]
            steps = [v.step(a) for v in venvs]
            for buf, o, (new, r, d, infos) in zip(bufs, cur, steps):
                nxt = new.copy()
                if d[0]:
                    nxt[0] = infos[0]['terminal_observation']
                buf.add(o, nxt, a, r, d, infos)
            names.append(name)
            cur = [s[0] for s in steps]
            if steps[0][2][0]:
                break
    return bufs[0], bufs[1], names, cfgs
