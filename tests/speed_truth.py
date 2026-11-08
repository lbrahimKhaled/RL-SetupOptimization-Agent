"""Ground truth for the R4 speed relabel tests (docs/specs/R4_review.md req 4, 7): identical seeds,
starts and actions under a plain time_pbrs DrivingEnv (source) and one with the speed potential +
corner entry / exit terms (target), stored like SB3 stores them (DummyVecEnv, terminal_observation
as next_obs, TimeLimit.truncated as timeout) into an NStepReplayBuffer (source) and a
BonusNStepReplayBuffer (target), with the env's Track.path s at every stored obs. The corner table
is synthetic (the default av21 car has no corner table): v_qss = V_SCALE x the p0 QSS speed on
Track.path, windows UNITS on Shanghai."""
import math
import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup.driving import corners as C
from rlsetup.driving.buffers import BonusNStepReplayBuffer
from rlsetup.driving.config import EnvConfig, RewardConfig, CornerBonusConfig

GAMMA = RewardConfig().gamma
V_SCALE = .45
UNITS = ((4600., 4700.), (300., 450.), (600., 760.), (1000., 1200.), (5250., 30.))   # (entry, exit) [m]


def table_for(track, qss, units=UNITS, v_scale=V_SCALE):
    L = float(track.length)
    n = int(math.ceil(L))
    s = np.arange(n) * L / n
    v = v_scale * np.asarray(qss.speed(s), dtype=float)
    t = C.CornerTable(L, s, v, [], 90., {'key': 'synthetic'})
    t.units = [{'name': f'U{k + 1}', 'apexes_m': [(a + (b - a) % L / 2) % L], 'v_apex': [20.], 'start_m': a,
                'exit_m': b, 'v_qss_exit': float(t.speed(b))} for k, (a, b) in enumerate(units)]
    return t


def configs(max_episode_s=20., speed_k=20.):
    src = EnvConfig(max_episode_s=max_episode_s, settle_s=0., reward=RewardConfig(mode='time_pbrs'))
    dst = EnvConfig(max_episode_s=max_episode_s, settle_s=0.,
                    reward=RewardConfig(mode='time_pbrs', speed_k=speed_k),
                    corner_bonus=CornerBonusConfig(enabled=True, mode='entry_exit'))
    return src, dst


def _pursuit(env, table, f=.9):
    """Demo-driver action towards f x v_qss(s + 15 m) of the table."""
    return env.settle_driver.act(env.state, env.s_drive, f * float(table.speed(env.s + 15.)))


def scenarios():
    """(name, seed, reset options, policy(env, table, k) -> action). Clean drive to the time limit
    across entries / exits (incl. one window through s = 0), a hand-over inside a window, a crash
    after an entry before its exit (take-back), a stall after an entry (truncation, no take-back)."""
    crash = np.array([1., 1., -1.], np.float32)
    stop = np.array([0., -1., .25], np.float32)           # brake pedal 0.5
    return [
        ('clean', 1, {'start_s': 4530., 'start_speed': 25.}, lambda env, t, k: _pursuit(env, t)),
        ('wrap', 2, {'start_s': 5150., 'start_speed': 18.}, lambda env, t, k: _pursuit(env, t, 1.)),
        ('inside', 3, {'start_s': 650., 'start_speed': 20.}, lambda env, t, k: _pursuit(env, t, 1.)),
        ('crash', 4, {'start_s': 250., 'start_speed': 22.},
         lambda env, t, k: _pursuit(env, t) if env.s < 340. else crash),
        ('stall', 5, {'start_s': 950., 'start_speed': 18.},
         lambda env, t, k: _pursuit(env, t, 1.) if env.s < 1005. else stop),
    ]


def collect(make_env, track, qss, *, max_episode_s=20., speed_k=20., table=None):
    """Returns (source buffer, target buffer, s at each stored obs (rows,), row names, table, (src, dst),
    target corner events [(row, event)], target infos per row)."""
    table = table_for(track, qss) if table is None else table
    cfgs = configs(max_episode_s, speed_k)
    envs = [make_env(c, corners=table) for c in cfgs]
    venvs = [DummyVecEnv([lambda e=e: e]) for e in envs]
    sp, ap = venvs[0].observation_space, venvs[0].action_space
    kw = dict(device='cpu', n_envs=1, n_steps=3, gamma=GAMMA, handle_timeout_termination=True)
    bufs = [NStepReplayBuffer(6000, sp, ap, **kw), BonusNStepReplayBuffer(6000, sp, ap, **kw)]
    s_true, names, events, infos_dst = [], [], [], []
    for name, seed, options, policy in scenarios():
        cur = []
        for v in venvs:
            v.seed(seed)
            v.set_options(options)
            cur.append(v.reset())
        s = envs[0].s
        for k in range(1000):
            a = np.asarray(policy(envs[0], table, k), np.float32)[None]
            steps = [v.step(a) for v in venvs]
            for buf, o, (new, r, d, infos) in zip(bufs, cur, steps):
                nxt = new.copy()
                if d[0]:
                    nxt[0] = infos[0]['terminal_observation']
                buf.add(o, nxt, a, r, d, infos)
            s_true.append(s)
            names.append(name)
            info = steps[1][3][0]
            infos_dst.append(info)
            events += [(len(names) - 1, ev) for ev in info.get('corner_events', ())]
            s = envs[0].s
            cur = [st[0] for st in steps]
            if steps[0][2][0]:
                break
    return bufs[0], bufs[1], np.asarray(s_true), names, table, cfgs, events, infos_dst
