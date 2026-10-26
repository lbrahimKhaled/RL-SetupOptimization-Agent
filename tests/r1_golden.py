"""Legacy (ref_tracking) rollout digest for the R1 bit-identity test (R1 spec §7 test 1).

Kept separate from the test so the same function generated the golden hash on the pre-R1 code
(see tests/test_r1.py::test_legacy_rollout_bit_identical)."""
import hashlib
import math
import numpy as np
from rlsetup.driving.config import EnvConfig

COST_ORDER = ('progress', 'time', 'smoothness', 'tracking', 'penalty')
N_DECISIONS = 300


def _sweep(k, amplitude=.3):
    """Steering sine sweep (period 1 s) and pedal steps (throttle / brake every 0.8 s)."""
    steer = amplitude * math.sin(2 * math.pi * k / 25.)
    if (k // 20) % 2 == 0:
        return np.array([steer, .6, -1.], np.float32)
    return np.array([steer, -.5, .2], np.float32)


def _action(kind, k):
    if kind == 'off_track':
        return _sweep(k) if k < 30 else np.array([1., 1., -1.], np.float32)   # forced off track
    if kind == 'stall':
        return np.array([0., -1., 1.], np.float32)
    if kind == 'gentle':
        return _sweep(k, .01)
    return _sweep(k)


# (episode kind, reset kwargs); cycled until N_DECISIONS decisions are taken
EPISODES = (('off_track', dict(seed=21)),
            ('stall', dict(seed=22, options={'start_s': 4900., 'start_speed': 3.5})),
            ('gentle', dict(seed=23, options={'start_s': 4500., 'start_speed': 10.})),   # main straight
            ('sweep', dict(seed=24)))


def legacy_rollout(make_env, config=None):
    """Run the fixed rollout on `make_env(config)` (default EnvConfig(max_episode_s=4) =
    100-decision episodes, ref_tracking reward). Returns (sha256 hex of reset obs + per decision
    (obs float32, reward float64, step_cost float64 in COST_ORDER, terminated, truncated), list of
    episode end causes)."""
    env = make_env(config or EnvConfig(max_episode_s=4.))
    h = hashlib.sha256()
    ends, n, ep = [], 0, 0
    while n < N_DECISIONS:
        kind, kw = EPISODES[ep % len(EPISODES)]
        ep += 1
        obs, _ = env.reset(**kw)
        h.update(np.ascontiguousarray(obs, np.float32).tobytes())
        k = 0
        while n < N_DECISIONS:
            obs, r, term, trunc, info = env.step(_action(kind, k))
            k += 1
            n += 1
            h.update(np.ascontiguousarray(obs, np.float32).tobytes())
            h.update(np.float64(r).tobytes())
            h.update(np.array([info['step_cost'][c] for c in COST_ORDER], np.float64).tobytes())
            h.update(bytes([bool(term), bool(trunc)]))
            if term or trunc:
                ends.append(info['termination'] or info['truncation'])
                break
    return h.hexdigest(), ends
