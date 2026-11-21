"""bcn_f1_r5 env stream digest for the R6 bit-identity test (docs/specs/R6_throttle_remap.md §4.1).

Kept separate from the test so the same function generated the golden hash on the pre-R6 code (HEAD 7bf3cd0,
extracted with git archive, ocd_first_lap-f1). Needs the sf70h_f1 car (ocd_first_lap branch sf70h-f1)."""
import dataclasses
import hashlib
import math
import numpy as np

N_DECISIONS = 200
THROTTLES = (-1., -.5, .3, .6, .7, .75, .85, .9, 1.)     # crosses the R6 knee and saturation


def bcn_r5_env(env_cfg=None, **replace):
    """DrivingEnv of preset bcn_f1_r5 (training mode, 8 s episodes), built from the caches."""
    from ocd_lap.track import Track
    from rlsetup.driving import corners, qss_profile
    from rlsetup.driving.env import DrivingEnv
    from rlsetup.driving.train_config import PRESETS
    cfg = env_cfg if env_cfg is not None else PRESETS['bcn_f1_r5']().env
    cfg = dataclasses.replace(cfg, max_episode_s=8., **replace)
    track = Track(cfg.track)
    qss = qss_profile.for_env(track, cfg)
    table = corners.for_env(track, cfg, qss)
    return DrivingEnv(cfg, track=track, qss=qss, corners=table)


def action(k):
    steer = .05 * math.sin(2 * math.pi * k / 25.)
    thr = THROTTLES[(k // 7) % len(THROTTLES)]
    return np.array([steer, thr, -1. if thr > -.9 else .1], np.float32)


def stream(env):
    """sha256 of reset obs + per decision (obs float32, reward float64, terminated, truncated, executed pedals),
    episodes with reset seeds 31, 32, ... until N_DECISIONS decisions."""
    h = hashlib.sha256()
    n, ep = 0, 0
    while n < N_DECISIONS:
        obs, _ = env.reset(seed=31 + ep)
        ep += 1
        h.update(np.ascontiguousarray(obs, np.float32).tobytes())
        k = 0
        while n < N_DECISIONS:
            a = action(k)
            inp = env.to_inputs(a)
            obs, r, term, trunc, info = env.step(a)
            k += 1
            n += 1
            h.update(np.ascontiguousarray(obs, np.float32).tobytes())
            h.update(np.float64(r).tobytes())
            h.update(np.array([inp.throttle, inp.brake, inp.steering_rad], np.float64).tobytes())
            h.update(bytes([bool(term), bool(trunc)]))
            if term or trunc:
                break
    return h.hexdigest()


if __name__ == '__main__':
    print(stream(bcn_r5_env()))
