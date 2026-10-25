#!/usr/bin/env python3
"""D1 environment benchmark: throughput (random actions, demo driver), reset cost, and full
demo-driver flying laps (evaluation mode) at f = 0.5 and 0.7 of the p0 QSS speed.

Single process, one core. Steps are 25 Hz decisions (2 base ticks of 10 OCD steps each).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import numpy as np
from ocd_lap.track import Track
from rlsetup.driving import qss_profile
from rlsetup.driving.config import EnvConfig
from rlsetup.driving.demo_driver import DemoPolicy
from rlsetup.driving.env import DrivingEnv
from rlsetup.provenance import run_provenance


def throughput_random(env, n_decisions, seed):
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    stepped = resets = 0
    t_step = t_reset = 0.
    causes = {}
    while stepped < n_decisions:
        t = time.perf_counter()
        env.reset(seed=int(rng.integers(2 ** 31)))
        t_reset += time.perf_counter() - t
        resets += 1
        while True:
            a = env.action_space.sample()
            t = time.perf_counter()
            _, _, term, trunc, info = env.step(a)
            t_step += time.perf_counter() - t
            stepped += 1
            if term or trunc:
                key = info['termination'] or info['truncation']
                causes[key] = causes.get(key, 0) + 1
                break
    return dict(decisions=stepped, decisions_per_s=stepped / t_step,
                decisions_per_s_incl_resets=stepped / (t_step + t_reset),
                resets=resets, reset_ms=1e3 * t_reset / resets, mean_episode_decisions=stepped / resets,
                termination_causes=causes)


def throughput_demo(env, n_decisions, seed, fraction=.5):
    env.reset(seed=seed)
    policy = DemoPolicy(env, fraction)
    policy.reset()
    stepped, t_step, t_policy = 0, 0., 0.
    while stepped < n_decisions:
        t = time.perf_counter()
        a = policy()
        t_policy += time.perf_counter() - t
        t = time.perf_counter()
        _, _, term, trunc, _ = env.step(a)
        t_step += time.perf_counter() - t
        stepped += 1
        if term or trunc:
            env.reset(seed=seed + stepped)
            policy.reset()
    return dict(decisions=stepped, decisions_per_s=stepped / t_step,
                decisions_per_s_incl_policy=stepped / (t_step + t_policy))


def flying_lap(env, fraction):
    _, info0 = env.reset(options={'evaluation': True})
    policy = DemoPolicy(env, fraction)
    policy.reset()
    clipped0, values0 = env.obs_clipped_total, env.obs_values_total
    ret, max_e, n = 0., 0., 0
    t = time.perf_counter()
    while True:
        _, r, term, trunc, info = env.step(policy())
        ret += r
        n += 1
        if 'e_ref' in info:
            max_e = max(max_e, abs(info['e_ref']))
        if term or trunc:
            break
    wall = time.perf_counter() - t
    laps = info.get('laps', [])
    return dict(fraction=fraction, termination=info['termination'], truncation=info['truncation'],
                start=info0['start'], decisions=n, sim_time_s=info['time_s'], wall_s=wall,
                return_=ret, max_abs_e_m=max_e, line_crossings_s=info.get('line_crossings_s'),
                lap_times_s=[l['lap_time_s'] for l in laps],
                flying_lap=laps[1] if len(laps) > 1 else None,
                lap1_minus_lap2_s=laps[0]['lap_time_s'] - laps[1]['lap_time_s'] if len(laps) > 1 else None,
                clip_fraction=(env.obs_clipped_total - clipped0) / (env.obs_values_total - values0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decisions', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', type=Path, help='also write the results here')
    args = p.parse_args()
    track = Track('Shanghai')                     # once, before any env (R1)
    qss = qss_profile.load_or_build(track)
    env = DrivingEnv(EnvConfig(), track=track, qss=qss)
    out = dict(config=EnvConfig().to_dict(), qss_lap_time_s=qss.lap_time_s,
               observation_size=env.observation_space.shape[0],
               machine=platform.platform(), python=platform.python_version(),
               provenance=run_provenance(track.name))
    print(f'obs size {out["observation_size"]}, QSS p0 lap {qss.lap_time_s:.3f} s', flush=True)
    out['random'] = throughput_random(env, args.decisions, args.seed)
    print('random actions:', json.dumps(out['random'], indent=1), flush=True)
    out['demo'] = throughput_demo(env, args.decisions, args.seed)
    print('demo driver f=0.5:', json.dumps(out['demo'], indent=1), flush=True)
    out['flying_laps'] = []
    for f in (.5, .7):
        lap = flying_lap(env, f)
        out['flying_laps'].append(lap)
        fl = lap['flying_lap'] or {}
        print(f'\nflying lap f={f}: termination={lap["termination"]} truncation={lap["truncation"]}\n'
              f'  lap times {np.round(lap["lap_times_s"], 3).tolist()} s (lap 2 = flying lap; '
              f'lap1 - lap2 = {lap["lap1_minus_lap2_s"]:.3f} s)\n'
              f'  flying-lap cost {fl.get("cost", float("nan")):.3f} = time {fl.get("time", 0):.3f} + '
              f'smoothness {fl.get("smoothness", 0):.4f} + tracking {fl.get("tracking", 0):.3f}\n'
              f'  episode return {lap["return_"]:.1f}, max |e| {lap["max_abs_e_m"]:.3f} m, '
              f'clip fraction {lap["clip_fraction"]:.2e}, {lap["decisions"]} decisions in '
              f'{lap["wall_s"]:.1f} s wall', flush=True)
    print('\nprovenance:', json.dumps(out['provenance'], indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float))


if __name__ == '__main__':
    main()
