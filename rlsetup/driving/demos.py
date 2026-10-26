"""Demonstration data for the instructive initialization (D2 spec §4 step 2, §5). No torch.

Collection: episode ep (0, 1, ...) = `env.reset(seed=seed + ep)` on the TRAINING distribution
(random s0, f_env, settle) -> `DemoPolicy(env, f_demo).reset()` with f_demo ~ U(0.35, 0.65)
-> each decision executes and records a = clip(demo + N(0, noise_std), -1, 1) until
termination / truncation (60 s). f_demo and the noise come from
`np.random.default_rng([seed, ep])`, so an episode does not depend on which worker ran it;
whole episodes are taken in order until >= n_transitions, so the dataset does not depend on the
number of workers either.

Storage `<demo_dir>/demos_<key12>.npz` + `.json` (metadata, summary, provenance). The key is
sha256 of (EnvConfig dict, DemoDataConfig without `workers`, rlsetup commit, ocd_lap
code_version, sha256 of the QSS cache key). Arrays:
  obs            (sum(T_i + 1), obs_dim) float32: each episode's T_i + 1 observations, the last
                 being the true terminal / truncation observation
  actions        (N, 3) float32 executed (noisy, clipped) actions, action space [-1, 1]
  actions_clean  (N, 3) float32 the demo driver's action before noise (addition to the spec:
                 BC's validation MSE is measured against it)
  rewards        (N,) float64 env reward; terminated, truncated (N,) bool
  ep_start_obs, ep_start_tr (E,) int64 first row of episode e in obs / in the transition arrays
  f_demo, s0     (E,) float64;  ep_cause (E,) str: failure cause, 'time_limit' or 'none'
Transition t of episode e: obs row ep_start_obs[e] + (t - ep_start_tr[e]), next obs = that + 1.

Gate: >= min_clean_fraction of the episodes end without a failure; else `ensure_demos`
recollects once with steering noise `fallback_steer_noise` (default 0.05 -> 0.025; with the
default noise the fallback runs only if 0.05 fails the gate).
"""
from collections import Counter, deque
from dataclasses import asdict, replace
from pathlib import Path
import datetime
import hashlib
import json
import multiprocessing
import numpy as np
from ..provenance import REPO_ROOT, run_provenance, sha256_file
from .config import EnvConfig
from .env import FAILURES
from .train_config import DemoDataConfig, config_from_dict
from .vec import make_env

ARRAYS = ('obs', 'actions', 'actions_clean', 'rewards', 'terminated', 'truncated', 'ep_start_obs',
          'ep_start_tr', 'f_demo', 's0', 'ep_cause')


class DemoGateFailed(RuntimeError):
    pass


def dataset_key(env_cfg, demo_cfg, qss, provenance=None):
    """(sha256 hex, the dict it hashes)."""
    prov = provenance or run_provenance()
    d = {'env': env_cfg.to_dict(),
         'demos': {k: v for k, v in asdict(demo_cfg).items() if k != 'workers'},
         'rlsetup_commit': prov['rlsetup']['commit'],
         'ocd_lap_code_version': prov['ocd_lap_code_version'],
         'qss_cache_key_sha256': hashlib.sha256(json.dumps(qss.meta['key'], sort_keys=True).encode()).hexdigest()}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(), d


def run_episode(env, demo_cfg, ep):
    """One demonstration episode on `env` (a DrivingEnv in training mode). Returns a dict of
    per-episode arrays and scalars."""
    from .demo_driver import DemoPolicy
    rng = np.random.default_rng([demo_cfg.seed, ep])
    f = float(rng.uniform(*demo_cfg.fraction_range))
    std = np.asarray(demo_cfg.noise_std, dtype=float)
    obs, info = env.reset(seed=demo_cfg.seed + ep)
    s0 = float(info['start']['s0'])
    policy = DemoPolicy(env, f)
    policy.reset()
    e = env.unwrapped
    observations, actions, clean_actions, rewards, terms, truncs = [obs], [], [], [], [], []
    speed_fraction = []
    while True:
        clean = policy()
        a = np.clip(clean + rng.normal(0., std), -1., 1.).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        observations.append(obs)
        actions.append(a)
        clean_actions.append(np.asarray(clean, np.float32))
        rewards.append(r)
        terms.append(term)
        truncs.append(trunc)
        if np.isfinite(info['speed']):
            speed_fraction.append(info['speed'] / float(e.qss.speed(info['s_track'])))
        if term or trunc:
            break
    cause = info['termination'] or info['truncation'] or 'none'
    return {'obs': np.asarray(observations, np.float32), 'actions': np.asarray(actions, np.float32),
            'actions_clean': np.asarray(clean_actions, np.float32), 'rewards': np.asarray(rewards, np.float64),
            'terminated': np.asarray(terms, bool), 'truncated': np.asarray(truncs, bool),
            'f_demo': f, 's0': s0, 'cause': cause, 'speed_fraction_sum': float(np.sum(speed_fraction)),
            'speed_fraction_n': len(speed_fraction)}


_WORKER = {}


def _init_worker(env_cfg_dict, track_sha, qss):
    _WORKER['env'] = make_env(env_cfg_dict, 0, track_sha, qss)


def _worker_episode(args):
    demo_cfg_dict, ep = args
    return run_episode(_WORKER['env'], config_from_dict(DemoDataConfig, demo_cfg_dict), ep)


def _collect_episodes(env_cfg, demo_cfg, qss, track_sha, start_method):
    """Episodes 0, 1, ... in order until >= n_transitions."""
    episodes, total = [], 0
    if demo_cfg.workers <= 1:
        env = make_env(env_cfg.to_dict(), 0, track_sha, qss)
        ep = 0
        while total < demo_cfg.n_transitions:
            episodes.append(run_episode(env, demo_cfg, ep))
            total += len(episodes[-1]['actions'])
            ep += 1
        env.close()
        return episodes
    ctx = multiprocessing.get_context(start_method)
    d = asdict(demo_cfg)
    with ctx.Pool(demo_cfg.workers, initializer=_init_worker,
                  initargs=(env_cfg.to_dict(), track_sha, qss)) as pool:
        pending, ep = deque(), 0
        while total < demo_cfg.n_transitions:
            while len(pending) < 2 * demo_cfg.workers:
                pending.append(pool.apply_async(_worker_episode, ((d, ep),)))
                ep += 1
            episodes.append(pending.popleft().get())
            total += len(episodes[-1]['actions'])
        pool.terminate()
    return episodes


def summarize(episodes):
    causes = Counter(e['cause'] for e in episodes)
    n_tr = sum(len(e['actions']) for e in episodes)
    failed = sum(causes[c] for c in FAILURES)
    sf_n = sum(e['speed_fraction_n'] for e in episodes)
    return {'episodes': len(episodes), 'transitions': n_tr, 'causes': dict(sorted(causes.items())),
            'failures': failed, 'clean_fraction': 1. - failed / max(len(episodes), 1),
            'mean_reward_per_decision': float(sum(e['rewards'].sum() for e in episodes) / max(n_tr, 1)),
            'mean_speed_fraction': sum(e['speed_fraction_sum'] for e in episodes) / max(sf_n, 1),
            'mean_episode_decisions': n_tr / max(len(episodes), 1)}


def _assemble(episodes):
    lens = np.array([len(e['actions']) for e in episodes], np.int64)
    ep_start_tr = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    out = {k: np.concatenate([e[k] for e in episodes]) for k in
           ('obs', 'actions', 'actions_clean', 'rewards', 'terminated', 'truncated')}
    out.update(ep_start_tr=ep_start_tr, ep_start_obs=ep_start_tr + np.arange(len(episodes), dtype=np.int64),
               f_demo=np.array([e['f_demo'] for e in episodes]), s0=np.array([e['s0'] for e in episodes]),
               ep_cause=np.array([e['cause'] for e in episodes], dtype='U16'))
    return out


def demo_path(demo_dir, key):
    d = Path(demo_dir)
    d = d if d.is_absolute() else REPO_ROOT / d
    return d / f'demos_{key[:12]}.npz'


def collect_demos(env_cfg, demo_cfg, qss, track_sha, demo_dir, *, start_method='forkserver', log=print):
    """Collect (or reuse the cached) dataset. Returns (npz path, metadata dict incl. summary)."""
    prov = run_provenance()
    key, key_dict = dataset_key(env_cfg, demo_cfg, qss, prov)
    path = demo_path(demo_dir, key)
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('key') == key and meta.get('npz_sha256') == sha256_file(path):
            log(f'demos: cached {path.name} ({meta["summary"]["transitions"]} transitions)')
            return path, dict(meta, cached=True)
    t0 = datetime.datetime.now(datetime.timezone.utc)
    episodes = _collect_episodes(env_cfg, demo_cfg, qss, track_sha, start_method)
    arrays = _assemble(episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)
    wall = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    summary = summarize(episodes)
    meta = {'key': key, 'key_inputs': key_dict, 'path': str(path), 'npz_sha256': sha256_file(path),
            'summary': summary, 'gate_passed': summary['clean_fraction'] >= demo_cfg.min_clean_fraction,
            'demo_config': asdict(demo_cfg), 'track_path_sha256': track_sha, 'wall_s': wall,
            'created_utc': t0.isoformat(timespec='seconds'), 'provenance': prov}
    meta_path.write_text(json.dumps(meta, indent=1))
    log(f'demos: collected {path.name}: {summary["episodes"]} episodes, {summary["transitions"]} '
        f'transitions, clean {summary["clean_fraction"]:.2f}, {wall:.0f} s')
    return path, meta


def ensure_demos(env_cfg, demo_cfg, qss, track_sha, demo_dir, *, start_method='forkserver', log=print):
    """`collect_demos` + the gate (spec §4 step 2): below min_clean_fraction, recollect once with
    steering noise `fallback_steer_noise`; raise DemoGateFailed if that fails too.
    Returns (path, meta, demo_cfg actually used)."""
    path, meta = collect_demos(env_cfg, demo_cfg, qss, track_sha, demo_dir, start_method=start_method, log=log)
    if meta['summary']['clean_fraction'] >= demo_cfg.min_clean_fraction:
        return path, meta, demo_cfg
    first = meta['summary']
    if demo_cfg.noise_std[0] <= demo_cfg.fallback_steer_noise:
        raise DemoGateFailed(f'demo gate failed: clean fraction {first["clean_fraction"]:.2f}')
    fb = replace(demo_cfg, noise_std=(demo_cfg.fallback_steer_noise, *demo_cfg.noise_std[1:]))
    log(f'demos: gate failed (clean {first["clean_fraction"]:.2f}); recollecting with steering noise '
        f'{demo_cfg.fallback_steer_noise}')
    path, meta = collect_demos(env_cfg, fb, qss, track_sha, demo_dir, start_method=start_method, log=log)
    if meta['summary']['clean_fraction'] < demo_cfg.min_clean_fraction:
        raise DemoGateFailed(f'demo gate failed after the steering-noise fallback: clean fraction '
                             f'{meta["summary"]["clean_fraction"]:.2f}')
    meta['fallback_from'] = first
    return path, meta, fb


def load_demos(path):
    """Dict of the stored arrays plus derived index arrays: obs_index (N,) obs row of each
    transition, episode (N,) episode id of each transition, ep_len (E,). 'meta' = the JSON
    sidecar if present."""
    path = Path(path)
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    lens = np.diff(np.r_[d['ep_start_tr'], len(d['actions'])])
    d['ep_len'] = lens
    d['episode'] = np.repeat(np.arange(len(lens)), lens)
    d['obs_index'] = np.arange(len(d['actions'])) + d['episode']
    meta = path.with_suffix('.json')
    d['meta'] = json.loads(meta.read_text()) if meta.exists() else None
    return d


def with_index(d):
    """Add the derived index arrays of `load_demos` to an in-memory dataset dict."""
    lens = np.diff(np.r_[d['ep_start_tr'], len(d['actions'])])
    d = dict(d, ep_len=lens, episode=np.repeat(np.arange(len(lens)), lens))
    d['obs_index'] = np.arange(len(d['actions'])) + d['episode']
    return d


def prefill_replay_buffer(rb, demos):
    """Write the demo transitions into SB3 replay buffer `rb` (spec §5).

    Episodes are split round-robin into rb.n_envs columns, each column concatenated and all
    trimmed to the shortest column length T; row r = transition r of every column.
    done = terminated or truncated; TimeLimit.truncated = truncated and not terminated. The
    last row of every column that does not end on a true termination gets done = 1 and
    TimeLimit.truncated = True, so the n-step sampler never chains a cut demo into that
    column's first online transition (it bootstraps from the stored next obs instead).
    Returns counts."""
    if 'obs_index' not in demos:
        demos = with_index(demos)
    n = rb.n_envs
    E = len(demos['ep_start_tr'])
    if E < n:
        raise ValueError(f'{E} demo episodes cannot fill {n} buffer columns')
    starts, lens = demos['ep_start_tr'], demos['ep_len']
    cols = [np.concatenate([np.arange(starts[e], starts[e] + lens[e]) for e in range(c, E, n)])
            for c in range(n)]
    T = min(len(c) for c in cols)
    if T > rb.buffer_size:
        raise ValueError(f'{T} rows exceed the buffer size {rb.buffer_size} per column')
    idx = np.stack([c[:T] for c in cols], axis=1)                    # (T, n) transition ids
    obs_i = demos['obs_index'][idx]
    term, trunc = demos['terminated'][idx], demos['truncated'][idx]
    done = term | trunc
    timeout = trunc & ~term
    tail = ~term[-1]
    done[-1, tail] = True
    timeout[-1, tail] = True
    O, A, R = demos['obs'], demos['actions'], demos['rewards']
    for r in range(T):
        rb.add(O[obs_i[r]], O[obs_i[r] + 1], A[idx[r]], R[idx[r]].astype(np.float32),
               done[r].astype(np.float32), [{'TimeLimit.truncated': bool(x)} for x in timeout[r]])
    return {'rows': int(T), 'transitions': int(T * n), 'dropped': int(len(A) - T * n),
            'flagged_tails': int(tail.sum()), 'columns': n}
