"""Evaluation of a driving policy (D2 spec §6): flying laps, probe episodes, Q vs Monte Carlo,
lap metrics.

Conventions
  - Inference runs under torch.set_num_threads(1) (restored afterwards). Deterministic action =
    tanh(mean). A stochastic lap seeds torch with `seed` and restores the global torch RNG
    afterwards, so evaluation never shifts the training RNG stream. Eval transitions never
    enter a replay buffer (separate envs).
  - Lap index of a decision = line crossings recorded at the END of that decision (0 = run-in
    before the first crossing, 1 = lap 1, 2 = lap 2 (the flying lap), eval_laps + 1 = the
    decision that completes the last lap).
  - Edge margin [m] = (1 - |track_position|) w(s) / 2 - car half width (1.0 m), w(s) = track
    width of the grid cross-section; negative = the car's side is over the edge (approximate:
    uses the centre position, ignores yaw).
  - Progress jump (R1 §2.3, projection exploits on Track.path): a decision whose Track.path
    progress exceeds 2 x speed x dt + 1 m (speed = max of the speeds at its start and end) or is
    below -1 m.
  - Front lock: min(kappa_FL, kappa_FR) < -0.12 (longitudinal slip ratio, telemetry).
    Spin upshift: the gear went up during a decision whose end state has max |rear slip| > 0.12.
  - Soft Monte-Carlo returns from decision k (alpha = current entropy coeff.), both logged
    (QA D2 M1, lead decision: log both):
        per-step ('mc', SAC objective):
          MC_k   = sum_{j>=0} gamma^j r_{k+j} - alpha sum_{j>=1} gamma^j log pi(a_{k+j} | s_{k+j})
        n-step-consistent ('mc_nstep', n = model.n_steps):
          MC^n_k = sum_{j>=0} gamma^j r_{k+j} - alpha sum_{m>=1} gamma^{mn} log pi(a_{k+mn} | s_{k+mn})
    MC^n is the fixed point, along the trajectory, of SB3 2.9.0's n-step critic target
    (NStepReplayBuffer + SAC.train: y = sum_{j<n} gamma^j r_{k+j} + gamma^n (Q(s_{k+n}, a') -
    alpha log pi(a' | s_{k+n}))), which adds the entropy bonus only every n-th decision; so with
    n = 3 the critic learns ~1/3 of the per-step entropy bonus and Q - MC (per-step) is biased
    by about alpha E[log pi] (gamma/(1-gamma) - gamma^n/(1-gamma^n)). Compare Q with MC^n for
    over/underestimation; MC is the SAC-objective reference. Both are truncated at the episode
    end (a completed evaluation lap is NOT a terminal state of the training MDP, hence the >= 600
    decisions-left rule: gamma^600 = 0.008).
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
import numpy as np
import torch as th

TRACE_KEYS = ('s', 'x', 'y', 'speed', 'e', 'gear', 'slip_rl', 'slip_rr', 'track_position',
              'track_width', 'reward', 'lap_index', 'time_s', 'shaping', 'progress_m')


@contextmanager
def torch_threads(n):
    old = th.get_num_threads()
    th.set_num_threads(n)
    try:
        yield
    finally:
        th.set_num_threads(old)


@contextmanager
def torch_seed(seed):
    """th.manual_seed(seed) inside; the global torch RNG state is restored afterwards."""
    state = th.get_rng_state()
    th.manual_seed(int(seed))
    try:
        yield
    finally:
        th.set_rng_state(state)


def actor_of(policy):
    """SB3 Actor of a model / SACPolicy / Actor."""
    while hasattr(policy, 'actor'):
        policy = policy.actor
    return policy


def critic_of(policy):
    return policy.critic if hasattr(policy, 'critic') else policy.policy.critic


def act(actor, obs, deterministic):
    """(action (3,) float32, log pi(action|obs) or None for deterministic)."""
    o = th.as_tensor(np.asarray(obs, np.float32)[None], device=actor.device)
    with th.no_grad():
        mean, log_std, _ = actor.get_action_dist_params(o)
        if deterministic:
            return th.tanh(mean).cpu().numpy()[0].astype(np.float32), None
        a, logp = actor.action_dist.log_prob_from_params(mean, log_std)
    return a.cpu().numpy()[0].astype(np.float32), float(logp[0])


@dataclass
class LapReport:
    clean: bool                    # all eval laps timed, no failure / truncation
    lap_times_s: list
    laps: list                     # env info 'laps': per-lap time / cost components
    termination: str | None
    truncation: str | None
    decisions: int
    episode_return: float
    trace: dict                    # per-decision arrays (TRACE_KEYS + actions, kappa, Fz)
    obs: np.ndarray = field(repr=False, default=None)    # (T + 1, obs_dim) incl. reset obs
    logp: np.ndarray = field(repr=False, default=None)   # (T,) log pi of the taken actions (stochastic)

    def lap_time(self, k):
        """Time of lap k (1-based) or None."""
        return self.lap_times_s[k - 1] if len(self.lap_times_s) >= k else None

    @property
    def lap1_s(self):
        return self.lap_time(1)

    @property
    def lap2_s(self):
        return self.lap_time(2)


def flying_lap(policy, env, *, deterministic, seed, setup=None, threads=1):
    """Run one evaluation episode of `env` (DrivingEnv; evaluation mode is forced) with the
    actor of `policy`. `seed`: env reset seed and, for stochastic laps, the torch seed."""
    actor = actor_of(policy)
    e = env.unwrapped
    options = {'evaluation': True}
    if setup is not None:
        options['setup'] = setup
    rec = {k: [] for k in TRACE_KEYS}
    actions, kappa, fz, logps = [], [], [], []
    with torch_threads(threads), (torch_seed(seed) if not deterministic else _null()):
        obs, _ = env.reset(seed=int(seed), options=options)
        observations = [obs]
        total = 0.
        while True:
            a, logp = act(actor, obs, deterministic)
            obs, r, term, trunc, info = env.step(a)
            observations.append(obs)
            total += r
            st = e.state
            tel = info.get('telemetry')
            rec['s'].append(info['s_track'])
            rec['x'].append(st['x'])
            rec['y'].append(st['y'])
            rec['speed'].append(info['speed'])
            rec['e'].append(info['e_ref'])
            rec['gear'].append(info['gear'])
            rec['slip_rl'].append(st['slip_rl'])
            rec['slip_rr'].append(st['slip_rr'])
            rec['track_position'].append(info.get('track_position', math.nan))
            rec['track_width'].append(info.get('track_width_m', math.nan))
            rec['reward'].append(r)
            rec['lap_index'].append(len(info['line_crossings_s']))
            rec['time_s'].append(info['time_s'])
            rec['shaping'].append(info.get('shaping', 0.))
            rec['progress_m'].append(info['progress_m'])
            actions.append(a)
            kappa.append([tel[f'kappa_{w}'] for w in ('FL', 'FR', 'RL', 'RR')] if tel else [math.nan] * 4)
            fz.append([tel[f'Fz_{w}_N'] for w in ('FL', 'FR', 'RL', 'RR')] if tel else [math.nan] * 4)
            logps.append(math.nan if logp is None else logp)
            if term or trunc:
                break
    trace = {k: np.asarray(v, dtype=np.int64 if k in ('gear', 'lap_index') else np.float64) for k, v in rec.items()}
    trace.update(actions=np.asarray(actions, np.float32), kappa=np.asarray(kappa), Fz=np.asarray(fz))
    laps = [dict(l) for l in info['laps']]
    times = [l['lap_time_s'] for l in laps]
    clean = info['termination'] == 'lap_complete' and len(times) == e.cfg.eval_laps
    return LapReport(clean, times, laps, info['termination'], info['truncation'], len(actions), total,
                     trace, np.asarray(observations, np.float32), np.asarray(logps))


@contextmanager
def _null():
    yield


# --- metric helpers -----------------------------------------------------------------------------
def edge_margin(track_position, width_m, half_width_m=1.):
    """(1 - |track_position|) w / 2 - half_width [m] (elementwise)."""
    return (1. - np.abs(track_position)) * np.asarray(width_m) / 2. - half_width_m


def front_lock_fraction(kappa_fl, kappa_fr, threshold=-.12):
    k = np.minimum(np.asarray(kappa_fl), np.asarray(kappa_fr))
    return float(np.mean(k < threshold)) if k.size else math.nan


def spin_upshifts(gear, max_rear_slip, threshold=.12):
    """Number of decisions k >= 1 with gear[k] > gear[k-1] and max |rear slip|[k] > threshold."""
    g = np.asarray(gear)
    up = np.diff(g) > 0
    return int(np.sum(up & (np.asarray(max_rear_slip)[1:] > threshold)))


def progress_jumps(progress_m, time_s, speed):
    """Boolean per decision: Track.path progress of the decision > 2 v dt + 1 m or < -1 m, with
    v = max(speed at its start, at its end); decision 0 starts at the hand-over (progress 0,
    time 0, start speed = its end speed). NaN speeds (non-finite state) never count."""
    p, t, v = (np.asarray(x, dtype=float) for x in (progress_m, time_s, speed))
    if p.size == 0:
        return np.zeros(0, bool)
    dp, dt = np.diff(p, prepend=0.), np.diff(t, prepend=0.)
    vmax = np.maximum(v, np.r_[v[0], v[:-1]])
    return (dp > 2. * vmax * dt + 1.) | (dp < -1.)


def lap_metrics(rep, cfg, lap=2):
    """Metrics of lap `lap` of a LapReport (None where the lap was not reached). cost = legacy
    lap cost (time + smoothness + tracking), cost_r1 = time; shaping_sum = sum of the shaping F
    over the lap's decisions (0 in ref_tracking)."""
    tr = rep.trace
    m = tr['lap_index'] == lap
    out = {'lap': lap, 'decisions': int(m.sum())}
    if lap <= len(rep.laps):
        l = rep.laps[lap - 1]
        out.update({k: float(l[k]) for k in ('lap_time_s', 'time', 'smoothness', 'tracking', 'cost')})
        out['cost_r1'] = float(l.get('cost_r1', l['time']))
        out['smoothness_share'] = float(l['smoothness'] / l['cost']) if l['cost'] else math.nan
    if not m.any():
        return out
    e = np.abs(tr['e'][m])
    slip = np.maximum(np.abs(tr['slip_rl']), np.abs(tr['slip_rr']))
    # the decision before the lap supplies the previous gear of its first decision
    idx = np.flatnonzero(m)                     # contiguous: the lap index never decreases
    lo, hi = max(idx[0] - 1, 0), idx[-1] + 1
    out.update(mean_abs_e=float(np.nanmean(e)), max_abs_e=float(np.nanmax(e)),
               min_edge_margin_m=float(np.nanmin(edge_margin(tr['track_position'][m], tr['track_width'][m],
                                                             cfg.car_half_width_m))),
               front_lock_fraction=front_lock_fraction(tr['kappa'][m, 0], tr['kappa'][m, 1], cfg.front_lock_kappa),
               spin_upshifts=spin_upshifts(tr['gear'][lo:hi], slip[lo:hi], cfg.spin_slip),
               share_abs_e_gt_1m=float(np.mean(e > 1.)), share_abs_e_gt_3m=float(np.mean(e > 3.)))
    if 'shaping' in tr:
        out['shaping_sum'] = float(np.sum(tr['shaping'][m]))
    if 'progress_m' in tr:
        out['progress_jumps'] = int(np.sum(progress_jumps(tr['progress_m'], tr['time_s'], tr['speed'])[m]))
    return out


def soft_mc_returns(rewards, logp, gamma, alpha, n_steps=1):
    """MC_k for every k (module docstring): n_steps = 1 per-step soft MC, n_steps = n the
    n-step-consistent soft MC (entropy at offsets n, 2n, ... only). Rewards and log pi past the
    last decision count as 0. Computed as G_k - alpha E_k with G_k = r_k + gamma G_{k+1} and
    E_k = gamma^n (log pi_{k+n} + E_{k+n}) (E_k = 0 for k + n > T - 1)."""
    r = np.asarray(rewards, dtype=float)
    lp = np.asarray(logp, dtype=float)
    n = int(n_steps)
    if n < 1:
        raise ValueError(f'n_steps must be >= 1, got {n_steps}')
    T = len(r)
    g = np.empty_like(r)
    acc = 0.
    for k in range(T - 1, -1, -1):
        acc = r[k] + gamma * acc
        g[k] = acc
    ent = np.zeros_like(r)
    gn = gamma ** n
    for k in range(T - 1 - n, -1, -1):
        ent[k] = gn * (lp[k + n] + ent[k + n])
    return g - alpha * ent


def mc_indices(n, failed, stride=100, min_left=600):
    """Decisions k = 0, stride, 2 stride, ... with >= min_left decisions left (k .. n-1 counts
    n - k), or all of them if the episode ended in a failure."""
    ks = np.arange(0, n, stride)
    return ks if failed else ks[n - ks >= min_left]


def q_values(critic, obs, actions):
    """(Q_min, Q_mean) arrays for batches of (obs, action)."""
    with th.no_grad():
        dev = getattr(critic, 'device', None)
        qs = th.cat(critic(th.as_tensor(np.asarray(obs, np.float32), device=dev),
                           th.as_tensor(np.asarray(actions, np.float32), device=dev)), dim=1).cpu()
    return qs.min(1).values.numpy().astype(float), qs.mean(1).numpy().astype(float)


def q_vs_mc(critic, obs, actions, rewards, logp, *, gamma, alpha, failed, stride=100, min_left=600, n_steps=1):
    """Pairs (Q(s_k, a_k), MC_k, MC^n_k) for one episode; obs[k] / actions[k] / rewards[k] /
    logp[k] per decision k; n_steps = the critic's n-step horizon (model.n_steps). Returns dict
    of arrays (empty if no k qualifies): k, q_min, q_mean, mc (per-step), mc_nstep, n_steps."""
    ks = mc_indices(len(rewards), failed, stride, min_left)
    if len(ks) == 0:
        z = np.zeros(0)
        return {'k': ks, 'q_min': z, 'q_mean': z, 'mc': z, 'mc_nstep': z, 'n_steps': int(n_steps)}
    mc = soft_mc_returns(rewards, logp, gamma, alpha)[ks]
    mcn = soft_mc_returns(rewards, logp, gamma, alpha, n_steps)[ks]
    q_min, q_mean = q_values(critic, np.asarray(obs)[ks], np.asarray(actions)[ks])
    return {'k': ks, 'q_min': q_min, 'q_mean': q_mean, 'mc': mc, 'mc_nstep': mcn, 'n_steps': int(n_steps)}


def summarize_q_mc(pairs):
    """Mean differences and ratios over a list of q_vs_mc results, against the per-step soft MC
    (keys without suffix) and the n-step-consistent soft MC (suffix '_nstep')."""
    cat = lambda key: np.concatenate([p[key] for p in pairs]) if pairs else np.zeros(0)
    q_min, q_mean, mc = cat('q_min'), cat('q_mean'), cat('mc')
    if mc.size == 0:
        return {'n': 0}
    mcn = cat('mc_nstep')
    ratio = lambda q, m: float(q.mean() / m.mean()) if m.mean() != 0 else math.nan
    return {'n': int(mc.size), 'n_steps': int(pairs[0]['n_steps']),
            'mc_mean': float(mc.mean()), 'q_min_mean': float(q_min.mean()),
            'q_mean_mean': float(q_mean.mean()), 'q_min_minus_mc': float((q_min - mc).mean()),
            'q_mean_minus_mc': float((q_mean - mc).mean()), 'q_min_ratio': ratio(q_min, mc),
            'q_mean_ratio': ratio(q_mean, mc),
            'mc_nstep_mean': float(mcn.mean()), 'q_min_minus_mc_nstep': float((q_min - mcn).mean()),
            'q_mean_minus_mc_nstep': float((q_mean - mcn).mean()), 'q_min_ratio_nstep': ratio(q_min, mcn),
            'q_mean_ratio_nstep': ratio(q_mean, mcn)}


# --- probe ---------------------------------------------------------------------------------------
@dataclass
class ProbeReport:
    progress_m: list
    causes: list
    mean_progress_m: float
    wall_s: float


def make_probe_starts(env, gcfg):
    """CollapseGuard start set: s_k = k L / n + offset on Track.path, reset seed seed_base + k;
    a start whose settle raises is shifted by start_shift_m (up to max_start_shifts times)."""
    L = float(env.unwrapped.track.length)
    starts = []
    for k in range(gcfg.n_starts):
        s = k * L / gcfg.n_starts + gcfg.start_offset_m
        for shift in range(gcfg.max_start_shifts + 1):
            try:
                env.reset(seed=gcfg.seed_base + k, options={'start_s': s % L})
                break
            except RuntimeError:
                s += gcfg.start_shift_m
        else:
            raise RuntimeError(f'probe start {k}: no settle-able start found')
        starts.append({'k': k, 's': float(s % L), 'seed': gcfg.seed_base + k, 'shifts': shift})
    return starts


def probe(policy, env, starts, *, threads=1):
    """Deterministic probe episodes (env: training mode, max_episode_s = probe length) from
    `starts`; progress = Track.path distance driven [m]."""
    t0 = time.perf_counter()
    actor = actor_of(policy)
    progress, causes = [], []
    with torch_threads(threads):
        for st in starts:
            obs, _ = env.reset(seed=st['seed'], options={'start_s': st['s']})
            while True:
                a, _ = act(actor, obs, True)
                obs, _, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            progress.append(float(info['progress_m']))
            causes.append(info['termination'] or info['truncation'])
    return ProbeReport(progress, causes, float(np.mean(progress)), time.perf_counter() - t0)


# --- full evaluation (DrivingEvalCallback) -------------------------------------------------------
def save_lap_npz(rep, path):
    tr = rep.trace
    np.savez_compressed(path, **{k: tr[k] for k in tr}, lap_times_s=np.asarray(rep.lap_times_s),
                        clean=rep.clean, termination=str(rep.termination))


def evaluate_model(model, envs, cfg, *, laps_path=None):
    """Deterministic 2-lap run on envs['eval_det'] + cfg.stochastic_laps 1-lap runs on
    envs['eval_sto'] (torch seeds cfg.seed + k) + Q vs MC on the stochastic episodes.
    Returns a JSON-ready dict."""
    t0 = time.perf_counter()
    det = flying_lap(model, envs['eval_det'], deterministic=True, seed=cfg.seed)
    if laps_path is not None:
        save_lap_npz(det, laps_path)
    out = {'det_clean': det.clean, 'det_termination': det.termination, 'det_truncation': det.truncation,
           'lap1_s': det.lap1_s, 'lap2_s': det.lap2_s,
           'lap_diff_s': abs(det.lap1_s - det.lap2_s) if det.lap2_s is not None else None,
           'det_return': det.episode_return, 'det_decisions': det.decisions,
           'lap2': lap_metrics(det, cfg, lap=2)}
    critic = critic_of(model)
    alpha, gamma = float(model.alpha), float(model.gamma)
    n_steps = int(getattr(model, 'n_steps', 1) or 1)          # SB3 n-step horizon of the critic target
    sto, pairs, logps = [], [], []
    from .env import FAILURES
    for k in range(cfg.stochastic_laps):
        rep = flying_lap(model, envs['eval_sto'], deterministic=False, seed=cfg.seed + k)
        sto.append({'clean': rep.clean, 'lap_s': rep.lap1_s, 'termination': rep.termination,
                    'truncation': rep.truncation})
        logps.append(rep.logp)
        with torch_threads(1):
            pairs.append(q_vs_mc(critic, rep.obs[:-1], rep.trace['actions'], rep.trace['reward'], rep.logp,
                                 gamma=gamma, alpha=alpha, failed=rep.termination in FAILURES,
                                 stride=cfg.q_mc_stride, min_left=cfg.q_mc_min_left, n_steps=n_steps))
    clean_laps = [s['lap_s'] for s in sto if s['clean']]
    env_cfg = getattr(getattr(envs['eval_det'], 'unwrapped', None), 'cfg', None)
    out.update(reward_mode=env_cfg.reward.mode if env_cfg is not None else None,
               sto_laps=sto, sto_clean=len(clean_laps), sto_n=len(sto),
               sto_clean_rate=len(clean_laps) / len(sto) if sto else math.nan,
               sto_lap_mean_s=float(np.mean(clean_laps)) if clean_laps else None,
               q_vs_mc=summarize_q_mc(pairs), alpha=alpha,
               policy_entropy=float(-np.mean(np.concatenate(logps))) if logps else math.nan,
               n_updates=int(model._n_updates), eval_wall_s=time.perf_counter() - t0)
    return out
