"""Relabel a ref_tracking replay buffer to the time_pbrs reward (R1 spec Amendment 1 §A).

numpy only, in place, in-process (nothing written to disk). A stored transition is one decision
(action_repeat base ticks of dt_tick = 0.02 s). From the stored 1-step fields:
    R_new = R_old + S + T + [fail_old] P_old - [term_new] P_new + F
  S  smoothness b sum_i w_i da_i^2 / dt_tick; only the first tick of a decision has a nonzero
     action rate, da = action - last action (obs block [9 + 3 + 6 n_preview : + 3])
  T  tracking c sum_j e_j^2 over the decision's ticks, e_j = cubic Hermite interpolation between
     (e, de/dt) at obs and (e', de'/dt) at next_obs, de/dt = v_x sin(h) + v_y cos(h) (h = heading
     error; curvature of xi_ref ignored). The spec's linear midpoint (e + e')/2 left errors up to
     2.6e-3 per decision under a hard steering sweep; Hermite keeps them below 1e-3 (tests).
  fail_old = dones and not timeouts; P_old = RewardConfig.failure_penalty(e', dt_tick) of the
     source (50 + 3 e'^2 with the defaults); a non-finite end (next_obs all zero) uses e and T = 0
  stall    = fail_old, finite end, speed < stall_speed_mps at both ends -> timeouts = 1 (dones stay
     1): truncation, no P_new, real Phi'
  term_new = fail_old and not stall: P_new = penalty_const, Phi' = 0
  F = gamma Phi' - Phi, Phi = -shaping_k |e|  (gamma, shaping_k, penalty_const of the target)
  Failure decisions can end after one tick or with a shortened tick (off track mid-tick). Their
  duration D is recovered from next_obs's body accelerations, which env.py computes over the true
  elapsed time: a_x' = dv_x/D - r v_y, a_y' = dv_y/D + r v_x (midpoint r, v) -> least-squares D,
  rounded to the 2 ms physics step. Ticks end at min(j dt_tick, D); S uses the first tick's
  length. Where D is not identifiable (|dv| tiny, e.g. a stalled car) the full-length formula is
  used. (Addition to the spec's formula: without it off-track rows were off by up to 2.3.)
Approximations (bounded by tests/test_relabel.py against env ground truth): the intermediate
offsets are interpolated; stall vs off-track is classified from the end speeds; a non-finite end
uses T = 0; old time-limit rows (timeouts = 1) keep their flags. SB3's NStepReplayBuffer builds n-step returns at sampling time
from these 1-step fields, so the n-step targets follow the relabelled rewards and flags.
"""
import math
import numpy as np
from ocd_lap.backend import DT
from ocd_lap.simulation import CONTROL_STEPS

EGO_SIZE = 9          # = env.EGO_SIZE (not imported: env.py pulls in the simulator)


MIN_ACCEL_NORM = 1.          # [m/s^2] below this the decision duration is not identified


def _layout(obs_cfg):
    return EGO_SIZE, EGO_SIZE + 3 + 6 * obs_cfg.n_preview          # e index, last-action index


def _kin(x, o):
    """(v_x, v_y, yaw rate, a_x, a_y, heading error) of observations x (unscaled, float64)."""
    f = lambda i, sc: x[..., i].astype(np.float64) * sc
    return (f(1, o.speed_scale_mps), f(2, o.vy_scale_mps), f(3, o.yaw_rate_scale_radps),
            f(4, o.accel_scale_mps2), f(5, o.accel_scale_mps2), f(EGO_SIZE + 1, o.heading_scale_rad))


def decision_duration(obs, nxt, o):
    """Least-squares duration [s] of each decision from the body accelerations of next_obs (see
    module docstring); NaN where |(A_x, A_y)| < MIN_ACCEL_NORM."""
    vx, vy, r, _, _, _ = _kin(obs, o)
    vx1, vy1, r1, ax1, ay1, _ = _kin(nxt, o)
    rm, vxm, vym = (r + r1) / 2, (vx + vx1) / 2, (vy + vy1) / 2
    ax, ay = ax1 + rm * vym, ay1 - rm * vxm                 # = dv_x / D, dv_y / D
    n2 = ax ** 2 + ay ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        d = ((vx1 - vx) * ax + (vy1 - vy) * ay) / n2
    return np.where(n2 >= MIN_ACCEL_NORM ** 2, d, np.nan)


def _hermite(e0, d0, e1, d1, s, h):
    """Cubic Hermite value at fraction s of an interval of length h."""
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * e0 + (s ** 3 - 2 * s ** 2 + s) * h * d0
            + (-2 * s ** 3 + 3 * s ** 2) * e1 + (s ** 3 - s ** 2) * h * d1)


def relabel_to_time_pbrs(rb, env_cfg_src, env_cfg_dst):
    """Relabel `rb` (SB3 ReplayBuffer filled under env_cfg_src, reward 'ref_tracking') in place
    to env_cfg_dst ('time_pbrs'); rows: all if rb.full else [:pos]. The two EnvConfigs must be
    equal apart from the reward, and a, b, w, c, gamma must be equal. Returns JSON-ready stats."""
    rs, rd = env_cfg_src.reward, env_cfg_dst.reward
    if rs.mode != 'ref_tracking' or rd.mode != 'time_pbrs':
        raise ValueError(f'relabel needs ref_tracking -> time_pbrs, got {rs.mode} -> {rd.mode}')
    if (rs.a, rs.b, tuple(rs.w), rs.c, rs.gamma) != (rd.a, rd.b, tuple(rd.w), rd.c, rd.gamma):
        raise ValueError('relabel: a, b, w, c and gamma must be equal in source and target')
    if env_cfg_src.obs != env_cfg_dst.obs or env_cfg_src.action_repeat != env_cfg_dst.action_repeat:
        raise ValueError('relabel: observation layout / action repeat differ')
    o = env_cfg_src.obs
    i_e, i_la = _layout(o)
    k = env_cfg_src.action_repeat
    dt = CONTROL_STEPS * DT
    n = rb.buffer_size if rb.full else rb.pos
    obs, nxt = rb.observations[:n], rb.next_observations[:n]
    e = obs[..., i_e].astype(np.float64) * o.e_scale_m
    e1 = nxt[..., i_e].astype(np.float64) * o.e_scale_m
    v = obs[..., 0].astype(np.float64) * o.speed_scale_mps
    v1 = nxt[..., 0].astype(np.float64) * o.speed_scale_mps
    da = rb.actions[:n].astype(np.float64) - obs[..., i_la:i_la + 3].astype(np.float64)
    r_old = rb.rewards[:n].astype(np.float64)
    dones, touts = rb.dones[:n] > .5, rb.timeouts[:n] > .5

    fail_old = dones & ~touts
    nonfinite = fail_old & ~np.any(nxt != 0., axis=-1)
    stall = fail_old & ~nonfinite & (v < env_cfg_src.stall_speed_mps) & (v1 < env_cfg_src.stall_speed_mps)
    term_new = fail_old & ~stall
    # decision duration: k full ticks, or (failure rows, if identifiable) the recovered one
    dur = np.full(e.shape, k * dt)
    est = np.round(decision_duration(obs, nxt, o) / DT) * DT
    use = fail_old & ~nonfinite & np.isfinite(est)
    dur[use] = np.clip(est[use], DT, k * dt)
    *_, h0 = _kin(obs, o)
    vx, vy, *_ = _kin(obs, o)
    vx1, vy1, *_, h1 = _kin(nxt, o)
    de0, de1 = vx * np.sin(h0) + vy * np.cos(h0), vx1 * np.sin(h1) + vy1 * np.cos(h1)
    smooth = rs.b * np.sum(np.asarray(rs.w) * da ** 2, axis=-1) / np.minimum(dt, dur)
    track = np.zeros_like(e)
    for j in range(1, k + 1):                               # tick j ends at min(j dt, D)
        t_end = np.minimum(j * dt, dur)
        live = (j - 1) * dt < dur - 1e-9                    # the decision reached tick j
        ej = _hermite(e, de0, e1, de1, t_end / dur, dur)
        track += np.where(live, rs.c * ej ** 2, 0.)
    track = np.where(nonfinite, 0., track)
    p_old = np.where(fail_old, rs.failure_penalty(np.where(nonfinite, e, e1), dt), 0.)
    phi, phi1 = -rd.shaping_k * np.abs(e), np.where(term_new, 0., -rd.shaping_k * np.abs(e1))
    shaping = rd.gamma * phi1 - phi
    r_new = r_old + smooth + track + p_old - np.where(term_new, rd.penalty_const, 0.) + shaping
    rb.rewards[:n] = r_new.astype(rb.rewards.dtype)
    rb.timeouts[:n] = np.where(stall, 1., rb.timeouts[:n]).astype(rb.timeouts.dtype)

    ok = ~fail_old
    mean = lambda x, m: float(x[m].mean()) if m.any() else math.nan
    return {'rows': int(n), 'transitions': int(n * rb.n_envs), 'failures_old': int(fail_old.sum()),
            'stalls': int(stall.sum()), 'nonfinite': int(nonfinite.sum()), 'terminations_new': int(term_new.sum()),
            'timeouts_old': int(touts.sum()), 'mean_r_old': float(r_old.mean()), 'mean_r_new': float(r_new.mean()),
            'nonfailure_mean_delta': mean(r_new - r_old, ok), 'nonfailure_mean_smoothness': mean(smooth, ok),
            'nonfailure_mean_tracking': mean(track, ok), 'nonfailure_mean_shaping': mean(shaping, ok),
            'failure_mean_p_old': mean(p_old, fail_old),
            'failure_duration_identified': int(use.sum())}


# --- R4: speed-profile potential + corner entry / exit terms (docs/specs/R4_review.md req 4) ---------
# Track position s is not in the observation; it is recovered per chain (one env column, in time
# order from rb.pos, split at dones and wherever next_obs != the next row's obs):
#   relative s inside a chain from the stored rewards of non-terminal rows (time_pbrs, a, b, k):
#       ds = (r + b dt - (gamma Phi_e(e') - Phi_e(e))) / a,   Phi_e = -k |e|,  e = obs[9] e_scale_m
#   one anchor per chain: 2-D rigid fit (Kabsch, rotation + translation) of the 60 preview points of
#   a row (xi_ref, left and right edge at the N stations, car frame) to TrackGrid.at(s + d_k) over
#   candidate s (1 m grid, then two finer searches and a parabola); the anchor = the sampled row with
#   the sharpest fit (largest residual curvature); verified when >= ANCHOR_MIN_ROWS sampled rows give
#   the same chain origin within ANCHOR_AGREE_M and no well-conditioned row disagrees. Else the whole
#   chain is unresolved (left unchanged).
# Then per row F_v = gamma Phi_v(s', v') - Phi_v(s, v) (Phi_v' = 0 on terminal rows = dones & ~timeouts;
# v = obs[0] speed_scale_mps), added to r in float64 -> float32, and the entry / exit terms
# (speed_profile.EntryExitTracker, crossings interpolated per decision, chain start = hand-over,
# terminal rows forfeit their crossings and take pending entries back) -> rb.bonuses.
ANCHOR_SAMPLES = 24          # rows per chain tried as anchor / verification (evenly spaced; all if fewer)
ANCHOR_MATCH_RMS_M = .05     # rigid-fit RMS [m] over the 60 points at the best s: above -> row unusable
ANCHOR_UNIQUE_RMS_M = .25    # ... and the best fit farther than ANCHOR_UNIQUE_DIST_M must be worse than this
ANCHOR_UNIQUE_DIST_M = 10.
ANCHOR_CURV_H_M = .5         # residual curvature (res(s* + h) + res(s* - h) - 2 res(s*)) / h^2 [m^2 / m^2] ...
ANCHOR_MIN_CURV = .05        # ... below this a row is not well conditioned (straight preview: s not identified)
ANCHOR_AGREE_M = .05         # review req 4: verified within 0.05 m
ANCHOR_MIN_ROWS = 3          # review req 4: at >= 3 rows
_FINE = ((1.5, .05), (.06, .002))   # (half width, step) [m] of the two refinement searches


def preview_local(obs, obs_cfg):
    """(..., 3 N, 2) preview points in the car frame [m] (x forward, y left) of observations."""
    n = obs_cfg.n_preview
    i0 = EGO_SIZE + 3
    k = np.arange(1, n + 1)
    scale = np.maximum(obs_cfg.preview_horizon_m * (k / n) ** obs_cfg.preview_exponent, obs_cfg.preview_min_scale_m)
    p = obs[..., i0:i0 + 6 * n].astype(np.float64).reshape(*obs.shape[:-1], n, 3, 2) * scale[:, None, None]
    return p.reshape(*obs.shape[:-1], 3 * n, 2)


def preview_world(grid, s, obs_cfg):
    """(..., 3 N, 2) world points TrackGrid.at(s + d_k) (xi_ref, left, right per station) for s (...)."""
    n = obs_cfg.n_preview
    d = obs_cfg.preview_horizon_m * (np.arange(1, n + 1) / n) ** obs_cfg.preview_exponent
    ref, left, right = grid.at(np.asarray(s, dtype=float)[..., None] + d)
    return np.stack([ref, left, right], axis=-2).reshape(*np.shape(s), 3 * n, 2)


def rigid_residual(q, w):
    """Sum of squared distances [m^2] after the best 2-D rigid fit of point sets q -> w (..., m, 2)."""
    qc = q - q.mean(-2, keepdims=True)
    wc = w - w.mean(-2, keepdims=True)
    a = (qc * wc).sum((-1, -2))
    b = (qc[..., 0] * wc[..., 1] - qc[..., 1] * wc[..., 0]).sum(-1)
    return (qc ** 2).sum((-1, -2)) + (wc ** 2).sum((-1, -2)) - 2. * np.sqrt(a * a + b * b)


def fit_track_s(grid, local, obs_cfg, batch=256):
    """Best Track.path arc length s per row of `local` (n, 3 N, 2): coarse search over grid.s, then
    the _FINE searches and a parabola. Returns dict of arrays: s, rms [m], unique_rms [m] (best fit
    farther than ANCHOR_UNIQUE_DIST_M), curv (residual curvature at s, m^2 / m^2)."""
    L, m = grid.length, local.shape[-2]
    cand = grid.s
    wc = preview_world(grid, cand, obs_cfg)
    wc = wc - wc.mean(-2, keepdims=True)
    sw = (wc ** 2).sum((-1, -2))
    wx, wy = wc[..., 0], wc[..., 1]
    out = {k: np.empty(len(local)) for k in ('s', 'rms', 'unique_rms', 'curv')}
    for lo in range(0, len(local), batch):
        q = local[lo:lo + batch]
        qc = q - q.mean(-2, keepdims=True)
        sq = (qc ** 2).sum((-1, -2))
        a = wx @ qc[..., 0].T + wy @ qc[..., 1].T
        b = wy @ qc[..., 0].T - wx @ qc[..., 1].T
        res = sq[None] + sw[:, None] - 2. * np.sqrt(a * a + b * b)        # (n_cand, nb)
        j = res.argmin(0)
        s = cand[j]
        far = np.abs((cand[:, None] - s[None] + L / 2) % L - L / 2) > ANCHOR_UNIQUE_DIST_M
        out['unique_rms'][lo:lo + len(q)] = np.sqrt(np.maximum(np.where(far, res, np.inf).min(0), 0.) / m)
        for half, step in _FINE:
            off = np.arange(-half, half + step / 2, step)
            r = rigid_residual(q[:, None], preview_world(grid, s[:, None] + off, obs_cfg))
            k = np.clip(r.argmin(1), 1, len(off) - 2)
            i = np.arange(len(q))
            r0, r1, r2 = r[i, k - 1], r[i, k], r[i, k + 1]
            den = r0 - 2 * r1 + r2
            frac = np.where(den > 0, .5 * (r0 - r2) / np.where(den > 0, den, 1.), 0.)
            s = s + off[k] + np.clip(frac, -1., 1.) * step
        h = ANCHOR_CURV_H_M
        r = rigid_residual(q[:, None], preview_world(grid, s[:, None] + np.array([-h, 0., h]), obs_cfg))
        out['s'][lo:lo + len(q)] = s % L
        out['rms'][lo:lo + len(q)] = np.sqrt(np.maximum(r[:, 1], 0.) / m)
        out['curv'][lo:lo + len(q)] = (r[:, 0] + r[:, 2] - 2 * r[:, 1]) / h ** 2
    return out


def _check_speed_relabel(src, dst):
    rs, rd = src.reward, dst.reward
    if rs.mode != 'time_pbrs' or rd.mode != 'time_pbrs':
        raise ValueError(f'speed relabel needs time_pbrs -> time_pbrs, got {rs.mode} -> {rd.mode}')
    if rs.speed_k != 0.:
        raise ValueError(f'speed relabel needs a source without the speed potential (speed_k {rs.speed_k:g})')
    if (rs.a, rs.b, rs.shaping_k, rs.gamma) != (rd.a, rd.b, rd.shaping_k, rd.gamma):
        raise ValueError('speed relabel: a, b, shaping_k and gamma must be equal in source and target')
    if rs.a <= 0:
        raise ValueError('speed relabel needs a > 0 (progress term) to recover the distance')
    if src.obs != dst.obs or src.action_repeat != dst.action_repeat:
        raise ValueError('speed relabel: observation layout / action repeat differ')


def _chains(rb, c, n):
    """(order, starts, ends) of env column c: rows in time order and the [start, end) chains."""
    order = (rb.pos + np.arange(n)) % rb.buffer_size if rb.full else np.arange(n)
    brk = rb.dones[order, c] > .5
    link = np.zeros(n, bool)                       # next_obs[order[i]] == obs[order[i + 1]]
    for lo in range(0, n - 1, 20_000):
        hi = min(lo + 20_000, n - 1)
        link[lo:hi] = np.all(rb.next_observations[order[lo:hi], c] == rb.observations[order[lo + 1:hi + 1], c], axis=-1)
    broken = ~brk[:-1] & ~link[:-1]
    cut = np.flatnonzero(brk[:-1] | ~link[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, n]
    return order, starts, ends, int(broken.sum())


def _anchor(fit, d_rel, L):
    """(s0 or None, reason, info) from the fits of a chain's sampled rows (d_rel = their offset)."""
    ok = (fit['rms'] <= ANCHOR_MATCH_RMS_M) & (fit['unique_rms'] > ANCHOR_UNIQUE_RMS_M)
    good = ok & (fit['curv'] >= ANCHOR_MIN_CURV)
    if not good.any():
        return None, 'no_conditioned_row', {}
    s0 = (fit['s'] - d_rel) % L
    a = int(np.flatnonzero(good)[np.argmax(fit['curv'][good])])
    dev = np.abs((s0 - s0[a] + L / 2) % L - L / 2)
    agree = good & (dev <= ANCHOR_AGREE_M)
    if (good & ~agree).any():
        return None, 'conflict', {'max_dev_m': float(dev[good].max())}
    if agree.sum() < ANCHOR_MIN_ROWS:
        return None, 'too_few_rows', {}
    return float(s0[a]), None, {'rms_m': float(fit['rms'][a]), 'curv': float(fit['curv'][a]),
                                'agree_max_m': float(dev[agree].max()), 'rows': int(agree.sum())}


def relabel_speed_profile(rb, env_cfg_src, env_cfg_dst, table, grid, *, max_unresolved=None):
    """R4 relabel of the time_pbrs buffer `rb` (a BonusNStepReplayBuffer, e.g. an adopted pre-R2
    buffer) to env_cfg_dst's speed potential (reward.speed_k, speed_sigma) and, if
    env_cfg_dst.entry_exit_on, its corner entry / exit terms (rb.bonuses). `table`: the run's
    CornerTable, `grid`: the TrackGrid of xi_ref (DrivingEnv.grid). Rows of unresolved chains
    stay unchanged. If max_unresolved is given and the unresolved share of the stored transitions
    exceeds it, nothing is written ('aborted': True). Returns (stats, s) with s (buffer_size,
    n_envs) = recovered Track.path s [m] at each stored obs (NaN: unresolved / not stored).
    ValueError per _check_speed_relabel."""
    from .speed_profile import EntryExitTracker, speed_potential
    _check_speed_relabel(env_cfg_src, env_cfg_dst)
    rw, o = env_cfg_dst.reward, env_cfg_dst.obs
    L = float(table.length)
    if abs(L - grid.length) > 1e-6:
        raise ValueError(f'corner table length {L} != track grid length {grid.length}')
    dt = env_cfg_src.action_repeat * CONTROL_STEPS * DT
    cb = env_cfg_dst.corner_bonus
    terms_on = env_cfg_dst.entry_exit_on
    n = rb.buffer_size if rb.full else rb.pos
    s_rec = np.full((rb.buffer_size, rb.n_envs), np.nan)
    f_v = np.zeros((rb.buffer_size, rb.n_envs))
    bon = np.zeros((rb.buffer_size, rb.n_envs))
    resolved = np.zeros((rb.buffer_size, rb.n_envs), bool)
    reasons, anchors, kinds = {}, [], {}
    sums = {'entry': 0., 'exit': 0., 'taken_back': 0.}
    n_chains = n_breaks = 0
    dist = 0.
    for c in range(rb.n_envs):
        order, starts, ends, breaks = _chains(rb, c, n)
        n_chains += len(starts)
        n_breaks += breaks
        e = rb.observations[order, c, EGO_SIZE].astype(np.float64) * o.e_scale_m
        v = rb.observations[order, c, 0].astype(np.float64) * o.speed_scale_mps
        e1 = rb.next_observations[order, c, EGO_SIZE].astype(np.float64) * o.e_scale_m
        v1 = rb.next_observations[order, c, 0].astype(np.float64) * o.speed_scale_mps
        r = rb.rewards[order, c].astype(np.float64)
        term = (rb.dones[order, c] > .5) & ~(rb.timeouts[order, c] > .5)
        ds = (r + rw.b * dt - (rw.gamma * -rw.shaping_k * np.abs(e1) + rw.shaping_k * np.abs(e))) / rw.a
        ds = np.where(term, np.nan, ds)
        # sampled rows of every chain -> one batched fit
        samp, owner = [], []
        for k, (a, b) in enumerate(zip(starts, ends)):
            idx = np.arange(a, b) if b - a <= ANCHOR_SAMPLES else np.unique(np.linspace(a, b - 1, ANCHOR_SAMPLES).round().astype(int))
            samp.append(idx)
            owner.append(np.full(len(idx), k))
        samp, owner = np.concatenate(samp), np.concatenate(owner)
        fit = fit_track_s(grid, preview_local(rb.observations[order[samp], c], o), o)
        for k, (a, b) in enumerate(zip(starts, ends)):
            d = np.r_[0., np.cumsum(np.nan_to_num(ds[a:b - 1]))]          # offset of each row from the chain start
            sel = owner == k
            s0, why, info = _anchor({key: val[sel] for key, val in fit.items()}, d[samp[sel] - a], L)
            if s0 is None:
                reasons[why] = reasons.get(why, 0) + (b - a)
                continue
            anchors.append(info)
            rows = order[a:b]
            s = (s0 + d) % L
            s_rec[rows, c] = s
            resolved[rows, c] = True
            last_term = bool(term[b - 1])
            d_next = d + np.nan_to_num(ds[a:b])
            phi = speed_potential(table, s, v[a:b], rw.speed_k, rw.speed_sigma)[0]
            phi1 = speed_potential(table, (s0 + d_next) % L, v1[a:b], rw.speed_k, rw.speed_sigma)[0]
            if last_term:
                phi1[-1] = 0.
            f_v[rows, c] = rw.gamma * phi1 - phi
            dist += float(d_next[-1] if not last_term else d[-1])
            if terms_on:
                tr = EntryExitTracker(table, cb.bonus_in, cb.bonus_out, cb.sigma)
                tr.reset(s0)
                for i in range(b - a):
                    t = bool(term[a + i])
                    cross = [] if t else tr.tick(d[i], d_next[i], v[a + i], v1[a + i])
                    total, events = tr.settle(cross, t)
                    bon[rows[i], c] = total
                    for ev in events:
                        kinds[f"{ev['point']}_{ev['kind']}"] = kinds.get(f"{ev['point']}_{ev['kind']}", 0) + 1
                        if ev['kind'] == 'paid':
                            sums[ev['point']] += ev['bonus']
                        elif ev['kind'] == 'taken_back':
                            sums['taken_back'] -= ev['bonus']
    stored = np.zeros((rb.buffer_size, rb.n_envs), bool)
    stored[:n] = True
    total_tr = int(stored.sum())
    unresolved = total_tr - int(resolved.sum())
    fv = f_v[resolved]
    q = lambda x, p: float(np.percentile(x, p)) if x.size else None
    laps = dist / L
    stats = {'transitions': total_tr, 'chains': n_chains, 'chains_resolved': len(anchors),
             'unresolved_transitions': unresolved, 'unresolved_fraction': unresolved / total_tr if total_tr else 0.,
             'unresolved_by_reason': reasons, 'integrity_breaks': n_breaks,
             'anchor': {'rms_m_median': q(np.array([x['rms_m'] for x in anchors]), 50),
                        'rms_m_max': max((x['rms_m'] for x in anchors), default=None),
                        'agree_max_m': max((x['agree_max_m'] for x in anchors), default=None),
                        'rows_median': q(np.array([x['rows'] for x in anchors]), 50),
                        'curv_min': min((x['curv'] for x in anchors), default=None)},
             'speed_k': rw.speed_k, 'speed_sigma': rw.speed_sigma, 'laps_driven': laps,
             'f_v': {'mean': float(fv.mean()) if fv.size else None, 'std': float(fv.std()) if fv.size else None,
                     'min': float(fv.min()) if fv.size else None, 'max': float(fv.max()) if fv.size else None,
                     'p1': q(fv, 1), 'p99': q(fv, 99), 'sum_per_lap': float(fv.sum()) / laps if laps else None},
             'terms': None, 'aborted': False}
    if terms_on:
        b = bon[resolved]
        stats['terms'] = {'bonus_in': cb.bonus_in, 'bonus_out': cb.bonus_out, 'sigma': cb.sigma, 'events': kinds,
                          'entry_paid_sum': sums['entry'], 'exit_paid_sum': sums['exit'],
                          'taken_back_sum': sums['taken_back'], 'net_sum': float(b.sum()),
                          'net_per_lap': float(b.sum()) / laps if laps else None,
                          'rows_nonzero': int((b != 0).sum()), 'rows_negative': int((b < 0).sum())}
    if max_unresolved is not None and stats['unresolved_fraction'] > max_unresolved:
        stats['aborted'] = True
        return stats, s_rec
    m = resolved[:n]
    rb.rewards[:n] = np.where(m, rb.rewards[:n].astype(np.float64) + f_v[:n], rb.rewards[:n]).astype(rb.rewards.dtype)
    if terms_on:
        rb.bonuses[:n] = np.where(m, bon[:n], rb.bonuses[:n]).astype(rb.bonuses.dtype)
    import hashlib
    stats['sha256_rewards'] = hashlib.sha256(np.ascontiguousarray(rb.rewards[:n]).tobytes()).hexdigest()
    if terms_on:
        stats['sha256_bonuses'] = hashlib.sha256(np.ascontiguousarray(rb.bonuses[:n]).tobytes()).hexdigest()
    return stats, s_rec
