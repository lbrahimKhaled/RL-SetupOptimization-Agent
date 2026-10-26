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
