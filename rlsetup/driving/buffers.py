"""R2 replay buffer: SB3 2.9.0 NStepReplayBuffer + a separately stored, weighted corner bonus
(docs/specs/R2_review.md §R2-4).

`BonusNStepReplayBuffer` stores per transition the env's unweighted info['corner_bonus']
(`bonuses`, float32 [buffer_size, n_envs]) and whether the row was written by this class
(`labelled`, bool). At sampling time each 1-step reward of the n-step sum becomes
    r_j + w b_j            (float32; w = `bonus_weight`, set by BonusScheduleCallback)
so the fade of w(t) also applies to old transitions. `_get_samples` is a copy of SB3 2.9.0's
NStepReplayBuffer._get_samples (pinned: agent.check_sb3_version) with that one change; masks,
timeouts, the temporary pos - 1 truncation and the bootstrap discounts are unchanged, and with
w = 0 the samples are bit-identical to NStepReplayBuffer's.
`adopt(rb)` turns a loaded NStepReplayBuffer (a pre-R2 run's replay_buffer.pkl) into this class
in place (no copy of the observation arrays): bonuses 0, labelled False, bonus_weight 0. The
class lives in an importable module, so pickling (save / load / resume) keeps bonuses and labelled.
`relabel_bonuses(rb, (B_s, f_s), (B, f))` (R3_review req 5) rewrites the stored bonuses of a
buffer written with bonus B_s / floor f_s to bonus B / floor f, in place (exact up to rounding).
R4: the stored bonus may be negative (an entry term taken back after a crash; speed_profile.py);
the sampling rule r + w b is unchanged (w = 1 for a never-fade schedule).
"""
import hashlib
import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


BONUS_SUM_TOL = 1e-6      # relative slack on b <= B_s (float32 storage of B_s f, f <= 1)


def relabel_bonus_values(b, src, dst):
    """Exact bonus map (R3_review req 5). `b`: stored unweighted bonuses B_s f_s(rho) (array);
    src = (B_s, f_s), dst = (B, f) as (bonus, floor_ratio). In float64:
        rho = f_s + (1 - f_s) b / B_s,   b' = B clip((rho - f) / (1 - f), 0, 1)   -> float32.
    b = 0 (rho <= f_s, forfeit, no crossing, unlabelled row) -> 0 since f >= f_s; b = B_s (rho >= 1)
    -> B. Error vs B f(rho) on the true rho: float32 rounding of the stored b, scaled by
    (B / B_s)(1 - f_s) / (1 - f), plus the float32 rounding of b' (<= ~4e-6 for (3, .75) -> (30, .85)).
    Assumes <= 1 crossing per decision (exits >= 150 m apart, <= ~4 m per decision): the caller
    refuses b > B_s (1 + BONUS_SUM_TOL)."""
    (bs, fs), (bd, fd) = (float(src[0]), float(src[1])), (float(dst[0]), float(dst[1]))
    rho = fs + (1. - fs) * np.asarray(b, np.float64) / bs
    return (bd * np.clip((rho - fd) / (1. - fd), 0., 1.)).astype(np.float32)


def check_bonus_relabel(src, dst):
    """Config-level refusals of a relabel src -> dst ((bonus, floor_ratio) pairs): ValueError if
    the new floor is below the old one (rho below the old floor is not recoverable: b = 0 there),
    B_s <= 0 (no rho information stored) or a floor outside [0, 1)."""
    (bs, fs), (bd, fd) = (float(src[0]), float(src[1])), (float(dst[0]), float(dst[1]))
    if bs <= 0.:
        raise ValueError(f'cannot relabel bonuses written with B = {bs:g} (no exit-speed information stored)')
    if not (0. <= fs < 1. and 0. <= fd < 1.):
        raise ValueError(f'floor ratios must lie in [0, 1): {fs:g} -> {fd:g}')
    if fd < fs:
        raise ValueError(f'cannot relabel to floor {fd:g} below the source floor {fs:g} '
                         '(exit-speed ratios below the source floor were stored as 0)')
    if bd < 0.:
        raise ValueError(f'negative bonus {bd:g}')


def relabel_bonuses(rb, src, dst):
    """Relabel the stored rows of BonusNStepReplayBuffer `rb` in place from src = (B_s, f_s) to
    dst = (B, f) (relabel_bonus_values). Refuses (ValueError, nothing changed) per
    check_bonus_relabel, or if any stored b < 0 or b > B_s (1 + BONUS_SUM_TOL) (summed events).
    Returns provenance stats over the stored transitions (rows x env columns): n_transitions,
    n_b_pos (b > 0 before), n_to_zero (b > 0 before, 0 after), mean / max before and after,
    sha256_after (of the relabelled float32 bonuses of the stored rows, C order)."""
    check_bonus_relabel(src, dst)
    n = rb.buffer_size if rb.full else rb.pos
    old = rb.bonuses[:n]
    bmax = float(old.max()) if old.size else 0.
    if old.size and (float(old.min()) < 0. or bmax > float(src[0]) * (1. + BONUS_SUM_TOL)):
        raise ValueError(f'stored bonuses outside [0, B_s = {float(src[0]):g}] (min {float(old.min()):g}, '
                         f'max {bmax:g}): more than one exit per decision?')
    new = relabel_bonus_values(old, src, dst)
    pos = old > 0
    stats = {'from': {'bonus': float(src[0]), 'floor_ratio': float(src[1])},
             'to': {'bonus': float(dst[0]), 'floor_ratio': float(dst[1])},
             'n_transitions': int(old.size), 'n_b_pos': int(pos.sum()), 'n_to_zero': int((pos & (new == 0)).sum()),
             'mean_before': float(old.mean()) if old.size else 0., 'mean_after': float(new.mean()) if new.size else 0.,
             'max_before': bmax, 'max_after': float(new.max()) if new.size else 0.,
             'sha256_after': hashlib.sha256(np.ascontiguousarray(new).tobytes()).hexdigest()}
    rb.bonuses[:n] = new
    return stats


def pre_exit_cells(rb, horizon):
    """(rows, cols) of the stored transitions of `rb` (a BonusNStepReplayBuffer) that have a nonzero
    bonus within the next `horizon` decisions of the same env column (offset 0 .. horizon - 1, the
    row itself included) with no episode end (dones) before that bonus row (R3_review req 6:
    pre-exit probe). Rows are taken in time order (oldest first after a wrap), so a chain never
    crosses the write position."""
    n = rb.buffer_size if rb.full else rb.pos
    order = (rb.pos + np.arange(n)) % rb.buffer_size if rb.full else np.arange(n)
    t = np.arange(n)
    big = np.iinfo(np.int64).max
    rows, cols = [], []
    for c in range(rb.n_envs):
        b, d = rb.bonuses[order, c] != 0, rb.dones[order, c] != 0
        nb = np.minimum.accumulate(np.where(b, t, big)[::-1])[::-1]     # next bonus row (>= t)
        nd = np.minimum.accumulate(np.where(d, t, big)[::-1])[::-1]     # next episode end (>= t)
        ok = (nb < big) & (nb - t < int(horizon)) & (nd >= nb)
        rows.append(order[ok])
        cols.append(np.full(int(ok.sum()), c))
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)


def pre_point_cells(s, points, length, before_m):
    """(rows, cols) of a recovered Track.path s array (rows x env columns, NaN = unknown) that lie
    at most `before_m` [m] before any of `points` [m] (0 < (p - s) mod length <= before_m; R4_review
    req 5: pre-entry / pre-exit probe rows of the offline critic phase)."""
    s = np.asarray(s, dtype=float)
    ok = np.zeros(s.shape, bool)
    fin = np.isfinite(s)
    for p in points:
        d = np.where(fin, (float(p) - np.where(fin, s, 0.)) % float(length), np.inf)
        ok |= (d > 0.) & (d <= float(before_m))
    rows, cols = np.nonzero(ok)
    return rows.astype(np.int64), cols.astype(np.int64)


class BonusNStepReplayBuffer(NStepReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonuses = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.labelled = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.bonus_weight = 0.

    @classmethod
    def adopt(cls, rb):
        """Convert the NStepReplayBuffer `rb` in place (R2-4.4) and return it."""
        if not isinstance(rb, NStepReplayBuffer):
            raise TypeError(f'adopt needs an NStepReplayBuffer, got {type(rb).__name__}')
        if not isinstance(rb, cls):
            rb.__class__ = cls
            rb.bonuses = np.zeros((rb.buffer_size, rb.n_envs), dtype=np.float32)
            rb.labelled = np.zeros((rb.buffer_size, rb.n_envs), dtype=bool)
            rb.bonus_weight = 0.
        return rb

    def labelled_fraction(self):
        """Share of the stored transitions written with a bonus label (0 for adopted rows)."""
        n = self.buffer_size if self.full else self.pos
        return float(self.labelled[:n].mean()) if n else 0.

    def add(self, obs, next_obs, action, reward, done, infos):
        self.bonuses[self.pos] = np.array([float(info.get('corner_bonus', 0.)) for info in infos], dtype=np.float32)
        self.labelled[self.pos] = True
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        # --- copy of SB3 2.9.0 NStepReplayBuffer._get_samples; only the bonus line is new ---
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)
        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(original_timeout_values, np.logical_not(self.dones[last_valid_index]))
        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (batch_inds[:, None] + steps) % self.buffer_size
        rewards_seq = self._normalize_reward(self.rewards[indices, env_indices[:, None]], env)
        if self.bonus_weight != 0:     # R2-4.2: r + w b per 1-step reward (float32)
            rewards_seq = rewards_seq + np.float32(self.bonus_weight) * self.bonuses[indices, env_indices[:, None]]
        dones_seq = self.dones[indices, env_indices[:, None]]
        truncated_seq = self.timeouts[indices, env_indices[:, None]]
        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.argmax(axis=1)
        has_done_or_truncated = done_or_truncated.any(axis=1)
        done_idx = np.where(has_done_or_truncated, done_idx, self.n_steps - 1)
        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]
        target_q_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)
        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=1, keepdims=True)
        last_indices = (batch_inds + done_idx) % self.buffer_size
        next_obs = self._normalize_obs(self.next_observations[last_indices, env_indices], env)
        next_dones = self.dones[last_indices, env_indices][:, None].astype(np.float32)
        next_timeouts = self.timeouts[last_indices, env_indices][:, None].astype(np.float32)
        final_dones = next_dones * (1.0 - next_timeouts)
        self.timeouts[last_valid_index] = original_timeout_values
        obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        actions = self.actions[batch_inds, env_indices]
        return ReplayBufferSamples(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(final_dones),
            rewards=self.to_torch(n_step_returns),
            discounts=self.to_torch(target_q_discounts),
        )
