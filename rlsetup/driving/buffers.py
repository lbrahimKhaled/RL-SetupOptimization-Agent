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
"""
import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


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
