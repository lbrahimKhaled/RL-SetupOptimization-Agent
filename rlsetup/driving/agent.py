"""StagedSAC (SB3 2.9.0 SAC with freezable actor / entropy coefficient), model factory and the
actor output-bias initialization (D2 spec §0, §3, §4 step 1).

StagedSAC relies on SB3 2.9.0 `SAC.train()` internals (pinned in pyproject; tests assert it):
  - frozen actor: actor params get requires_grad=False. `actor_loss.backward()` still works
    (min_qf_pi depends on the critic), Adam skips params whose grad is None, so the actor stays
    bit-identical; the stray critic grads from the actor loss are zeroed by
    `critic.optimizer.zero_grad()` before the next critic step (standard SB3).
  - frozen alpha: during train() `ent_coef_optimizer` is set to None, so sac.py takes its
    fixed-ent_coef branch with `ent_coef_tensor = exp(log_ent_coef)`; it is restored afterwards,
    so `save()` (outside train()) still stores log_ent_coef and its optimizer.
  - `actor_lr` overrides the actor's lr after SB3 resets every lr at the start of train().
The freeze flags and actor_lr are pickled with the model; `_setup_model` re-applies
requires_grad after `load()`.
"""
import numpy as np
import stable_baselines3
import torch as th
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.utils import update_learning_rate
from .policy import LNSACPolicy

SB3_VERSION = '2.9.0'


class StagedSAC(SAC):
    def __init__(self, *args, **kwargs):
        self.freeze_actor = False
        self.freeze_ent_coef = False
        self.actor_lr = None
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        self.set_frozen(self.freeze_actor, self.freeze_ent_coef)   # after load(): requires_grad

    def set_frozen(self, actor: bool, ent_coef: bool):
        self.freeze_actor, self.freeze_ent_coef = bool(actor), bool(ent_coef)
        for p in self.actor.parameters():
            p.requires_grad_(not actor)

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)          # SB3 resets all lrs each train()
        if self.actor_lr is not None:
            update_learning_rate(self.actor.optimizer, self.actor_lr)
            self.logger.record('train/actor_lr', self.actor_lr)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if not (self.freeze_ent_coef and self.ent_coef_optimizer is not None):
            return super().train(gradient_steps, batch_size)
        opt = self.ent_coef_optimizer
        self.ent_coef_optimizer = None                      # sac.py: fixed-ent_coef branch
        self.ent_coef_tensor = th.exp(self.log_ent_coef.detach())
        try:
            super().train(gradient_steps, batch_size)
        finally:
            self.ent_coef_optimizer = opt

    @property
    def alpha(self):
        """Current entropy coefficient (float)."""
        if self.log_ent_coef is not None:
            return float(th.exp(self.log_ent_coef.detach()).item())
        return float(self.ent_coef_tensor.item())


def check_sb3_version():
    if stable_baselines3.__version__ != SB3_VERSION:
        raise RuntimeError(f'StagedSAC needs stable-baselines3=={SB3_VERSION} (train() internals), '
                           f'found {stable_baselines3.__version__}')


def build_model(cfg, vec_env):
    """StagedSAC for TrainConfig `cfg` on `vec_env` (CPU, seeded with cfg.seed: SB3 seeds
    python/numpy/torch and env i with seed + i)."""
    check_sb3_version()
    a = cfg.agent
    arch = list(a.net_arch)
    policy_kwargs = dict(net_arch=dict(pi=arch, qf=list(arch)), activation_fn=nn.ReLU,
                         n_critics=a.n_critics, share_features_extractor=False)
    policy = LNSACPolicy if a.critic_layernorm else 'MlpPolicy'
    return StagedSAC(policy, vec_env, learning_rate=a.learning_rate, buffer_size=a.buffer_size,
                     learning_starts=a.learning_starts, batch_size=a.batch_size, tau=a.tau,
                     gamma=cfg.env.reward.gamma, train_freq=(a.train_freq, 'step'),
                     gradient_steps=cfg.gradient_steps, n_steps=a.n_steps, ent_coef=a.ent_coef,
                     target_entropy=a.target_entropy, use_sde=False, policy_kwargs=policy_kwargs,
                     seed=cfg.seed, device='cpu', verbose=0)


def init_output_bias(actor, *, pedal_deadzone, action=(0., -.275, -.8), log_std=-1., weight_scale=.01):
    """Instructive init step 1 (DECISIONS #6): scale the mu / log_std output weights by
    `weight_scale` and set mu.bias = atanh(action), log_std.bias = log_std, so the deterministic
    action is ~`action` (default straight, 15 % throttle, brake -0.8 = inside the pedal deadzone)
    for any observation and sigma ~ exp(log_std) (pre-tanh). Valid only with the pedal deadzone
    (with the plain mapping -0.8 would be 10 % brake)."""
    assert pedal_deadzone, 'init_output_bias assumes the pedal deadzone (DECISIONS #10)'
    target = np.asarray(action, dtype=np.float64)
    assert target.shape == (actor.mu.out_features,) and np.all(np.abs(target) < 1)
    with th.no_grad():
        actor.mu.weight.mul_(weight_scale)
        actor.mu.bias.copy_(th.as_tensor(np.arctanh(target), dtype=actor.mu.bias.dtype))
        actor.log_std.weight.mul_(weight_scale)
        actor.log_std.bias.fill_(log_std)
