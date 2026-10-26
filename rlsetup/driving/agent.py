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
  - `min_train_transitions` > 0: train() does nothing (no gradient step, _n_updates unchanged)
    while the replay buffer holds fewer transitions (rows x n_envs). Set by a resume with an
    empty buffer (TrainConfig.resume_empty_buffer_min_batches); 0 = always train.
The freeze flags, actor_lr and min_train_transitions are pickled with the model;
`_setup_model` re-applies requires_grad after `load()`.

`warm_start` (R1 spec §2.5) copies actor, critics, target critics and log alpha of a trained
model into a freshly built one (optimizers, replay buffer and step counters stay fresh) and
freezes actor + alpha for the critic-only phase.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import stable_baselines3
import torch as th
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.utils import update_learning_rate
from .policy import LNSACPolicy, LayerNormContinuousCritic

SB3_VERSION = '2.9.0'


class InitMismatch(ValueError):
    """init_from cannot initialize this model (spaces, gamma, architecture or env config differ,
    or the source run's config.json is missing)."""


class StagedSAC(SAC):
    def __init__(self, *args, **kwargs):
        self.freeze_actor = False
        self.freeze_ent_coef = False
        self.actor_lr = None
        self.min_train_transitions = 0
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

    def buffer_transitions(self):
        """Transitions in the replay buffer (rows x n_envs)."""
        rb = self.replay_buffer
        return 0 if rb is None else int(rb.size()) * int(rb.n_envs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.min_train_transitions and self.buffer_transitions() < self.min_train_transitions:
            return                                          # refilling after an empty-buffer resume
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


def param_digest(obj):
    """sha256 of the parameters of a module (state_dict, key order) or of one tensor: the A4
    check that actor / log alpha stay bit-identical through a frozen phase."""
    h = hashlib.sha256()
    items = obj.state_dict().items() if isinstance(obj, nn.Module) else [('', obj)]
    for k, v in items:
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _env_without_reward(env):
    return json.loads(json.dumps({k: v for k, v in env.items() if k != 'reward'}, sort_keys=True))


def warm_start(model, init_from, cfg):
    """Initialize the fresh StagedSAC `model` (built for TrainConfig `cfg`) from the saved model
    `init_from` (R1 spec §2.5). Refuses (InitMismatch) unless observation / action space, gamma,
    policy class, net_arch, n_critics and critic LayerNorm are equal, and the source run's
    config.json (at Path(init_from).parents[1]) has the same EnvConfig apart from `reward`,
    with an equal reward.gamma. Copies the policy state (actor, critic, critic_target; only
    actor.* with cfg.warm.load_critics=False) and, with cfg.warm.load_alpha, log_ent_coef in
    place. Optimizers, replay buffer, num_timesteps and _n_updates stay fresh; no output-bias
    init. Sets model.min_train_transitions = batch_size * min_buffer_batches and freezes actor +
    alpha. Returns {'path', 'sha256', 'src_num_timesteps', 'src_alpha', 'src_config_sha256',
    'src_rlsetup_commit' (from the source provenance.json, None if absent), 'actor_sha256',
    'log_ent_coef_sha256' (after loading)}."""
    w = cfg.warm
    path = Path(init_from).resolve()
    if not path.is_file():
        raise InitMismatch(f'init_from {path} does not exist')
    run_dir = path.parents[1]
    cfg_path = run_dir / 'config.json'
    if not cfg_path.is_file():
        raise InitMismatch(f'{cfg_path} missing: cannot check the source env config')
    cfg_text = cfg_path.read_text()
    src_cfg = json.loads(cfg_text)
    src = StagedSAC.load(path, device='cpu')
    bad = []
    if src.observation_space != model.observation_space:
        bad.append(f'observation space {src.observation_space} != {model.observation_space}')
    if src.action_space != model.action_space:
        bad.append(f'action space {src.action_space} != {model.action_space}')
    if src.gamma != model.gamma:
        bad.append(f'gamma {src.gamma} != {model.gamma}')
    if type(src.policy) is not type(model.policy):
        bad.append(f'policy class {type(src.policy).__name__} != {type(model.policy).__name__}')
    for key in ('net_arch', 'n_critics'):
        a, b = src.policy_kwargs.get(key), model.policy_kwargs.get(key)
        if a != b:
            bad.append(f'{key} {a} != {b}')
    ln = lambda m: isinstance(m.critic, LayerNormContinuousCritic)
    if ln(src) != ln(model):
        bad.append(f'critic LayerNorm {ln(src)} != {ln(model)}')
    src_env = src_cfg.get('env') or {}
    if _env_without_reward(src_env) != _env_without_reward(cfg.env.to_dict()):
        bad.append('env config (without reward) differs from the source run')
    if (src_env.get('reward') or {}).get('gamma') != cfg.env.reward.gamma:
        bad.append(f"reward.gamma {(src_env.get('reward') or {}).get('gamma')} != {cfg.env.reward.gamma}")
    if bad:
        raise InitMismatch(f'cannot warm start from {path}: ' + '; '.join(bad))
    state = src.policy.state_dict()
    if w.load_critics:
        model.policy.load_state_dict(state)
    else:
        missing, unexpected = model.policy.load_state_dict(
            {k: v for k, v in state.items() if k.startswith('actor.')}, strict=False)
        assert not unexpected and all(not k.startswith('actor.') for k in missing)
    if w.load_alpha:
        if src.log_ent_coef is None or model.log_ent_coef is None:
            raise InitMismatch('load_alpha needs a learned entropy coefficient in both models')
        with th.no_grad():
            model.log_ent_coef.copy_(src.log_ent_coef)
    model.min_train_transitions = cfg.agent.batch_size * w.min_buffer_batches
    model.set_frozen(True, True)
    prov = {}
    if (run_dir / 'provenance.json').is_file():
        prov = json.loads((run_dir / 'provenance.json').read_text())
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'src_num_timesteps': int(src.num_timesteps), 'src_alpha': float(src.alpha),
            'src_config_sha256': hashlib.sha256(cfg_text.encode()).hexdigest(),
            'src_rlsetup_commit': (prov.get('rlsetup') or {}).get('commit'),
            'actor_sha256': param_digest(model.actor),
            'log_ent_coef_sha256': None if model.log_ent_coef is None else param_digest(model.log_ent_coef)}
