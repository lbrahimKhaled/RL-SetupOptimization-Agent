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
R2-8 low-noise rollouts: `low_noise_envs` (env indices) act with a = tanh(mean + scale sigma eps),
eps ~ N(0, I) from the global torch RNG, scale = `low_noise_std_scale` (0 = the deterministic
action exactly); `_sample_action` overwrites those rows after SB3's own sample, passing them
through SB3's unscale / scale steps, so the stored action is the executed action. Empty
`low_noise_envs` (default) = SB3's _sample_action, untouched. `bonus_weight` mirrors the replay
buffer's corner-bonus weight (audit only); `bonus_trigger_step` / `bonus_report_from` are the
Amendment-1 schedule state (also in bonus_state.json). All are pickled with the model.

`warm_start` (R1 spec §2.5, Amendment 1 §B) copies actor, critics, target critics, log alpha
and (WarmStartConfig.load_optimizers) the Adam states of a trained model into a freshly built
one (step counters stay 0) and freezes actor + alpha for the critic phases. `preload_buffer`
loads the source's replay buffer, relabels it to the new reward (relabel.py) and cuts the
n-step chain at its last row. R2 (corner_bonus on): the model uses buffers.BonusNStepReplayBuffer
and a pre-R2 NStepReplayBuffer is adopted in place (bonus 0, unlabelled). R4 (warm.speed_relabel):
after the adoption the buffer is relabelled to the speed potential and the corner entry / exit terms
(relabel.relabel_speed_profile); too many unresolved rows -> RelabelUnresolved (train.py then
starts from a fresh buffer).
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


class RelabelUnresolved(RuntimeError):
    """R4: the speed relabel could not recover s for more than warm.speed_relabel_max_unresolved of
    the stored transitions; nothing was written and the model keeps its own (empty) buffer.
    `stats` = relabel_speed_profile stats ('aborted': True)."""

    def __init__(self, stats):
        super().__init__(f"speed relabel: {stats['unresolved_fraction']:.4%} of the transitions unresolved")
        self.stats = stats


class StagedSAC(SAC):
    def __init__(self, *args, **kwargs):
        self.freeze_actor = False
        self.freeze_ent_coef = False
        self.actor_lr = None
        self.min_train_transitions = 0
        self.low_noise_envs = ()
        self.low_noise_std_scale = .3
        self.bonus_weight = 0.
        self.bonus_trigger_step = None
        self.bonus_report_from = None
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

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        rows = [i for i in self.low_noise_envs if i < n_envs]
        if not rows or self.num_timesteps < learning_starts:
            return action, buffer_action
        obs = self.policy.obs_to_tensor(self._last_obs)[0]      # whole batch, as predict() (same kernels)
        with th.no_grad():
            mean, log_std, _ = self.actor.get_action_dist_params(obs)
            mean, log_std = mean[rows], log_std[rows]
            a = th.tanh(mean + self.low_noise_std_scale * th.exp(log_std) * th.randn_like(mean))
        unscaled = self.policy.unscale_action(a.cpu().numpy().reshape((-1, *self.action_space.shape)))
        scaled = self.policy.scale_action(unscaled)            # the same chain as SB3 for its rows
        action, buffer_action = action.copy(), buffer_action.copy()
        buffer_action[rows] = scaled
        action[rows] = self.policy.unscale_action(scaled)
        return action, buffer_action

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
    rb = {}
    if cfg.env.corner_bonus.enabled:   # R2-4.3: SB3 adds n_steps / gamma only for its own buffer class
        from .buffers import BonusNStepReplayBuffer
        rb = dict(replay_buffer_class=BonusNStepReplayBuffer,
                  replay_buffer_kwargs={'n_steps': a.n_steps, 'gamma': cfg.env.reward.gamma})
    if any(not (isinstance(i, int) and 0 <= i < vec_env.num_envs) for i in a.low_noise_envs):
        raise ValueError(f'low_noise_envs {a.low_noise_envs} not in range(n_envs = {vec_env.num_envs})')
    model = StagedSAC(policy, vec_env, learning_rate=a.learning_rate, buffer_size=a.buffer_size,
                      learning_starts=a.learning_starts, batch_size=a.batch_size, tau=a.tau,
                      gamma=cfg.env.reward.gamma, train_freq=(a.train_freq, 'step'),
                      gradient_steps=cfg.gradient_steps, n_steps=a.n_steps, ent_coef=a.ent_coef,
                      target_entropy=a.target_entropy, use_sde=False, policy_kwargs=policy_kwargs,
                      seed=cfg.seed, device=cfg.device, verbose=0, **rb)
    model.low_noise_envs = tuple(int(i) for i in a.low_noise_envs)
    model.low_noise_std_scale = float(a.low_noise_std_scale)
    return model


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
    """Env config dict without `reward` and the R2 training-aid blocks (TRAINING_AID_KEYS): the
    part that must equal the source's for a warm start (missing aid blocks = defaults)."""
    from .config import TRAINING_AID_KEYS
    skip = ('reward', *TRAINING_AID_KEYS)
    return json.loads(json.dumps({k: v for k, v in env.items() if k not in skip}, sort_keys=True))


def warm_start(model, init_from, cfg):
    """Initialize the fresh StagedSAC `model` (built for TrainConfig `cfg`) from the saved model
    `init_from` (R1 spec §2.5). Refuses (InitMismatch) unless observation / action space, gamma,
    n_steps, target_entropy, policy class, net_arch, n_critics and critic LayerNorm are equal, and
    the source run's
    config.json (at Path(init_from).parents[1]) has the same EnvConfig apart from `reward` and
    the R2 training-aid blocks (corner_bonus, focus), with an equal reward.gamma. Copies the policy state (actor, critic, critic_target; only
    actor.* with cfg.warm.load_critics=False) and, with cfg.warm.load_alpha, log_ent_coef in
    place. Optimizers, replay buffer, num_timesteps and _n_updates stay fresh; no output-bias
    init. cfg.warm.load_optimizers: also the actor / critic / ent_coef optimizer state_dicts.
    Sets model.min_train_transitions = batch_size * min_buffer_batches (preload_buffer sets it to
    0), model.actor_lr = cfg.warm.actor_lr (it only acts after the unfreeze) and freezes actor +
    alpha. Returns {'path', 'sha256', 'src_num_timesteps', 'src_alpha', 'src_config_sha256',
    'src_rlsetup_commit' (from the source provenance.json, None if absent), 'actor_sha256',
    'log_ent_coef_sha256' (after loading), 'src_lifetime_steps_offset' (the source run's own
    status.json lifetime_steps_offset, 0 if absent), 'lifetime_steps_offset' (= that +
    src_num_timesteps: this run's step 0 on the lifetime axis, chained over warm starts; R3_review req 4)}."""
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
    for key in ('gamma', 'n_steps', 'target_entropy'):            # QA m7: + n_steps, target_entropy
        a, b = getattr(src, key, None), getattr(model, key, None)
        if a != b:
            bad.append(f'{key} {a} != {b}')
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
        if unexpected or any(k.startswith('actor.') for k in missing):     # QA m6: no assert
            raise InitMismatch(f'actor state_dict mismatch: unexpected {unexpected}, missing {missing}')
    if w.load_alpha:
        if src.log_ent_coef is None or model.log_ent_coef is None:
            raise InitMismatch('load_alpha needs a learned entropy coefficient in both models')
        with th.no_grad():
            model.log_ent_coef.copy_(src.log_ent_coef)
    if w.load_optimizers:
        model.actor.optimizer.load_state_dict(src.actor.optimizer.state_dict())
        model.critic.optimizer.load_state_dict(src.critic.optimizer.state_dict())
        if model.ent_coef_optimizer is not None and src.ent_coef_optimizer is not None:
            model.ent_coef_optimizer.load_state_dict(src.ent_coef_optimizer.state_dict())
    model.actor_lr = w.actor_lr
    model.min_train_transitions = cfg.agent.batch_size * w.min_buffer_batches
    model.set_frozen(True, True)
    prov = {}
    if (run_dir / 'provenance.json').is_file():
        prov = json.loads((run_dir / 'provenance.json').read_text())
    src_offset = 0
    if (run_dir / 'status.json').is_file():
        src_offset = int(json.loads((run_dir / 'status.json').read_text()).get('lifetime_steps_offset') or 0)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'src_num_timesteps': int(src.num_timesteps), 'src_alpha': float(src.alpha),
            'src_config_sha256': hashlib.sha256(cfg_text.encode()).hexdigest(),
            'src_rlsetup_commit': (prov.get('rlsetup') or {}).get('commit'),
            'actor_sha256': param_digest(model.actor),
            'log_ent_coef_sha256': None if model.log_ent_coef is None else param_digest(model.log_ent_coef),
            'src_lifetime_steps_offset': src_offset, 'lifetime_steps_offset': src_offset + int(src.num_timesteps)}


def default_init_buffer(init_from):
    """<run folder of init_from>/replay_buffer.pkl (init_from = <run>/checkpoints/<name>.zip)."""
    return str(Path(init_from).resolve().parents[1] / 'replay_buffer.pkl')


def preload_buffer(model, path, init_from, cfg, *, table=None, grid=None, aux=None):
    """Amendment 1 §B step 3: load the replay buffer at `path` into `model`; refuse (InitMismatch)
    unless buffer class, buffer_size, n_envs, n_steps, gamma, observation and action shapes equal
    the model's own buffer; with cfg.warm.relabel relabel it from the source env config (config.json
    of init_from's run folder) to cfg.env (relabel.relabel_to_time_pbrs; refused unless the source
    reward is ref_tracking); mark the last row of every env column as an episode end
    (callbacks.cut_nstep_chains); min_train_transitions = 0. R2-4.4: a plain NStepReplayBuffer is
    accepted for a BonusNStepReplayBuffer model and adopted in place (bonus 0, labelled False).
    R3_review req 5: a loaded BonusNStepReplayBuffer whose source config.json has env.corner_bonus
    enabled with (bonus, floor_ratio) != cfg.env.corner_bonus gets its stored bonuses relabelled exactly
    (buffers.relabel_bonuses); refused (InitMismatch) if the new floor is below the source floor or a
    stored bonus exceeds the source B. Returns {'path', 'sha256', 'rows', 'transitions', 'full',
    'cut_row', 'relabel' (stats or None), 'adopted', 'labelled_fraction' (None without the bonus
    buffer), 'bonus_relabel' (relabel_bonuses stats or None), 'speed_relabel' (stats or None)}.
    R4 (cfg.warm.speed_relabel; review req 4): after the adoption, relabel.relabel_speed_profile with
    the run's corner `table` and xi_ref TrackGrid `grid`; the source must be time_pbrs without the
    speed potential (refused before the load otherwise); an unresolved share above
    cfg.warm.speed_relabel_max_unresolved raises RelabelUnresolved (model buffer untouched).
    `aux` (dict): receives 's', the recovered Track.path s of every stored row (NaN: unresolved)."""
    from stable_baselines3.common.save_util import load_from_pkl
    from .callbacks import cut_nstep_chains
    from .config import EnvConfig
    from .relabel import relabel_to_time_pbrs, relabel_speed_profile, _check_speed_relabel
    from .train_config import config_from_dict
    from .buffers import BonusNStepReplayBuffer, relabel_bonuses, check_bonus_relabel
    from stable_baselines3.common.buffers import NStepReplayBuffer
    path = Path(path).resolve()
    if not path.is_file():
        raise InitMismatch(f'init_buffer {path} missing')
    src_env = None
    speed = cfg.warm.speed_relabel
    if cfg.warm.relabel or speed or isinstance(model.replay_buffer, BonusNStepReplayBuffer):
        cfg_path = Path(init_from).resolve().parents[1] / 'config.json'
        src_env = config_from_dict(EnvConfig, json.loads(cfg_path.read_text())['env'])
    if speed:                # R4: refuse before loading the 1.3 GB file
        if table is None or grid is None:
            raise ValueError('warm.speed_relabel needs the corner table and the track grid')
        if cfg.warm.relabel:
            raise InitMismatch('warm.relabel (R1) and warm.speed_relabel (R4) cannot be combined')
        try:
            _check_speed_relabel(src_env, cfg.env)
        except ValueError as e:
            raise InitMismatch(f'cannot speed-relabel {path}: {e}') from None
    if cfg.env.entry_exit_on and not speed and src_env.corner_bonus.enabled:
        raise InitMismatch(f'{path} holds exit bonuses; the R4 entry / exit terms need warm.speed_relabel')
    pair = lambda cb: (float(cb.bonus), float(cb.floor_ratio))
    bonus_src = bonus_dst = None
    if (isinstance(model.replay_buffer, BonusNStepReplayBuffer) and src_env.corner_bonus.enabled
            and cfg.env.corner_bonus.mode == 'exit' and pair(src_env.corner_bonus) != pair(cfg.env.corner_bonus)):
        bonus_src, bonus_dst = pair(src_env.corner_bonus), pair(cfg.env.corner_bonus)
        try:                 # config-level refusal before loading the 1.3 GB file
            check_bonus_relabel(bonus_src, bonus_dst)
        except ValueError as e:
            raise InitMismatch(f'cannot relabel the bonuses of {path}: {e}') from None
    if cfg.warm.relabel:     # R2-10.10: refuse before loading the 1.3 GB file
        if src_env.reward.mode != 'ref_tracking':
            raise InitMismatch(f'cannot relabel {path}: the source reward is already {src_env.reward.mode!r} '
                               '(relabel needs ref_tracking; set warm.relabel=False)')
    rb, cur = load_from_pkl(path), model.replay_buffer
    adopted = False
    if isinstance(cur, BonusNStepReplayBuffer) and type(rb) is NStepReplayBuffer:
        BonusNStepReplayBuffer.adopt(rb)
        adopted = True
    bad = []
    if type(rb) is not type(cur):
        bad.append(f'buffer class {type(rb).__name__} != {type(cur).__name__}')
    for key in ('buffer_size', 'n_envs', 'n_steps', 'gamma', 'obs_shape', 'action_dim'):
        a, b = getattr(rb, key, None), getattr(cur, key, None)
        if a != b:
            bad.append(f'{key} {a} != {b}')
    if bad:
        raise InitMismatch(f'cannot preload {path}: ' + '; '.join(bad))
    bonus_stats = None
    if bonus_src is not None and not adopted:      # an adopted buffer holds no bonuses (all 0)
        try:
            bonus_stats = relabel_bonuses(rb, bonus_src, bonus_dst)
        except ValueError as e:
            raise InitMismatch(f'cannot relabel the bonuses of {path}: {e}') from None
    stats = None
    if cfg.warm.relabel:
        stats = relabel_to_time_pbrs(rb, src_env, cfg.env)
    speed_stats = None
    if speed:
        speed_stats, s_rec = relabel_speed_profile(rb, src_env, cfg.env, table, grid,
                                                   max_unresolved=cfg.warm.speed_relabel_max_unresolved)
        if speed_stats['aborted']:
            raise RelabelUnresolved(speed_stats)
        if aux is not None:
            aux['s'] = s_rec
    rb.device = model.device
    model.replay_buffer = rb
    cut = cut_nstep_chains(rb)
    model.min_train_transitions = 0
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    lab = rb.labelled_fraction() if isinstance(rb, BonusNStepReplayBuffer) else None
    return {'path': str(path), 'sha256': h.hexdigest(), 'rows': int(rb.size()), 'transitions': int(rb.size() * rb.n_envs),
            'full': bool(rb.full), 'cut_row': None if cut is None else int(cut), 'relabel': stats,
            'adopted': adopted, 'labelled_fraction': lab, 'bonus_relabel': bonus_stats, 'speed_relabel': speed_stats}
