"""All numbers of D2 (SAC driving-agent training). Nothing numeric lives in train.py.

Spec: docs/specs/D2_agent.md (binding). Step counts are transitions summed over envs (SB3
`num_timesteps`); one transition = one 25 Hz decision of one env. Times in s unless the name
says otherwise. PLACEHOLDER marks numbers not yet justified by data or a decision.

The discount gamma is NOT here: it is `EnvConfig.reward.gamma` (0.992 per decision), part of the
frozen driver definition (D1 config.py).

This module must not import torch (rlsetup.driving.vec imports `config_from_dict`).
"""
from dataclasses import dataclass, field, asdict, fields, is_dataclass, replace
import ast
import json
import typing
from .config import EnvConfig


@dataclass(frozen=True)
class AgentConfig:
    """SB3 2.9.0 SAC hyperparameters (spec §0) and the instructive initialization (§3, §4)."""
    learning_rate: float = 3e-4          # every Adam (actor, critics, alpha) unless actor_lr overrides
    batch_size: int = 512
    buffer_size: int = 1_000_000         # transitions (SB3 divides by n_envs per column)
    tau: float = .005
    n_steps: int = 3                     # n-step returns (SB3 NStepReplayBuffer when > 1)
    train_freq: int = 1                  # vector steps between train() calls
    learning_starts: int = 0             # > 0 would make SB3 act uniformly at random (off_policy_algorithm.py)
    ent_coef: str = 'auto_0.1'           # learned alpha, initial 0.1
    target_entropy: str = 'auto'         # = -dim(A) = -3
    net_arch: tuple = (256, 256, 256)    # actor and each critic, ReLU
    n_critics: int = 2
    critic_layernorm: bool = True        # LayerNorm after every hidden Linear of the critics
    init_mode: str = 'bc'                # 'bc' (default) | 'bias_only' (ablation/fallback: steps 1 + 5)
    init_action: tuple = (0., -.275, -.8)   # deterministic action after init_output_bias:
                                         # straight, 15 % throttle, brake in the deadzone
    init_log_std: float = -1.            # log_std bias -> sigma ~ 0.37 (pre-tanh)
    init_weight_scale: float = .01       # mu / log_std output weights scaled by this


@dataclass(frozen=True)
class DemoDataConfig:
    """Demonstration data (spec §4 step 2). Everything except `workers` enters the dataset key."""
    n_transitions: int = 200_000         # whole episodes are collected until at least this many
    fraction_range: tuple = (.35, .65)   # f_demo ~ U(lo, hi) per episode (speed fraction of v_qss)
    noise_std: tuple = (.05, .1, .1)     # i.i.d. Gaussian per dim [steer, throttle, brake] per decision,
                                         # action units. Steering 0.05 = lead decision 26 Oct (spec said
                                         # 0.1): measured clean fraction 0.41 at 0.1, 0.86 at 0.05
    fallback_steer_noise: float = .025   # gate failed -> recollect once with this steering noise (only
                                         # if noise_std[0] is larger). PLACEHOLDER: 0.025 not measured
    min_clean_fraction: float = .8       # gate: share of episodes without a failure
    seed: int = 12345                    # independent of the training seed; reset seed = seed + ep
    workers: int = 4                     # collection processes (does not change the data)


@dataclass(frozen=True)
class BCConfig:
    """Behaviour cloning of the actor (spec §4 step 3)."""
    learning_rate: float = 3e-4          # own Adam, not actor.optimizer
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 3                    # early stop on validation NLL; best epoch is kept
    val_fraction: float = .1             # split by episode
    split_seed: int = 0                  # same split for every training seed
    action_clip: float = .995            # |a| clip before atanh in the log-likelihood
    gate_stochastic_laps: int = 3        # informative stochastic laps after the deterministic gate lap
    gate_required: bool = True           # False: record a failed gate and continue (tests only)
    slip_bins: tuple = (0., .04, .08, .12, .16, .2, .3, .5, 1.)   # A1 report: max rear slip bins


@dataclass(frozen=True)
class WarmupConfig:
    """Critic warm-up (spec §4 steps 4a/4b) and unfreeze (step 5)."""
    offline_blocks: int = 20             # step 4a: blocks of model.train(offline_steps_per_block)
    offline_steps_per_block: int = 1000
    q_probe_batch: int = 512             # fixed demo batch for the mean-Q log per block
    q_mc_max_episodes: int = 20          # demo episodes for Q vs soft MC at the end of step 4a
    online_warmup_steps: int = 20_000    # step 4b: actor + alpha frozen until num_timesteps >= this


@dataclass(frozen=True)
class GuardConfig:
    """CollapseGuard (spec §6): probe starts, schedule and rollback."""
    n_starts: int = 8                    # s_k = k L / n + start_offset_m
    start_offset_m: float = 37.
    start_shift_m: float = 50.           # shift a start that raises during settle by this
    max_start_shifts: int = 10
    seed_base: int = 1000                # reset seed of start k = seed_base + k
    probe_s: float = 30.                 # probe episode length (max_episode_s of the probe env)
    interval: int = 10_000               # probe every this many steps ...
    window: int = 200_000                # ... from unfreeze to unfreeze + window
    threshold: float = .5                # collapse: mean probe progress < threshold x BC baseline
    rollback_actor_lr: float = 1e-4
    max_rollbacks: int = 1               # the next collapse stops the run (status 'collapsed')


@dataclass(frozen=True)
class EvalConfig:
    """DrivingEvalCallback (spec §6)."""
    interval: int = 50_000               # also at num_timesteps = 0 and at the end
    stochastic_laps: int = 10
    seed: int = 0                        # stochastic lap k: th.manual_seed(seed + k), reset seed seed + k
    q_mc_stride: int = 100               # Q vs MC every this many decisions ...
    q_mc_min_left: int = 600             # ... with at least this many decisions left (or a failure end)
    front_lock_kappa: float = -.12       # front lock: min(kappa_FL, kappa_FR) below this
    spin_slip: float = .12               # spin upshift: gear up while max |rear slip| above this
    car_half_width_m: float = 1.         # edge margin = (1 - |track_position|) w / 2 - this
    stats_window: int = 10_000           # TrainStatsCallback window (steps)


@dataclass(frozen=True)
class TrainConfig:
    preset: str = 'full'
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    demos: DemoDataConfig = field(default_factory=DemoDataConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # sim-runner benchmark (D1_sim_runner §4): 4 workers = P-cores, 1 torch thread is fastest,
    # gradient_steps 2 per vector step = UTD 0.5 (~330 decisions/s measured end to end).
    n_envs: int = 4
    gradient_steps: int = 2
    torch_threads: int = 1
    total_timesteps: int = 2_000_000     # PLACEHOLDER until the smoke-run extrapolation (§11)
    max_wall_h: float | None = None      # PLACEHOLDER: per seed; set from Ibrahim's compute budget
                                         # (D2 question 2). None = no wall-clock limit
    checkpoint_interval: int = 100_000   # checkpoints/step_<n>.zip
    save_replay_buffer: bool = True      # replay_buffer.pkl at the end / wall-limit stop (D3)
    demo_dir: str = 'data/demos'         # relative to the repo root unless absolute
    runs_root: str = 'runs/D2'           # relative to the repo root unless absolute
    vec_start_method: str = 'forkserver'

    def to_dict(self):
        return asdict(self)

    @property
    def utd(self):
        """Gradient steps per env transition."""
        return self.gradient_steps / (self.n_envs * self.agent.train_freq)


def _smoke():
    """~20 min single-seed check (spec §11 step 1): demos as full; BC <= 10 epochs + gate;
    offline 5k; online warm-up 10k; RL until 20 min wall; probe every 5k; eval every 25k with
    3 stochastic laps."""
    c = TrainConfig(preset='smoke')
    return replace(c, total_timesteps=10_000_000, max_wall_h=20 / 60, checkpoint_interval=25_000,
                   bc=replace(c.bc, max_epochs=10),
                   warmup=replace(c.warmup, offline_blocks=5, online_warmup_steps=10_000),
                   guard=replace(c.guard, interval=5_000),
                   eval=replace(c.eval, interval=25_000, stochastic_laps=3))


PRESETS = {'full': lambda: TrainConfig(preset='full'), 'smoke': _smoke}


# --- dict / override helpers (JSON round trip of nested frozen dataclasses) -------------------------
def config_from_dict(cls, d):
    """Rebuild dataclass `cls` from `asdict` output (lists back to tuples where the default is a
    tuple, nested dataclasses recursively). Unknown keys raise."""
    hints = typing.get_type_hints(cls)
    names = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(names)
    if unknown:
        raise KeyError(f'{cls.__name__}: unknown fields {sorted(unknown)}')
    kw = {}
    for name, value in d.items():
        t = hints[name]
        if is_dataclass(t) and isinstance(value, dict):
            kw[name] = config_from_dict(t, value)
        elif isinstance(value, list):
            kw[name] = tuple(value)
        else:
            kw[name] = value
    return cls(**kw)


def train_config_from_dict(d):
    return config_from_dict(TrainConfig, d)


def _parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(cfg, overrides):
    """`overrides`: iterable of 'dotted.path=value' (value parsed as a Python literal, else a
    string), e.g. 'agent.batch_size=256', 'env.reward.c=0.05', 'max_wall_h=None'."""
    for item in overrides or ():
        if '=' not in item:
            raise ValueError(f'override {item!r} is not key=value')
        path, text = item.split('=', 1)
        keys = path.strip().split('.')
        value = _parse_value(text.strip())
        if isinstance(value, list):
            value = tuple(value)
        cfg = _replace_path(cfg, keys, value)
    return cfg


def _replace_path(obj, keys, value):
    name = keys[0]
    if name not in {f.name for f in fields(obj)}:
        raise KeyError(f'{type(obj).__name__} has no field {name!r}')
    if len(keys) == 1:
        return replace(obj, **{name: value})
    return replace(obj, **{name: _replace_path(getattr(obj, name), keys[1:], value)})


def config_json(cfg):
    """Canonical JSON text of a TrainConfig (sorted keys; hashed into provenance)."""
    return json.dumps(cfg.to_dict(), sort_keys=True, indent=1)
