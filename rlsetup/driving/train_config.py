"""All numbers of D2 (SAC driving-agent training). Nothing numeric lives in train.py.

Spec: docs/specs/D2_agent.md (binding); warm start, stop rule and the r1_* presets:
docs/specs/R1_time_optimal.md §2.4. Step counts are transitions summed over envs (SB3
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
                                         # | 'warm' (R1 §2.5: actor, critics, alpha from TrainConfig.init_from)
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
    stats_slow_mps: float = 15.          # TrainStats: failure counted "slow" below this speed ...
    stats_wide_m: float = 3.             # ... or "wide" above this |e| (fail_share_slow_or_wide)
    stats_low_v0_mps: float = 5.         # TrainStats: stalls counted among episodes starting below this


@dataclass(frozen=True)
class WarmStartConfig:
    """init_mode 'warm' (R1 spec §2.5, §3 as amended by Amendment 1 §B): load a trained model and
    its Adam states, preload its replay buffer relabelled to the new reward, offline critic phase,
    online critic phase (actor + alpha frozen), unfreeze."""
    critic_warmup_steps: int = 20_000    # online critic phase: actor + alpha frozen (Amendment 1: was 100k)
    load_critics: bool = True            # critics + target critics from init_from
    load_alpha: bool = True              # log_ent_coef from init_from
    load_optimizers: bool = True         # actor / critic / ent_coef Adam states from init_from
    init_buffer: str | None = None       # replay_buffer.pkl to preload (CLI --init-buffer); None =
                                         # <init_from>/../../replay_buffer.pkl; missing -> refuse
    relabel: bool = True                 # relabel the preloaded buffer ref_tracking -> time_pbrs (relabel.py)
    offline_critic_grad_steps: int = 25_000   # critic grad steps on the preloaded buffer before learn()
    offline_block: int = 1_000           # ... in blocks of this many (critic loss + mean Q logged per block)
    actor_lr: float | None = None        # actor Adam lr from the unfreeze on (None = agent.learning_rate)
    min_buffer_batches: int = 10         # no grad step until buffer >= agent.batch_size * this (0 with a
                                         # preloaded buffer)
    critic_check_tol: float = .08        # |Q_min - MC| / |MC| at the unfreeze eval; above -> warning


@dataclass(frozen=True)
class StopConfig:
    """Plateau stop (R1 spec §5): from min_steps on, stop when over the last `window` steps
    neither the best clean deterministic lap 2 nor the best stochastic mean lap improved by
    >= min_improvement_s [s]."""
    plateau: bool = False
    min_steps: int = 1_000_000
    window: int = 500_000
    min_improvement_s: float = .1


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
    warm: WarmStartConfig = field(default_factory=WarmStartConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    init_from: str | None = None         # model .zip for init_mode 'warm' (absolute; CLI --init-from);
                                         # its run folder (zip's parents[1]) must hold config.json
    strict_provenance: bool = True       # refuse a run / resume with uncommitted tracked rlsetup changes
                                         # (unless --allow-dirty); False for the smoke presets
    # sim-runner benchmark (D1_sim_runner §4): 4 workers = P-cores, 1 torch thread is fastest,
    # gradient_steps 2 per vector step = UTD 0.5 (~330 decisions/s measured end to end).
    n_envs: int = 4
    gradient_steps: int = 2
    torch_threads: int = 1
    device: str = 'cpu'                  # torch device of the SAC networks ('cuda' on the Windows RTX box)
    total_timesteps: int = 2_000_000     # PLACEHOLDER until the smoke-run extrapolation (§11)
    max_wall_h: float | None = None      # PLACEHOLDER: per seed; set from Ibrahim's compute budget
                                         # (D2 question 2). None = no wall-clock limit
    checkpoint_interval: int = 100_000   # checkpoints/step_<n>.zip
    save_replay_buffer: bool = True      # replay_buffer.pkl at the end / wall-limit stop (D3); False
                                         # disables every buffer save, incl. the periodic one below
    replay_buffer_save_every: int | None = None   # steps between periodic replay_buffer.pkl saves
                                         # (atomic, for crash-safe resume); None = checkpoint_interval,
                                         # 0 = only at the end. ~1.3 GB per save at buffer_size 1M.
                                         # When > 0 the buffer is also saved at every evaluation,
                                         # together with last.zip (QA D2 m4)
    resume_empty_buffer_min_batches: int = 10   # resume WITHOUT replay_buffer.pkl: no gradient updates
                                         # until the buffer holds agent.batch_size x this many
                                         # transitions (lead decision, QA D2 follow-up)
    demo_dir: str = 'data/demos'         # relative to the repo root unless absolute
    runs_root: str = 'runs/D2'           # relative to the repo root unless absolute
    vec_start_method: str = 'forkserver'

    def to_dict(self):
        return asdict(self)

    @property
    def buffer_save_interval(self):
        """Steps between periodic replay-buffer saves; 0 = none (only the final save)."""
        if not self.save_replay_buffer:
            return 0
        return self.checkpoint_interval if self.replay_buffer_save_every is None else int(self.replay_buffer_save_every)

    @property
    def utd(self):
        """Gradient steps per env transition."""
        return self.gradient_steps / (self.n_envs * self.agent.train_freq)


def _smoke():
    """~20 min single-seed check (spec §11 step 1): demos as full; BC <= 10 epochs + gate;
    offline 5k; online warm-up 10k; RL until 20 min wall; probe every 5k; eval every 25k with
    3 stochastic laps."""
    c = TrainConfig(preset='smoke', strict_provenance=False)
    return replace(c, total_timesteps=10_000_000, max_wall_h=20 / 60, checkpoint_interval=25_000,
                   bc=replace(c.bc, max_epochs=10),
                   warmup=replace(c.warmup, offline_blocks=5, online_warmup_steps=10_000),
                   guard=replace(c.guard, interval=5_000),
                   eval=replace(c.eval, interval=25_000, stochastic_laps=3))


def _r1_warm():
    """R1 (time_pbrs reward) warm-started from a trained ref_tracking driver (--init-from) with
    its Adam states and its replay buffer relabelled to R1: offline critic 25k grad steps, online
    critic phase 20k, then full SAC; guard 0.65 x the loaded actor's probe; 3M cap, 4 h, plateau
    stop (R1 spec §2.4, §5, Amendment 1)."""
    c = TrainConfig(preset='r1_warm')
    return replace(c, total_timesteps=3_000_000, max_wall_h=4.,
                   env=replace(c.env, reward=replace(c.env.reward, mode='time_pbrs')),
                   agent=replace(c.agent, init_mode='warm'),
                   guard=replace(c.guard, threshold=.65),
                   stop=replace(c.stop, plateau=True))


def _r1_warm_smoke():
    """r1_warm to 150k env steps (Amendment 1 §D: offline critic 25k grad steps, online critic
    20k, then 130k RL; ~15 min): eval every 25k with 3 stochastic laps, probe every 5k, no
    plateau stop, dirty tree allowed. max_wall_h 0.5 is only a safety cap (the step count ends it)."""
    c = _r1_warm()
    return replace(c, preset='r1_warm_smoke', total_timesteps=150_000, max_wall_h=.5,
                   checkpoint_interval=25_000, strict_provenance=False,
                   eval=replace(c.eval, interval=25_000, stochastic_laps=3),
                   guard=replace(c.guard, interval=5_000),
                   stop=replace(c.stop, plateau=False))


def _p1_tum():
    """sf70h_tum car with the TUM-like setup p1_TUM (rlsetup/driving/setups.py) and 15 kN m brakes,
    ERS / DRS off; otherwise the settings of run sf70h_mac_r1: time_pbrs reward from scratch, demo
    speed fraction (0.5, 0.8), BC gate recorded but not required, no step / wall-clock cap (runs
    until stopped). Run name: p1_TUM."""
    c = TrainConfig(preset='p1_TUM')
    return replace(c, total_timesteps=1_000_000_000, max_wall_h=None,
                   env=replace(c.env, car='sf70h_tum', ers=False, setup='p1_TUM', max_brake_torque_Nm=15000.,
                               reward=replace(c.env.reward, mode='time_pbrs')),
                   demos=replace(c.demos, fraction_range=(.5, .8)),
                   bc=replace(c.bc, gate_required=False))


def _p0_tumref():
    """Like p1_TUM but with the old setup p0 (12 kN m brakes) and TUM's raceline as xi_ref
    (EnvConfig.reference = 'raceline', Track.reference): the reference-line counterpart of
    p1_TUM. Run name: p0_TUMref."""
    c = _p1_tum()
    return replace(c, preset='p0_TUMref',
                   env=replace(c.env, setup='p0', max_brake_torque_Nm=12000., reference='raceline'))


PRESETS = {'full': lambda: TrainConfig(preset='full'), 'smoke': _smoke, 'r1_warm': _r1_warm,
           'r1_warm_smoke': _r1_warm_smoke, 'p1_TUM': _p1_tum,
           'p0_TUMref': _p0_tumref}


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
