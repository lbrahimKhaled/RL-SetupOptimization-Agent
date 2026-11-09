"""All numbers of D2 (SAC driving-agent training). Nothing numeric lives in train.py.

Spec: docs/specs/D2_agent.md (binding); warm start, stop rule and the r1_* presets:
docs/specs/R1_time_optimal.md §2.4; R2 training aids (bonus schedule, focus weights, low-noise
rollouts, presets f1_r2*): docs/specs/R2_review.md incl. Amendment 1; R3 (preset f1_r3: B = 30, floor
0.85, never-fade schedule, bonus relabel, focus init_state): docs/specs/R3_review.md; R4 (preset f1_r4:
speed-profile potential, corner entry / exit terms, speed relabel of the init buffer, fresh-buffer
fallback): docs/specs/R4_review.md; preset f1_r4_scratch: the R4 reward from scratch. Step counts are transitions summed over envs (SB3
`num_timesteps`); one transition = one 25 Hz decision of one env. Times in s unless the name
says otherwise. PLACEHOLDER marks numbers not yet justified by data or a decision.

The discount gamma is NOT here: it is `EnvConfig.reward.gamma` (0.992 per decision), part of the
frozen driver definition (D1 config.py).

This module must not import torch (rlsetup.driving.vec imports `config_from_dict`).
"""
from dataclasses import dataclass, field, asdict, fields, is_dataclass, replace
from pathlib import Path
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
    low_noise_envs: tuple = ()           # R2-8: env indices that act with tanh(mean + scale sigma eps)
    low_noise_std_scale: float = .3      # ... scale (0 = the deterministic action) [P: research C.3]


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
    # R4 eval logging (review req 6; written to config.json only for R4 runs or when changed):
    full_throttle_pedal: float = .95     # full-throttle share per straight: throttle pedal >= this
    brake_pedal: float = .1              # brake share / brake onset: brake pedal >= this. PLACEHOLDER


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
    actor_lr: float | None = None        # actor Adam lr from the unfreeze on (None = agent.learning_rate);
                                         # also honoured by init_mode 'bc' (callbacks.PhaseCallback)
    min_buffer_batches: int = 10         # no grad step until buffer >= agent.batch_size * this (0 with a
                                         # preloaded buffer)
    critic_check_tol: float = .08        # |Q_min - MC| / |MC| at the unfreeze eval; above -> warning
    # R3 req 6: pre-exit probe of the offline critic phase (bonus buffer with nonzero bonuses only):
    # fixed batch of rows that have a nonzero bonus within this many decisions ahead (same env column,
    # no episode end before it); mean Q_min per block; warning when its range over the last
    # pre_exit_warn_blocks blocks exceeds pre_exit_warn_q [reward units] (critic still moving near exits).
    pre_exit_horizon: int = 30
    pre_exit_warn_q: float = .5
    pre_exit_warn_blocks: int = 5
    # R4 (docs/specs/R4_review.md req 4-5): relabel the preloaded time_pbrs buffer to the speed potential
    # and the corner entry / exit terms (relabel.relabel_speed_profile); more than
    # speed_relabel_max_unresolved of the stored transitions unresolved -> fresh buffer: fallback_steps
    # transitions from the frozen actor, then the offline critic phase and the usual warm-up. The
    # offline phase probes mean Q_min on rows <= probe_before_m before an entry / exit (recovered s).
    speed_relabel: bool = False
    speed_relabel_max_unresolved: float = .001
    fallback_steps: int = 200_000
    probe_before_m: float = 150.


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
class BonusScheduleConfig:
    """R2-5 / Amendment 1: corner-bonus weight w(t), t = this run's num_timesteps: 1 for
    t < hold_until, linear 1 -> 0 on [hold_until, fade_until], 0 after; reportable from
    report_from = fade_until + bonus_free_before_report (8M). Trigger (Amendment 1): the first
    evaluation with a clean deterministic lap 2 <= trigger_lap2_s while w > 0 or t < report_from
    sets w = 0 for the rest of the run and report_from = trigger step + bonus_free_before_report;
    the run then stops itself at report_from.
    R3 (DECISIONS #25, R3_review req 1): hold_until = None = never fade: w = 1 for the whole run,
    report_from None (no evaluation is reportable, best.zip is never written, only best_aided.zip),
    no trigger; fade_until and bonus_free_before_report are then unused. trigger_lap2_s stays the
    keeper limit (scripts/keeper_check.py counts every evaluation of such a run)."""
    hold_until: int | None = 2_000_000
    fade_until: int = 4_000_000
    bonus_free_before_report: int = 4_000_000   # Decision 2 (DECISIONS #22)
    trigger_lap2_s: float = 95.          # Decision 5: clean deterministic lap 2 <= this

    @property
    def never_fade(self):
        return self.hold_until is None

    @property
    def report_from(self):
        """First reportable step; None for a never-fade schedule."""
        return None if self.never_fade else self.fade_until + self.bonus_free_before_report


@dataclass(frozen=True)
class FocusUpdateConfig:
    """R2-7.3/7.4: FocusedStartCallback (main process). Counters per focus bin: passes n_b and
    crashes c_b (off_track / wrong_way / nonfinite), decayed every update_every steps by
    0.5^(update_every / half_life); r_b = c_b / n_b; P = (1 - hot_floor_share) r / sum r +
    hot_floor_share Uniform(H0). Seed: crashes of the failed deterministic eval laps with
    step >= seed_min_step in <seed_run>/laps/step_*_det.npz: n_b = prior_passes,
    c_b = prior_passes crashes_b / N_evals; H0 = bins with >= hot_min_crashes seed crashes.
    R3 req 7: init_state = a focus_state.json of an earlier run (relative to the repo root unless
    absolute): its decayed counters n, c and H0 are the initial state (continued, not re-seeded);
    takes precedence over seed_run."""
    update_every: int = 10_000
    half_life: int = 1_000_000
    prior_passes: float = 200.
    seed_run: str | None = None          # seed folder (relative to the repo root unless absolute)
    init_state: str | None = None        # focus_state.json to continue from (R3; precedence over seed_run)
    seed_min_step: int = 2_000_000
    hot_min_crashes: int = 5
    hot_floor_share: float = .2


R2_TRAIN_KEYS = ('bonus_schedule', 'focus_update')      # TrainConfig blocks of the R2 training aids
R2_AGENT_KEYS = ('low_noise_envs', 'low_noise_std_scale')
R3_WARM_KEYS = ('pre_exit_horizon', 'pre_exit_warn_q', 'pre_exit_warn_blocks')   # omitted like the R2 blocks
R4_WARM_KEYS = ('speed_relabel', 'speed_relabel_max_unresolved', 'fallback_steps', 'probe_before_m')
R4_EVAL_KEYS = ('full_throttle_pedal', 'brake_pedal')   # omitted (at defaults) unless an R4 term is on


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
    bonus_schedule: BonusScheduleConfig = field(default_factory=BonusScheduleConfig)   # R2 (corner_bonus)
    focus_update: FocusUpdateConfig = field(default_factory=FocusUpdateConfig)         # R2 (focus)
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

    @property
    def r2_aids(self):
        """True if any R2 training aid is on (corner bonus, focused starts, low-noise envs)."""
        return bool(self.env.corner_bonus.enabled or self.env.focus.enabled or self.agent.low_noise_envs)

    def to_dict(self):
        """asdict with EnvConfig.to_dict for `env`; without any R2 aid on, the R2 blocks that equal
        their defaults are left out too, so pre-R2 configs (and their JSON) are unchanged. Without
        an R4 term (EnvConfig.r4) and without warm.speed_relabel, the R4 warm / eval fields at their
        defaults are left out (pre-R4 configs round-trip byte-equal)."""
        d = asdict(self)
        d['env'] = self.env.to_dict()
        if not (self.env.r4 or self.warm.speed_relabel):
            for block, keys, default in (('warm', R4_WARM_KEYS, WarmStartConfig()), ('eval', R4_EVAL_KEYS, EvalConfig())):
                for k in keys:
                    if getattr(getattr(self, block), k) == getattr(default, k):
                        del d[block][k]
        if not self.r2_aids:
            for k in R2_TRAIN_KEYS:
                if getattr(self, k) == type(getattr(self, k))():
                    del d[k]
            for k in R2_AGENT_KEYS:
                if getattr(self.agent, k) == getattr(AgentConfig(), k):
                    del d['agent'][k]
            for k in R3_WARM_KEYS:
                if getattr(self.warm, k) == getattr(WarmStartConfig(), k):
                    del d['warm'][k]
        return d

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


def _bcn():
    """p1_TUM's recipe on Barcelona (racetrack-database Catalunya) with the QSS-best setup
    bcn_best and 15 kN m brakes (docs/results/bcn/SETUP_SELECTION.md). Run name: bcn."""
    c = _p1_tum()
    return replace(c, preset='bcn', env=replace(c.env, track='Catalunya', setup='bcn_best', max_brake_torque_Nm=15000.))


def _f1_sha():
    """p1_TUM's recipe (time_pbrs from scratch, demo f (0.5, 0.8), BC gate not required, ERS off,
    15 kN m, safe line, no step / wall cap) on Shanghai with the car sf70h_f1 (ocd_first_lap branch
    sf70h-f1; run with PYTHONPATH=<that worktree>) and its setup p1_TUM_f1 (setups.py: car bias
    0.58, heave damping 0.3 PLACEHOLDER). Steering action +-1 = the car's 0.30 rad lock (env.py).
    Demo / settle traction control re-tuned for the peakier tyre (Fx peak at kappa 0.075-0.081,
    was 0.14-0.27): cut starts at rear slip 0.06 and is complete at 0.09 (span 0.03), just past
    the peak. PLACEHOLDER; 0.06 = the start that drove the closed-loop TUM lap on this car
    (SF70H_F1_UPGRADE.md §3). Run name: f1_sha."""
    c = _p1_tum()
    return replace(c, preset='f1_sha',
                   env=replace(c.env, car='sf70h_f1', setup='p1_TUM_f1',
                               demo=replace(c.env.demo, traction_slip_start=.06, traction_slip_span=.03)))


LINE_IQP_W35 = 'csv:docs/results/shanghai_line_setup/line_iqp_w3.5.csv'   # ReferenceLine digest eaf1792a00be2da0


def _f1_sha_w35():
    """f1_sha with only xi_ref changed: the converged (iterated-QP) minimum-curvature line with
    a 3.5 m width allowance (docs/results/shanghai_line_setup/NOTES.md: footprint fully inside,
    min clearance +0.10 m; QSS p1_TUM_f1 96.01 s vs 102.53 s on the safe line). The start
    placement, settle, demonstrations and evaluation start follow it (env.py drive_line);
    progress, the lap line and the QSS start-speed profile stay on Track.path. Run name: f1_sha_w35."""
    c = _f1_sha()
    return replace(c, preset='f1_sha_w35', env=replace(c.env, reference=LINE_IQP_W35))


def _f1_best_w35():
    """f1_sha_w35 with only the setup changed to the QSS-best Shanghai setup sha_f1_best
    (docs/results/shanghai_line_setup/NOTES.md; QSS 93.83 s on this line). Run name: f1_best_w35."""
    c = _f1_sha_w35()
    return replace(c, preset='f1_best_w35', env=replace(c.env, setup='sha_f1_best'))


F1_BEST_W35_RUN = 'runs/mac/D2/f1_best_w35/seed0'   # R2 warm-start source and focus seed
# R2 Amendment 2 (DECISIONS #24): warm start from the source's last.zip (num_timesteps 12,350,000; its
# critic matches the end-of-run replay buffer), not best.zip (9.2M). Absolute, like --init-from.
F1_R2_INIT = str(Path(__file__).resolve().parents[2] / F1_BEST_W35_RUN / 'checkpoints' / 'last.zip')


def _f1_r2():
    """R2 (docs/specs/R2_review.md + Amendments 1-2; DECISIONS #22-24): f1_best_w35 (same driver
    definition) + clean-corner exit bonus (B = 3, w = 1 until 2M, fade to 0 at 4M, reportable from
    8M; early trigger at a clean lap 2 <= 95 s) + focused starts (25 %, seeded from f1_best_w35's
    crashes) + low-noise rollouts on env 3 (0.3 sigma); warm start by default from f1_best_w35
    last.zip (F1_R2_INIT, 12.35M steps = lifetime_steps_offset; --init-from overrides) with its
    replay buffer adopted unrelabelled (already time_pbrs), actor lr 1e-4 from the unfreeze, guard
    0.65 (baseline = the loaded actor's probe); no step / wall cap, no plateau stop. Run name: f1_r2."""
    c = _f1_best_w35()
    return replace(c, preset='f1_r2', total_timesteps=1_000_000_000, max_wall_h=None, init_from=F1_R2_INIT,
                   env=replace(c.env, corner_bonus=replace(c.env.corner_bonus, enabled=True),
                               focus=replace(c.env.focus, enabled=True)),
                   agent=replace(c.agent, init_mode='warm', low_noise_envs=(3,)),
                   warm=replace(c.warm, relabel=False, actor_lr=1e-4),
                   guard=replace(c.guard, threshold=.65),
                   focus_update=replace(c.focus_update, seed_run=F1_BEST_W35_RUN),
                   stop=replace(c.stop, plateau=False))


F1_R2_RUN = 'runs/mac/D2/f1_r2/seed0'   # R3 warm-start source (stopped at 7.40M, DECISIONS #25)
F1_R3_INIT = str(Path(__file__).resolve().parents[2] / F1_R2_RUN / 'checkpoints' / 'last.zip')


def _f1_r3():
    """R3 (docs/specs/R3_bigger_bonus.md + R3_review.md; DECISIONS #25): f1_r2 with the corner
    bonus at B = 30 per corner, floor 0.85, kept on (w = 1, never fade: hold_until None, so no
    trigger, nothing reportable, only best_aided.zip); warm start by default from f1_r2 last.zip
    (F1_R3_INIT, 7.40M steps; lifetime offset 12.35M + 7.40M = 19.75M) with f1_r2's own replay buffer
    (warm.init_buffer None -> <f1_r2>/replay_buffer.pkl) whose bonuses are relabelled exactly from
    (3, 0.75) to (30, 0.85) (buffers.relabel_bonuses); offline critic 50k grad steps (pre-exit probe);
    focused starts continue from f1_r2's focus_state.json (init_state, not re-seeded). Everything
    else as f1_r2 (focus 25 %, low-noise env 3, actor lr 1e-4, guard 0.65, no plateau stop, no
    step / wall cap). Laps are aided while the bonus is on. Run name: f1_r3."""
    c = _f1_r2()
    return replace(c, preset='f1_r3', init_from=F1_R3_INIT,
                   env=replace(c.env, corner_bonus=replace(c.env.corner_bonus, bonus=30., floor_ratio=.85)),
                   warm=replace(c.warm, offline_critic_grad_steps=50_000),
                   bonus_schedule=replace(c.bonus_schedule, hold_until=None),
                   focus_update=replace(c.focus_update, seed_run=None, init_state=f'{F1_R2_RUN}/focus_state.json'))


def _f1_r4():
    """R4 (docs/specs/R4_speed_profile.md + binding R4_review.md; DECISIONS #26): f1_r2's recipe with the
    R2/R3 exit bonus replaced by (a) the speed-profile potential Phi_v = 20 g(v, v_qss(s)), sigma 0.3,
    in the env reward (potential-based, gamma Phi_v' - Phi_v) and (b) corner entry + exit terms
    B_in = B_out = 10 per unit (sigma 0.3; entry = window start, paid at the entry crossing, taken back
    on a terminal failure before the unit's exit; exit paid at the exit), stored as the buffer bonus
    with w = 1 (never fade: hold_until None, no trigger, nothing reportable, only best_aided.zip).
    Warm start from f1_best_w35 last.zip (F1_R2_INIT, 12.35M = lifetime offset 0 + 12,350,000) with its
    pre-R2 replay buffer adopted and relabelled exactly (warm.speed_relabel; fallback: fresh buffer of
    200k frozen-actor transitions); offline critic 50k grad steps with pre-entry / pre-exit probes
    (rows <= 150 m before, recovered s); focused starts seeded from f1_best_w35's crashes; low-noise
    env 3, actor lr 1e-4, guard 0.65, no plateau stop, no step / wall cap. Run name: f1_r4."""
    c = _f1_r2()
    return replace(c, preset='f1_r4', init_from=F1_R2_INIT,
                   env=replace(c.env, reward=replace(c.env.reward, speed_k=20., speed_sigma=.3),
                               corner_bonus=replace(c.env.corner_bonus, mode='entry_exit', bonus_in=10., bonus_out=10.,
                                                    sigma=.3)),
                   warm=replace(c.warm, relabel=False, speed_relabel=True, offline_critic_grad_steps=50_000),
                   bonus_schedule=replace(c.bonus_schedule, hold_until=None),
                   focus_update=replace(c.focus_update, seed_run=F1_BEST_W35_RUN, init_state=None))


def _f1_r4_scratch():
    """f1_best_w35 from scratch (demos + BC, no warm start, no focused starts, no low-noise rollouts, same
    guard and evaluation) with only the reward changed to f1_r4's, so that f1_best_w35 is the control
    run of the reward: speed potential Phi_v (k_v 20, sigma 0.3) + corner entry / exit terms (B_in = B_out = 10,
    sigma 0.3, QSS targets unchanged), never fading (w = 1 from step 0, no trigger, nothing reportable, only
    best_aided.zip). Demo rewards and buffer rows carry the same terms (demos.py). One more difference from
    f1_best_w35: actor lr 1e-4 from the unfreeze (warm.actor_lr), because the control trained at 1e-4 from
    its guard rollback at 30k on (docs/specs/R4_scratch_review.md B1), which would otherwise depend on whether
    this run's guard fires. Guard and rollback logic unchanged. Run name: f1_r4_scratch."""
    c = _f1_best_w35()
    return replace(c, preset='f1_r4_scratch',
                   env=replace(c.env, reward=replace(c.env.reward, speed_k=20., speed_sigma=.3),
                               corner_bonus=replace(c.env.corner_bonus, enabled=True, mode='entry_exit', bonus_in=10.,
                                                    bonus_out=10., sigma=.3)),
                   warm=replace(c.warm, actor_lr=1e-4),
                   bonus_schedule=replace(c.bonus_schedule, hold_until=None))


def _f1_r2_smoke():
    """f1_r2 compressed to 150k env steps (R2-9.2): w = 1 until 60k, fade to 0 at 120k,
    reportable from 150k (gap 30k); eval every 25k with 3 stochastic laps, probe every 5k,
    checkpoints every 25k, dirty tree allowed; max_wall_h 0.5 is only a safety cap. Same default
    warm start as f1_r2 (last.zip)."""
    c = _f1_r2()
    return replace(c, preset='f1_r2_smoke', total_timesteps=150_000, max_wall_h=.5,
                   checkpoint_interval=25_000, strict_provenance=False,
                   bonus_schedule=replace(c.bonus_schedule, hold_until=60_000, fade_until=120_000,
                                          bonus_free_before_report=30_000),
                   eval=replace(c.eval, interval=25_000, stochastic_laps=3),
                   guard=replace(c.guard, interval=5_000))


PRESETS = {'full': lambda: TrainConfig(preset='full'), 'smoke': _smoke, 'r1_warm': _r1_warm,
           'r1_warm_smoke': _r1_warm_smoke, 'p1_TUM': _p1_tum,
           'p0_TUMref': _p0_tumref, 'bcn': _bcn, 'f1_sha': _f1_sha, 'f1_sha_w35': _f1_sha_w35,
           'f1_best_w35': _f1_best_w35, 'f1_r2': _f1_r2, 'f1_r2_smoke': _f1_r2_smoke, 'f1_r3': _f1_r3,
           'f1_r4': _f1_r4, 'f1_r4_scratch': _f1_r4_scratch}


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
    """Canonical JSON text of a TrainConfig (sorted keys; hashed into provenance). R2 blocks that
    are off and at their defaults are omitted (TrainConfig.to_dict)."""
    return json.dumps(cfg.to_dict(), sort_keys=True, indent=1)
