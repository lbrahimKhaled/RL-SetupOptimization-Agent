"""Constants of stage C2b (critics, training, evaluation). Every number cites its addendum item.

Binding: docs/specs/C2b_addendum.md (B1-B20, Runs and budget, Bars) > C2_review.md > C2_style_critic.md;
spec S12 on top. PLACEHOLDER marks a number with no measurement or decision behind it.
"""
from dataclasses import dataclass, field

# Physics groups (spec S7 / C1 S4): aero + heights; springs + roll; alignment; diff; brake bias.
PHYSICS_GROUPS = (('ride_height_front_m', 'rake_m', 'aero_level', 'aero_front_share'),
                  ('ride_freq_front_hz', 'ride_freq_rear_hz', 'roll_stiffness_Nm_per_rad', 'roll_stiffness_front_share'),
                  ('camber_front_deg', 'camber_rear_deg', 'toe_front_deg', 'toe_rear_deg'),
                  ('lsd_lock_drive', 'lsd_lock_coast'),
                  ('brake_bias_front',))
BARCELONA = 'spain'                     # canonical key of the target track (spec S12.20)
ASSISTS_ABS_TC = ('anti_lock_brakes', 'traction_control')
BRAKING_ASSIST = 'braking_assist'


@dataclass(frozen=True)
class NuisanceConfig:
    """B7. Ridge penalty alpha / n_train on the mean loss (= alpha on the sum of squares)."""
    alpha_gts: float = 1e-2             # (game, track, session) fixed effect
    alpha_driver: float = 1.            # driver offset by (game, user)
    min_driver_rows: int = 2            # fewer training rows -> pooled into 'rare' per fold
    alpha_other: float = 1.             # controller x game, each assist x game, team x game, date terms
    alpha_style: float = 10.            # critics 1 and 2 only: linear style main effect m(s~), 9 -> 1
    days_per_month: float = 30.4375
    backfit_iters: int = 50             # closed-form initialisation (C1-R5), coordinate descent
    backfit_tol: float = 1e-10


@dataclass(frozen=True)
class PretrainConfig:
    """B8 (critic 3, and critic 2's pretrain on A; C1-R6)."""
    hidden: int = 16                    # critic 3 head width
    hidden_full: int = 32               # critic 2: 2 x 32 (B10)
    lr: float = 1e-3
    batch: int = 256
    max_epochs: int = 500
    patience: int = 30                  # B8 (the C1 amendment's 200 is not adopted: B8 says 30)
    inner_frac: float = .10             # inner validation on 10 % of training cells
    weight_decay_grid: tuple = (0., 1e-5, 1e-4)
    final_init_std: float = 1e-3        # C1 [ARCH]: small final-layer init
    smoke_epochs: int = 20              # R0 only
    smoke_patience: int = 5             # R0 only, PLACEHOLDER (R0 budget)


@dataclass(frozen=True)
class FineTuneConfig:
    """B9 / B10 fine-tune on exact rows."""
    n_e: int = 2                        # e = W s~, W 2 x 9 (parallel analysis k = 2, C2_review)
    lr: float = 3e-3
    max_steps: int = 2000
    patience: int = 100
    inner_frac: float = .20             # 20 % of training persons
    l2_grid: tuple = (1e-4, 1e-3, 1e-2, 1e-1)   # on (W, W1e)
    smoke_steps: int = 100              # R0 only


@dataclass(frozen=True)
class GuardConfig:
    """B12 = C1 amendment item 1 (paired guard), unchanged."""
    rows: int = 256
    prob: float = .3
    weight: float = .1
    target_pct: float = 1.


@dataclass(frozen=True)
class BcnHeadConfig:
    """B11 Barcelona residual head."""
    hidden: int = 16
    l2_grid: tuple = (1e-4, 1e-3, 1e-2)
    inner_frac: float = .20             # of Barcelona training cells
    # optimiser: B11 gives none; B8's settings are used (PLACEHOLDER reading, see the coder report)
    lr: float = 1e-3
    batch: int = 256
    max_epochs: int = 500
    patience: int = 30


@dataclass(frozen=True)
class EvalConfig:
    """B14."""
    n_boot: int = 2000
    boot_seed: int = 0
    mde_factor: float = 2.8             # MDE = (z_.975 + z_.80) SE, the addendum's SE -> MDE ratio
    shuffle_seeds: tuple = (0, 1, 2, 3, 4)
    r8_setups: int = 2000               # B14: g = F1 25 vs F1 26 Spearman over support-box setups
    ridge_alpha_a: float = 1.           # B15 ridge on A: nuisance + p + p x z
    ridge_alpha_exact: float = 10.      # B15 ridge on exact rows: nuisance + m(s) + p + s x PC4(p)
    ridge_pcs: int = 4
    bcn_delta_alphas: tuple = (3., 10., 30.)    # E4 pooled + Barcelona deviation
    group_k: int = 4                    # B13: PCA k = 4, varimax


@dataclass(frozen=True)
class MetaPriorConfig:
    """B21 in-fold meta prior Pi(p; t, g) = sum_k w_tk (p_k - m_tk)^2, added as lambda Pi to critic 3's J."""
    fast_frac: float = .20              # B21.1: fast set = lowest 20 % nuisance-only residual per (game, track)
    min_rows: int = 20                  # B21.1: fewer training rows in (game, track) -> the game's pooled fast set
    kappa: float = 300.                 # B21.2: shrinkage of the per-track win rate, in pairs
    dist_eps: float = 1e-9              # B21.2: a pair counts on knob k if the distances differ by more
    c_grid: tuple = (0., .03, .1, .3)   # B21.3: lambda = c SD_train(setup part, c = 0) / SD_train(Pi)


@dataclass(frozen=True)
class SessionConfig:
    """B22."""
    s22b_alpha: float = 10.             # p x 1[Race], p x 1[Quali] nuisance deviation (training only)
    s22b_sessions: tuple = ('Race', 'Quali')     # TT is the reference level
    s22a_drop: str = 'Quali'            # S22a: critic 3 trained without these rows


@dataclass(frozen=True)
class LadderConfig:
    """B23 rung 2: e' = gamma clip(e); fixed (gamma, L2 on (W, W1e), alpha of m(s~)) per level."""
    levels: tuple = (('L1', .3, 1., 100.), ('L2', .1, 10., 300.), ('L3', .03, 100., 1000.))
    rung_order: tuple = ('rung0_B9_exact', 'rung1_rows_S', 'rung2_selected_level', 'rung2_L3')


@dataclass(frozen=True)
class C2bConfig:
    nuisance: NuisanceConfig = field(default_factory=NuisanceConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    bcn: BcnHeadConfig = field(default_factory=BcnHeadConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    meta: MetaPriorConfig = field(default_factory=MetaPriorConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)


# Bars (addendum "Bars"; written to bars.json before any training; read-only to the report).
BARS = {
    'critic3_A_pooled': {'pair_acc_min': .56, 'pair_acc_ci_low_gt': .52, 'vs_ridge_margin': -.02,
                         'strict_pair_acc_min': .54, 'heldout_setup_r2_gain_gt': 0.},
    'barcelona': {'head_vs_pooled_max_drop': .03, 'pair_acc_min': .55, 'mde_expected': .13},
    'critic1_exact': {'delta_r2_ci_low_gt': -.02, 'fold_sets': ['F_person', 'F_cell'], 'e0_pair_acc_A_max_abs_diff': .01},
    'style_helps': {'claim_if_ci_low_gt': 0., 'fold_set': 'F_person', 'seen_driver_only': 'F_cell',
                    'mde_F_person': .04, 'mde_F_cell': .02, 'null_expected': True},
    'critic2': {'mae_below_nuisance_ci_excludes_0': True, 'fold_sets': ['F_cell', 'F_person'],
                'F_person_miss_expected': True, 'grouping_cost': 'report only'},
    'groups': {'rule': 'C1-R12: keep PCA unless the paired CI of (PCA - physics) pair accuracy lies entirely below 0, '
                       'or the held-out-setup R2 gain trails physics by more than 0.01', 'r2_margin': .01},
    'fallback_trigger': {'delta_r2_point_lt': -.02, 'e0_pair_acc_A_abs_gt': .01,
                         'does_not_trigger': ['null style effect', 'critic 2 missing its bar vs nuisance-only',
                                              'missed Barcelona bar'],
                         'td3_q': 'always critic 1 (spec 12.21); the B16 clause "neither passes -> critic 3" is void (B23)'},
    # B21-B23 (addendum items of 16 Nov 2026)
    'b21_adoption': {'F_person_delta_pair_ci_low_gt': 0., 'F_cell_delta_pair_point_ge': 0.,
                     'barcelona_max_drop_point': .01, 'F_hash_r2_gain_max_drop': .005,
                     'compare': 'critic 3 with B21 at the selected c vs plain critic 3, twin 0, repeat 0, same folds / seeds'},
    'b21_trigger': {'any_R11_ranking_bar_missed_on_A': ['pair_acc_min', 'pair_acc_ci_low_gt', 'vs_ridge_margin',
                                                        'strict_pair_acc_min'],
                    'not_triggering': ['F_hash R2 bar', 'Barcelona bar']},
    'b21_triggered_delivery': {'F_person_delta_pair_ci_low_gt': -.01, 'barcelona_max_drop_point': .01},
    'b21_grid': {'c': [0., .03, .1, .3], 'kappa': 300., 'fast_frac': .2, 'tie': 'smaller c'},
    's22a': {'drop_quali_iff': 'paired delta (TT+Race-only - all) on TT+Race pairs: CI low > 0 on F_person AND point > 0 on F_cell'},
    's22b': {'escalate_iff': 'delta on TT pairs (q_TT - plain q): CI low > 0 on F_person'},
    'b23': {'rung_order': ['rung0_B9_exact', 'rung1_rows_S (only if the B16 trigger fires)', 'rung2_selected_level', 'rung2_L3'],
            'levels': {'L1': {'gamma': .3, 'l2': 1., 'alpha_m_s': 100.}, 'L2': {'gamma': .1, 'l2': 10., 'alpha_m_s': 300.},
                       'L3': {'gamma': .03, 'l2': 100., 'alpha_m_s': 1000.}},
            'bars': 'the critic 1 bars (delta R2 CI low > -0.02 on F_person and F_cell; e = 0 pair accuracy on A within '
                    '0.01 of the delivered critic 3)', 'td3_q': 'critic1 always; L3 failing = implementation fault, stop',
            'comparator': "critic 3 + exact-row nuisance at the level's alpha of m(s~)",
            'selection_set': ['L1', 'L2', 'L3'],
            'critic2': 'rungs 0-1 only in C2b (architect ruling 29); rung 2 deferred to the OCD/AC stage'},
}

__all__ = ['ASSISTS_ABS_TC', 'BARCELONA', 'BARS', 'BRAKING_ASSIST', 'BcnHeadConfig', 'C2bConfig', 'EvalConfig',
           'FineTuneConfig', 'GuardConfig', 'LadderConfig', 'MetaPriorConfig', 'NuisanceConfig', 'PHYSICS_GROUPS',
           'PretrainConfig', 'SessionConfig']
