"""Constants of stage C2a (data, knob map, style extraction, geometry gate). No model in this stage.

Spec: docs/specs/C2_style_critic.md (S) amended by docs/specs/C2_review.md (R1-R13, binding).
PLACEHOLDER marks a number with no measurement behind it. Source-specific constants (F1 25/26 click
map, column names) live in `sources/f1_game_map.py`, not here: nothing in this file knows a game.
"""
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# OCD knobs the data covers, in the critic's input order (S4, as C1 S3).
KNOB_ORDER = ('ride_freq_front_hz', 'ride_freq_rear_hz', 'roll_stiffness_Nm_per_rad',
              'roll_stiffness_front_share', 'ride_height_front_m', 'rake_m', 'aero_level', 'aero_front_share',
              'brake_bias_front', 'camber_front_deg', 'camber_rear_deg', 'toe_front_deg', 'toe_rear_deg',
              'lsd_lock_drive', 'lsd_lock_coast')

# Session / controller / assist vocabularies are adapter outputs (sources/*.py, QA M12), not constants here.
UNKNOWN = 'unknown'                                 # level of a descriptor a source does not publish
TRACK_FEATURES = ('n_corners', 'longest_straight_m', 'straight_share', 'slow_corner_share', 'fast_corner_share')

# Reason codes for a lap or a track whose geometry numbers (S_corner_peak_angle, L_*) are NaN, and for
# a lap with no style at all. Nothing is dropped because of them (Decision 7).
OK = 'ok'
NO_GEOMETRY = 'no_geometry'                         # no geometry file for the track
GATE_LOW_CORR = 'gate_low_corr'                     # R2: median |corr(steer, kappa)| below the bar
GATE_LENGTH = 'gate_length'                         # R2: |L_game / L_geo - 1| above the bar
GATE_SIGN = 'gate_sign'                             # median signed corr has the wrong sign for the source's
                                                    # declared steer convention: mirror-image geometry (QA M2)
GATE_NO_LAPS = 'gate_no_laps'                       # no usable telemetry lap to check the geometry with
LAP_NO_GRAPH = 'lap_no_graph'                       # the lap has no graph samples at all (never reaches the adapter)
LAP_NO_SAMPLES = 'lap_no_samples'                   # a channel or the time mapping is missing
LAP_TIME_CLOSURE = 'lap_time_closure'               # R3: |t_end / lap - 1| above the bar
LAP_SPACING = 'lap_spacing'                         # R3: median sample spacing above the bar


@dataclass(frozen=True)
class DataConfig:
    """Setup pipeline (S2 as amended by C2b_addendum B1, B2).

    Order: dry + complete -> `drop_tracks` (B2, at load) -> OCD box (S2 step 4) -> real-world hard cut
    (B1) -> two-sided, two-pass robust label filter per (game, track, session) (B1, replaces S2 steps
    2-3 and the one-sided 8 % cut). The label filter is a selection on the label, computed globally
    (all rows, not per fold; QA M10)."""
    drop_tracks: tuple = ('mexico',)            # B2 / spec 12.12: no accepted geometry
    label_z_max: float = 3.5                    # B1: drop |robust z| > 3.5, both sides
    label_passes: int = 2                       # B1: m and d recomputed once on the kept rows
    # B1 reading: a row dropped in an earlier pass stays dropped ("drop |z| > 3.5 again"). True re-tests
    # every row against the recomputed m, d (the architect's measurement script; 2 more rows kept).
    label_retest_all: bool = False
    mad_to_sd: float = 1.4826                   # B1: d = 1.4826 median |gap - 0|
    real_ref_min_ratio: float = .92             # B1 hard cut: lap < 0.92 T_ref dropped [J, architect]
    real_ref_flag_band: tuple = (.97, 1.05)     # B1: flag r = fastest TT top-10 lap / T_ref outside (review only)
    real_ref_max_group_drop: float = .01        # B1: the hard cut raises if it removes > 1 % of a group
    bounds_tol: float = 1e-9                    # S2 step 4; keeps rake = 0 rows despite rounding error
    max_bounds_drop_frac: float = .05           # S2 step 4 stop rule, applied per game (R1)


CAR_SOURCE_BRANCH = 'sf70h-f1'                      # ocd_first_lap branch the CarConfig values come from


@dataclass(frozen=True)
class CarConfig:
    """Fine-tune car geometry for the frequency mediator (C1-R1).

    m_c and t: `CarGeometry.from_params(OCDBackend(car='sf70h_f1').base_params)`, which needs the
    worktree ocd_first_lap-f1 (branch `CAR_SOURCE_BRANCH`) on PYTHONPATH; the venv's default ocd_first_lap
    does not define that car, so the values are kept here and tests/test_c2_knobs.py checks them against
    the worktree. Runs record the worktree commit and a hash of these values (critic/provenance.py)."""
    corner_sprung_mass_front_kg: float = 151.14666666666668
    corner_sprung_mass_rear_kg: float = 185.35333333333332
    track_front_m: float = 1.6
    track_rear_m: float = 1.6


@dataclass(frozen=True)
class TrackFeatureConfig:
    """Track summary z from the centre line (C1 review [ARCH] table, C1-R15). PLACEHOLDER radius bands
    from PDF Table VIII (PROJECT_CONTEXT 11.7 is open; C2_review "Needs Ibrahim 5")."""
    resample_m: float = 1.
    smooth_window_m: float = 40.
    smooth_passes: int = 2
    corner_radius_m: float = 500.
    merge_gap_m: float = 40.
    min_corner_m: float = 15.
    slow_radius_m: float = 50.
    fast_radius_m: float = 150.                 # 150-500 m counts as fast; 50-150 m (medium) is left out
    z_clip: float = 3.


@dataclass(frozen=True)
class GeometryConfig:
    """Where track geometry comes from and the R2 gate.

    A geometry file is TUM format (`# x_m,y_m,w_tr_right_m,w_tr_left_m`, closed, driving direction).
    Candidates of a track: `extra_dir/<stem>.csv` first, then `tum_dir/tracks/<stem>.csv`; every
    candidate is gated and the first that passes is used (the gate table reports all of them)."""
    extra_dir: str = str(REPO_ROOT / 'data/tracks_extra')
    tum_dir: str = ''                           # '' = ocd_first_lap/vendor/racetrack-database
    min_abs_corr: float = .85                   # R2: median over laps of the best |corr(steer, kappa)|
    max_length_rel: float = .01                 # R2: |L_game / L_geo - 1|
    # R2 fit: shift over the whole lap on the geometry's ~2 m grid (circular cross-correlation),
    # scale on a grid around the closed-lap value L_geo / L_game (relative steps below).
    scale_rel_grid: tuple = (-.01, -.0075, -.005, -.0025, 0., .0025, .005, .0075, .01)
    min_gate_laps: int = 1
    # B3: z uses the layout only (no alignment), so for z alone a candidate that failed the gate is
    # accepted with the length test plus median |corr| >= z_min_abs_corr; flagged z_provisional.
    # The style geometry numbers keep the full gate above.
    z_min_abs_corr: float = .80


@dataclass(frozen=True)
class ExtraGeometryConfig:
    """Centre lines for the tracks TUM lacks or that fail the gate (scripts/fetch_track_geometry.py),
    recipe of docs/research/c2_track_geometry.md section 5. bacinger/f1-circuits, MIT licence.

    Width: PLACEHOLDER, provisional pending Ibrahim (C2_review / c2_track_geometry 4.1 option W2):
    ONE constant per side = half the median total width of the 25 vendored TUM tracks (median over
    tracks of each track's median w_tr_right + w_tr_left = 11.646 m, computed by
    `geometry.tum_median_total_width`; tests/test_c2_geometry.py checks this value)."""
    repo: str = 'bacinger/f1-circuits'
    commit: str = '8094e957b8d6aa8b9f4510c93968556cccd30e02'
    url: str = 'https://raw.githubusercontent.com/{repo}/{commit}/circuits/{id}.geojson'
    licence_url: str = 'https://raw.githubusercontent.com/{repo}/{commit}/LICENSE.md'
    raw_dir: str = str(REPO_ROOT / 'data/tracks_extra/raw/bacinger')
    assumed_half_width_m: float = 5.823         # PLACEHOLDER (see class docstring), both sides
    resample_m: float = 5.                      # arc-length spacing of the written centre line
    spline_rms_m: float = 2.                    # target RMS vertex deviation of the periodic smoothing spline
    max_download_bytes: int = 2_000_000         # a circuit GeoJSON is ~10 kB; anything larger is refused


@dataclass(frozen=True)
class StyleInputConfig:
    """Telemetry lap -> style-rl input (S3, R3)."""
    resample_hz: float = 25.                    # style-rl's rate (SMOOTH_SAMPLES = 3 = 0.12 s)
    max_time_closure_rel: float = .01           # R3: |t_end / lap - 1|
    max_median_spacing_m: float = 8.            # R3
    cache_dir: str = str(REPO_ROOT / 'runs/style_cache')


@dataclass(frozen=True)
class StyleTransformConfig:
    """R4: the 9 B/T/S numbers enter; log of the six rate numbers, robust z, clip. L_* held out."""
    log_keys: tuple = ('B_apply_rate', 'B_release_rate', 'T_apply_rate', 'T_release_rate',
                       'S_reversals_5deg_per_km', 'S_mean_abs_rate')
    masked_keys: tuple = ('L_border_mean_m', 'L_inside_dist_apex_m', 'L_outside_dist_exit_m')
    log_floor: float = 1e-3                     # PLACEHOLDER: a rate of exactly 0 (no event) enters as log(floor)
    # robust z = (x - median) / (IQR / iqr_to_sd); 1.349 makes the IQR an SD-equivalent for normal data,
    # so the +-3 clip is "3 SD". The review says "median and IQR" without the factor: 1.0 would be raw IQR.
    iqr_to_sd: float = 1.349
    z_clip: float = 3.


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    car: CarConfig = field(default_factory=CarConfig)
    track: TrackFeatureConfig = field(default_factory=TrackFeatureConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    extra_geometry: ExtraGeometryConfig = field(default_factory=ExtraGeometryConfig)
    style_input: StyleInputConfig = field(default_factory=StyleInputConfig)
    style_transform: StyleTransformConfig = field(default_factory=StyleTransformConfig)
