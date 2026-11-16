"""Adapter contract between a data source and the critics (S4, Decision 9).

The critics work in OCD physical knobs and style-rl style numbers. Each source (the F1 25/26 game
exports first, Assetto Corsa later) brings its own adapter; nothing in model / train / evaluate may
depend on a source's column names. A "table" is a mapping column name -> array of equal length.
Source-independent containers (Frame, gap, cells, ranking groups, lap Protocols) are in
`critic/dataset.py`; an adapter builds them without importing another adapter (QA M12).
"""
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable
import numpy as np
from ..config import OK, StyleInputConfig


@dataclass
class StyleInput:
    """One lap in the form style-rl's `style_vector` takes, plus what the geometry gate needs.

    The uniform arrays are sampled every 1 / resample_hz seconds; `reason` is `config.OK` or the code
    of the lap QC rule that rejected the lap (then the uniform arrays may be empty)."""
    time_s: np.ndarray                  # uniform clock [s], starts at the first sample
    throttle: np.ndarray                # 0..1
    brake: np.ndarray                   # 0..1
    steer_deg: np.ndarray               # steering wheel angle [deg]; sign as the source gives it
    distance_m: np.ndarray              # source lap distance at each uniform sample [m]
    raw_distance_m: np.ndarray          # source lap distance of the raw steering samples [m]
    raw_steer_deg: np.ndarray           # raw steering samples [deg]
    lap_time_s: float                   # published lap time [s]
    t_end_s: float = float('nan')       # reconstructed time at the last time key [s]
    lap_length_m: float = float('nan')  # largest source distance of the lap [m]
    median_spacing_m: float = float('nan')      # median distance between raw samples [m]
    reason: str = OK
    digest: str = ''                    # sha256 of the raw samples (cache key part)
    # samples dropped because their distance did not increase (first sample of a tie kept, QA M5):
    # {channel or 'time_map': count}
    dropped_ties: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.reason == OK

    @property
    def n_dropped_ties(self) -> int:
        """Steering samples dropped as distance ties (the steering samples define the time base)."""
        return int(self.dropped_ties.get('steer', 0))


def resample_uniform(t: np.ndarray, channels: Mapping[str, np.ndarray], hz: float):
    """Linear resample of `channels` (arrays over the increasing clock `t` [s]) to a uniform clock of
    `hz` samples per second starting at t[0]: (grid, {name: values})."""
    grid = np.arange(t[0], t[-1], 1. / hz)
    return grid, {k: np.interp(grid, t, v) for k, v in channels.items()}


@runtime_checkable
class SourceAdapter(Protocol):
    """What a data source must provide. The vocabularies are the adapter's outputs (QA M12):
    `meta` returns values from them and a Frame stores them."""
    name: str
    sessions: tuple                     # session labels `meta` can return
    controllers: tuple                  # controller labels `meta` can return
    assist_levels: dict                 # assist -> levels, index 0 = config.UNKNOWN
    steer_kappa_sign: int               # declared sign of corr(steer, track curvature kappa (+ = left turn)):
                                        # -1 when the source's steer + is a right turn (R2 sign check, QA M2)

    def to_knobs(self, table: Mapping[str, np.ndarray]) -> np.ndarray:
        """(n, 15) physical OCD knobs in `config.KNOB_ORDER` from the source's setup parameters."""

    def style_inputs(self, lap, cfg: StyleInputConfig = StyleInputConfig()) -> StyleInput:
        """One telemetry lap of the source -> `StyleInput` (time base, 25 Hz, steering in degrees)."""

    def meta(self, table: Mapping[str, np.ndarray]) -> dict:
        """Nuisance descriptors per row, already normalised: 'source' (game / simulator name),
        'session' (one of `sessions`), 'controller' (one of `controllers`), 'assists' ({name: level
        array}, levels from `assist_levels`, `config.UNKNOWN` where the source does not publish it)."""


@dataclass
class Provenance:
    """Input files of a load: {label: sha256}; labels never contain usernames."""
    files: dict = field(default_factory=dict)
    skipped_duplicates: list = field(default_factory=list)
