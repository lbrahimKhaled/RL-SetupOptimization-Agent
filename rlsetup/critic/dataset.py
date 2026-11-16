"""Source-independent dataset containers of C2 (QA M12): what every source adapter produces.

`Frame` (rows A, rows S = `cell >= 0`), the gap label, cells, the R6 ranking groups, and the minimal
shapes of a telemetry lap (`LapLike`) and of a loaded data set (`LapSource`) that `style_data` needs.
An Assetto Corsa adapter builds these from its own files without importing `sources.f1laps`. The
session / controller / assist vocabularies are the adapter's outputs and travel with the Frame.
"""
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class LapLike(Protocol):
    """One telemetry lap as `style_data.lap_styles` reads it (plus whatever the adapter's
    `style_inputs` needs)."""
    game: str                       # source name (game / simulator)
    track: str                      # the source's track name
    driver: int                     # integer driver id (no names)

    @property
    def has_samples(self) -> bool: ...


@runtime_checkable
class LapSource(Protocol):
    """A loaded data set: `laps` maps an integer lap key to a `LapLike`."""
    laps: Mapping


def gaps(lap_s, group, reference=None) -> np.ndarray:
    """% lap-time gap 100 (lap / median - 1), median per `group` value over the rows where `reference`
    is True (default: all rows). C2b can pass a training-fold mask as `reference` (QA M10): rows
    of a group without reference rows get NaN."""
    lap_s, group = np.asarray(lap_s, float), np.asarray(group)
    ref = np.ones(len(lap_s), bool) if reference is None else np.asarray(reference, bool)
    med = {g: np.median(lap_s[ref & (group == g)]) for g in np.unique(group[ref])}
    return 100. * (lap_s / np.array([med.get(g, np.nan) for g in group]) - 1.)


def robust_label(lap_s, group, z_max: float = 3.5, passes: int = 2, mad_to_sd: float = 1.4826,
                 retest_all: bool = False):
    """C2b_addendum B1: two-sided robust filter of the lap-time label within each `group` value
    ((game, track, session) in C2b), and the final % gap.

    Pass 1 on all rows of a group: m = median lap, gap = 100 (lap / m - 1), d = mad_to_sd * median |gap|,
    robust z = gap / d; rows with |z| > z_max are dropped. Every further pass (passes - 1 of them)
    recomputes m and d on the rows kept so far and drops |z| > z_max among them (a dropped row never
    comes back); with `retest_all` every row of the group is re-tested against the new m, d instead
    (a row dropped earlier can return). The final gap uses the last m. A group whose d is 0 keeps every row of that pass.

    Returns (keep, gap, median_s, report): keep (n,) bool; gap (n,) % to the final m (NaN never: every
    row gets a gap, dropped rows too); median_s (n,) the final m of the row's group [s]; report =
    {'passes': [{'kept', 'dropped_low', 'dropped_high'}], 'groups': n_groups}. Lower gap = faster."""
    lap_s, group = np.asarray(lap_s, float), np.asarray(group)
    keep = np.ones(len(lap_s), bool)
    gap, med_s = np.full(len(lap_s), np.nan), np.full(len(lap_s), np.nan)
    report = {'groups': int(len(np.unique(group))), 'passes': []}
    inv = np.unique(group, return_inverse=True)[1].ravel()
    for _ in range(passes):
        z = np.zeros(len(lap_s))
        for g in range(inv.max() + 1 if len(inv) else 0):
            rows = np.flatnonzero(inv == g)
            ref = rows[keep[rows]]
            if not len(ref):
                continue
            m = np.median(lap_s[ref])
            gap[rows] = 100. * (lap_s[rows] / m - 1.)
            med_s[rows] = m
            d = mad_to_sd * np.median(np.abs(gap[ref]))
            z[rows] = gap[rows] / d if d > 0 else 0.
        cand = np.ones(len(lap_s), bool) if retest_all else keep
        low, high = cand & (z < -z_max), cand & (z > z_max)
        keep = cand & ~(low | high)
        report['passes'].append({'kept': int(keep.sum()), 'dropped_low': int(low.sum()), 'dropped_high': int(high.sum())})
    return keep, gap, med_s, report


def cells_of(laps: Mapping, games, tracks):
    """(cells, cell_laps): cells = sorted (game index, driver id, track index) with >= 1 lap that has
    samples (S5); cell_laps = per cell the lap keys, ascending."""
    acc = {}
    for key in sorted(laps):
        lap = laps[key]
        if lap.has_samples and lap.game in games and lap.track in tracks:
            acc.setdefault((games.index(lap.game), lap.driver, tracks.index(lap.track)), []).append(key)
    cells = tuple(sorted(acc))
    return cells, tuple(tuple(acc[c]) for c in cells)


@dataclass
class Frame:
    """Rows A: every setup kept by S2 + B1. `gap` is % above the (game, track, session) median lap
    (B1), lower = faster.
    Rows S are `cell >= 0` (the driver has a telemetry lap with samples on that track, S5)."""
    knobs: np.ndarray               # (n, 15) physical, config.KNOB_ORDER
    p: np.ndarray                   # (n, 15) normalised to [-1, 1]
    gap: np.ndarray                 # % lap-time gap
    lap_s: np.ndarray               # lap time [s] (C2b can recompute `gap` per fold with `gaps`)
    game: np.ndarray                # index into games
    track: np.ndarray               # index into track_names
    driver: np.ndarray              # integer id (adapter-defined, no names)
    team: np.ndarray                # index into team_names
    session: np.ndarray             # index into sessions
    controller: np.ndarray          # index into controllers
    assists: dict                   # assist -> index into assist_levels[assist] (0 = config.UNKNOWN)
    date_days: np.ndarray           # days since 1970-01-01, -1 = not published
    setup_hash: np.ndarray          # C1-R3 hash per row
    cell: np.ndarray                # index into cells, -1 = not a style row
    exact_lap: np.ndarray           # lap key of the exact telemetry match, -1 = none
    source_row: np.ndarray          # index of the row in the adapter's loaded setups
    games: tuple                    # source names
    track_names: tuple              # source track names, sorted
    team_names: tuple
    sessions: tuple                 # adapter vocabulary
    controllers: tuple              # adapter vocabulary
    assist_levels: dict             # adapter vocabulary: assist -> levels, index 0 = config.UNKNOWN
    cells: tuple                    # (game index, driver id, track index) with >= 1 telemetry lap with samples
    cell_laps: tuple                # per cell: lap keys
    person: np.ndarray = None       # cross-source integer person id (B6); never written next to a name
    label_median_s: np.ndarray = None   # B1: final median lap of the row's (game, track, session) [s]
    counts: list = field(default_factory=list)          # [(step, rows)]
    box_report: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.gap)

    @property
    def in_s(self) -> np.ndarray:
        return self.cell >= 0


def ranking_stats(frame: Frame, mask=None) -> dict:
    """R6 ranking groups: same (game, driver, track, session) with >= 2 different setups. Returns
    {'groups', 'rows', 'pairs', 'drivers'}; a pair = two rows of one group with different setup hash
    AND different gap. Pairs never cross sessions."""
    idx = np.arange(len(frame)) if mask is None else np.flatnonzero(mask)
    groups = {}
    for i in idx:
        groups.setdefault((frame.game[i], frame.driver[i], frame.track[i], frame.session[i]), []).append(i)
    ng, n_rows, pairs, drivers = 0, 0, 0, set()
    for key, ii in groups.items():
        hashes = frame.setup_hash[ii]
        if len(set(hashes)) < 2:
            continue
        ng += 1
        n_rows += len(ii)
        drivers.add(key[:2])
        pairs += sum(1 for a in range(len(ii)) for b in range(a + 1, len(ii))
                     if hashes[a] != hashes[b] and frame.gap[ii[a]] != frame.gap[ii[b]])
    return {'groups': ng, 'rows': n_rows, 'pairs': pairs, 'drivers': len(drivers)}


__all__ = ['Frame', 'LapLike', 'LapSource', 'cells_of', 'gaps', 'ranking_stats', 'robust_label']
