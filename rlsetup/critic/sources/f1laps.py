"""f1laps exports (F1 25, F1 26, later more) -> setups, telemetry laps and the C2 dataset (S2, S5).

An export is a folder (or a folder inside a zip archive, read in place) holding `setups.csv` and,
optionally, `laps.csv`, `setup_lap_map.csv`, `time_mapping.csv`, `graph_{throttle,brake,steer}.csv`.
A bare setups CSV file is an export with the setups table only. A bare CSV with a `matched_lap_id`
column (the 16 Nov telemetry-enhanced setup-lap pairs file) is an explicit setup -> lap map only: its
pairs take precedence over `setup_lap_map.csv` for the exact-row pairing; the lap's telemetry must come
from an export already read (nothing is downloaded). The game is a column. New export folders need no
code change: every `setups.csv` found under a given path is one export.

Privacy: usernames are mapped to integer ids keyed by (game, user) (`driver`) and to a cross-game integer
`person` id (the same name in two games = one person, C2b_addendum B6) while the files are read, and the
strings are discarded; no username, source URL or lap / setup id is put in a Frame or written anywhere.
`graph_speed.csv` is not read: no C2 number uses speed.

This module is the F1 adapter only. The containers it fills (`Frame`, cells, ranking groups) are the
source-independent ones of `critic/dataset.py`; its session / controller / assist vocabularies below
are adapter outputs that travel with the Frame (QA M12).
"""
from dataclasses import dataclass, field
from pathlib import Path
import csv
import glob as globlib
import hashlib
import io
import zipfile
import numpy as np
from ..config import Config, KNOB_ORDER, LAP_NO_SAMPLES, LAP_SPACING, LAP_TIME_CLOSURE, OK, StyleInputConfig, UNKNOWN
from ..dataset import Frame, cells_of, ranking_stats, robust_label
from ..knobs import normalise, setup_hash
from .base import Provenance, StyleInput, resample_uniform
from .f1_game_map import F1GameMapConfig, GAME_COLUMNS, box_check, clip_to_box, game_to_knobs
from .f1_reference import REAL_POLE_S

CHANNELS = ('throttle', 'brake', 'steer')
# Output vocabularies of this adapter (S2 step 5).
SESSIONS = ('Race', 'Quali', 'TT')
CONTROLLERS = ('Wheel', 'Controller', 'blank')
# Assist columns and their known levels (F1 26; F1 25 publishes none -> all UNKNOWN). Index 0 = UNKNOWN.
ASSIST_LEVELS = {'traction_control': (UNKNOWN, 'Off', 'Medium', 'Full'),
                 'anti_lock_brakes': (UNKNOWN, 'Off', 'On'),
                 'steering_assist': (UNKNOWN, 'Off', 'On'),
                 'braking_assist': (UNKNOWN, 'Off', 'Low', 'Medium', 'High'),
                 'gearbox': (UNKNOWN, 'Manual', 'Manual & Suggested', 'Automatic'),
                 'racing_line_assist': (UNKNOWN, 'Off', 'Corners', 'Full')}
# raw setup columns `meta` reads
META_COLUMNS = ('game', 'session', 'session_group', 'controller', *ASSIST_LEVELS)
STEER_DEG_PER_UNIT = 115.       # Decision 5: style-rl's OCD conversion (0.15 rad x ratio 13.4), game steer -1..1
# F1 25/26 steer + = right turn (measured: corr(steer, kappa) < 0 on all 753 gated laps), kappa + = left turn
STEER_KAPPA_SIGN = -1
# time_mapping values are milliseconds (S0: 77,237 at 5,270 m for a 77.323 s lap). The export declares no
# unit ('not_declared_by_endpoint'); a wrong unit would fail the R3 time-closure check on every lap.
TIME_MAP_UNIT_S = 1e-3
# declared `unit` of each graph channel in the exports read so far; anything else stops the load
GRAPH_UNITS = {'throttle': ('fraction',), 'brake': ('fraction',), 'steer': ('source_normalized',)}
# explicit setup -> lap pairs file (16 Nov): a bare CSV with this column is read as the table PAIRS_TABLE
PAIRS_LAP_COLUMN, PAIRS_TABLE = 'matched_lap_id', 'setup_lap_pairs.csv'


# ---------------------------------------------------------------------------------------------------
# reading exports
# ---------------------------------------------------------------------------------------------------
class Export:
    """One export: `read(name)` -> bytes of the table `name` or None. `label` identifies it in logs
    and provenance (path relative to the given source, no usernames)."""

    def __init__(self, label, reader, names):
        self.label, self._reader, self.names = label, reader, frozenset(names)

    def read(self, name):
        return self._reader(name) if name in self.names else None


def expand_sources(specs):
    """CLI list -> sorted existing paths; each item is a path or a glob pattern."""
    out = []
    for spec in specs:
        hits = sorted(globlib.glob(str(spec))) if any(c in str(spec) for c in '*?[') else [str(spec)]
        if not hits:
            raise FileNotFoundError(f'no export matches {spec!r}')
        out += [h for h in hits if not Path(h).name.startswith('.')]
    return [Path(p) for p in dict.fromkeys(out)]


def discover(path) -> list:
    """Exports under `path`: a directory (every `setups.csv` below it), a .zip (every `setups.csv`
    member; nothing is extracted) or a setups .csv file."""
    path = Path(path)
    if path.is_dir():
        out = []
        for s in sorted(path.rglob('setups.csv')):
            root = s.parent
            names = {p.name for p in root.iterdir() if p.is_file()}
            out.append(Export(f'{path.name}/{root.relative_to(path).as_posix()}'.rstrip('/.'),
                              lambda n, root=root: (root / n).read_bytes(), names))
        return out
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist() if not m.endswith('/')]
        out = []
        for m in sorted(m for m in members if m.rsplit('/', 1)[-1] == 'setups.csv'):
            prefix = m[:-len('setups.csv')]
            names = {x[len(prefix):] for x in members if x.startswith(prefix) and '/' not in x[len(prefix):]}

            def reader(n, prefix=prefix):
                with zipfile.ZipFile(path) as z:
                    return z.read(prefix + n)
            out.append(Export(f'{path.name}:{prefix}'.rstrip('/:'), reader, names))
        return out
    if path.suffix.lower() == '.csv' and path.is_file():
        with open(path, encoding='utf-8-sig', newline='') as f:
            header = next(csv.reader(f), [])
        name = PAIRS_TABLE if PAIRS_LAP_COLUMN in header else 'setups.csv'
        return [Export(path.name, lambda n: path.read_bytes(), {name})]
    raise ValueError(f'{path}: not an export folder, a .zip archive or a setups .csv file')


def _rows(data: bytes):
    return csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))


def _num(value):
    """float, or None if blank / not finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def lap_time_s(text):
    """'1:17.323' or '77.323' -> seconds; None if blank or malformed."""
    text = (text or '').strip()
    try:
        if ':' in text:
            m, s = text.split(':')
            v = 60. * float(m) + float(s)
        else:
            v = float(text)
    except ValueError:
        return None
    return v if np.isfinite(v) and v > 0 else None


def session_of(row):
    """Normalised session {Race, Quali, TT} (S2 step 5): the export's `session_group` when present,
    else the detailed `session` label (Qualifying 1-3 and Sprint Shootout count as Quali)."""
    for value in (row.get('session_group') or '', row.get('session') or ''):
        if value == 'Race':
            return 'Race'
        if value == 'Time Trial':
            return 'TT'
        if value.startswith('Qualifying') or value.startswith('Sprint Shootout'):
            return 'Quali'
    raise ValueError(f"unknown session {row.get('session')!r} / group {row.get('session_group')!r}")


def controller_of(value):
    value = (value or '').strip()
    if value == '':
        return 'blank'
    if value in CONTROLLERS:
        return value
    raise ValueError(f'unknown controller {value!r}')


def assists_of(row):
    """{assist: level string from ASSIST_LEVELS[assist]}; UNKNOWN for a missing column or a value outside
    the known levels (the F1 26 export has a few garbled cells)."""
    return {a: (v if (v := (row.get(a) or '').strip()) in levels[1:] else UNKNOWN) for a, levels in ASSIST_LEVELS.items()}


def _date_days(text):
    """ISO date -> days since 1970-01-01, -1 if blank / malformed."""
    try:
        return int(np.datetime64((text or '').strip(), 'D').astype(int))
    except ValueError:
        return -1


@dataclass
class TelemetryLap:
    """One public leaderboard lap (a `dataset.LapLike`). Distances in metres along the game's lap;
    `channels[c]` is (n, 2) [distance_m, value] sorted by sample index; `time_map` is (m, 2)
    [distance_m, time in source units]."""
    game: str
    track: str
    driver: int
    lap_time_s: float
    controller: str
    assists: dict
    date_days: int
    channels: dict = field(default_factory=dict)
    time_map: np.ndarray = None
    person: int = -1                # cross-game person id (B6)

    @property
    def has_samples(self) -> bool:
        return 'steer' in self.channels and len(self.channels['steer']) > 0


@dataclass
class F1LapsData:
    """Everything read from the exports (a `dataset.LapSource`). `setups` rows are dicts without any
    username."""
    setups: list                    # one dict per setup_id (deduplicated), raw (no filter applied yet)
    laps: dict                      # lap key (int) -> TelemetryLap, deduplicated by lap_id
    exact: dict                     # setup index -> lap key, accepted exact matches with samples
    counts: list                    # [(step, rows)]
    provenance: Provenance
    n_drivers: int
    n_persons: int = 0
    dropped_tracks: dict = field(default_factory=dict)   # B2: {track: {'setups': n, 'laps': n}}


def _parse_graph(data: bytes, wanted, units=None):
    """{lap_id: (n, 2) [distance_m, value]} sorted by sample index; null values dropped. `units`: the
    accepted values of the `unit` column (when the file has one); another unit raises ValueError."""
    acc = {}
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    header = next(reader)
    i_lap, i_idx, i_d, i_v = (header.index(c) for c in ('lap_id', 'sample_index', 'distance_m', 'source_value'))
    i_unit = header.index('unit') if units and 'unit' in header else None
    for row in reader:
        if row[i_lap] in wanted:
            if i_unit is not None and row[i_unit] not in units:
                raise ValueError(f'graph unit {row[i_unit]!r} not in {units}')
            try:
                acc.setdefault(row[i_lap], []).append((int(row[i_idx]), float(row[i_d]), float(row[i_v])))
            except ValueError:
                continue                # null sample: stays a gap
    out = {}
    for lap_id, a in acc.items():
        a = np.array(sorted(a))
        a = a[np.isfinite(a).all(axis=1)]
        if len(a):
            out[lap_id] = a[:, 1:]
    return out


def _parse_time_map(data: bytes, wanted):
    acc = {}
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    header = next(reader)
    i_lap, i_d, i_t = (header.index(c) for c in ('lap_id', 'source_distance_key', 'source_time_value'))
    for row in reader:
        if row[i_lap] in wanted:
            try:
                acc.setdefault(row[i_lap], []).append((float(row[i_d]), float(row[i_t])))
            except ValueError:
                continue
    return {k: np.array(sorted(v)) for k, v in acc.items()}


def game_label(game):
    """Canonical game label: 'F1_25' (16 Nov files) and 'F1 25' (f1laps exports) are one game, 'F1 25'."""
    return (game or '').strip().replace('_', ' ')


def load_exports(paths, log=print, drop_tracks=(), games=None) -> F1LapsData:
    """Read all exports under `paths` (see `discover`). Setups are deduplicated by `setup_id`, laps by
    `lap_id` (first export wins for the lap's metadata). Telemetry (each graph channel, the time map)
    is taken from the first export that has it for that lap: a lap listed earlier without samples is
    filled from a later export that lists it with samples (QA M1). A setups / laps table whose bytes
    were already read from another export (same name and sha256) is not counted again. sha256 of every
    file read goes to `provenance`. Setups, laps and exact matches on `drop_tracks` (B2) are removed
    after the ids are assigned, before anything else uses the data (counted in `dropped_tracks`).
    `person` (B6): integer per username across games, from the sorted names, built before the names
    are discarded. `games` (DataConfig.games): setups of other games are skipped before any id is given
    (counted per game). Exact pairs: explicit pairs files first, then `setup_lap_map.csv`; the first
    accepted pair (lap with samples) per setup wins."""
    exports = [e for p in paths for e in discover(p)]
    if not exports:
        raise ValueError(f'no export with a setups.csv under {[str(p) for p in paths]}')
    prov, seen_files = Provenance(), {}
    users = {}                      # (game, user) -> None; replaced by integer ids below, then cleared
    setups, setup_ids, laps, lap_ids, exact_pairs, explicit_pairs = [], {}, {}, {}, [], []
    keep_games = None if games is None else {game_label(g) for g in games}
    skipped_games = {}
    n_setup_rows = n_lap_rows = 0
    filled_later, seen = set(), set()   # lap keys filled by a later export / listed by an earlier one

    def fetch(export, name, dedup=True):
        """Bytes of `name` (sha recorded); None when absent, or when `dedup` and the same bytes were read."""
        data = export.read(name)
        if data is None:
            return None
        sha = hashlib.sha256(data).hexdigest()
        prov.files[f'{export.label}/{name}'] = sha
        if (name, sha) in seen_files:
            if dedup:
                prov.skipped_duplicates.append(f'{export.label}/{name} == {seen_files[(name, sha)]}')
                return None
        else:
            seen_files[(name, sha)] = f'{export.label}/{name}'
        return data

    for export in exports:
        games = set()
        data = fetch(export, 'setups.csv')
        if data is not None:
            for r in _rows(data):
                n_setup_rows += 1
                r['game'] = game_label(r['game'])
                if keep_games is not None and r['game'] not in keep_games:
                    skipped_games[r['game']] = skipped_games.get(r['game'], 0) + 1
                    continue
                games.add(r['game'])
                if r['setup_id'] in setup_ids:
                    continue
                users.setdefault((r['game'], r['user']))
                setup_ids[r['setup_id']] = len(setups)
                setups.append({'game': r['game'], 'track': r['track'], 'team': r.get('team', ''),
                               '_user': (r['game'], r['user']), 'conditions': r.get('conditions', ''),
                               'meta_raw': {c: (r.get(c) or '') for c in META_COLUMNS},
                               'date_days': _date_days(r.get('date_key')), 'lap_s': lap_time_s(r.get('lap_time')),
                               'values': {c: _num(r.get(c)) for c in GAME_COLUMNS}})
        elif 'setups.csv' in export.names:      # duplicate bytes: the game is still needed for laps.csv
            games = {game_label(r['game']) for r in _rows(export.read('setups.csv'))}
        data = fetch(export, 'laps.csv')
        lap_bytes = data if data is not None else export.read('laps.csv')   # duplicate bytes: ids still needed
        listed = {}                 # lap_id -> key of every lap this export lists
        if lap_bytes is not None:
            for r in _rows(lap_bytes):
                n_lap_rows += data is not None
                if r['lap_id'] in lap_ids:
                    listed[r['lap_id']] = lap_ids[r['lap_id']]
                    continue
                game = game_label(r.get('game')) or (next(iter(games)) if len(games) == 1 else None)
                if game is None:
                    raise ValueError(f'{export.label}/laps.csv has no game column and its setups.csv does not name one game')
                users.setdefault((game, r['user']))
                lap_ids[r['lap_id']] = key = len(lap_ids)
                listed[r['lap_id']] = key
                lap = TelemetryLap(game=game, track=r['track'], driver=-1, lap_time_s=lap_time_s(r.get('lap_time')),
                                   controller=controller_of(r.get('controller')), assists=assists_of(r),
                                   date_days=_date_days(r.get('date_key')))
                lap._user = (game, r['user'])
                laps[key] = lap
        for ch in CHANNELS:
            wanted = {lid: k for lid, k in listed.items() if ch not in laps[k].channels}
            data = fetch(export, f'graph_{ch}.csv', dedup=False) if wanted else None
            if data is not None:
                for lap_id, a in _parse_graph(data, wanted, GRAPH_UNITS[ch]).items():
                    laps[wanted[lap_id]].channels[ch] = a
                    if wanted[lap_id] in seen:
                        filled_later.add(wanted[lap_id])
        wanted = {lid: k for lid, k in listed.items() if laps[k].time_map is None}
        data = fetch(export, 'time_mapping.csv', dedup=False) if wanted else None
        if data is not None:
            for lap_id, a in _parse_time_map(data, wanted).items():
                laps[wanted[lap_id]].time_map = a
                if wanted[lap_id] in seen:
                    filled_later.add(wanted[lap_id])
        seen |= set(listed.values())
        data = fetch(export, 'setup_lap_map.csv')
        if data is not None:
            exact_pairs += [(r['setup_id'], r['lap_id']) for r in _rows(data) if r.get('has_telemetry') == 'true']
        data = fetch(export, PAIRS_TABLE)
        if data is not None:
            explicit_pairs += [(r['setup_id'], r[PAIRS_LAP_COLUMN]) for r in _rows(data)
                               if r.get('has_telemetry') == 'true' and r[PAIRS_LAP_COLUMN]]
        log(f'read {export.label}: setups so far {len(setups)}, laps so far {len(laps)}')

    # usernames -> integer ids keyed by (game, user) and cross-game person ids (B6); the strings end here
    ids = {key: i for i, key in enumerate(sorted(users))}
    persons = {name: i for i, name in enumerate(sorted({u for _, u in users}))}
    for s in setups:
        key = s.pop('_user')
        s['driver'], s['person'] = ids[key], persons[key[1]]
    for lap in laps.values():
        lap.driver, lap.person = ids[lap._user], persons[lap._user[1]]
        del lap._user
    n_drivers, n_persons = len(ids), len(persons)
    users.clear()
    ids.clear()
    persons.clear()

    exact = {}
    for setup_id, lap_id in explicit_pairs + exact_pairs:
        if setup_id in setup_ids and lap_id in lap_ids and laps[lap_ids[lap_id]].has_samples:
            exact.setdefault(setup_ids[setup_id], lap_ids[lap_id])
    usable = [(a, b) for a, b in explicit_pairs if a in setup_ids and b in lap_ids and laps[lap_ids[b]].has_samples]
    counts = [('setup rows read (all exports)', n_setup_rows),
              *((f'setup rows skipped: game {g} (no S4 map)', n) for g, n in sorted(skipped_games.items())),
              ('unique setup_id', len(setups)),
              ('lap rows read (all exports)', n_lap_rows), ('unique lap_id', len(laps)),
              ('laps with graph samples', sum(l.has_samples for l in laps.values())),
              ('laps given telemetry by a later export', len(filled_later)),
              ('explicit setup-lap pairs read (pairs files)', len(explicit_pairs)),
              ('explicit pairs with the setup kept and the lap telemetry available', len(usable)),
              ('explicit pairs not in any setup_lap_map.csv (new exact pairs)', len(set(usable) - set(exact_pairs))),
              ('exact setup-lap matches with samples', len(exact))]
    dropped = {}
    if drop_tracks:                     # B2: before anything else uses the data
        drop = set(drop_tracks)
        for t in sorted(drop):
            dropped[t] = {'setups': sum(s['track'] == t for s in setups),
                          'laps': sum(l.track == t for l in laps.values())}
        keep_rows = [i for i, s in enumerate(setups) if s['track'] not in drop]
        new_index = {old: new for new, old in enumerate(keep_rows)}
        setups = [setups[i] for i in keep_rows]
        laps = {k: l for k, l in laps.items() if l.track not in drop}
        exact = {new_index[i]: k for i, k in exact.items() if i in new_index and k in laps}
        counts += [(f'dropped tracks {sorted(drop)}: setups', sum(d['setups'] for d in dropped.values())),
                   (f'dropped tracks {sorted(drop)}: laps', sum(d['laps'] for d in dropped.values())),
                   ('unique setup_id after drop_tracks', len(setups)), ('unique lap_id after drop_tracks', len(laps)),
                   ('exact setup-lap matches with samples after drop_tracks', len(exact))]
    for step, n in counts:
        log(f'{step}: {n}')
    return F1LapsData(setups=setups, laps=laps, exact=exact, counts=counts, provenance=prov, n_drivers=n_drivers,
                      n_persons=n_persons, dropped_tracks=dropped)


# ---------------------------------------------------------------------------------------------------
# setup pipeline (S2) -> dataset.Frame
# ---------------------------------------------------------------------------------------------------
def real_world_check(data: F1LapsData, cfg: Config = Config(), real_pole=None) -> dict:
    """B1 real-world check, review only: per (game, track) r = fastest f1laps TT top-10 lap (the
    telemetry laps of laps.csv) / T_ref (season-matched real pole, `f1_reference.REAL_POLE_S`); `flag`
    when r is outside `real_ref_flag_band` (possible layout or label mismatch: for review, never an
    automatic drop). Tracks without a reference are listed under 'skipped'."""
    real_pole = REAL_POLE_S if real_pole is None else real_pole
    lo, hi = cfg.data.real_ref_flag_band
    fastest = {}
    for lap in data.laps.values():
        if lap.lap_time_s:
            k = (lap.game, lap.track)
            fastest[k] = min(fastest.get(k, np.inf), lap.lap_time_s)
    keys = sorted({(s['game'], s['track']) for s in data.setups} | set(fastest))
    out = {'rows': [], 'skipped': []}
    for k in keys:
        if k not in real_pole or k not in fastest:
            out['skipped'].append({'game': k[0], 'track': k[1], 'has_reference': k in real_pole,
                                   'has_tt_laps': k in fastest})
            continue
        r = fastest[k] / real_pole[k]
        out['rows'].append({'game': k[0], 'track': k[1], 'fastest_tt_s': fastest[k], 'T_ref_s': real_pole[k],
                            'r': r, 'flag': bool(r < lo or r > hi)})
    return out


def build_frame(data: F1LapsData, cfg: Config = Config(), map_cfg: F1GameMapConfig = F1GameMapConfig(),
                adapter: 'F1LapsAdapter' = None, log=print, real_pole=None) -> Frame:
    """S2 pipeline as amended by C2b_addendum B1/B2, row counts logged per step:
    1. dry, lap time and all 15 mapped parameters present;
    2. `drop_tracks` (B2; normally already applied by `load_exports`);
    3. inside the OCD box (S2 step 4; raises ValueError when a game loses more than
       `max_bounds_drop_frac`, the stop rule);
    4. B1 hard cut: lap < real_ref_min_ratio * T_ref (season-matched real pole; tracks without a
       reference are not cut); raises ValueError when it removes more than `real_ref_max_group_drop`
       of any (game, track, session) group;
    5. B1 label: two-sided two-pass robust filter per (game, track, session) (`dataset.robust_label`),
       gap = % to the final group median. This is a selection on the label, computed on all rows.
    Knobs come from `adapter.to_knobs`, the nuisance descriptors from `adapter.meta` (QA M7)."""
    adapter = F1LapsAdapter(map_cfg, cfg) if adapter is None else adapter
    real_pole = REAL_POLE_S if real_pole is None else real_pole
    counts = list(data.counts)
    dc = cfg.data

    def count(step, n, extra=''):
        counts.append((step, int(n)))
        log(f'{step}: {n} rows{extra}')

    def per_game(mask):
        return ''.join(f'; {g} {int(mask[game == i].sum())}' for i, g in enumerate(games))

    rows = [(i, s) for i, s in enumerate(data.setups)
            if s['conditions'] == 'Dry' and s['lap_s'] is not None and all(v is not None for v in s['values'].values())]
    count('1. dry, lap time and all 15 mapped parameters present (unique setup_id)', len(rows))
    rows = [(i, s) for i, s in rows if s['track'] not in dc.drop_tracks]
    count(f'2. tracks not in drop_tracks {list(dc.drop_tracks)} (B2)', len(rows))
    games = tuple(sorted({s['game'] for _, s in rows}))
    tracks = tuple(sorted(({s['track'] for _, s in rows} | {l.track for l in data.laps.values()}) - set(dc.drop_tracks)))
    game = np.array([games.index(s['game']) for _, s in rows], dtype=int)
    table = {c: np.array([s['values'][c] for _, s in rows], dtype=float) for c in GAME_COLUMNS}
    knobs = adapter.to_knobs(table)
    keep, report = box_check(knobs, np.array([games[g] for g in game], dtype=object), dc, log)
    count('3. inside the OCD box', keep.sum(), per_game(keep))
    rows, knobs = [r for r, m in zip(rows, keep) if m], clip_to_box(knobs[keep])
    game = game[keep]
    track = np.array([tracks.index(s['track']) for _, s in rows], dtype=int)
    lap = np.array([s['lap_s'] for _, s in rows])
    meta = adapter.meta({c: np.array([s['meta_raw'][c] for _, s in rows], dtype=object) for c in META_COLUMNS})
    session = np.array([adapter.sessions.index(v) for v in meta['session']], dtype=int)
    group = (game * len(tracks) + track) * len(adapter.sessions) + session      # (game, track, session)

    # B1 hard cut against the season-matched real pole
    t_ref = np.array([real_pole.get((games[g], tracks[t]), np.nan) for g, t in zip(game, track)])
    cut = np.isfinite(t_ref) & (lap < dc.real_ref_min_ratio * t_ref)
    hard = {'n_cut': int(cut.sum()), 'groups_over_limit': [], 'rows_without_reference': int((~np.isfinite(t_ref)).sum())}
    for gk in np.unique(group[cut]):
        m = group == gk
        frac = float(cut[m].mean())
        g_, rest = divmod(int(gk), len(tracks) * len(adapter.sessions))
        t_, s_ = divmod(rest, len(adapter.sessions))
        if frac > dc.real_ref_max_group_drop:
            hard['groups_over_limit'].append({'game': games[g_], 'track': tracks[t_], 'session': adapter.sessions[s_],
                                              'cut': int(cut[m].sum()), 'rows': int(m.sum())})
    if hard['groups_over_limit']:
        raise ValueError(f"B1 hard cut (lap < {dc.real_ref_min_ratio:g} T_ref) removes more than "
                         f"{100 * dc.real_ref_max_group_drop:g} % of a group: {hard['groups_over_limit']}")
    keep = ~cut
    count(f'4. lap >= {dc.real_ref_min_ratio:g} x season-matched real pole (B1 hard cut; '
          f"{hard['rows_without_reference']} rows have no reference)", keep.sum(), per_game(keep))

    # B1 robust label per (game, track, session)
    sub = np.flatnonzero(keep)
    k2, gap_sub, med_sub, label_report = robust_label(lap[sub], group[sub], dc.label_z_max, dc.label_passes, dc.mad_to_sd,
                                                  dc.label_retest_all)
    sess_sub = session[sub]
    for j, pr in enumerate(label_report['passes']):
        log(f"   B1 pass {j + 1}: kept {pr['kept']}, dropped low {pr['dropped_low']}, high {pr['dropped_high']}")
    label_report['hard_cut'] = hard
    label_report['groups_min_rows'] = int(np.bincount(np.unique(group[sub], return_inverse=True)[1].ravel()).min())
    dropped = ~k2
    # dropped rows by session, split by the sign of their gap to the final median
    label_report['dropped_by_session'] = {
        adapter.sessions[si]: {'low': int((dropped & (sess_sub == si) & (gap_sub < 0)).sum()),
                               'high': int((dropped & (sess_sub == si) & (gap_sub > 0)).sum())}
        for si in range(len(adapter.sessions))}
    sel = sub[k2]
    count(f'5. |robust z| <= {dc.label_z_max:g} per (game, track, session), {dc.label_passes} passes (B1; rows A)',
          len(sel), per_game(np.isin(np.arange(len(lap)), sel)))
    gap, med_s = np.full(len(lap), np.nan), np.full(len(lap), np.nan)
    gap[sub], med_s[sub] = gap_sub, med_sub
    rows = [rows[i] for i in sel]
    knobs, game, track, lap, gap, med_s, session = knobs[sel], game[sel], track[sel], lap[sel], gap[sel], med_s[sel], session[sel]
    meta = {'source': meta['source'][sel], 'controller': meta['controller'][sel],
            'assists': {a: v[sel] for a, v in meta['assists'].items()}}
    teams = tuple(sorted({s['team'] for _, s in rows}))
    driver = np.array([s['driver'] for _, s in rows], dtype=int)
    person = np.array([s.get('person', -1) for _, s in rows], dtype=int)

    cells, cell_laps = cells_of(data.laps, games, tracks)          # S5
    cell_index = {c: i for i, c in enumerate(cells)}
    cell = np.array([cell_index.get((g, d, t), -1) for g, d, t in zip(game, driver, track)], dtype=int)
    exact = np.array([data.exact.get(i, -1) for i, _ in rows], dtype=int)
    report = dict(report, label=label_report)
    frame = Frame(knobs=knobs, p=normalise(knobs), gap=gap, lap_s=lap, game=game, track=track, driver=driver,
                  team=np.array([teams.index(s['team']) for _, s in rows], dtype=int), session=session,
                  controller=np.array([adapter.controllers.index(v) for v in meta['controller']], dtype=int),
                  assists={a: np.array([adapter.assist_levels[a].index(v) for v in meta['assists'][a]], dtype=int)
                           for a in adapter.assist_levels},
                  date_days=np.array([s['date_days'] for _, s in rows], dtype=int), setup_hash=setup_hash(knobs),
                  cell=cell, exact_lap=exact, source_row=np.array([i for i, _ in rows], dtype=int), games=games,
                  track_names=tracks, team_names=teams, sessions=tuple(adapter.sessions),
                  controllers=tuple(adapter.controllers), assist_levels=dict(adapter.assist_levels), cells=cells,
                  cell_laps=cell_laps, person=person, label_median_s=med_s, counts=counts, box_report=report)
    if list(meta['source']) != [games[g] for g in game]:
        raise AssertionError('adapter meta source != setup game')
    count('6. session, controller and assists normalised (rows A)', len(frame))
    count('rows S (driver has a telemetry lap with samples on that track)', frame.in_s.sum(),
          ''.join(f'; {g} {int((frame.in_s & (game == i)).sum())}' for i, g in enumerate(games)))
    count('rows S that are exact telemetry rows', (frame.in_s & (exact >= 0)).sum())
    return frame


def s_rows_before_filters(data: F1LapsData) -> dict:
    """{game: setups (all unique setup_id, no filter) by a driver on a track where that driver has a
    telemetry lap with samples} - the number S5 / the review quote before filters."""
    cells = {(l.game, l.driver, l.track) for l in data.laps.values() if l.has_samples}
    out = {}
    for s in data.setups:
        if (s['game'], s['driver'], s['track']) in cells:
            out[s['game']] = out.get(s['game'], 0) + 1
    return out


# ---------------------------------------------------------------------------------------------------
# telemetry lap -> style input (S3, R3)
# ---------------------------------------------------------------------------------------------------
def _increasing(a: np.ndarray):
    """(rows of (n, 2) [distance, value] whose distance exceeds every earlier one, number dropped).
    np.interp and the distance-based time base need strictly increasing distance; of samples that tie
    (or step back) the first is kept (QA M5). Measured on the 859 real laps with samples: 295 have steer
    ties (40,897 samples); 78 lose > 1 % (64 of them F1 25), 42 lose > 10 % (all F1 25), max 30.5 %;
    no time-map ties. The count is stored per lap (`StyleInput.dropped_ties`, `LapStyles.n_dropped_ties`)."""
    if not len(a):
        return a, 0
    prev = np.maximum.accumulate(np.r_[-np.inf, a[:-1, 0]])
    keep = a[:, 0] > prev
    return a[keep], int((~keep).sum())


def lap_style_input(lap: TelemetryLap, cfg: StyleInputConfig = StyleInputConfig()) -> StyleInput:
    """S3 + R3. Time of each sample from `time_mapping` with (0 m, 0 s) prepended, linear in distance
    (past the last key: continued at the speed of the last key interval). Pedals are put on the
    steering samples' distances, then everything is resampled linearly to a uniform `resample_hz`
    clock. steer_deg = game steer x 115 (sign as the game gives it: + = right turn in F1 25/26).
    Distance ties: first sample kept, counts in `dropped_ties`.
    QC (R3): a missing channel / time map, |t_end / lap - 1| > 1 % (t_end = time at the last key) or a
    median sample spacing > 8 m rejects the lap; `reason` says which."""
    empty = np.zeros(0)
    ch, ties = {}, {}
    for c in CHANNELS:
        if c in lap.channels:
            ch[c], ties[c] = _increasing(lap.channels[c])
    tm = None
    if lap.time_map is not None:
        tm, ties['time_map'] = _increasing(lap.time_map)
    h = hashlib.sha256()
    for c in CHANNELS:
        h.update(np.ascontiguousarray(ch.get(c, empty)).tobytes())
    h.update(np.ascontiguousarray(empty if tm is None else tm).tobytes())
    out = StyleInput(empty, empty, empty, empty, empty, empty, empty, lap_time_s=float(lap.lap_time_s or np.nan),
                     reason=LAP_NO_SAMPLES, digest=h.hexdigest(), dropped_ties=ties)
    if 'steer' in ch and len(ch['steer']) >= 2:         # recorded even when the lap is rejected below
        out.median_spacing_m = float(np.median(np.diff(ch['steer'][:, 0])))
    if any(c not in ch or len(ch[c]) < 4 for c in CHANNELS) or tm is None or len(tm) < 2 or not lap.lap_time_s:
        return out
    dist, steer = ch['steer'][:, 0], ch['steer'][:, 1] * STEER_DEG_PER_UNIT
    out.raw_distance_m, out.raw_steer_deg = dist, steer
    out.lap_length_m = float(dist[-1])
    key_d, key_t = np.r_[0., tm[:, 0]], np.r_[0., tm[:, 1] * TIME_MAP_UNIT_S]
    if key_d[1] <= 0:                                   # the source already has a key at 0 m
        key_d, key_t = key_d[1:], key_t[1:]
    out.t_end_s = float(key_t[-1])
    if not np.all(np.diff(key_t) > 0) or abs(out.t_end_s / lap.lap_time_s - 1) > cfg.max_time_closure_rel:
        out.reason = LAP_TIME_CLOSURE
        return out
    if out.median_spacing_m > cfg.max_median_spacing_m:
        out.reason = LAP_SPACING
        return out
    t = np.interp(dist, key_d, key_t)
    past = dist > key_d[-1]
    if past.any():
        v_last = (key_d[-1] - key_d[-2]) / (key_t[-1] - key_t[-2])
        t[past] = key_t[-1] + (dist[past] - key_d[-1]) / v_last
    raw = {'throttle': np.interp(dist, ch['throttle'][:, 0], ch['throttle'][:, 1]),
           'brake': np.interp(dist, ch['brake'][:, 0], ch['brake'][:, 1]), 'steer_deg': steer, 'distance_m': dist}
    grid, u = resample_uniform(t, raw, cfg.resample_hz)
    out.time_s, out.throttle, out.brake, out.steer_deg, out.distance_m = grid, u['throttle'], u['brake'], u['steer_deg'], \
        u['distance_m']
    out.reason = OK
    return out


class F1LapsAdapter:
    """`SourceAdapter` of the F1 25 / F1 26 f1laps exports."""
    name = 'f1laps'
    sessions = SESSIONS
    controllers = CONTROLLERS
    assist_levels = ASSIST_LEVELS
    steer_kappa_sign = STEER_KAPPA_SIGN

    def __init__(self, map_cfg: F1GameMapConfig = F1GameMapConfig(), cfg: Config = Config()):
        self.map_cfg, self.cfg = map_cfg, cfg

    def to_knobs(self, table) -> np.ndarray:
        return game_to_knobs(table, self.map_cfg, self.cfg.car)

    def style_inputs(self, lap: TelemetryLap, cfg: StyleInputConfig = None) -> StyleInput:
        return lap_style_input(lap, self.cfg.style_input if cfg is None else cfg)

    def meta(self, table) -> dict:
        """Per row (columns META_COLUMNS, missing columns read as blank): 'source', 'session',
        'controller' (strings from SESSIONS / CONTROLLERS), 'assists' ({assist: level strings})."""
        n = len(next(iter(table.values())))
        rows = [{k: table[k][i] for k in table} for i in range(n)]
        assists = [assists_of(r) for r in rows]
        return {'source': np.array([r['game'] for r in rows], dtype=object),
                'session': np.array([session_of(r) for r in rows], dtype=object),
                'controller': np.array([controller_of(r.get('controller')) for r in rows], dtype=object),
                'assists': {a: np.array([x[a] for x in assists], dtype=object) for a in ASSIST_LEVELS}}


__all__ = ['ASSIST_LEVELS', 'CHANNELS', 'CONTROLLERS', 'Export', 'F1LapsAdapter', 'F1LapsData', 'Frame', 'KNOB_ORDER',
           'META_COLUMNS', 'SESSIONS', 'STEER_KAPPA_SIGN', 'TelemetryLap', 'build_frame', 'discover', 'expand_sources',
           'real_world_check',
           'lap_style_input', 'load_exports', 'ranking_stats', 's_rows_before_filters']
