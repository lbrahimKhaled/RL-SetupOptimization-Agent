"""R2 corner table and clean-corner exit bonus (docs/specs/R2_review.md §R2-2, §R2-3).

Corner table (built once in the main process, like the QSS profile, and handed picklable to the
vector-env workers and the evaluation envs):
  1 v_qss on xi_ref: `qss_lap.lap_time(setup, track, ggv_data=(ggv, traction, None), car, ers,
    drs=False, path=ref.path)['v_profile']` with the g-g-v / traction of the env's cached QSS
    profile (the same call as `qss_profile.line_lap_time`); t_line = cumulative QSS time along
    the line (trapezoid in 1/v, = TUM calc_t_profile; t_line[-1] = the line's QSS lap time).
  2 each line node is matched to Track.path (`track.match` with a running hint), unwrapped and
    v is interpolated onto a regular Track.path grid (step L / ceil(L / grid_m), period L).
  3 apexes = `scipy.signal.find_peaks(-v, prominence=prominence_mps)` on the grid rolled to
    start at the global maximum of v.
  4 consecutive apexes closer than merge_m are one unit.
  5 window start b_k = argmax of v between the last apex of unit k-1 and the first of unit k
    (the start of braking).
  6 exit s_out,k = last apex + min(exit_after_apex_m, (b_{k+1} - last apex) - exit_before_brake_m);
    window_k = [b_k, s_out,k]; windows must be disjoint and >= MIN_WINDOW_M long.
Positions are Track.path arc length [m] in [0, L) (the env's progress coordinate); a window
may wrap through s = 0 (end < start). Units are numbered U1.. in order of their first apex on
Track.path. Speeds in m/s.

Cache: data/corner_table_<key12>.json, key = sha256 of {QSS profile key, xi_ref digest, the
detection fields of CornerBonusConfig, sha256 of this file}. A cached file is used only if its
full key matches; otherwise the table is recomputed in memory (a missing file is written once).

Exit bonus (`CornerTracker`, pure, no simulator): f(rho) = clip((rho - floor) / (1 - floor), 0, 1),
rho = v_exit / v_qss(s_out); bonus = B f (capped at rho = 1, DECISIONS #22). Paid once per pass
when the high-water mark of the env's Track.path distance since hand-over first exceeds the
unit's next threshold d_out + j L (d_out = (s_out - s_handover) mod L); v_exit is the speed
interpolated linearly inside the 50 Hz tick at the crossing. A decision that ends in a terminal
failure forfeits every exit it crossed (the env applies that rule; R2-3.4).
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import datetime
import hashlib
import json
import math
import warnings
import numpy as np

MIN_WINDOW_M = 100.                # R2-2.6: every window at least this long
DETECTION_FIELDS = ('prominence_mps', 'merge_m', 'exit_after_apex_m', 'exit_before_brake_m', 'grid_m')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exit_fraction(rho, floor_ratio):
    """f(rho) = clip((rho - floor) / (1 - floor), 0, 1): 0 at or below the floor, 1 at or above v_qss."""
    return float(min(1., max(0., (float(rho) - floor_ratio) / (1. - floor_ratio))))


def exit_bonus(rho, bonus, floor_ratio):
    """B f(rho) (unweighted, reward units)."""
    return float(bonus) * exit_fraction(rho, floor_ratio)


@dataclass
class CornerTable:
    """Corner units on Track.path (see module docstring). Picklable (numpy + plain lists)."""
    length: float                    # Track.path lap length L [m]
    s: np.ndarray                    # grid stations [m], step = length / len(s)
    v: np.ndarray                    # v_qss of xi_ref mapped to the grid [m/s]
    units: list                      # dicts: name, apexes_m, v_apex, start_m, exit_m, v_qss_exit
    lap_time_s: float                # QSS lap time on xi_ref [s]
    meta: dict = field(default_factory=dict)

    @property
    def step(self):
        return self.length / len(self.s)

    def speed(self, s):
        """v_qss (xi_ref, mapped) at Track.path arc length(s) s (wraps, linear)."""
        return np.interp(np.asarray(s, dtype=float) % self.length, np.r_[self.s, self.length],
                         np.r_[self.v, self.v[0]])

    def in_window(self, s, k):
        """True if s (wrapped) lies in window k = [start, exit] (which may wrap through 0)."""
        u = self.units[k]
        return (float(s) - u['start_m']) % self.length <= (u['exit_m'] - u['start_m']) % self.length

    def unit_at(self, s):
        """Index of the unit whose window contains s, or None."""
        for k in range(len(self.units)):
            if self.in_window(s, k):
                return k
        return None

    def summary(self):
        """JSON-ready table (provenance, logs)."""
        return {'key': self.meta.get('key'), 'lap_time_s': self.lap_time_s, 'length_m': self.length,
                'units': [dict(u) for u in self.units]}


# --- build ---------------------------------------------------------------------------------------
def cache_key(qss, ref, cb_cfg):
    """(sha256 hex, the dict it hashes) of everything the table depends on (R2-2.8)."""
    d = json.loads(json.dumps({'qss_key': qss.meta.get('key'), 'reference_digest': ref.digest,
                               'detection': {k: getattr(cb_cfg, k) for k in DETECTION_FIELDS},
                               'corners_py_sha256': _sha(__file__)}, sort_keys=True))
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(), d


def line_speed_profile(track, ref, qss, env_cfg):
    """(v at the line's nodes (n,), QSS lap time [s]) on xi_ref (R2-2.1)."""
    from ocd_lap import qss_lap
    from .setups import get as get_setup
    ggv_data = (np.asarray(qss.meta['ggv'], dtype=float), np.asarray(qss.meta['traction'], dtype=float), None)
    r = qss_lap.lap_time(get_setup(env_cfg.setup), track, ggv_data=ggv_data, car=env_cfg.car, ers=env_cfg.ers,
                         drs=False, path=np.asarray(ref.path, dtype=float))
    return np.asarray(r['v_profile'], dtype=float), float(r['lap_time_s'])


def cumulative_time(path, v):
    """Cumulative QSS time [s] at the closed line's nodes (len(path)), trapezoid in 1/v between
    nodes: dt = 2 ds / (v_i + v_{i+1}) (TUM calc_t_profile)."""
    el = np.diff(np.asarray(path, dtype=float)[:, 0])
    vc = np.r_[v, v[0]]
    return np.r_[0., np.cumsum(2. * el / (vc[:-1] + vc[1:]))]


def map_to_track(track, ref_path, v, grid_m):
    """(grid s [m], v on the grid): line node i is matched to Track.path with a running hint,
    unwrapped, and v is interpolated (period L) onto stations i * L / ceil(L / grid_m)."""
    L = float(track.length)
    xy = np.asarray(ref_path, dtype=float)[:-1, 1:]
    s = np.empty(len(xy))
    hint = None
    for i, p in enumerate(xy):
        hint = s[i] = float(np.asarray(track.match(p, hint)[0]))
    s = np.unwrap(s, period=L)
    if s[-1] < s[0]:
        raise ValueError('reference line runs against Track.path direction')
    n = int(math.ceil(L / grid_m))
    grid = np.arange(n) * (L / n)
    return grid, np.interp(grid, s % L, v, period=L)


def detect_units(grid_v, step, cb_cfg):
    """Units from v on a periodic grid (R2-2.3 - 2.6). Returns a list of dicts with grid indices
    'apexes', 'start' and 'exit_m_rel' (exit [m] in the rolled frame), and the roll offset i0
    (rolled index 0 = grid index i0, the global maximum of v)."""
    from scipy.signal import find_peaks
    v = np.asarray(grid_v, dtype=float)
    n = len(v)
    i0 = int(np.argmax(v))
    rolled = np.roll(v, -i0)
    peaks, _ = find_peaks(-rolled, prominence=cb_cfg.prominence_mps)
    if len(peaks) == 0:
        return [], i0
    groups = [[int(peaks[0])]]
    for p in peaks[1:]:
        if (p - groups[-1][-1]) * step < cb_cfg.merge_m:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    K = len(groups)
    starts = []
    for k in range(K):                          # argmax between the previous unit's last apex and this first
        lo = groups[k - 1][-1] - (n if k == 0 else 0)
        idx = np.arange(lo + 1, groups[k][0])
        starts.append(int(idx[np.argmax(rolled[idx % n])]))   # rolled index (unit 0: <= 0 allowed)
    units = []
    for k in range(K):
        last = groups[k][-1]
        nxt = starts[k + 1] if k + 1 < K else starts[0] + n
        d = min(cb_cfg.exit_after_apex_m, (nxt - last) * step - cb_cfg.exit_before_brake_m)
        units.append({'apexes': [(a + i0) for a in groups[k]], 'start': starts[k] + i0,
                      'exit_m_rel': last * step + d})
    return units, i0


def build(track, ref, qss, env_cfg, cb_cfg):
    """CornerTable for the env config's car / setup / ERS and xi_ref `ref` (not cached)."""
    v_line, lap = line_speed_profile(track, ref, qss, env_cfg)
    t_line = cumulative_time(ref.path, v_line)
    grid, v = map_to_track(track, ref.path, v_line, cb_cfg.grid_m)
    L = float(track.length)
    step = L / len(grid)
    raw, i0 = detect_units(v, step, cb_cfg)
    units = []
    for u in raw:
        apexes = [float((a % len(grid)) * step) for a in u['apexes']]
        start = float((u['start'] % len(grid)) * step)
        exit_ = float((u['exit_m_rel'] + i0 * step) % L)
        units.append({'apexes_m': apexes, 'start_m': start, 'exit_m': exit_})
    units.sort(key=lambda u: u['apexes_m'][0])
    table = CornerTable(L, grid, v, [], lap, {})
    for k, u in enumerate(units):
        u.update(name=f'U{k + 1}', v_apex=[float(table.speed(a)) for a in u['apexes_m']],
                 v_qss_exit=float(table.speed(u['exit_m'])))
        table.units.append({k2: u[k2] for k2 in ('name', 'apexes_m', 'v_apex', 'start_m', 'exit_m', 'v_qss_exit')})
    check_windows(table)
    key, key_dict = cache_key(qss, ref, cb_cfg)
    table.meta = {'key': key, 'key_dict': key_dict, 'line_lap_time_s': lap, 't_line_end_s': float(t_line[-1]),
                  'reference': ref.name}
    return table


def check_windows(table):
    """Assert every window is >= MIN_WINDOW_M and the windows are disjoint (R2-2.6)."""
    L = table.length
    spans = []
    for u in table.units:
        length = (u['exit_m'] - u['start_m']) % L
        if length < MIN_WINDOW_M:
            raise ValueError(f"{u['name']}: window {length:.1f} m < {MIN_WINDOW_M} m")
        spans.append((u['start_m'], length))
    spans.sort()
    for (a, la), (b, _) in zip(spans, spans[1:] + [(spans[0][0] + L, 0.)] if spans else []):
        if a + la >= b:
            raise ValueError(f'corner windows overlap at s = {b % L:.1f} m')


# --- cache ---------------------------------------------------------------------------------------
def default_path(key):
    from .qss_profile import DATA_DIR
    return DATA_DIR / f'corner_table_{key[:12]}.json'


def save(table, path):
    from ..provenance import run_provenance
    from ..runs import write_json
    write_json(path, {'length': table.length, 's': table.s.tolist(), 'v': table.v.tolist(), 'units': table.units,
                      'lap_time_s': table.lap_time_s, 'meta': table.meta,
                      'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                      'provenance': run_provenance()})


def load(path):
    d = json.loads(Path(path).read_text())
    return CornerTable(float(d['length']), np.asarray(d['s'], dtype=float), np.asarray(d['v'], dtype=float),
                       d['units'], float(d['lap_time_s']), d['meta'])


def load_or_build(track, ref, qss, env_cfg, cb_cfg=None, *, path=None):
    """Cached table if its key matches, else recomputed in memory (a missing file is written)."""
    cb_cfg = env_cfg.corner_bonus if cb_cfg is None else cb_cfg
    key, _ = cache_key(qss, ref, cb_cfg)
    path = Path(path) if path is not None else default_path(key)
    if path.exists():
        cached = load(path)
        if cached.meta.get('key') == key:
            return cached
        warnings.warn(f'{path.name}: corner table key does not match; recomputing in memory', stacklevel=2)
        return build(track, ref, qss, env_cfg, cb_cfg)
    table = build(track, ref, qss, env_cfg, cb_cfg)
    save(table, path)
    return load(path)


def for_env(track, env_cfg, qss, ref=None):
    """Corner table for an EnvConfig (xi_ref = env_cfg.reference unless `ref` is given)."""
    from .reference import ReferenceLine
    ref = ReferenceLine.for_track(track, env_cfg.reference) if ref is None else ref
    return load_or_build(track, ref, qss, env_cfg)


# --- exit bonus ----------------------------------------------------------------------------------
class CornerTracker:
    """Once-per-pass exit crossings on the env's cumulative Track.path distance d since hand-over.

    reset(s_handover): threshold_k = (exit_k - s_handover) mod L; high-water mark h = 0.
    tick(d0, d1, v0, v1): for every unit whose threshold lies in (h, d1] (in order of distance):
    phi = (X - d0) / (d1 - d0) clipped to [0, 1], v_exit = v0 + phi (v1 - v0), rho = v_exit /
    v_qss_exit, bonus = B f(rho); then threshold += L (also past d1 after a projection jump of
    more than a lap, so one tick pays a unit at most once) and h = max(h, d1). Driving backwards
    and re-crossing never pays twice (h only grows). Returns the crossing events (dicts: unit,
    v_exit, v_qss_exit, rho, bonus); the env decides paid / forfeit per decision."""

    def __init__(self, table, bonus, floor_ratio):
        self.table, self.B, self.floor = table, float(bonus), float(floor_ratio)
        self.L = float(table.length)
        self.exits = np.array([u['exit_m'] for u in table.units], dtype=float)
        self.v_exit_qss = np.array([u['v_qss_exit'] for u in table.units], dtype=float)
        self.reset(0.)

    def reset(self, s_handover):
        self.next = (self.exits - float(s_handover)) % self.L
        self.high = 0.

    def tick(self, d0, d1, v0, v1):
        events = []
        if not d1 > self.high:
            return events
        hit = np.flatnonzero((self.next > self.high) & (self.next <= d1))
        for k in hit[np.argsort(self.next[hit])]:
            x = self.next[k]
            phi = min(1., max(0., (x - d0) / (d1 - d0))) if d1 > d0 else 1.
            v = float(v0 + phi * (v1 - v0))
            rho = v / self.v_exit_qss[k]
            events.append({'unit': int(k), 'v_exit': v, 'v_qss_exit': float(self.v_exit_qss[k]), 'rho': float(rho),
                           'bonus': exit_bonus(rho, self.B, self.floor)})
            while self.next[k] <= d1:
                self.next[k] += self.L
        self.high = float(d1)
        return events


def settle_events(events, failed):
    """Decision-level payment rule (R2-3.4): returns (paid sum, events with 'kind'): every crossing
    of a decision that ends in a terminal failure is 'forfeit' (bonus 0); otherwise 'paid' when
    the bonus is > 0 and 'exit_unpaid' at or below the floor ratio."""
    out, total = [], 0.
    for e in events:
        e = dict(e)
        if failed:
            e.update(kind='forfeit', bonus=0.)
        else:
            e['kind'] = 'paid' if e['bonus'] > 0. else 'exit_unpaid'
            total += e['bonus']
        out.append(e)
    return total, out
