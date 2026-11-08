"""R4 speed-profile terms (docs/specs/R4_speed_profile.md §1, binding review docs/specs/R4_review.md;
DECISIONS #26). Pure numpy, no simulator; corners.py is reused unchanged (its cache key stays).

Closeness score (squared, symmetric, non-negative):
    g(v, v*) = max(0, 1 - ((v - v*) / (sigma v*))^2)        1 at v = v*, 0 at and beyond +-sigma v*
Speed potential (env reward, RewardConfig.speed_k / speed_sigma):
    Phi_v(s, v) = k_v g(v, v_qss(s)),  v = |(v_x, v_y)| [m/s], s = Track.path arc length [m],
    v_qss = CornerTable.speed (QSS on xi_ref mapped to Track.path).
Corner entry + exit terms (CornerBonusConfig mode 'entry_exit'), per corner unit:
    entry = window start (corners.py step 5, the QSS braking point), exit = the unit's exit_m;
    B_in g(v_entry, v_qss(entry)) paid at the entry crossing, B_out g(v_exit, v_qss_exit) at the exit.
    A terminal failure (off track, non-finite, wrong way) forfeits every crossing of the failing
    decision (corners.settle_events) and takes back every entry paid earlier in this pass whose
    exit has not been crossed yet (a negative term in the failing decision). A truncation (time
    limit, stall) takes nothing back. Hand-over at or after an entry (inside the window): only the
    exit is paid. Crossings come from two corners.CornerTracker instances (threshold / high-water
    mark / speed interpolation inside the 50 Hz tick, unchanged).
"""
import numpy as np
from .corners import CornerTable, CornerTracker, settle_events


def closeness(v, v_star, sigma):
    """g(v, v*) (see module docstring); float for scalars, array otherwise. v* > 0."""
    x = (np.asarray(v, dtype=float) - np.asarray(v_star, dtype=float)) / (sigma * np.asarray(v_star, dtype=float))
    g = np.maximum(0., 1. - x * x)
    return float(g) if g.ndim == 0 else g


def closeness_rho(rho, sigma):
    """g as a function of rho = v / v*: max(0, 1 - ((rho - 1) / sigma)^2)."""
    x = (np.asarray(rho, dtype=float) - 1.) / sigma
    g = np.maximum(0., 1. - x * x)
    return float(g) if g.ndim == 0 else g


def speed_potential(table, s, v, k, sigma):
    """(Phi_v, g) at Track.path arc length s [m] and speed v [m/s] (scalars or arrays)."""
    g = closeness(v, table.speed(s), sigma)
    return k * g, g


def entry_points(table):
    """[(unit name, entry_m = window start, v_qss there)] of a CornerTable."""
    return [(u['name'], float(u['start_m']), float(table.speed(u['start_m']))) for u in table.units]


def _entry_view(table):
    """CornerTable whose units' 'exit' is the entry point, so a CornerTracker detects entries."""
    units = [dict(u, exit_m=float(u['start_m']), v_qss_exit=float(table.speed(u['start_m']))) for u in table.units]
    return CornerTable(table.length, table.s, table.v, units, table.lap_time_s, dict(table.meta))


class EntryExitTracker:
    """Entry / exit crossings and their terms (module docstring). Same interface as
    corners.CornerTracker for the env: reset(s_handover), tick(d0, d1, v0, v1) per 50 Hz tick
    (d = Track.path distance since hand-over), plus settle(events, failed) per decision, which
    returns (signed term sum, events). Event dicts: unit, point ('entry' | 'exit'), v, v_qss, rho,
    g, bonus (term paid; negative for a take-back), kind ('paid' | 'unpaid' (g = 0) | 'forfeit' |
    'taken_back'). Episode counters: taken_back (sum, >= 0), n_taken_back."""

    def __init__(self, table, bonus_in, bonus_out, sigma):
        self.table = table
        self.B_in, self.B_out, self.sigma = float(bonus_in), float(bonus_out), float(sigma)
        self._in = CornerTracker(_entry_view(table), 1., 0.)
        self._out = CornerTracker(table, 1., 0.)
        self.pending = np.zeros(len(table.units))       # entry term paid in this pass, exit not yet crossed
        self.reset(0.)

    def reset(self, s_handover):
        self._in.reset(s_handover)
        self._out.reset(s_handover)
        self.pending[:] = 0.
        self.taken_back, self.n_taken_back = 0., 0

    def _events(self, raw, point, B):
        out = []
        for e in raw:
            g = closeness_rho(e['rho'], self.sigma)
            out.append({'unit': e['unit'], 'point': point, 'v': e['v_exit'], 'v_qss': e['v_qss_exit'],
                        'rho': e['rho'], 'g': g, 'bonus': B * g})
        return out

    def tick(self, d0, d1, v0, v1):
        """Crossings inside one tick: entries first, then exits (a unit's entry and exit are
        >= 100 m apart, so both in one tick only after a projection jump)."""
        return (self._events(self._in.tick(d0, d1, v0, v1), 'entry', self.B_in)
                + self._events(self._out.tick(d0, d1, v0, v1), 'exit', self.B_out))

    def settle(self, events, failed):
        """Decision-level rule. Not failed: entries are paid (and become pending), exits are paid
        and clear their unit's pending entry. Failed (terminal failure): every crossing of the
        decision is forfeited (corners.settle_events) and every pending entry is taken back."""
        total, out = settle_events(events, failed)
        for e in out:
            if e['kind'] == 'exit_unpaid':
                e['kind'] = 'unpaid'
        if not failed:
            for e in out:
                if e['point'] == 'entry':
                    self.pending[e['unit']] += e['bonus']
                else:
                    self.pending[e['unit']] = 0.
            return total, out
        for k in np.flatnonzero(self.pending != 0.):
            amount = float(self.pending[k])
            total -= amount
            self.taken_back += amount
            self.n_taken_back += 1
            out.append({'unit': int(k), 'point': 'entry', 'bonus': -amount, 'kind': 'taken_back'})
        self.pending[:] = 0.
        return total, out
