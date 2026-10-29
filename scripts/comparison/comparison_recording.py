"""Read-only 50 Hz recorder shared by offline RL and pure-pursuit comparisons.

Uses DrivingEnv's genuine reset/settle, per-physics-step footprint stop, and interpolated
lap clock. No torch import and no simulator changes.
"""
import numpy as np
from ocd_lap.env import OCDRacingEnv
from rlsetup.driving.env import DrivingEnv
import ocd_trace


class _TickHook(OCDRacingEnv):
    def step(self, action):
        out = super().step(action)
        self._on_tick(action, out[4])
        return out


class RecordingEnv(DrivingEnv, _TickHook):
    """MRO interposes only a read-only hook after each genuine 50 Hz base tick."""

    def __init__(self, config, **kw):
        super().__init__(config, **kw)
        self.geo = ocd_trace.Geometry(self.track)

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        self.rows, self.decision, self.command_context = [], 0, {}
        self.geo.reset(s_path=self.s)
        self.clock = ocd_trace.LapClock(self.track, self.s, len(self._lines))
        self._rec_dist, self._rec_s, self._t_prev = 0., self.s, 0.
        self.handover = {'state': dict(self.state), 's_path_m': self.s, 's_ref_m': self._s_ref,
                         'settle_integral_m': self.settle_driver.integral, 'start': info['start']}
        return obs, info

    def _on_tick(self, action, base):
        a = np.asarray(action, dtype=float)
        row = dict(t_s=self.time_s, decision=self.decision, decision_end=0,
                   a_steer=a[0], a_throttle=a[1], a_brake=a[2], **self.command_context)
        if not base['nonfinite']:
            st, length = self.state, float(self.track.length)
            geometry = self.geo.locate(st['x'], st['y'], st['yaw'], s_path=self.s)
            d0 = self._rec_dist
            self._rec_dist += (self.s - self._rec_s + length / 2) % length - length / 2
            self._rec_s = self.s
            self.clock.update(d0, self._rec_dist, self._t_prev, self.time_s)
            row.update(lap=self.clock.lap, t_lap_s=self.clock.t_lap(self.time_s), progress_m=self._rec_dist,
                       **geometry, **ocd_trace.backend_sample(self.backend, st, self.to_inputs(a)),
                       on_track=int(not base['off_track']))
        self.rows.append(row)
        self._t_prev = self.time_s

    def step(self, action):
        self._t_prev = self.time_s
        obs, reward, term, trunc, info = super().step(action)
        costs = info['step_cost']
        self.rows[-1].update(decision_end=1, reward=reward, r_progress=costs['progress'], r_time=costs['time'],
                             r_smoothness=costs['smoothness'], r_tracking=costs['tracking'],
                             r_penalty=costs['penalty'], r_shaping=info['shaping'], e_ref_env_m=info['e_ref'],
                             termination=info['termination'] or '', truncation=info['truncation'] or '')
        self.decision += 1
        return obs, reward, term, trunc, info


def check(env, report):
    """Compare the recorder to the configured reference, including raceline-based p0 runs."""
    times = report.lap_times_s
    lap_ok = len(env.clock.lap_times) == len(times) and np.allclose(
        env.clock.lap_times, times, rtol=0., atol=1e-9)
    column = {'path': 'e_path_m', 'raceline': 'e_raceline_m'}[env.cfg.reference]
    decisions = [row for row in env.rows if row['decision_end'] and column in row]
    deviation = max((abs(row[column] - row['e_ref_env_m']) for row in decisions), default=0.)
    return {'lap_times_match_env': bool(lap_ok), 'configured_reference': env.cfg.reference,
            'max_abs_configured_ref_minus_env_e_ref_m': float(deviation),
            'recorder_crossings_s': list(env.clock.crossings)}
