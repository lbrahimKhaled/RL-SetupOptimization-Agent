"""One OCD sample row (common.OCD_COLUMNS) from an `OCDBackend`, for the pure-pursuit and RL exports.

Read-only: it calls `backend.car.signals()` / `backend.get_telemetry()` (OCD debug-log reads) and
pure helpers of `ocd_lap.powertrain`, never anything that steps or syncs the car, so recording
does not change the trajectory (checked: the RL lap times reproduce the training evaluation).

Line geometry uses the repo's own helpers with their own match hints:
  s_path, progress   Track.match on Track.path (as ocd_lap.env.OCDRacingEnv)
  e_path             ReferenceLine.from_track_path(track) (as DrivingEnv._line_errors with the
                     default xi_ref = Track.path)
  s_raceline, e_raceline, heading_err_raceline   ReferenceLine(Track.reference)
  track_position     same formula as DrivingEnv.track_position on the TrackGrid of Track.path
"""
import math
import numpy as np
from ocd_lap.backend import WHEELS
from rlsetup.driving.reference import ReferenceLine, track_grid

_LONG = dict(FL='front_left', FR='front_right', RL='rear_left', RR='rear_right')
SIGNALS = {'steer_cmd_rad': 'steering_actuator/target_steering_angle_rad',
           'drive_torque_axle_Nm': 'drivetrain/input/transmission_output_torque_Nm',
           **{f'brake_torque_{w}_Nm': f'drivetrain/input/brake_torque_per_wheel_Nm/{_LONG[w]}' for w in WHEELS},
           'dvx': 'vehicle_dynamics/x_dot_vec/v_x_mps', 'dvy': 'vehicle_dynamics/x_dot_vec/v_y_mps'}


def raceline_of(track):
    """TUM's raceline (Track.reference, the racetrack-database file TUM's simulator also reads)."""
    return ReferenceLine(track.reference, f'{track.name}:Track.reference')


class Geometry:
    """Track-relative quantities of a pose; keeps its own match hints (call `reset` per run)."""

    def __init__(self, track):
        self.track = track
        self.raceline = raceline_of(track)
        self.path_ref = ReferenceLine.from_track_path(track)
        self.grid = track_grid(track, self.path_ref, 1.)
        self.reset()

    def reset(self, s_path=None, s_raceline=None, s_path_ref=None):
        self.s_path, self.s_raceline, self.s_path_ref = s_path, s_raceline, s_path_ref

    def locate(self, x, y, yaw, s_path=None):
        """dict of s_path_m, s_raceline_m, e_raceline_m, heading_err_raceline_rad, e_path_m,
        track_position. `s_path`: use this Track.path arc length (e.g. the env's own) instead
        of matching."""
        if s_path is None:
            s_path = float(self.track.match([x, y], self.s_path)[0])
        self.s_path = s_path
        self.s_raceline = self.raceline.match((x, y), self.s_raceline)
        self.s_path_ref = self.path_ref.match((x, y), self.s_path_ref)
        e_r, h_r = self.raceline.errors_at(self.s_raceline, x, y, yaw)
        e_p, _ = self.path_ref.errors_at(self.s_path_ref, x, y, yaw)
        _, left, right = self.grid.at(s_path)
        d = left - right
        p = np.array([x, y]) - right
        return dict(s_path_m=s_path, s_raceline_m=self.s_raceline, e_raceline_m=e_r,
                    heading_err_raceline_rad=h_r, e_path_m=e_p,
                    track_position=float(2. * (p @ d) / (d @ d) - 1.))


def backend_sample(backend, state, inputs):
    """Vehicle columns of OCD_COLUMNS at the current backend state; `inputs` = the pedals held
    over the tick that ended here."""
    sig = dict(zip(SIGNALS, backend.car.signals(list(SIGNALS.values()))))
    tel = backend.get_telemetry()
    vx, vy, r = state['vx'], state['vy'], state['yaw_rate']
    speed = math.hypot(vx, vy)
    pt = backend.powertrain
    rpm = pt.rpm(backend.car.rear_axle_omega())
    row = dict(x_m=state['x'], y_m=state['y'], yaw_rad=state['yaw'], speed_mps=speed, vx_mps=vx, vy_mps=vy,
               yaw_rate_radps=r, ax_mps2=sig['dvx'] - r * vy, ay_mps2=sig['dvy'] + r * vx,
               steer_cmd_rad=sig['steer_cmd_rad'], steer_actual_rad=state['steer_actual'],
               throttle=inputs.throttle, brake=inputs.brake, drive_torque_axle_Nm=sig['drive_torque_axle_Nm'],
               gear=backend.gear, engine_rpm=rpm,
               engine_torque_crank_Nm=inputs.throttle * pt.engine_torque_Nm(rpm),
               e_motor_torque_crank_Nm=inputs.throttle * pt.e_motor_torque_Nm(rpm, speed),
               ers_energy_J=backend.ers_energy_J, slip_rl=state['slip_rl'], slip_rr=state['slip_rr'])
    for w in WHEELS:
        row[f'brake_torque_{w}_Nm'] = sig[f'brake_torque_{w}_Nm']
        row[f'kappa_{w}'] = tel[f'kappa_{w}']
        row[f'alpha_{w}_rad'] = tel[f'alpha_{w}_rad']
        row[f'Fz_{w}_N'] = tel[f'Fz_{w}_N']
    return row


class LapClock:
    """Line crossings on Track.path progress, interpolated between ticks (as DrivingEnv /
    ocd_lap.simulation.drive): lines at d_line + k L, d_line = distance from the start to s = 0."""

    def __init__(self, track, s_start, n_lines):
        L = float(track.length)
        d_line = float((L - s_start) % L)
        self.lines = [d_line + k * L for k in range(n_lines)]
        self.crossings = [0.] if d_line == 0. else []

    def update(self, d0, d1, t0, t1):
        """Record a crossing between progress d0 -> d1 (times t0 -> t1); True if one happened."""
        j = len(self.crossings)
        if j < len(self.lines) and d0 < self.lines[j] <= d1:
            self.crossings.append(float(t0 + (self.lines[j] - d0) / max(d1 - d0, 1e-12) * (t1 - t0)))
            return True
        return False

    @property
    def lap(self):
        return len(self.crossings)

    def t_lap(self, t):
        return t - self.crossings[-1] if self.crossings else t

    @property
    def lap_times(self):
        return [b - a for a, b in zip(self.crossings, self.crossings[1:])]
