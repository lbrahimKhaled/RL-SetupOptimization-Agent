"""Thin AC collector: explicit geometric guidance, frozen actor, bounded UDP control.

Coordinates are (AC z, AC x), x forward/y left in the body frame. All times
are seconds, distances metres and angles radians. No game reset or gear command.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time

import numpy as np

from .actor import ActorRuntime, SOURCE_OBS, _json, source_contract_sha256
from .management import ConnectionConfig, ManagementClient
from .telemetry import CaptureConfig, _numeric, _open_socket, _packet

RELEASE_ACTION = np.array([0., -.8, -.8], dtype=np.float32)
GAMMA = .992
SOURCE_CONTRACT_HASH = hashlib.sha256(json.dumps(asdict(SOURCE_OBS), sort_keys=True).encode()).hexdigest()


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def number(packet, key):
    if key not in packet or not _numeric(packet[key]):
        raise ValueError(f'missing/nonfinite telemetry: {key}')
    return float(packet[key])


REFERENCES = ('centerline', 'fast_lane', 'ocd_path')
OCD_REFERENCE_SCHEMA = 'rlsetup.ac.ocd-reference.v1'
# v2 (scripts/ac_ocd_reference.py --source-run): the source run's own xi_ref (Track.path or a CSV
# line) plus, per AC row, the OCD Track.path station, the R4 corner-table speed and corner-window
# flag, and the source env's settle/stall settings; bound to the source config by SHA-256.
OCD_REFERENCE_SCHEMA_V2 = 'rlsetup.ac.ocd-reference.v2'
SETTLE_KEYS = ('rear_axle_m', 'wheelbase_m', 'steer_lock_rad', 'lookahead_m', 'lookahead_s', 'speed_kp',
               'speed_ki', 'integral_limit_m', 'brake_cap', 'traction_control', 'traction_slip_start',
               'traction_slip_span', 'released_pedal_action', 'settle_s', 'start_speed_fraction',
               'eval_speed_fraction', 'settle_window_margin_m', 'stall_speed_mps', 'stall_time_s',
               'wrong_way_rad')


def ac_borders_sha256(left, right):
    """Identity of the exact AC border rows (float64, row order) a reference file is bound to."""
    return hashlib.sha256(np.ascontiguousarray(
        np.stack([np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)])).tobytes()).hexdigest()


def load_reference(track, reference, reference_file=None):
    """Return (track dict usable by Geometry, reference file sha256 or None).

    'ocd_path' rows come from scripts/ac_ocd_reference.py and must be bound to these exact
    AC borders; the other references come from the live AC track data and take no file.
    """
    if reference not in REFERENCES:
        raise ValueError('explicit reference=centerline, fast_lane or ocd_path required')
    if reference != 'ocd_path':
        if reference_file is not None:
            raise ValueError('a reference file is only used with reference=ocd_path')
        return track, None
    if reference_file is None:
        raise ValueError('reference=ocd_path requires --reference-file')
    data = Path(reference_file).read_bytes()
    saved = _json(data)
    rows = np.asarray(saved.get('rows'), dtype=float)
    if (saved.get('schema') not in (OCD_REFERENCE_SCHEMA, OCD_REFERENCE_SCHEMA_V2)
            or saved.get('reference') != 'ocd_path'
            or rows.ndim != 2 or rows.shape != np.asarray(track['left_lane'], dtype=float).shape):
        raise ValueError('invalid OCD reference file for this AC track')
    if saved.get('ac_borders_sha256') != ac_borders_sha256(track['left_lane'], track['right_lane']):
        raise ValueError('OCD reference file was made for different AC track borders')
    result = dict(track, ocd_path=rows.tolist())
    if saved['schema'] == OCD_REFERENCE_SCHEMA_V2:
        result['ocd_profile'] = _profile_fields(saved, len(rows))
    return result, hashlib.sha256(data).hexdigest()


def _profile_fields(saved, n):
    """Validated v2 per-row profile, settle settings and source binding (JSON numbers/strings)."""
    profile, settle, source = saved.get('profile'), saved.get('settle'), saved.get('source')
    if not all(isinstance(x, dict) for x in (profile, settle, source)):
        raise ValueError('v2 reference needs profile, settle and source objects')
    arrays = {}
    for key in ('s_path_m', 'v_qss_mps', 'v_start_mps', 'corner'):
        value = np.asarray(profile.get(key), dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError(f'v2 reference profile.{key} must have one finite value per row')
        arrays[key] = value
    if (np.any(np.diff(arrays['s_path_m']) <= 0) or np.any(arrays['v_qss_mps'] <= 0)
            or np.any(arrays['v_start_mps'] <= 0) or not np.isin(arrays['corner'], (0., 1.)).all()):
        raise ValueError('v2 reference profile out of range (increasing s_path, speeds > 0, corner 0/1)')
    length = profile.get('path_length_m')
    # A closing duplicate row (first = last point) sits exactly one lap on.
    if not _numeric(length) or length <= 0 or arrays['s_path_m'][-1]-arrays['s_path_m'][0] > length+1e-6:
        raise ValueError('v2 reference profile needs the OCD Track.path length')
    if set(settle) != set(SETTLE_KEYS) or not all(
            (isinstance(settle[k], bool) if k == 'traction_control' else
             isinstance(settle[k], list) and len(settle[k]) == 2 and all(_numeric(x) for x in settle[k])
             if k == 'start_speed_fraction' else _numeric(settle[k])) for k in SETTLE_KEYS):
        raise ValueError('v2 reference settle settings incomplete or not numeric')
    digest = source.get('source_config_sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('v2 reference must name its source config SHA-256')
    return dict(arrays, path_length_m=float(length), settle=dict(settle), source_config_sha256=digest)


class Profile:
    """OCD quantities at an AC reference station (v2 reference rows; linear, wrapping).

    s_path(station): OCD Track.path arc length [m] (unwrapped by row); progress for the reward uses
    its wrapped difference, as OCD's ds. v_qss(station): R4 corner-table speed [m/s] (QSS on xi_ref).
    v_start_min(station, distance): minimum of the env's QSS profile on Track.path, which sets
    OCD's start speed. corner(station): 1 inside a corner window (flag of the row segment).
    """
    def __init__(self, fields, geometry):
        n = len(geometry.ref)  # Geometry drops a closing duplicate row
        self.geometry, self.length = geometry, fields['path_length_m']
        self.s_path_rows = fields['s_path_m'][:n]
        self.v_rows, self.corner_rows = fields['v_qss_mps'][:n], fields['corner'][:n]
        self.v_start_rows = fields['v_start_mps'][:n]
        self.settle = fields['settle']
        self._x = np.r_[geometry.stations[:-1], geometry.length]
        self._s = np.r_[self.s_path_rows, self.s_path_rows[0]+self.length]

    def s_path(self, station):
        return float(np.interp(float(station) % self.geometry.length, self._x, self._s) % self.length)

    def path_progress(self, old_station, new_station):
        d = self.s_path(new_station)-self.s_path(old_station)
        return float((d+self.length/2) % self.length-self.length/2)

    def v_qss(self, station):
        return float(np.interp(np.asarray(station, dtype=float) % self.geometry.length, self._x,
                               np.r_[self.v_rows, self.v_rows[0]]).min())

    def v_start(self, station):
        return float(np.interp(np.asarray(station, dtype=float) % self.geometry.length, self._x,
                               np.r_[self.v_start_rows, self.v_start_rows[0]]).min())

    def v_start_min(self, start, distance):
        """Minimum start-QSS speed over [start, start + distance] stations (1 m spacing)."""
        return self.v_start(start+np.arange(0., max(distance, 0.)+1., 1.))

    def corner(self, station):
        station = float(station) % self.geometry.length
        i = int(np.searchsorted(self.geometry.stations, station, side='right')-1)
        return bool(self.corner_rows[min(i, len(self.corner_rows)-1)])


class Geometry:
    """Closed corresponding borders and midpoint, AC AI fast-lane or OCD training-line reference.

    Labels are validated against forward tangents, never automatically reversed.
    Four oriented body corners are checked against local border cross-sections.
    """
    def __init__(self, track, *, reference):
        if reference not in REFERENCES:
            raise ValueError('explicit reference=centerline, fast_lane or ocd_path required')
        self.left = np.asarray(track['left_lane'], dtype=float)
        self.right = np.asarray(track['right_lane'], dtype=float)
        if (self.left.ndim != 2 or self.left.shape[1] != 2 or len(self.left) < 4
                or self.right.shape != self.left.shape
                or not np.isfinite(self.left).all() or not np.isfinite(self.right).all()):
            raise ValueError('finite corresponding Nx2 borders required')
        if reference == 'centerline':
            if np.linalg.norm((self.left[0]+self.right[0]-self.left[-1]-self.right[-1])/2) < 1e-6:
                self.left, self.right = self.left[:-1], self.right[:-1]
            self.ref = (self.left + self.right) / 2
        else:
            self.ref = np.asarray(track.get(reference), dtype=float)
            if self.ref.shape != self.left.shape or not np.isfinite(self.ref).all():
                raise ValueError(f'finite {reference} Nx2 matching border rows required')
            if np.linalg.norm(self.ref[0]-self.ref[-1]) < 1e-6:
                if (np.linalg.norm(self.left[0]-self.left[-1]) >= 1e-6
                        or np.linalg.norm(self.right[0]-self.right[-1]) >= 1e-6):
                    raise ValueError(f'{reference} closure must match corresponding borders')
                self.ref, self.left, self.right = self.ref[:-1], self.left[:-1], self.right[:-1]
        self.delta = np.roll(self.ref, -1, axis=0) - self.ref
        self.lengths = np.linalg.norm(self.delta, axis=1)
        width = np.linalg.norm(self.left-self.right, axis=1)
        cross = self.delta[:, 0]*(self.left-self.right)[:, 1] - self.delta[:, 1]*(self.left-self.right)[:, 0]
        if np.any(self.lengths <= 1e-6) or np.any(width <= 0) or np.any(cross <= 0):
            raise ValueError('degenerate geometry or incorrectly labelled left/right borders')
        if reference != 'centerline':
            left, right = self.left-self.ref, self.right-self.ref
            if (np.any(self.delta[:, 0]*left[:, 1]-self.delta[:, 1]*left[:, 0] <= 0)
                    or np.any(self.delta[:, 0]*right[:, 1]-self.delta[:, 1]*right[:, 0] >= 0)):
                raise ValueError(f'{reference} must lie between corresponding left/right borders')
        self.stations = np.r_[0., np.cumsum(self.lengths)]
        self.length = float(self.stations[-1])
        self.preview = SOURCE_OBS.preview_horizon_m * (np.arange(1, 21)/20)**SOURCE_OBS.preview_exponent

    def at(self, station):
        station = np.asarray(station) % self.length
        i = np.searchsorted(self.stations, station, side='right') - 1
        t = (station-self.stations[i])/self.lengths[i]
        def interp(points):
            return points[i] + t[..., None]*(points[(i+1) % len(points)]-points[i])
        return interp(self.ref), interp(self.left), interp(self.right)

    def project(self, position):
        offset = np.asarray(position)-self.ref
        t = np.clip(np.sum(offset*self.delta, axis=1)/self.lengths**2, 0, 1)
        nearest = self.ref+t[:, None]*self.delta
        i = int(np.argmin(np.sum((nearest-position)**2, axis=1)))
        tangent = self.delta[i]/self.lengths[i]
        e = float(tangent[0]*(position[1]-nearest[i, 1])-tangent[1]*(position[0]-nearest[i, 0]))
        return float((self.stations[i]+t[i]*self.lengths[i]) % self.length), e, math.atan2(tangent[1], tangent[0])

    def track_position(self, position, station):
        _, left, right = self.at(station)
        across = left-right
        return float(2*np.dot(position-(left+right)/2, across)/np.dot(across, across))

    def inside(self, position, yaw, length, width):
        c, s = math.cos(yaw), math.sin(yaw)
        for x in (-length/2, length/2):
            for y in (-width/2, width/2):
                corner = position + np.array([c*x-s*y, s*x+c*y])
                station, _, _ = self.project(corner)
                if abs(self.track_position(corner, station)) > 1:
                    return False
        return True

    def progress(self, old, new):
        return float((new-old+self.length/2) % self.length-self.length/2)


class SteeringMap:
    """Measured road-wheel angle -> vJoy command; no extrapolation."""
    def __init__(self, data):
        required = ('angle_rad', 'command', 'baseline_angle_rad', 'car_name', 'track_name',
                    'track_configuration', 'setup_id', 'sampling_hz')
        if data.get('schema_version') != 1 or data.get('validated') is not True or any(k not in data for k in required):
            raise ValueError('validated schema_version=1 steering calibration required')
        self.data = data
        self.angles, self.commands = np.asarray(data['angle_rad'], float), np.asarray(data['command'], float)
        if (self.angles.ndim != 1 or len(self.angles) < 3 or self.commands.shape != self.angles.shape
                or not np.isfinite(self.angles).all() or not np.isfinite(self.commands).all()
                or np.any(np.diff(self.angles) <= 0) or np.any(abs(self.commands) > 1)
                or not (np.all(np.diff(self.commands) > 0) or np.all(np.diff(self.commands) < 0))
                or not self.angles[0] <= 0 <= self.angles[-1]
                or not _numeric(data['baseline_angle_rad'])):
            raise ValueError('invalid measured steering domain')
        self.baseline = float(data['baseline_angle_rad'])

    def verify(self, static, config, setup_id):
        for key, source in (('car_name', 'CarName'), ('track_name', 'TrackName'), ('track_configuration', 'TrackConfiguration')):
            if self.data[key] != static.get(source):
                raise ValueError(f'calibration identity mismatch: {key}')
        if self.data['setup_id'] != setup_id or self.data['sampling_hz'] != config.get('ego_sampling_freq'):
            raise ValueError('calibration setup/sampling mismatch')
        if config.get('vjoy_executed_by_server') is not True:
            raise ValueError('server-side vJoy controls required')

    def command(self, angle, *, saturation='reject'):
        if saturation not in ('clip', 'reject') or not math.isfinite(angle):
            raise ValueError('invalid steering request/saturation mode')
        if saturation == 'reject' and not self.angles[0] <= angle <= self.angles[-1]:
            raise ValueError('desired steering outside measured calibration domain')
        return float(np.interp(np.clip(angle, self.angles[0], self.angles[-1]), self.angles, self.commands))


def road_angle(packet, baseline):
    """Front-minus-rear circular mean heading, positive left, minus neutral baseline."""
    def axle(names):
        values = []
        for name in names:
            x, z = number(packet, f'tyreContactHeading_{name}_x'), number(packet, f'tyreContactHeading_{name}_z')
            if math.hypot(x, z) < 1e-6:
                raise ValueError('degenerate tyre heading')
            values.append(math.atan2(x, z))
        sx, cx = sum(math.sin(x) for x in values), sum(math.cos(x) for x in values)
        if math.hypot(sx, cx) < 1e-6:
            raise ValueError('ambiguous tyre heading mean')
        return math.atan2(sx, cx)
    return float(wrap(axle(('FL', 'FR'))-axle(('RL', 'RR'))-baseline))


class Observation:
    """Source 159 columns; previous blocks newest first, from real samples only."""
    def __init__(self, geometry, baseline):
        self.geometry, self.baseline = geometry, baseline
        self.history = deque(maxlen=2)
        self.previous = None
        self.previous_kin = None

    def build(self, packet, last_action):
        o = SOURCE_OBS
        position = np.array([number(packet, 'world_position_x'), number(packet, 'world_position_y')])
        yaw, stamp = number(packet, 'yaw'), number(packet, 'timestamp_ac')
        c, s = math.cos(yaw), math.sin(yaw)
        velocity = np.array([number(packet, 'velocity_z'), number(packet, 'velocity_x')])
        vx, vy = c*velocity[0]+s*velocity[1], -s*velocity[0]+c*velocity[1]
        if self.previous is None:
            self.previous = stamp, yaw
            return None
        dt = stamp-self.previous[0]
        if dt <= 0:
            raise ValueError('nonadvancing observation timestamp')
        rate = float(wrap(yaw-self.previous[1])/dt)
        self.previous = stamp, yaw
        kin = np.array([vx, vy, rate])
        if self.previous_kin is None:
            self.previous_kin = kin
            return None
        midpoint = (kin+self.previous_kin)/2
        ax, ay = (kin[:2]-self.previous_kin[:2])/dt + np.array([-midpoint[2]*midpoint[1], midpoint[2]*midpoint[0]])
        self.previous_kin = kin
        ego = np.array([math.hypot(vx, vy)/o.speed_scale_mps, vx/o.speed_scale_mps,
                        vy/o.vy_scale_mps, rate/o.yaw_rate_scale_radps,
                        ax/o.accel_scale_mps2, ay/o.accel_scale_mps2,
                        road_angle(packet, self.baseline)/o.steer_scale_rad,
                        np.clip(o.slip_gain*number(packet, 'tyre_slip_ratio_rl'), -o.slip_clip, o.slip_clip),
                        np.clip(o.slip_gain*number(packet, 'tyre_slip_ratio_rr'), -o.slip_clip, o.slip_clip)])
        station, e, heading = self.geometry.project(position)
        ref, left, right = self.geometry.at(station+self.geometry.preview)
        pts = np.stack([ref, left, right], axis=1)-position
        local = np.stack([c*pts[..., 0]+s*pts[..., 1], -s*pts[..., 0]+c*pts[..., 1]], axis=-1)
        preview = (local/np.maximum(self.geometry.preview, o.preview_min_scale_m)[:, None, None]).ravel()
        result = None
        if len(self.history) == 2:
            raw = np.concatenate([ego, [e/o.e_scale_m, wrap(yaw-heading)/o.heading_scale_rad,
                                      self.geometry.track_position(position, station)], preview, last_action, *self.history])
            if raw.shape != (159,) or not np.isfinite(raw).all():
                raise ValueError('invalid source observation')
            result = np.clip(raw, -o.clip, o.clip).astype(np.float32)
        self.history.appendleft(np.r_[ego, last_action])
        return result


def reward(progress, dt, old_e, new_e, *, failure=False):
    """R1 per decision: metres - seconds + PBRS; failure P=100 per approved R1."""
    phi_next = 0. if failure else -abs(new_e)
    return float(progress-dt+GAMMA*phi_next+abs(old_e)-(100. if failure else 0.))


@dataclass(frozen=True)
class RewardTerms:
    """The source env.reward terms added to R1 (rlsetup/driving/config.py RewardConfig, time_pbrs):

    speed potential  F_v = gamma Phi_v(s') - Phi_v(s), Phi_v = speed_k g(v, v_qss(s)),
                     g = max(0, 1 - ((v - v*) / (speed_sigma v*))^2); Phi_v(s') = 0 after a terminal failure.
    line penalty     min(line_k max(0, |e| - line_free_m), line_cap) max(0, ds) when the decision ends in a
                     corner window (OCD applies it per 50 Hz tick; here once per 25 Hz decision).
    All zero (R1 only) for sources trained without them.
    """
    speed_k: float = 0.
    speed_sigma: float = .3
    line_k: float = 0.
    line_free_m: float = .5
    line_cap: float = .5

    @classmethod
    def from_source(cls, config):
        rw = config.get('env', {}).get('reward')
        if not isinstance(rw, dict):
            raise ValueError('source config has no env.reward')
        # The AC R1 above hard-codes these source values; any other source reward is unsupported.
        for key, value in dict(mode='time_pbrs', a=1., b=1., shaping_k=1., penalty_const=100., gamma=GAMMA).items():
            if rw.get(key) != value:
                raise ValueError(f'unsupported source env.reward.{key}={rw.get(key)!r} (AC implements {value!r})')
        values = {name: rw.get(name, getattr(cls, name)) for name in
                  ('speed_k', 'speed_sigma', 'line_k', 'line_free_m', 'line_cap')}
        if any(not _numeric(v) or v < 0 for v in values.values()) or values['speed_sigma'] <= 0:
            raise ValueError('source reward terms must be finite and nonnegative')
        return cls(**{k: float(v) for k, v in values.items()})

    @property
    def active(self):
        return self.speed_k > 0 or self.line_k > 0

    def phi_v(self, speed, v_qss):
        if not self.speed_k:
            return 0.
        x = (speed-v_qss)/(self.speed_sigma*v_qss)
        return self.speed_k*max(0., 1.-x*x)

    def line_penalty(self, abs_e, ds, in_corner):
        if not self.line_k or not in_corner or ds <= 0:
            return 0.
        return min(self.line_k*max(0., abs_e-self.line_free_m), self.line_cap)*ds


def controls(action, steering, seq, *, saturation='reject', steer_lock_rad=.6):
    """AC command for a source action; steering action +-1 = +-steer_lock_rad road-wheel angle
    (the source car's lock: 0.6 rad sf70h_tum, 0.30 rad sf70h_f1; ActorRuntime.steer_lock_rad)."""
    action = np.asarray(action)
    if action.shape != (3,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
        raise ValueError('finite normalized action[3] required')
    if not _numeric(steer_lock_rad) or not 0 < steer_lock_rad <= 1:
        raise ValueError('steer_lock_rad must be in (0, 1] rad')
    pedals = np.clip((action[1:]+.5)/1.5, 0, 1)
    return dict(steer=steering.command(steer_lock_rad*float(action[0]), saturation=saturation), acc=float(2*pedals[0]-1),
                brake=float(2*pedals[1]-1), server_steps=int(seq),
                enable_gear_shift=False, shift_up=False, shift_down=False)


def body_vx(packet):
    """Forward speed in the car frame [m/s] (same formula as Observation.build)."""
    yaw = number(packet, 'yaw')
    return math.cos(yaw)*number(packet, 'velocity_z')+math.sin(yaw)*number(packet, 'velocity_x')


class SettleDriver:
    """OCD's settle / demo controller (rlsetup/driving/demo_driver.py DemoDriver) on the AC reference.

    Pure pursuit from the rear axle to the reference point lookahead_m + lookahead_s v_x ahead,
    steering atan(2 L sin(alpha) / d) within the lock; speed PI on the pedals (speed_kp per m/s,
    speed_ki per m, integral clipped), brake pedal capped at brake_cap, throttle scaled down by
    rear-slip traction control. Returns source actions (pedal_deadzone inverse, released pedal
    action when a pedal is 0), so the policy's history at hand-over holds settle actions as in OCD.
    """
    def __init__(self, settle, geometry):
        self.c, self.geometry = settle, geometry
        self.integral = 0.

    def act(self, packet, station, target_mps, dt):
        c = self.c
        position = np.array([number(packet, 'world_position_x'), number(packet, 'world_position_y')])
        yaw, vx = number(packet, 'yaw'), body_vx(packet)
        rear = position-c['rear_axle_m']*np.array([math.cos(yaw), math.sin(yaw)])
        goal, _, _ = self.geometry.at(station-c['rear_axle_m']+c['lookahead_m']+c['lookahead_s']*max(vx, 0.))
        d = goal-rear
        alpha = math.atan2(d[1], d[0])-yaw
        lock = c['steer_lock_rad']
        steer = float(np.clip(math.atan2(2.*c['wheelbase_m']*math.sin(alpha), np.linalg.norm(d)), -lock, lock))
        error = target_mps-vx
        self.integral = float(np.clip(self.integral+error*dt, -c['integral_limit_m'], c['integral_limit_m']))
        u = c['speed_kp']*error+c['speed_ki']*self.integral
        throttle = min(1., max(0., u))
        slip = max(abs(number(packet, 'tyre_slip_ratio_rl')), abs(number(packet, 'tyre_slip_ratio_rr')))
        if c['traction_control'] and slip > c['traction_slip_start']:
            throttle *= max(0., 1.-(slip-c['traction_slip_start'])/c['traction_slip_span'])
        brake = min(c['brake_cap'], max(0., -u))
        pedal = lambda p: -.5+1.5*p if p > 0. else c['released_pedal_action']
        return np.array([np.clip(steer/lock, -1., 1.), pedal(throttle), pedal(brake)], dtype=np.float32)


@dataclass(frozen=True)
class CollectConfig:
    host: str
    max_steps: int
    steering_saturation: str = 'reject'
    max_dt: float = .08
    min_dt: float = .02
    max_wall_s: float = 180.
    timeout_s: float = 3.
    port: int = 2345
    management_port: int = 2347
    setup_id: str = 'default_user_confirmed'
    final_brake: bool = False
    exploration_std_scale: float = 0.  # 0 = deterministic policy mean
    exploration_seed: int = 0
    runup_max_s: float = 30.  # v2 reference: settle run-up must reach hand-over within this

    def __post_init__(self):
        if type(self.final_brake) is not bool:
            raise ValueError('final_brake must be a boolean')
        if (isinstance(self.exploration_std_scale, bool)
                or not isinstance(self.exploration_std_scale, (int, float))
                or not 0 <= self.exploration_std_scale <= 1
                or type(self.exploration_seed) is not int or self.exploration_seed < 0):
            raise ValueError('exploration_std_scale must be in [0, 1] and seed a nonnegative int')
        CaptureConfig(self.host, port=self.port)
        if type(self.max_steps) is not int or not 0 < self.max_steps <= 100000:
            raise ValueError('max_steps must be in 1..100000')
        for value in (self.min_dt, self.max_dt, self.max_wall_s, self.timeout_s, self.runup_max_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('dimensions/timing must be finite positive')
        if self.steering_saturation not in ('clip', 'reject'):
            raise ValueError('steering_saturation must be clip or reject')
        if self.min_dt > self.max_dt or self.max_wall_s > 3600:
            raise ValueError('invalid timing bounds')


class Session:
    """Fresh contiguous UDP telemetry and counter echo; releases before disconnect."""
    def __init__(self, config, steering, steer_lock_rad=.6):
        self.config, self.steering = config, steering
        self.steer_lock_rad = steer_lock_rad  # source car lock; parked/hold commands only steer 0
        self.sock = None
        self.last = None
        self.received = None

    def __enter__(self):
        deadline = time.monotonic()+self.config.timeout_s
        self.sock = _open_socket(CaptureConfig(self.config.host, port=self.config.port), deadline)
        try:
            self.sock.send(b'connect')
            self.sock.settimeout(max(.001, deadline-time.monotonic()))
            reply = self.sock.recv(65508)
            if time.monotonic() > deadline:
                raise TimeoutError('handshake deadline exceeded')
            if reply != b'identified':
                raise ValueError('expected identified handshake')
        except BaseException:
            self.close()
            raise
        return self

    def receive(self, deadline):
        # The server initializes after identified; active freshness starts at the first sample.
        wait_s = self.config.timeout_s if self.last is None else self.config.max_dt
        self.sock.settimeout(max(.001, min(deadline-time.monotonic(), wait_s)))
        if time.monotonic() >= deadline:
            raise TimeoutError('wall limit')
        payload = self.sock.recv(65508)
        # Select the newest already queued sample, with a strict work bound. A
        # skipped sequence after handover is rejected below, never replayed across.
        self.sock.settimeout(0.)
        for _ in range(256):
            try:
                payload = self.sock.recv(65508)
            except (BlockingIOError, socket.timeout):
                break
        else:
            raise ValueError('UDP backlog exceeds drain bound')
        if time.monotonic() > deadline:
            raise TimeoutError('wall limit')
        if len(payload) > 65507:
            raise ValueError('oversized UDP telemetry')
        data = _packet(payload)
        seq, stamp = data.get('steps'), number(data, 'timestamp_ac')
        if type(seq) is not int or seq < 0:
            raise ValueError('invalid telemetry sequence')
        now = time.monotonic()
        if self.last is not None:
            if seq != self.last['steps']+1:
                raise ValueError('telemetry sequence gap/reset/duplicate')
            dt = stamp-self.last['timestamp_ac']
            if not self.config.min_dt <= dt <= self.config.max_dt or now-self.received > self.config.max_dt:
                raise ValueError('telemetry timing outside accepted bounds')
        self.last, self.received = data, now
        return data

    def send(self, action):
        if self.last is None or time.monotonic()-self.received > self.config.max_dt:
            raise ValueError('cannot control stale telemetry')
        command = controls(action, self.steering, self.last['steps'], saturation=self.config.steering_saturation,
                           steer_lock_rad=self.steer_lock_rad)
        self.sock.send(json.dumps(command, allow_nan=False).encode())
        return command

    def close(self):
        if self.sock is not None:
            try:
                # Release does not require a supported steering target: neutral command is
                # measured at zero road angle; pedals are always explicitly released.
                release = dict(steer=self.steering.command(0.), acc=-1.,
                               brake=1. if self.config.final_brake else -1.,
                               server_steps=int(self.last['steps']) if self.last else 0,
                               enable_gear_shift=False, shift_up=False, shift_down=False)
                self.sock.send(json.dumps(release).encode())
            finally:
                try:
                    self.sock.send(b'disconnect')
                finally:
                    self.sock.close()
                    self.sock = None

    def __exit__(self, *exc):
        self.close()


DTYPES = dict(obs=np.float32, next_obs=np.float32, actions=np.float32, rewards=np.float32,
              terminated=np.bool_, truncated=np.bool_, dt=np.float64, episode_id=np.int64, seq=np.int64)


def write_chunk(path, rows, metadata):
    """Atomic exclusive NPZ publication; numeric arrays and Unicode JSON only."""
    path = Path(path)
    arrays = {}
    for key, dtype in DTYPES.items():
        shape = (len(rows), 159) if key in ('obs', 'next_obs') else (len(rows), 3) if key == 'actions' else (len(rows),)
        arrays[key] = np.asarray([row[key] for row in rows], dtype=dtype).reshape(shape)
    for key, shape in (('timestamp_ac', (len(rows),)), ('next_timestamp_ac', (len(rows),)),
                       ('command', (len(rows), 3)), ('desired_road_angle', (len(rows),)),
                       ('executed_road_angle', (len(rows),))):
        arrays[key] = np.asarray([row[key] for row in rows], dtype=np.float64).reshape(shape)
    arrays['metadata_json'] = np.array(json.dumps(metadata, allow_nan=False, sort_keys=True))
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.npz', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # same-directory exclusive, atomic publication
    finally:
        os.unlink(temporary)


def _settle(session, observation, profile, geometry, state, current, config, deadline):
    """OCD's reset on AC (rlsetup/driving/env.py 'reset', training): the settle driver drives the car
    from the parked hand-over up to v0 = f min v_qss over [s, s + f v_qss(s) settle_s + margin] and
    holds it for settle_s, then the policy takes over. The target is re-evaluated at the car's
    station (OCD places the car at v0; AC must accelerate first, so the window moves with it).
    f = eval_speed_fraction for deterministic drives, else f ~ U(start_speed_fraction) seeded by the
    drive's exploration seed. Nothing of the run-up is stored. Returns (first policy observation or
    None when the run-up failed, latest packet, start metadata)."""
    c = profile.settle
    if config.exploration_std_scale == 0:
        fraction = float(c['eval_speed_fraction'])
    else:
        fraction = float(np.random.default_rng(config.exploration_seed).uniform(*c['start_speed_fraction']))
    driver, action, obs = SettleDriver(c, geometry), RELEASE_ACTION, None
    elapsed, hold_from, dt = 0., None, .04
    start = dict(mode='ocd_settle', fraction=fraction, outcome=None)
    _, _, (station, _, _) = state(current)
    while True:
        speed = math.hypot(number(current, 'velocity_z'), number(current, 'velocity_x'))
        window = fraction*profile.v_start(station)*c['settle_s']+c['settle_window_margin_m']
        target = fraction*profile.v_start_min(station, window)
        start.update(runup_s=elapsed, speed_mps=speed, target_mps=target, station_m=station)
        if hold_from is None and speed >= .9*target:
            hold_from = elapsed
        if obs is not None and hold_from is not None and elapsed-hold_from >= c['settle_s']:
            start['outcome'] = 'handover'
            return obs, current, start
        if elapsed >= config.runup_max_s:
            start['outcome'] = 'runup_timeout'
            return None, current, start
        action = driver.act(current, station, target, dt)
        session.send(action)
        following = session.receive(deadline)
        dt = following['timestamp_ac']-current['timestamp_ac']
        elapsed += dt
        obs = observation.build(following, action)
        _, yaw, (station, _, heading) = state(following)
        current = following
        if number(following, 'numberOfTyresOut') >= 1 or abs(wrap(yaw-heading)) > c['wrong_way_rad']:
            start.update(outcome='runup_off_track', runup_s=elapsed)
            return None, current, start


def collect(config, actor_bundle, calibration, *, reference, output, execute=False, reference_file=None):
    """One explicit handover/chunk; dry run validates local inputs without connecting."""
    steering = SteeringMap(json.loads(Path(calibration).read_text()))
    actor = ActorRuntime(actor_bundle)
    if Path(output).exists():
        raise FileExistsError(output)
    if reference not in REFERENCES:
        raise ValueError('explicit --reference centerline, fast_lane or ocd_path required')
    if (reference == 'ocd_path') != (reference_file is not None):
        raise ValueError('--reference-file is required with, and only with, --reference ocd_path')
    if not execute:
        return {'execute': False, 'message': 'local actor/calibration valid; no server connection or control'}
    client = ManagementClient(ConnectionConfig(config.host, port=config.management_port, timeout_s=config.timeout_s))
    static, server, track = (client.query(name) for name in ('get_static_info', 'get_config', 'get_track_info'))
    steering.verify(static, server, config.setup_id)
    if server.get('ego_sampling_freq') != 25:
        raise ValueError('source requires 25 Hz decision telemetry')
    track, reference_file_sha256 = load_reference(track, reference, reference_file)
    geometry = Geometry(track, reference=reference)
    observation = Observation(geometry, steering.baseline)
    profile = Profile(track['ocd_profile'], geometry) if 'ocd_profile' in track else None
    terms = RewardTerms.from_source(actor.source_config)
    source_env = actor.source_config['env']
    stall_speed, stall_time = float(source_env.get('stall_speed_mps', 3.)), float(source_env.get('stall_time_s', 2.))
    if terms.active and profile is None:
        raise ValueError('the source reward has R4 terms: use a v2 reference file (ac_ocd_reference.py --source-run)')
    if profile is not None:
        if track['ocd_profile']['source_config_sha256'] != actor.manifest['source']['config_sha256']:
            raise ValueError('v2 reference file was made for a different source run config')
        if profile.settle['steer_lock_rad'] != actor.steer_lock_rad:
            raise ValueError('v2 reference settle lock differs from the source actor lock')
        stall_speed, stall_time = profile.settle['stall_speed_mps'], profile.settle['stall_time_s']
    if actor.steer_lock_rad > max(-steering.angles[0], steering.angles[-1])+1e-9:
        print(json.dumps(dict(warning='source steering lock exceeds the measured AC calibration domain; '
                                      'large actions will saturate', steer_lock_rad=actor.steer_lock_rad,
                              calibration_domain_rad=[float(steering.angles[0]), float(steering.angles[-1])])),
              flush=True)
    rows, reason = [], 'max_steps'
    deadline = time.monotonic()+config.max_wall_s
    progress_total = 0.
    metadata = dict(schema_version=1, actor_manifest_sha256=hashlib.sha256((Path(actor_bundle)/'manifest.json').read_bytes()).hexdigest(),
                    source_config_sha256=hashlib.sha256((Path(actor_bundle)/'config.json').read_bytes()).hexdigest(),
                    target={'car_name': static['CarName'], 'track_name': static['TrackName'], 'track_configuration': static['TrackConfiguration'], 'setup_id': config.setup_id},
                    source_contract_sha256=source_contract_sha256(actor.manifest), reward_mode='time_pbrs', reference=reference, gamma=GAMMA, n_steps=1, actor_fixed=True,
                    min_dt_s=config.min_dt, max_dt_s=config.max_dt, timing_bounds=asdict(config),
                    off_track_criterion='numberOfTyresOut>=1 (wheel-off-track)',
                    steering_saturation=config.steering_saturation, saturation_count=0,
                    release_history_action=RELEASE_ACTION.tolist(),
                    calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
                    # v2 profile arrays are identified by reference_file_sha256, not hashed here.
                    track_geometry_sha256=hashlib.sha256(json.dumps(
                        {k: v for k, v in track.items() if k != 'ocd_profile'}, sort_keys=True).encode()).hexdigest(),
                    reference_geometry_sha256=hashlib.sha256(np.ascontiguousarray(
                        np.stack([geometry.ref, geometry.left, geometry.right])).tobytes()).hexdigest(),
                    reference_length_m=geometry.length, reference_file_sha256=reference_file_sha256,
                    terminal_evidence=None,
                    rollout_mode='explore' if config.exploration_std_scale > 0 else 'deterministic',
                    exploration_std_scale=float(config.exploration_std_scale),
                    exploration_seed=config.exploration_seed,
                    steer_lock_rad=actor.steer_lock_rad, start=None)
    if profile is not None:
        # MDP terms beyond legacy R1; part of the replay identity (learn.replay_identity).
        metadata['source_terms'] = dict(asdict(terms), progress='ocd_track_path', start='ocd_settle',
                                        stall_speed_mps=stall_speed, stall_time_s=stall_time)
    generator = None
    if config.exploration_std_scale > 0:
        import torch
        generator = torch.Generator(device='cpu').manual_seed(config.exploration_seed)
    try:
        with Session(config, steering, actor.steer_lock_rad) as session:
            def state(packet):
                pos = np.array([number(packet, 'world_position_x'), number(packet, 'world_position_y')])
                yaw = number(packet, 'yaw')
                return pos, yaw, geometry.project(pos)
            obs = None
            while obs is None:
                current = session.receive(deadline)
                # AC exports reverse=0, neutral=1, first=2. No automatic shifts.
                if number(current, 'actualGear') < 2:
                    raise ValueError('handover requires forward gear; select first manually before collect')
                for field in ('accStatus', 'brakeStatus'):
                    if number(current, field) > .01:
                        raise ValueError('handover requires released throttle/brake')
                session.send(RELEASE_ACTION)
                obs = observation.build(current, RELEASE_ACTION)
            if profile is not None:
                obs, current, start = _settle(session, observation, profile, geometry, state, current, config, deadline)
                metadata['start'] = start
            old_pos, old_yaw, (old_s, old_e, _) = state(current)
            if obs is not None and number(current, 'numberOfTyresOut') >= 1:  # obs None: failed run-up
                raise ValueError('handover wheels outside track')
            stall_s = 0.
            phi_v = (terms.phi_v(math.hypot(number(current, 'velocity_z'), number(current, 'velocity_x')),
                                 profile.v_qss(old_s)) if terms.active else 0.)
            for _ in range(config.max_steps if obs is not None else 0):
                action = (actor.predict(obs) if generator is None else
                          actor.predict(obs, deterministic=False, generator=generator,
                                        std_scale=config.exploration_std_scale))
                command = session.send(action)
                following = session.receive(deadline)
                next_obs = observation.build(following, action)
                if next_obs is None:
                    raise ValueError('missing next observation')
                pos, yaw, (station, e, heading) = state(following)
                ds = geometry.progress(old_s, station)
                dt = following['timestamp_ac']-current['timestamp_ac']
                speed = math.hypot(number(following, 'velocity_z'), number(following, 'velocity_x'))
                if abs(ds) > speed*dt*2+1:
                    raise ValueError('projection jump/reset; transition discarded')
                tyres_out = number(following, 'numberOfTyresOut')
                heading_error = float(wrap(yaw-heading))
                wheel_off_track, wrong_way = tyres_out >= 1, abs(heading_error) > math.pi/2
                failure = wheel_off_track or wrong_way
                progress_total += ds
                lap = progress_total >= geometry.length
                # OCD progress is Track.path arc length; v2 references map AC stations onto it.
                ds_reward = profile.path_progress(old_s, station) if profile is not None else ds
                value = reward(ds_reward, dt, old_e, e, failure=failure)
                if terms.active:
                    phi_v_next = 0. if failure else terms.phi_v(speed, profile.v_qss(station))
                    value += GAMMA*phi_v_next-phi_v
                    value -= terms.line_penalty(abs(e), ds_reward, profile.corner(station))
                    phi_v = phi_v_next
                # OCD (time_pbrs): speed < stall_speed for > stall_time is a truncation, no penalty.
                stall_s = stall_s+dt if speed < stall_speed else 0.
                stalled = profile is not None and not failure and not lap and stall_s > stall_time
                desired = actor.steer_lock_rad*float(action[0])
                effective = float(np.clip(desired, steering.angles[0], steering.angles[-1]))
                metadata['saturation_count'] += int(effective != desired)
                rows.append(dict(obs=obs, next_obs=next_obs, actions=action, rewards=value,
                                 terminated=failure, truncated=(lap or stalled) and not failure, dt=dt, episode_id=0,
                                 seq=current['steps'],
                                 timestamp_ac=current['timestamp_ac'], next_timestamp_ac=following['timestamp_ac'],
                                 command=[command['steer'], command['acc'], command['brake']],
                                 desired_road_angle=desired, executed_road_angle=effective))
                current, obs, old_s, old_e = following, next_obs, station, e
                if failure or lap or stalled:
                    if failure:
                        reason = ('wheel_off_track_and_wrong_way' if wheel_off_track and wrong_way
                                  else 'wheel_off_track' if wheel_off_track else 'wrong_way')
                        metadata['terminal_evidence'] = dict(numberOfTyresOut=tyres_out,
                                                             heading_error_rad=heading_error)
                    else:
                        reason = 'lap_complete' if lap else 'stall'
                    break
            if obs is None:
                reason = metadata['start']['outcome']
    except KeyboardInterrupt:
        reason = 'interrupted'
    except (OSError, ValueError, TimeoutError) as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        if rows and not rows[-1]['terminated']:
            rows[-1]['truncated'] = True
        metadata['stop_reason'] = reason
        metadata['progress_m'] = float(progress_total)
        write_chunk(output, rows, metadata)
    return dict(transitions=len(rows), stop_reason=reason, progress_m=float(progress_total),
                mode=metadata['rollout_mode'], output=str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('collect')
    for name in ('actor-bundle', 'calibration', 'output'):
        run.add_argument('--'+name, type=Path, required=True)
    run.add_argument('--host', default=os.environ.get('AC_HOST'))
    run.add_argument('--reference', choices=list(REFERENCES), required=True)
    run.add_argument('--reference-file', type=Path, help='required with --reference ocd_path')
    run.add_argument('--steering-saturation', choices=['clip', 'reject'], default='reject')
    run.add_argument('--max-steps', type=int, required=True)
    run.add_argument('--max-dt', type=float, default=.08)
    run.add_argument('--min-dt', type=float, default=.02)
    run.add_argument('--max-wall-s', type=float, default=180.)
    run.add_argument('--setup-id', default='default_user_confirmed')
    run.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    config = CollectConfig(args.host, args.max_steps, steering_saturation=args.steering_saturation,
                           max_dt=args.max_dt, min_dt=args.min_dt, max_wall_s=args.max_wall_s, setup_id=args.setup_id)
    print(json.dumps(collect(config, args.actor_bundle, args.calibration, reference=args.reference,
                             output=args.output, execute=args.execute, reference_file=args.reference_file)))


if __name__ == '__main__':
    main()
