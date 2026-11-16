"""Thin AC collector: explicit centerline guidance, frozen actor, bounded UDP control.

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

from .actor import ActorRuntime, SOURCE_OBS, source_contract_sha256
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


class Geometry:
    """Closed corresponding borders; reference is explicitly their midpoint.

    Labels are validated against forward tangents, never automatically reversed.
    Four oriented body corners are checked against local border cross-sections.
    """
    def __init__(self, track, *, reference):
        if reference != 'centerline':
            raise ValueError('explicit reference=centerline required')
        self.left = np.asarray(track['left_lane'], dtype=float)
        self.right = np.asarray(track['right_lane'], dtype=float)
        if (self.left.ndim != 2 or self.left.shape[1] != 2 or len(self.left) < 4
                or self.right.shape != self.left.shape
                or not np.isfinite(self.left).all() or not np.isfinite(self.right).all()):
            raise ValueError('finite corresponding Nx2 borders required')
        if np.linalg.norm((self.left[0]+self.right[0]-self.left[-1]-self.right[-1])/2) < 1e-6:
            self.left, self.right = self.left[:-1], self.right[:-1]
        self.ref = (self.left + self.right) / 2
        self.delta = np.roll(self.ref, -1, axis=0) - self.ref
        self.lengths = np.linalg.norm(self.delta, axis=1)
        width = np.linalg.norm(self.left-self.right, axis=1)
        cross = self.delta[:, 0]*(self.left-self.right)[:, 1] - self.delta[:, 1]*(self.left-self.right)[:, 0]
        if np.any(self.lengths <= 1e-6) or np.any(width <= 0) or np.any(cross <= 0):
            raise ValueError('degenerate geometry or incorrectly labelled left/right borders')
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


def controls(action, steering, seq, *, saturation='reject'):
    action = np.asarray(action)
    if action.shape != (3,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
        raise ValueError('finite normalized action[3] required')
    pedals = np.clip((action[1:]+.5)/1.5, 0, 1)
    return dict(steer=steering.command(.6*float(action[0]), saturation=saturation), acc=float(2*pedals[0]-1),
                brake=float(2*pedals[1]-1), server_steps=int(seq),
                enable_gear_shift=False, shift_up=False, shift_down=False)


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

    def __post_init__(self):
        CaptureConfig(self.host, port=self.port)
        if type(self.max_steps) is not int or not 0 < self.max_steps <= 100000:
            raise ValueError('max_steps must be in 1..100000')
        for value in (self.min_dt, self.max_dt, self.max_wall_s, self.timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('dimensions/timing must be finite positive')
        if self.steering_saturation not in ('clip', 'reject'):
            raise ValueError('steering_saturation must be clip or reject')
        if self.min_dt > self.max_dt or self.max_wall_s > 3600:
            raise ValueError('invalid timing bounds')


class Session:
    """Fresh contiguous UDP telemetry and counter echo; releases before disconnect."""
    def __init__(self, config, steering):
        self.config, self.steering = config, steering
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
        self.sock.settimeout(max(.001, min(deadline-time.monotonic(), self.config.max_dt)))
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
        command = controls(action, self.steering, self.last['steps'], saturation=self.config.steering_saturation)
        self.sock.send(json.dumps(command, allow_nan=False).encode())
        return command

    def close(self):
        if self.sock is not None:
            try:
                # Release does not require a supported steering target: neutral command is
                # measured at zero road angle; pedals are always explicitly released.
                release = dict(steer=self.steering.command(0.), acc=-1., brake=-1.,
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


def collect(config, actor_bundle, calibration, *, reference, output, execute=False):
    """One explicit handover/chunk; dry run validates local inputs without connecting."""
    steering = SteeringMap(json.loads(Path(calibration).read_text()))
    actor = ActorRuntime(actor_bundle)
    if Path(output).exists():
        raise FileExistsError(output)
    if reference != 'centerline':
        raise ValueError('explicit --reference centerline required')
    if not execute:
        return {'execute': False, 'message': 'local actor/calibration valid; no server connection or control'}
    client = ManagementClient(ConnectionConfig(config.host, port=config.management_port, timeout_s=config.timeout_s))
    static, server, track = (client.query(name) for name in ('get_static_info', 'get_config', 'get_track_info'))
    steering.verify(static, server, config.setup_id)
    if server.get('ego_sampling_freq') != 25:
        raise ValueError('source requires 25 Hz decision telemetry')
    geometry = Geometry(track, reference=reference)
    observation = Observation(geometry, steering.baseline)
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
                    track_geometry_sha256=hashlib.sha256(json.dumps(track, sort_keys=True).encode()).hexdigest())
    try:
        with Session(config, steering) as session:
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
            def state(packet):
                pos = np.array([number(packet, 'world_position_x'), number(packet, 'world_position_y')])
                yaw = number(packet, 'yaw')
                return pos, yaw, geometry.project(pos)
            old_pos, old_yaw, (old_s, old_e, _) = state(current)
            if number(current, 'numberOfTyresOut') >= 1:
                raise ValueError('handover wheels outside track')
            for _ in range(config.max_steps):
                action = actor.predict(obs)
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
                failure = (number(following, 'numberOfTyresOut') >= 1
                           or abs(wrap(yaw-heading)) > math.pi/2)
                progress_total += ds
                lap = progress_total >= geometry.length
                desired = .6*float(action[0])
                effective = float(np.clip(desired, steering.angles[0], steering.angles[-1]))
                metadata['saturation_count'] += int(effective != desired)
                rows.append(dict(obs=obs, next_obs=next_obs, actions=action,
                                 rewards=reward(ds, dt, old_e, e, failure=failure),
                                 terminated=failure, truncated=lap and not failure, dt=dt, episode_id=0, seq=current['steps'],
                                 timestamp_ac=current['timestamp_ac'], next_timestamp_ac=following['timestamp_ac'],
                                 command=[command['steer'], command['acc'], command['brake']],
                                 desired_road_angle=desired, executed_road_angle=effective))
                current, obs, old_s, old_e = following, next_obs, station, e
                if failure or lap:
                    reason = 'failure' if failure else 'lap_complete'
                    break
    except KeyboardInterrupt:
        reason = 'interrupted'
    except (OSError, ValueError, TimeoutError) as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        if rows and not rows[-1]['terminated']:
            rows[-1]['truncated'] = True
        metadata['stop_reason'] = reason
        write_chunk(output, rows, metadata)
    return dict(transitions=len(rows), stop_reason=reason, output=str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('collect')
    for name in ('actor-bundle', 'calibration', 'output'):
        run.add_argument('--'+name, type=Path, required=True)
    run.add_argument('--host', default=os.environ.get('AC_HOST'))
    run.add_argument('--reference', choices=['centerline'], required=True)
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
                             output=args.output, execute=args.execute)))


if __name__ == '__main__':
    main()
