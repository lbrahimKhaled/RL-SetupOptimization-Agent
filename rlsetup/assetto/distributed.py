"""Stop-and-wait AC worker (Windows) and offline SAC learner (Mac).

Only the server imports SB3/OCD. TCP carries bounded numeric NPZ and allowlisted
actor tensors, never pickle. Game commands remain on the worker's local AC link.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import shutil
import socket
import struct
import threading
import time
import zipfile

import numpy as np

from .actor import ActorRuntime, source_contract_sha256, _json
from .driving import (CollectConfig, Geometry, REFERENCES, RELEASE_ACTION, Session, SteeringMap,
                      collect, load_reference, number, wrap)
from .management import ConnectionConfig, ManagementClient, _connect
from .telemetry import _packet

MAX_PAYLOAD = 32 * 1024 * 1024
ACTOR_FILES = {'actor.pt', 'manifest.json', 'config.json', 'training.json'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def exact(sock, length, deadline):
    chunks = bytearray()
    while len(chunks) < length:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('TCP frame deadline exceeded')
        sock.settimeout(remaining)
        block = sock.recv(min(length-len(chunks), 65536))
        if not block:
            raise ConnectionError('TCP peer disconnected')
        chunks.extend(block)
    return bytes(chunks)


def send(sock, token, header, payload=b''):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError('TCP payload exceeds 32 MiB')
    header = dict(header, token=token, size=len(payload), sha256=sha(payload), protocol=1)
    encoded = json.dumps(header, allow_nan=False).encode()
    if len(encoded) > 65536:
        raise ValueError('TCP header too large')
    sock.sendall(struct.pack('!I', len(encoded))+encoded+payload)


def receive(sock, token):
    timeout = sock.gettimeout()
    deadline = time.monotonic()+(timeout if timeout is not None else 300.)
    try:
        return _receive(sock, token, deadline)
    finally:
        sock.settimeout(timeout)


def _receive(sock, token, deadline):
    length = struct.unpack('!I', exact(sock, 4, deadline))[0]
    if not 0 < length <= 65536:
        raise ValueError('invalid TCP header length')
    header = _json(exact(sock, length, deadline))
    if (not isinstance(header, dict) or header.get('protocol') != 1
            or not isinstance(header.get('token'), str)
            or not hmac.compare_digest(header['token'].encode('utf-8'), token.encode('utf-8'))):
        raise ValueError('TCP authentication/protocol failed')
    size = header.get('size')
    if type(size) is not int or not 0 <= size <= MAX_PAYLOAD:
        raise ValueError('invalid TCP payload length')
    payload = exact(sock, size, deadline)
    if header.get('sha256') != sha(payload):
        raise ValueError('TCP payload hash mismatch')
    header.pop('token', None)  # authentication material must never reach receipts or status logs
    return header, payload


def pack_actor(path):
    ActorRuntime(path)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(ACTOR_FILES):
            file = Path(path)/name
            if file.is_file():
                archive.writestr(name, file.read_bytes())
    return output.getvalue()


def unpack_actor(payload, target):
    """Allowlist extraction and full ActorRuntime validation before activation."""
    target = Path(target)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or not set(names) <= ACTOR_FILES
                or not {'actor.pt','config.json','manifest.json'} <= set(names)
                or sum(info.file_size for info in archive.infolist()) > MAX_PAYLOAD):
            raise ValueError('invalid actor archive members')
    if target.exists():
        if pack_actor(target) != payload:
            # ZIP timestamps need not match; compare extracted member bytes instead.
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if (set(archive.namelist()) != {p.name for p in target.iterdir() if p.is_file()}
                        or any((target/name).read_bytes() != archive.read(name) for name in archive.namelist())):
                    raise ValueError('actor version already exists with different contents')
        ActorRuntime(target)
        return target
    temporary = target.with_name(target.name+'.incoming')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if (len(names) != len(set(names)) or not set(names) <= ACTOR_FILES
                    or not {'actor.pt', 'config.json', 'manifest.json'} <= set(names)
                    or sum(info.file_size for info in archive.infolist()) > MAX_PAYLOAD):
                raise ValueError('invalid actor archive members')
            for name in names:
                (temporary/name).write_bytes(archive.read(name))
        ActorRuntime(temporary)
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


SCHEDULE_DEFAULTS = dict(warmup_transitions=0, transitions_per_update=0, keep_learners=0)


class LearnerServer:
    """Disk receipts make retries and restart return the same committed actor.

    Legacy schedule (transitions_per_update=0): every training cycle runs the fixed
    critic_updates then updates. Ratio schedule (transitions_per_update=k>0): the
    actor stays at the source until warmup_transitions valid transitions are pending,
    then critic_updates critic-only updates run once; afterwards each chunk triggers
    floor((new+carry)/k) joint updates. The cycle budget is not part of the work-dir
    identity, so a stopped run can be continued with a different budget.
    """
    def __init__(self, args):
        from .learn import LearnerConfig, read_chunk, replay_identity
        self.args, self.root = args, Path(args.work_dir).resolve()
        self.schedule = {key: getattr(args, key, value) for key, value in SCHEDULE_DEFAULTS.items()}
        if any(type(v) is not int or v < 0 for v in self.schedule.values()):
            raise ValueError('schedule values must be nonnegative integers')
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = LearnerConfig(batch_size=args.batch_size, buffer_size=args.buffer_size)
        self.config.validate()
        source = pack_actor(args.actor_bundle)
        settings = dict(source_sha256=sha(source), critic_updates=args.critic_updates,
                        updates=args.updates, batch_size=args.batch_size,
                        buffer_size=args.buffer_size, **self.schedule)
        # Hash member contents, not ZIP timestamps, for restart compatibility.
        settings['source_sha256'] = sha((Path(args.actor_bundle)/'manifest.json').read_bytes())
        config_path = self.root/'server.json'
        if config_path.exists():
            saved = dict(SCHEDULE_DEFAULTS, **json.loads(config_path.read_text()))
            saved.pop('max_cycles', None)  # budget may change between restarts
            if saved != settings:
                raise ValueError('server work-dir belongs to different settings/source')
        else:
            unpack_actor(source, self.root/'source')
            atomic_json(config_path, settings)
        self._learner = None  # in-memory cache of the learner saved at state['learner']
        self.state = dict(cycle=0, actor='source', learner=None, pending=[], no_update=0, carry=0)
        for receipt in sorted(self.root.glob('cycle-*/receipt.json')):
            saved = json.loads(receipt.read_text())
            if saved['cycle'] != self.state['cycle']+1:
                raise ValueError('noncontiguous server receipts')
            self.state = saved['state']
        if 'replay_identity' not in self.state:
            identity = None
            if self.state['learner']:
                identity = _json((self.root/self.state['learner']/'manifest.json').read_bytes())['provenance']
            actor_path = self.root/self.state['actor']
            runtime = ActorRuntime(actor_path)
            for relative in self.state['pending']:
                _, metadata, _ = read_chunk(self.root/relative, sha((actor_path/'manifest.json').read_bytes()),
                                           source_contract_sha256(runtime.manifest), self.config.gamma,
                                           runtime.manifest['source']['config_sha256'])
                candidate = replay_identity(metadata)
                if identity is not None and identity != candidate:
                    raise ValueError('legacy pending replay contains incompatible chunk identities')
                identity = candidate
            self.state['replay_identity'] = identity

    def process(self, header, payload):
        from .learn import ACLearner, read_chunk, replay_identity
        if header.get('op') == 'hello':
            return dict(ok=True, cycle=self.state['cycle'], trained=False,
                        batch_size=self.args.batch_size), pack_actor(self.root/self.state['actor'])
        cycle = header.get('cycle')
        if header.get('op') != 'chunk' or type(cycle) is not int or cycle < 1:
            raise ValueError('expected chunk and positive integer cycle')
        directory = self.root/f'cycle-{cycle:08d}'
        receipt = directory/'receipt.json'
        digest = sha(payload)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['chunk_sha256'] != digest:
                raise ValueError('cycle reused with a different chunk')
            return saved['response'], (directory/'actor.zip').read_bytes()
        max_cycles = getattr(self.args, 'max_cycles', None)
        if cycle != self.state['cycle']+1 or (max_cycles is not None and cycle > max_cycles):
            raise ValueError('unexpected cycle or server cycle budget reached')
        if directory.exists():
            shutil.rmtree(directory)  # only an uncommitted local transaction
        directory.mkdir()
        chunk_path = directory/'chunk.npz'
        chunk_path.write_bytes(payload)
        actor_path = self.root/self.state['actor']
        runtime = ActorRuntime(actor_path)
        # Empty startup diagnostics use the same strict schema but cannot establish replay identity.
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 256*1024*1024:
                raise ValueError('expanded chunk exceeds 256 MiB')
        arrays, meta, _ = read_chunk(chunk_path, sha((actor_path/'manifest.json').read_bytes()),
                                    source_contract_sha256(runtime.manifest), self.config.gamma,
                                    runtime.manifest['source']['config_sha256'], allow_empty=True)
        count = len(arrays['obs'])
        state = dict(self.state, cycle=cycle, pending=list(self.state['pending']))
        if count:
            identity = replay_identity(meta)
            if state['replay_identity'] is not None and state['replay_identity'] != identity:
                raise ValueError('chunk target/reference/timing differs from pending/existing AC replay')
            state['replay_identity'] = identity
            state['pending'].append(str(chunk_path.relative_to(self.root)))
        total = 0
        for path in state['pending']:
            with np.load(self.root/path, allow_pickle=False) as archive:
                total += len(archive['obs'])
        ratio = self.schedule['transitions_per_update']
        state.setdefault('carry', 0)
        critic_updates = updates = 0
        if not ratio:
            trained = total >= self.args.batch_size
            if trained:
                critic_updates, updates = self.args.critic_updates, self.args.updates
        elif state['learner'] is None:
            # Frozen-actor warmup: collect with the source actor, then prepare fresh critics.
            trained = total >= max(self.schedule['warmup_transitions'], self.args.batch_size)
            if trained:
                critic_updates = self.args.critic_updates
        else:
            updates, state['carry'] = divmod(state['carry']+total, ratio)
            trained = updates > 0
            if not trained:  # rare: a tiny chunk; keep its data pending with the same actor
                state['carry'] -= total
        if trained:
            try:
                learner = self._load_learner(state['learner'], actor_path)
                for path in state['pending']:
                    learner.import_chunk(self.root/path)
                learner.train(critic_updates, updates)
                learner.save(directory/'learner')
            except BaseException:
                self._learner = None  # a failed transaction must not leak mutated state
                raise
            # Match ACLearner.resume(saved, saved/actor): the next chunk comes from the new actor.
            learner.bundle = directory/'learner'/'actor'
            learner.actor_hash = sha((learner.bundle/'manifest.json').read_bytes())
            self._learner = (str((directory/'learner').relative_to(self.root)), learner)
            state.update(learner=self._learner[0],
                         actor=str((directory/'learner'/'actor').relative_to(self.root)),
                         pending=[], no_update=0)
        elif count:
            state['no_update'] = 0
        else:
            state['no_update'] += 1
            if state['no_update'] > self.args.max_no_update:
                raise ValueError('too many consecutive empty chunks; inspect diagnostics')
        rewards = arrays['rewards'].astype(float)
        response = dict(ok=True, cycle=cycle, trained=trained, transitions=count,
                        accumulated_transitions=total, critic_updates=critic_updates, updates=updates,
                        mode=meta.get('rollout_mode', 'deterministic'),
                        stop_reason=meta.get('stop_reason'), episode_return=float(rewards.sum()),
                        progress_m=meta.get('progress_m'),
                        complete=max_cycles is not None and cycle >= max_cycles)
        actor = pack_actor(self.root/state['actor'])
        (directory/'actor.zip').write_bytes(actor)
        atomic_json(receipt, dict(cycle=cycle, chunk_sha256=digest, response=response, state=state))
        self.state = state
        with (self.root/'progress.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(dict(response, unix_time_s=time.time()), allow_nan=False)+'\n')
        print(json.dumps(response), flush=True)
        self._prune()
        return response, actor

    def _load_learner(self, saved, actor_path):
        from .learn import ACLearner
        if saved is None:
            return ACLearner(actor_path, self.config)
        if self._learner is not None and self._learner[0] == saved:
            return self._learner[1]
        return ACLearner.resume(self.root/saved, actor_path, self.config)

    def _prune(self):
        """Drop replay/model snapshots of older cycles; actors, chunks and receipts stay."""
        keep = self.schedule['keep_learners']
        if not keep:
            return
        learners = sorted(p for p in self.root.glob('cycle-*/learner') if (p/'replay.pkl').exists())
        current = self.root/self.state['learner'] if self.state['learner'] else None
        for path in learners[:-keep]:
            if path != current:
                for name in ('replay.pkl', 'model.zip'):
                    (path/name).unlink(missing_ok=True)


class BrakeSession(Session):
    """A parked session ends with full brake unless handover was verified."""
    handover_complete = False

    def __init__(self, *args):
        super().__init__(*args)
        self.hold_mode, self.hold_normal = False, True
        self.wait_phase = 'READY_HOLD'
        self.hold_config = ReadinessConfig()
        self.hold_validator, self.hold_log = None, None

    def receive(self, deadline):
        if not self.hold_mode or self.last is None or self.received is None:
            establishing = self.last is None or self.received is None
            packet = super().receive(deadline)
            self.hold_normal = not establishing
            return packet
        g = self.hold_config
        while True:
            end = min(deadline, self.received+g.hold_freshness_s)
            remaining = end-time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'{self.wait_phase} advancing telemetry freshness exceeded 0.25 s')
            self.sock.settimeout(remaining)
            try:
                pending = [(self.sock.recv(65508), time.monotonic())]
            except socket.timeout as exc:
                raise TimeoutError(f'{self.wait_phase} advancing telemetry freshness exceeded 0.25 s') from exc
            drained_limit = True
            self.sock.settimeout(0.)
            for _ in range(g.hold_max_packets-1):
                try:
                    pending.append((self.sock.recv(65508), time.monotonic()))
                except (BlockingIOError, socket.timeout):
                    drained_limit = False
                    break
            accepted, normal = None, len(pending) == 1
            for payload, received in pending:
                try:
                    if len(payload) > 65507:
                        raise ValueError('oversized UDP telemetry')
                    packet = _packet(payload)
                    seq, stamp = packet.get('steps'), number(packet, 'timestamp_ac')
                    if type(seq) is not int or seq < 0:
                        raise ValueError(f'invalid {self.wait_phase} telemetry sequence')
                    if self.hold_validator:
                        self.hold_validator(packet)
                    previous_seq, previous_stamp = self.last['steps'], self.last['timestamp_ac']
                    if seq < previous_seq or stamp < previous_stamp:
                        raise ValueError(f'{self.wait_phase} telemetry counter/timestamp regression')
                    if seq == previous_seq or stamp == previous_stamp:
                        if seq != previous_seq or stamp != previous_stamp:
                            raise ValueError(f'{self.wait_phase} inconsistent duplicate telemetry')
                        if self.hold_log:
                            self.hold_log('hold_packet', disposition='duplicate', telemetry=packet)
                        normal = False
                        continue  # duplicates never refresh the advancing-sample deadline
                    dt, local_gap = stamp-previous_stamp, received-self.received
                    if local_gap >= g.hold_freshness_s or dt > g.hold_freshness_s or received > deadline:
                        raise TimeoutError(f'{self.wait_phase} telemetry gap exceeded 0.25 s')
                    contiguous = seq == previous_seq+1
                    packet_normal = (contiguous and self.config.min_dt <= dt <= self.config.max_dt
                                     and local_gap <= self.config.max_dt)
                    normal = normal and packet_normal
                    if self.hold_log:
                        self.hold_log('hold_packet', disposition='advancing', telemetry=packet,
                                      seq_delta=seq-previous_seq, server_dt_s=dt, local_gap_s=local_gap,
                                      normal_cadence=packet_normal, drained_packets=len(pending))
                    self.last, self.received, accepted = packet, received, packet
                except BaseException:
                    if self.hold_log:
                        self.hold_log('hold_rejected_packet', raw_payload=payload.decode('utf-8', errors='replace'))
                    raise
            if drained_limit:
                raise ValueError(f'{self.wait_phase} telemetry drain bound exceeded')
            if accepted is not None:
                self.hold_normal = normal
                return accepted

    def close(self):
        if self.sock is not None:
            try:
                command = dict(steer=self.steering.command(0.), acc=-1.,
                               brake=-1. if self.handover_complete else 1.,
                               server_steps=int(self.last['steps']) if self.last else 0,
                               enable_gear_shift=False, shift_up=False, shift_down=False)
                self.sock.send(json.dumps(command).encode())
            finally:
                try:
                    self.sock.send(b'disconnect')
                finally:
                    self.sock.close()
                    self.sock = None


@dataclass(frozen=True)
class ReadinessConfig:
    """Diagnostic guards, SI units; handoff creep is not a validated vehicle parameter."""
    stationary_mps: float = .05
    full_brake: float = .9
    released_pedal: float = .01
    stable_samples: int = 3
    reset_settle_s: float = .4
    handoff_s: float = 1.
    handoff_max_mps: float = 1.
    handoff_samples: int = 2
    shift_confirmation_s: float = 1.
    hold_freshness_s: float = .25
    hold_max_packets: int = 32


class Parked:
    """One UDP owner with distinct reset-settle, strict hold and released handoff phases."""
    def __init__(self, args):
        self.args, self.guards = args, ReadinessConfig()
        self.ready, self.release, self.stop = threading.Event(), threading.Event(), threading.Event()
        self.error, self.phase, self.last_speed = None, 'BRAKE_TO_STOP', None
        root = Path(getattr(args, 'work_dir', Path(args.calibration).parent))
        self.trace_path = root/f'readiness-{time.time_ns()}.jsonl'
        self.thread = threading.Thread(target=self.run, name='ac-park', daemon=True)

    def start(self):
        self.thread.start()
        deadline = time.monotonic()+self.args.prepare_timeout
        while not self.ready.wait(.05):
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError(f'automatic AC preparation timed out; trace={self.trace_path}')
        self.check()
        return self

    def check(self):
        if self.error:
            raise RuntimeError(f'parked control failed: {self.error}')
        if not self.thread.is_alive():
            raise RuntimeError(f'parked control ended unexpectedly; trace={self.trace_path}')

    def handoff(self):
        self.release.set()
        self.thread.join(self.guards.handoff_s+2.)
        if self.thread.is_alive() or self.error:
            raise RuntimeError(f'AC handover failed: {self.error or "timeout"}; trace={self.trace_path}')

    def close(self):
        self.stop.set()
        self.thread.join(3.)

    def run(self):
        peak = {}
        success = False
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open('x', encoding='utf-8') as stream:
                def log(kind, **data):
                    row = dict(type=kind, phase=self.phase, monotonic=time.monotonic(),
                               unix_time_s=time.time(), **data)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()

                def phase(name):
                    self.phase = name
                    log('phase')

                log('metadata', guards=asdict(self.guards), prepare_timeout_s=self.args.prepare_timeout,
                    note='Reset is send-only; final brake transmission is best effort, not acknowledgement.')
                try:
                    success = self._run(log, phase, peak)
                except BaseException as exc:
                    speed = ('unknown' if self.last_speed is None else
                             f'{self.last_speed:.6f} m/s ({3.6*self.last_speed:.6f} km/h)')
                    self.error = (f'{type(exc).__name__}: {exc}; phase={self.phase}; speed={speed}; '
                                  f'trace={self.trace_path}')
                    log('failure', error=self.error)
                finally:
                    log('summary', success=success, stopped=self.stop.is_set(), error=self.error,
                        peak_speed_mps_by_phase=peak,
                        post_reset_max_displacement_m=getattr(self,'reset_displacement',0.),
                        handoff_max_displacement_m=getattr(self,'handoff_displacement',0.))
        except BaseException as exc:
            self.error = self.error or f'{type(exc).__name__}: {exc}; phase={self.phase}; trace={self.trace_path}'

    def _run(self, log, phase, peak):
        a, g = self.args, self.guards
        steering = SteeringMap(json.loads(Path(a.calibration).read_text()))
        client = ManagementClient(ConnectionConfig(a.ac_host, port=a.management_port, timeout_s=2.))
        static, server, track = [client.query(op) for op in ('get_static_info','get_config','get_track_info')]
        log('management', static=static, server=server)
        steering.verify(static, server, a.setup_id)
        if (static.get('CarName'), static.get('TrackName'), static.get('TrackConfiguration')) != (
                'ks_ferrari_sf70h', 'ks_barcelona', 'layout_gp'):
            raise ValueError('automatic runner requires SF70H / Barcelona GP')
        if server.get('sampling_freq') != 100 or server.get('ego_sampling_freq') != 25:
            raise ValueError('automatic runner requires 100/25 Hz AC server')
        track, _ = load_reference(track, a.reference, getattr(a, 'reference_file', None))
        geometry = Geometry(track, reference=a.reference)
        config = CollectConfig(a.ac_host, 1, port=a.ac_port, management_port=a.management_port,
                               steering_saturation='clip', max_wall_s=3600., setup_id=a.setup_id)
        deadline = time.monotonic()+a.prepare_timeout
        reset_at, stable, pulse, pulse_deadline, released, handoff_at = None, 0, None, None, 0, None
        strict_after_shift = False
        hold_qualified = g.stable_samples
        reset_origin, handoff_origin = None, None
        with BrakeSession(config, steering) as session:
            def validate_hold(packet):
                speed = number(packet, 'speed')
                self.last_speed = speed
                peak['READY_HOLD'] = max(peak.get('READY_HOLD', 0.), speed)
                position = np.array([number(packet, 'world_position_x'), number(packet, 'world_position_y')])
                _, _, heading = geometry.project(position)
                if (not 0 <= speed < g.stationary_mps
                        or not 0 <= number(packet, 'accStatus') <= g.released_pedal
                        or not g.full_brake <= number(packet, 'brakeStatus') <= 1.01
                        or number(packet, 'actualGear') != 2
                        or number(packet, 'numberOfTyresOut') != 0
                        or abs(wrap(number(packet, 'yaw')-heading)) >= math.pi/2):
                    raise ValueError('ready-hold stationary/control/geometry guard failed')
            def validate_wait(packet):
                if self.phase == 'READY_HOLD':
                    return validate_hold(packet)
                # Before readiness, reset motion/pedal release/geometry transients are allowed.
                speed, gear = number(packet, 'speed'), number(packet, 'actualGear')
                self.last_speed = speed
                peak[self.phase] = max(peak.get(self.phase, 0.), speed)
                if (speed < 0 or gear != int(gear) or not 0 <= gear <= 9
                        or not 0 <= number(packet, 'accStatus') <= 1.01
                        or not 0 <= number(packet, 'brakeStatus') <= 1.01):
                    raise ValueError('invalid observed speed, gear or pedal range')
                for key in ('world_position_x', 'world_position_y', 'yaw', 'numberOfTyresOut'):
                    number(packet, key)
            session.hold_validator, session.hold_log = validate_wait, log
            while not self.stop.is_set():
                session.wait_phase = self.phase
                session.hold_mode = (self.phase in ('BRAKE_TO_STOP','RESET_SETTLING','READY_HOLD')
                                     and pulse is None and not strict_after_shift)
                packet = session.receive(time.monotonic()+1.)
                strict_after_shift = False
                if self.phase == 'READY_HOLD':
                    validate_hold(packet)
                    normal = getattr(session, 'hold_normal', True)
                    previous_qualified = hold_qualified
                    hold_qualified = min(g.stable_samples, hold_qualified+1) if normal else 0
                    if hold_qualified != previous_qualified or not normal:
                        log('hold_qualification', normal_cadence=normal, qualified_samples=hold_qualified,
                            release_pending=self.release.is_set())
                    if self.release.is_set() and hold_qualified >= g.stable_samples:
                        phase('HANDOFF')
                        session.hold_mode = False
                        handoff_at = time.monotonic()
                releasing = self.phase == 'HANDOFF'
                action = RELEASE_ACTION if releasing else np.array([0., -.8, 1.], np.float32)
                log('sample', telemetry=packet)
                command = session.send(action)
                log('command', command=command, action=action.tolist())
                speed, gear = number(packet,'speed'), number(packet,'actualGear')
                gas, brake = number(packet,'accStatus'), number(packet,'brakeStatus')
                self.last_speed = speed
                peak[self.phase] = max(peak.get(self.phase, 0.), speed)
                if (speed < 0 or gear != int(gear) or not 0 <= gear <= 9
                        or not 0 <= gas <= 1.01 or not 0 <= brake <= 1.01):
                    raise ValueError('invalid observed speed, gear or pedal range')
                position = np.array([number(packet,'world_position_x'),number(packet,'world_position_y')])
                _, _, heading = geometry.project(position)
                tyres_out = number(packet,'numberOfTyresOut')
                heading_error = float(wrap(number(packet,'yaw')-heading))
                on_track = tyres_out == 0 and abs(heading_error) < math.pi/2
                if reset_at is not None:
                    if reset_origin is None:
                        reset_origin = position.copy()
                    self.reset_displacement = max(getattr(self,'reset_displacement',0.),
                                                  float(np.linalg.norm(position-reset_origin)))
                if releasing:
                    if handoff_origin is None:
                        handoff_origin = position.copy()
                    self.handoff_displacement = max(getattr(self,'handoff_displacement',0.),
                                                    float(np.linalg.norm(position-handoff_origin)))
                    if time.monotonic()-handoff_at >= g.handoff_s:
                        raise TimeoutError('released-pedal handoff deadline exceeded')
                    if speed >= g.handoff_max_mps or gas > g.released_pedal or gear != 2 or not on_track:
                        raise ValueError('released-pedal handoff speed/control/geometry guard failed')
                    released = released+1 if brake <= g.released_pedal else 0
                    if released >= g.handoff_samples:
                        session.handover_complete = True
                        log('handover_verified', speed_mps=speed, heading_error_rad=heading_error)
                        return True
                    continue
                stopped = speed < g.stationary_mps and gas <= g.released_pedal and brake >= g.full_brake
                if self.phase == 'READY_HOLD':
                    if not stopped or gear != 2 or not on_track:
                        raise ValueError('ready-hold stationary/control/geometry guard failed')
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError('stationary/reset/gear readiness deadline exceeded')
                if not getattr(session, 'hold_normal', True):
                    stable = 0
                    log('prep_qualification', normal_cadence=False, qualified_samples=0)
                    continue
                if pulse is not None:
                    if gear == pulse:
                        pulse, stable = None, 0
                        strict_after_shift = True
                        log('shift_confirmed', gear=gear)
                        continue  # cleared flags persist for a complete telemetry interval
                    if time.monotonic() >= pulse_deadline:
                        raise TimeoutError('gear pulse was not confirmed by telemetry')
                    stable = 0
                    continue
                if reset_at is None:
                    stable = stable+1 if stopped else 0
                    if stable < g.stable_samples:
                        continue
                    conn = ConnectionConfig(a.ac_host, port=a.management_port, timeout_s=2.)
                    reset_at = time.monotonic()
                    log('reset_intent', timestamp_ac=number(packet,'timestamp_ac'))
                    with _connect(conn, time.monotonic()+2.) as sock:
                        sock.sendall(b'reset')
                    log('reset_sent', note='No acknowledgement; reset may release pedals and invalidate Hotlap.')
                    stable = 0
                    phase('RESET_SETTLING')
                    continue
                stable = stable+1 if (stopped and on_track and
                                          time.monotonic()-reset_at >= g.reset_settle_s) else 0
                log('prep_qualification', normal_cadence=True, qualified_samples=stable)
                if stable < g.stable_samples:
                    continue
                if gear != 2:
                    up = gear < 2
                    command.update(enable_gear_shift=True, shift_up=up, shift_down=not up)
                    log('shift_intent', command=command, observed_gear=gear)
                    session.sock.send(json.dumps(command).encode())
                    pulse, pulse_deadline = gear+(1 if up else -1), time.monotonic()+g.shift_confirmation_s
                    stable = 0
                    continue
                phase('READY_HOLD')
                self.ready.set()
            return False


def exchange(args, token, header, payload, parked):
    last = None
    for attempt in range(args.reconnect_attempts):
        parked.check()
        try:
            with socket.create_connection((args.server,args.port),timeout=10.) as sock:
                sock.settimeout(args.network_timeout)
                send(sock,token,header,payload)
                response, actor = receive(sock,token)
            parked.check()
            if response.get('ok') is not True:
                raise ValueError(response.get('error','server rejected transaction'))
            return response, actor
        except (OSError, ConnectionError) as exc:
            last = exc
            time.sleep(min(1.+attempt,5.))
    raise ConnectionError(f'TCP retry budget exhausted: {last}')


# Telemetry stalls, readiness timeouts, AC/learner connection loss. Not retried: server
# rejections and contract/identity errors (ValueError), and Ctrl+C (KeyboardInterrupt).
RETRYABLE = (RuntimeError, TimeoutError, ConnectionError, OSError)


def worker(args, token):
    if not args.execute:
        print('Offline plan only; add --execute to authorize automatic brake/reset/gear/control.')
        return
    import torch
    torch.set_num_threads(1)
    print(json.dumps(dict(worker_inference_threads=torch.get_num_threads())), flush=True)
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root/'worker.json'
    identity = dict(server=args.server,port=args.port,reference=args.reference,
                    calibration_sha256=sha(Path(args.calibration).read_bytes()),
                    max_steps=args.max_steps,max_wall_s=args.max_wall_s,setup_id=args.setup_id)
    reference_file = getattr(args,'reference_file',None)
    if reference_file is not None:
        identity['reference_file_sha256'] = sha(Path(reference_file).read_bytes())
    state = json.loads(path.read_text()) if path.exists() else None
    if state is not None and state['identity'] != identity:
        raise ValueError('worker work-dir belongs to different settings/calibration')
    limit = getattr(args,'max_consecutive_failures',0)
    failures = 0
    while True:
        # worker.json is rewritten atomically after every committed step: disk is the truth.
        state = json.loads(path.read_text()) if path.exists() else None
        parked = Parked(args)
        try:
            if _worker_cycle(args,token,root,path,identity,state,parked) is None:
                return
            failures = 0  # a drive completed: the loop is healthy again
            continue
        except RETRYABLE as exc:
            if not limit:
                raise
            failure = exc
        finally:
            parked.close()
        failures += 1
        wait = min(60., 5.*2**(failures-1))
        error = f'{type(failure).__name__}: {failure}'
        with (root/'retries.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps(dict(unix_time_s=time.time(),consecutive_failures=failures,limit=limit,
                                      error=error,gave_up=failures > limit),allow_nan=False)+'\n')
        if failures > limit:
            raise RuntimeError(f'giving up after {failures} consecutive failures; last: {error}') from failure
        print(f'Recoverable failure {failures}/{limit}: {error}\n'
              f'Car left on full brake; restarting stop/reset/gear preparation in {wait:.0f} s.',flush=True)
        time.sleep(wait)


def _worker_cycle(args, token, root, path, identity, state, parked):
    """One prepare/exchange/drive cycle; returns the new worker state, or None when done."""
    parked.start()
    if state is None:
        response, actor = exchange(args,token,dict(op='hello'),b'',parked)
        cycle = response['cycle']
        unpack_actor(actor,root/f'actor-{cycle:08d}')
        state = dict(identity=identity,cycle=cycle,actor=f'actor-{cycle:08d}')
        atomic_json(path,state)
    pending = root/f'chunk-{state["cycle"]+1:08d}.npz'
    if pending.exists():
        cycle = state['cycle']+1
        response, actor = exchange(args,token,dict(op='chunk',cycle=cycle),pending.read_bytes(),parked)
        unpack_actor(actor,root/f'actor-{cycle:08d}')
        state.update(cycle=cycle,actor=f'actor-{cycle:08d}')
        atomic_json(path,state)
        print(json.dumps(response),flush=True)
    max_cycles = getattr(args,'max_cycles',None)
    if max_cycles is not None and state['cycle'] >= max_cycles:
        print(f'Requested cycle limit ({max_cycles}) reached; restart with a larger '
              '--max-cycles (or none) to continue from the latest actor. '
              'Car held with best-effort full brake.',flush=True)
        return None
    parked.handoff()
    next_cycle = state['cycle']+1
    eval_every = getattr(args,'eval_every',0)
    evaluate = bool(eval_every) and next_cycle % eval_every == 0
    scale = 0. if evaluate else getattr(args,'explore_std_scale',0.)
    config = CollectConfig(args.ac_host,args.max_steps,port=args.ac_port,
                           management_port=args.management_port,setup_id=args.setup_id,
                           steering_saturation='clip',max_wall_s=args.max_wall_s,final_brake=True,
                           exploration_std_scale=scale,exploration_seed=next_cycle)
    result = collect(config,root/state['actor'],args.calibration,reference=args.reference,
                     output=root/f'chunk-{next_cycle:08d}.npz',execute=True,
                     reference_file=getattr(args,'reference_file',None))
    print(json.dumps(result),flush=True)
    if result['stop_reason'] == 'interrupted':
        raise KeyboardInterrupt
    return state


def server(args, token):
    learner = LearnerServer(args)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind((args.bind,args.port))
        listener.listen(1)
        print(f'AC learner listening on {args.bind}:{args.port}',flush=True)
        while True:
            connection,_ = listener.accept()
            with connection:
                connection.settimeout(args.network_timeout)
                authenticated = False
                try:
                    header,payload = receive(connection,token)
                    authenticated = True
                    response,actor = learner.process(header,payload)
                    send(connection,token,response,actor)
                except (ValueError,OSError,KeyError,TypeError,EOFError,zipfile.BadZipFile) as exc:
                    if not authenticated:
                        continue
                    try:
                        send(connection,token,dict(ok=False,error=f'{type(exc).__name__}: {exc}'))
                    except OSError:
                        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='role',required=True)
    for role in ('server','worker'):
        p = sub.add_parser(role)
        p.add_argument('--port',type=int,default=2350)
        p.add_argument('--token-file',type=Path,required=True)
        p.add_argument('--work-dir',type=Path,required=True)
        p.add_argument('--max-cycles',type=int,default=None,
                       help='optional stop after this cycle; omit to run until stopped (Ctrl+C)')
        p.add_argument('--network-timeout',type=float,default=300.)
        if role == 'server':
            p.add_argument('--bind',default='0.0.0.0')
            p.add_argument('--actor-bundle',type=Path,required=True)
            p.add_argument('--critic-updates',type=int,required=True)
            p.add_argument('--updates',type=int,required=True)
            p.add_argument('--batch-size',type=int,required=True)
            p.add_argument('--buffer-size',type=int,default=100000)
            p.add_argument('--max-no-update',type=int,default=20,
                           help='abort after this many consecutive empty chunks')
            p.add_argument('--warmup-transitions',type=int,default=0,
                           help='ratio schedule: keep the source actor until this many transitions')
            p.add_argument('--transitions-per-update',type=int,default=0,
                           help='0 = legacy fixed updates per cycle; k>0 = one joint update per k new transitions')
            p.add_argument('--keep-learners',type=int,default=0,
                           help='keep replay/model snapshots of only the newest N cycles (0 = keep all)')
        else:
            p.add_argument('--server',required=True)
            p.add_argument('--calibration',type=Path,required=True)
            p.add_argument('--ac-host',default='127.0.0.1')
            p.add_argument('--ac-port',type=int,default=2345)
            p.add_argument('--management-port',type=int,default=2347)
            p.add_argument('--reference',choices=list(REFERENCES),default='fast_lane')
            p.add_argument('--reference-file',type=Path,help='required with --reference ocd_path')
            p.add_argument('--setup-id',default='default_user_confirmed')
            p.add_argument('--max-steps',type=int,required=True)
            p.add_argument('--max-wall-s',type=float,required=True)
            p.add_argument('--prepare-timeout',type=float,default=20.)
            p.add_argument('--reconnect-attempts',type=int,default=5)
            p.add_argument('--max-consecutive-failures',type=int,default=0,
                           help='retry preparation after telemetry/readiness/connection failures this many '
                                'times in a row (car kept braked; 0 = stop on first failure)')
            p.add_argument('--explore-std-scale',type=float,default=0.,
                           help='training rollouts sample with this fraction of the policy std (0 = deterministic)')
            p.add_argument('--eval-every',type=int,default=0,
                           help='every Nth cycle is a deterministic evaluation rollout (0 = never)')
            p.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    if (not 1 <= args.port <= 65535 or not 1 <= args.network_timeout <= 3600
            or (args.max_cycles is not None and not 1 <= args.max_cycles <= 10_000_000)):
        parser.error('invalid port/cycle/time budget')
    if args.role == 'server' and (args.critic_updates < 0 or args.updates < 0 or args.max_no_update < 1
                                  or args.warmup_transitions < 0 or args.transitions_per_update < 0
                                  or args.keep_learners < 0):
        parser.error('update/schedule budgets must be nonnegative; max-no-update must be positive')
    if args.role == 'worker' and (not 0 <= args.explore_std_scale <= 1 or args.eval_every < 0
                                  or not 0 <= args.max_consecutive_failures <= 100):
        parser.error('explore-std-scale must be in [0, 1]; eval-every nonnegative; failures 0..100')
    if args.role == 'worker' and (not 1 <= args.prepare_timeout <= 60 or not 1 <= args.reconnect_attempts <= 20):
        parser.error('invalid preparation/reconnect budget')
    if args.role == 'worker' and (args.reference == 'ocd_path') != (args.reference_file is not None):
        parser.error('--reference-file is required with, and only with, --reference ocd_path')
    token = args.token_file.read_text().strip()
    if len(token) < 32:
        parser.error('shared token must contain at least 32 characters')
    (server if args.role == 'server' else worker)(args,token)


if __name__ == '__main__':
    main()
