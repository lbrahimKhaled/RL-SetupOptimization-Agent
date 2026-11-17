"""Small offline transport/receipt/readiness smoke; no AC connection or actuation."""
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_ac_learn import bundle, chunk
from test_ac_driving import geometry, steering
from rlsetup.assetto.driving import CollectConfig, Session
from rlsetup.assetto import distributed as d


class ScheduledDatagrams:
    """Deterministic local arrival times; nonblocking drain sees only already-arrived packets."""
    def __init__(self, clock, events):
        self.clock,self.events,self.timeout=clock,list(events),None
    def settimeout(self,value):self.timeout=value
    def recv(self,_size):
        if self.events and self.events[0][0] <= self.clock[0]+self.timeout:
            arrival,packet=self.events.pop(0)
            self.clock[0]=max(self.clock[0],arrival)
            return json.dumps(packet,allow_nan=False).encode()
        if self.timeout==0:raise BlockingIOError()
        self.clock[0]+=self.timeout
        raise socket.timeout()
    def send(self,_value):pass
    def close(self):pass


def hold_session(monkeypatch,events):
    clock=[0.]
    monkeypatch.setattr(d.time,'monotonic',lambda:clock[0])
    session=d.BrakeSession(CollectConfig('localhost',1),steering())
    session.last={'steps':1,'timestamp_ac':.04};session.received=0.;session.hold_mode=True
    session.sock=ScheduledDatagrams(clock,events)
    return session


def test_ready_hold_120ms_gap_recovers_but_active_driving_stays_strict(monkeypatch):
    events=[(.12,dict(steps=4,timestamp_ac=.16)),(.16,dict(steps=5,timestamp_ac=.20)),
            (.20,dict(steps=6,timestamp_ac=.24)),(.24,dict(steps=7,timestamp_ac=.28))]
    session=hold_session(monkeypatch,events)
    assert session.receive(1.)['steps']==4 and not session.hold_normal
    for seq in (5,6,7):
        assert session.receive(1.)['steps']==seq and session.hold_normal
    strict=hold_session(monkeypatch,events)
    strict.hold_mode=False
    with pytest.raises(TimeoutError):strict.receive(1.)
    assert Session.receive is not d.BrakeSession.receive


def test_hold_long_silence_duplicate_freshness_and_backwards_are_rejected(monkeypatch):
    session=hold_session(monkeypatch,[(.26,dict(steps=2,timestamp_ac=.08))])
    with pytest.raises(TimeoutError,match='freshness'):session.receive(1.)
    session=hold_session(monkeypatch,[(.04,dict(steps=1,timestamp_ac=.04)),
                                      (.20,dict(steps=1,timestamp_ac=.04))])
    with pytest.raises(TimeoutError,match='freshness'):session.receive(1.)
    session=hold_session(monkeypatch,[(.04,dict(steps=0,timestamp_ac=0.))])
    with pytest.raises(ValueError,match='regression'):session.receive(1.)
    session=hold_session(monkeypatch,[(.04,dict(steps=2,timestamp_ac=float('inf')))])
    with pytest.raises(ValueError):session.receive(1.)


def test_hold_coalescing_validates_every_packet_and_clears_qualification(monkeypatch):
    session=hold_session(monkeypatch,[(.04,dict(steps=2,timestamp_ac=.08)),
                                      (.04,dict(steps=3,timestamp_ac=.12))])
    seen=[];session.hold_validator=lambda packet:seen.append(packet['steps'])
    assert session.receive(1.)['steps']==3
    assert seen==[2,3] and not session.hold_normal


def test_real_reset_excerpt_keeps_3d_stop_guard_and_recovers_120ms_gap(monkeypatch):
    excerpt=json.loads((Path(__file__).parent/'fixtures/ac_reset_settling_excerpt.json').read_text())
    rows=excerpt['samples']
    events=[(row['arrival_s'],dict(steps=row['steps'],timestamp_ac=row['timestamp_ac'])) for row in rows]
    end=rows[-1]
    events.append((end['arrival_s']+.12,dict(steps=end['steps']+3,timestamp_ac=end['timestamp_ac']+.12)))
    for offset in (1,2,3):
        events.append((end['arrival_s']+.12+.04*offset,
                       dict(steps=end['steps']+3+offset,timestamp_ac=end['timestamp_ac']+.12+.04*offset)))
    session=hold_session(monkeypatch,events)
    session.last=dict(excerpt['anchor']);session.wait_phase='RESET_SETTLING'
    for row in rows:
        assert session.receive(2.)['steps']==row['steps']
        assert row['speed'] > d.ReadinessConfig().stationary_mps
    assert max(row['speed'] for row in rows)>2.2
    assert rows[-1]['velocity_planar_mps']<.05<rows[-1]['speed']
    assert session.receive(2.)['steps']==end['steps']+3 and not session.hold_normal
    for offset in (1,2,3):
        assert session.receive(2.)['steps']==end['steps']+3+offset and session.hold_normal
    assert d.ReadinessConfig().stationary_mps==.05


def test_worker_limits_only_its_inference_thread_count_before_preparation(tmp_path):
    calibration=tmp_path/'calibration.json';calibration.write_text('{}')
    code='''import sys,torch
from types import SimpleNamespace
from pathlib import Path
from rlsetup.assetto import distributed as d
torch.set_num_threads(2)
class StopBeforeNetworking:
 def __init__(self,args):
  assert torch.get_num_threads()==1
  print('bounded-before-preparation')
  raise SystemExit(0)
d.Parked=StopBeforeNetworking
args=SimpleNamespace(execute=True,work_dir=Path(sys.argv[1]),calibration=Path(sys.argv[2]),
 server='127.0.0.1',port=1,reference='fast_lane',max_steps=1,max_wall_s=1.,setup_id='test')
d.worker(args,'not-used')
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'worker'),str(calibration)],
                          check=True,capture_output=True,text=True)
    assert '"worker_inference_threads": 1' in result.stdout and 'bounded-before-preparation' in result.stdout


def test_frames_and_bounds():
    left,right = socket.socketpair()
    with left,right:
        thread = threading.Thread(target=d.send,args=(left,'a'*32,dict(op='hello'),b'data'))
        thread.start()
        header,payload = d.receive(right,'a'*32)
        thread.join()
        assert header['op'] == 'hello' and payload == b'data'
        assert 'token' not in header
        left.sendall(struct.pack('!I',65537))
        with pytest.raises(ValueError,match='length'):
            d.receive(right,'a'*32)


def test_frame_deadline_is_absolute_and_unicode_auth_is_rejected():
    left,right=socket.socketpair()
    with left,right:
        right.settimeout(.08)
        def trickle():
            for value in struct.pack('!I',1):
                try:left.sendall(bytes([value]))
                except OSError:return
                time.sleep(.04)
        thread=threading.Thread(target=trickle)
        started=time.monotonic();thread.start()
        with pytest.raises(TimeoutError):d.receive(right,'a'*32)
        assert time.monotonic()-started < .14
        thread.join()
    left,right=socket.socketpair()
    with left,right:
        d.send(left,'é',dict(op='hello'))
        with pytest.raises(ValueError,match='authentication'):d.receive(right,'a'*32)


def test_deep_header_and_chunk_metadata_are_invalid_input_not_parser_crashes(bundle,tmp_path):
    nested='['*15000+']'*15000
    header=('{"op":"hello","nested":'+nested+'}').encode()
    left,right=socket.socketpair()
    with left,right:
        sender=threading.Thread(target=left.sendall,args=(struct.pack('!I',len(header))+header,))
        sender.start()
        with pytest.raises(ValueError,match='JSON'):d.receive(right,'a'*32)
        sender.join(1.)
        assert not sender.is_alive()
    path=chunk(tmp_path/'deep.npz',bundle)
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    arrays['metadata_json']=np.array('{"nested":'+nested+'}')
    np.savez(path,**arrays)
    args=SimpleNamespace(work_dir=tmp_path/'server',actor_bundle=bundle,batch_size=4,
                         buffer_size=32,critic_updates=1,updates=1,max_cycles=2,max_no_update=20)
    server=d.LearnerServer(args)
    with pytest.raises(ValueError,match='JSON'):
        server.process(dict(op='chunk',cycle=1),path.read_bytes())
    assert server.state['cycle']==0


def test_pending_replay_identity_checked_before_receipt(bundle,tmp_path):
    args=SimpleNamespace(work_dir=tmp_path/'server',actor_bundle=bundle,batch_size=32,
                         buffer_size=64,critic_updates=1,updates=1,max_cycles=5,max_no_update=20)
    server=d.LearnerServer(args)
    server.process(dict(op='chunk',cycle=1),chunk(tmp_path/'one.npz',bundle).read_bytes())
    incompatible=chunk(tmp_path/'two.npz',bundle,reference='fast_lane',
                        reference_geometry_sha256='a'*64,reference_length_m=4500.)
    with pytest.raises(ValueError,match='differs'):
        server.process(dict(op='chunk',cycle=2),incompatible.read_bytes())
    assert not (args.work_dir/'cycle-00000002/receipt.json').exists()
    resumed=d.LearnerServer(args)
    good=chunk(tmp_path/'three.npz',bundle)
    response,_=resumed.process(dict(op='chunk',cycle=2),good.read_bytes())
    assert response['cycle']==2 and not response['trained']


def test_committed_cycle_retry_and_restart_never_retrain(bundle,tmp_path,monkeypatch):
    args = SimpleNamespace(work_dir=tmp_path/'server',actor_bundle=bundle,batch_size=4,
                           buffer_size=32,critic_updates=1,updates=1,max_cycles=2,max_no_update=20)
    server = d.LearnerServer(args)
    data = chunk(tmp_path/'chunk.npz',bundle).read_bytes()
    response,actor = server.process(dict(op='chunk',cycle=1),data)
    assert response['trained'] and response['cycle'] == 1
    from rlsetup.assetto.learn import ACLearner
    monkeypatch.setattr(ACLearner,'train',lambda *a: pytest.fail('duplicate training'))
    assert server.process(dict(op='chunk',cycle=1),data) == (response,actor)
    restarted = d.LearnerServer(args)
    assert restarted.process(dict(op='chunk',cycle=1),data) == (response,actor)
    assert restarted.state['cycle'] == 1


def test_worker_import_has_no_sb3_or_ocd():
    code = '''import sys
class Reject:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'stable_baselines3','sb3_contrib','ocd_lap','ocd_native'}:
   raise ImportError('forbidden worker dependency: '+fullname)
sys.meta_path.insert(0,Reject())
import rlsetup.assetto.distributed
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


@pytest.fixture
def park_harness(tmp_path,monkeypatch):
    calibration = tmp_path/'steering.json'
    calibration.write_text(json.dumps(steering().data))
    g = geometry()
    class Management:
        def __init__(self,*a): pass
        def query(self,name):
            return {'get_static_info':dict(CarName='ks_ferrari_sf70h',TrackName='ks_barcelona',TrackConfiguration='layout_gp'),
                    'get_config':dict(sampling_freq=100,ego_sampling_freq=25,vjoy_executed_by_server=True),
                    'get_track_info':dict(left_lane=g.left,right_lane=g.right)}[name]
    monkeypatch.setattr(d,'ManagementClient',Management)
    monkeypatch.setattr(d.SteeringMap,'verify',lambda *a: None)
    h = SimpleNamespace(resets=[],shifts=[],commands=[],mode='normal',post_reset=0,
                        gap_seen=False,after_gap=0,release_after_hold_samples=[])
    h.real_reset_rows=json.loads((Path(__file__).parent/'fixtures/ac_reset_settling_excerpt.json').read_text())['samples']
    class Reset:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def sendall(self,value):
            h.resets.append(value)
            if h.mode=='real_reset':h.session.gear=1
    monkeypatch.setattr(d,'_connect',lambda *a: Reset())
    class Fake:
        def __init__(self,*a):
            self.i,self.gear,self.brake=0,4,1.;self.sock=self;self.hold_normal=True
            h.session=self
        def __enter__(self): return self
        def __exit__(self,*a):
            h.final_brake = not getattr(self,'handover_complete',False)
        def receive(self,*a):
            delay=.001
            if h.mode=='real_reset' and h.resets:
                if h.post_reset < len(h.real_reset_rows):
                    previous=h.real_reset_rows[h.post_reset-1]['arrival_s'] if h.post_reset else 0.
                    delay=h.real_reset_rows[h.post_reset]['arrival_s']-previous
                else:delay=.12 if h.post_reset==len(h.real_reset_rows) else .04
            time.sleep(delay)
            self.i += 1
            if getattr(self,'hold_mode',False) and h.mode=='gap':
                self.hold_normal=h.gap_seen
                if h.gap_seen:h.after_gap+=1
                else:h.gap_seen=True
            if h.resets:
                h.post_reset += 1
            packet=dict(steps=self.i,timestamp_ac=self.i*.04,speed=0.,actualGear=self.gear,
                        accStatus=0.,brakeStatus=self.brake,numberOfTyresOut=0,
                        world_position_x=100.,world_position_y=0.,yaw=1.5707963267948966)
            if h.mode == 'transient' and h.resets and h.post_reset <= 5:
                packet.update(speed=.08,brakeStatus=0.,accStatus=.02,numberOfTyresOut=1)
            if h.mode == 'persistent' and h.resets:
                packet['speed']=.08
            if h.mode=='real_reset' and h.resets:
                if h.post_reset<=len(h.real_reset_rows):
                    measured=h.real_reset_rows[h.post_reset-1]
                    packet.update(speed=measured['speed'],brakeStatus=measured['brakeStatus'],
                                  accStatus=measured['accStatus'])
                else:
                    # Synthetic continuation: one 120ms gap, then stationary normal samples.
                    self.hold_normal=h.post_reset!=len(h.real_reset_rows)+1
            if h.mode == 'creep' and self.brake == 0:
                packet.update(speed=.2,world_position_y=.001*self.i)
            if h.mode == 'fast_creep' and self.brake == 0:
                packet['speed']=1.
            if h.mode.startswith('bad_'):
                key,value={'bad_speed':('speed',.05),'bad_gas':('accStatus',.02),
                           'bad_brake':('brakeStatus',.8),'bad_gear':('actualGear',3),
                           'bad_tyres':('numberOfTyresOut',1),'bad_heading':('yaw',-1.5707963267948966)}[h.mode]
                packet[key]=value
            return packet
        def send(self,action):
            if isinstance(action,bytes):
                command=json.loads(action);h.shifts.append((command['shift_up'],h.post_reset))
                self.gear += 1 if command['shift_up'] else -1
                return
            h.commands.append(action.copy());self.brake=1. if action[2] == 1 else 0.
            if action[2] < 0 and h.mode=='gap':h.release_after_hold_samples.append(h.after_gap)
            return dict(server_steps=self.i)
    monkeypatch.setattr(d,'BrakeSession',Fake)
    h.args=SimpleNamespace(calibration=calibration,ac_host='localhost',ac_port=2345,
                           management_port=2347,reference='centerline',setup_id='fixed',
                           prepare_timeout=2.,work_dir=tmp_path/'worker')
    return h


def trace_rows(parked):
    return [json.loads(line) for line in parked.trace_path.read_text().splitlines()]


def test_park_resets_once_observes_high_gear_downshifts_and_releases(park_harness):
    h=park_harness
    parked=d.Parked(h.args)
    try:
        parked.start();parked.handoff()
    finally:
        parked.close()
    assert h.resets == [b'reset'] and [up for up,_ in h.shifts] == [False,False]
    assert h.commands[-1][2] == -.8 and parked.error is None
    assert not h.final_brake
    rows=trace_rows(parked)
    assert {row['phase'] for row in rows} == {'BRAKE_TO_STOP','RESET_SETTLING','READY_HOLD','HANDOFF'}
    assert rows[-1]['type'] == 'summary' and rows[-1]['success']


def test_reset_transient_recovers_and_deliberate_handoff_creep_is_allowed(park_harness):
    h=park_harness;h.mode='transient'
    parked=d.Parked(h.args)
    try:
        parked.start()
        assert h.shifts and all(at > 5 for _,at in h.shifts)
        h.mode='creep'
        parked.handoff()
    finally:
        parked.close()
    rows=trace_rows(parked)
    assert parked.error is None and not h.final_brake
    assert any(row.get('telemetry',{}).get('speed') == .08 for row in rows)
    summary=rows[-1]
    assert summary['peak_speed_mps_by_phase']['RESET_SETTLING'] == .08
    assert summary['peak_speed_mps_by_phase']['HANDOFF'] == .2
    assert summary['handoff_max_displacement_m'] > 0


def test_real_reset_bounce_needs_three_synthetic_fresh_3d_stopped_samples_before_shift(park_harness):
    h=park_harness;h.mode='real_reset';parked=d.Parked(h.args)
    try:
        parked.start();parked.handoff()
    finally:
        parked.close()
    assert parked.error is None and h.shifts
    assert all(at>=len(h.real_reset_rows)+4 for _,at in h.shifts)
    rows=trace_rows(parked)
    shift_index=next(i for i,row in enumerate(rows) if row['type']=='shift_intent')
    qualifications=[row for row in rows[:shift_index] if row['type']=='prep_qualification']
    assert qualifications[-1]['qualified_samples']==3
    assert any(row['qualified_samples']==0 and not row['normal_cadence'] for row in qualifications)


@pytest.mark.parametrize('mode',['bad_speed','bad_gas','bad_brake','bad_gear','bad_tyres','bad_heading'])
def test_ready_hold_guards_remain_strict(park_harness,mode):
    h=park_harness;parked=d.Parked(h.args)
    try:
        parked.start();h.mode=mode
        parked.thread.join(1.)
        with pytest.raises(RuntimeError,match='ready-hold'):
            parked.check()
    finally:
        parked.close()
    assert h.final_brake and 'phase=READY_HOLD' in parked.error
    assert str(parked.trace_path) in parked.error and 'm/s' in parked.error
    assert trace_rows(parked)[-1]['type'] == 'summary'


def test_persistent_reset_movement_times_out_with_trace_and_no_shifts(park_harness):
    h=park_harness;h.mode='persistent';h.args.prepare_timeout=.15
    parked=d.Parked(h.args)
    try:
        with pytest.raises((RuntimeError,TimeoutError)):
            parked.start()
    finally:
        parked.close()
    assert not h.shifts and h.final_brake
    rows=trace_rows(parked)
    assert any(row.get('telemetry',{}).get('speed') == .08 for row in rows)
    assert not rows[-1]['success']
    retry=d.Parked(h.args)
    assert retry.trace_path != parked.trace_path


def test_handoff_speed_limit_aborts_and_brakes(park_harness):
    h=park_harness;parked=d.Parked(h.args)
    try:
        parked.start();h.mode='fast_creep'
        with pytest.raises(RuntimeError,match='handoff speed'):
            parked.handoff()
    finally:
        parked.close()
    assert h.final_brake and '1.000000 m/s' in parked.error


def test_release_pending_during_hold_gap_waits_for_three_normal_samples(park_harness):
    h=park_harness;parked=d.Parked(h.args)
    try:
        parked.start();h.mode='gap';parked.handoff()
    finally:
        parked.close()
    assert parked.error is None and h.gap_seen
    assert h.release_after_hold_samples[0]==3
    rows=trace_rows(parked)
    assert any(row.get('qualified_samples')==0 and row.get('release_pending') for row in rows)


def _ratio_args(tmp_path,bundle,**changes):
    values=dict(work_dir=tmp_path/'server',actor_bundle=bundle,batch_size=4,buffer_size=64,
                critic_updates=3,updates=0,max_cycles=None,max_no_update=2,
                warmup_transitions=16,transitions_per_update=3,keep_learners=1)
    values.update(changes)
    return SimpleNamespace(**values)


def test_ratio_schedule_warmup_carry_restart_and_prune(bundle,tmp_path):
    args=_ratio_args(tmp_path,bundle)
    server=d.LearnerServer(args)
    def send(cycle):
        actor=server.root/server.state['actor']
        data=chunk(tmp_path/f'c{cycle}.npz',actor,rewards=np.full(8,cycle,np.float32)).read_bytes()
        return server.process(dict(op='chunk',cycle=cycle),data)[0]
    first=send(1)
    assert not first['trained'] and server.state['actor']=='source'  # frozen-actor warmup
    second=send(2)
    assert second['trained'] and (second['critic_updates'],second['updates'])==(3,0)
    assert server.state['actor']!='source'
    third=send(3)
    assert (third['critic_updates'],third['updates'],server.state['carry'])==(0,2,2)
    fourth=send(4)
    assert (fourth['updates'],server.state['carry'])==(3,1)
    assert fourth['complete'] is False and fourth['episode_return']==32.
    snapshots=[p for p in server.root.glob('cycle-*/learner') if (p/'replay.pkl').exists()]
    assert snapshots==[server.root/server.state['learner']]
    assert all((p/'actor/actor.pt').exists() for p in server.root.glob('cycle-*/learner'))
    assert len((server.root/'progress.jsonl').read_text().splitlines())==4
    # Stop/continue: a different cycle budget resumes the same lineage from disk.
    server=d.LearnerServer(_ratio_args(tmp_path,bundle,max_cycles=5))
    assert server.state['cycle']==4 and server.state['carry']==1
    fifth=send(5)
    assert fifth['updates']==3 and fifth['complete'] is True
    with pytest.raises(ValueError,match='budget'):send(6)
    with pytest.raises(ValueError,match='different settings'):
        d.LearnerServer(_ratio_args(tmp_path,bundle,transitions_per_update=4))


def test_ratio_schedule_empty_chunks_abort_but_valid_ones_reset_counter(bundle,tmp_path):
    server=d.LearnerServer(_ratio_args(tmp_path,bundle,warmup_transitions=1000))
    def empty(cycle):
        arrays={key:np.zeros((0,*shape),dtype) for key,(dtype,shape) in
                __import__('rlsetup.assetto.learn',fromlist=['_ARRAYS'])._ARRAYS.items()}
        return chunk(tmp_path/f'e{cycle}.npz',bundle,**arrays).read_bytes()
    server.process(dict(op='chunk',cycle=1),empty(1))
    server.process(dict(op='chunk',cycle=2),chunk(tmp_path/'v.npz',bundle).read_bytes())
    assert server.state['no_update']==0
    server.process(dict(op='chunk',cycle=3),empty(3))
    server.process(dict(op='chunk',cycle=4),empty(4))
    with pytest.raises(ValueError,match='empty chunks'):
        server.process(dict(op='chunk',cycle=5),empty(5))


def _retry_worker(tmp_path,monkeypatch,failures_before_success,limit):
    """Fake preparation fails N times, then the stored cycle budget ends the loop."""
    tmp_path.mkdir(exist_ok=True)
    calibration=tmp_path/'steering.json';calibration.write_text('{}')
    args=SimpleNamespace(execute=True,work_dir=tmp_path/'worker',calibration=calibration,server='127.0.0.1',
                         port=1,reference='fast_lane',max_steps=1,max_wall_s=1.,setup_id='test',
                         max_cycles=3,max_consecutive_failures=limit)
    identity=dict(server='127.0.0.1',port=1,reference='fast_lane',calibration_sha256=d.sha(b'{}'),
                  max_steps=1,max_wall_s=1.,setup_id='test')
    args.work_dir.mkdir()
    d.atomic_json(args.work_dir/'worker.json',dict(identity=identity,cycle=3,actor='actor-00000003'))
    events=[]
    class FlakyParked:
        def __init__(self,_args):pass
        def start(self):
            events.append('start')
            if events.count('start') <= failures_before_success:
                raise RuntimeError('parked control failed: TimeoutError: BRAKE_TO_STOP freshness')
        def close(self):events.append('close')
    monkeypatch.setattr(d,'Parked',FlakyParked)
    sleeps=[];monkeypatch.setattr(d.time,'sleep',sleeps.append)
    return args,events,sleeps


def test_worker_retries_transient_preparation_failures_then_continues(tmp_path,monkeypatch):
    args,events,sleeps=_retry_worker(tmp_path,monkeypatch,2,5)
    d.worker(args,'token')
    assert events==['start','close']*3 and sleeps==[5.,10.]
    log=[json.loads(l) for l in (args.work_dir/'retries.jsonl').read_text().splitlines()]
    assert [r['consecutive_failures'] for r in log]==[1,2] and not any(r['gave_up'] for r in log)


def test_worker_gives_up_after_consecutive_limit_and_default_is_no_retry(tmp_path,monkeypatch):
    args,events,sleeps=_retry_worker(tmp_path/"a",monkeypatch,99,2)
    with pytest.raises(RuntimeError,match='giving up after 3'):d.worker(args,'token')
    assert events.count('start')==3 and events.count('close')==3 and sleeps==[5.,10.]
    args,events,sleeps=_retry_worker(tmp_path/'b',monkeypatch,99,0)
    with pytest.raises(RuntimeError,match='BRAKE_TO_STOP'):d.worker(args,'token')
    assert events==['start','close'] and sleeps==[]


def test_worker_never_retries_contract_errors_or_ctrl_c(tmp_path,monkeypatch):
    args,events,_=_retry_worker(tmp_path,monkeypatch,0,5)
    for exc in (ValueError('server rejected'),KeyboardInterrupt()):
        def start(self,exc=exc):
            events.append('start');raise exc
        monkeypatch.setattr(d.Parked,'start',start)
        events.clear()
        with pytest.raises(type(exc)):d.worker(args,'token')
        assert events==['start','close']

