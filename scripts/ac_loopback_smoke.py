"""Real worker/learner subprocesses against a localhost-only AC protocol simulator.

This validates transport, collection, updates and recovery. It is not vehicle
validation, and never connects to an installed game or a non-loopback address.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlsetup.assetto import distributed as d
from rlsetup.assetto.actor import ActorRuntime


def wait_for(predicate, seconds, message):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError(message)


class FakeAC:
    """25 Hz ego packets plus the existing read-only/reset management protocol."""
    def __init__(self):
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(('127.0.0.1', 0)); self.udp.settimeout(.005)
        self.management = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.management.bind(('127.0.0.1',0)); self.management.listen(8); self.management.settimeout(.05)
        self.client, self.session_kind = None, None
        self.seq, self.stamp, self.gear = 0, 0., 4
        self.gas, self.brake, self.steer, self.speed = 0., 1., 0., 0.
        self.position, self.yaw = np.array([100.,0.]), math.pi/2
        self.sessions, self.collectors, self.resets = 0, 0, 0
        self.samples, self.overlaps, self.commands = [], [], []
        self.retired_clients, self.stale_replacements = set(), 0
        self.between_session_brakes, self.delayed_management = [], False
        self.pause_until, self.telemetry_pauses = 0., []
        self.reset_transient, self.reset_recovered = 0, False
        excerpt=json.loads((ROOT/'tests/fixtures/ac_reset_settling_excerpt.json').read_text())
        self.vertical_profile=[row['velocity_vertical_mps'] for row in excerpt['samples']]+[0.,0.,0.]
        self.measured_reset_samples=len(excerpt['samples'])
        self.vertical_speed,self.reset_pause_duration,self.reset_pause_injected=0.,.12,False
        self.last_sent,self.fresh_stopped,self.unsafe_shifts=None,0,[]
        a=np.arange(720)*2*np.pi/720
        self.track={key:np.c_[r*np.cos(a),r*np.sin(a)].tolist()
                    for key,r in [('left_lane',90.),('right_lane',110.),('fast_lane',96.)]}
        self.threads=[threading.Thread(target=self.ego_loop,daemon=True),
                      threading.Thread(target=self.management_loop,daemon=True)]

    @property
    def ego_port(self): return self.udp.getsockname()[1]
    @property
    def management_port(self): return self.management.getsockname()[1]

    def start(self):
        for thread in self.threads: thread.start()
        return self

    def pause_telemetry(self, duration_s):
        with self.lock:
            self.pause_until=time.monotonic()+duration_s
            self.telemetry_pauses.append(dict(duration_s=duration_s,brake=self.brake,gas=self.gas))

    def close(self):
        self.stop.set()
        for thread in self.threads: thread.join(2.)
        self.udp.close(); self.management.close()

    def management_loop(self):
        while not self.stop.is_set():
            try: conn,_=self.management.accept()
            except socket.timeout: continue
            with conn:
                name=conn.recv(1024).decode()
                delay=False
                with self.lock:
                    if self.session_kind=='collector' and self.client is None:
                        self.between_session_brakes.append(self.brake)
                        if not self.delayed_management:
                            delay=True;self.delayed_management=True
                    if name=='reset':
                        self.resets+=1;self.position=np.array([100.,0.]);self.yaw=math.pi/2
                        self.speed=0.;self.gas=0.;self.brake=0.;self.reset_transient=len(self.vertical_profile)
                        continue
                    if name=='get_config':
                        data=repr(dict(sampling_freq=100,ego_sampling_freq=25,vjoy_executed_by_server=True))
                    elif name=='get_static_info':
                        data=json.dumps(dict(CarName='ks_ferrari_sf70h',TrackName='ks_barcelona',
                                             TrackConfiguration='layout_gp',autoShifterOn=1))
                    elif name=='get_track_info': data=json.dumps(self.track)
                    else: continue
                if delay:time.sleep(.2)  # old collector releases used to coast during this gap
                conn.sendall(data.encode())

    def ego_loop(self):
        tick=time.monotonic()+.04
        while not self.stop.is_set():
            try:
                raw,address=self.udp.recvfrom(65536)
                with self.lock:
                    if raw==b'connect':
                        if self.client is not None and self.client != address:
                            if self.client in self.retired_clients:self.stale_replacements+=1
                            else:self.overlaps.append([self.client,address])
                        self.client=address;self.seq=0;self.sessions+=1;self.session_kind=None
                        self.last_sent,self.fresh_stopped=None,0
                        self.udp.sendto(b'identified',address)
                    elif raw==b'disconnect':
                        if self.client==address:self.client=None
                    elif self.client==address:
                        command=json.loads(raw)
                        if command['server_steps'] > self.seq: raise AssertionError('future control counter')
                        if self.session_kind is None:
                            self.session_kind='park' if command['brake']>.9 else 'collector'
                            if self.session_kind=='collector':self.collectors+=1
                        self.gas=(command['acc']+1)/2;self.brake=(command['brake']+1)/2
                        self.steer=.3*command['steer']
                        if command.get('enable_gear_shift'):
                            if math.hypot(self.speed,self.vertical_speed)>=.05 or self.fresh_stopped<3:
                                self.unsafe_shifts.append(dict(speed_3d=math.hypot(self.speed,self.vertical_speed),
                                                               fresh_stopped_samples=self.fresh_stopped))
                            self.gear+=int(command.get('shift_up',False))-int(command.get('shift_down',False))
                        self.commands.append(dict(session=self.sessions,kind=self.session_kind,**command))
            except socket.timeout: pass
            now=time.monotonic()
            if now<tick:continue
            tick+=.04
            if tick<now-.04:tick=now+.04
            with self.lock:
                self.stamp+=.04
                if self.reset_transient:
                    index=len(self.vertical_profile)-self.reset_transient
                    if index==self.measured_reset_samples and not self.reset_pause_injected:
                        self.pause_telemetry(self.reset_pause_duration);self.reset_pause_injected=True
                    self.reset_transient-=1
                    self.speed=0.;self.vertical_speed=self.vertical_profile[index]
                    if not self.reset_transient:self.reset_recovered=True
                elif self.brake>=.9:self.speed=max(0.,self.speed-1.)
                elif self.session_kind=='collector':
                    self.speed=max(0.,self.speed+.04*(3*self.gas-5*self.brake))
                else:self.speed=0.
                self.yaw+=self.speed/3.58*math.tan(self.steer)*.04
                self.position+=self.speed*.04*np.array([math.cos(self.yaw),math.sin(self.yaw)])
                if self.client is None:continue
                self.seq+=1
                gear=1 if self.session_kind=='collector' and self.collectors==3 else self.gear
                packet=dict(steps=self.seq,packetId=round(self.stamp*100),timestamp_ac=self.stamp,
                            world_position_x=float(self.position[0]),world_position_y=float(self.position[1]),
                            velocity_z=self.speed*math.cos(self.yaw),velocity_x=self.speed*math.sin(self.yaw),
                            velocity_y=self.vertical_speed,speed=math.hypot(self.speed,self.vertical_speed),
                            yaw=self.yaw,actualGear=gear,accStatus=self.gas,brakeStatus=self.brake,
                            numberOfTyresOut=0,tyre_slip_ratio_rl=0.,tyre_slip_ratio_rr=0.)
                for wheel in ('FL','FR','RL','RR'):
                    angle=self.yaw+(self.steer if wheel in ('FL','FR') else 0.)
                    packet[f'tyreContactHeading_{wheel}_z']=math.cos(angle)
                    packet[f'tyreContactHeading_{wheel}_x']=math.sin(angle)
                self.samples.append(dict(session=self.sessions,kind=self.session_kind,**packet))
                if time.monotonic()<self.pause_until:continue
                normal=(self.last_sent is not None and self.seq==self.last_sent[0]+1
                        and .02<=self.stamp-self.last_sent[1]<=.08 and time.monotonic()-self.last_sent[2]<=.08)
                stopped=packet['speed']<.05 and self.gas<=.01 and self.brake>=.9
                self.fresh_stopped=self.fresh_stopped+1 if normal and stopped else 0
                self.last_sent=(self.seq,self.stamp,time.monotonic())
                try:self.udp.sendto(json.dumps(packet).encode(),self.client)
                except OSError:pass


class FaultProxy:
    """Interrupt pending cycle 1, then discard cycle 2's already committed reply."""
    def __init__(self, backend, token, fake_ac=None, long_hold_loss=False):
        self.backend,self.token=backend,token
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.bind(('127.0.0.1',0));self.socket.listen(8);self.socket.settimeout(.05)
        self.stop,self.pending,self.committed=threading.Event(),threading.Event(),threading.Event()
        self.pending_dropped,self.reply_dropped=False,False
        self.requests=[];self.error=None;self.saved_reply=None
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.fake_ac,self.long_hold_loss,self.jitter_injected=fake_ac,long_hold_loss,False

    @property
    def port(self):return self.socket.getsockname()[1]
    def start(self):self.thread.start();return self
    def close(self):self.stop.set();self.thread.join(5.);self.socket.close()

    def run(self):
        while not self.stop.is_set():
            try:conn,_=self.socket.accept()
            except socket.timeout:continue
            with conn:
                conn.settimeout(30.)
                try:
                    header,payload=d.receive(conn,self.token)
                    self.requests.append(dict(op=header.get('op'),cycle=header.get('cycle'),sha256=d.sha(payload)))
                    if self.fake_ac and not self.jitter_injected:
                        if self.long_hold_loss and header.get('op')=='hello':
                            self.fake_ac.pause_telemetry(.35);self.jitter_injected=True
                        elif header.get('cycle')==4:
                            self.fake_ac.pause_telemetry(.12);self.jitter_injected=True
                    if header.get('cycle')==1 and not self.pending_dropped:
                        self.pending_dropped=True;self.pending.set();continue
                    with socket.create_connection(('127.0.0.1',self.backend),timeout=5.) as target:
                        target.settimeout(30.);d.send(target,self.token,header,payload)
                        response,actor=d.receive(target,self.token)
                    if header.get('cycle')==2:
                        if not self.reply_dropped:
                            self.reply_dropped=True;self.saved_reply=(response,actor);self.committed.set();continue
                        assert (response,actor)==self.saved_reply,'retry changed committed actor or receipt'
                    d.send(conn,self.token,response,actor)
                except OSError:
                    pass  # backend restart intentionally drops this attempt
                except BaseException as exc:
                    self.error=repr(exc);return


def free_tcp_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def stop_process(process):
    if process and process.poll() is None:
        process.terminate()
        try:process.wait(5.)
        except subprocess.TimeoutExpired:process.kill();process.wait(5.)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--actor-bundle',type=Path,default=ROOT/'runs/ac/source_actor')
    args=parser.parse_args(argv)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    token=secrets.token_hex(32);token_file=output/'session.key';token_file.write_text(token)
    calibration=output/'calibration.json'
    calibration.write_text(json.dumps(dict(schema_version=1,validated=True,angle_rad=[-.3,0.,.3],
                            command=[-1.,0.,1.],baseline_angle_rad=0.,car_name='ks_ferrari_sf70h',
                            track_name='ks_barcelona',track_configuration='layout_gp',
                            setup_id='localhost_simulated',sampling_hz=25)))
    server_port=free_tcp_port();fake=FakeAC().start();proxy=FaultProxy(server_port,token,fake).start()
    environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    base=[sys.executable,'-m','rlsetup.assetto.distributed']
    common=['--token-file',str(token_file),'--max-cycles','5','--network-timeout','30']
    server_cmd=base+['server','--bind','127.0.0.1','--port',str(server_port),
                '--actor-bundle',str(args.actor_bundle.resolve()),'--work-dir',str(output/'server'),
                '--critic-updates','1','--updates','1','--batch-size','4','--buffer-size','64']+common
    server_cmd[server_cmd.index('--max-cycles')+1]='6'  # cycle 6 reserved for malformed-input probes
    worker_cmd=base+['worker','--server','127.0.0.1','--port',str(proxy.port),
                '--calibration',str(calibration),'--work-dir',str(output/'worker'),
                '--ac-host','127.0.0.1','--ac-port',str(fake.ego_port),'--management-port',str(fake.management_port),
                '--setup-id','localhost_simulated','--reference','fast_lane','--max-steps','2',
                '--max-wall-s','5','--prepare-timeout','10','--execute']+common
    handles=[];processes=[];extra_proxies=[];server_process=worker_process=None
    def launch(command,name):
        stream=(output/f'{name}.log').open('ab');handles.append(stream)
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        processes.append(process);return process
    def server_ready():
        if server_process.poll() is not None:raise AssertionError('learner server exited; inspect server.log')
        try:
            with socket.create_connection(('127.0.0.1',server_port),timeout=.1):return True
        except OSError:return False
    report=dict(simulation_only=True,real_worker_cli=True,real_server_cli=True,real_gradient_updates=True,
                output=str(output),started_unix_s=time.time(),
                ports=dict(learner_tcp=server_port,worker_proxy_tcp=proxy.port,
                           fake_ac_management_tcp=fake.management_port,fake_ac_ego_udp=fake.ego_port))
    report['source_sha256']={str(path.relative_to(ROOT)):d.sha(path.read_bytes()) for path in
                            (Path(__file__).resolve(),ROOT/'rlsetup/assetto/distributed.py',
                             ROOT/'rlsetup/assetto/driving.py',ROOT/'rlsetup/assetto/learn.py',
                             ROOT/'rlsetup/assetto/actor.py')}
    try:
        server_process=launch(server_cmd,'server');wait_for(server_ready,20.,'server startup timeout')
        worker_process=launch(worker_cmd,'worker')
        wait_for(proxy.pending.is_set,30.,'pending chunk was not reached')
        stop_process(worker_process)
        with fake.lock:
            fake.retired_clients.add(fake.client)  # the old OS process is confirmed exited
        pending=output/'worker/chunk-00000001.npz';pending_hash=d.sha(pending.read_bytes())
        assert not list((output/'server').glob('cycle-*/receipt.json'))
        worker_process=launch(worker_cmd,'worker')
        wait_for(proxy.committed.is_set,30.,'committed reply was not reached')
        receipt2=output/'server/cycle-00000002/receipt.json'
        committed_before=d.sha(receipt2.read_bytes())
        stop_process(server_process)
        server_process=launch(server_cmd,'server');wait_for(server_ready,20.,'server resume timeout')
        try:worker_process.wait(60.)
        except subprocess.TimeoutExpired:raise AssertionError('worker did not finish finite loop')
        assert worker_process.returncode==0,'worker failed; inspect worker.log'
        assert proxy.error is None,proxy.error
        assert d.sha(receipt2.read_bytes())==committed_before
        assert d.sha(pending.read_bytes())==pending_hash
        receipts=[json.loads(path.read_text()) for path in sorted((output/'server').glob('cycle-*/receipt.json'))]
        assert len(receipts)==5
        counts=[entry['response']['transitions'] for entry in receipts]
        trained=[entry['cycle'] for entry in receipts if entry['response']['trained']]
        assert counts==[2,2,0,2,2],counts
        assert trained==[2,5],trained
        source=ActorRuntime(args.actor_bundle)
        previous=source
        changed=[]
        for cycle in trained:
            path=output/f'server/cycle-{cycle:08d}/learner/actor'
            actor=ActorRuntime(path)
            changed.append(any(not np.array_equal(value.numpy(),previous._actor.state_dict()[key].numpy())
                               for key,value in actor._actor.state_dict().items()))
            previous=actor
        assert all(changed),'actor tensors did not change'
        for i,entry in enumerate(receipts,1):
            with np.load(output/f'worker/chunk-{i:08d}.npz',allow_pickle=False) as data:
                metadata=json.loads(data['metadata_json'].item())
            previous=output/f'worker/actor-{i-1:08d}/manifest.json'
            assert metadata['actor_manifest_sha256']==d.sha(previous.read_bytes())
        final=json.loads((output/'server/cycle-00000005/learner/manifest.json').read_text())
        assert final['critic_updates']==2 and final['updates']==2
        assert fake.reset_recovered and not fake.overlaps
        wait_for(lambda:fake.client is None,2.,'final UDP subscriber did not disconnect')
        assert fake.brake==1. and fake.gas==0.,'final controls did not retain full brake'
        assert fake.between_session_brakes and all(value==1. for value in fake.between_session_brakes)
        qualification=[]
        prep_qualification=[]
        for path in (output/'worker').glob('readiness-*.jsonl'):
            qualification.extend(json.loads(line) for line in path.read_text().splitlines()
                                 if 'hold_qualification' in line)
            prep_qualification.extend(json.loads(line) for line in path.read_text().splitlines()
                                      if 'prep_qualification' in line)
        assert proxy.jitter_injected and any(not row['normal_cadence'] for row in qualification)
        assert any(row.get('qualified_samples')==3 for row in qualification)
        assert any(row['phase']=='RESET_SETTLING' and not row['normal_cadence'] for row in prep_qualification)
        assert not fake.unsafe_shifts,'gear command preceded three fresh 3D-stationary samples'
        # Malformed network requests must not kill the listening server.
        probes=[]
        for case in ('unicode_auth','wrong_auth','bad_hash','oversize','corrupt_npz','scalar_obs','metadata_list',
                     'deep_header','deep_metadata'):
            with socket.create_connection(('127.0.0.1',server_port),timeout=2.) as sock:
                sock.settimeout(2.)
                if case=='deep_header':
                    encoded=('{"op":"hello","nested":'+'['*15000+']'*15000+'}').encode()
                    sock.sendall(struct.pack('!I',len(encoded))+encoded)
                elif case=='corrupt_npz':d.send(sock,token,dict(op='chunk',cycle=6),b'not npz')
                elif case in ('scalar_obs','metadata_list','deep_metadata'):
                    with np.load(output/'worker/chunk-00000005.npz',allow_pickle=False) as archive:
                        arrays={key:archive[key] for key in archive.files}
                    if case=='scalar_obs':arrays['obs']=np.array(0.,dtype=np.float32)
                    elif case=='metadata_list':arrays['metadata_json']=np.array('[]')
                    else:arrays['metadata_json']=np.array('{"nested":'+'['*15000+']'*15000+'}')
                    stream=io.BytesIO();np.savez(stream,**arrays)
                    d.send(sock,token,dict(op='chunk',cycle=6),stream.getvalue())
                else:
                    header=dict(protocol=1,token=token,size=0,sha256=d.sha(b''),op='hello')
                    if case=='unicode_auth':header['token']='é'
                    if case=='wrong_auth':header['token']='wrong'
                    if case=='bad_hash':header['sha256']='0'*64
                    if case=='oversize':header['size']=d.MAX_PAYLOAD+1
                    encoded=json.dumps(header).encode();sock.sendall(struct.pack('!I',len(encoded))+encoded)
                try:sock.recv(65536)
                except (OSError,socket.timeout):pass
            with socket.create_connection(('127.0.0.1',server_port),timeout=2.) as sock:
                sock.settimeout(2.);d.send(sock,token,dict(op='hello'));header,_=d.receive(sock,token)
                assert header['ok'] and header['cycle']==5
            probes.append(case)
        # The same real worker must abort a >250 ms hold outage, without collecting another chunk.
        loss_proxy=FaultProxy(server_port,token,fake,long_hold_loss=True).start();extra_proxies.append(loss_proxy)
        loss_cmd=list(worker_cmd)
        for flag,value in [('--port',str(loss_proxy.port)),('--work-dir',str(output/'long-loss-worker')),
                           ('--max-cycles','6')]:
            loss_cmd[loss_cmd.index(flag)+1]=value
        loss_process=launch(loss_cmd,'long-loss-worker')
        loss_process.wait(20.)
        assert loss_process.returncode!=0 and loss_proxy.jitter_injected
        loss_traces=list((output/'long-loss-worker').glob('readiness-*.jsonl'))
        assert loss_traces
        loss_rows=[json.loads(line) for line in loss_traces[-1].read_text().splitlines()]
        assert any(row['type']=='failure' and row['phase']=='READY_HOLD' for row in loss_rows)
        assert not list((output/'long-loss-worker').glob('chunk-*.npz'))
        wait_for(lambda:fake.client is None,2.,'long-loss worker subscriber did not disconnect')
        assert fake.brake==1. and fake.gas==0.
        assert len(list((output/'server').glob('cycle-*/receipt.json')))==5
        # A second preparation attempt loses telemetry while the reset is still settling.
        with fake.lock:
            fake.reset_pause_duration=.35;fake.reset_pause_injected=False
        reset_proxy=FaultProxy(server_port,token,fake).start();extra_proxies.append(reset_proxy)
        reset_cmd=list(loss_cmd)
        reset_cmd[reset_cmd.index('--port')+1]=str(reset_proxy.port)
        reset_cmd[reset_cmd.index('--work-dir')+1]=str(output/'long-reset-loss-worker')
        reset_process=launch(reset_cmd,'long-reset-loss-worker');reset_process.wait(20.)
        assert reset_process.returncode!=0
        reset_traces=list((output/'long-reset-loss-worker').glob('readiness-*.jsonl'))
        assert reset_traces
        reset_rows=[json.loads(line) for line in reset_traces[-1].read_text().splitlines()]
        assert any(row['type']=='failure' and row['phase']=='RESET_SETTLING' for row in reset_rows)
        assert not list((output/'long-reset-loss-worker').glob('chunk-*.npz'))
        wait_for(lambda:fake.client is None,2.,'reset-loss worker subscriber did not disconnect')
        assert fake.brake==1. and fake.gas==0.
        assert len(list((output/'server').glob('cycle-*/receipt.json')))==5
        report.update(success=True,transition_counts=counts,trained_cycles=trained,
                      actor_tensors_changed=changed,critic_updates=final['critic_updates'],updates=final['updates'],
                      worker_restarted_with_pending=True,learner_restarted_after_committed_reply_loss=True,
                      duplicate_receipt_unchanged=True,single_udp_owner=True,final_full_brake=True,
                      full_brake_across_delayed_management_gap=True,
                      ready_hold_120ms_pause_recovered=True,release_recognition_recovers_three_samples=True,
                      ready_hold_350ms_loss_aborted=True,
                      reset_settling_120ms_pause_recovered=True,reset_settling_350ms_loss_aborted=True,
                      vertical_bounce_prevents_early_gears=True,
                      reset_transient_recovered=True,malformed_requests_survived=probes,
                      proxy_requests=proxy.requests,udp_sessions=fake.sessions)
        report['processes']=[dict(pid=process.pid,role='learner' if 'server' in process.args else 'worker',
                                  returncode=process.poll()) for process in processes]
        report['stale_subscription_replacements_after_confirmed_worker_exit']=fake.stale_replacements
    except BaseException as exc:
        report.update(success=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for process in processes:stop_process(process)
        report['processes']=[dict(pid=process.pid,role='learner' if 'server' in process.args else 'worker',
                                  returncode=process.poll()) for process in processes]
        proxy.close();fake.close()
        for proxy in extra_proxies:proxy.close()
        for handle in handles:handle.close()
        (output/'fake-ac.json').write_text(json.dumps(dict(commands=fake.commands,samples=fake.samples,
                                            overlaps=fake.overlaps,resets=fake.resets),indent=2))
        report['elapsed_s']=time.time()-report['started_unix_s']
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
