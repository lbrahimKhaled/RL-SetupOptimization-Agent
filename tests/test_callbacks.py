"""D2 §10: CollapseGuard, PhaseCallback, WallClockLimit, TrainStats / Eval callbacks, metric
helpers, run folders, training provenance."""
import dataclasses
import json
import math
import time
import numpy as np
import pytest
import torch as th
from rlsetup import runs
from rlsetup.driving import evaluate as ev
from rlsetup.driving.callbacks import (CollapseGuard, PhaseCallback, WallClockLimit, TrainStatsCallback,
                                       DrivingEvalCallback, CheckpointCallback, snapshot_state)
from rlsetup.driving.train_config import GuardConfig, EvalConfig
from sb3_helpers import make_model


def start(cb, model):
    cb.init_callback(model)
    cb.on_training_start({}, {})


def step(cb, model, n, **locals_):
    model.num_timesteps = n
    cb.update_locals(locals_)
    return cb.on_step()


def state_equal(a, b):
    return all(th.equal(a[k], b[k]) for k in a)


def opt_equal(a, b):
    sa, sb = a['state'], b['state']
    return sa.keys() == sb.keys() and all(
        all(th.equal(sa[i][k], sb[i][k]) if th.is_tensor(sa[i][k]) else sa[i][k] == sb[i][k] for k in sa[i])
        for i in sa)


# --- CollapseGuard ------------------------------------------------------------------------------
def test_collapse_guard_rollback_then_stop():
    model = make_model()
    values = iter([100., 10., 100., 10.])
    calls, events = [], []
    guard = CollapseGuard(lambda m: calls.append(m.num_timesteps) or next(values), baseline=100., cfg=GuardConfig(),
                          on_event=lambda kind, **d: events.append((kind, d)))
    start(guard, model)
    assert guard.start_step == 0 and guard.snapshot is not None
    model.train(2, 64)
    assert step(guard, model, 9_996)                       # not yet a probe step
    assert calls == []
    assert step(guard, model, 10_000)                      # 100 >= 0.5 x 100: new last-good snapshot
    good = snapshot_state(model)
    rb_pos = model.replay_buffer.pos
    model.train(3, 64)                                     # drift away from the good state
    assert not state_equal(good['policy'], model.policy.state_dict())
    assert step(guard, model, 20_000)                      # 10 < 50: rollback
    assert events[0][0] == 'rollback' and events[0][1]['step'] == 20_000
    assert state_equal(good['policy'], model.policy.state_dict())
    assert opt_equal(good['actor_opt'], model.actor.optimizer.state_dict())
    assert opt_equal(good['critic_opt'], model.critic.optimizer.state_dict())
    assert opt_equal(good['ent_opt'], model.ent_coef_optimizer.state_dict())
    assert th.equal(good['log_ent_coef'], model.log_ent_coef.detach())
    assert model.actor_lr == 1e-4 and guard.rollbacks == 1
    assert model.replay_buffer.pos == rb_pos              # buffer never rolled back
    assert step(guard, model, 30_000)                      # recovered
    assert step(guard, model, 40_000) is False             # second collapse stops
    assert guard.collapsed and events[-1][0] == 'collapsed'
    assert calls == [10_000, 20_000, 30_000, 40_000]


def test_collapse_guard_probe_at_threshold_is_ok():
    """QA mutant C1: mean probe progress exactly threshold x baseline is not a collapse."""
    model = make_model(fill=10)
    events = []
    guard = CollapseGuard(lambda m: 50., baseline=100., cfg=GuardConfig(threshold=.5),
                          on_event=lambda kind, **d: events.append(kind))
    start(guard, model)
    assert step(guard, model, 10_000) and step(guard, model, 20_000)
    assert events == [] and guard.rollbacks == 0 and all(h['ok'] for h in guard.history)


def test_collapse_guard_waits_for_unfreeze_and_stops_after_window():
    model = make_model(fill=10)
    model.set_frozen(True, True)
    calls = []
    guard = CollapseGuard(lambda m: calls.append(m.num_timesteps) or 0., baseline=1., cfg=GuardConfig(window=30_000))
    start(guard, model)
    assert step(guard, model, 10_000) and guard.snapshot is None   # frozen: inactive
    model.set_frozen(False, False)
    assert step(guard, model, 12_000) and guard.start_step == 12_000
    assert step(guard, model, 21_996) and calls == []
    assert step(guard, model, 22_000) is True and calls == [22_000]   # first collapse -> rollback
    assert step(guard, model, 32_000) is False                       # second -> stop
    guard2 = CollapseGuard(lambda m: calls.append(-1) or 0., baseline=1., cfg=GuardConfig(window=30_000))
    start(guard2, model)
    assert step(guard2, model, 31_000 + model.num_timesteps)          # past the window: no probe
    assert -1 not in calls


# --- Phase / wall clock ---------------------------------------------------------------------------
def test_phase_callback_unfreezes_at_step():
    model = make_model(fill=0)
    model.set_frozen(True, True)
    events = []
    cb = PhaseCallback(20_000, on_event=lambda k, **d: events.append((k, d)))
    start(cb, model)
    assert step(cb, model, 19_996) and model.freeze_actor
    assert all(not p.requires_grad for p in model.actor.parameters())
    assert step(cb, model, 20_000) and not model.freeze_actor and not model.freeze_ent_coef
    assert all(p.requires_grad for p in model.actor.parameters())
    from rlsetup.driving.agent import param_digest
    assert events == [('unfreeze', {'step': 20_000, 'actor_sha256': param_digest(model.actor),     # R1 A4
                                    'log_ent_coef_sha256': param_digest(model.log_ent_coef)})]
    assert cb.unfrozen_at == 20_000


def test_wall_clock_limit():
    model = make_model(fill=0)
    cb = WallClockLimit(1 / 3600, t0=time.time() - .5)
    start(cb, model)
    assert step(cb, model, 4)
    cb.t0 = time.time() - 2.
    assert step(cb, model, 8) is False and cb.triggered
    none = WallClockLimit(None)
    start(none, model)
    assert step(none, model, 12)


# --- TrainStats / Eval / Checkpoint callbacks ------------------------------------------------------
def test_train_stats_window(tmp_path):
    from rlsetup.driving.vec import base_venv
    model = make_model(fill=0)
    venv = base_venv(model.get_env())
    venv.reset_infos = [{'start': {'v0': 4.}}]
    cb = TrainStatsCallback(tmp_path / 'ts.jsonl', window=8)
    start(cb, model)
    info = {'termination': 'stall', 'truncation': None, 'speed': 2., 'e_ref': -.5, 'episode': {'r': -40., 'l': 60}}
    venv.reset_infos = [{'start': {'v0': 20.}}]
    step(cb, model, 4, infos=[info], dones=np.array([True]))
    info2 = {'termination': 'off_track', 'truncation': None, 'speed': 30., 'e_ref': 4., 'episode': {'r': 10., 'l': 90}}
    step(cb, model, 8, infos=[info2], dones=np.array([True]))
    rec = runs.read_jsonl(tmp_path / 'ts.jsonl')
    assert len(rec) == 1
    r = rec[0]
    assert r['step'] == 8 and r['episodes'] == 2 and r['causes'] == {'stall': 1, 'off_track': 1}
    assert r['fail_speed_mean'] == 16. and r['fail_abs_e_mean'] == 2.25 and r['fail_share_slow_or_wide'] == 1.
    assert r['low_v0_episodes'] == 1 and r['low_v0_stalls'] == 1        # second episode started at 20 m/s
    assert r['ep_return_mean'] == -15. and r['ep_length_mean'] == 75.
    assert cb.totals == {'stall': 1, 'off_track': 1}
    # QA m6: thresholds from EvalConfig
    e = EvalConfig()
    assert (e.stats_slow_mps, e.stats_wide_m, e.stats_low_v0_mps) == (15., 3., 5.)
    cb2 = TrainStatsCallback(tmp_path / 'ts2.jsonl', window=8, slow_mps=1., wide_m=5., low_v0_mps=3.)
    venv.reset_infos = [{'start': {'v0': 4.}}]
    model.num_timesteps = 0
    start(cb2, model)
    step(cb2, model, 4, infos=[info], dones=np.array([True]))
    step(cb2, model, 8, infos=[info2], dones=np.array([True]))
    r2 = runs.read_jsonl(tmp_path / 'ts2.jsonl')[0]
    assert r2['fail_share_slow_or_wide'] == 0. and r2['low_v0_episodes'] == 0


def test_eval_callback_best_and_regression(tmp_path):
    model = make_model(fill=10)
    results = iter([{'det_clean': True, 'lap2_s': 150.}, {'det_clean': True, 'lap2_s': 145.},
                    {'det_clean': False, 'lap2_s': None, 'det_termination': 'off_track'},
                    {'det_clean': True, 'lap2_s': 146.}])
    events = []
    cb = DrivingEvalCallback(lambda m, n: dict(next(results)), 100, tmp_path / 'evals.jsonl', tmp_path,
                             on_event=lambda k, **d: events.append(k))
    start(cb, model)                                      # baseline eval at step 0
    assert (tmp_path / 'best.zip').exists() and (tmp_path / 'last.zip').exists()
    best0 = (tmp_path / 'best.zip').stat().st_mtime_ns
    step(cb, model, 100)
    step(cb, model, 200)
    step(cb, model, 250)                                  # not an eval step
    cb.on_training_end()                                  # final eval at 250
    rec = runs.read_jsonl(tmp_path / 'evals.jsonl')
    assert [r['step'] for r in rec] == [0, 100, 200, 250]
    assert [r.get('new_best', False) for r in rec] == [True, True, False, False]
    assert cb.best_lap2 == 145. and (tmp_path / 'best.zip').stat().st_mtime_ns > best0
    assert events == ['det_regression']
    assert all(k in rec[1] for k in ('env_steps_per_s', 'wall_train_s', 'wall_eval_s', 'train_obs_clip_fraction'))


def test_checkpoint_callback(tmp_path):
    model = make_model(fill=0)
    cb = CheckpointCallback(100, tmp_path)
    start(cb, model)
    for n in (52, 104, 156, 208):
        step(cb, model, n)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['last.zip', 'step_104.zip', 'step_208.zip']


def test_checkpoint_callback_saves_replay_buffer(tmp_path):
    from stable_baselines3.common.save_util import load_from_pkl
    from rlsetup.driving.train_config import TrainConfig
    model = make_model(fill=300)
    saved = []
    buf = tmp_path / 'replay_buffer.pkl'
    cb = CheckpointCallback(100, tmp_path / 'ck', buffer_path=buf, buffer_every=50, on_buffer_saved=saved.append)
    (tmp_path / 'ck').mkdir()
    start(cb, model)
    for n in (20, 52, 76, 104, 130):
        step(cb, model, n)
    assert saved == [52, 104]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ck', 'replay_buffer.pkl']      # no .tmp left over
    assert sorted(p.name for p in (tmp_path / 'ck').iterdir()) == ['last.zip', 'step_104.zip']
    rb = load_from_pkl(buf)
    assert rb.pos == model.replay_buffer.pos == 300
    np.testing.assert_array_equal(rb.observations[:300], model.replay_buffer.observations[:300])
    # resumed at 104: the next buffer save is at 150, not immediately
    cb2 = CheckpointCallback(100, tmp_path / 'ck', buffer_path=buf, buffer_every=50, on_buffer_saved=saved.append)
    model.num_timesteps = 104
    start(cb2, model)
    step(cb2, model, 120)
    step(cb2, model, 150)
    assert saved == [52, 104, 150]
    # no buffer path / interval 0: never
    cb3 = CheckpointCallback(100, tmp_path / 'ck', buffer_path=buf, buffer_every=0, on_buffer_saved=saved.append)
    start(cb3, model)
    step(cb3, model, 1000)
    assert saved == [52, 104, 150]
    # TrainConfig: None = checkpoint_interval, explicit value, disabled with save_replay_buffer=False
    c = TrainConfig()
    assert c.replay_buffer_save_every is None and c.buffer_save_interval == c.checkpoint_interval == 100_000
    assert dataclasses.replace(c, replay_buffer_save_every=30_000).buffer_save_interval == 30_000
    assert dataclasses.replace(c, replay_buffer_save_every=0).buffer_save_interval == 0
    assert dataclasses.replace(c, save_replay_buffer=False).buffer_save_interval == 0


CODE = {'rlsetup': {'commit': 'a' * 40, 'dirty': False, 'diff_sha256': 'e' * 64}, 'ocd_lap_code_version': 'c' * 12}


def resume_fixture(tmp_path, monkeypatch, *, code=CODE, status=None, fill=100, cfg=None):
    """A run folder (tiny config, preset 'full') with last.zip at step 5,000 and provenance.json
    recording CODE; heavy pieces (track, QSS, vec / eval envs, learn) and code_state stubbed.
    Returns (run dir, model, seen): seen collects the model handed to _learn_and_finish and the
    _callbacks kwargs. `cfg`: TrainConfig to store (default tiny_cfg())."""
    import ocd_lap.track
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rlsetup import provenance
    from rlsetup.driving import train as T, qss_profile, vec
    from rlsetup.driving.train_config import config_json
    from sb3_helpers import TinyEnv, tiny_cfg
    cfg = cfg or tiny_cfg()
    model = make_model(cfg, fill=fill)
    model.num_timesteps = 5_000
    d = runs.make_run_dir(tmp_path, 'r', 0)
    model.save(d / 'checkpoints' / 'last.zip')
    (d / 'config.json').write_text(config_json(cfg))
    runs.write_json(d / 'provenance.json', CODE)
    runs.update_status(d, **(status or dict(phase='rl', state='running', bc_baseline_m=50., unfrozen_at=1_000,
                                           rollbacks=0)))
    runs.write_json(d / 'probe_starts.json', [])
    monkeypatch.setattr(ocd_lap.track, 'Track', lambda name: None)
    monkeypatch.setattr(provenance, 'code_state', lambda track=None: code)
    monkeypatch.setattr(qss_profile, 'load_or_build', lambda track, **kw: None)
    monkeypatch.setattr(T, 'eval_envs', lambda *a: {})
    monkeypatch.setattr(vec, 'build_vec_env', lambda *a, **k: DummyVecEnv([TinyEnv]))
    seen = {}

    def fake_callbacks(*a, **k):
        seen['callbacks'] = k
        return [], None, None, None
    monkeypatch.setattr(T, '_callbacks', fake_callbacks)

    def fake_finish(run, m, cbs, guard, wall, plateau, total):
        seen.update(model=m, size=m.replay_buffer.size(), step=m.num_timesteps, total=total)
        return T.EXIT_OK
    monkeypatch.setattr(T, '_learn_and_finish', fake_finish)
    return d, model, seen


def status_of(d):
    return json.loads((d / 'status.json').read_text())


def test_resume_without_replay_buffer_warns(tmp_path, monkeypatch):
    """Hard kill before the first buffer save: resume continues with an empty buffer, a warning
    in status.json and no gradient updates until batch_size x 10 transitions are stored."""
    from rlsetup.driving import train as T
    d, model, seen = resume_fixture(tmp_path, monkeypatch)
    assert T.resume_pipeline(d, ['total_timesteps=6000'], argv=['x']) == T.EXIT_OK
    assert {k: seen[k] for k in ('size', 'step', 'total')} == {'size': 0, 'step': 5_000, 'total': 1_000}
    assert seen['model'].min_train_transitions == 512 * 10
    s = status_of(d)
    assert len(s['warnings']) == 1 and s['warnings'][0]['step'] == 5_000
    assert 'empty replay buffer' in s['warnings'][0]['message']
    kinds = [e['kind'] for e in s['events']]
    assert kinds == ['resume_without_replay_buffer', 'resume'] and s['events'][1]['replay_buffer'] is False
    assert s['events'][1]['min_train_transitions'] == 5_120 and s['events'][1]['code'] == CODE
    assert s['events'][1]['code_mismatches'] == [] and s['events'][1]['allow_code_change'] is False
    # with the buffer present (saved at the last.zip step): loaded, cut, no new warning
    model.save_replay_buffer(d / 'replay_buffer.pkl')
    runs.update_status(d, replay_buffer_step=5_000)
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_OK
    assert seen['size'] == 100 and seen['model'].min_train_transitions == 0
    rb = seen['model'].replay_buffer
    assert rb.dones[99, 0] == 1 and rb.timeouts[99, 0] == 1                 # m1: resume cut
    s = status_of(d)
    ev = s['events'][-1]
    assert len(s['warnings']) == 1 and ev['replay_buffer'] is True and ev['replay_buffer_cut_row'] == 99
    assert ev['replay_buffer_lag'] is False and seen['callbacks']['buffer_saved_step'] == 5_000
    # no last.zip: FileNotFoundError
    (d / 'checkpoints' / 'last.zip').unlink()
    with pytest.raises(FileNotFoundError):
        T.resume_pipeline(d, [], argv=['x'])


def test_resume_during_frozen_warmup_stays_frozen(tmp_path, monkeypatch):
    """QA mutant T3: a seed killed during the online warm-up (actor + alpha frozen, no unfreeze
    yet) resumes frozen and unfreezes at warmup.online_warmup_steps; after the unfreeze the
    resumed guard starts at the recorded unfreeze step."""
    from rlsetup.driving import train as T
    d, model, seen = resume_fixture(tmp_path, monkeypatch, status=dict(phase='online_warmup', state='running',
                                                                       bc_baseline_m=50., rollbacks=0))
    model.set_frozen(actor=True, ent_coef=True)
    model.save(d / 'checkpoints' / 'last.zip')
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_OK
    assert seen['callbacks']['unfreeze_at'] == 20_000 and seen['callbacks']['guard_start'] is None
    assert seen['model'].freeze_actor and seen['model'].freeze_ent_coef
    assert not any(p.requires_grad for p in seen['model'].actor.parameters())
    runs.update_status(d, unfrozen_at=1_000)
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_OK
    assert seen['callbacks']['unfreeze_at'] == 0 and seen['callbacks']['guard_start'] == 1_000


def test_resume_warns_when_buffer_older_than_last_zip(tmp_path, monkeypatch):
    """QA m4: last.zip (every eval) newer than replay_buffer.pkl by more than one vector step."""
    from rlsetup.driving import train as T
    d, model, seen = resume_fixture(tmp_path, monkeypatch)
    model.save_replay_buffer(d / 'replay_buffer.pkl')
    runs.update_status(d, replay_buffer_step=4_999)                 # n - n_envs (n_envs = 1): fine
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_OK
    assert 'warnings' not in status_of(d) and status_of(d)['events'][-1]['replay_buffer_lag'] is False
    runs.update_status(d, replay_buffer_step=4_998)
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_OK
    s = status_of(d)
    assert s['events'][-1]['replay_buffer_lag'] is True and len(s['warnings']) == 1
    assert 'from step 4998 but last.zip from step 5000' in s['warnings'][0]['message']
    assert seen['callbacks']['buffer_saved_step'] is None


def test_resume_refuses_changed_or_dirty_code(tmp_path, monkeypatch):
    """QA M2: resume compares the code with provenance.json and refuses a dirty full preset;
    the override flags are recorded."""
    from rlsetup.driving import train as T
    changed = {'rlsetup': dict(CODE['rlsetup'], commit='b' * 40), 'ocd_lap_code_version': 'd' * 12}
    d, model, seen = resume_fixture(tmp_path, monkeypatch, code=changed)
    cfg0 = (d / 'config.json').read_text()
    assert T.resume_pipeline(d, ['total_timesteps=9000'], argv=['x']) == T.EXIT_CODE_CHANGED == 5
    s = status_of(d)
    assert s['events'][-1]['kind'] == 'resume_refused' and 'model' not in seen
    assert 'rlsetup commit' in s['events'][-1]['reasons'][0] and 'ocd_lap_code_version' in s['events'][-1]['reasons'][0]
    assert s['state'] == 'running' and (d / 'config.json').read_text() == cfg0   # nothing else changed
    assert T.resume_pipeline(d, [], argv=['x'], allow_code_change=True) == T.EXIT_OK
    ev = status_of(d)['events'][-1]
    assert ev['kind'] == 'resume' and ev['allow_code_change'] is True and len(ev['code_mismatches']) == 2
    assert ev['code'] == changed and any('different code' in w['message'] for w in status_of(d)['warnings'])
    # dirty tree with the full preset
    dirty = {'rlsetup': dict(CODE['rlsetup'], dirty=True), 'ocd_lap_code_version': CODE['ocd_lap_code_version']}
    monkeypatch.setattr('rlsetup.provenance.code_state', lambda track=None: dirty)
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_DIRTY
    assert T.resume_pipeline(d, [], argv=['x'], allow_dirty=True) == T.EXIT_OK
    assert status_of(d)['events'][-1]['allow_dirty'] is True
    # missing provenance.json counts as a code change
    (d / 'provenance.json').unlink()
    monkeypatch.setattr('rlsetup.provenance.code_state', lambda track=None: CODE)
    assert T.resume_pipeline(d, [], argv=['x']) == T.EXIT_CODE_CHANGED
    # CLI flags reach resume_pipeline
    args = T.parse_args(['--resume', '--run-name', 'r', '--allow-code-change', '--allow-dirty'])
    assert args.allow_code_change and args.allow_dirty


def test_code_mismatches():
    from rlsetup.provenance import code_mismatches
    assert code_mismatches(CODE, CODE) == []
    other = {'rlsetup': dict(CODE['rlsetup'], diff_sha256='f' * 64), 'ocd_lap_code_version': CODE['ocd_lap_code_version']}
    assert code_mismatches(CODE, other) == [f"rlsetup diff_sha256: run {'e' * 64} != now {'f' * 64}"]
    assert len(code_mismatches({}, CODE)) == 3


def test_resume_cut_stops_nstep_chains(tmp_path):
    """QA m1: after load + cut, n-step samples never reach from the saved transitions into the
    episodes the resumed env starts (QA measured 69/12,800 crossing samples without the cut)."""
    import itertools
    import gymnasium as gym
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rlsetup.driving.agent import StagedSAC, build_model
    from rlsetup.driving.callbacks import load_replay_buffer_for_resume, cut_nstep_chains
    from sb3_helpers import tiny_cfg
    ids = itertools.count(1)

    class EpEnv(gym.Env):
        """obs[0] = episode id (new on every reset), obs[1] = step; episodes never end."""
        observation_space = gym.spaces.Box(-1e6, 1e6, (159,), np.float32)
        action_space = gym.spaces.Box(-1., 1., (3,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.id, self.t = next(ids), 0
            return self._o(), {}

        def _o(self):
            o = np.zeros(159, np.float32)
            o[:2] = self.id, self.t
            return o

        def step(self, a):
            self.t += 1
            return self._o(), 1., False, False, {}
    cfg = dataclasses.replace(tiny_cfg(), gradient_steps=0)
    m = build_model(cfg, DummyVecEnv([EpEnv]))
    m.set_logger(configure(None, []))
    m.learn(200)
    m.save(tmp_path / 'm.zip')
    m.save_replay_buffer(tmp_path / 'rb.pkl')
    l = StagedSAC.load(tmp_path / 'm.zip', env=DummyVecEnv([EpEnv]), device='cpu')
    assert load_replay_buffer_for_resume(l, tmp_path / 'rb.pkl') == 199
    l.set_logger(configure(None, []))
    l.learn(200, reset_num_timesteps=False)                      # as resume_pipeline
    rb = l.replay_buffer
    assert rb.pos == 400 and rb.observations[199, 0, 0] != rb.observations[200, 0, 0]
    smp = rb._get_samples(np.arange(400))
    o, n = smp.observations.numpy(), smp.next_observations.numpy()
    np.testing.assert_array_equal(o[:, 0], n[:, 0])              # never across the cut
    assert smp.dones.numpy()[199, 0] == 0 and smp.discounts.numpy()[199, 0] == pytest.approx(.992)  # bootstrapped
    # a true termination at the cut stays terminal; an empty buffer is untouched; wrap-around
    rb2 = make_model(fill=10).replay_buffer
    rb2.dones[9], rb2.timeouts[9] = 1., 0.
    assert cut_nstep_chains(rb2) == 9 and rb2.dones[9, 0] == 1 and rb2.timeouts[9, 0] == 0
    assert cut_nstep_chains(make_model(fill=0).replay_buffer) is None
    rb3 = make_model(fill=10).replay_buffer
    rb3.pos, rb3.full = 0, True
    assert cut_nstep_chains(rb3) == rb3.buffer_size - 1 and rb3.timeouts[-1, 0] == 1


def test_train_skipped_until_buffer_refilled():
    """Empty-buffer resume: no gradient step while the buffer holds < min_train_transitions."""
    model = make_model(fill=100)
    model.min_train_transitions = 128
    critic0 = {k: v.clone() for k, v in model.critic.state_dict().items()}
    model.train(2, 64)
    assert model._n_updates == 0 and state_equal(critic0, model.critic.state_dict())
    assert model.buffer_transitions() == 100
    rng = np.random.default_rng(1)
    for _ in range(28):
        model.replay_buffer.add(rng.uniform(-1, 1, (1, 159)).astype(np.float32),
                                rng.uniform(-1, 1, (1, 159)).astype(np.float32),
                                np.zeros((1, 3), np.float32), np.zeros(1, np.float32), np.array([False]), [{}])
    model.train(2, 64)
    assert model._n_updates == 2


def test_rollback_saves_last_zip_atomically(tmp_path):
    """QA m3: the rolled-back model (and its rollback actor_lr) is on disk before the event."""
    from rlsetup.driving.agent import StagedSAC
    model = make_model()
    values = iter([100., 10.])
    order = []
    last = tmp_path / 'last.zip'
    guard = CollapseGuard(lambda m: next(values), baseline=100., cfg=GuardConfig(), last_path=last,
                          on_event=lambda kind, **d: order.append((kind, last.exists())))
    start(guard, model)
    step(guard, model, 10_000)                            # ok: new snapshot, nothing saved
    assert not last.exists()
    good = snapshot_state(model)
    model.train(3, 64)
    step(guard, model, 20_000)                            # rollback
    assert order == [('rollback', True)] and not list(tmp_path.glob('*.tmp'))
    loaded = StagedSAC.load(last, device='cpu')
    assert loaded.actor_lr == 1e-4 and state_equal(good['policy'], loaded.policy.state_dict())


def test_eval_saves_buffer_once_per_step(tmp_path):
    """QA m4: the replay buffer is saved with last.zip at every evaluation; a checkpoint at the
    same step does not save it twice; the step of an earlier save (prefill) is skipped."""
    from rlsetup.driving.callbacks import ReplayBufferSaver
    model = make_model(fill=50)
    saved = []
    saver = ReplayBufferSaver(tmp_path / 'replay_buffer.pkl', saved.append, last_step=0)
    ev_cb = DrivingEvalCallback(lambda m, n: {'det_clean': False, 'lap2_s': None}, 100, tmp_path / 'evals.jsonl',
                                tmp_path, buffer_saver=saver)
    ck = CheckpointCallback(200, tmp_path, buffer_every=200, buffer_saver=saver)
    start(ev_cb, model)                                   # step-0 eval: buffer already saved at 0
    start(ck, model)
    assert saved == [] and (tmp_path / 'last.zip').exists()
    for n in (100, 200, 300):
        step(ev_cb, model, n)
        step(ck, model, n)
    assert saved == [100, 200, 300] and (tmp_path / 'replay_buffer.pkl').exists()


# --- metric helpers -----------------------------------------------------------------------------------
def test_soft_mc_exact():
    rng = np.random.default_rng(0)
    r, lp = rng.normal(size=50), rng.normal(size=50)
    g, a = .9, .3
    mc = ev.soft_mc_returns(r, lp, g, a)
    for k in (0, 7, 49):
        brute = sum(g ** j * r[k + j] for j in range(50 - k)) - a * sum(g ** j * lp[k + j] for j in range(1, 50 - k))
        assert mc[k] == pytest.approx(brute, rel=1e-12)
    np.testing.assert_array_equal(ev.mc_indices(1250, False), [0, 100, 200, 300, 400, 500, 600])
    np.testing.assert_array_equal(ev.mc_indices(250, True), [0, 100, 200])
    assert len(ev.mc_indices(599, False)) == 0


def sb3_nstep_fixed_point(rewards, logp, gamma, alpha, n_steps):
    """Brute force: Q on a deterministic chain s_0 .. s_{T-1} (s_{T-1} terminal) that SB3 2.9.0's
    n-step critic target maps to itself. The targets come from a real NStepReplayBuffer
    (`_get_samples`: n-step reward sum, discount gamma^m, done flag, index of next_obs) and
    SAC.train's backup y = R + (1 - d) disc (Q(s') - alpha log pi(a'|s')), with a' = the chain's
    own next action (log pi = logp of that state). Iterated until nothing changes."""
    import gymnasium as gym
    from stable_baselines3.common.buffers import NStepReplayBuffer
    T = len(rewards)
    rb = NStepReplayBuffer(T + 8, gym.spaces.Box(-1e6, 1e6, (1,), np.float32),
                           gym.spaces.Box(-1, 1, (1,), np.float32), device='cpu', n_envs=1,
                           n_steps=n_steps, gamma=gamma)
    for k in range(T):
        rb.add(np.array([[k]], np.float32), np.array([[k + 1]], np.float32), np.zeros((1, 1), np.float32),
               np.array([rewards[k]], np.float32), np.array([k == T - 1]), [{}])
    smp = rb._get_samples(np.arange(T))
    R, disc = smp.rewards.numpy()[:, 0].astype(float), smp.discounts.numpy()[:, 0].astype(float)
    done, nxt = smp.dones.numpy()[:, 0], smp.next_observations.numpy()[:, 0].astype(int)
    lp = np.r_[logp, 0.]                                          # s_T (after the terminal) unused
    q = np.zeros(T + 1)
    for _ in range(T + 2):
        new = q.copy()
        new[:T] = R + (1. - done) * disc * (q[nxt] - alpha * lp[nxt])
        if np.array_equal(new, q):
            break
        q = new
    return q[:T]


def test_nstep_soft_mc_is_sb3_fixed_point():
    """QA D2 M1: MC^n (entropy only at offsets n, 2n, ...) equals the fixed point of SB3's n-step
    target; the per-step soft MC does not (by about alpha E[log pi] (g/(1-g) - g^n/(1-g^n)))."""
    rng = np.random.default_rng(3)
    T, g, a = 50, .9, .3                                          # T not a multiple of n
    r, lp = rng.normal(1., .5, T), rng.normal(-2., 1., T)
    for n in (1, 2, 3):
        fp = sb3_nstep_fixed_point(r, lp, g, a, n)
        np.testing.assert_allclose(ev.soft_mc_returns(r, lp, g, a, n_steps=n), fp, rtol=1e-5, atol=1e-5)
        brute = [sum(g ** j * r[k + j] for j in range(T - k)) - a * sum(g ** j * lp[k + j] for j in range(n, T - k, n))
                 for k in range(T)]
        np.testing.assert_allclose(ev.soft_mc_returns(r, lp, g, a, n_steps=n), brute, rtol=1e-12)
    per_step = ev.soft_mc_returns(r, lp, g, a)
    fp3 = sb3_nstep_fixed_point(r, lp, g, a, 3)
    np.testing.assert_allclose(per_step, sb3_nstep_fixed_point(r, lp, g, a, 1), rtol=1e-5, atol=1e-5)
    assert np.abs(per_step - fp3)[:30].min() > .5                # far from the end: clearly biased
    # constant log pi far from the episode end: offset = alpha |log pi| (g/(1-g) - g^3/(1-g^3))
    c = np.full(2000, -2.)
    off = ev.soft_mc_returns(np.zeros(2000), c, g, a)[0] - ev.soft_mc_returns(np.zeros(2000), c, g, a, 3)[0]
    assert off == pytest.approx(a * 2. * (g / (1 - g) - g ** 3 / (1 - g ** 3)), rel=1e-9)
    with pytest.raises(ValueError):
        ev.soft_mc_returns(r, lp, g, a, n_steps=0)


def test_q_vs_mc_with_fake_critic():
    class Critic:
        def __call__(self, obs, act):
            return obs[:, :1] * 2., obs[:, :1] * 2. + 1.      # Q1 = 2 o, Q2 = 2 o + 1
    n = 700
    obs = np.arange(n, dtype=np.float32)[:, None].repeat(3, 1)
    r, lp = np.ones(n), np.zeros(n)
    out = ev.q_vs_mc(Critic(), obs, np.zeros((n, 3)), r, lp, gamma=.5, alpha=.1, failed=False)
    np.testing.assert_array_equal(out['k'], [0, 100])
    np.testing.assert_allclose(out['q_min'], [0., 200.])
    np.testing.assert_allclose(out['q_mean'], [.5, 200.5])
    np.testing.assert_allclose(out['mc'], [2., 2.], rtol=1e-12)
    np.testing.assert_allclose(out['mc_nstep'], [2., 2.], rtol=1e-12)          # log pi = 0: identical
    s = ev.summarize_q_mc([out])
    assert s['n'] == 2 and s['q_min_minus_mc'] == pytest.approx(98.) and s['q_min_ratio'] == pytest.approx(50.)
    assert s['n_steps'] == 1 and s['q_min_minus_mc_nstep'] == pytest.approx(98.)
    lp = np.full(n, -1.)                                     # entropy bonus: n-step MC gets less of it
    o3 = ev.q_vs_mc(Critic(), obs, np.zeros((n, 3)), r, lp, gamma=.5, alpha=.1, failed=False, n_steps=3)
    np.testing.assert_allclose(o3['mc'], ev.soft_mc_returns(r, lp, .5, .1)[[0, 100]])
    np.testing.assert_allclose(o3['mc_nstep'], ev.soft_mc_returns(r, lp, .5, .1, 3)[[0, 100]])
    assert (o3['mc'] > o3['mc_nstep']).all()
    s3 = ev.summarize_q_mc([o3])
    assert s3['n_steps'] == 3 and s3['mc_nstep_mean'] == pytest.approx(o3['mc_nstep'].mean())
    assert s3['q_min_minus_mc_nstep'] == pytest.approx((o3['q_min'] - o3['mc_nstep']).mean())
    assert s3['q_mean_ratio_nstep'] == pytest.approx(o3['q_mean'].mean() / o3['mc_nstep'].mean())
    assert ev.q_vs_mc(Critic(), obs[:10], np.zeros((10, 3)), r[:10], lp[:10], gamma=.5, alpha=.1,
                      failed=False, n_steps=3)['mc_nstep'].size == 0


def test_evaluate_model_q_pairs_failures_entropy_alpha(monkeypatch):
    """QA mutants E1 (Q paired with the NEXT obs), E2 (failed episodes lose their tail), E7
    (entropy sign), E8 (alpha not from the model): evaluate_model with stubbed laps and critic."""
    import types

    class Critic:
        def __call__(self, obs, act):                       # Q1 = 2 obs[0], Q2 = 2 obs[0] + 1
            return obs[:, :1] * 2., obs[:, :1] * 2. + 1.
    rng = np.random.default_rng(4)

    def sto_rep(T, termination):
        obs = np.zeros((T + 1, 3), np.float32)
        obs[:, 0] = np.arange(T + 1)
        trace = {'actions': np.zeros((T, 3), np.float32), 'reward': rng.normal(1., .3, T),
                 'lap_index': np.zeros(T, int)}
        return ev.LapReport(termination == 'lap_complete', [100.] if termination == 'lap_complete' else [], [],
                            termination, None, T, 0., trace, obs, rng.normal(-2., .5, T))
    det = ev.LapReport(False, [], [], 'off_track', None, 5, 0., {'lap_index': np.zeros(5, int)})
    sto = [sto_rep(700, 'lap_complete'), sto_rep(250, 'off_track')]
    reps = iter([det] + sto)
    monkeypatch.setattr(ev, 'flying_lap', lambda *a, **k: next(reps))
    model = types.SimpleNamespace(critic=Critic(), alpha=.37, gamma=.9, n_steps=3, _n_updates=7)
    cfg = EvalConfig(stochastic_laps=2)
    out = ev.evaluate_model(model, {'eval_det': None, 'eval_sto': None}, cfg)
    pairs = [ev.q_vs_mc(Critic(), r.obs[:-1], r.trace['actions'], r.trace['reward'], r.logp, gamma=.9, alpha=.37,
                        failed=f, n_steps=3) for r, f in zip(sto, (False, True))]
    assert [list(p['k']) for p in pairs] == [[0, 100], [0, 100, 200]]
    assert out['q_vs_mc'] == ev.summarize_q_mc(pairs) and out['q_vs_mc']['n'] == 5
    np.testing.assert_allclose(pairs[1]['q_min'], [0., 200., 400.])        # Q(s_k, a_k), not s_{k+1}
    assert out['alpha'] == .37 and out['n_updates'] == 7
    assert out['policy_entropy'] > 0 and out['policy_entropy'] == pytest.approx(-np.mean(np.r_[sto[0].logp, sto[1].logp]))
    assert out['sto_clean'] == 1 and out['sto_n'] == 2 and not out['det_clean']


def test_lap_tagging_and_lap_metrics():
    # 2 decisions run-in, lap 1 = 3 decisions, lap 2 = 4 decisions, final decision completes lap 2
    lap_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3])
    n = len(lap_index)
    gear = np.array([1, 1, 2, 2, 2, 2, 3, 3, 4, 4])
    slip = np.array([0, 0, .2, 0, 0, 0, .2, 0, .05, 0])       # spin upshift only at index 6 in lap 2
    kappa = np.zeros((n, 4))
    kappa[5, 0] = -.2                                          # FL locked at one lap-2 decision
    kappa[7, :2] = -.13                                        # both fronts locked
    kappa[8, 1] = -.11                                         # not below -0.12
    tp = np.array([0, 0, 0, 0, 0, .5, -.8, 0, 0, 0.])
    trace = {'lap_index': lap_index, 'gear': gear, 'slip_rl': slip, 'slip_rr': -slip / 2, 'kappa': kappa,
             'e': np.array([0, 0, 0, 0, 0, .1, -.4, .2, .3, 9.]), 'track_position': tp,
             'track_width': np.full(n, 12.)}
    laps = [{'lap_time_s': 100., 'time': 100., 'smoothness': 1., 'tracking': 2., 'cost': 103.},
            {'lap_time_s': 99., 'time': 99., 'smoothness': 3., 'tracking': 1., 'cost': 103.}]
    rep = ev.LapReport(True, [100., 99.], laps, 'lap_complete', None, n, 0., trace)
    m = ev.lap_metrics(rep, EvalConfig(), lap=2)
    assert m['decisions'] == 4 and m['lap_time_s'] == 99. and rep.lap2_s == 99. and rep.lap1_s == 100.
    assert m['mean_abs_e'] == pytest.approx(.25) and m['max_abs_e'] == pytest.approx(.4)
    assert m['min_edge_margin_m'] == pytest.approx((1 - .8) * 12 / 2 - 1.)
    assert m['front_lock_fraction'] == pytest.approx(.5)
    assert m['spin_upshifts'] == 1
    assert m['smoothness_share'] == pytest.approx(3 / 103)
    assert ev.edge_margin(0., 10.) == 4. and ev.edge_margin(1., 10.) == -1.
    # QA mutant E4: an upshift at the FIRST decision of lap 2 (gear compared with the last decision
    # of lap 1) counts; one at the first decision after lap 2 does not
    li = np.array([0, 1, 1, 2, 2, 3])
    tr2 = {'lap_index': li, 'gear': np.array([1, 1, 2, 3, 3, 4]), 'slip_rl': np.array([0, 0, 0, .2, 0, .2]),
           'slip_rr': np.zeros(6), 'kappa': np.zeros((6, 4)), 'e': np.zeros(6), 'track_position': np.zeros(6),
           'track_width': np.full(6, 12.)}
    assert ev.lap_metrics(ev.LapReport(True, [], [], 'lap_complete', None, 6, 0., tr2), EvalConfig())['spin_upshifts'] == 1
    assert ev.spin_upshifts([1, 2, 3], [0, .13, .12]) == 1
    assert ev.front_lock_fraction([-.2, 0.], [-.13, -.05]) == .5


# --- runs / provenance ----------------------------------------------------------------------------------
def test_run_dir_and_status(tmp_path):
    d = runs.make_run_dir(tmp_path, 'r', 3)
    assert d == tmp_path / 'r' / 'seed3' and (d / 'checkpoints').is_dir() and (d / 'bc').is_dir()
    with pytest.raises(FileNotFoundError):
        runs.make_run_dir(tmp_path, 'r', 3, resume=True)
    runs.update_status(d, phase='bc', state='running', x=math.nan)
    runs.add_event(d, 'rollback', step=10)
    runs.record_phase(d, 'bc', wall_s=1.5, env_steps=0)
    s = json.loads((d / 'status.json').read_text())
    assert s['phase'] == 'bc' and s['x'] is None and s['events'][0]['kind'] == 'rollback'
    assert s['phases'] == {'bc': {'wall_s': 1.5, 'env_steps': 0}}
    with pytest.raises(FileExistsError):
        runs.make_run_dir(tmp_path, 'r', 3)
    assert runs.make_run_dir(tmp_path, 'r', 3, resume=True) == d


def test_training_provenance(track, qss):
    from rlsetup.provenance import training_provenance
    p = training_provenance(track, n_envs=4, qss=qss, reference_digest='abc', track_sha256='f' * 64,
                            config_json='{}', argv=['x'])
    assert p['packages']['stable-baselines3'] == '2.9.0' and p['n_envs'] == 4 and p['argv'] == ['x']
    assert len(p['qss']['cache_key_sha256']) == 64 and p['qss']['lap_time_s'] == pytest.approx(133.916, abs=1e-3)
    assert p['torch_num_threads'] >= 1 and 'hostname' in p and p['demo_dataset_sha256'] is None
    assert {'rlsetup', 'ocd_first_lap', 'ocd_lap_code_version'} <= set(p)
    json.dumps(p)
    # D2_smoke.md finding: the code version includes the track CSVs (Track.name, not the object)
    from rlsetup.provenance import ocd_lap_code_version, code_state
    assert p['ocd_lap_code_version'] == ocd_lap_code_version('Shanghai') != ocd_lap_code_version(None)
    assert code_state(track)['ocd_lap_code_version'] == p['ocd_lap_code_version']


# --- flying_lap / probe on the real env (short limits) ------------------------------------------------
def test_flying_lap_and_probe_short(make_env):
    from rlsetup.driving.agent import init_output_bias
    from rlsetup.driving.config import EnvConfig
    model = make_model(fill=0)
    init_output_bias(model.actor, pedal_deadzone=True)
    env = make_env(EnvConfig(evaluation=True, record_telemetry=True, eval_max_s=2.))
    th.manual_seed(123)
    before = th.get_rng_state()
    threads = th.get_num_threads()
    sto = ev.flying_lap(model, env, deterministic=False, seed=5)
    assert th.equal(before, th.get_rng_state()) and th.get_num_threads() == threads
    sto2 = ev.flying_lap(model, env, deterministic=False, seed=5)
    np.testing.assert_array_equal(sto.trace['actions'], sto2.trace['actions'])
    det = ev.flying_lap(model, env, deterministic=True, seed=0)
    assert not det.clean and det.truncation == 'time_limit' and det.decisions == 50
    assert det.obs.shape == (51, 159) and np.isnan(det.logp).all() and np.isfinite(sto.logp).all()
    assert np.isfinite(det.trace['track_position']).all() and (det.trace['track_width'] > 5).all()
    assert np.isfinite(det.trace['kappa']).all() and (det.trace['lap_index'] == 0).all()
    np.testing.assert_allclose(det.trace['actions'][0], [0., -.275, -.8], atol=.02)
    probe_env = make_env(EnvConfig(max_episode_s=2.))
    starts = ev.make_probe_starts(probe_env, GuardConfig(n_starts=2))
    assert [s['seed'] for s in starts] == [1000, 1001] and starts[0]['s'] == pytest.approx(37.)
    p1, p2 = ev.probe(model, probe_env, starts), ev.probe(model, probe_env, starts)
    assert p1.progress_m == p2.progress_m and all(p > 0 for p in p1.progress_m)
