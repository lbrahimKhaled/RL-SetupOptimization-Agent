"""R2 (docs/specs/R2_review.md R2-10 incl. Amendment 1): corner table, exit bonus, buffer, schedule,
low-noise rollouts, focused starts, warm-start compatibility, evaluation, keeper.

Tests marked needs_f1 need ocd_first_lap branch sf70h-f1 (PYTHONPATH=<ocd_first_lap-f1 worktree>);
needs_src additionally the f1_best_w35 run folder (runs/ is not in git)."""
import dataclasses
import hashlib
import importlib.util
import json
import math
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
import torch as th
from ocd_lap import cars
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup import runs
from rlsetup.provenance import REPO_ROOT
from rlsetup.driving import corners as C
from rlsetup.driving.buffers import BonusNStepReplayBuffer
from rlsetup.driving.callbacks import (BonusScheduleCallback, FocusedStartCallback, DrivingEvalCallback,
                                       TrainStatsCallback, bonus_weight, focus_seed, load_bonus_state)
from rlsetup.driving.config import EnvConfig, RewardConfig, CornerBonusConfig, FocusConfig, TRAINING_AID_KEYS
from rlsetup.driving.train_config import (PRESETS, TrainConfig, BonusScheduleConfig, FocusUpdateConfig, EvalConfig,
                                          config_json, train_config_from_dict)
from r1_golden import legacy_rollout
from sb3_helpers import TinyEnv, tiny_cfg, make_model, OBS

HAS_F1 = 'sf70h_f1' in cars.CARS
needs_f1 = pytest.mark.skipif(not HAS_F1, reason='sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path')
SRC = REPO_ROOT / 'runs' / 'mac' / 'D2' / 'f1_best_w35' / 'seed0'
needs_src = pytest.mark.skipif(not (HAS_F1 and (SRC / 'checkpoints' / 'best.zip').exists()),
                               reason='needs sf70h_f1 and runs/mac/D2/f1_best_w35')
GOLDEN_LEGACY = '0007e2813d2d28971223460790b45852513768865f08d6d0b3685f5868a1c4c2'   # = test_r1.py
B, FLOOR = 3., .75
ON = dict(corner_bonus=CornerBonusConfig(enabled=True), focus=FocusConfig(enabled=True))


def synthetic_table(L=5388.773713834052, units=((517., 1027., 55.), (1468., 1735., 61.), (4635., 4900., 60.))):
    """CornerTable with given (start, exit, v_qss_exit) windows on a constant 60 m/s profile."""
    n = int(math.ceil(L))
    us = [{'name': f'U{k + 1}', 'apexes_m': [(a + b) / 2], 'v_apex': [20.], 'start_m': a, 'exit_m': b,
           'v_qss_exit': v} for k, (a, b, v) in enumerate(units)]
    return C.CornerTable(L, np.arange(n) * L / n, np.full(n, 60.), us, 90., {'key': 'synthetic'})


# --- R2-10.4: f(rho) ---------------------------------------------------------------------------------
def test_exit_fraction_and_bonus():
    assert C.exit_fraction(.75, FLOOR) == 0. and C.exit_fraction(.5, FLOOR) == 0.
    assert C.exit_fraction(1., FLOOR) == C.exit_fraction(1.3, FLOOR) == 1.
    assert C.exit_fraction(.875, FLOOR) == pytest.approx(.5) and C.exit_fraction(.8, FLOOR) == pytest.approx(.2)
    assert C.exit_bonus(.9, B, FLOOR) == pytest.approx(3. * .6) and C.exit_bonus(2., B, FLOOR) == B


# --- R2-10.3: CornerTracker ---------------------------------------------------------------------------
def tracker(L=5000., exit_m=1000., v_exit=50.):
    t = C.CornerTable(L, np.arange(5000) * 1., np.full(5000, 60.),
                      [{'name': 'U1', 'apexes_m': [900.], 'v_apex': [20.], 'start_m': 800., 'exit_m': exit_m,
                        'v_qss_exit': v_exit}], 90.)
    return C.CornerTracker(t, B, FLOOR)


def drive(tr, d_from, d_to, step=2., v=45.):
    ev, d = [], d_from
    while d < d_to:
        ev += tr.tick(d, d + step, v, v)
        d += step
    return ev


def test_tracker_one_payment_per_exit_per_lap():
    tr = tracker()
    tr.reset(0.)                                       # exit 1000 m ahead, then every 5000 m
    ev = drive(tr, 0., 12_000.)
    assert len(ev) == 3 and all(e['unit'] == 0 for e in ev)
    e = ev[0]
    assert e['rho'] == pytest.approx(.9) and e['bonus'] == pytest.approx(B * .6) and e['v_qss_exit'] == 50.


def test_tracker_interpolates_speed_in_the_tick():
    tr = tracker()
    tr.reset(0.)
    assert tr.tick(0., 999., 40., 40.) == []
    (e,) = tr.tick(999., 1003., 40., 48.)               # crossing at phi = 0.25
    assert e['v_exit'] == pytest.approx(42.) and e['rho'] == pytest.approx(.84)


def test_tracker_back_and_forth_paid_once():
    tr = tracker()
    tr.reset(0.)
    ev = tr.tick(0., 999., 45., 45.) + tr.tick(999., 1001., 45., 45.) + tr.tick(1001., 990., 5., 5.) \
        + tr.tick(990., 1005., 5., 45.) + tr.tick(1005., 998., 5., 5.) + tr.tick(998., 1010., 5., 45.)
    assert len(ev) == 1


def test_tracker_start_inside_window_and_past_exit():
    tr = tracker()
    tr.reset(900.)                                     # hand-over inside the window [800, 1000] before s_out
    (e,) = drive(tr, 0., 150.)
    assert e['bonus'] > 0
    tr.reset(1050.)                                    # past s_out: next pass, one lap later
    assert drive(tr, 0., 4900.) == [] and len(drive(tr, 4900., 5000.)) == 1


def test_tracker_projection_jump_pays_at_most_B():
    tr = tracker()
    tr.reset(0.)
    (e,) = tr.tick(0., 3000., 10., 500.)                # a projection jump with an absurd speed
    assert e['bonus'] <= B and e['bonus'] == B
    tr.reset(0.)
    ev = tr.tick(0., 2 * 5000. + 1500., 45., 45.)      # more than two laps in one tick: still one payment
    assert len(ev) == 1 and tr.next[0] > 2 * 5000. + 1500.


def test_settle_events_forfeit_on_failure():
    ev = [{'unit': 0, 'v_exit': 45., 'v_qss_exit': 50., 'rho': .9, 'bonus': 1.8},
          {'unit': 1, 'v_exit': 30., 'v_qss_exit': 50., 'rho': .6, 'bonus': 0.}]
    total, out = C.settle_events(ev, failed=True)
    assert total == 0. and [e['kind'] for e in out] == ['forfeit', 'forfeit'] and all(e['bonus'] == 0. for e in out)
    total, out = C.settle_events(ev, failed=False)
    assert total == 1.8 and [e['kind'] for e in out] == ['paid', 'exit_unpaid']


def test_env_forfeits_crossings_of_the_failing_decision(make_env, monkeypatch):
    """Env wiring: a crossing in a decision that ends off track is 'forfeit' (bonus 0); earlier
    decisions pay; the env reward never contains the bonus."""
    env = make_env(EnvConfig(settle_s=0., reward=RewardConfig(mode='time_pbrs'), **ON), corners=synthetic_table())
    ref = make_env(EnvConfig(settle_s=0., reward=RewardConfig(mode='time_pbrs')))
    for e in (env, ref):
        e.reset(seed=3, options={'start_s': 4500., 'start_speed': 30.})
    fake = {'unit': 0, 'v_exit': 45., 'v_qss_exit': 50., 'rho': .9, 'bonus': 1.8}
    monkeypatch.setattr(env._tracker, 'tick', lambda *a: [dict(fake)])
    crash = np.array([1., 1., -1.], np.float32)
    total = 0.
    for k in range(400):
        o1, r1, te1, tr1, i1 = env.step(crash)
        o0, r0, te0, tr0, i0 = ref.step(crash)
        assert o1.tobytes() == o0.tobytes() and r1 == r0 and (te1, tr1) == (te0, tr0)
        assert 'corner_bonus' not in i0
        if te1:
            assert i1['termination'] == 'off_track' and i1['corner_bonus'] == 0.
            assert {e['kind'] for e in i1['corner_events']} == {'forfeit'}
            assert i1['episode_corner_bonus'] == pytest.approx(total)
            break
        assert {e['kind'] for e in i1['corner_events']} == {'paid'}
        assert i1['corner_bonus'] == pytest.approx(1.8 * env.cfg.action_repeat)
        total += i1['corner_bonus']
    else:
        pytest.fail('no off-track')
    assert total > 0


# --- R2-10.1: corner table --------------------------------------------------------------------------
def test_detect_units_on_a_synthetic_profile():
    """Apex detection, merge (< 150 m), window start = argmax between units, exit rule incl. the
    30 m-before-braking cap and the wrap through the global maximum."""
    n = 3000
    s = np.arange(n, dtype=float)
    v = 80. - 4. * np.exp(-((s - 2900.) / 200.) ** 2)       # long gentle curve (< 5 m/s: no apex)
    dip = lambda c, depth, w=25.: depth * np.exp(-((s - c) / w) ** 2)
    v = v - dip(500, 50) - dip(600, 40) - dip(1200, 45) - dip(1300, 4) - dip(1390, 45)
    v[200] = 95.                                          # global maximum (braking point of the first unit)
    units, i0 = C.detect_units(v, 1., CornerBonusConfig())
    assert i0 == 200
    ap = [sorted(a % n for a in u['apexes']) for u in units]
    assert ap == [[500, 600], [1200], [1390]]            # 500 + 600 merged; the 4 m/s kink at 1300 is no apex
    starts = [u['start'] % n for u in units]
    assert starts[0] == 200
    assert starts[2] == 1200 + int(np.argmax(v[1201:1390])) + 1
    exits = [(u['exit_m_rel'] + i0) % n for u in units]
    assert starts[1] == 600 + int(np.argmax(v[601:1200])) + 1
    for k, last in enumerate((600, 1200, 1390)):         # last apex + min(150, next start - last - 30)
        nxt = starts[k + 1] if k < 2 else starts[0] + n
        assert exits[k] == pytest.approx(last + min(150., nxt - last - 30.))
    assert exits[2] == pytest.approx(1540.)


@needs_f1
def test_corner_table_matches_r2_2_7(tmp_path):
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from rlsetup.driving.reference import ReferenceLine
    env = PRESETS['f1_r2']().env
    track = Track(env.track)
    qss = qss_profile.for_env(track, env)
    ref = ReferenceLine.for_track(track, env.reference)
    table = C.load_or_build(track, ref, qss, env, path=tmp_path / 'ct.json')     # built and written
    expect = [((813, 877), 517, 1027, 55.2), ((1585,), 1467, 1735, 60.9), ((2329,), 2062, 2419, 57.5),
              ((2531,), 2449, 2681, 57.5), ((3112, 3167), 3007, 3317, 56.0), ((4749,), 4634, 4899, 60.4),
              ((5128,), 5042, 5278, 63.4)]
    assert [u['name'] for u in table.units] == [f'U{k}' for k in range(1, 8)]
    for u, (apexes, b, out, v) in zip(table.units, expect):
        assert np.allclose(u['apexes_m'], apexes, atol=2.) and abs(u['start_m'] - b) <= 2. and abs(u['exit_m'] - out) <= 2.
        assert u['v_qss_exit'] == pytest.approx(v, abs=.2)
    C.check_windows(table)
    line = qss_profile.line_lap_time(qss, track, ref.path, env)
    assert table.lap_time_s == pytest.approx(line, abs=1e-6) and table.meta['t_line_end_s'] == pytest.approx(line, abs=1e-6)
    again = C.load_or_build(track, ref, qss, env, path=tmp_path / 'ct.json')       # cache hit
    assert again.meta['key'] == table.meta['key'] and again.units == table.units
    other = dataclasses.replace(env.corner_bonus, merge_m=10.)
    assert C.cache_key(qss, ref, other)[0] != table.meta['key']
    assert C.cache_key(qss, ref, dataclasses.replace(env.corner_bonus, bonus=6.))[0] == table.meta['key']


def test_check_windows_rejects_overlap_and_short():
    with pytest.raises(ValueError, match='overlap'):
        C.check_windows(synthetic_table(units=((500., 1000., 50.), (900., 1200., 50.))))
    with pytest.raises(ValueError, match='< 100'):
        C.check_windows(synthetic_table(units=((500., 550., 50.),)))
    C.check_windows(synthetic_table(units=((5300., 200., 50.), (300., 600., 50.))))     # wraps through 0


# --- R2-10.2: bit identity, configs, demo keys ------------------------------------------------------
def test_golden_rollout_unchanged_with_aids_on(make_env):
    table = synthetic_table()
    digest, _ = legacy_rollout(lambda c: make_env(dataclasses.replace(c, **ON), corners=table))
    assert digest == GOLDEN_LEGACY
    assert legacy_rollout(make_env)[0] == GOLDEN_LEGACY


def test_time_pbrs_rollout_bitwise_equal_with_bonus(make_env):
    table = synthetic_table()
    cfg = EnvConfig(max_episode_s=8., reward=RewardConfig(mode='time_pbrs'))
    a = make_env(dataclasses.replace(cfg, **ON), corners=table)
    b = make_env(cfg)
    a.set_focus_weights(None)
    paid = 0.
    for seed in (5, 6):
        (oa, ia), (ob, ib) = a.reset(seed=seed), b.reset(seed=seed)
        assert oa.tobytes() == ob.tobytes() and ia['start']['s0'] == ib['start']['s0']
        for k in range(200):
            act = np.array([.05 * math.sin(k / 9.), .4, -1.], np.float32)
            ra, rb_ = a.step(act), b.step(act)
            assert ra[0].tobytes() == rb_[0].tobytes() and ra[1:4] == rb_[1:4]
            paid += ra[4]['corner_bonus']
            if ra[2] or ra[3]:
                break
    assert 'corner_bonus' not in rb_[4]


def test_config_dicts_without_aids_are_unchanged():
    assert set(TRAINING_AID_KEYS) == {'corner_bonus', 'focus'}
    d = EnvConfig().to_dict()
    assert not set(TRAINING_AID_KEYS) & set(d)
    on = EnvConfig(**ON).to_dict()
    assert on['corner_bonus']['enabled'] and on['focus']['fraction'] == .25
    for name, make in PRESETS.items():
        cfg = make()
        j = json.loads(config_json(cfg))
        assert train_config_from_dict(j) == cfg, name
        if not name.startswith('f1_r2'):
            assert not {'bonus_schedule', 'focus_update'} & set(j) and not set(TRAINING_AID_KEYS) & set(j['env'])
            assert not {'low_noise_envs', 'low_noise_std_scale'} & set(j['agent'])
        else:
            assert {'bonus_schedule', 'focus_update'} <= set(j) and set(TRAINING_AID_KEYS) <= set(j['env'])
    old = json.loads(config_json(PRESETS['f1_best_w35']()))
    assert train_config_from_dict(old).env.corner_bonus == CornerBonusConfig()          # missing -> default


def test_demo_key_env_part_unchanged():
    """The env dict that enters demos.dataset_key is unchanged by R2: every local demo dataset whose
    stored key inputs have the current (pre-R2) field set round-trips to the same dict and key, and
    the f1_best_w35 dataset's env dict equals the preset's. (The key also hashes the source of
    env.py / config.py, which R2 had to edit: see docs/handoff/R2_coder.md.)"""
    from rlsetup.driving.train_config import config_from_dict
    cur = json.loads(json.dumps(EnvConfig().to_dict()))
    shape = lambda d: {k: shape(v) if isinstance(v, dict) else None for k, v in d.items()}
    checked = 0
    for meta in sorted((REPO_ROOT / 'data' / 'demos').glob('demos_*.json')):
        d = json.loads(meta.read_text())
        inputs = d.get('key_inputs') or {}
        env = inputs.get('env')
        if env is None or shape(env) != shape(cur):
            continue                                      # written before R1 / other fields: not comparable
        assert json.loads(json.dumps(config_from_dict(EnvConfig, env).to_dict())) == env, meta.name
        assert hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest() == d['key'], meta.name
        checked += 1
    if (SRC / 'demos.json').exists():
        key = json.loads((SRC / 'demos.json').read_text())['key']
        meta = REPO_ROOT / 'data' / 'demos' / f'demos_{key[:12]}.json'
        if meta.exists():
            env = json.loads(meta.read_text())['key_inputs']['env']
            assert env == json.loads(json.dumps(PRESETS['f1_best_w35']().env.to_dict()))
            checked += 1
    if not checked:
        pytest.skip('no local demo datasets with the current env fields')


def test_presets_f1_r2():
    base, r2, smoke = PRESETS['f1_best_w35'](), PRESETS['f1_r2'](), PRESETS['f1_r2_smoke']()
    assert dataclasses.replace(r2.env, corner_bonus=CornerBonusConfig(), focus=FocusConfig()) == base.env
    assert r2.env.corner_bonus.enabled and r2.env.corner_bonus.bonus == 3. and r2.env.focus.enabled
    assert (r2.agent.init_mode, r2.agent.low_noise_envs, r2.agent.low_noise_std_scale) == ('warm', (3,), .3)
    assert (r2.warm.relabel, r2.warm.actor_lr, r2.guard.threshold, r2.stop.plateau) == (False, 1e-4, .65, False)
    assert (r2.total_timesteps, r2.max_wall_h, r2.focus_update.seed_run) == (1_000_000_000, None, 'runs/mac/D2/f1_best_w35/seed0')
    assert r2.bonus_schedule.report_from == 8_000_000 and r2.strict_provenance
    s = smoke.bonus_schedule
    assert (s.hold_until, s.fade_until, s.report_from, smoke.total_timesteps) == (60_000, 120_000, 150_000, 150_000)
    assert (smoke.eval.interval, smoke.eval.stochastic_laps, smoke.guard.interval, smoke.strict_provenance) == \
        (25_000, 3, 5_000, False)
    assert smoke.env == r2.env and smoke.agent == r2.agent


# --- R2-10.5: replay buffer ------------------------------------------------------------------------
def _buffers(n_envs=1, size=64, n_steps=3, gamma=.9, rows=50, seed=0):
    """Identical plain / bonus buffers with `size` rows per env (SB3 divides buffer_size by n_envs)."""
    from gymnasium import spaces
    obs_sp, act_sp = spaces.Box(-1, 1, (4,), np.float32), spaces.Box(-1, 1, (2,), np.float32)
    kw = dict(n_envs=n_envs, n_steps=n_steps, gamma=gamma)
    plain = NStepReplayBuffer(size * n_envs, obs_sp, act_sp, **kw)
    bonus = BonusNStepReplayBuffer(size * n_envs, obs_sp, act_sp, **kw)
    rng = np.random.default_rng(seed)
    for t in range(rows):
        o, o2 = rng.uniform(-1, 1, (n_envs, 4)).astype(np.float32), rng.uniform(-1, 1, (n_envs, 4)).astype(np.float32)
        a, r = rng.uniform(-1, 1, (n_envs, 2)).astype(np.float32), rng.normal(size=n_envs).astype(np.float32)
        done = rng.uniform(size=n_envs) < .15
        infos = [{'corner_bonus': float(rng.uniform(0, 3)) if rng.uniform() < .3 else 0.,
                  'TimeLimit.truncated': bool(done[i] and rng.uniform() < .3)} for i in range(n_envs)]
        for buf in (plain, bonus):
            buf.add(o, o2, a, r, done, infos)
    return plain, bonus


def _samples_equal(a, b):
    return all(th.equal(x, y) for x, y in zip(a, b))


def _sample(buf, inds, seed=0):
    np.random.seed(seed)
    return buf._get_samples(np.asarray(inds))


def test_bonus_buffer_w0_bitwise_equal_to_nstep():
    for n_envs, rows in ((1, 50), (3, 100)):          # 100 rows > 64: wrapped (full) buffer
        plain, bonus = _buffers(n_envs=n_envs, rows=rows)
        assert bonus.bonus_weight == 0. and bonus.labelled[:min(rows, 64)].all()
        inds = np.random.default_rng(1).integers(0, 50 if rows == 50 else 64, 256)
        assert _samples_equal(_sample(plain, inds), _sample(bonus, inds))


def test_bonus_buffer_nstep_returns_with_weight():
    plain, bonus = _buffers(rows=50)
    bonus.bonus_weight = .7
    g, n = bonus.gamma, bonus.n_steps
    inds = np.arange(0, 45)
    got = _sample(bonus, inds).rewards.numpy()[:, 0]
    for k, i in enumerate(inds):
        tot = 0.
        for j in range(n):
            r = bonus.rewards[i + j, 0] + np.float32(.7) * bonus.bonuses[i + j, 0]
            tot += g ** j * r
            if bonus.dones[i + j, 0] or bonus.timeouts[i + j, 0] or (i + j == bonus.pos - 1):
                break
        assert got[k] == pytest.approx(tot, rel=1e-5, abs=1e-5)
    ref = _sample(plain, inds).rewards.numpy()[:, 0]
    assert not np.allclose(got, ref)                    # the bonus does enter


def test_adopt_in_place_and_pickle(tmp_path):
    from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
    plain, _ = _buffers(rows=50)
    save_to_pkl(tmp_path / 'p.pkl', plain)
    rb = load_from_pkl(tmp_path / 'p.pkl')
    obs_id = id(rb.observations)
    inds = np.arange(40)
    before = _sample(rb, inds)
    assert BonusNStepReplayBuffer.adopt(rb) is rb and type(rb) is BonusNStepReplayBuffer and id(rb.observations) == obs_id
    assert not rb.bonuses.any() and not rb.labelled.any() and rb.bonus_weight == 0. and rb.labelled_fraction() == 0.
    assert _samples_equal(before, _sample(rb, inds))
    rb.bonus_weight = 1.
    assert _samples_equal(before, _sample(rb, inds))      # all bonuses 0
    rb.add(np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32), np.zeros((1, 2), np.float32),
           np.zeros(1, np.float32), np.array([False]), [{'corner_bonus': 2.5}])
    assert rb.bonuses[50, 0] == 2.5 and rb.labelled[50, 0] and rb.labelled_fraction() == pytest.approx(1 / 51)
    save_to_pkl(tmp_path / 'b.pkl', rb)
    back = load_from_pkl(tmp_path / 'b.pkl')
    assert type(back) is BonusNStepReplayBuffer and np.array_equal(back.bonuses, rb.bonuses)
    assert np.array_equal(back.labelled, rb.labelled) and back.bonus_weight == 1.
    with pytest.raises(TypeError):
        BonusNStepReplayBuffer.adopt(object())


def test_build_model_buffer_class():
    c = tiny_cfg()
    on = dataclasses.replace(c, env=dataclasses.replace(c.env, corner_bonus=CornerBonusConfig(enabled=True)))
    m = make_model(on, fill=10)
    assert type(m.replay_buffer) is BonusNStepReplayBuffer and m.replay_buffer.n_steps == 3
    assert m.replay_buffer.gamma == on.env.reward.gamma
    assert type(make_model(c, fill=0).replay_buffer) is NStepReplayBuffer


# --- R2-10.6: schedule + Amendment 1 -------------------------------------------------------------------
def test_bonus_weight_schedule():
    cfg = BonusScheduleConfig()
    assert [bonus_weight(cfg, n) for n in (0, 1_999_999, 2_000_000, 3_000_000, 4_000_000, 9_000_000)] == \
        [1., 1., 1., .5, 0., 0.]
    cb = BonusScheduleCallback(cfg)
    assert cb.report_from == 8_000_000 and not cb.reportable(7_999_999) and cb.reportable(8_000_000)


def _sched_model(**sched):
    c = tiny_cfg()
    cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, corner_bonus=CornerBonusConfig(enabled=True)),
                              bonus_schedule=BonusScheduleConfig(**sched))
    return cfg, make_model(cfg, fill=600)


def test_weight_set_before_every_train_call():
    cfg, model = _sched_model(hold_until=40, fade_until=120, bonus_free_before_report=40)
    seen = []
    orig = model.train

    def train(gradient_steps, batch_size=64):
        seen.append((model.num_timesteps, model.replay_buffer.bonus_weight, model.bonus_weight))
        return orig(gradient_steps, batch_size)
    model.train = train
    cb = BonusScheduleCallback(cfg.bonus_schedule)
    model.learn(160, callback=[cb])
    assert len(seen) == 160
    for n, w, w2 in seen:
        assert w == w2 == bonus_weight(cfg.bonus_schedule, n)
    assert {w for _, w, _ in seen} >= {1., 0., .5}


def test_resume_mid_fade_and_after_trigger(tmp_path):
    cfg = BonusScheduleConfig()
    cb = BonusScheduleCallback(cfg)                        # resume at 3.0M, no trigger
    assert cb.weight(3_000_000) == .5
    events = []
    cb = BonusScheduleCallback(cfg, tmp_path / 'bonus_state.json', on_event=lambda k, **d: events.append(k))
    rec = cb.on_eval({'step': 1_000_000, 'det_clean': True, 'lap2_s': 95.2})
    assert rec['bonus_w'] == 1. and rec['reportable'] is False and 'aided_target_lap' not in rec
    rec = cb.on_eval({'step': 1_050_000, 'det_clean': True, 'lap2_s': 95.0})          # <= 95: trigger
    assert rec['aided_target_lap'] and rec['bonus_w'] == 1. and rec['report_from'] == 5_050_000
    assert rec['reportable'] is False and events == ['aided_target_lap'] and cb.weight(1_060_000) == 0.
    assert cb.on_eval({'step': 1_100_000, 'det_clean': True, 'lap2_s': 94.})['bonus_w'] == 0.
    assert events == ['aided_target_lap']                                                   # once
    st = runs.read_json(tmp_path / 'bonus_state.json')
    assert (st['trigger_step'], st['report_from']) == (1_050_000, 5_050_000)
    trig, rf = load_bonus_state(tmp_path)
    resumed = BonusScheduleCallback(cfg, trigger_step=trig, report_from=rf)
    assert resumed.weight(1_200_000) == 0. and resumed.weight(500_000) == 0. and resumed.report_from == 5_050_000
    model = SimpleNamespace(num_timesteps=5_049_996, replay_buffer=SimpleNamespace(bonus_weight=1.),
                            logger=SimpleNamespace(record=lambda *a: None))
    resumed.model = model
    assert resumed._on_step() and model.replay_buffer.bonus_weight == 0.
    model.num_timesteps = 5_050_000
    assert resumed._on_step() is False and resumed.stopped
    assert model.bonus_trigger_step == 1_050_000 and model.bonus_report_from == 5_050_000
    # no trigger once w = 0 and step >= report_from (the keeper stops on that reportable lap)
    late = BonusScheduleCallback(cfg)
    rec = late.on_eval({'step': 8_000_000, 'det_clean': True, 'lap2_s': 94.})
    assert 'aided_target_lap' not in rec and rec['reportable'] and late.trigger_step is None
    # model attributes are the fallback without bonus_state.json
    m = SimpleNamespace(bonus_trigger_step=7, bonus_report_from=4_000_007)
    assert load_bonus_state(tmp_path / 'none', m) == (7, 4_000_007)


# --- R2-10.7: low-noise rollouts -----------------------------------------------------------------------
def _low_noise_model(scale):
    c = tiny_cfg()
    cfg = dataclasses.replace(c, n_envs=4, agent=dataclasses.replace(c.agent, low_noise_envs=(3,), low_noise_std_scale=scale))
    from rlsetup.driving.agent import build_model
    model = build_model(cfg, DummyVecEnv([TinyEnv] * 4))
    model._last_obs = model.env.reset()
    return model


def test_low_noise_scale0_is_the_deterministic_action():
    model = _low_noise_model(0.)
    assert model.low_noise_envs == (3,)
    th.manual_seed(0)
    ref_a, ref_b = SAC._sample_action(model, 0, None, 4)
    th.manual_seed(0)
    a, b = model._sample_action(0, None, 4)
    assert np.array_equal(a[:3], ref_a[:3]) and np.array_equal(b[:3], ref_b[:3])
    det = model.predict(model._last_obs, deterministic=True)[0]
    sc = model.policy.scale_action(det)
    assert np.array_equal(b[3], sc[3]) and np.array_equal(a[3], model.policy.unscale_action(sc)[3])
    assert not np.array_equal(a[3], ref_a[3])


def test_low_noise_std_is_scaled():
    model = _low_noise_model(.3)
    obs = model.policy.obs_to_tensor(model._last_obs)[0]
    with th.no_grad():
        mean, log_std, _ = model.actor.get_action_dist_params(obs)
    pre = []
    for _ in range(3000):
        a, _ = model._sample_action(0, None, 4)
        pre.append(np.arctanh(np.clip(a[3], -.999999, .999999)))
    z = (np.asarray(pre) - mean[3].numpy()) / th.exp(log_std[3]).numpy()
    assert np.allclose(z.std(axis=0), .3, rtol=.08) and np.allclose(z.mean(axis=0), 0., atol=.03)


def test_low_noise_envs_out_of_range_refused():
    c = tiny_cfg()
    from rlsetup.driving.agent import build_model
    with pytest.raises(ValueError, match='low_noise_envs'):
        build_model(dataclasses.replace(c, agent=dataclasses.replace(c.agent, low_noise_envs=(3,))), DummyVecEnv([TinyEnv]))


# --- R2-10.8: focused starts ---------------------------------------------------------------------------
def test_focus_sampler_share_and_handover_targets(make_env):
    table = synthetic_table()
    env = make_env(EnvConfig(**ON), corners=table)
    env.reset(seed=0)
    assert env.n_focus_bins == 108
    p = np.zeros(108)
    p[[16, 31, 93]] = [.5, .3, .2]                      # 800 m (U1), 1550 m (U2), 4650 m (U3)
    env.set_focus_weights(p)
    plans = [env._focus_plan({}) for _ in range(10_000)]
    focused = [q for q in plans if q]
    assert abs(len(focused) / 10_000 - .25) <= .015
    assert {q['bin'] for q in focused} == {16, 31, 93}
    assert {q['target'] for q in focused} == {517., 1468., 4635.}
    L = float(env.track.length)
    for q in focused[:2000]:
        st = env.focus_start(q['target'])
        lead = (q['target'] - st['s_handover']) % L
        assert 50. <= lead <= 300. and .5 <= st['f'] <= .8 and st['v0'] > 0.
    assert env.focus_target(50) == 2500.                # bin centre 2525 m in no window: the bin start
    for opts in ({'start_s': 100.}, {'start_speed': 5.}, {'evaluation': True}):
        env.evaluation = bool(opts.get('evaluation'))
        assert all(env._focus_plan(opts) is None for _ in range(200))
    with pytest.raises(ValueError):
        env.set_focus_weights(np.ones(5))


def test_focused_resets_integration(make_env):
    """N = 20 focused resets (fraction 1): hand-over within -330 .. -20 m of the target; eval,
    probe (start_s) and explicit resets never focused."""
    table = synthetic_table()
    env = make_env(EnvConfig(corner_bonus=CornerBonusConfig(enabled=True), focus=FocusConfig(enabled=True, fraction=1.)),
                   corners=table)
    p = np.zeros(108)
    p[[16, 31, 93]] = 1 / 3
    env.set_focus_weights(p)
    ho = []
    for k in range(20):
        _, info = env.reset(seed=100 + k)
        st = info['start']
        if st['focus_fallback']:
            continue
        assert st['focused'] and st['focus_target_s'] in (517., 1468., 4635.)
        ho.append(st['handover_to_target_m'])
    assert len(ho) >= 18 and all(-330. <= h <= -20. for h in ho), ho
    _, info = env.reset(seed=1, options={'start_s': 300.})
    assert info['start']['focused'] is False
    _, info = env.reset(seed=1, options={'evaluation': True})
    assert 'focused' not in info['start']


# --- R2-10.9: FocusedStartCallback ---------------------------------------------------------------------
class FakeVenv:
    def __init__(self, n):
        self.num_envs = n
        self.reset_infos = [{'s_track': 10., 'start': {'focused': False}} for _ in range(n)]


def _focus_cb(ucfg=FocusUpdateConfig(update_every=100, half_life=200), **kw):
    pushed = []
    cb = FocusedStartCallback(ucfg, 5388.77, 50., push=pushed.append, **kw)
    venv = FakeVenv(2)
    cb.model = SimpleNamespace(num_timesteps=0, get_env=lambda: venv)
    return cb, pushed


def _step(cb, n, infos, dones):
    cb.model.num_timesteps = n
    cb.locals = {'infos': infos, 'dones': np.asarray(dones)}
    return cb._on_step()


def test_focus_callback_passes_crashes_decay(tmp_path):
    cb, pushed = _focus_cb(state_path=tmp_path / 'focus_state.json')
    cb._on_training_start()
    assert len(pushed) == 1 and np.allclose(pushed[0], 1 / 108)          # no seed, no crashes: uniform
    n0 = cb.n.copy()
    _step(cb, 2, [{'s_track': 60.}, {'s_track': 20.}], [False, False])   # env 0: bin 0 -> 1
    _step(cb, 4, [{'s_track': 110.}, {'s_track': 30.}], [False, False])  # env 0: bin 1 -> 2
    assert cb.n[1] - n0[1] == 1 and cb.n[2] - n0[2] == 1 and cb.n[0] == n0[0]
    cb.model.get_env().reset_infos[0] = {'s_track': 4000., 'start': {'focused': True, 'handover_to_target_m': -120.}}
    _step(cb, 6, [{'s_track': 120., 'termination': 'off_track'}, {'s_track': 40., 'termination': None,
                                                                  'truncation': 'stall'}], [True, False])
    assert cb.c[2] == 1. and cb.c.sum() == 1. and cb.last[0] == 80 and cb.focused[0]
    cb.model.get_env().reset_infos[1] = {'s_track': 900., 'start': {'focused': False}}
    _step(cb, 8, [{'s_track': 4003.}, {'s_track': 45., 'termination': None, 'truncation': 'stall'}], [False, True])
    assert cb.c.sum() == 1.                                               # a stall is no crash
    c2, n2 = cb.c[2], cb.n[2]
    _step(cb, 100, [{'s_track': 4005.}, {'s_track': 905.}], [False, False])
    d = .5 ** (100 / 200)
    assert cb.c[2] == pytest.approx(c2 * d) and cb.n[2] == pytest.approx(n2 * d) and len(pushed) == 2
    r = cb.c / cb.n
    assert np.allclose(pushed[-1], .8 * r / r.sum() + .2 / 108)
    out = cb.eval_extra(None, 100)['focus']
    assert out['share_resets'] == pytest.approx(1 / 4) and out['share_transitions'] == pytest.approx(2 / 10)
    assert out['handover_mean_m'] == -120. and out['top5'][0] == [100, pytest.approx(float(pushed[-1][2]))]
    st = runs.read_json(tmp_path / 'focus_state.json')
    back, _ = _focus_cb(state=st)
    assert np.array_equal(back.n, cb.n) and np.array_equal(back.c, cb.c) and np.allclose(back.weights(), cb.weights())


def _fake_laps(d, rows):
    (d / 'laps').mkdir(parents=True)
    for step, clean, term, s_end in rows:
        np.savez_compressed(d / 'laps' / f'step_{step}_det.npz', s=np.array([0., s_end]), clean=clean,
                            termination=str(term))


def test_focus_seed_and_prior(tmp_path):
    rows = [(1_000_000, False, 'off_track', 760.)] + [(2_000_000 + k, False, 'off_track', 760. + k) for k in range(6)] \
        + [(3_000_000, False, 'wrong_way', 2410.), (3_100_000, False, 'stall', 2410.), (3_200_000, True, 'lap_complete', 5000.)]
    _fake_laps(tmp_path, rows)
    u = FocusUpdateConfig()
    s = focus_seed(tmp_path, u, 50., 5388.77)
    assert s['n_evals'] == 9 and s['total_crashes'] == 7 and s['hot'] == [15] and s['crashes'][48] == 1
    cb, _ = _focus_cb(u, seed=s)
    assert np.all(cb.n == 200.) and cb.c[15] == pytest.approx(200. * 6 / 9) and cb.c[48] == pytest.approx(200. / 9)
    p = cb.weights()
    assert p[15] == pytest.approx(.8 * 6 / 7 + .2) and p[48] == pytest.approx(.8 / 7) and p.sum() == pytest.approx(1.)


@needs_src
def test_focus_seed_of_f1_best_w35():
    s = focus_seed(SRC, FocusUpdateConfig(), 50., 5388.773713834052)
    assert s['total_crashes'] == 166 and [b * 50 for b in s['hot']] == \
        [650, 700, 750, 800, 1550, 2350, 2400, 2500, 3100, 4700, 5100]


# --- TrainStats groups (R2-8.4) ------------------------------------------------------------------------
def test_train_stats_groups(tmp_path):
    from rlsetup.driving.vec import base_venv
    model = make_model(fill=0)
    venv = base_venv(model.get_env())
    venv.reset_infos = [{'start': {'v0': 20., 'focused': True}}]
    cb = TrainStatsCallback(tmp_path / 'ts.jsonl', window=8, groups=True, low_noise_envs=(0,))
    cb.init_callback(model)
    cb.on_training_start({}, {})
    venv.reset_infos = [{'start': {'v0': 20., 'focused': False}}]
    model.num_timesteps = 4
    cb.update_locals({'infos': [{'termination': 'off_track', 's_track': 812., 'episode_corner_bonus': 2., 'speed': 40., 'e_ref': 2.}],
                      'dones': np.array([True])})
    cb.on_step()
    model.num_timesteps = 8
    cb.update_locals({'infos': [{'termination': None, 'truncation': 'time_limit', 's_track': 10.,
                                 'episode_corner_bonus': 6., 'speed': 50., 'e_ref': .1}], 'dones': np.array([True])})
    cb.on_step()
    g = runs.read_jsonl(tmp_path / 'ts.jsonl')[0]['groups']
    assert g['low_noise']['episodes'] == 2 and g['normal']['episodes'] == 0 and g['low_noise']['failure_rate'] == .5
    assert g['focused'] == {'episodes': 1, 'failures': 1, 'failure_rate': 1., 'crash_bins': {'800': 1},
                            'bonus_per_episode': 2.}
    assert g['uniform']['bonus_per_episode'] == 6. and cb.eval_groups()['low_noise']['episodes'] == 2
    assert cb.eval_groups()['low_noise']['episodes'] == 0


# --- R2-10.10: warm start / preload -------------------------------------------------------------------
def _r2_cfg(src_path, **warm):
    from test_r1 import _warm_cfg
    c = _warm_cfg(src_path, **warm)
    return dataclasses.replace(c, env=dataclasses.replace(c.env, **ON))


def test_warm_start_accepts_aid_blocks_refuses_driver_change(tmp_path):
    from test_r1 import _save_source
    from rlsetup.driving.agent import build_model, warm_start, InitMismatch
    c = tiny_cfg()
    src_cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, reward=RewardConfig(mode='time_pbrs')))
    _, path = _save_source(tmp_path, 'src', src_cfg)
    cfg = _r2_cfg(path)
    warm_start(build_model(cfg, DummyVecEnv([TinyEnv])), path, cfg)                # accepted
    bad = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, max_episode_s=30.))
    with pytest.raises(InitMismatch, match='env config'):
        warm_start(build_model(bad, DummyVecEnv([TinyEnv])), path, bad)
    if (SRC / 'config.json').exists():                                            # the real source config
        from rlsetup.driving.agent import _env_without_reward
        src = json.loads((SRC / 'config.json').read_text())['env']
        assert _env_without_reward(src) == _env_without_reward(PRESETS['f1_r2']().env.to_dict())
        assert src['reward'] == json.loads(json.dumps(dataclasses.asdict(PRESETS['f1_r2']().env.reward)))


def test_preload_adopts_plain_buffer_and_refuses_relabel_of_time_pbrs(tmp_path):
    from test_r1 import _save_source
    from rlsetup.driving.agent import build_model, preload_buffer, default_init_buffer, InitMismatch
    c = tiny_cfg()
    src_cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, reward=RewardConfig(mode='time_pbrs')))
    src, path = _save_source(tmp_path, 'src', src_cfg, buffer=True)
    with pytest.raises(InitMismatch, match='already'):
        cfg = _r2_cfg(path)                                                    # relabel True (default)
        preload_buffer(build_model(cfg, DummyVecEnv([TinyEnv])), default_init_buffer(path), path, cfg)
    cfg = _r2_cfg(path, relabel=False)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    info = preload_buffer(model, default_init_buffer(path), path, cfg)
    rb = model.replay_buffer
    assert type(rb) is BonusNStepReplayBuffer and info['adopted'] and info['labelled_fraction'] == 0.
    assert info['relabel'] is None and rb.pos == 600 and not rb.bonuses.any()
    assert np.array_equal(rb.rewards[:599], src.replay_buffer.rewards[:599])


# --- R2-10.11: evaluation (q_vs_mc with w b), best / best_aided, keeper ----------------------------------
def test_evaluate_model_uses_r_plus_w_b(monkeypatch):
    from rlsetup.driving import evaluate as E
    table = synthetic_table()
    T = 5

    def fake_lap(model, env, *, deterministic, seed, **_):
        tr = {k: np.zeros(T) for k in E.TRACE_KEYS}
        tr.update(reward=np.arange(1., T + 1), corner_bonus=np.array([0., 2., 0., 1., 0.]),
                  s=np.array([0., 1., 2., 3., 800.]), lap_index=np.array([0, 1, 2, 2, 2]),
                  actions=np.zeros((T, 3), np.float32), kappa=np.zeros((T, 4)), Fz=np.zeros((T, 4)),
                  time_s=np.arange(T) * .04, speed=np.full(T, 50.), progress_m=np.arange(T) * 2.)
        ev = [{'unit': 0, 'kind': 'paid', 'v_exit': 45., 'v_qss_exit': 55., 'rho': .82, 'bonus': 2., 'lap_index': 2,
               'decision': 1}]
        return E.LapReport(deterministic, [100., 101.] if deterministic else [], [], 'lap_complete' if deterministic
                           else 'off_track', None, T, float(tr['reward'].sum()), tr, np.zeros((T + 1, OBS), np.float32),
                           np.zeros(T), ev)
    seen = []
    monkeypatch.setattr(E, 'flying_lap', fake_lap)
    monkeypatch.setattr(E, 'q_vs_mc', lambda critic, o, a, r, logp, **k: seen.append(np.array(r)) or
                        {'k': np.zeros(0), 'q_min': np.zeros(0), 'q_mean': np.zeros(0), 'mc': np.zeros(0),
                         'mc_nstep': np.zeros(0), 'n_steps': 3})
    env = SimpleNamespace(corners=table, _tracker=object(), unwrapped=None)
    env.unwrapped = env
    model = SimpleNamespace(alpha=.05, gamma=.992, n_steps=3, _n_updates=0, bonus_weight=.5, critic=None,
                            policy=SimpleNamespace(critic=None))
    out = E.evaluate_model(model, {'eval_det': env, 'eval_sto': env}, EvalConfig(stochastic_laps=2))
    assert len(seen) == 2 and np.allclose(seen[0], np.arange(1., 6.) + .5 * np.array([0., 2., 0., 1., 0.]))
    assert out['det_return'] == 15. and out['det_bonus_sum'] == 3.
    assert out['lap2_corners'] == {'U1': {'v_exit': 45., 'v_qss_exit': 55., 'rho': .82, 'bonus': 2., 'kind': 'paid'}}
    assert out['lap2_bonus_sum'] == 2. and out['q_vs_mc_bonus_w'] == .5
    assert [s['fail_s'] for s in out['sto_laps']] == [800., 800.]
    assert out['sto_corners']['U1'] == {'passes': 2, 'crashes': 2} and out['sto_corners']['U2']['passes'] == 0


def test_best_and_best_aided_gating(tmp_path):
    """Amendment-1 flow on the eval callback: aided evaluations -> best_aided.zip; the first clean
    lap 2 <= 95 s triggers (w = 0, report_from = trigger + gap, best_aided.zip); the run stops
    itself at report_from with a final reportable evaluation -> best.zip."""
    model = make_model(fill=0)
    laps = {0: 101.9, 25: 101.5, 50: 101.7, 75: 94.0, 95: 96.}
    cfg = BonusScheduleConfig(hold_until=40, fade_until=80, bonus_free_before_report=20)    # report_from 100
    sched = BonusScheduleCallback(cfg, tmp_path / 'bonus_state.json')
    cb = DrivingEvalCallback(lambda m, n: {'det_clean': True, 'lap2_s': laps[n]}, 25, tmp_path / 'evals.jsonl',
                             tmp_path, schedule=sched)
    for c in (sched, cb):
        c.init_callback(model)
    model.num_timesteps = 0
    sched.on_training_start({}, {})
    cb.on_training_start({}, {})
    for n in range(1, 200):
        model.num_timesteps = n
        go = sched.on_step()
        cb.on_step()
        if not go:
            break
    cb.on_training_end()
    recs = runs.read_jsonl(tmp_path / 'evals.jsonl')
    assert [r['step'] for r in recs] == [0, 25, 50, 75, 95] and sched.stopped
    assert [r['bonus_w'] for r in recs] == [1., 1., .75, .125, 0.]
    assert [r['reportable'] for r in recs] == [False, False, False, False, True]
    assert recs[3]['aided_target_lap'] and recs[3]['report_from'] == 95 and sched.trigger_step == 75
    assert [bool(r.get('new_best_aided')) for r in recs] == [True, True, False, True, False]
    assert [bool(r.get('new_best')) for r in recs] == [False, False, False, False, True]
    assert (tmp_path / 'best_aided.zip').exists() and (tmp_path / 'best.zip').exists()
    assert cb.best_aided_lap2 == 94. and cb.best_lap2 == 96.
    assert model.bonus_weight == 0. and model.bonus_trigger_step == 75


def test_best_zip_from_reportable_only(tmp_path):
    model = make_model(fill=0)
    cfg = BonusScheduleConfig(hold_until=10, fade_until=20, bonus_free_before_report=30)   # report_from 50
    sched = BonusScheduleCallback(cfg)
    cb = DrivingEvalCallback(lambda m, n: {'det_clean': True, 'lap2_s': 100. - n / 100}, 25, tmp_path / 'e.jsonl',
                             tmp_path, schedule=sched)
    cb.init_callback(model)
    sched.init_callback(model)
    model.num_timesteps = 0
    cb.on_training_start({}, {})
    for n in (25, 50, 75):
        model.num_timesteps = n
        cb.evaluate()
    recs = runs.read_jsonl(tmp_path / 'e.jsonl')
    assert [r['reportable'] for r in recs] == [False, False, True, True]
    assert [bool(r.get('new_best')) for r in recs] == [False, False, True, True]
    assert cb.best_lap2 == 99.25 and cb.best_aided_lap2 == 99.75
    plain = DrivingEvalCallback(lambda m, n: {'det_clean': True, 'lap2_s': 100.}, 25, tmp_path / 'p.jsonl', tmp_path)
    plain.init_callback(model)
    model.num_timesteps = 0
    plain.on_training_start({}, {})
    rec = plain.records[0]
    assert rec['new_best'] and 'reportable' not in rec and 'bonus_w' not in rec        # pre-R2 behaviour


def _keeper():
    spec = importlib.util.spec_from_file_location('keeper_check', REPO_ROOT / 'scripts' / 'keeper_check.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_keeper_checks(tmp_path):
    K = _keeper()
    ev = lambda **r: runs.append_jsonl(tmp_path / 'evals.jsonl', r)
    ev(step=0, det_clean=True, lap2_s=94.)                  # pre-R2 record: reportable
    assert K.reached(tmp_path) and not K.stopped(tmp_path)
    (tmp_path / 'evals.jsonl').unlink()
    ev(step=0, det_clean=True, lap2_s=101.9, reportable=False)
    ev(step=50, det_clean=True, lap2_s=94.5, reportable=False, aided_target_lap=True)
    ev(step=100, det_clean=False, lap2_s=None, reportable=True)
    ev(step=150, det_clean=True, lap2_s=95.5, reportable=True)
    assert not K.reached(tmp_path)
    ev(step=200, det_clean=True, lap2_s=95.0, reportable=True)
    assert K.reached(tmp_path) and not K.reached(tmp_path, 94.9)
    runs.write_json(tmp_path / 'bonus_state.json', {'trigger_step': 50, 'report_from': 4_000_050})
    assert not K.stopped(tmp_path)
    ev(step=4_000_050, det_clean=False, lap2_s=None, reportable=True)
    assert K.stopped(tmp_path)
    (tmp_path / 'bonus_state.json').unlink()
    runs.update_status(tmp_path, stop_reason='report_complete')
    assert K.stopped(tmp_path)


def test_evals_to_tb_r2_scalars_and_lifetime(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('evals_to_tb', REPO_ROOT / 'scripts' / 'evals_to_tb.py')
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    written = []

    class W:
        def __init__(self, d):
            self.d = d

        def add_scalar(self, tag, v, step, walltime=None):
            written.append((self.d.split('/')[-1], tag, v, step))

        def close(self):
            pass
    monkeypatch.setattr(M, 'SummaryWriter', W)
    rec = {'step': 50_000, 'utc': '2026-11-07T10:00:00Z', 'det_clean': True, 'lap2_s': 101.5, 'bonus_w': 1.,
           'reportable': False, 'lap2_bonus_sum': 12.4, 'lap2_corners': {'U1': {'rho': .91}},
           'sto_corners': {'U3': {'passes': 2, 'crashes': 1}}, 'buffer_labelled_fraction': .05,
           'focus': {'share_resets': .26, 'share_transitions': .2, 'top5': [[2350, .3]], 'r_hot': {'2350': .01}},
           'train_groups': {'low_noise': {'failure_rate': .1, 'episodes': 9, 'bonus_per_episode': 5.}}}
    runs.append_jsonl(tmp_path / 'evals.jsonl', rec)
    runs.update_status(tmp_path, lifetime_steps_offset=9_200_000)
    M.main(tmp_path)
    tags = {(d, t): (v, s) for d, t, v, s in written}
    for t in ('bonus/w', 'bonus/lap2_sum', 'corner/rho_U1', 'corner/sto_crash_U3', 'focus/share_resets',
              'focus/top1_P', 'lownoise/failure_rate', 'buffer/labelled_fraction', 'lap/deterministic_lap2_s'):
        assert tags[('tb_evals', t)][1] == 50_000 and tags[('tb_evals_lifetime', t)][1] == 9_250_000, t
    assert (tmp_path / 'tb_evals_lifetime' / 'last_step.txt').read_text() == '50000'
    written.clear()
    M.main(tmp_path)                                         # nothing new
    assert written == []


# --- resume wiring (R2-5.3, R2-7.5, R2-9.3, Amendment 1) -------------------------------------------------
def test_resume_restores_r2_state(tmp_path, monkeypatch):
    from test_callbacks import resume_fixture
    from rlsetup.driving import train as T
    c = tiny_cfg()
    cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, **ON))
    monkeypatch.setattr(T, 'corner_table', lambda *a: None)
    d, model, seen = resume_fixture(tmp_path, monkeypatch, cfg=cfg)
    for r in ({'step': 0, 'det_clean': True, 'lap2_s': 101., 'reportable': False},
              {'step': 4_000, 'det_clean': True, 'lap2_s': 99., 'reportable': False},
              {'step': 5_000, 'det_clean': True, 'lap2_s': 100., 'reportable': True}):
        runs.append_jsonl(d / 'evals.jsonl', r)
    runs.write_json(d / 'bonus_state.json', {'trigger_step': 4_000, 'report_from': 8_000})
    assert T.resume_pipeline(d, ['total_timesteps=9000'], argv=['x']) == T.EXIT_OK
    k = seen['callbacks']
    assert k['bonus_state'] == (4_000, 8_000) and k['best_lap2'] == 100. and k['best_aided_lap2'] == 99.
    assert k['focus_state'] is None and k['focus_seed'] is None                       # seed_run None
    assert any('focus_state.json missing' in w['message'] for w in runs.read_json(d / 'status.json')['warnings'])
    runs.write_json(d / 'focus_state.json', {'n': [1.] * 108, 'c': [0.] * 108, 'hot': [], 'updates': 3})
    runs.write_json(d / 'bonus_state.json', {'trigger_step': 1_000, 'report_from': 5_000})   # ended itself
    seen.clear()
    assert T.resume_pipeline(d, ['total_timesteps=9000'], argv=['x']) == T.EXIT_OK
    assert 'callbacks' not in seen and 'model' not in seen


# --- Fix batch (8 Nov): QA R2 MINORs + tests 1-6, Amendment 2 ---------------------------------------------
def test_tracker_handover_exactly_at_exit():
    """QA test 1: a hand-over exactly at s_out counts that exit as passed; the unit is paid one lap
    later (it was never paid before the fix)."""
    tr = tracker()
    tr.reset(1000.)
    assert tr.next[0] == 5000.
    ev = drive(tr, 0., 12_000.)
    assert len(ev) == 2 and all(e['unit'] == 0 for e in ev)
    tr.reset(1000. + 5000.)                            # same point one lap on (mod L)
    assert tr.next[0] == 5000.


def test_env_stall_in_paying_decision_does_not_forfeit(make_env, monkeypatch):
    """QA test 2: a crossing in a decision that ends in a stall (time_pbrs: truncation) is paid;
    only terminal failures forfeit (R2-3.4)."""
    env = make_env(EnvConfig(settle_s=0., reward=RewardConfig(mode='time_pbrs'), **ON), corners=synthetic_table())
    env.reset(seed=3, options={'start_s': 4900., 'start_speed': 3.5})
    fake = {'unit': 0, 'v_exit': 45., 'v_qss_exit': 50., 'rho': .9, 'bonus': 1.8}
    monkeypatch.setattr(env._tracker, 'tick', lambda *a: [dict(fake)])
    brake = np.array([0., -1., 1.], np.float32)
    for _ in range(200):
        _, r, term, trunc, info = env.step(brake)
        if term or trunc:
            break
    assert trunc and not term and info['truncation'] == 'stall' and info['termination'] is None
    assert info['corner_bonus'] >= 1.8 and {e['kind'] for e in info['corner_events']} == {'paid'}
    assert info['step_cost']['penalty'] == 0.


def test_learn_and_finish_report_complete(tmp_path):
    """QA test 3: after a trigger the run stops itself at report_from; _learn_and_finish records
    stop reason 'report_complete' (not 'completed'), exit 0."""
    from rlsetup.driving import train as T
    cfg, model = _sched_model(hold_until=40, fade_until=80, bonus_free_before_report=20)
    cfg = dataclasses.replace(cfg, save_replay_buffer=False)
    d = runs.make_run_dir(tmp_path, 'r', 0)
    run = T.Run(cfg, d, lambda *a: None)
    sched = BonusScheduleCallback(cfg.bonus_schedule, d / 'bonus_state.json', trigger_step=10, report_from=30,
                                  on_event=run.event)
    stub = SimpleNamespace(collapsed=False, triggered=False)
    assert T._learn_and_finish(run, model, [sched], stub, stub, stub, 1_000) == T.EXIT_OK
    st = runs.read_json(d / 'status.json')
    assert st['stop_reason'] == st['state'] == 'report_complete' and st['num_timesteps'] == 30
    assert [e['kind'] for e in st['events']] == ['report_complete'] and (d / 'checkpoints' / 'last.zip').exists()


def test_keeper_pattern_does_not_match_prefix(tmp_path):
    """QA test 4: the keeper's process pattern for f1_r2 matches only --run-name f1_r2 (followed by a
    space or the end), never f1_r2_smoke; checked with Python's re and with the real pgrep -f."""
    import re
    import shutil
    import subprocess
    import sys
    import time
    K = _keeper()
    pat = K.pattern('f1_r2')
    cmd = '/x/.venv/bin/python -m rlsetup.driving.train --set runs_root=/r --preset {p} --seed 0 --run-name {n}{tail}'
    for name, tail, hit in (('f1_r2', ' --init-from a.zip', True), ('f1_r2', '', True),
                            ('f1_r2_smoke', ' --init-from a.zip', False), ('f1_r2_smoke', '', False),
                            ('f1_r2x', '', False)):
        assert bool(re.search(pat, cmd.format(p=name, n=name, tail=tail))) is hit, (name, tail)
    assert K.pattern('a.b') == r'rlsetup\.driving\.train .*--run-name a\.b( |$)'
    out = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts' / 'keeper_check.py'), 'pattern', 'f1_r2'],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == pat
    if shutil.which('pgrep') is None:
        return
    tag = f'kp{tmp_path.name[-12:]}'                      # unique per test run
    script = tmp_path / 'rlsetup.driving.train'           # a sleeping process whose argv mimics train.py
    script.write_text('import time\ntime.sleep(30)\n')
    p = subprocess.Popen([sys.executable, str(script), '--run-name', f'{tag}_smoke', '--seed', '0'])
    try:
        time.sleep(.3)
        def pgrep(name):
            return subprocess.run(['pgrep', '-f', K.pattern(name)], capture_output=True, text=True).stdout.split()
        assert str(p.pid) in pgrep(f'{tag}_smoke') and str(p.pid) not in pgrep(tag)
    finally:
        p.kill()
        p.wait()


def test_keeper_limit_single_source(tmp_path):
    """QA MINOR (95 s in two places): the keeper's limit is the run's bonus_schedule.trigger_lap2_s
    (config.json), the default for pre-R2 configs equals BonusScheduleConfig's."""
    K = _keeper()
    assert K.LIMIT_S_DEFAULT == BonusScheduleConfig().trigger_lap2_s == 95.
    assert K.limit_s(tmp_path) == 95.                                          # no config.json
    (tmp_path / 'config.json').write_text(config_json(PRESETS['f1_best_w35']()))   # pre-R2: block omitted
    assert K.limit_s(tmp_path) == 95.
    cfg = PRESETS['f1_r2']()
    cfg = dataclasses.replace(cfg, bonus_schedule=dataclasses.replace(cfg.bonus_schedule, trigger_lap2_s=96.5))
    (tmp_path / 'config.json').write_text(config_json(cfg))
    assert K.limit_s(tmp_path) == 96.5
    runs.append_jsonl(tmp_path / 'evals.jsonl', {'step': 9_000_000, 'det_clean': True, 'lap2_s': 96., 'reportable': True})
    assert K.reached(tmp_path) and not K.reached(tmp_path, 95.)


def test_focus_weights_pushed_to_subproc_workers(track, qss):
    """QA test 5: FocusedStartCallback (default push = env_method) reaches DrivingEnvs living in
    SubprocVecEnv workers behind VecMonitor; with fraction 1 the workers then plan focused starts."""
    from rlsetup.driving.vec import build_vec_env
    cfg = EnvConfig(corner_bonus=CornerBonusConfig(enabled=True), focus=FocusConfig(enabled=True, fraction=1.))
    vec = build_vec_env(cfg, 2, track, qss, corners=synthetic_table())
    try:
        assert vec.get_attr('focus_weights') == [None, None]
        vec.seed(0)
        vec.reset()
        cb = FocusedStartCallback(FocusUpdateConfig(), float(track.length), 50.)
        cb.model = SimpleNamespace(num_timesteps=0, get_env=lambda: vec)
        assert cb.training_env is vec
        cb._on_training_start()
        p = cb.weights()
        for w in vec.get_attr('focus_weights'):
            assert np.allclose(w, p) and w.shape == (108,)
        plans = vec.env_method('_focus_plan', {})
        assert all(q is not None and 0 <= q['bin'] < 108 for q in plans)
    finally:
        vec.close()


def test_bonus_buffer_weighted_wrapped_multi_env():
    """QA test 6: w != 0 on a wrapped (full) 3-env buffer: the n-step sums of r + w b follow the
    ring (wrap at buffer_size, cut at done / timeout / pos - 1) per sampled env, and equal a plain
    NStepReplayBuffer whose rewards hold r + w b (float32), bit for bit."""
    w = .7
    plain, bonus = _buffers(n_envs=3, rows=100)            # 64 rows per env, pos = 36, full
    assert bonus.full and bonus.pos == 36
    bonus.bonus_weight = w
    size, g, n = bonus.buffer_size, bonus.gamma, bonus.n_steps
    inds = np.concatenate([np.arange(64), np.arange(60, 64), [33, 34, 35]])   # across the wrap and pos - 1
    got = _sample(bonus, inds, seed=5).rewards.numpy()[:, 0]
    np.random.seed(5)
    envs = np.random.randint(0, 3, size=inds.shape)
    for k, (i, e) in enumerate(zip(inds, envs)):
        tot = 0.
        for j in range(n):
            q = (i + j) % size
            tot += g ** j * (bonus.rewards[q, e] + np.float32(w) * bonus.bonuses[q, e])
            if bonus.dones[q, e] or bonus.timeouts[q, e] or q == bonus.pos - 1:
                break
        assert got[k] == pytest.approx(tot, rel=1e-5, abs=1e-5), (i, e)
    plain.rewards = (plain.rewards + np.float32(w) * bonus.bonuses).astype(np.float32)
    assert _samples_equal(_sample(plain, inds, seed=5), _sample(bonus, inds, seed=5))
    assert len(set(envs.tolist())) == 3


def test_result_stretch_after_trigger(tmp_path):
    """Amendment 2: after a trigger, best.zip comes from the clean evaluations with w = 0 in
    [trigger, report_from] (report_from included); the trigger evaluation (w > 0) only gives
    best_aided.zip; bonus_state.json and on_result carry the result block."""
    model = make_model(fill=0)
    laps = {0: 101.9, 25: 94.0, 50: 96.5, 75: None, 100: 95.8, 125: 96.1}
    cfg = BonusScheduleConfig(hold_until=40, fade_until=80, bonus_free_before_report=100)    # trigger at 25
    results = []
    sched = BonusScheduleCallback(cfg, tmp_path / 'bonus_state.json', on_result=results.append)
    cb = DrivingEvalCallback(lambda m, n: {'det_clean': laps[n] is not None, 'lap2_s': laps[n]}, 25,
                             tmp_path / 'evals.jsonl', tmp_path, schedule=sched)
    for c in (sched, cb):
        c.init_callback(model)
    model.num_timesteps = 0
    sched.on_training_start({}, {})
    cb.on_training_start({}, {})
    for n in range(1, 300):
        model.num_timesteps = n
        go = sched.on_step()
        cb.on_step()
        if not go:
            break
    cb.on_training_end()
    recs = runs.read_jsonl(tmp_path / 'evals.jsonl')
    assert [r['step'] for r in recs] == [0, 25, 50, 75, 100, 125] and sched.stopped and sched.report_from == 125
    assert [r.get('result_eval') for r in recs] == [None, False, True, True, True, True]
    assert [r['reportable'] for r in recs] == [False] * 5 + [True]
    assert [bool(r.get('new_best')) for r in recs] == [False, False, True, False, True, False]
    assert [bool(r.get('new_best_aided')) for r in recs] == [True, True, False, False, False, False]
    assert cb.best_lap2 == 95.8 and cb.best_aided_lap2 == 94.
    res = runs.read_json(tmp_path / 'bonus_state.json')['result']
    assert res == {'trigger_step': 25, 'report_from': 125, 'n_evals': 4, 'best_lap2_s': 95.8, 'best_step': 100,
                   'clean_laps': [{'step': 50, 'lap2_s': 96.5}, {'step': 100, 'lap2_s': 95.8},
                                  {'step': 125, 'lap2_s': 96.1}]}
    assert results[-1] == res and len(results) == 5                 # every evaluation from the trigger on
    # resume: the stretch is rebuilt from the records (records written before this fix lack result_eval)
    old = [{k: v for k, v in r.items() if k != 'result_eval'} for r in recs[:4]]
    back = BonusScheduleCallback(cfg, trigger_step=25, report_from=125, history=old)
    assert back.result()['clean_laps'] == res['clean_laps'][:1] and back.result()['n_evals'] == 2
    # trigger after the fade (w already 0): the trigger evaluation itself is in the stretch
    late = BonusScheduleCallback(BonusScheduleConfig(hold_until=10, fade_until=20, bonus_free_before_report=100))
    rec = late.on_eval({'step': 50, 'det_clean': True, 'lap2_s': 94.5})
    assert rec['aided_target_lap'] and rec['bonus_w'] == 0. and rec['result_eval'] and not rec['reportable']
    assert late.result()['best_lap2_s'] == 94.5
    # without a trigger: no result block, best.zip from reportable evaluations only (R2-5 / R2-9.3)
    plain = BonusScheduleCallback(cfg)
    rec = plain.on_eval({'step': 25, 'det_clean': True, 'lap2_s': 96.})
    assert 'result_eval' not in rec and plain.result() is None and 'result' not in plain.state(25)


def test_resume_rebuilds_best_from_result_stretch(tmp_path, monkeypatch):
    """Amendment 2 on resume: best_lap2 from the clean w = 0 evaluations of [trigger, report_from]
    (and reportable ones), best_aided from the others incl. the trigger; the result stretch goes
    to the schedule as eval_history."""
    from test_callbacks import resume_fixture
    from rlsetup.driving import train as T
    c = tiny_cfg()
    cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, **ON))
    monkeypatch.setattr(T, 'corner_table', lambda *a: None)
    d, model, seen = resume_fixture(tmp_path, monkeypatch, cfg=cfg)
    recs = [{'step': 0, 'det_clean': True, 'lap2_s': 101., 'reportable': False, 'bonus_w': 1.},
            {'step': 2_000, 'det_clean': True, 'lap2_s': 94.5, 'reportable': False, 'bonus_w': 1.,
             'aided_target_lap': True},
            {'step': 3_000, 'det_clean': True, 'lap2_s': 96., 'reportable': False, 'bonus_w': 0.},
            {'step': 4_000, 'det_clean': True, 'lap2_s': 95.5, 'reportable': False, 'bonus_w': 0.},
            {'step': 5_000, 'det_clean': False, 'lap2_s': None, 'reportable': False, 'bonus_w': 0.}]
    for r in recs:
        runs.append_jsonl(d / 'evals.jsonl', r)
    runs.write_json(d / 'bonus_state.json', {'trigger_step': 2_000, 'report_from': 8_000})
    assert T.resume_pipeline(d, ['total_timesteps=9000'], argv=['x']) == T.EXIT_OK
    k = seen['callbacks']
    assert k['bonus_state'] == (2_000, 8_000) and k['best_lap2'] == 95.5 and k['best_aided_lap2'] == 94.5
    sched = BonusScheduleCallback(BonusScheduleConfig(), trigger_step=2_000, report_from=8_000,
                                  history=k['eval_history'])
    assert [x['step'] for x in sched.result()['clean_laps']] == [3_000, 4_000] and sched.result()['n_evals'] == 3


def test_f1_r2_presets_warm_start_from_last_zip():
    """Amendment 2: f1_r2 / f1_r2_smoke default to f1_best_w35's last.zip (absolute path)."""
    want = REPO_ROOT / 'runs' / 'mac' / 'D2' / 'f1_best_w35' / 'seed0' / 'checkpoints' / 'last.zip'
    for name in ('f1_r2', 'f1_r2_smoke'):
        cfg = PRESETS[name]()
        assert cfg.init_from == str(want) and cfg.agent.init_mode == 'warm'
        assert train_config_from_dict(json.loads(config_json(cfg))).init_from == str(want)
    from rlsetup.driving.train import parse_args
    from rlsetup.driving.agent import default_init_buffer
    assert default_init_buffer(str(want)) == str(want.parents[1] / 'replay_buffer.pkl')
    assert parse_args(['--preset', 'f1_r2', '--run-name', 'f1_r2']).init_from is None   # preset default used


@needs_src
def test_f1_r2_init_is_12_35M_steps():
    """lifetime_steps_offset = the init checkpoint's num_timesteps (train.py) = 12,350,000 for last.zip."""
    import zipfile
    with zipfile.ZipFile(PRESETS['f1_r2']().init_from) as z:
        assert json.loads(z.read('data'))['num_timesteps'] == 12_350_000
