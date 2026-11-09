"""R4 (docs/specs/R4_speed_profile.md §3 + binding R4_review.md req 1-7): closeness g, speed potential
(telescoping, terminal convention, bit identity when off), corner entry / exit terms with take-back,
exact buffer relabel vs env ground truth (+ abort / fallback pieces), preset f1_r4, never-fade
schedule, keeper, configs, logging.

needs_f1: ocd_first_lap branch sf70h-f1 on PYTHONPATH; needs_src: also runs/mac/D2/f1_best_w35."""
import dataclasses
import importlib.util
import json
import math
import pickle
import zipfile
from types import SimpleNamespace
import numpy as np
import pytest
from ocd_lap import cars
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup import runs
from rlsetup.provenance import REPO_ROOT
from rlsetup.driving import corners as C
from rlsetup.driving import speed_profile as SP
from rlsetup.driving.buffers import BonusNStepReplayBuffer, pre_point_cells
from rlsetup.driving.callbacks import DeferredOfflineCritic, TrainStatsCallback, bonus_weight
from rlsetup.driving.config import EnvConfig, RewardConfig, CornerBonusConfig, FocusConfig
from rlsetup.driving.relabel import relabel_speed_profile
from rlsetup.driving.train_config import (PRESETS, BonusScheduleConfig, EvalConfig, config_json,
                                          train_config_from_dict, F1_R2_INIT, F1_BEST_W35_RUN)
from r1_golden import legacy_rollout
from sb3_helpers import TinyEnv, tiny_cfg, make_model
import speed_truth as ST

HAS_F1 = 'sf70h_f1' in cars.CARS
needs_f1 = pytest.mark.skipif(not HAS_F1, reason='sf70h_f1 needs ocd_first_lap branch sf70h-f1 on the path')
RUNS = REPO_ROOT / 'runs' / 'mac' / 'D2'
SRC = RUNS / 'f1_best_w35' / 'seed0'
needs_src = pytest.mark.skipif(not (HAS_F1 and (SRC / 'checkpoints' / 'last.zip').exists()),
                               reason='needs sf70h_f1 and runs/mac/D2/f1_best_w35')
GOLDEN_LEGACY = '0007e2813d2d28971223460790b45852513768865f08d6d0b3685f5868a1c4c2'   # = test_r1.py
# time_pbrs digest of legacy_rollout(EnvConfig(max_episode_s=4, reward=time_pbrs)), generated with the
# pre-R4 code (commit ad3b879, both ocd_first_lap branches)
GOLDEN_PBRS = '3104af11981e51be74c2a9475c402e9c273340794ccd4dc62baf836e13e9e7aa'
TABLE_KEY = '6c61b28a4876'
E3 = CornerBonusConfig(enabled=True, mode='entry_exit')


@pytest.fixture(scope='module')
def truth(make_env, track, qss):
    """Ground-truth buffers of speed_truth.collect (+ a plain time_pbrs env with the same table)."""
    return ST.collect(make_env, track, qss)


# --- g ---------------------------------------------------------------------------------------------------
def test_closeness_symmetric_zero_beyond_sigma_one_at_target():
    v = 60.
    assert SP.closeness(v, v, .3) == 1.
    for d in (.05, .1, .2, .29):
        assert SP.closeness(v * (1 + d), v, .3) == pytest.approx(SP.closeness(v * (1 - d), v, .3), abs=1e-12)
        assert SP.closeness(v * (1 + d), v, .3) == pytest.approx(1 - (d / .3) ** 2)
    for d in (.3, .31, .5, 1., 2.):
        assert SP.closeness(v * (1 + d), v, .3) == 0. and SP.closeness(v * max(0., 1 - d), v, .3) == 0.
    x = np.linspace(0., 150., 301)
    g = SP.closeness(x, v, .3)
    assert g.shape == x.shape and np.all(g >= 0.) and np.all(g <= 1.)
    assert np.allclose(SP.closeness_rho(x / v, .3), g)


def test_config_validation():
    with pytest.raises(ValueError, match='time_pbrs'):
        RewardConfig(speed_k=1.)                                      # ref_tracking
    for kw in (dict(speed_k=-1.), dict(speed_k=math.inf), dict(speed_sigma=0.), dict(speed_sigma=-.1)):
        with pytest.raises(ValueError):
            RewardConfig(mode='time_pbrs', **kw)
    for kw in (dict(mode='both'), dict(sigma=0.), dict(bonus_in=-1.), dict(bonus_out=math.nan)):
        with pytest.raises(ValueError):
            CornerBonusConfig(**kw)
    on = EnvConfig(reward=RewardConfig(mode='time_pbrs', speed_k=20.), corner_bonus=E3)
    assert on.speed_on and on.entry_exit_on and on.r4 and not EnvConfig().r4


# --- req 1: configs ----------------------------------------------------------------------------------------
def test_to_dict_omits_r4_defaults_and_round_trips():
    d = EnvConfig().to_dict()
    assert 'speed_k' not in d['reward'] and 'speed_sigma' not in d['reward']
    ex = EnvConfig(corner_bonus=CornerBonusConfig(enabled=True)).to_dict()
    assert not {'mode', 'bonus_in', 'bonus_out', 'sigma'} & set(ex['corner_bonus'])
    on = EnvConfig(reward=RewardConfig(mode='time_pbrs', speed_k=20.), corner_bonus=E3).to_dict()
    assert on['reward']['speed_k'] == 20. and on['reward']['speed_sigma'] == .3
    assert on['corner_bonus']['mode'] == 'entry_exit' and on['corner_bonus']['bonus_in'] == 10.
    for name, make in PRESETS.items():
        cfg = make()
        j = json.loads(config_json(cfg))
        assert train_config_from_dict(j) == cfg, name
        if name not in ('f1_r4', 'f1_r4_scratch'):     # no R4 key in any pre-R4 preset's config.json
            text = config_json(cfg)
            for k in ('"speed_k"', '"speed_sigma"', '"bonus_in"', '"bonus_out"', '"mode": "e', '"speed_relabel"',
                      '"fallback_steps"', '"probe_before_m"', '"full_throttle_pedal"', '"brake_pedal"'):
                assert k not in text, (name, k)


@pytest.mark.parametrize('run', ['f1_best_w35', 'f1_r3'])
def test_real_run_configs_round_trip_byte_equal(run):
    p = RUNS / run / 'seed0' / 'config.json'
    if not p.exists():
        pytest.skip(f'needs {p}')
    text = p.read_text()
    assert config_json(train_config_from_dict(json.loads(text))) == text


def test_real_f1_r2_config_differs_only_by_the_r3_keys():
    """f1_r2's config.json was written before R3, which added focus_update.init_state and the
    warm.pre_exit_* keys to aid presets (commit 3228c91): those four lines are the only difference
    (pre-existing; R4 adds none)."""
    p = RUNS / 'f1_r2' / 'seed0' / 'config.json'
    if not p.exists():
        pytest.skip(f'needs {p}')
    import difflib
    text = p.read_text()
    new = config_json(train_config_from_dict(json.loads(text)))
    added = [l for l in difflib.ndiff(text.splitlines(), new.splitlines()) if l[:2] in ('+ ', '- ')]
    assert sorted(l[0] + ' ' + l[2:].strip() for l in added) == sorted(['+ "init_state": null,', '+ "pre_exit_horizon": 30,',
                                                       '+ "pre_exit_warn_blocks": 5,', '+ "pre_exit_warn_q": 0.5,'])


# --- req 2: speed potential in the env ------------------------------------------------------------------
def test_golden_rollouts_unchanged(make_env, track, qss):
    table = ST.table_for(track, qss)
    assert legacy_rollout(make_env)[0] == GOLDEN_LEGACY
    off = EnvConfig(max_episode_s=4., reward=RewardConfig(mode='time_pbrs', speed_sigma=.5))   # speed_k = 0
    assert legacy_rollout(make_env, off)[0] == GOLDEN_PBRS
    assert legacy_rollout(make_env, EnvConfig(max_episode_s=4., reward=RewardConfig(mode='time_pbrs')))[0] == GOLDEN_PBRS
    # entry / exit terms on, speed_k = 0: the reward and observations stay bit-identical
    with_terms = dataclasses.replace(off, corner_bonus=E3)
    assert legacy_rollout(lambda c: make_env(c, corners=table), with_terms)[0] == GOLDEN_PBRS
    assert legacy_rollout(lambda c: make_env(dataclasses.replace(c, corner_bonus=E3), corners=table))[0] == GOLDEN_LEGACY


def _episode(env, options, policy, seed=1, n=600):
    obs, info = env.reset(seed=seed, options=options)
    out = {'phi0': info['phi_v'], 'g0': info['g_v'], 'F': [], 'phi': [], 'r': [], 'info': []}
    for k in range(n):
        obs, r, te, tr, info = env.step(policy(env, k))
        out['F'].append(info['shaping_v'])
        out['phi'].append(info['phi_v'])
        out['r'].append(r)
        out['info'].append(info)
        if te or tr:
            break
    return out


@pytest.mark.parametrize('kind', ['time_limit', 'crash'])
def test_speed_potential_telescopes_and_terminal_convention(make_env, track, qss, kind):
    table = ST.table_for(track, qss)
    k, sigma, gamma = 20., .3, RewardConfig().gamma
    on = make_env(EnvConfig(max_episode_s=4., settle_s=0., reward=RewardConfig(mode='time_pbrs', speed_k=k)),
                  corners=table)
    off = make_env(EnvConfig(max_episode_s=4., settle_s=0., reward=RewardConfig(mode='time_pbrs')))
    crash = np.array([1., 1., -1.], np.float32)
    pol = (lambda e, i: ST._pursuit(e, table, 1.)) if kind == 'time_limit' else \
        (lambda e, i: ST._pursuit(e, table, 1.) if i < 30 else crash)
    opts = {'start_s': 4530., 'start_speed': 25.}
    a = _episode(on, opts, pol)
    acts = []
    b_r = []
    off.reset(seed=1, options=opts)
    on.reset(seed=1, options=opts)
    for i in range(len(a['F'])):
        act = pol(on, i)
        acts.append(act)
        r1 = on.step(act)
        r0 = off.step(act)
        b_r.append(r1[1] - r0[1] - r1[4]['shaping_v'])
        assert r1[0].tobytes() == r0[0].tobytes()
    assert np.max(np.abs(b_r)) < 1e-9                       # reward = time_pbrs reward + F_v exactly
    T = len(a['F'])
    disc = sum(gamma ** t * f for t, f in enumerate(a['F']))
    assert disc == pytest.approx(gamma ** T * a['phi'][-1] - a['phi0'], abs=1e-9)
    last = a['info'][-1]
    if kind == 'crash':
        assert last['termination'] == 'off_track' and a['phi'][-1] == 0. and a['F'][-1] == pytest.approx(-a['phi'][-2])
    else:
        assert last['truncation'] == 'time_limit' and a['phi'][-1] > 0.
        assert a['phi'][-1] == pytest.approx(k * last['g_v']) and 0. < last['g_v'] <= 1.
        assert last['episode_shaping_v'] == pytest.approx(sum(a['F']), abs=1e-9)


def test_phi_v_uses_track_s_and_speed(make_env, track, qss):
    table = ST.table_for(track, qss)
    env = make_env(EnvConfig(settle_s=0., reward=RewardConfig(mode='time_pbrs', speed_k=20.)), corners=table)
    _, info = env.reset(seed=1, options={'start_s': 4530., 'start_speed': 25.})
    v = math.hypot(env.state['vx'], env.state['vy'])
    assert info['phi_v'] == pytest.approx(20. * SP.closeness(v, float(table.speed(env.s)), .3), abs=1e-12)


# --- req 3: entry / exit tracker -------------------------------------------------------------------------
def _table(units=((800., 1000.),), L=5000., v=50.):
    t = C.CornerTable(L, np.arange(int(L)) * 1., np.full(int(L), v), [], 90.)
    t.units = [{'name': f'U{k + 1}', 'apexes_m': [(a + b) / 2], 'v_apex': [20.], 'start_m': a, 'exit_m': b,
                'v_qss_exit': v} for k, (a, b) in enumerate(units)]
    return t


def _drive(tr, d0, d1, v=45., step=2., fail_at=None):
    """Decisions of `step` m at speed v; returns [(total, events)] per decision."""
    out, d = [], d0
    while d < d1:
        ev = tr.tick(d, d + step, v, v)
        failed = fail_at is not None and d + step >= fail_at
        out.append(tr.settle(ev, failed))
        d += step
        if failed:
            break
    return out


def test_entry_exit_paid_once_per_pass():
    tr = SP.EntryExitTracker(_table(), 10., 10., .3)
    tr.reset(0.)
    res = _drive(tr, 0., 15_000.)
    ev = [e for _, es in res for e in es]
    assert [(e['point'], e['kind']) for e in ev] == [('entry', 'paid'), ('exit', 'paid')] * 3
    g = SP.closeness(45., 50., .3)
    assert ev[0]['g'] == pytest.approx(g) and ev[0]['bonus'] == pytest.approx(10. * g) and ev[0]['rho'] == .9
    assert sum(t for t, _ in res) == pytest.approx(6 * 10. * g)
    assert not tr.pending.any() and tr.n_taken_back == 0


def test_crash_after_entry_takes_back_and_forfeits():
    g = SP.closeness(45., 50., .3)
    tr = SP.EntryExitTracker(_table(), 10., 10., .3)
    tr.reset(0.)
    res = _drive(tr, 0., 2000., fail_at=900.)                  # entry 800 paid, crash at 900 before exit 1000
    totals = [t for t, _ in res]
    assert totals.count(pytest.approx(10. * g)) == 1 and totals[-1] == pytest.approx(-10. * g)
    assert sum(totals) == pytest.approx(0., abs=1e-12)
    assert res[-1][1] == [{'unit': 0, 'point': 'entry', 'bonus': pytest.approx(-10. * g), 'kind': 'taken_back'}]
    assert tr.n_taken_back == 1 and tr.taken_back == pytest.approx(10. * g)
    tr.reset(0.)                                                # crossing inside the failing decision: forfeit
    res = _drive(tr, 2., 2000., step=4., fail_at=801.)          # decision 798 -> 802 fails
    assert res[-1][0] == 0. and [e['kind'] for e in res[-1][1]] == ['forfeit'] and tr.n_taken_back == 0
    tr.reset(0.)                                                # crash after the exit: nothing taken back
    res = _drive(tr, 0., 2000., fail_at=1100.)
    assert sum(t for t, _ in res) == pytest.approx(20. * g) and tr.n_taken_back == 0 and res[-1][0] == 0.
    tr.reset(0.)                                                # exit crossed in the failing decision
    res = _drive(tr, 2., 2000., step=4., fail_at=1001.)         # 998 -> 1002 fails
    assert [e['kind'] for e in res[-1][1]] == ['forfeit', 'taken_back'] and sum(t for t, _ in res) == pytest.approx(0.)


def test_truncation_takes_nothing_back():
    tr = SP.EntryExitTracker(_table(), 10., 10., .3)
    tr.reset(0.)
    res = _drive(tr, 0., 900.)                                  # entry paid, episode truncated before the exit
    total, events = tr.settle([], False)                        # the truncating decision: not a failure
    assert total == 0. and events == [] and sum(t for t, _ in res) > 0 and tr.n_taken_back == 0


def test_handover_at_inside_and_before_entry():
    for s_h, want in ((800., ['exit']), (850., ['exit']), (799., ['entry', 'exit']), (1000., ['entry'])):
        tr = SP.EntryExitTracker(_table(), 10., 10., .3)
        tr.reset(s_h)
        ev = [e for _, es in _drive(tr, 0., 4900.) for e in es]   # less than one lap
        assert [e['point'] for e in ev] == want, s_h
    tr = SP.EntryExitTracker(_table(), 10., 10., .3)
    tr.reset(1000.)                                             # at the exit: next pass one lap on (entry first)
    ev = [e for _, es in _drive(tr, 0., 4900.) for e in es]
    assert [e['point'] for e in ev] == ['entry'] and tr.pending[0] > 0


def test_unpaid_outside_sigma_and_window_through_zero():
    tr = SP.EntryExitTracker(_table(((4900., 100.),)), 10., 10., .3)
    tr.reset(4800.)
    res = _drive(tr, 0., 400., v=30.)                           # rho 0.6: g = 0
    ev = [e for _, es in res for e in es]
    assert [(e['point'], e['kind']) for e in ev] == [('entry', 'unpaid'), ('exit', 'unpaid')]
    assert sum(t for t, _ in res) == 0.


def test_env_entry_exit_terms_and_reward_unchanged(make_env, track, qss):
    table = ST.table_for(track, qss)
    base = EnvConfig(max_episode_s=20., settle_s=0., reward=RewardConfig(mode='time_pbrs', speed_k=20.))
    env = make_env(dataclasses.replace(base, corner_bonus=E3), corners=table)
    ref = make_env(base, corners=table)
    by_name = {name: (seed, opts, pol) for name, seed, opts, pol in ST.scenarios()}
    for name in ('crash', 'stall', 'wrap'):
        seed, opts, pol = by_name[name]
        env.reset(seed=seed, options=opts)
        ref.reset(seed=seed, options=opts)
        evs = []
        for k in range(1000):
            a = np.asarray(pol(env, table, k), np.float32)
            o1, r1, te, tr, i1 = env.step(a)
            o0, r0, *_ = ref.step(a)
            assert o1.tobytes() == o0.tobytes() and r1 == r0      # terms never enter the env reward
            evs += i1.get('corner_events', [])
            if te or tr:
                break
        if name == 'crash':
            assert i1['termination'] == 'off_track'
            assert [(e['point'], e['kind']) for e in evs] == [('entry', 'paid'), ('entry', 'taken_back')]
            assert i1['corner_bonus'] == pytest.approx(-evs[0]['bonus']) and i1['episode_corner_bonus'] == pytest.approx(0.)
            assert i1['episode_corner_takebacks'] == 1 and i1['episode_corner_taken_back'] == pytest.approx(evs[0]['bonus'])
        elif name == 'stall':
            assert i1['truncation'] == 'stall' and [e['kind'] for e in evs] == ['paid']
            assert i1['episode_corner_bonus'] == pytest.approx(evs[0]['bonus']) and i1['episode_corner_takebacks'] == 0
        else:
            assert [(e['unit'], e['point']) for e in evs] == [(4, 'entry'), (4, 'exit'), (1, 'entry')]
            assert all(e['kind'] == 'paid' for e in evs)
            assert table.units[4]['exit_m'] < table.units[4]['start_m']            # window through s = 0


# --- req 4: buffer relabel vs env ground truth ----------------------------------------------------------
def test_relabel_matches_env_ground_truth(truth, make_env):
    src, dst, s_true, names, table, (cs, cd), events, _ = truth
    rb = pickle.loads(pickle.dumps(src))
    BonusNStepReplayBuffer.adopt(rb)
    grid = make_env(cd, corners=table).grid
    r_before = rb.rewards.copy()
    stats, s = relabel_speed_profile(rb, cs, cd, table, grid)
    n = rb.pos
    assert stats['unresolved_transitions'] == 0 and stats['chains'] == stats['chains_resolved'] == len(set(names))
    assert stats['integrity_breaks'] == 0 and not stats['aborted']
    ds = np.abs((s[:n, 0] - s_true + table.length / 2) % table.length - table.length / 2)
    assert ds.max() <= .01
    f_v = rb.rewards[:n, 0].astype(np.float64) - r_before[:n, 0].astype(np.float64)
    f_true = dst.rewards[:n, 0].astype(np.float64) - src.rewards[:n, 0].astype(np.float64)
    assert np.abs(f_v - f_true).max() <= 1e-4
    assert np.abs(rb.rewards[:n, 0] - dst.rewards[:n, 0]).max() <= 1e-4
    assert np.abs(rb.bonuses[:n, 0] - dst.bonuses[:n, 0]).max() <= 2e-3
    kinds = {f"{e['point']}_{e['kind']}" for _, e in events}
    assert {'entry_paid', 'exit_paid', 'entry_taken_back'} <= kinds          # scenarios cover the cases
    assert (rb.bonuses[:n, 0] < 0).sum() == 1 and stats['terms']['rows_negative'] == 1
    assert np.array_equal(rb.dones, src.dones) and np.array_equal(rb.timeouts, src.timeouts)
    assert not rb.labelled[:n].any()                                          # relabel does not mark rows


def test_relabel_wrapped_buffer_and_abort(truth, make_env):
    """Rows of a full (wrapped) buffer are ordered from rb.pos (oldest row mid-episode); a chain with
    only a straight preview
    stays unresolved, and with max_unresolved below its share nothing is written."""
    src, dst, s_true, names, table, (cs, cd), _, _ = truth
    from stable_baselines3.common.buffers import NStepReplayBuffer
    n, k0, shift = src.pos, 30, 700        # the oldest stored row is row 30 of the first episode
    m = n - k0
    rb = NStepReplayBuffer(m, src.observation_space, src.action_space, device='cpu', n_envs=1, n_steps=3,
                           gamma=ST.GAMMA, handle_timeout_termination=True)
    for k in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
        getattr(rb, k)[:] = np.roll(getattr(src, k)[k0:n], -shift, axis=0)
    rb.full, rb.pos = True, m - shift      # chronological order starts at pos
    BonusNStepReplayBuffer.adopt(rb)
    grid = make_env(cd, corners=table).grid
    stats, s = relabel_speed_profile(rb, cs, cd, table, grid)
    want = np.roll(dst.rewards[k0:n], -shift, axis=0)
    ok = np.isfinite(s[:, 0])
    assert stats['chains'] == len(set(names)) and ok.all() and stats['integrity_breaks'] == 0
    assert np.abs(rb.rewards[:, 0] - want[:, 0]).max() <= 1e-4
    st = np.roll(s_true[k0:n], -shift)
    assert np.abs((s[:, 0] - st + table.length / 2) % table.length - table.length / 2).max() <= .01
    # abort: a straight-only chain (main straight, 300 m preview without a curve)
    env = make_env(ST.configs()[0], corners=table)
    venv = DummyVecEnv([lambda: env])
    straight = NStepReplayBuffer(200, src.observation_space, src.action_space, device='cpu', n_envs=1, n_steps=3,
                                 gamma=ST.GAMMA, handle_timeout_termination=True)
    venv.seed(9)
    venv.set_options({'start_s': 5300., 'start_speed': 30.})
    o = venv.reset()
    for k in range(40):
        a = np.asarray(env.settle_driver.act(env.state, env.s_drive, 30.), np.float32)[None]
        new, r, d, infos = venv.step(a)
        straight.add(o, new, a, r, d, infos)
        o = new
    BonusNStepReplayBuffer.adopt(straight)
    before = straight.rewards.copy()
    stats, s = relabel_speed_profile(straight, cs, cd, table, grid, max_unresolved=.001)
    assert stats['aborted'] and stats['unresolved_fraction'] == 1. and np.isnan(s[:40]).all()
    assert np.array_equal(straight.rewards, before) and not straight.bonuses.any()


def test_relabel_refusals(truth, make_env):
    from rlsetup.driving.relabel import _check_speed_relabel
    _, _, _, _, table, (cs, cd), _, _ = truth
    for bad_src, bad_dst in ((dataclasses.replace(cs, reward=RewardConfig()), cd),
                             (cd, cd),                                            # source already has Phi_v
                             (cs, dataclasses.replace(cd, reward=dataclasses.replace(cd.reward, shaping_k=2.))),
                             (cs, dataclasses.replace(cd, action_repeat=1))):
        with pytest.raises(ValueError):
            _check_speed_relabel(bad_src, bad_dst)


def test_pre_point_cells():
    s = np.array([[100., np.nan], [960., 990.], [999.9, 1000.], [4990., 30.]])
    rows, cols = pre_point_cells(s, [1000., 50.], 5000., 150.)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]


# --- preload_buffer integration (agent.py) ------------------------------------------------------------
def _src_run(tmp_path, truth):
    src, _, _, _, _, (cs, _), _, _ = truth
    from rlsetup.driving.train_config import TrainConfig
    d = tmp_path / 'src' / 'seed0'
    (d / 'checkpoints').mkdir(parents=True)
    (d / 'config.json').write_text(config_json(dataclasses.replace(TrainConfig(), env=cs)))
    with open(d / 'replay_buffer.pkl', 'wb') as f:
        pickle.dump(src, f)
    return d / 'checkpoints' / 'last.zip', d / 'replay_buffer.pkl'


def _r4_cfg(truth, **warm):
    _, _, _, _, _, (_, cd), _, _ = truth
    c = tiny_cfg()
    c = dataclasses.replace(c, agent=dataclasses.replace(c.agent, buffer_size=6000))
    return dataclasses.replace(c, env=cd, bonus_schedule=BonusScheduleConfig(hold_until=None),
                               agent=dataclasses.replace(c.agent, init_mode='warm'),
                               warm=dataclasses.replace(c.warm, relabel=False, speed_relabel=True, **warm))


def test_preload_buffer_speed_relabel(tmp_path, truth, make_env):
    from rlsetup.driving.agent import build_model, preload_buffer, InitMismatch, RelabelUnresolved
    src, dst, _, _, table, (cs, cd), _, _ = truth
    init_from, path = _src_run(tmp_path, truth)
    grid = make_env(cd, corners=table).grid
    cfg = _r4_cfg(truth)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    aux = {}
    info = preload_buffer(model, path, init_from, cfg, table=table, grid=grid, aux=aux)
    rb = model.replay_buffer
    n = src.pos
    assert info['adopted'] and info['speed_relabel']['unresolved_transitions'] == 0 and info['relabel'] is None
    assert np.abs(rb.rewards[:n, 0] - dst.rewards[:n, 0]).max() <= 1e-4
    assert np.abs(rb.bonuses[:n, 0] - dst.bonuses[:n, 0]).max() <= 2e-3
    assert aux['s'].shape == (6000, 1) and np.isfinite(aux['s'][:n]).all() and np.isnan(aux['s'][n:]).all()
    json.dumps(info)                                                # JSON-ready (status.json / provenance)
    with pytest.raises(ValueError, match='corner table'):
        preload_buffer(build_model(cfg, DummyVecEnv([TinyEnv])), path, init_from, cfg)
    both = _r4_cfg(truth)
    both = dataclasses.replace(both, warm=dataclasses.replace(both.warm, relabel=True))
    with pytest.raises(InitMismatch, match='cannot be combined'):
        preload_buffer(build_model(both, DummyVecEnv([TinyEnv])), path, init_from, both, table=table, grid=grid)
    # source already has the speed potential: refused before the load
    bad_dir = tmp_path / 'bad' / 'seed0'
    (bad_dir / 'checkpoints').mkdir(parents=True)
    from rlsetup.driving.train_config import TrainConfig
    (bad_dir / 'config.json').write_text(config_json(dataclasses.replace(TrainConfig(), env=cd)))
    junk = bad_dir / 'replay_buffer.pkl'
    junk.write_bytes(b'not a pickle')
    with pytest.raises(InitMismatch, match='speed'):
        preload_buffer(build_model(cfg, DummyVecEnv([TinyEnv])), junk, bad_dir / 'checkpoints' / 'last.zip', cfg,
                       table=table, grid=grid)
    # unresolved above the limit: RelabelUnresolved, the model keeps its own empty buffer
    tight = _r4_cfg(truth, speed_relabel_max_unresolved=-1.)
    m2 = build_model(tight, DummyVecEnv([TinyEnv]))
    fresh = m2.replay_buffer
    with pytest.raises(RelabelUnresolved) as e:
        preload_buffer(m2, path, init_from, tight, table=table, grid=grid)
    assert e.value.stats['aborted'] and m2.replay_buffer is fresh and fresh.size() == 0


# --- fallback pieces ---------------------------------------------------------------------------------------
def test_deferred_offline_critic_runs_once_after_collection():
    from stable_baselines3.common.logger import configure
    from rlsetup.driving.agent import build_model
    cfg = tiny_cfg()
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    model.set_logger(configure(None, []))
    model.min_train_transitions = 300
    seen = []
    cb = DeferredOfflineCritic(300, lambda: seen.append((model.num_timesteps, model._n_updates, model.min_train_transitions)))
    model.learn(450, callback=[cb])
    assert seen == [(300, 0, 0)] and cb.done                  # no gradient step before it, once
    assert model._n_updates > 0


def test_offline_critic_probes_by_recovered_s(tmp_path):
    from rlsetup.driving import train as T
    c = tiny_cfg()
    cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, reward=RewardConfig(mode='time_pbrs', speed_k=20.),
                                                         corner_bonus=E3),
                              bonus_schedule=BonusScheduleConfig(hold_until=None),
                              warm=dataclasses.replace(c.warm, offline_critic_grad_steps=12, offline_block=2,
                                                       pre_exit_warn_q=0., pre_exit_warn_blocks=5, speed_relabel=True))
    model = make_model(cfg, fill=600)
    model.replay_buffer.bonus_weight = 1.
    table = _table(((800., 1000.), (2000., 2300.)))
    s = np.full((model.replay_buffer.buffer_size, 1), np.nan)
    s[:600, 0] = np.arange(600) * 5.                                     # 0 .. 2995 m
    d = runs.make_run_dir(tmp_path, 'r', 0)
    runs.update_status(d, phase='offline_warmup')
    out = T._offline_critic(T.Run(cfg, d, lambda *a: None), model, s, table)
    assert len(out) == 6 and all(np.isfinite(r['mean_q_pre_entry']) and np.isfinite(r['mean_q_pre_exit']) for r in out)
    msgs = [w['message'] for w in runs.read_json(d / 'status.json')['warnings']]
    assert any('pre-entry mean Q_min moved' in m for m in msgs) and any('pre-exit mean Q_min moved' in m for m in msgs)


# --- req 5: preset, schedule, keeper, offset --------------------------------------------------------------
def test_preset_f1_r4():
    r2, r4 = PRESETS['f1_r2'](), PRESETS['f1_r4']()
    assert r4.init_from == F1_R2_INIT == str(SRC / 'checkpoints' / 'last.zip')
    assert r4.warm.init_buffer is None and not r4.warm.relabel and r4.warm.speed_relabel
    assert r4.warm.offline_critic_grad_steps == 50_000 and r4.warm.fallback_steps == 200_000
    assert r4.warm.speed_relabel_max_unresolved == .001 and r4.warm.probe_before_m == 150.
    assert r4.env.reward.speed_k == 20. and r4.env.reward.speed_sigma == .3 and r4.env.reward.mode == 'time_pbrs'
    cb = r4.env.corner_bonus
    assert cb.enabled and cb.mode == 'entry_exit' and (cb.bonus_in, cb.bonus_out, cb.sigma) == (10., 10., .3)
    assert r4.env.entry_exit_on and r4.env.speed_on
    assert r4.bonus_schedule.never_fade and r4.bonus_schedule.report_from is None
    assert r4.focus_update.seed_run == F1_BEST_W35_RUN and r4.focus_update.init_state is None
    assert r4.env.focus.enabled and r4.env.focus.fraction == .25 and r4.agent.low_noise_envs == (3,)
    assert r4.warm.actor_lr == 1e-4 and r4.guard.threshold == .65 and not r4.stop.plateau
    back = dataclasses.replace(r4, preset='f1_r2', env=r2.env, warm=r2.warm, bonus_schedule=r2.bonus_schedule)
    assert back == r2                                        # nothing else differs from f1_r2
    j = json.loads(config_json(r4))
    assert train_config_from_dict(j) == r4 and j['warm']['speed_relabel'] and j['eval']['full_throttle_pedal'] == .95
    from rlsetup.driving.agent import default_init_buffer
    assert default_init_buffer(r4.init_from) == str(SRC / 'replay_buffer.pkl')
    from rlsetup.driving.train import parse_args
    assert parse_args(['--preset', 'f1_r4', '--run-name', 'f1_r4']).preset == 'f1_r4'
    for n in (0, 1, 10 ** 6, 10 ** 9):                       # never fade: w = 1 (entry / exit terms always on)
        assert bonus_weight(r4.bonus_schedule, n) == 1.


def test_keeper_counts_aided_evals_of_f1_r4(tmp_path):
    spec = importlib.util.spec_from_file_location('keeper_check', REPO_ROOT / 'scripts' / 'keeper_check.py')
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    d = tmp_path / 'r4'
    d.mkdir()
    (d / 'config.json').write_text(config_json(PRESETS['f1_r4']()))
    runs.append_jsonl(d / 'evals.jsonl', {'step': 0, 'det_clean': True, 'lap2_s': 101.9, 'reportable': False, 'bonus_w': 1.})
    assert K.never_fade(d) and K.limit_s(d) == 95. and not K.reached(d)
    runs.append_jsonl(d / 'evals.jsonl', {'step': 50, 'det_clean': True, 'lap2_s': 94.9, 'reportable': False, 'bonus_w': 1.})
    assert K.reached(d) and not K.stopped(d)


@needs_f1
def test_f1_r4_corner_table_key_unchanged():
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from rlsetup.driving.train import corner_table
    cfg = PRESETS['f1_r4']()
    track = Track(cfg.env.track)
    table = corner_table(cfg, track, qss_profile.for_env(track, cfg.env))
    assert table.meta['key'][:12] == TABLE_KEY and (REPO_ROOT / 'data' / f'corner_table_{TABLE_KEY}.json').exists()
    want = [(517.0, 82.56), (1467.9, 80.15), (2061.9, 78.73), (2449.9, 61.32), (3006.9, 78.11), (4634.8, 83.60),
            (5041.8, 72.00)]                                 # R4_review [ARCH] 3
    for (name, s_in, v_in), (s, v) in zip(SP.entry_points(table), want):
        assert s_in == pytest.approx(s, abs=.1) and v_in == pytest.approx(v, abs=.01)


@needs_src
def test_f1_r4_lifetime_offset_is_12_35M():
    with zipfile.ZipFile(F1_R2_INIT) as z:
        assert json.loads(z.read('data'))['num_timesteps'] == 12_350_000
    st = json.loads((SRC / 'status.json').read_text())
    assert int(st.get('lifetime_steps_offset') or 0) == 0                      # source offset 0 -> 0 + 12.35M


# --- req 6: logging -------------------------------------------------------------------------------------
def _fake_report(table):
    from rlsetup.driving import evaluate as E
    s = np.r_[np.arange(0., 5000., 10.), np.arange(0., 5000., 10.)]          # lap 1 then lap 2, 10 m per decision
    T = len(s)
    lap = np.r_[np.ones(500, int), np.full(500, 2)]
    speed = np.full(T, 50.)
    speed[(lap == 2) & (s >= 850.) & (s <= 900.)] = 30.                       # apex v_min of U1
    actions = np.tile(np.array([0., 1., -1.], np.float32), (T, 1))             # full throttle, no brake
    actions[(lap == 2) & (s >= 760.) & (s < 860.), 1:] = (-1., 1.)              # braking from 760 m (entry 800)
    tr = {k: np.zeros(T) for k in E.TRACE_KEYS}
    tr.update(s=s, lap_index=lap, speed=speed, actions=actions, kappa=np.zeros((T, 4)), Fz=np.zeros((T, 4)),
              g_v=np.full(T, .5), shaping_v=np.full(T, .1), shaping=np.full(T, -.01))
    ev = [{'unit': 0, 'point': 'entry', 'kind': 'paid', 'rho': .9, 'g': .8, 'bonus': 8., 'lap_index': 2},
          {'unit': 0, 'point': 'exit', 'kind': 'paid', 'rho': .95, 'g': .9, 'bonus': 9., 'lap_index': 2},
          {'unit': 1, 'point': 'entry', 'kind': 'paid', 'rho': 1.1, 'g': .7, 'bonus': 7., 'lap_index': 2},
          {'unit': 1, 'point': 'entry', 'kind': 'taken_back', 'bonus': -7., 'lap_index': 2},
          {'unit': 0, 'point': 'entry', 'kind': 'paid', 'rho': .5, 'g': 0., 'bonus': 5., 'lap_index': 1}]
    return E.LapReport(True, [100., 100.], [], 'lap_complete', None, T, 0., tr, None, None, ev)


def test_lap_r4_metrics():
    from rlsetup.driving import evaluate as E
    table = _table(((800., 1000.), (2000., 2300.)))
    units, lap = E.lap_r4(_fake_report(table), table, EvalConfig(), pedal_deadzone=True)
    u1, u2 = units['U1'], units['U2']
    assert (u1['rho_in'], u1['g_in'], u1['kind_in'], u1['rho_out'], u1['kind_out']) == (.9, .8, 'paid', .95, 'paid')
    assert u1['brake_onset_m'] == pytest.approx(-40.) and u1['v_min_m_s'] == 30. and u2['v_min_m_s'] == 50.
    assert u2['taken_back'] == 7. and u2['kind_in'] == 'paid' and u2['brake_onset_m'] is None
    assert lap['entry_sum'] == 15. and lap['exit_sum'] == 9. and lap['taken_back'] == 7. and lap['terms_sum'] == 17.
    assert lap['mean_g'] == .5 and lap['shaping_v_sum'] == pytest.approx(50.) and lap['shaping_e_sum'] == pytest.approx(-5.)
    s1, s2 = lap['straights']['S1'], lap['straights']['S2']                    # 1000 -> 2000, 2300 -> 800
    assert s1['full_throttle_share'] == 1. and s1['brake_share'] == 0. and s1['top_speed'] == 50. and s1['mean_g'] == .5
    assert s2['full_throttle_share'] < 1. and s2['brake_share'] > 0.
    _, speed_only = E.lap_r4(_fake_report(table), table, EvalConfig(), pedal_deadzone=True, entry_exit=False)
    assert 'terms_sum' not in speed_only and speed_only['mean_g'] == .5


def test_evals_to_tb_r4_scalars():
    spec = importlib.util.spec_from_file_location('evals_to_tb', REPO_ROOT / 'scripts' / 'evals_to_tb.py')
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    rec = {'step': 0, 'det_clean': True, 'lap2_s': 101.,
           'lap2_r4': {'terms_sum': 17., 'mean_g': .5, 'shaping_v_sum': 50., 'shaping_e_sum': -5.,
                       'front_lock_fraction': .01, 'straights': {'S1': {'full_throttle_share': 1., 'top_speed': 80.}}},
           'lap2_entry_exit': {'U1': {'rho_in': .9, 'g_in': .8, 'brake_onset_m': 20., 'v_min_m_s': 30.}},
           'train_r4': {'ep_shaping_v_mean': -3., 'takebacks': 2}}
    sc = M.scalars_of(rec)
    assert sc['r4/lap2_terms_sum'] == 17. and sc['straight/full_throttle_share_S1'] == 1.
    assert sc['corner/brake_onset_m_U1'] == 20. and sc['corner/rho_in_U1'] == .9 and sc['train_r4/takebacks'] == 2
    assert not any(k.startswith(('r4/', 'straight/', 'train_r4/')) for k in M.scalars_of({'step': 0}))


def test_train_stats_r4_per_episode():
    cb = TrainStatsCallback(None, 10 ** 9)
    infos = [{'termination': 'off_track', 'episode_shaping_v': -20., 'episode_corner_bonus': 0.,
              'episode_corner_taken_back': 9., 'episode_corner_takebacks': 1},
             {'truncation': 'time_limit', 'episode_shaping_v': -2., 'episode_corner_bonus': 30.,
              'episode_corner_taken_back': 0., 'episode_corner_takebacks': 0}]
    for info in infos:
        cb._add_r4(cb.w['r4'], info)
        cb._add_r4(cb._eval_r4, info)
    r4 = cb.eval_r4()
    assert r4 == {'episodes': 2, 'ep_shaping_v_mean': -11., 'ep_corner_terms_mean': 15., 'ep_taken_back_mean': 4.5,
                  'takebacks': 1, 'episodes_with_takeback': 1}
    assert cb.eval_r4() is None and cb._summarize_r4(cb._empty_r4()) is None
