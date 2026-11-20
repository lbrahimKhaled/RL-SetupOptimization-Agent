"""R5 reward on the AC path (docs/specs/R5_apex_split.md §2): RewardTerms vs the OCD env formulas,
post_apex_w in the v2 reference, loud failures for missing fields and unknown reward keys.

Synthetic track / fake car from test_ac_transfer_v2; the OCD side uses DrivingEnv's own methods on a
stand-in object (as tests/test_r5.py). No game, network or simulator stepping.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from rlsetup.assetto.driving import GAMMA, Geometry, Profile, RewardTerms, load_reference, reward
from rlsetup.assetto.learn import read_chunk, replay_identity
from rlsetup.driving import corners
from rlsetup.driving.env import DrivingEnv, post_apex_weight

from test_ac_transfer_v2 import F1_REWARD, N, harness, run, write_reference
from test_r5 import fake

ROOT = Path(__file__).resolve().parents[1]
R5_REWARD = dict(F1_REWARD, apex_split=True, line_k_pre=.141, line_free_pre_m=.2, exit_k=20/150, exit_sigma=.3,
                 apex_blend_m=20., split_units=[1], split_at_m=[930.])
INERT = dict(c=.1, w=[.01, .001, .001], penalty_base=50., penalty_kappa=1.5, recovery_time_s=1.2)


def source(rw):
    return {'env': {'reward': rw}}


# --- RewardTerms.from_source -------------------------------------------------------------------------

@pytest.mark.parametrize('path', ['models/bcn_f1_r4_step6400k/seed0/config.json',
                                  'runs/ac/source_actor/config.json',
                                  'runs/ac/source_actor_bcn_f1_r4/config.json',
                                  'runs/ac/source_actor_bcn_f1_r5_7m/config.json',
                                  'models/bcn_f1_r5_step7000k/seed0/config.json'])
def test_existing_source_configs_load(path):
    if not (ROOT/path).exists():
        pytest.skip(f'{path} not in this checkout')
    rw = json.loads((ROOT/path).read_text())['env']['reward']
    terms = RewardTerms.from_source(source(rw))
    assert terms.apex_split == bool(rw.get('apex_split', False))
    if not terms.apex_split:  # R4 / R1: the chunk identity is the R4 dict, as before R5 support
        assert terms.identity() == {k: float(rw.get(k, getattr(RewardTerms, k))) for k in
                                    ('speed_k', 'speed_sigma', 'line_k', 'line_free_m', 'line_cap')}
    else:
        assert (terms.line_k_pre, terms.line_free_pre_m, terms.exit_k, terms.exit_sigma) == (
            rw['line_k_pre'], rw['line_free_pre_m'], rw['exit_k'], rw['exit_sigma'])


def test_r4_terms_ignore_w():
    r4 = RewardTerms.from_source(source(dict(F1_REWARD, **INERT)))
    assert not r4.apex_split and r4.exit_term(30., 20., 2., True, 1.) == 0.
    for w in (0., .5, 1.):
        assert r4.line_penalty(1.2, 2., True, w) == r4.line_penalty(1.2, 2., True) == min(.0705*.7, .5)*2.


def test_unknown_or_unsupported_reward_keys_raise():
    RewardTerms.from_source(source(dict(F1_REWARD, **INERT)))           # ref_tracking-only keys: inert
    RewardTerms.from_source(source(dict(F1_REWARD, apex_blend_m=20., split_units=[], split_at_m=[])))  # neutral
    with pytest.raises(ValueError, match='new_term'):
        RewardTerms.from_source(source(dict(F1_REWARD, new_term=1.)))
    with pytest.raises(ValueError, match='new_term'):                   # unknown: no neutral value known
        RewardTerms.from_source(source(dict(F1_REWARD, new_term=0.)))
    for key, value in (('split_units', [1]), ('apex_blend_m', 10.)):     # only meaningful with apex_split
        with pytest.raises(ValueError, match=key):
            RewardTerms.from_source(source(dict(F1_REWARD, **{key: value})))
    with pytest.raises(ValueError, match='exit_k'):
        RewardTerms.from_source(source(dict(F1_REWARD, exit_k=.1)))
    with pytest.raises(ValueError, match='apex_split'):
        RewardTerms.from_source(source(dict(F1_REWARD, apex_split=1)))
    with pytest.raises(ValueError, match='line_k'):
        RewardTerms.from_source(source(dict(R5_REWARD, line_k=0.)))
    with pytest.raises(ValueError):
        RewardTerms.from_source(source(dict(R5_REWARD, exit_sigma=0.)))


# --- R5 terms equal the OCD env formulas ---------------------------------------------------------------

U = [(100., [200.], 350.)]
OCD = dict(mode='time_pbrs', line_k=.0705, line_free_m=.5, line_cap=.5, apex_split=True, line_k_pre=.141,
           line_free_pre_m=.2, exit_k=20/150, exit_sigma=.3, apex_blend_m=20.)


@pytest.mark.parametrize('s,w', [(150., 0.), (190., 0.), (200., .5), (205., .75), (300., 1.)])
@pytest.mark.parametrize('abs_e', [.1, .35, 1.2, 50.])
def test_line_penalty_equals_ocd(s, w, abs_e):
    terms = RewardTerms.from_source(source(R5_REWARD))
    env = fake(U, s, **OCD)
    assert post_apex_weight(env.corners, s, 0, 20.) == w
    for ds in (2., 0., -1.):
        assert terms.line_penalty(abs_e, ds, True, w) == pytest.approx(DrivingEnv._line_penalty(env, abs_e, ds),
                                                                     abs=1e-15)
    assert terms.line_penalty(abs_e, 2., False, w) == 0. == DrivingEnv._line_penalty(fake(U, 400., **OCD), abs_e, 2.)


@pytest.mark.parametrize('s,w', [(150., 0.), (200., .5), (205., .75), (300., 1.)])
@pytest.mark.parametrize('speed', [30., 40., 42.5, 47., 50., 60.])   # rho = speed / 50 below and above 1
def test_exit_term_equals_ocd(s, w, speed):
    terms = RewardTerms.from_source(source(R5_REWARD))
    env = fake(U, s, speed, **OCD)
    for ds in (3., 0., -1.):
        assert terms.exit_term(speed, 50., ds, True, w) == pytest.approx(DrivingEnv._exit_term(env, ds), abs=1e-15)
    assert terms.exit_term(speed, 50., 3., False, w) == 0.
    if w == 1. and speed >= 50.:
        assert terms.exit_term(speed, 50., 3., True, w) == pytest.approx(20/150*3.)   # one-sided, h = 1


# --- reference generation --------------------------------------------------------------------------

def _script():
    sys.path.insert(0, str(ROOT/'scripts'))
    import ac_ocd_reference
    return ac_ocd_reference


def test_post_apex_profile_on_the_barcelona_table():
    table = corners.load(ROOT/'data'/'corner_table_6313f2151e24.json')   # same U1 window/apex as R5's table
    w, how = _script().post_apex_profile(table, R5_REWARD, np.array([700., 875., 920., 930., 940., 980., 995.]))
    np.testing.assert_allclose(w, [0., 0., 0., .5, 1., 1., 0.], atol=1e-9)   # 700, 995: outside U1
    assert how['split_units'] == [1] and how['split_at_m'] == [930.] and how['apex_blend_m'] == 20.
    w4, _ = _script().post_apex_profile(table, dict(R5_REWARD, split_units=[], split_at_m=[]), np.array([840.]))
    assert w4[0] == pytest.approx(.5, abs=.01)                                 # no split: U1's last apex 839.9 m
    with pytest.raises(SystemExit, match='U1'):
        _script().post_apex_profile(table, dict(R5_REWARD, split_at_m=[1100.]), np.array([840.]))


def test_generated_r5_reference_file():
    path = ROOT/'runs/ac/ocd_line_reference_bcn_f1_r5_7m.json'
    if not path.exists():
        pytest.skip('R5 reference not generated in this checkout')
    p = json.loads(path.read_text())['profile']
    s, w = np.asarray(p['s_path_m']) % p['path_length_m'], np.asarray(p['post_apex_w'])
    for x, want in ((875., 0.), (920., 0.), (940., 1.)):
        assert w[np.argmin(abs(s-x))] == pytest.approx(want, abs=.05)
    i = np.argmin(abs(s-930.))
    assert w[i] == pytest.approx(.5+(s[i]-930.)/20., abs=1e-9)                 # rows ~1.5 m apart
    assert np.all(w[np.asarray(p['corner']) == 0] == 0)


# --- collector -------------------------------------------------------------------------------------

def _add_w(reference, w):
    saved = json.loads(reference.read_text())
    saved['profile']['post_apex_w'] = list(map(float, w))
    reference.write_text(json.dumps(saved))


def test_apex_split_source_without_post_apex_w_raises(monkeypatch, tmp_path):
    bundle, calibration, reference, *_ = harness(monkeypatch, tmp_path, actions=[[0., -.4, -.8]],
                                                 reward_config=R5_REWARD)
    with pytest.raises(ValueError, match='post_apex_w'):
        run(bundle, calibration, reference, tmp_path/'x.npz')


def test_post_apex_w_validation(tmp_path):
    corner = (np.arange(N) % 100 < 70).astype(float)
    info = write_reference(tmp_path/'ok.json', corner=corner)
    for bad in (np.where(corner == 1, 1.5, 0.), np.ones(N), np.zeros(N-1)):    # > 1; nonzero off corner; length
        _add_w(tmp_path/'ok.json', bad)
        with pytest.raises(ValueError, match='post_apex_w'):
            load_reference(info, 'ocd_path', tmp_path/'ok.json')


def test_r5_drive_rewards_match_the_formulas(monkeypatch, tmp_path):
    # Accelerate past v_qss = 20 m/s (rho below and above 1), then brake to a stall.
    bundle, calibration, reference, info, car, _ = harness(
        monkeypatch, tmp_path, actions=[[0., .5, -.8]]*60+[[0., -.8, 1.]]*200, reward_config=R5_REWARD,
        reference_kwargs=dict(corner=(np.arange(N) % 100 < 70).astype(float)))
    i = np.arange(N)
    w_rows = np.where(i % 100 < 70, np.clip((i % 100-40)/20., 0., 1.), 0.)
    _add_w(reference, w_rows)
    run(bundle, calibration, reference, tmp_path/'chunk.npz')
    with np.load(tmp_path/'chunk.npz', allow_pickle=False) as chunk:
        meta = json.loads(chunk['metadata_json'].item())
        rewards = chunk['rewards'].astype(float)
    assert meta['source_terms'] == dict(speed_k=20., speed_sigma=.3, line_k=.0705, line_free_m=.5, line_cap=.5,
                                        apex_split='reference_post_apex_w', line_k_pre=.141, line_free_pre_m=.2,
                                        exit_k=20/150, exit_sigma=.3, progress='ocd_track_path',
                                        start='ocd_settle', stall_speed_mps=3., stall_time_s=2.)
    # the learner accepts the chunk and keeps R5 replay apart from R4 replay
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    _, read_meta, _ = read_chunk(tmp_path/'chunk.npz', sha(bundle/'manifest.json'), 'a'*64, GAMMA,
                                 sha(bundle/'config.json'))
    assert replay_identity(read_meta)['source_terms']['exit_k'] == 20/150
    # independent recomputation of every decision reward (formulas of R5 spec §2, written out here)
    track, _ = load_reference(info, 'ocd_path', reference)
    g = Geometry(track, reference='ocd_path')
    p = Profile(track['ocd_profile'], g)
    row = lambda st: min(int(np.searchsorted(g.stations, st % g.length, side='right')-1), N-1)
    packets = car.packets[-(len(rewards)+1):]
    speed = lambda q: math.hypot(q['velocity_z'], q['velocity_x'])
    phi_v = lambda v: 20.*max(0., 1.-((v-20.)/(.3*20.))**2)
    s0, e0, _ = g.project(np.array([packets[0]['world_position_x'], packets[0]['world_position_y']]))
    phi, expected, seen_w, seen_rho = phi_v(speed(packets[0])), [], set(), []
    for q in packets[1:]:
        s1, e1, _ = g.project(np.array([q['world_position_x'], q['world_position_y']]))
        ds, v, k = p.path_progress(s0, s1), speed(q), row(s1)
        w, inside = w_rows[k], i[k] % 100 < 70
        value = reward(ds, .04, e0, e1)+GAMMA*phi_v(v)-phi
        if inside and ds > 0:
            pre = min(.141*max(0., abs(e1)-.2), .5)
            post = min(.0705*max(0., abs(e1)-.5), .5)
            h = max(0., 1.-((1.-min(v/20., 1.))/.3)**2)
            value += -((1.-w)*pre+w*post)*ds+20/150*w*h*ds
            seen_w.add(round(float(w), 2))
            seen_rho.append(v/20.)
        expected.append(value)
        s0, e0, phi = s1, e1, phi_v(v)
    np.testing.assert_allclose(rewards, expected, rtol=1e-5, atol=1e-4)
    assert {0., 1.} <= seen_w and len(seen_w) > 5                     # pre, blend and post rows driven
    assert min(seen_rho) < .7 and max(seen_rho) > 1.05                # exit term below and above v_qss
