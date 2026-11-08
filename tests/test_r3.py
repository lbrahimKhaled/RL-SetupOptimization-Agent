"""R3 (docs/specs/R3_bigger_bonus.md + binding R3_review.md reqs 1-9): preset f1_r3, exact bonus relabel,
never-fade schedule, keeper on aided evaluations, lifetime offset chaining, focus init_state, pre-exit probe.

Tests that read runs/mac/D2/f1_r2 skip when that run folder is absent (runs/ is not in git)."""
import dataclasses
import importlib.util
import json
import zipfile
from types import SimpleNamespace
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup import runs
from rlsetup.provenance import REPO_ROOT
from rlsetup.driving import corners as C
from rlsetup.driving.buffers import (BonusNStepReplayBuffer, relabel_bonus_values, relabel_bonuses,
                                     check_bonus_relabel, pre_exit_cells, BONUS_SUM_TOL)
from rlsetup.driving.callbacks import (BonusScheduleCallback, DrivingEvalCallback, FocusedStartCallback, bonus_weight,
                                       load_bonus_state)
from rlsetup.driving.config import CornerBonusConfig, FocusConfig, RewardConfig
from rlsetup.driving.train_config import (PRESETS, BonusScheduleConfig, FocusUpdateConfig, WarmStartConfig,
                                          config_json, train_config_from_dict)
from sb3_helpers import TinyEnv, tiny_cfg, make_model

R2_RUN = REPO_ROOT / 'runs' / 'mac' / 'D2' / 'f1_r2' / 'seed0'
needs_r2_run = pytest.mark.skipif(not (R2_RUN / 'config.json').exists(), reason='needs runs/mac/D2/f1_r2/seed0')
SRC, DST = (3., .75), (30., .85)
NEVER = BonusScheduleConfig(hold_until=None)
ON = dict(corner_bonus=CornerBonusConfig(enabled=True), focus=FocusConfig(enabled=True))


def _tracker_bonus(rho, bonus, floor, v_qss=50.):
    """Bonus of one exit crossing at exit-speed ratio rho, as CornerTracker computes it."""
    t = C.CornerTable(5000., np.arange(5000) * 1., np.full(5000, 60.),
                      [{'name': 'U1', 'apexes_m': [900.], 'v_apex': [20.], 'start_m': 800., 'exit_m': 1000.,
                        'v_qss_exit': v_qss}], 90.)
    ev = C.CornerTracker(t, bonus, floor).tick(999., 1001., rho * v_qss, rho * v_qss)
    assert len(ev) == 1 and ev[0]['rho'] == pytest.approx(rho, abs=1e-15)
    return ev[0]['bonus']


# --- req 5: exact relabel map --------------------------------------------------------------------------
RHO_GRID = [0., .5, .74, .75, .75 + 1e-7, .8, .85 - 1e-7, .85, .85 + 1e-7, .86, .9, .91, .95, .999, 1., 1. + 1e-7,
            1.3] + list(np.linspace(.7, 1.05, 351))


def test_relabel_map_matches_tracker_on_rho_grid():
    """Stored float32 bonus at (3, 0.75) relabelled to (30, 0.85) = float32 of the tracker's bonus at
    (30, 0.85) for the same rho, within the float32 rounding bound (~16.7 ulp(3)/2 + ulp(30)/2 < 4e-6)."""
    old = np.array([_tracker_bonus(r, *SRC) for r in RHO_GRID], np.float32)        # what the buffer holds
    want = np.array([_tracker_bonus(r, *DST) for r in RHO_GRID], np.float32)
    got = relabel_bonus_values(old, SRC, DST)
    assert got.dtype == np.float32
    assert np.abs(got.astype(float) - want.astype(float)).max() <= 4e-6
    by = dict(zip(RHO_GRID[:17], got[:17]))
    # near the new floor: float32(b) at rho = 0.85 lies a few 1e-9 above it in rho -> b' ~ 8e-7 (rounding bound)
    assert by[.85 - 1e-7] == 0. and 0. <= by[.85] <= 4e-6 and 0. < by[.85 + 1e-7] < 1e-4
    assert by[.75] == by[.8] == 0. and by[1.] == by[1.3] == by[1. + 1e-7] == 30.          # b = 0 -> 0, b = 3 -> 30
    assert by[.91] == pytest.approx(12., abs=4e-6)
    assert relabel_bonus_values(np.float32([0., 3.]), SRC, DST).tolist() == [0., 30.]
    assert not relabel_bonus_values(old, SRC, (0., .85)).any()                            # B = 0
    assert np.abs(relabel_bonus_values(old, SRC, SRC) - old).max() <= 4e-7                 # identity map
    assert relabel_bonus_values(np.float32([3.]), (3., .75), (3., .75)).tolist() == [3.]


def test_relabel_refusals():
    with pytest.raises(ValueError, match='below the source floor'):
        check_bonus_relabel(SRC, (30., .7))
    with pytest.raises(ValueError, match='no exit-speed'):
        check_bonus_relabel((0., .75), DST)
    with pytest.raises(ValueError, match='floor'):
        check_bonus_relabel(SRC, (30., 1.))
    check_bonus_relabel(SRC, (30., .75))                                                 # equal floor is fine
    rb = _bonus_buffer([[0., 1.5], [3. * (1 + .5 * BONUS_SUM_TOL), 0.]])
    relabel_bonuses(rb, SRC, DST)                                                       # float32 slack accepted
    for bad in (6., -.1):                                                               # two exits summed / negative
        rb = _bonus_buffer([[0., 1.5], [bad, 0.]])
        before = rb.bonuses.copy()
        with pytest.raises(ValueError, match='outside'):
            relabel_bonuses(rb, SRC, DST)
        assert np.array_equal(rb.bonuses, before)                                       # nothing changed


def _bonus_buffer(rows, size=8, dones=None):
    """BonusNStepReplayBuffer with len(rows[0]) env columns and the given per-row bonuses."""
    from gymnasium import spaces
    rows = np.asarray(rows, np.float32)
    n_envs = rows.shape[1]
    rb = BonusNStepReplayBuffer(size * n_envs, spaces.Box(-1, 1, (2,), np.float32), spaces.Box(-1, 1, (1,), np.float32),
                                n_envs=n_envs, n_steps=3, gamma=.99)
    for t, b in enumerate(rows):
        done = np.zeros(n_envs, bool) if dones is None else np.asarray(dones[t], bool)
        rb.add(np.full((n_envs, 2), t, np.float32), np.zeros((n_envs, 2), np.float32), np.zeros((n_envs, 1), np.float32),
               np.zeros(n_envs, np.float32), done, [{'corner_bonus': float(x)} for x in b])
    return rb


def test_relabel_bonuses_stats_and_stored_rows_only():
    rb = _bonus_buffer([[0., .6], [1.44, 3.], [.3, 0.]])         # rho .75, .8 / .87, 1. / .775, .75
    rb.bonuses[5:] = 9.                                           # beyond pos: not stored, untouched
    st = relabel_bonuses(rb, SRC, DST)
    assert (st['n_transitions'], st['n_b_pos'], st['n_to_zero']) == (6, 4, 2)          # .6 and .3 -> 0
    assert st['mean_before'] == pytest.approx(5.34 / 6) and st['max_before'] == 3. and st['max_after'] == 30.
    assert np.allclose(rb.bonuses[:3], [[0., 0.], [4., 30.], [0., 0.]], atol=4e-6)
    assert st['mean_after'] == pytest.approx(34. / 6, abs=1e-5) and (rb.bonuses[5:] == 9.).all()
    assert st['from'] == {'bonus': 3., 'floor_ratio': .75} and st['to'] == {'bonus': 30., 'floor_ratio': .85}
    assert len(st['sha256_after']) == 64


def _bonus_source(tmp_path, bonuses=None):
    """Source run with corner_bonus (3, 0.75) on and a bonus replay buffer; returns (cfg, path)."""
    from test_r1 import _save_source
    c = tiny_cfg()
    src_cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, reward=RewardConfig(mode='time_pbrs'), **ON))
    src, path = _save_source(tmp_path, 'src', src_cfg, buffer=False)
    rb = src.replay_buffer
    assert type(rb) is BonusNStepReplayBuffer
    rng = np.random.default_rng(0)
    for t in range(600):
        rb.add(rng.uniform(-1, 1, (1, 159)).astype(np.float32), rng.uniform(-1, 1, (1, 159)).astype(np.float32),
               rng.uniform(-1, 1, (1, 3)).astype(np.float32), rng.normal(size=1).astype(np.float32),
               rng.uniform(size=1) < .02, [{'corner_bonus': float(bonuses(t)) if bonuses else 0.}])
    src.save_replay_buffer(path.parents[1] / 'replay_buffer.pkl')
    return src, path


def _r3_cfg(path, bonus=30., floor=.85):
    from test_r1 import _warm_cfg
    c = _warm_cfg(path, relabel=False)
    return dataclasses.replace(c, env=dataclasses.replace(c.env, corner_bonus=CornerBonusConfig(
        enabled=True, bonus=bonus, floor_ratio=floor), focus=FocusConfig(enabled=True)))


def test_preload_relabels_bonus_buffer_with_provenance(tmp_path):
    from rlsetup.driving.agent import build_model, preload_buffer, default_init_buffer, InitMismatch
    src, path = _bonus_source(tmp_path, bonuses=lambda t: [0., 1.44, 3., .6][t % 4] if t % 7 == 0 else 0.)
    old = src.replay_buffer.bonuses[:600, 0].copy()
    cfg = _r3_cfg(path)
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    info = preload_buffer(model, default_init_buffer(path), path, cfg)
    rb = model.replay_buffer
    assert not info['adopted'] and info['relabel'] is None
    st = info['bonus_relabel']
    assert st['n_b_pos'] == int((old > 0).sum()) and st['n_to_zero'] == int((old == np.float32(.6)).sum()) > 0
    assert np.array_equal(rb.bonuses[:600, 0], relabel_bonus_values(old, SRC, DST)) and rb.bonuses.max() == 30.
    assert st['mean_before'] == pytest.approx(float(old.mean()))
    # same (B, floor): untouched; lower floor: refused before loading; summed events: refused
    same = _r3_cfg(path, 3., .75)
    m2 = build_model(same, DummyVecEnv([TinyEnv]))
    assert preload_buffer(m2, default_init_buffer(path), path, same)['bonus_relabel'] is None
    assert np.array_equal(m2.replay_buffer.bonuses[:600, 0], old)
    low = _r3_cfg(path, 30., .7)
    junk = tmp_path / 'not_a_pickle.pkl'                     # refused before the (large) buffer is loaded
    junk.write_bytes(b'not a pickle')
    with pytest.raises(InitMismatch, match='below the source floor'):
        preload_buffer(build_model(low, DummyVecEnv([TinyEnv])), str(junk), path, low)


def test_preload_refuses_summed_bonuses(tmp_path):
    from rlsetup.driving.agent import build_model, preload_buffer, default_init_buffer, InitMismatch
    _, path = _bonus_source(tmp_path, bonuses=lambda t: 6. if t == 10 else 0.)
    cfg = _r3_cfg(path)
    with pytest.raises(InitMismatch, match='outside'):
        preload_buffer(build_model(cfg, DummyVecEnv([TinyEnv])), default_init_buffer(path), path, cfg)


# --- req 1, 2, 8: never-fade schedule --------------------------------------------------------------------
def test_never_fade_schedule_weight_trigger_report():
    assert all(bonus_weight(NEVER, n) == 1. for n in (0, 1_999_999, 2_000_000, 3_000_000, 4_000_000, 10 ** 9))
    assert NEVER.report_from is None and NEVER.never_fade and not BonusScheduleConfig().never_fade
    cb = BonusScheduleCallback(NEVER)
    assert cb.report_from is None and not cb.reportable(0) and not cb.reportable(10 ** 9)
    events = []
    cb = BonusScheduleCallback(NEVER, on_event=lambda k, **d: events.append(k))
    for n, lap in ((0, 101.), (50_000, 94.), (8_000_000, 90.)):                     # fast laps: no trigger
        rec = cb.on_eval({'step': n, 'det_clean': True, 'lap2_s': lap})
        assert rec['bonus_w'] == 1. and rec['reportable'] is False and rec['report_from'] is None
        assert 'aided_target_lap' not in rec and 'result_eval' not in rec
    assert events == [] and cb.trigger_step is None and cb.result() is None
    st = cb.state(8_000_000)
    assert st['report_from'] is None and st['bonus_w'] == 1. and st['schedule']['hold_until'] is None
    model = SimpleNamespace(num_timesteps=10 ** 9, replay_buffer=SimpleNamespace(bonus_weight=0.),
                            logger=SimpleNamespace(record=lambda *a: None))
    cb.model = model
    assert cb._on_step() and not cb.stopped and model.replay_buffer.bonus_weight == 1.
    assert model.bonus_report_from is None


def test_never_fade_weight_before_every_train_call():
    c = tiny_cfg()
    cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, corner_bonus=CornerBonusConfig(enabled=True)),
                              bonus_schedule=NEVER)
    model = make_model(cfg, fill=600)
    seen = []
    orig = model.train

    def train(gradient_steps, batch_size=64):
        seen.append(model.replay_buffer.bonus_weight)
        return orig(gradient_steps, batch_size)
    model.train = train
    model.learn(60, callback=[BonusScheduleCallback(NEVER)])
    assert len(seen) == 60 and set(seen) == {1.}


def test_never_fade_only_best_aided(tmp_path):
    """Every evaluation aided: best.zip is never written, best_aided.zip on each better clean lap,
    also for a lap <= 95 s (no trigger, no stop)."""
    model = make_model(fill=0)
    laps = {0: 101.9, 25: 99.0, 50: None, 75: 94.0, 100: 96.0}
    sched = BonusScheduleCallback(NEVER, tmp_path / 'bonus_state.json')
    cb = DrivingEvalCallback(lambda m, n: {'det_clean': laps[n] is not None, 'lap2_s': laps[n]}, 25,
                             tmp_path / 'evals.jsonl', tmp_path, schedule=sched)
    for c in (sched, cb):
        c.init_callback(model)
    model.num_timesteps = 0
    sched.on_training_start({}, {})
    cb.on_training_start({}, {})
    for n in range(1, 101):
        model.num_timesteps = n
        assert sched.on_step()
        cb.on_step()
    recs = runs.read_jsonl(tmp_path / 'evals.jsonl')
    assert [r['step'] for r in recs] == [0, 25, 50, 75, 100]
    assert {r['bonus_w'] for r in recs} == {1.} and not any(r['reportable'] for r in recs)
    assert [bool(r.get('new_best_aided')) for r in recs] == [True, True, False, True, False]
    assert not any(r.get('new_best') for r in recs) and not (tmp_path / 'best.zip').exists()
    assert (tmp_path / 'best_aided.zip').exists() and cb.best_aided_lap2 == 94.
    st = runs.read_json(tmp_path / 'bonus_state.json')
    assert st['trigger_step'] is None and st['report_from'] is None and 'result' not in st
    assert load_bonus_state(tmp_path) == (None, None)


def _never_cfg(**focus_update):
    c = tiny_cfg()
    return dataclasses.replace(c, env=dataclasses.replace(c.env, **ON), bonus_schedule=NEVER,
                               focus_update=FocusUpdateConfig(**focus_update))


def test_never_fade_resume(tmp_path, monkeypatch):
    """Resume of a never-fade run: bonus_state.json report_from null -> (None, None), nothing
    reportable so best_lap2 stays inf and best_aided = best clean lap; the run is not 'ended itself';
    the schedule rebuilt from that state keeps w = 1; focus counters fall back to init_state."""
    from test_callbacks import resume_fixture
    from rlsetup.driving import train as T
    init = tmp_path / 'init_focus.json'
    runs.write_json(init, {'step': 7, 'n': [2.] * 108, 'c': [1.] * 108, 'hot': [3], 'updates': 9, 'bin_m': 50.,
                           'n_bins': 108})
    cfg = _never_cfg(init_state=str(init))
    monkeypatch.setattr(T, 'corner_table', lambda *a: None)
    d, model, seen = resume_fixture(tmp_path, monkeypatch, cfg=cfg)
    sched = BonusScheduleCallback(NEVER, d / 'bonus_state.json')
    for r in ({'step': 0, 'det_clean': True, 'lap2_s': 101.}, {'step': 2_000, 'det_clean': True, 'lap2_s': 94.},
              {'step': 4_000, 'det_clean': False, 'lap2_s': None}):
        runs.append_jsonl(d / 'evals.jsonl', sched.on_eval(dict(r)))
    assert runs.read_json(d / 'bonus_state.json')['report_from'] is None
    assert T.resume_pipeline(d, ['total_timesteps=9000'], argv=['x']) == T.EXIT_OK
    k = seen['callbacks']
    assert k['bonus_state'] == (None, None) and k['best_lap2'] == float('inf') and k['best_aided_lap2'] == 94.
    assert seen['total'] == 4_000 and k['focus_seed'] is None and k['focus_state']['updates'] == 9
    assert any('re-seeded from ' + str(init) in w['message'] for w in runs.read_json(d / 'status.json')['warnings'])
    back = BonusScheduleCallback(NEVER, trigger_step=k['bonus_state'][0], report_from=k['bonus_state'][1],
                                 history=k['eval_history'])
    assert back.weight(10 ** 8) == 1. and back.report_from is None and back.result() is None
    spec = importlib.util.spec_from_file_location('evals_to_tb', REPO_ROOT / 'scripts' / 'evals_to_tb.py')
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    sc = M.scalars_of(runs.read_jsonl(d / 'evals.jsonl')[1])
    assert sc['bonus/w'] == 1. and sc['bonus/reportable'] == 0. and sc['lap/deterministic_lap2_s'] == 94.


def test_old_f1_r2_config_still_loads():
    """config.json written by f1_r2 (before R3: no warm.pre_exit_*, no focus_update.init_state) loads and
    keeps its fading schedule; pre-R2 presets' JSON gains no R3 keys."""
    want = PRESETS['f1_r2']()
    j = json.loads(config_json(want))
    for k in ('pre_exit_horizon', 'pre_exit_warn_q', 'pre_exit_warn_blocks'):
        del j['warm'][k]
    del j['focus_update']['init_state']
    old = train_config_from_dict(j)
    assert old == want and old.bonus_schedule.report_from == 8_000_000 and bonus_weight(old.bonus_schedule, 3_000_000) == .5
    for name in ('full', 'r1_warm', 'f1_best_w35'):
        w = json.loads(config_json(PRESETS[name]()))['warm']
        assert not {'pre_exit_horizon', 'pre_exit_warn_q', 'pre_exit_warn_blocks'} & set(w), name
    if (R2_RUN / 'config.json').exists():                       # the real file
        real = train_config_from_dict(json.loads((R2_RUN / 'config.json').read_text()))
        assert real.preset == 'f1_r2' and real.bonus_schedule == BonusScheduleConfig()
        assert real.focus_update.init_state is None and real.warm.pre_exit_horizon == WarmStartConfig().pre_exit_horizon


# --- req 3: keeper -----------------------------------------------------------------------------------------
def _keeper():
    spec = importlib.util.spec_from_file_location('keeper_check', REPO_ROOT / 'scripts' / 'keeper_check.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_keeper_reached_on_aided_evals_for_never_fade(tmp_path):
    K = _keeper()
    r3, r2 = tmp_path / 'r3', tmp_path / 'r2'
    for d, name in ((r3, 'f1_r3'), (r2, 'f1_r2')):
        d.mkdir()
        (d / 'config.json').write_text(config_json(PRESETS[name]()))
        for r in ({'step': 0, 'det_clean': True, 'lap2_s': 101., 'reportable': False, 'bonus_w': 1.},
                  {'step': 50, 'det_clean': True, 'lap2_s': 95.5, 'reportable': False, 'bonus_w': 1.}):
            runs.append_jsonl(d / 'evals.jsonl', r)
    assert K.never_fade(r3) and not K.never_fade(r2) and not K.never_fade(tmp_path)       # no config: pre-R2
    assert K.limit_s(r3) == 95. and not K.reached(r3) and not K.reached(r2)
    for d in (r3, r2):
        runs.append_jsonl(d / 'evals.jsonl', {'step': 100, 'det_clean': True, 'lap2_s': 95.0, 'reportable': False,
                                              'bonus_w': 1.})
    assert K.reached(r3) and not K.reached(r3, 94.9)
    assert not K.reached(r2)                                     # f1_r2 unchanged: aided laps never stop it
    runs.append_jsonl(r3 / 'evals.jsonl', {'step': 150, 'det_clean': False, 'lap2_s': 90., 'reportable': False})
    assert not K.stopped(r3)


# --- req 4: lifetime offset chaining ---------------------------------------------------------------------
def test_lifetime_offset_chaining(tmp_path):
    from test_r1 import _save_source
    from rlsetup.driving.agent import build_model, warm_start
    c = tiny_cfg()
    src_cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, reward=RewardConfig(mode='time_pbrs')))
    _, path = _save_source(tmp_path, 'src', src_cfg)                  # num_timesteps 2,600,000
    cfg = _r3_cfg(path)
    info = warm_start(build_model(cfg, DummyVecEnv([TinyEnv])), path, cfg)
    assert (info['src_lifetime_steps_offset'], info['lifetime_steps_offset']) == (0, 2_600_000)   # no status.json
    runs.update_status(path.parents[1], lifetime_steps_offset=12_350_000)
    info = warm_start(build_model(cfg, DummyVecEnv([TinyEnv])), path, cfg)
    assert info['src_num_timesteps'] == 2_600_000 and info['src_lifetime_steps_offset'] == 12_350_000
    assert info['lifetime_steps_offset'] == 14_950_000


@needs_r2_run
def test_f1_r3_lifetime_offset_is_19_75M():
    status = json.loads((R2_RUN / 'status.json').read_text())
    with zipfile.ZipFile(PRESETS['f1_r3']().init_from) as z:
        n = json.loads(z.read('data'))['num_timesteps']
    assert (status['lifetime_steps_offset'], n) == (12_350_000, 7_400_000)
    assert status['lifetime_steps_offset'] + n == 19_750_000


# --- req 7: focus init_state ---------------------------------------------------------------------------
def test_focus_init_state(tmp_path):
    from rlsetup.driving import train as T
    src = FocusedStartCallback(FocusUpdateConfig(), 5388.77, 50.)
    src.c[13], src.hot, src.updates = 4., [13, 31], 740
    path = tmp_path / 'focus_state.json'
    runs.write_json(path, src.state(7_400_000))
    cfg = _never_cfg(init_state=str(path), seed_run='runs/mac/D2/f1_best_w35/seed0')
    st, prov = T.focus_init_state_for(cfg)
    assert prov['path'] == str(path) and prov['step'] == 7_400_000 and prov['n_bins'] == 108 and prov['hot'] == [13, 31]
    assert len(prov['sha256']) == 64 and T.focus_seed_for(cfg, None) is None          # init_state wins over seed_run
    cb = FocusedStartCallback(cfg.focus_update, 5388.77, 50., state=st)
    assert np.array_equal(cb.n, src.n) and np.array_equal(cb.c, src.c) and cb.hot == [13, 31] and cb.updates == 740
    assert np.allclose(cb.weights(), src.weights())
    off = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, focus=FocusConfig(enabled=False)))
    assert T.focus_init_state_for(off) == (None, None)
    bad = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, focus=FocusConfig(enabled=True, bin_m=25.)))
    with pytest.raises(ValueError, match='bin_m'):
        T.focus_init_state_for(bad)
    r2 = T.r2_provenance(cfg, None, None, 1., prov)
    assert r2['focus_init_state'] == prov and r2['bonus_schedule']['report_from'] is None
    gone = dataclasses.replace(cfg, focus_update=FocusUpdateConfig(init_state=str(tmp_path / 'missing.json')),
                               runs_root=str(tmp_path / 'runs'), strict_provenance=False)
    assert T.run_pipeline(gone, 'x') == T.EXIT_INIT_MISMATCH and not (tmp_path / 'runs').exists()   # no run folder


@needs_r2_run
def test_f1_r3_focus_init_state_is_f1_r2s():
    from rlsetup.driving import train as T
    st, prov = T.focus_init_state_for(PRESETS['f1_r3']())
    assert prov['path'] == str(R2_RUN / 'focus_state.json') and prov['n_bins'] == 108 and st['bin_m'] == 50.
    FocusedStartCallback(PRESETS['f1_r3']().focus_update, 5388.773713834052, 50., state=st)    # 108 bins accepted


# --- req 6: pre-exit probe --------------------------------------------------------------------------------
def test_pre_exit_cells():
    b = np.zeros((40, 2), np.float32)
    b[35, 0], b[10, 1] = 2., 1.
    dones = np.zeros((40, 2), bool)
    dones[20, 0] = True                                     # episode end before the exit at 35 (column 0)
    rb = _bonus_buffer(b, size=64, dones=dones)
    r, c = pre_exit_cells(rb, 30)
    got = set(zip(r.tolist(), c.tolist()))
    assert got == {(t, 0) for t in range(21, 36)} | {(t, 1) for t in range(0, 11)}
    r, c = pre_exit_cells(rb, 5)
    assert set(zip(r.tolist(), c.tolist())) == {(t, 0) for t in range(31, 36)} | {(t, 1) for t in range(6, 11)}
    # wrapped buffer: time order starts at pos; a chain never crosses the write position
    w = np.zeros((12, 1), np.float32)
    w[9, 0] = 1.                                            # rows 8..11 -> slots 0..3 after the wrap (size 8)
    wrb = _bonus_buffer(w, size=8)
    assert wrb.full and wrb.pos == 4
    r, _ = pre_exit_cells(wrb, 3)
    assert sorted(r.tolist()) == [0, 1, 7]                  # slot 1 = row 9 (bonus), 0 = row 8, 7 = row 7
    assert not len(pre_exit_cells(_bonus_buffer(np.zeros((5, 1))), 30)[0])


def test_offline_critic_pre_exit_probe(tmp_path):
    from rlsetup.driving import train as T
    c = tiny_cfg()
    cfg = dataclasses.replace(c, env=dataclasses.replace(c.env, corner_bonus=CornerBonusConfig(enabled=True)),
                              bonus_schedule=NEVER,
                              warm=dataclasses.replace(c.warm, offline_critic_grad_steps=12, offline_block=2,
                                                       pre_exit_warn_q=0., pre_exit_warn_blocks=5))
    model = make_model(cfg, fill=600)
    model.replay_buffer.bonuses[::50, 0] = 30.
    model.replay_buffer.bonus_weight = 1.
    d = runs.make_run_dir(tmp_path, 'r', 0)
    runs.update_status(d, phase='offline_warmup')
    out = T._offline_critic(T.Run(cfg, d, lambda *a: None), model)
    assert len(out) == 6 and all(np.isfinite(r['mean_q_pre_exit']) for r in out)
    assert [r['grad_steps'] for r in runs.read_jsonl(d / 'warmup.jsonl')] == [2, 4, 6, 8, 10, 12]
    assert any('pre-exit mean Q_min moved' in w['message'] for w in runs.read_json(d / 'status.json')['warnings'])
    calm = dataclasses.replace(cfg, warm=dataclasses.replace(cfg.warm, pre_exit_warn_q=1e9))
    d2 = runs.make_run_dir(tmp_path, 'r2', 0)
    runs.update_status(d2, phase='offline_warmup')
    T._offline_critic(T.Run(calm, d2, lambda *a: None), model)
    assert not (runs.read_json(d2 / 'status.json') or {}).get('warnings')
    plain = make_model(tiny_cfg(), fill=600)                 # no bonus buffer: no probe
    d3 = runs.make_run_dir(tmp_path, 'r3', 0)
    plain_cfg = dataclasses.replace(tiny_cfg(), warm=calm.warm)
    assert 'mean_q_pre_exit' not in T._offline_critic(T.Run(plain_cfg, d3, lambda *a: None), plain)[0]


# --- preset ----------------------------------------------------------------------------------------------
def test_preset_f1_r3():
    r2, r3 = PRESETS['f1_r2'](), PRESETS['f1_r3']()
    assert r3.env.corner_bonus == dataclasses.replace(r2.env.corner_bonus, bonus=30., floor_ratio=.85)
    assert r3.init_from == str(REPO_ROOT / 'runs' / 'mac' / 'D2' / 'f1_r2' / 'seed0' / 'checkpoints' / 'last.zip')
    assert r3.warm.init_buffer is None and r3.warm.offline_critic_grad_steps == 50_000 and not r3.warm.relabel
    assert r3.bonus_schedule.never_fade and r3.bonus_schedule.report_from is None
    assert r3.bonus_schedule.trigger_lap2_s == 95.
    assert r3.focus_update.init_state == 'runs/mac/D2/f1_r2/seed0/focus_state.json' and r3.focus_update.seed_run is None
    back = dataclasses.replace(r3, preset='f1_r2', init_from=r2.init_from, env=r2.env, warm=r2.warm,
                               bonus_schedule=r2.bonus_schedule, focus_update=r2.focus_update)
    assert back == r2                                        # nothing else differs from f1_r2
    j = json.loads(config_json(r3))
    assert j['bonus_schedule']['hold_until'] is None and train_config_from_dict(j) == r3
    from rlsetup.driving.agent import default_init_buffer
    assert default_init_buffer(r3.init_from) == str(R2_RUN / 'replay_buffer.pkl')
    from rlsetup.driving.train import parse_args
    assert parse_args(['--preset', 'f1_r3', '--run-name', 'f1_r3']).preset == 'f1_r3'
