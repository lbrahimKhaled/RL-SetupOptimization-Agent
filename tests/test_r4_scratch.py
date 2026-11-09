"""R4 reward trained from scratch (preset f1_r4_scratch): the preset differs from f1_best_w35 only in the
reward, no warm start, never-fade schedule; demo rewards hold the speed potential, the demo dataset and
the prefilled bonus buffer hold the corner entry / exit terms, n-step targets use them, w(0) is in force
during the offline warm-up, actor lr 1e-4 from the unfreeze (warm.actor_lr in the BC path). Simulator tests use the synthetic corner table of speed_truth (default av21)."""
from types import SimpleNamespace as NS
import dataclasses
import importlib.util
import json
import time
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup import runs
from rlsetup.driving import demos as D
from rlsetup.driving import train as T
from rlsetup.driving import evaluate as E
from rlsetup.driving.agent import StagedSAC, build_model
from rlsetup.driving.buffers import BonusNStepReplayBuffer
from rlsetup.driving.callbacks import BonusScheduleCallback, PhaseCallback, bonus_weight
from rlsetup.driving.config import EnvConfig, RewardConfig, CornerBonusConfig
from rlsetup.driving.train_config import (PRESETS, BonusScheduleConfig, DemoDataConfig, config_json,
                                          train_config_from_dict, apply_overrides)
from rlsetup.provenance import REPO_ROOT
from sb3_helpers import TinyEnv, OBS, tiny_cfg, make_model
import speed_truth as ST

CONTROL_RUN = REPO_ROOT / 'runs' / 'mac' / 'D2' / 'f1_best_w35' / 'seed0' / 'config.json'
GAMMA = .992


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f'{prefix}{k}.') if isinstance(v, dict) else {f'{prefix}{k}': v})
    return out


# --- preset --------------------------------------------------------------------------------------------
def test_preset_differs_from_control_only_in_the_reward():
    base, c = PRESETS['f1_best_w35'](), PRESETS['f1_r4_scratch']()
    assert c.env.reward == dataclasses.replace(base.env.reward, speed_k=20., speed_sigma=.3)
    assert c.env.corner_bonus == dataclasses.replace(base.env.corner_bonus, enabled=True, mode='entry_exit',
                                                     bonus_in=10., bonus_out=10., sigma=.3)
    assert c.env.corner_bonus == PRESETS['f1_r4']().env.corner_bonus and c.env.reward == PRESETS['f1_r4']().env.reward
    env = dataclasses.replace(c.env, reward=base.env.reward, corner_bonus=base.env.corner_bonus)
    assert dataclasses.replace(c, preset=base.preset, env=env, bonus_schedule=base.bonus_schedule, warm=base.warm) == base
    assert c.warm == dataclasses.replace(base.warm, actor_lr=1e-4) and base.warm.actor_lr is None
    assert c.bonus_schedule == dataclasses.replace(base.bonus_schedule, hold_until=None)


def test_preset_is_from_scratch_with_every_aid_off():
    c = PRESETS['f1_r4_scratch']()
    assert c.agent.init_mode == 'bc' and c.init_from is None and c.warm.init_buffer is None
    assert not c.warm.speed_relabel and not c.env.focus.enabled and c.agent.low_noise_envs == ()
    assert c.focus_update.seed_run is None and c.focus_update.init_state is None
    assert c.guard.threshold == .5 and c.eval.interval == 50_000 and c.seed == 0 and not c.stop.plateau
    assert c.demos == PRESETS['f1_best_w35']().demos and c.demos.fraction_range == (.5, .8)
    assert c.env.r4 and c.env.speed_on and c.env.entry_exit_on and c.env.track == 'Shanghai'
    assert train_config_from_dict(json.loads(config_json(c))) == c
    assert T.parse_args(['--preset', 'f1_r4_scratch', '--run-name', 'f1_r4_scratch']).preset == 'f1_r4_scratch'


def test_schedule_never_fades_no_trigger_nothing_reportable(tmp_path):
    bs = PRESETS['f1_r4_scratch']().bonus_schedule
    assert bs.never_fade and bs.report_from is None
    for n in (0, 1, 2_000_000, 4_000_000, 10 ** 9):
        assert bonus_weight(bs, n) == 1.
    sched = BonusScheduleCallback(bs, tmp_path / 'bonus_state.json')
    rec = sched.on_eval({'step': 50_000, 'det_clean': True, 'lap2_s': 90.})    # would trigger a fading schedule
    assert sched.trigger_step is None and rec['bonus_w'] == 1. and rec['reportable'] is False
    assert rec['report_from'] is None and not rec.get('aided_target_lap') and sched.weight(10 ** 9) == 1.


def test_keeper_treats_it_as_never_fade(tmp_path):
    spec = importlib.util.spec_from_file_location('keeper_check', REPO_ROOT / 'scripts' / 'keeper_check.py')
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    (tmp_path / 'config.json').write_text(config_json(PRESETS['f1_r4_scratch']()))
    runs.append_jsonl(tmp_path / 'evals.jsonl', {'step': 0, 'det_clean': True, 'lap2_s': 101.9, 'reportable': False})
    assert K.never_fade(tmp_path) and K.limit_s(tmp_path) == 95. and not K.reached(tmp_path)
    runs.append_jsonl(tmp_path / 'evals.jsonl', {'step': 50, 'det_clean': True, 'lap2_s': 94.9, 'reportable': False})
    assert K.reached(tmp_path) and not K.stopped(tmp_path)


def test_new_fields_against_the_real_control_config():
    """Fields added since f1_best_w35 ran take values equivalent to its behaviour (all aids off)."""
    if not CONTROL_RUN.exists():
        pytest.skip(f'needs {CONTROL_RUN}')
    real = flat(json.loads(CONTROL_RUN.read_text()))
    over = (REPO_ROOT / 'platforms' / 'mac' / 'train.overrides').read_text().splitlines()
    over = [x.split('#')[0].strip() for x in over if x.split('#')[0].strip()]
    mine = flat(json.loads(config_json(apply_overrides(PRESETS['f1_r4_scratch'](), over))))
    differ = {k for k in set(real) & set(mine) if real[k] != mine[k]}
    assert differ == {'preset', 'runs_root', 'warm.actor_lr'}  # shared keys; runs_root = --set of train.sh
    assert real['warm.actor_lr'] is None and mine['warm.actor_lr'] == 1e-4
    assert not set(real) - set(mine)
    added = set(mine) - set(real)
    reward = {k for k in added if k.startswith('env.corner_bonus.')} | {'env.reward.speed_k', 'env.reward.speed_sigma'}
    assert {k for k in added if k.startswith(('env.corner_bonus', 'env.reward'))} == reward
    other = {k for k in added - reward if not k.startswith(('bonus_schedule', 'focus_update'))}
    assert other == {'agent.low_noise_envs', 'agent.low_noise_std_scale', 'eval.brake_pedal', 'eval.full_throttle_pedal',
                     'warm.fallback_steps', 'warm.pre_exit_horizon', 'warm.pre_exit_warn_blocks', 'warm.pre_exit_warn_q',
                     'warm.probe_before_m', 'warm.speed_relabel', 'warm.speed_relabel_max_unresolved'}
    assert mine['agent.low_noise_envs'] == [] and mine['warm.speed_relabel'] is False
    assert mine['focus_update.seed_run'] is None and mine['focus_update.init_state'] is None


# --- item 1: dataset key and demo rewards ------------------------------------------------------------------
def r4_env_cfg(**kw):
    return EnvConfig(max_episode_s=20., settle_s=0., reward=RewardConfig(mode='time_pbrs', speed_k=20.),
                     corner_bonus=CornerBonusConfig(enabled=True, mode='entry_exit'), **kw)


def test_dataset_key_changes_with_the_reward_and_reward_sources(qss, monkeypatch):
    prov = {'ocd_lap_code_version': 'a'}
    demo = DemoDataConfig(n_transitions=100, workers=1)
    plain = EnvConfig(reward=RewardConfig(mode='time_pbrs'))
    k0, d0 = D.dataset_key(plain, demo, qss, prov)
    assert 'rlsetup_reward_sources_sha256' not in d0              # non-bonus keys hash the same inputs as before
    on = r4_env_cfg()
    k1, d1 = D.dataset_key(on, demo, qss, prov)
    assert k1 != k0 and set(d1['rlsetup_reward_sources_sha256']) == set(D.REWARD_SOURCES)
    assert d1['env']['reward']['speed_k'] == 20. and d1['env']['corner_bonus']['mode'] == 'entry_exit'
    for other in (dataclasses.replace(on, reward=dataclasses.replace(on.reward, speed_k=10.)),
                  dataclasses.replace(on, corner_bonus=dataclasses.replace(on.corner_bonus, bonus_in=5.))):
        assert D.dataset_key(other, demo, qss, prov)[0] != k1
    sha = D.sha256_file
    monkeypatch.setattr(D, 'sha256_file', lambda p: '0' * 64 if str(p).endswith('speed_profile.py') else sha(p))
    assert D.dataset_key(on, demo, qss, prov)[0] != k1 and D.dataset_key(plain, demo, qss, prov)[0] == k0


@pytest.fixture(scope='module')
def rollouts(make_env, track, qss):
    """Demo episodes (v_qss of the table = 0.45 x p0 QSS, demos at f = 0.45: g near 1) on the R4 env and on
    the plain time_pbrs env, with the R4 env's per-decision info."""
    table = ST.table_for(track, qss)
    demo = DemoDataConfig(n_transitions=1, workers=1, fraction_range=(.45, .45), noise_std=(.02, .02, .02))
    cfg = r4_env_cfg()
    on, off = make_env(cfg, corners=table), make_env(EnvConfig(max_episode_s=20., settle_s=0.,
                                                               reward=RewardConfig(mode='time_pbrs')), corners=table)
    infos = []
    step = on.step

    def recording(a):
        out = step(a)
        infos.append(out[-1])
        return out
    on.step = recording
    eps = []
    for ep in range(6):
        infos.clear()
        a, b = D.run_episode(on, demo, ep), D.run_episode(off, demo, ep)
        eps.append((a, b, list(infos)))
    return eps


def test_demo_rewards_hold_the_speed_potential(rollouts):
    nonzero = 0
    for a, b, infos in rollouts:
        np.testing.assert_array_equal(a['actions'], b['actions'])
        shaping = np.array([i['shaping_v'] for i in infos])
        assert len(shaping) == len(a['rewards'])
        np.testing.assert_allclose(a['rewards'] - b['rewards'], shaping, atol=1e-9)
        nonzero += int(np.count_nonzero(shaping))
    assert nonzero > 100


def test_demo_dataset_stores_the_corner_terms(rollouts):
    terms = []
    for a, b, infos in rollouts:
        assert 'corner_bonus' not in b                            # plain env: no array, dataset unchanged
        np.testing.assert_array_equal(a['corner_bonus'], [i['corner_bonus'] for i in infos])
        assert a['corner_bonus'].sum() == pytest.approx(infos[-1]['episode_corner_bonus'], abs=1e-9)
        terms.append(a['corner_bonus'])
    all_terms = np.concatenate(terms)
    assert np.count_nonzero(all_terms) >= 3 and np.max(all_terms) > 1.   # entries / exits really paid
    assembled = D._assemble([e[0] for e in rollouts])
    np.testing.assert_array_equal(assembled['corner_bonus'], all_terms)
    assert 'corner_bonus' not in D._assemble([e[1] for e in rollouts])


# --- item 2: prefill and n-step targets -------------------------------------------------------------------
def synthetic(lengths, terminal, seed=0):
    rng = np.random.default_rng(seed)
    obs, act, rew, bon, term, trunc = [], [], [], [], [], []
    for e, (n, is_term) in enumerate(zip(lengths, terminal)):
        obs += [[e, t, 0., 0., 0.] for t in range(n + 1)]
        act += [[0., 0., 0.]] * n
        rew += rng.normal(1., .5, n).tolist()
        b = np.where(rng.uniform(size=n) < .2, rng.uniform(-7., 10., n), 0.)     # signed, as a take-back is
        bon += b.tolist()
        term += [False] * (n - 1) + [is_term]
        trunc += [False] * (n - 1) + [not is_term]
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    return D.with_index({'obs': np.array(obs, np.float32), 'actions': np.array(act, np.float32),
                         'rewards': np.array(rew), 'corner_bonus': np.array(bon), 'terminated': np.array(term),
                         'truncated': np.array(trunc), 'ep_start_tr': starts,
                         'ep_start_obs': starts + np.arange(len(lengths))})


def buffer(cls, n_envs, size=2000):
    return cls(size * n_envs, gym.spaces.Box(-1e3, 1e3, (5,), np.float32), gym.spaces.Box(-1, 1, (3,), np.float32),
               device='cpu', n_envs=n_envs, n_steps=3, gamma=GAMMA)


def test_prefill_writes_the_terms_to_the_right_rows():
    d = synthetic([40, 7, 25, 60, 13, 33, 52, 9, 18, 30], [True, False, True, True, False, False, True, False, True, True])
    rb = buffer(BonusNStepReplayBuffer, 3)
    info = D.prefill_replay_buffer(rb, d)
    T_ = info['rows']
    assert T_ == 29 and rb.labelled[:T_].all() and rb.labelled_fraction() == 1.
    cols = [np.concatenate([np.arange(d['ep_start_tr'][e], d['ep_start_tr'][e] + d['ep_len'][e]) for e in range(c, 10, 3)])
            for c in range(3)]
    for c in range(3):
        np.testing.assert_array_equal(rb.bonuses[:T_, c], d['corner_bonus'][cols[c][:T_]].astype(np.float32))
        np.testing.assert_array_equal(rb.rewards[:T_, c], d['rewards'][cols[c][:T_]].astype(np.float32))


def test_prefill_refuses_a_bonus_buffer_without_terms_and_ignores_them_otherwise():
    d = synthetic([40, 30], [True, False])
    plain = buffer(NStepReplayBuffer, 2)
    D.prefill_replay_buffer(plain, d)
    assert plain.pos == 30
    no = {k: v for k, v in d.items() if k != 'corner_bonus'}
    rb = buffer(BonusNStepReplayBuffer, 2)
    with pytest.raises(ValueError, match='corner_bonus'):
        D.prefill_replay_buffer(rb, no)
    assert rb.pos == 0 and not rb.labelled.any()


def test_n_step_returns_on_prefilled_demos_include_the_terms():
    d = synthetic([70, 45, 90], [True, False, True])
    rb = buffer(BonusNStepReplayBuffer, 1)
    n = D.prefill_replay_buffer(rb, d)['rows']
    assert n == 205
    batch = np.arange(0, n - 4)                                   # away from the temporary pos - 1 cut
    def by_hand(w):
        out = np.zeros(len(batch))
        for i, b in enumerate(batch):
            g, ret = 1., 0.
            for j in range(3):
                ret += g * (float(rb.rewards[b + j, 0]) + w * float(rb.bonuses[b + j, 0]))
                if rb.dones[b + j, 0] or rb.timeouts[b + j, 0]:
                    break
                g *= GAMMA
            out[i] = ret
        return out
    rb.bonus_weight = 1.
    np.testing.assert_allclose(rb._get_samples(batch).rewards.numpy().ravel(), by_hand(1.), atol=2e-5)
    assert np.abs(by_hand(1.) - by_hand(0.)).max() > 1.           # the terms do change the targets
    rb.bonus_weight = 0.
    np.testing.assert_allclose(rb._get_samples(batch).rewards.numpy().ravel(), by_hand(0.), atol=2e-5)


# --- item 3: the BC path with the schedule -----------------------------------------------------------------
def bonus_model():
    c = tiny_cfg()
    env = dataclasses.replace(c.env, reward=RewardConfig(mode='time_pbrs', speed_k=20.),
                              corner_bonus=CornerBonusConfig(enabled=True, mode='entry_exit'))
    c = dataclasses.replace(c, env=env, bonus_schedule=BonusScheduleConfig(hold_until=None),
                            warmup=dataclasses.replace(c.warmup, offline_blocks=1, offline_steps_per_block=2,
                                                       q_mc_max_episodes=2))
    from stable_baselines3.common.logger import configure
    model = build_model(c, DummyVecEnv([TinyEnv]))
    model.set_logger(configure(None, []))
    return c, model


def test_w0_is_set_before_the_offline_phase():
    c, model = bonus_model()
    assert isinstance(model.replay_buffer, BonusNStepReplayBuffer) and model.replay_buffer.bonus_weight == 0.
    T._set_bonus_w0(c, model)
    assert model.replay_buffer.bonus_weight == 1. and model.bonus_weight == 1.
    plain_cfg = tiny_cfg()
    plain = build_model(plain_cfg, DummyVecEnv([TinyEnv]))
    T._set_bonus_w0(plain_cfg, plain)
    assert not hasattr(plain.replay_buffer, 'bonus_weight') and plain.bonus_weight == 0.


class _Stop(Exception):
    pass


def pipeline_until_offline(tmp_path, monkeypatch, name):
    """Real run_pipeline of preset `name` (real build_model, prefill and w(0) step) with the simulator, demo
    collection, BC and provenance stubbed. A spy replaces _offline_warmup, records what it finds and stops."""
    import ocd_lap.track as track_mod
    from rlsetup import provenance as P
    from rlsetup.driving import bc, qss_profile, vec
    c = PRESETS[name]()
    cfg = dataclasses.replace(c, n_envs=1, strict_provenance=False, runs_root=str(tmp_path / 'runs'),
                              agent=dataclasses.replace(c.agent, buffer_size=5_000))
    n, rng = 300, np.random.default_rng(0)
    data = {'obs': rng.uniform(-1, 1, (n + 2, OBS)).astype(np.float32), 'actions': rng.uniform(-.9, .9, (n, 3)).astype(np.float32),
            'rewards': rng.normal(1., .1, n), 'terminated': np.arange(n) == 149, 'truncated': np.arange(n) == n - 1,
            'ep_start_tr': np.array([0, 150]), 'ep_start_obs': np.array([0, 151]), 'ep_cause': np.array(['off_track', 'time_limit'])}
    if cfg.env.corner_bonus.enabled:
        data['corner_bonus'] = np.where(np.arange(n) % 37 == 5, 8., 0.)
    lap = NS(clean=True, termination=None, lap1_s=100., lap2_s=99., decisions=10)
    meta = {'npz_sha256': 'x', 'key': 'k', 'summary': {'transitions': n}, 'cached': True}
    seen = {}

    def spy(run, model, demos, val):
        rb = model.replay_buffer
        seen.update(w=getattr(rb, 'bonus_weight', None), mirror=model.bonus_weight, frozen=model.freeze_actor, actor_lr=model.actor_lr,
                    nonzero=int(np.count_nonzero(rb.bonuses[:rb.pos])) if hasattr(rb, 'bonuses') else None)
        raise _Stop
    for owner, attr, value in ((track_mod, 'Track', lambda name: NS(name=name)),
                               (P, 'training_provenance', lambda *a, **k: {}),
                               (qss_profile, 'for_env', lambda *a: NS(meta={'key': {}}, lap_time_s=90.)),
                               (vec, 'track_sha256', lambda *a: '0'),
                               (vec, 'build_vec_env', lambda *a, **k: DummyVecEnv([TinyEnv])),
                               (T, 'eval_envs', lambda *a: {'eval_det': None, 'eval_sto': None, 'probe': NS(ref=NS(digest='d'))}),
                               (T, 'reference_metadata', lambda *a: None),
                               (T, 'corner_table', lambda *a: None),
                               (D, 'ensure_demos', lambda *a, **k: ('p', meta, cfg.demos)),
                               (D, 'load_demos', lambda p: D.with_index(data)),
                               (bc, 'train_bc', lambda *a, **k: {'val_episodes': [0]}),
                               (bc, 'plot_throttle_vs_slip', lambda *a, **k: None),
                               (E, 'flying_lap', lambda *a, **k: lap),
                               (E, 'lap_metrics', lambda *a, **k: {}),
                               (E, 'make_probe_starts', lambda *a, **k: []),
                               (E, 'probe', lambda *a, **k: NS(mean_progress_m=500., progress_m=[500.])),
                               (T, '_offline_warmup', spy)):
        monkeypatch.setattr(owner, attr, value)
    with pytest.raises(_Stop):
        T.run_pipeline(cfg, 'r')
    return seen


def test_w0_is_in_force_at_the_offline_warmup_entry(tmp_path, monkeypatch):
    seen = pipeline_until_offline(tmp_path, monkeypatch, 'f1_r4_scratch')
    assert seen['w'] == 1. and seen['mirror'] == 1. and seen['frozen'] and seen['nonzero'] > 0
    assert seen['actor_lr'] is None                               # the frozen phases keep their own settings


def test_no_bonus_weight_without_a_corner_bonus(tmp_path, monkeypatch):
    seen = pipeline_until_offline(tmp_path, monkeypatch, 'f1_best_w35')
    assert seen['w'] is None and seen['mirror'] == 0. and seen['frozen'] and seen['actor_lr'] is None


def test_offline_warmup_mc_uses_the_terms(tmp_path, monkeypatch):
    c, model = bonus_model()
    n = 800
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, (n + 2, OBS)).astype(np.float32)
    d = D.with_index({'obs': obs, 'actions': rng.uniform(-.9, .9, (n, 3)).astype(np.float32),
                      'rewards': rng.normal(1., .1, n), 'corner_bonus': np.where(np.arange(n) % 97 == 5, 8., 0.),
                      'terminated': np.arange(n) == 399, 'truncated': np.arange(n) == n - 1,
                      'ep_start_tr': np.array([0, 400]), 'ep_start_obs': np.array([0, 401]),
                      'ep_cause': np.array(['off_track', 'time_limit'])})
    D.prefill_replay_buffer(model.replay_buffer, d)
    model.set_frozen(True, True)
    seen, real = [], E.q_vs_mc

    def spy(critic, o, a, rewards, *args, **kw):
        seen.append(np.asarray(rewards))
        return real(critic, o, a, rewards, *args, **kw)
    monkeypatch.setattr(E, 'q_vs_mc', spy)
    run = T.Run(c, tmp_path, lambda *a: None)
    T._set_bonus_w0(c, model)
    T._offline_warmup(run, model, d, [0])
    np.testing.assert_allclose(seen[0], d['rewards'][:400] + d['corner_bonus'][:400])
    np.testing.assert_allclose(seen[1], d['rewards'][400:] + d['corner_bonus'][400:])
    seen.clear()
    model.replay_buffer.bonus_weight = 0.                         # w = 0: the rewards as stored
    T._offline_warmup(run, model, d, [0])
    np.testing.assert_array_equal(seen[0], d['rewards'][:400])


# --- actor lr from the unfreeze (docs/specs/R4_scratch_review.md B1) ------------------------------------------
BC_PRESETS = sorted(name for name, make in PRESETS.items() if make().agent.init_mode == 'bc')


def staged(name, tmp_path):
    """(cfg, tiny frozen model, its PhaseCallback as run_pipeline / resume_pipeline build it, the run)."""
    c = PRESETS[name]()
    cfg = dataclasses.replace(c, n_envs=1, agent=dataclasses.replace(c.agent, buffer_size=5_000))
    model = make_model(cfg, fill=600)
    model.set_frozen(True, True)
    run = T.Run(cfg, tmp_path, lambda *a: None)
    run.t_learn, run.step0 = time.time(), 0
    cbs = T._callbacks(run, model, {'probe': None}, [], 500., unfreeze_at=20_000)[0]
    phase = next(cb for cb in cbs if isinstance(cb, PhaseCallback))
    phase.init_callback(model)
    return cfg, model, phase, run


def lr_after_train(model):
    model.train(1, 64)
    return model.actor.optimizer.param_groups[0]['lr'], model.critic.optimizer.param_groups[0]['lr']


def test_only_the_new_preset_sets_an_actor_lr_on_the_bc_path():
    assert 'f1_best_w35' in BC_PRESETS and len(BC_PRESETS) >= 9
    assert {n for n in BC_PRESETS if PRESETS[n]().warm.actor_lr is not None} == {'f1_r4_scratch'}


@pytest.mark.parametrize('name', BC_PRESETS)
def test_actor_lr_right_after_the_unfreeze(name, tmp_path):
    _, model, phase, _ = staged(name, tmp_path)
    assert model.actor_lr is None and lr_after_train(model) == (3e-4, 3e-4)      # frozen phase: own settings
    model.num_timesteps = 19_999
    assert phase.on_step() and model.freeze_actor and model.actor_lr is None     # not yet
    model.num_timesteps = 20_000
    assert phase.on_step() and not model.freeze_actor
    want = 1e-4 if name == 'f1_r4_scratch' else 3e-4
    assert model.actor_lr == (1e-4 if name == 'f1_r4_scratch' else None)
    actor, critic = lr_after_train(model)
    assert actor == pytest.approx(want, abs=1e-12) and critic == 3e-4


def test_the_warm_path_ignores_the_phase_callback_setting(tmp_path):
    c = PRESETS['f1_r4']()
    cfg = dataclasses.replace(c, n_envs=1, agent=dataclasses.replace(c.agent, buffer_size=5_000, low_noise_envs=()))
    model = make_model(cfg, fill=600)
    run = T.Run(cfg, tmp_path, lambda *a: None)
    cbs = T._callbacks(run, model, {'probe': NS(unwrapped=NS(track=NS(length=5388.)))}, [], 500., unfreeze_at=20_000)[0]
    assert next(cb for cb in cbs if isinstance(cb, PhaseCallback)).actor_lr is None      # warm_start sets model.actor_lr


def test_resume_after_the_unfreeze_keeps_the_lr_from_the_saved_model(tmp_path):
    """Not config-driven and not the Adam state: `actor_lr` is a pickled attribute of the model, and
    StagedSAC._update_learning_rate re-applies it at every train()."""
    cfg, model, phase, run = staged('f1_r4_scratch', tmp_path)
    model.num_timesteps = 20_000
    phase.on_step()
    model.save(tmp_path / 'last.zip')
    loaded = StagedSAC.load(tmp_path / 'last.zip', env=DummyVecEnv([TinyEnv]), device='cpu')
    loaded.replay_buffer = model.replay_buffer
    loaded.set_logger(model.logger)
    assert loaded.actor_lr == 1e-4 and not loaded.freeze_actor
    cbs = T._callbacks(run, loaded, {'probe': None}, [], 500., unfreeze_at=0)[0]         # as resume_pipeline after the unfreeze
    again = next(cb for cb in cbs if isinstance(cb, PhaseCallback))
    again.init_callback(loaded)
    again.actor_lr = 5e-5                                         # would show if a resume re-applied the config value
    assert again.on_step() and loaded.actor_lr == 1e-4
    assert lr_after_train(loaded)[0] == pytest.approx(1e-4, abs=1e-12)


def test_resume_inside_the_frozen_phase_applies_it_at_the_unfreeze(tmp_path):
    cfg, model, _, run = staged('f1_r4_scratch', tmp_path)
    model.num_timesteps = 5_000
    model.save(tmp_path / 'last.zip')
    loaded = StagedSAC.load(tmp_path / 'last.zip', env=DummyVecEnv([TinyEnv]), device='cpu')
    loaded.replay_buffer = model.replay_buffer
    loaded.set_logger(model.logger)
    assert loaded.freeze_actor and loaded.actor_lr is None
    cbs = T._callbacks(run, loaded, {'probe': None}, [], 500., unfreeze_at=20_000)[0]
    phase = next(cb for cb in cbs if isinstance(cb, PhaseCallback))
    phase.init_callback(loaded)
    loaded.num_timesteps = 20_000
    assert phase.on_step() and loaded.actor_lr == 1e-4 and not loaded.freeze_actor
