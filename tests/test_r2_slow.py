"""R2-10.12 (slow, sf70h_f1 + the f1_best_w35 run folder): a mini f1_r2_smoke warm-started from
f1_best_w35's best.zip (with a small init buffer of real transitions instead of the 1.3 GB
replay_buffer.pkl), stopped mid-fade and resumed: step-0 lap 2 = 101.87194035987982 s (S1) with the
lap-2 exit ratios of R2-11 S2; the plain init buffer is adopted (unlabelled); w(t) and `reportable`
follow the compressed schedule through the resume (S7); best_aided.zip before report_from, best.zip
only from the reportable evaluation; bonus / focus state files, provenance 'r2', grouped train stats.
The actor stays frozen (critic phase longer than the run), so every evaluation drives the same lap.
"""
import json
import numpy as np
import pytest
from rlsetup.provenance import REPO_ROOT
from test_d2_slow import cli, jsonl
from test_r2 import needs_src, SRC

pytestmark = [pytest.mark.slow, needs_src]

S2_RHO = (.912, .910, .849, .897, .894, .897, .926)
MINI = ['--set', 'n_envs=2', '--set', 'agent.low_noise_envs=(1,)', '--set', 'agent.buffer_size=20000',
        '--set', 'max_wall_h=None', '--set', 'checkpoint_interval=1000', '--set', 'eval.interval=1000',
        '--set', 'eval.stochastic_laps=1', '--set', 'warm.offline_critic_grad_steps=200',
        '--set', 'warm.offline_block=100', '--set', 'warm.critic_warmup_steps=100000',
        '--set', 'guard.n_starts=2', '--set', 'guard.probe_s=4.0', '--set', 'guard.interval=1000',
        '--set', 'bonus_schedule.hold_until=1000', '--set', 'bonus_schedule.fade_until=4000',
        '--set', 'bonus_schedule.bonus_free_before_report=1000', '--set', 'focus_update.update_every=500']


def _init_buffer(path, rows=1500):
    """NStepReplayBuffer (pre-R2 class, 2 env columns) of `rows` real transitions of the source
    actor (stochastic) on the f1_best_w35 env."""
    from stable_baselines3.common.buffers import NStepReplayBuffer
    from stable_baselines3.common.save_util import save_to_pkl
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from rlsetup.driving.agent import StagedSAC
    from rlsetup.driving.env import DrivingEnv
    from rlsetup.driving.evaluate import act
    from rlsetup.driving.train_config import PRESETS
    cfg = PRESETS['f1_best_w35']().env
    track = Track(cfg.track)
    qss = qss_profile.for_env(track, cfg)
    envs = [DrivingEnv(cfg, track=track, qss=qss) for _ in range(2)]
    model = StagedSAC.load(SRC / 'checkpoints' / 'best.zip', device='cpu')
    rb = NStepReplayBuffer(20000, model.observation_space, model.action_space, n_envs=2, n_steps=3, gamma=.992)
    obs = [e.reset(seed=50 + i)[0] for i, e in enumerate(envs)]
    for t in range(rows):
        nxt, acts, rews, dones, infos = [], [], [], [], []
        for i, e in enumerate(envs):
            a, _ = act(model.actor, obs[i], False)
            o2, r, te, tr, info = e.step(a)
            nxt.append(o2), acts.append(a), rews.append(r), dones.append(te or tr)
            infos.append({'TimeLimit.truncated': bool(tr and not te)})
        rb.add(np.array(obs), np.array(nxt), np.array(acts), np.array(rews, np.float32), np.array(dones), infos)
        obs = [e.reset(seed=1000 + t)[0] if d else o for e, o, d in zip(envs, nxt, dones)]
    save_to_pkl(path, rb)
    return rb


def test_mini_f1_r2_smoke_resume_mid_fade(tmp_path):
    from stable_baselines3.common.save_util import load_from_pkl
    from rlsetup.driving.buffers import BonusNStepReplayBuffer
    root = tmp_path / 'runs'
    buf = tmp_path / 'init_buffer.pkl'
    _init_buffer(buf)
    best = SRC / 'checkpoints' / 'best.zip'
    cli('--preset', 'f1_r2_smoke', '--seed', '0', '--run-name', 'r2', '--init-from', str(best),
        '--init-buffer', str(buf), '--set', f'runs_root={root}', '--set', 'total_timesteps=2000', *MINI)
    d = root / 'r2' / 'seed0'
    for f in ('config.json', 'provenance.json', 'status.json', 'evals.jsonl', 'bonus_state.json', 'focus_state.json',
              'checkpoints/last.zip', 'checkpoints/best_aided.zip', 'replay_buffer.pkl'):
        assert (d / f).exists(), f
    assert not (d / 'checkpoints' / 'best.zip').exists()
    status = json.loads((d / 'status.json').read_text())
    assert status['state'] == 'completed' and status['num_timesteps'] == 2000
    assert status['lifetime_steps_offset'] == 9_200_000 and status['report_from'] == 5000
    assert status['init_buffer']['adopted'] and status['init_buffer']['labelled_fraction'] == 0.
    prov = json.loads((d / 'provenance.json').read_text())
    r2 = prov['r2']
    assert len(r2['corner_table']['units']) == 7 and r2['focus_seed']['total_crashes'] == 166
    assert r2['labelled_fraction_at_start'] == 0. and r2['low_noise'] == {'envs': [1], 'std_scale': .3}
    ev = jsonl(d / 'evals.jsonl')
    assert [r['step'] for r in ev] == [0, 1000, 2000]
    e0 = ev[0]
    assert e0['det_clean'] and abs(e0['lap2_s'] - 101.87194035987982) <= 1e-6                     # S1
    rho = [e0['lap2_corners'][f'U{k}']['rho'] for k in range(1, 8)]
    assert np.allclose(rho, S2_RHO, atol=.015), rho                                                # S2
    assert 0. < e0['lap2_bonus_sum'] <= 7 * 3.
    assert [r['bonus_w'] for r in ev] == pytest.approx([1., 1., 2 / 3])
    assert [r['reportable'] for r in ev] == [False] * 3 and ev[0]['new_best_aided']
    assert all(r['lap2_s'] == e0['lap2_s'] for r in ev)                                           # frozen actor
    assert ev[0]['focus']['resets'] == 2 and len(ev[0]['focus']['top5']) == 5           # the two initial resets
    assert ev[0]['focus']['top5'][0][0] == 2350 and ev[-1]['buffer_labelled_fraction'] > 0
    assert set(ev[-1]['train_groups']) == {'normal', 'low_noise', 'focused', 'uniform'}
    rb = load_from_pkl(d / 'replay_buffer.pkl')
    assert type(rb) is BonusNStepReplayBuffer and rb.pos == 1500 + 1000
    assert not rb.labelled[:1500].any() and rb.labelled[1500:2500].all()
    st = json.loads((d / 'bonus_state.json').read_text())
    assert st['trigger_step'] is None and st['report_from'] == 5000

    cli('--preset', 'f1_r2_smoke', '--resume', '--seed', '0', '--run-name', 'r2', '--set', f'runs_root={root}',
        '--set', 'total_timesteps=5000')
    ev = jsonl(d / 'evals.jsonl')
    assert [r['step'] for r in ev] == [0, 1000, 2000, 3000, 4000, 5000]
    assert [r['bonus_w'] for r in ev] == pytest.approx([1., 1., 2 / 3, 1 / 3, 0., 0.])          # S7
    assert [r['reportable'] for r in ev] == [False] * 5 + [True]
    assert [bool(r.get('new_best')) for r in ev] == [False] * 5 + [True]
    assert (d / 'checkpoints' / 'best.zip').exists()
    status = json.loads((d / 'status.json').read_text())
    assert status['state'] == 'completed' and status['num_timesteps'] == 5000
    assert [e['kind'] for e in status['events']].count('resume') == 1
    assert 'focus_state.json missing' not in json.dumps(status.get('warnings', []))
