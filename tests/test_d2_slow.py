"""D2 §10 (slow): end-to-end mini pipeline through the CLI (demos 5k with 2 workers, BC 2
epochs, offline 200 grad steps, online 3k with 2 vec-env workers, evaluation at 0 and at the
end, one guard probe), then resume from checkpoints/last.zip + replay_buffer.pkl, then the
report script. Plus: demo data independent of the number of collection workers."""
import json
import subprocess
import sys
import numpy as np
import pytest
from rlsetup.provenance import REPO_ROOT

pytestmark = pytest.mark.slow

MINI = ['--set', 'n_envs=2', '--set', 'total_timesteps=3000', '--set', 'max_wall_h=None',
        '--set', 'checkpoint_interval=2000', '--set', 'agent.buffer_size=20000',
        '--set', 'demos.n_transitions=5000', '--set', 'demos.workers=2', '--set', 'demos.min_clean_fraction=0.0',
        '--set', 'bc.max_epochs=2', '--set', 'bc.gate_required=False', '--set', 'bc.gate_stochastic_laps=1',
        '--set', 'warmup.offline_blocks=2', '--set', 'warmup.offline_steps_per_block=100',
        '--set', 'warmup.online_warmup_steps=1000', '--set', 'guard.n_starts=2', '--set', 'guard.probe_s=4.0',
        '--set', 'guard.interval=1000', '--set', 'guard.window=1500', '--set', 'eval.interval=1000000',
        '--set', 'eval.stochastic_laps=1', '--set', 'env.eval_max_s=40.0']


def cli(*args):
    r = subprocess.run([sys.executable, '-m', 'rlsetup.driving.train', *args], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_mini_pipeline_resume_and_report(tmp_path):
    root = tmp_path / 'runs'
    common = ['--set', f'runs_root={root}', '--set', f'demo_dir={tmp_path / "demos"}']
    cli('--preset', 'smoke', '--seed', '0', '--run-name', 'mini', *common, *MINI)
    d = root / 'mini' / 'seed0'
    for f in ('config.json', 'provenance.json', 'status.json', 'demos.json', 'probe_starts.json',
              'bc/actor_bc.pt', 'bc/report.json', 'sb3/progress.csv', 'train_stats.jsonl', 'evals.jsonl',
              'warmup.jsonl', 'probes.jsonl', 'laps/step_0_det.npz', 'laps/step_3000_det.npz',
              'checkpoints/step_2000.zip', 'checkpoints/last.zip', 'replay_buffer.pkl', 'stdout.log'):
        assert (d / f).exists(), f
    status = json.loads((d / 'status.json').read_text())
    assert status['state'] == 'completed' and status['num_timesteps'] == 3000 and status['unfrozen_at'] == 1000
    assert {'demos', 'bc', 'offline_warmup', 'online_warmup', 'rl'} <= set(status['phases'])
    assert status['phases']['offline_warmup']['grad_steps'] == 200
    prov = json.loads((d / 'provenance.json').read_text())
    assert prov['demo_dataset_sha256'] == json.loads((d / 'demos.json').read_text())['sha256']
    assert len(prov['bc_actor_sha256']) == 64 and prov['n_envs'] == 2 and prov['packages']['stable-baselines3'] == '2.9.0'
    ev = jsonl(d / 'evals.jsonl')
    assert [r['step'] for r in ev] == [0, 3000]
    assert ev[0]['freeze_actor'] is True and ev[1]['freeze_actor'] is False
    assert all(r['sto_n'] == 1 and 'q_vs_mc' in r and 'lap2' in r for r in ev)
    assert ev[1]['train_obs_clip_fraction'] is not None and ev[1]['train_obs_clip_fraction'] < .01
    assert [p['step'] for p in jsonl(d / 'probes.jsonl')] == [2000]
    assert len([r for r in jsonl(d / 'warmup.jsonl') if 'block' in r]) == 2
    lap = np.load(d / 'laps' / 'step_3000_det.npz')
    assert {'s', 'x', 'y', 'speed', 'e', 'actions', 'gear', 'kappa', 'Fz'} <= set(lap.files)
    cfg = json.loads((d / 'config.json').read_text())
    assert cfg['n_envs'] == 2 and cfg['seed'] == 0 and cfg['preset'] == 'smoke'

    # resume: +1000 steps from last.zip + replay buffer
    cli('--preset', 'smoke', '--resume', '--seed', '0', '--run-name', 'mini', '--set', f'runs_root={root}',
        '--set', 'total_timesteps=4000')
    status = json.loads((d / 'status.json').read_text())
    assert status['state'] == 'completed' and status['num_timesteps'] == 4000
    assert [e['kind'] for e in status['events']] == ['unfreeze', 'resume']
    assert [r['step'] for r in jsonl(d / 'evals.jsonl')] == [0, 3000, 4000]
    assert (d / 'sb3' / 'resume1' / 'progress.csv').exists()
    from stable_baselines3.common.save_util import load_from_pkl
    rb = load_from_pkl(d / 'replay_buffer.pkl')
    prefill_rows = status['phases']['offline_warmup']['prefill']['rows']
    assert rb.pos == prefill_rows + 4000 // 2 and not rb.full          # demo rows + online rows (2 envs)

    # report (re-evaluates last.zip): criteria FAIL, no crash, deterministic re-eval reproduces
    r = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts' / 'd2_report.py'), str(root / 'mini')],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    rep = json.loads((root / 'mini' / 'd2_report.json').read_text())
    assert rep['criteria']['vi_logged'] and not rep['pass']
    assert rep['seeds'][0]['reeval_return_delta'] == 0.


def test_demo_data_independent_of_workers(tmp_path, track, qss):
    import dataclasses
    from rlsetup.driving import demos as D
    from rlsetup.driving.config import EnvConfig
    from rlsetup.driving.train_config import DemoDataConfig
    from rlsetup.driving.vec import track_sha256
    env_cfg = EnvConfig(max_episode_s=4.)
    one = DemoDataConfig(n_transitions=500, workers=1)
    p1, _ = D.collect_demos(env_cfg, one, qss, track_sha256(track), tmp_path / 'a', log=lambda *_: None)
    p2, _ = D.collect_demos(env_cfg, dataclasses.replace(one, workers=3), qss, track_sha256(track), tmp_path / 'b',
                            log=lambda *_: None)
    a, b = D.load_demos(p1), D.load_demos(p2)
    assert p1.name == p2.name
    for k in D.ARRAYS:
        assert a[k].tobytes() == b[k].tobytes(), k
