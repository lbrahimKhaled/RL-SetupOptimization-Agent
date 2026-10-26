"""R1 spec §7 test 12 (slow): a mini r1_warm_smoke warm-started from the last.zip of a mini D2
pipeline (the test_d2_slow settings); files written; the step-0 deterministic evaluation equals
the source's final one; the source's replay buffer is preloaded and relabelled (Amendment 1),
the offline critic phase runs; the actor stays frozen (same digest) through the critic phases,
also across a resume, and unfreezes at warm.critic_warmup_steps with the critic check logged."""
import json
import pytest
from test_d2_slow import MINI, cli, jsonl

pytestmark = pytest.mark.slow

WARM = ['--set', 'n_envs=2', '--set', 'max_wall_h=None', '--set', 'checkpoint_interval=1000',
        '--set', 'agent.buffer_size=20000', '--set', 'warm.critic_warmup_steps=2000',
        '--set', 'warm.offline_critic_grad_steps=300', '--set', 'warm.offline_block=200',
        '--set', 'guard.n_starts=2', '--set', 'guard.probe_s=4.0',
        '--set', 'guard.interval=1000', '--set', 'guard.window=1500', '--set', 'eval.interval=1000000',
        '--set', 'eval.stochastic_laps=1', '--set', 'env.eval_max_s=40.0']   # env must equal the source's


def test_mini_r1_warm_smoke_from_mini_pipeline(tmp_path):
    from stable_baselines3 import SAC
    from rlsetup.driving.agent import StagedSAC, param_digest
    root = tmp_path / 'runs'
    mini = list(MINI)
    i = mini.index('total_timesteps=3000')
    mini[i] = 'total_timesteps=2000'
    cli('--preset', 'smoke', '--seed', '0', '--run-name', 'src', '--set', f'runs_root={root}',
        '--set', f'demo_dir={tmp_path / "demos"}', *mini)
    src_dir = root / 'src' / 'seed0'
    src_zip = src_dir / 'checkpoints' / 'last.zip'
    src_eval = jsonl(src_dir / 'evals.jsonl')[-1]
    assert src_eval['step'] == 2000

    cli('--preset', 'r1_warm_smoke', '--seed', '0', '--run-name', 'warm', '--init-from', str(src_zip),
        '--set', f'runs_root={root}', '--set', 'total_timesteps=1000', *WARM)
    d = root / 'warm' / 'seed0'
    for f in ('config.json', 'provenance.json', 'status.json', 'probe_starts.json', 'evals.jsonl',
              'train_stats.jsonl', 'laps/step_0_det.npz', 'checkpoints/step_1000.zip', 'checkpoints/last.zip',
              'replay_buffer.pkl', 'sb3/progress.csv', 'stdout.log'):
        assert (d / f).exists(), f
    assert not (d / 'bc').exists() or not any((d / 'bc').iterdir())          # no demos / BC
    cfg = json.loads((d / 'config.json').read_text())
    assert cfg['env']['reward']['mode'] == 'time_pbrs' and cfg['agent']['init_mode'] == 'warm'
    assert cfg['init_from'] == str(src_zip.resolve())
    prov = json.loads((d / 'provenance.json').read_text())
    status = json.loads((d / 'status.json').read_text())
    init = status['init_from']
    assert prov['init_from']['sha256'] == init['sha256'] and prov['init_from']['src_num_timesteps'] == 2000
    assert init['src_rlsetup_commit'] == json.loads((src_dir / 'provenance.json').read_text())['rlsetup']['commit']
    assert status['baseline_source'] == 'init_from' and status['state'] == 'completed'
    assert status['num_timesteps'] == 1000 and status['lifetime_timesteps'] == 3000
    assert status.get('unfrozen_at') is None
    from stable_baselines3.common.save_util import load_from_pkl
    src_rb = load_from_pkl(src_dir / 'replay_buffer.pkl')
    ib = status['init_buffer']
    assert ib['path'] == str((src_dir / 'replay_buffer.pkl').resolve()) and ib['rows'] == src_rb.size()
    assert ib['relabel']['rows'] == src_rb.size() and prov['init_buffer']['sha256'] == ib['sha256']
    assert cfg['warm']['init_buffer'] == ib['path']
    blocks = [r for r in jsonl(d / 'warmup.jsonl') if 'block' in r]
    assert [b['grad_steps'] for b in blocks] == [200, 300] and status['phases']['offline_warmup']['grad_steps'] == 300
    rb = load_from_pkl(d / 'replay_buffer.pkl')
    assert rb.pos == src_rb.pos + 1000 // 2                                  # preloaded rows + new rows
    ev = jsonl(d / 'evals.jsonl')
    assert [r['step'] for r in ev] == [0, 1000] and all(r['freeze_actor'] for r in ev)
    assert all(r['reward_mode'] == 'time_pbrs' for r in ev)
    for k in ('det_termination', 'det_decisions', 'lap1_s', 'lap2_s', 'det_clean'):   # same actor, same dynamics
        assert ev[0][k] == src_eval[k], k
    src_model = SAC.load(src_zip, device='cpu')
    last = StagedSAC.load(d / 'checkpoints' / 'last.zip', device='cpu')
    assert last.num_timesteps == 1000 and last._n_updates == 300 + 1000 // 2 * 2   # offline + online (UTD .5)
    assert param_digest(last.actor) == param_digest(src_model.actor) == init['actor_sha256']
    assert param_digest(last.log_ent_coef) == init['log_ent_coef_sha256']

    # resume inside the critic-only phase: still frozen until 2000, then unfreezes; critic check
    cli('--preset', 'r1_warm_smoke', '--resume', '--seed', '0', '--run-name', 'warm', '--set', f'runs_root={root}',
        '--set', 'total_timesteps=3000')
    status = json.loads((d / 'status.json').read_text())
    assert status['state'] == 'completed' and status['num_timesteps'] == 3000 and status['unfrozen_at'] == 2000
    unfreeze = [e for e in status['events'] if e['kind'] == 'unfreeze']
    assert len(unfreeze) == 1 and unfreeze[0]['step'] == 2000
    assert unfreeze[0]['actor_sha256'] == init['actor_sha256']                # A4: frozen through the resume
    assert unfreeze[0]['log_ent_coef_sha256'] == init['log_ent_coef_sha256']
    ev = jsonl(d / 'evals.jsonl')
    assert [r['step'] for r in ev] == [0, 1000, 2000, 3000]               # QA M1: extra eval at the unfreeze
    assert 'critic_check' in ev[2] and ev[2]['critic_check_tol'] == .08 and 'critic_check' not in ev[3]
    for k in ('det_decisions', 'lap1_s', 'lap2_s'):                          # actor not moved yet at the check
        assert ev[2][k] == ev[0][k], k
    assert param_digest(StagedSAC.load(d / 'checkpoints' / 'last.zip', device='cpu').actor) != init['actor_sha256']
