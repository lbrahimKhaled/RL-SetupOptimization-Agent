"""AC0 static inventory: no simulator, package imports or model deserialization."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'ac_preflight.py'
SPEC = importlib.util.spec_from_file_location('ac_preflight', SCRIPT)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def write_config(root, data):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'config.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps(data))
        archive.writestr('policy.pth', b'not a tensor archive: must never be loaded')


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_unknown_identity_is_not_filled_from_folder_or_preset(tmp_path):
    path = write_config(tmp_path / 'sf70h_f1' / 'bcn', {'preset': 'f1_r4', 'env': {}})
    run = P.inspect_run(path)
    assert run['config']['env']['car'] is None
    assert run['config']['env']['track'] is None
    assert run['config']['env']['reward'] is None
    assert run['config']['env']['corner_bonus'] is None
    assert all(not c['exists'] for c in run['checkpoints'].values())
    assert run['errors'] == []


@pytest.mark.parametrize('reward,corner,expected', [
    ({'mode': 'time_pbrs'}, None, (None, None, None)),
    ({'mode': 'time_pbrs'}, {'enabled': True, 'bonus': 30., 'floor_ratio': .85},
     (None, None, 30.)),
    ({'mode': 'time_pbrs', 'speed_k': 20., 'speed_sigma': .3},
     {'enabled': True, 'mode': 'entry_exit', 'bonus_in': 10., 'bonus_out': 10.},
     (20., 'entry_exit', None)),
])
def test_reward_flags_preserved_without_inventing_defaults(tmp_path, reward, corner, expected):
    env = {'car': 'sf70h_f1', 'track': 'Catalunya', 'reward': reward}
    if corner is not None:
        env['corner_bonus'] = corner
    run = P.inspect_run(write_config(tmp_path, {'env': env, 'secret': 'omit-me'}))
    selected = run['config']['env']
    assert (selected['reward']['speed_k'], (selected['corner_bonus'] or {}).get('mode'),
            (selected['corner_bonus'] or {}).get('bonus')) == expected
    assert 'omit-me' not in json.dumps(run)


def test_checkpoint_reads_only_literal_json_never_pickle(tmp_path, monkeypatch):
    path = write_config(tmp_path, {'env': {'car': 'sf70h_f1'}})
    serialized = base64.b64encode(pickle.dumps({'message': 'do not load'})).decode('ascii')
    for name in P.CHECKPOINT_NAMES:
        checkpoint(tmp_path / 'checkpoints' / name,
                   {'num_timesteps': 123, 'gamma': .992, 'n_steps': 3, 'batch_size': 512,
                    'policy_class': {':serialized:': serialized},
                    'observation_space': {':serialized:': 'not even valid base64'}})
    monkeypatch.setattr(pickle, 'loads', lambda *_a, **_k: pytest.fail('pickle decoded'))
    monkeypatch.setattr(base64, 'b64decode', lambda *_a, **_k: pytest.fail('base64 decoded'))
    before = snapshot(tmp_path)
    run = P.inspect_run(path)
    assert snapshot(tmp_path) == before
    assert run['errors'] == []
    for name, saved in run['checkpoints'].items():
        assert saved['metadata'] == {'num_timesteps': 123, 'gamma': .992, 'n_steps': 3,
                                     'batch_size': 512}
        assert saved['sha256'] == hashlib.sha256((tmp_path / 'checkpoints' / name).read_bytes()).hexdigest()
        assert saved['consistent_read'] is True
    assert serialized not in json.dumps(run)


def test_serialized_numeric_metadata_is_not_interpreted(tmp_path):
    path = tmp_path / 'last.zip'
    checkpoint(path, {'num_timesteps': {':serialized:': 'anything'}, 'gamma': True})
    errors = []
    result = P.inspect_checkpoint(path, errors)
    assert result['metadata']['num_timesteps'] is None
    assert result['metadata']['gamma'] is None
    assert len(errors) == 2


@pytest.mark.parametrize('contents', ['{', '[]', '{"env":NaN}', '{"env":"bad"}'])
def test_malformed_config_reported(tmp_path, contents):
    path = tmp_path / 'config.json'
    path.write_text(contents)
    assert P.inspect_run(path)['errors']


def test_missing_config_reported(tmp_path):
    result = P.inspect_run(tmp_path / 'config.json')
    assert result['config'] is None
    assert result['errors']


@pytest.mark.parametrize('kind', ['corrupt', 'missing_data', 'invalid_data', 'nonobject_data'])
def test_malformed_checkpoint_reported(tmp_path, kind):
    path = tmp_path / 'bad.zip'
    if kind == 'corrupt':
        path.write_bytes(b'broken')
    else:
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('other' if kind == 'missing_data' else 'data',
                             '[]' if kind == 'nonobject_data' else '{')
    errors = []
    result = P.inspect_checkpoint(path, errors)
    assert result['exists']
    assert errors


def test_evals_only_rank_valid_clean_deterministic_lap_two(tmp_path):
    path = tmp_path / 'evals.jsonl'
    rows = [
        {'step': 0, 'det_clean': False, 'lap2_s': None, 'sto_lap_mean_s': 1},
        {'step': 10, 'det_clean': True, 'lap2_s': 87.},
        {'step': 20, 'det_clean': True, 'lap2_s': 85.},
        {'step': 30, 'det_clean': False, 'lap2_s': 1.},
        {'step': 40, 'det_clean': 'true', 'lap2_s': 1.},
        {'step': True, 'det_clean': True, 'lap2_s': 1.},
        {'step': 50, 'det_clean': True, 'lap2_s': -1.},
        {'step': 60, 'det_clean': True, 'lap2_s': True},
        {'step': 70, 'det_clean': True, 'lap2_s': None},
        {'step': 80, 'det_clean': True, 'lap2_s': float('nan')},
        [],
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n{\n\n')
    result = P.inspect_evals(path, [])
    assert result['rows'] == 12
    assert result['invalid_rows'] == 8
    assert result['clean_rows'] == 2
    assert result['best_logged_clean_lap2'] == {'lap2_s': 85., 'step': 20}
    assert result['checkpoint_association_verified'] is False


def test_duplicate_run_roots_deduplicated_and_read_only(tmp_path):
    path = write_config(tmp_path / 'run' / 'seed0', {'env': {'track': 'Catalunya'}})
    before = snapshot(tmp_path)
    result = P.build_report(run_roots=[tmp_path, path.parent])
    assert len(result['runs']) == 1
    assert snapshot(tmp_path) == before


def test_no_paths_is_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        P.main([])
    assert exc.value.code == 2
    assert 'no default scan' in capsys.readouterr().err


def test_cli_json_errors_and_exit_code(tmp_path, capsys):
    assert P.main(['--run-root', str(tmp_path / 'missing')]) == 1
    report = json.loads(capsys.readouterr().out)
    assert 'does not exist' in report['errors'][0]
    assert P.main(['--run-root', str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert 'no config.json' in report['errors'][0]


def test_standalone_cli_works_without_site_packages(tmp_path):
    write_config(tmp_path, {'env': {'car': 'sf70h_f1', 'track': 'Catalunya'}})
    completed = subprocess.run([sys.executable, '-S', str(SCRIPT), '--run-root', str(tmp_path)],
                               capture_output=True, text=True, check=True)
    report = json.loads(completed.stdout)
    assert report['inspection_only']
    assert report['runs'][0]['config']['env']['car'] == 'sf70h_f1'
    assert completed.stderr == ''


def test_install_inspection_does_not_write_git_or_echo_paths(tmp_path):
    root = tmp_path / 'assetto_corsa_gym'
    root.mkdir()
    subprocess.run(['git', 'init', '--quiet', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '--quiet', '-m', 'fixture'], check=True)
    secret = root / 'personal-name.txt'
    secret.write_text('private')
    before = snapshot(root)
    result = P.inspect_install(root.resolve())
    assert result['errors'] == []
    assert len(result['commit']) == 40
    assert result['dirty'] is True
    assert result['installed_state_verified'] is False
    assert 'personal-name' not in json.dumps(result)
    assert snapshot(root) == before
    assert P.inspect_install(root / 'missing')['errors']


def test_subdirectory_is_not_misidentified_as_install_root(tmp_path):
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    child = tmp_path / 'child'
    child.mkdir()
    result = P.inspect_install(child.resolve())
    assert result['commit'] is None
    assert result['dirty'] is None
    assert result['errors']
