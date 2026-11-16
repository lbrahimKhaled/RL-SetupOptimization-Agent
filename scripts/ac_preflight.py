"""Read-only AC installation / OCD run inventory; Python 3.8+, standard library only.

Inspect only explicitly supplied trees. Print JSON to stdout; do not import the game,
RL packages, or checkpoint classes, and never deserialize pickle/base64 fields.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import zipfile


PACKAGES = ('torch', 'stable-baselines3', 'sb3-contrib', 'numpy', 'gym', 'gymnasium',
            'assetto-corsa-gym')
ENV_KEYS = ('car', 'track', 'setup', 'reference', 'action_repeat', 'obs_gear',
            'pedal_deadzone', 'ers', 'max_brake_torque_Nm', 'max_episode_s')
REWARD_KEYS = ('mode', 'a', 'b', 'c', 'w', 'gamma', 'shaping_k', 'penalty_const',
               'speed_k', 'speed_sigma')
CORNER_KEYS = ('enabled', 'mode', 'bonus', 'floor_ratio', 'bonus_in', 'bonus_out',
               'sigma', 'prominence_mps', 'merge_m', 'exit_after_apex_m',
               'exit_before_brake_m', 'grid_m')
OBS_KEYS = ('n_preview', 'preview_horizon_m', 'preview_exponent', 'preview_min_scale_m',
            'grid_m', 'history', 'speed_scale_mps', 'vy_scale_mps', 'accel_scale_mps2',
            'yaw_rate_scale_radps', 'steer_scale_rad', 'heading_scale_rad', 'slip_gain',
            'slip_clip', 'e_scale_m', 'gear_scale', 'clip')
AGENT_KEYS = ('batch_size', 'buffer_size', 'critic_layernorm', 'ent_coef', 'n_critics',
              'n_steps', 'net_arch', 'target_entropy', 'tau', 'train_freq',
              'learning_rate', 'learning_starts', 'init_mode', 'low_noise_envs',
              'low_noise_std_scale')
TRAIN_KEYS = ('preset', 'seed', 'device', 'total_timesteps', 'n_envs', 'gradient_steps',
              'torch_threads', 'vec_start_method')
CHECKPOINT_KEYS = ('num_timesteps', 'gamma', 'n_steps', 'batch_size')
CHECKPOINT_NAMES = ('best.zip', 'best_aided.zip', 'last.zip')
MAX_ZIP_METADATA_BYTES = 8 * 1024 * 1024  # Inspection bound; no model tensors are read.


def _reject_constant(value):
    raise ValueError('non-finite JSON number: ' + value)


def _json(text):
    return json.loads(text, parse_constant=_reject_constant)


def _read_object(path, errors):
    try:
        result = _json(path.read_text(encoding='utf-8-sig'))
        if not isinstance(result, dict):
            raise ValueError('expected a JSON object')
        return result
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f'{path}: {exc}')
        return None


def _plain(value):
    """Allow scalar values and flat scalar lists, never serialized object dictionaries."""
    return (value is None or type(value) in (str, bool, int)
            or (type(value) is float and math.isfinite(value)))


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _select(source, keys, label, errors):
    if source is None:
        return None
    if not isinstance(source, dict):
        errors.append(f'{label}: expected an object')
        return None
    result = {}
    for key in keys:
        value = source.get(key)
        if not (_plain(value) or (isinstance(value, list) and all(_plain(x) for x in value))):
            errors.append(f'{label}.{key}: expected a scalar or flat scalar list')
            value = None
        result[key] = value
    return result


def inspect_checkpoint(path, errors):
    result = {'path': str(path), 'exists': path.exists()}
    if not result['exists']:
        return result
    try:
        before = path.stat()
        result['size_bytes'] = before.st_size
        with path.open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result['sha256'] = digest.hexdigest()
        with zipfile.ZipFile(path) as archive:
            member = archive.getinfo('data')
            if member.file_size > MAX_ZIP_METADATA_BYTES:
                raise ValueError('checkpoint data JSON exceeds the 8 MiB inspection limit')
            data = _json(archive.read(member).decode('utf-8'))
        if not isinstance(data, dict):
            raise ValueError('checkpoint data must be a JSON object')
        result['metadata'] = {}
        for key in CHECKPOINT_KEYS:
            value = data.get(key)
            # Only literal numeric metadata; even these fields may be serialized objects.
            if value is not None and not _number(value):
                errors.append(f'{path}: {key} is not literal finite numeric metadata')
                value = None
            result['metadata'][key] = value
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            errors.append(f'{path}: changed during inspection; repeat when checkpoint saving stops')
            result['consistent_read'] = False
        else:
            result['consistent_read'] = True
    except (OSError, UnicodeError, ValueError, KeyError, zipfile.BadZipFile,
            RuntimeError, NotImplementedError) as exc:
        errors.append(f'{path}: {exc}')
    return result


def inspect_evals(path, errors):
    """Existing evals.jsonl schema: det_clean, lap2_s and step (callbacks.py)."""
    result = {'path': str(path), 'exists': path.exists(), 'rows': 0, 'invalid_rows': 0,
              'clean_rows': 0, 'best_logged_clean_lap2': None,
              'checkpoint_association_verified': False}
    if not result['exists']:
        return result
    try:
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():
                    continue
                result['rows'] += 1
                try:
                    row = _json(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected an object')
                    step, clean, lap = row.get('step'), row.get('det_clean'), row.get('lap2_s')
                    if type(step) is not int or step < 0 or type(clean) is not bool:
                        raise ValueError('invalid step or det_clean')
                    if not clean:
                        continue
                    if not _number(lap) or lap <= 0:
                        raise ValueError('invalid clean lap time')
                except (ValueError, TypeError):
                    result['invalid_rows'] += 1
                    continue
                result['clean_rows'] += 1
                best = result['best_logged_clean_lap2']
                if best is None or lap < best['lap2_s']:
                    result['best_logged_clean_lap2'] = {'lap2_s': lap, 'step': step}
    except (OSError, UnicodeError) as exc:
        errors.append(f'{path}: {exc}')
    return result


def inspect_run(config_path):
    errors = []
    root = config_path.parent
    result = {'run_dir': str(root), 'config_path': str(config_path), 'errors': errors}
    config = _read_object(config_path, errors)
    result['config'] = None
    if config is not None:
        selected = _select(config, TRAIN_KEYS, str(config_path), errors)
        env = config.get('env')
        selected['env'] = _select(env, ENV_KEYS, f'{config_path}:env', errors)
        if isinstance(env, dict):
            for name, keys in (('reward', REWARD_KEYS), ('corner_bonus', CORNER_KEYS), ('obs', OBS_KEYS)):
                selected['env'][name] = _select(env.get(name), keys, f'{config_path}:env.{name}', errors)
        selected['agent'] = _select(config.get('agent'), AGENT_KEYS, f'{config_path}:agent', errors)
        selected['bonus_schedule'] = _select(config.get('bonus_schedule'),
            ('hold_until', 'fade_until', 'trigger_lap2_s', 'bonus_free_before_report'),
            f'{config_path}:bonus_schedule', errors)
        result['config'] = selected
    # Expose existence/paths only: provenance has machine names and arbitrary command arguments.
    result['provenance'] = {'path': str(root / 'provenance.json'),
                            'exists': (root / 'provenance.json').is_file()}
    result['checkpoints'] = {name: inspect_checkpoint(root / 'checkpoints' / name, errors)
                             for name in CHECKPOINT_NAMES}
    result['evals'] = inspect_evals(root / 'evals.jsonl', errors)
    return result


def inspect_install(root):
    errors = []
    result = {'root': str(root), 'commit': None, 'dirty': None, 'errors': errors}
    if not root.is_dir():
        errors.append(f'{root}: installation directory does not exist')
        return result
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

    def git(*args):
        completed = subprocess.run(
            ['git', '-c', 'core.fsmonitor=false', '-C', str(root), *args],
            capture_output=True, text=True, timeout=10, env=env, check=True)
        return completed.stdout.strip()

    try:
        if Path(git('rev-parse', '--show-toplevel')).resolve() != root:
            raise ValueError('supplied path is not the root of its Git checkout')
        result['commit'] = git('rev-parse', 'HEAD')
        result['dirty'] = bool(git('status', '--porcelain', '--untracked-files=normal'))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # Do not echo arbitrary subprocess output (which can include local Git configuration).
        errors.append(f'{root}: cannot read Git identity ({type(exc).__name__}: {exc})')
    result['config_candidates'] = [str(p) for p in sorted(root.rglob('*'))
                                    if p.is_file() and p.name in ('config.yml', 'config.yaml')]
    result['installed_state_verified'] = False  # A checkout is not proof the game plugin runs.
    return result


def build_report(ac_gym_root=None, run_roots=()):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {'schema_version': 1, 'inspection_only': True,
              'host': {'os': platform.system(), 'os_release': platform.release(),
                       'machine': platform.machine(), 'python': platform.python_version(),
                       'packages': packages},
              'ac_gym': None, 'run_roots': [], 'runs': [], 'errors': []}
    if ac_gym_root is not None:
        report['ac_gym'] = inspect_install(Path(ac_gym_root).expanduser().resolve())
    configs = set()
    for raw in run_roots:
        root = Path(raw).expanduser().resolve()
        report['run_roots'].append(str(root))
        if not root.is_dir():
            report['errors'].append(f'{root}: run directory does not exist')
            continue
        found = set(root.rglob('config.json'))
        if not found:
            report['errors'].append(f'{root}: no config.json found')
        configs.update(found)
    report['runs'] = [inspect_run(p) for p in sorted(configs)]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ac-gym-root', type=Path, help='exact assetto_corsa_gym Git checkout root')
    parser.add_argument('--run-root', type=Path, action='append', default=[],
                        help='run/seed directory or runs tree to inspect (repeatable)')
    args = parser.parse_args(argv)
    if args.ac_gym_root is None and not args.run_root:
        parser.error('provide --ac-gym-root and/or --run-root; there is no default scan')
    report = build_report(args.ac_gym_root, args.run_root)
    print(json.dumps(report, indent=2, allow_nan=False))
    has_errors = (report['errors'] or any(r['errors'] for r in report['runs'])
                  or (report['ac_gym'] and report['ac_gym']['errors'])
                  or any(r['evals']['invalid_rows'] for r in report['runs']))
    return 1 if has_errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
