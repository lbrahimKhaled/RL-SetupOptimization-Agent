"""Immutable, run-specific inputs for offline comparisons (stdlib only; never loads torch)."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    """Exclusive create: historical artifacts must never be overwritten."""
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def checkpoint_step(path):
    """Read SB3's plain JSON timestep count, without deserializing executable objects."""
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError(f'invalid checkpoint ZIP: {path}')
        step = json.loads(z.read('data'))['num_timesteps']
    if type(step) is not int or step < 0:
        raise ValueError(f'checkpoint num_timesteps must be a nonnegative integer: {step!r}')
    return step


def select_eval(records, step):
    """The checkpoint must have its own clean logged deterministic evaluation."""
    matching = [r for r in records if r.get('step') == step and r.get('det_clean') is True
                and r.get('det_termination') == 'lap_complete' and r.get('det_truncation') is None
                and isinstance(r.get('lap2_s'), (int, float))
                and math.isfinite(r['lap2_s']) and r['lap2_s'] > 0.]
    if not matching:
        raise ValueError(f'no clean deterministic eval record for checkpoint num_timesteps={step}')
    if len({r['lap2_s'] for r in matching}) != 1:
        raise ValueError(f'conflicting clean eval records at checkpoint num_timesteps={step}')
    return matching[-1]


def read_evals(data):
    records = []
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A live trainer may be appending the final record during this read.
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                continue
            raise
    return records


def freeze_checkpoint(source, destination):
    """Copy from one open inode, so an atomic trainer rename cannot mix two checkpoints."""
    destination = Path(destination)
    with Path(source).open('rb') as src, destination.open('xb') as dst:
        before = os.fstat(src.fileno())
        while block := src.read(1024 * 1024):
            dst.write(block)
        after = os.fstat(src.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('checkpoint changed in place while copying; snapshot is not usable')
    if destination.stat().st_size != before.st_size:
        raise RuntimeError('checkpoint snapshot size mismatch')
    step = checkpoint_step(destination)
    destination.chmod(0o444)
    return {'source': str(Path(source).resolve()), 'snapshot': destination.name,
            'sha256': digest(destination.read_bytes()), 'bytes': before.st_size, 'num_timesteps': step}


def prepare_snapshot(run_dir, out, checkpoint='best.zip'):
    """Create a NEW output directory and freeze config, provenance, checkpoint and eval record.

    Missing records are retried briefly: the trainer atomically saves best.zip immediately
    before appending its evaluation. Never read the live checkpoint again after the copy.
    """
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if Path(checkpoint).name != checkpoint:
        raise ValueError('--checkpoint must be a filename in the run checkpoints directory')
    config = (run_dir / 'config.json').read_bytes()
    provenance = (run_dir / 'provenance.json').read_bytes()
    recorded_hash = json.loads(provenance).get('config_sha256')
    if recorded_hash != digest(config):
        raise ValueError(f'run config hash does not match saved provenance: {run_dir}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_bytes(config)
    (out / 'provenance.json').write_bytes(provenance)
    ckpt = freeze_checkpoint(run_dir / 'checkpoints' / checkpoint, out / 'checkpoint.zip')
    selected = None
    for attempt in range(21):
        eval_data = (run_dir / 'evals.jsonl').read_bytes()
        try:
            selected = select_eval(read_evals(eval_data), ckpt['num_timesteps'])
            break
        except ValueError:
            if attempt == 20:
                raise
            time.sleep(.1)
    (out / 'evals.snapshot.jsonl').write_bytes(eval_data)
    write_json(out / 'selected_eval.json', selected)
    snapshot = {'run': str(run_dir), 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'config_sha256': digest(config), 'provenance_sha256': digest(provenance),
                'evals_sha256': digest(eval_data), 'checkpoint': ckpt, 'selected_eval': selected}
    write_json(out / 'snapshot.json', snapshot)
    return snapshot


def source_differences(recorded, current):
    """Compare simulator/driver source identity, independent of unrelated comparison edits."""
    differences = []
    for group in ('rlsetup_sources_sha256', 'ocd_lap_extra_sources_sha256'):
        a, b = recorded.get(group, {}), current.get(group, {})
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                differences.append(f'{group}/{key}: saved {a.get(key)} != current {b.get(key)}')
    if recorded.get('ocd_lap_code_version') != current.get('ocd_lap_code_version'):
        differences.append('ocd_lap_code_version differs (includes native binary, simulator and track inputs)')
    return differences


def reproduction(measured, expected, clean, tolerance_s, training_device, inference_device='cpu'):
    """Measured-vs-logged deterministic result; CPU inference may differ from MPS training."""
    if not math.isfinite(tolerance_s) or tolerance_s < 0.:
        raise ValueError('lap tolerance must be finite and nonnegative')
    delta = None if measured is None else float(measured - expected)
    return {'expected_lap2_s': float(expected), 'measured_lap2_s': measured, 'difference_s': delta,
            'tolerance_s': tolerance_s,
            'within_tolerance': bool(clean and delta is not None and abs(delta) <= tolerance_s),
            'training_device': training_device, 'inference_device': inference_device,
            'device_changed': training_device != inference_device,
            'note': 'CPU and MPS inference can diverge; comparison uses the measured snapshot lap.'}
