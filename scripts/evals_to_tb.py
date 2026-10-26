"""Mirror a seed's evals.jsonl (lap times, clean rates, Q vs MC, alpha, throughput) into TensorBoard.

    .venv/bin/python scripts/evals_to_tb.py runs/D2/full1/seed0

Appends to <seed_dir>/tb_evals/: each call writes only evaluations newer than the last one
mirrored (kept in tb_evals/last_step.txt) into a new event file in the same folder. TensorBoard
merges the files of a folder; deleting and rewriting them instead leaves a running TensorBoard
on the stale, deleted file. Point TensorBoard at the seed folder to see these next to SB3's own
losses in sb3/.
"""
import calendar
import json
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main(seed_dir):
    seed_dir = Path(seed_dir)
    out = seed_dir / 'tb_evals'
    out.mkdir(exist_ok=True)
    marker = out / 'last_step.txt'
    last = int(marker.read_text()) if marker.exists() else -1
    writer = None
    for line in (seed_dir / 'evals.jsonl').read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:   # line being written right now
            continue
        step = e['step']
        if step <= last:
            continue
        writer = writer or SummaryWriter(str(out))
        wall = calendar.timegm(time.strptime(e['utc'], '%Y-%m-%dT%H:%M:%SZ')) if e.get('utc') else None
        scalars = {
            'lap/deterministic_lap2_s': e.get('lap2_s') if e.get('det_clean') else None,
            'lap/deterministic_clean': float(bool(e.get('det_clean'))),
            'lap/stochastic_mean_lap_s': e.get('sto_lap_mean_s'),
            'lap/stochastic_clean_rate': e.get('sto_clean_rate'),
            'train/alpha': e.get('alpha'),
            'train/policy_entropy': e.get('policy_entropy'),
            'speed/env_steps_per_s': e.get('env_steps_per_s'),
            'speed/grad_steps_per_s': e.get('grad_steps_per_s'),
        }
        q = e.get('q_vs_mc') or {}
        if q.get('mc_mean'):
            scalars['critic/q_min_over_mc'] = q['q_min_mean'] / q['mc_mean']
        for k, v in (e.get('train_causes_total') or {}).items():
            scalars[f'failures_cumulative/{k}'] = v
        for tag, value in scalars.items():
            if value is not None:
                writer.add_scalar(tag, float(value), step, walltime=wall)
        last = step
    if writer:
        writer.close()
        marker.write_text(str(last))


if __name__ == '__main__':
    main(sys.argv[1])
