"""Mirror a seed's evals.jsonl (lap times, clean rates, Q vs MC, alpha, throughput) into TensorBoard.

    .venv/bin/python scripts/evals_to_tb.py runs/D2/full1/seed0

Writes <seed_dir>/tb_evals/ (rebuilt from scratch on every call, so it is safe to rerun while
training). Point TensorBoard at the seed folder to see these next to SB3's own losses in sb3/.
"""
import json
import shutil
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main(seed_dir):
    seed_dir = Path(seed_dir)
    out = seed_dir / 'tb_evals'
    shutil.rmtree(out, ignore_errors=True)
    writer = SummaryWriter(str(out))
    for line in (seed_dir / 'evals.jsonl').read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:   # line being written right now
            continue
        step = e['step']
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
                writer.add_scalar(tag, float(value), step)
    writer.close()


if __name__ == '__main__':
    main(sys.argv[1])
