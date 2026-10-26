"""Per-seed progress summary of a D2 run (read-only on the run folder).

    .venv/bin/python scripts/seed_summary.py runs/D2/full1 [--out docs/results/D2_full1]

Writes <out>/SEEDS.md (one table per seed: every evaluation, plus status, BC and phase timings)
and <out>/lap2_vs_steps.png. Safe to run while training is in progress.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGET_S, QSS_S = 140.6, 133.9


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # line being written right now
            pass
    return rows


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def fmt(x, nd=2):
    return '—' if x is None else f'{x:.{nd}f}'


def seed_section(seed_dir):
    status = read_json(seed_dir / 'status.json')
    bc = read_json(seed_dir / 'bc' / 'report.json')
    evals = read_jsonl(seed_dir / 'evals.jsonl')
    out = [f'## {seed_dir.name}', '']
    out.append(f"- **State:** phase `{status.get('phase', '?')}`, state `{status.get('state', '?')}`, "
               f"rollbacks {status.get('rollbacks', 0)}, updated {status.get('updated_utc', '?')} UTC")
    gate = bc.get('gate') or {}
    if bc:
        out.append(f"- **BC (instructive init):** epochs {bc.get('epochs_run', '?')}, best val NLL "
                   f"{fmt(bc.get('best_val_nll'), 3)}, gate lap 2 {fmt(gate.get('lap2_s'))} s "
                   f"({'PASS, clean' if gate.get('det_clean') else 'FAIL: ' + str(gate.get('det_termination'))})")
    for ev in status.get('events', []):
        out.append(f"- event `{ev.get('kind')}` at step {ev.get('step')} ({ev.get('utc', '')[:19]})")
    clean = [e for e in evals if e.get('det_clean') and e.get('lap2_s')]
    if clean:
        best = min(clean, key=lambda e: e['lap2_s'])
        out.append(f"- **Best clean deterministic lap 2:** {best['lap2_s']:.2f} s at step {best['step']:,} "
                   f"(target ≤ {TARGET_S} s, QSS {QSS_S} s)")
    if evals:
        last = evals[-1]
        out.append(f"- **Latest eval:** step {last['step']:,}; deterministic "
                   f"{'clean ' + fmt(last.get('lap2_s')) + ' s' if last.get('det_clean') else last.get('det_termination')}; "
                   f"stochastic clean {last.get('sto_clean')}/{last.get('sto_n')}; "
                   f"train wall {fmt((last.get('wall_train_s') or 0) / 3600)} h; "
                   f"{fmt(last.get('env_steps_per_s'), 0)} env steps/s")
    out += ['', '| step | det. result | lap 1 [s] | lap 2 [s] | stoch. clean | stoch. mean lap [s] '
                '| α | Q_min / MC | failure causes (training, cumulative) |',
            '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for e in evals:
        q = e.get('q_vs_mc') or {}
        ratio = (q.get('q_min_mean') / q['mc_mean']) if q.get('mc_mean') else None
        det = 'clean' if e.get('det_clean') else (e.get('det_termination') or e.get('det_truncation'))
        causes = ', '.join(f'{k} {v}' for k, v in (e.get('train_causes_total') or {}).items())
        out.append(f"| {e['step']:,} | {det} | {fmt(e.get('lap1_s'))} | {fmt(e.get('lap2_s'))} | "
                   f"{e.get('sto_clean')}/{e.get('sto_n')} | {fmt(e.get('sto_lap_mean_s'))} | "
                   f"{fmt(e.get('alpha'), 3)} | {fmt(ratio)} | {causes} |")
    return out, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(p for p in args.run_dir.glob('seed*') if p.is_dir())
    lines = [f'# D2 run `{args.run_dir.name}` — per-seed progress', '',
             f'Generated {datetime.now():%Y-%m-%d %H:%M} (local) by `scripts/seed_summary.py`. '
             f'Done criterion (DECISIONS #12): clean deterministic flying lap 2 ≤ {TARGET_S} s on all 3 seeds, '
             'plateau < 0.2 s over the last 20 %, seed std ≤ 0.2 s, ≥ 9/10 stochastic laps clean. '
             'Evaluations every 50k steps; "det." = deterministic policy, flying lap 2 after a run-in lap.', '']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed_dir in seeds:
        section, evals = seed_section(seed_dir)
        lines += section + ['']
        pts = [(e['step'], e['lap2_s']) for e in evals if e.get('det_clean') and e.get('lap2_s')]
        if pts:
            ax.plot(*zip(*pts), 'o-', label=seed_dir.name)
    ax.axhline(TARGET_S, ls='--', c='tab:red', lw=1, label=f'target {TARGET_S} s')
    ax.axhline(QSS_S, ls=':', c='gray', lw=1, label=f'QSS {QSS_S} s')
    ax.set_xlabel('env steps'); ax.set_ylabel('clean deterministic lap 2 [s]')
    ax.set_title(f'{args.run_dir.name}: lap time vs training (clean laps only)')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out_dir / 'lap2_vs_steps.png', dpi=130)
    lines.append('![lap 2 vs steps](lap2_vs_steps.png)')
    (out_dir / 'SEEDS.md').write_text('\n'.join(lines) + '\n')
    print(out_dir / 'SEEDS.md')


if __name__ == '__main__':
    main()
