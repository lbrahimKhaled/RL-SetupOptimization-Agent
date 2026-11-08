"""Mirror a seed's evals.jsonl (lap times, clean rates, Q vs MC, alpha, throughput, lap-2 line / cost
metrics, critic check; R2: corner bonus, per-corner exit ratios, focus, low-noise and buffer
statistics; R4: entry / exit ratios and terms, brake onset, apex v_min, closeness g and pedal use per
straight, Phi_v / Phi_e shaping sums, front locking, training-episode R4 terms) into TensorBoard.

    .venv/bin/python scripts/evals_to_tb.py runs/D2/full1/seed0

Appends to <seed_dir>/tb_evals/: each call writes only evaluations newer than the last one
mirrored (kept in tb_evals/last_step.txt) into a new event file in the same folder. TensorBoard
merges the files of a folder; deleting and rewriting them instead leaves a running TensorBoard
on the stale, deleted file. Point TensorBoard at the seed folder to see these next to SB3's own
losses in sb3/.
A warm-started run (status.json 'lifetime_steps_offset' = the source's own offset + the source
model's steps, chained over warm starts; R3) also gets <seed_dir>/tb_evals_lifetime/ with the same
scalars at step + offset, so it overlays its source runs' curves on one step axis (R2-9.6):
    tensorboard --logdir_spec f1_r3:runs/mac/D2/f1_r3/seed0/tb_evals_lifetime,f1_r2:runs/mac/D2/f1_r2/seed0/tb_evals_lifetime,f1_best_w35:runs/mac/D2/f1_best_w35/seed0/tb_evals
    (R4: add f1_r4:runs/mac/D2/f1_r4/seed0/tb_evals_lifetime)
"""
import calendar
import json
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def scalars_of(e):
    """{tag: value or None} of one evaluation record."""
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
    lap2 = e.get('lap2') or {}          # metrics of the deterministic lap 2 (None if not reached)
    scalars.update({
        'line/mean_abs_e_m': lap2.get('mean_abs_e'),
        'line/max_abs_e_m': lap2.get('max_abs_e'),
        'line/share_abs_e_gt_3m': lap2.get('share_abs_e_gt_3m'),
        'lap/min_edge_margin_m': lap2.get('min_edge_margin_m'),
        'cost/lap2_legacy': lap2.get('cost'),                     # time + smoothness + tracking
        'cost/lap2_tracking': lap2.get('tracking'),
        'cost/lap2_smoothness': lap2.get('smoothness'),
        'cost/lap2_r1': lap2.get('cost_r1', lap2.get('time')),   # = lap time (pre-R1 records: 'time')
        'diag/progress_jumps': lap2.get('progress_jumps'),
        'critic/critic_check': e.get('critic_check'),
    })
    q = e.get('q_vs_mc') or {}
    if q.get('mc_mean'):
        scalars['critic/q_min_over_mc'] = q['q_min_mean'] / q['mc_mean']
    if q.get('mc_nstep_mean'):
        scalars['critic/q_min_over_mc_nstep'] = q['q_min_mean'] / q['mc_nstep_mean']
        scalars['critic/q_min_minus_mc_nstep'] = q.get('q_min_minus_mc_nstep')
    for k, v in (e.get('train_causes_total') or {}).items():
        scalars[f'failures_cumulative/{k}'] = v
    # --- R2 (docs/specs/R2_review.md R2-9.6); absent keys -> nothing written ---
    if 'bonus_w' in e:
        scalars.update({'bonus/w': e.get('bonus_w'), 'bonus/reportable': float(bool(e.get('reportable'))),
                        'bonus/lap2_sum': e.get('lap2_bonus_sum'), 'bonus/det_sum': e.get('det_bonus_sum')})
    for unit, c in (e.get('lap2_corners') or {}).items():
        scalars[f'corner/rho_{unit}'] = c.get('rho')
    for unit, c in (e.get('sto_corners') or {}).items():
        scalars[f'corner/sto_crash_{unit}'] = c.get('crashes')
        scalars[f'corner/sto_pass_{unit}'] = c.get('passes')
    f = e.get('focus') or {}
    for k in ('share_resets', 'share_transitions', 'fallback_rate', 'handover_mean_m', 'handover_median_m'):
        if k in f:
            scalars[f'focus/{k}'] = f[k]
    for i, (b, p) in enumerate(f.get('top5') or []):
        scalars[f'focus/top{i + 1}_P'] = p
        scalars[f'focus/top{i + 1}_bin_m'] = b
    for b, r in (f.get('r_hot') or {}).items():
        scalars[f'focus/r_hot_{b}'] = r
    groups = e.get('train_groups') or {}
    for g, v in groups.items():
        prefix = 'lownoise' if g == 'low_noise' else f'groups_{g}'
        scalars[f'{prefix}/failure_rate'] = v.get('failure_rate')
        scalars[f'{prefix}/episodes'] = v.get('episodes')
        scalars[f'{prefix}/bonus_per_episode'] = v.get('bonus_per_episode')
    if 'buffer_labelled_fraction' in e:
        scalars['buffer/labelled_fraction'] = e['buffer_labelled_fraction']
    # --- R4 (docs/specs/R4_review.md req 6); absent keys -> nothing written ---
    r4 = e.get('lap2_r4') or {}
    for k in ('entry_sum', 'exit_sum', 'taken_back', 'terms_sum', 'mean_g', 'shaping_v_sum', 'shaping_e_sum',
              'front_lock_fraction'):
        if k in r4:
            scalars[f'r4/lap2_{k}'] = r4[k]
    for name, st in (r4.get('straights') or {}).items():
        for k in ('mean_g', 'full_throttle_share', 'brake_share', 'top_speed'):
            scalars[f'straight/{k}_{name}'] = st.get(k)
    for unit, c in (e.get('lap2_entry_exit') or {}).items():
        for k in ('rho_in', 'g_in', 'rho_out', 'g_out', 'taken_back', 'brake_onset_m', 'v_min_m_s'):
            if k in c:
                scalars[f'corner/{k}_{unit}'] = c[k]
    tr = e.get('train_r4') or {}
    for k in ('ep_shaping_v_mean', 'ep_corner_terms_mean', 'ep_taken_back_mean', 'takebacks', 'episodes_with_takeback'):
        if k in tr:
            scalars[f'train_r4/{k}'] = tr[k]
    return scalars


def mirror(seed_dir, out_name='tb_evals', offset=0):
    """Write the evaluations newer than <out>/last_step.txt to a new event file in
    <seed_dir>/<out_name> at step + offset. Returns the number of evaluations written."""
    seed_dir = Path(seed_dir)
    out = seed_dir / out_name
    out.mkdir(exist_ok=True)
    marker = out / 'last_step.txt'
    last = int(marker.read_text()) if marker.exists() else -1
    writer, written = None, 0
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
        for tag, value in scalars_of(e).items():
            if value is not None:
                writer.add_scalar(tag, float(value), step + offset, walltime=wall)
        last = step
        written += 1
    if writer:
        writer.close()
        marker.write_text(str(last))
    return written


def main(seed_dir):
    seed_dir = Path(seed_dir)
    mirror(seed_dir)
    status = seed_dir / 'status.json'
    offset = json.loads(status.read_text()).get('lifetime_steps_offset') if status.exists() else None
    if offset is not None:
        mirror(seed_dir, 'tb_evals_lifetime', int(offset))


if __name__ == '__main__':
    main(sys.argv[1])
