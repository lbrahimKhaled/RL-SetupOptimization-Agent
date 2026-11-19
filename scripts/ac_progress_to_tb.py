"""Mirror an AC learner run's progress.jsonl (one row per drive) into TensorBoard.

    .venv/bin/python scripts/ac_progress_to_tb.py runs/ac/tcp-full-002 --ocd-run runs/mac/D2/bcn/seed0 --follow

Writes to <work_dir>/tb/ (step = learner cycle, one drive) and <work_dir>/tb_lineage/ (step = the
source OCD checkpoint's num_timesteps + AC transitions so far; tags renamed <group>/ac_<name>), so the
AC curves continue the OCD run's step axis next to its own charts. Shared `compare/` tags put OCD and
AC lap times in one chart: --ocd-run (the OCD run whose checkpoints/best.zip is the AC source, checked
by SHA-256) writes that run's evaluations to <work_dir>/tb_ocd_compare/ under the same tags:
  compare/noisy_lap_time_s          OCD stochastic-eval mean lap  | AC exploration laps (0.3 x policy std)
  compare/deterministic_lap_time_s  OCD deterministic lap 2 (clean) | AC deterministic (eval) laps
    .venv/bin/python -m tensorboard.main --port 6007 --logdir_spec \
        ocd_bcn:runs/mac/D2/bcn/seed0/tb_evals,ocd_bcn_sb3:runs/mac/D2/bcn/seed0/sb3,\
ocd_bcn_compare:runs/ac/tcp-full-002/tb_ocd_compare,ac_full_002:runs/ac/tcp-full-002/tb_lineage
OCD laps are flying laps of the OCD car model; AC laps start from a standstill, so they are not equal
conditions.
Each call appends only drives newer than tb/last_cycle.txt to a new event
file (TensorBoard merges a folder's files; rewriting them would leave a running TensorBoard on a
deleted file). --follow keeps polling for new drives until Ctrl+C.

Lap time = AC time of a drive that reached lap_complete: one full reference length (4602 m) from a
standing start wherever the previous drive left the car; not an official AC flying lap.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def drive_time_s(work_dir: Path, cycle: int) -> float | None:
    chunk = work_dir / f'cycle-{cycle:08d}' / 'chunk.npz'
    try:
        with np.load(chunk, allow_pickle=False) as z:
            return float(z['dt'].sum())
    except (OSError, KeyError, ValueError):
        return None


def scalars_of(row: dict, seconds: float | None, total: int, laps: int) -> dict:
    mode = 'eval' if row.get('mode') == 'deterministic' else 'explore'
    lap = row.get('stop_reason') == 'lap_complete'
    out = {
        f'{mode}/progress_m': row.get('progress_m'),
        f'{mode}/episode_return': row.get('episode_return'),
        f'{mode}/lap_complete': float(lap),
        'all/progress_m': row.get('progress_m'),
        'all/episode_return': row.get('episode_return'),
        'all/laps_completed_total': laps,
        'train/joint_updates': row.get('updates'),
        'train/critic_only_updates': row.get('critic_updates'),
        'train/transitions_this_drive': row.get('transitions'),
        'train/transitions_total': total,
        'failure/off_track': float(str(row.get('stop_reason', '')).startswith(('wheel_off', 'wrong_way'))),
        'failure/telemetry_or_timeout': float(str(row.get('stop_reason', '')).startswith(('TimeoutError', 'ValueError'))),
        'start/teleport': float(row.get('start_mode') == 'teleport'),
        'start/retry_failed_corner': float(row.get('start_kind') == 'retry'),
        'failure/teleport_start': float(str(row.get('stop_reason', '')).startswith(('teleport', 'warmup'))),
    }
    if seconds:
        out['all/drive_time_s'] = seconds
        if row.get('progress_m') is not None and seconds > 0:
            out['all/avg_speed_kmh'] = 3.6*row['progress_m']/seconds
        if lap and row.get('start_mode') == 'teleport':
            # A full reference length from a moving drop-in somewhere on the lap: not a lap time.
            out['lap/teleport_full_length_s'] = seconds
        elif lap:
            out['lap/lap_time_s'] = seconds
            out[f'lap/{mode}_lap_time_s'] = seconds
            out['compare/deterministic_lap_time_s' if mode == 'eval' else 'compare/noisy_lap_time_s'] = seconds
    return out


def source_steps(work_dir: Path) -> int:
    """OCD env steps of the checkpoint the AC actor was exported from (learner source bundle)."""
    manifest = json.loads((work_dir / 'source' / 'manifest.json').read_text())
    return int(manifest['source']['num_timesteps'])


def lineage_tag(tag: str) -> str:
    group, name = tag.split('/', 1)
    if group == 'compare':  # shared with the OCD run's compare folder
        return tag
    return f'{group}/ac_{name}'


def write_ocd_compare(work_dir: Path, ocd_run: Path) -> int:
    """OCD evaluations under the shared compare/ tags, once (the source run is finished)."""
    out = work_dir / 'tb_ocd_compare'
    if (out / 'done.txt').exists():
        return 0
    manifest = json.loads((work_dir / 'source' / 'manifest.json').read_text())
    best = hashlib.sha256((ocd_run / 'checkpoints' / 'best.zip').read_bytes()).hexdigest()
    if best != manifest['source']['checkpoint_sha256']:
        raise SystemExit(f'{ocd_run}/checkpoints/best.zip is not the AC source checkpoint')
    writer, n = SummaryWriter(str(out)), 0
    for line in (ocd_run / 'evals.jsonl').read_text().splitlines():
        e = json.loads(line)
        wall = calendar.timegm(time.strptime(e['utc'], '%Y-%m-%dT%H:%M:%SZ')) if e.get('utc') else None
        if e.get('det_clean') and e.get('lap2_s'):
            writer.add_scalar('compare/deterministic_lap_time_s', float(e['lap2_s']), e['step'], walltime=wall)
        if e.get('sto_lap_mean_s'):
            writer.add_scalar('compare/noisy_lap_time_s', float(e['sto_lap_mean_s']), e['step'], walltime=wall)
        n += 1
    writer.close()
    (out / 'done.txt').write_text(str(n))
    return n


def mirror(work_dir: Path, state: dict, writers: dict) -> int:
    """Append drives newer than state['last'] to both folders; returns the number written."""
    progress = work_dir / 'progress.jsonl'
    if not progress.exists():
        return 0
    written = 0
    for line in progress.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # line being written right now
            continue
        cycle = int(row['cycle'])
        if cycle <= state['last']:
            continue
        state['total'] += int(row.get('transitions') or 0)
        state['laps'] += row.get('stop_reason') == 'lap_complete' and row.get('start_mode') != 'teleport'
        if not writers:
            writers['tb'] = SummaryWriter(str(work_dir / 'tb'))
            writers['tb_lineage'] = SummaryWriter(str(work_dir / 'tb_lineage'))
        lineage_step = state['source_steps'] + state['total']
        for tag, value in scalars_of(row, drive_time_s(work_dir, cycle), state['total'], state['laps']).items():
            if value is not None:
                writers['tb'].add_scalar(tag, float(value), cycle, walltime=row.get('unix_time_s'))
                writers['tb_lineage'].add_scalar(lineage_tag(tag), float(value), lineage_step,
                                                 walltime=row.get('unix_time_s'))
        state['last'] = cycle
        written += 1
    if written:
        for w in writers.values():
            w.flush()
        (work_dir / 'tb' / 'last_cycle.txt').write_text(json.dumps(
            dict(last=state['last'], total=state['total'], laps=state['laps'])))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('work_dir', type=Path, help='learner --work-dir, e.g. runs/ac/tcp-full-002')
    parser.add_argument('--follow', action='store_true', help='keep adding new drives until Ctrl+C')
    parser.add_argument('--ocd-run', type=Path, help='OCD seed dir of the source checkpoint, e.g. runs/mac/D2/bcn/seed0')
    parser.add_argument('--every-s', type=float, default=10., help='poll interval with --follow')
    args = parser.parse_args(argv)
    marker = args.work_dir / 'tb' / 'last_cycle.txt'
    state = dict(last=-1, total=0, laps=0)
    if marker.exists():
        state.update(json.loads(marker.read_text()))
    state['source_steps'] = source_steps(args.work_dir)
    if args.ocd_run:
        print(f'{write_ocd_compare(args.work_dir, args.ocd_run)} OCD evaluation(s) written', flush=True)
    writers = {}
    try:
        while True:
            n = mirror(args.work_dir, state, writers)
            if n:
                print(f'{n} drive(s) added; last cycle {state["last"]}, laps {state["laps"]}', flush=True)
            if not args.follow:
                break
            time.sleep(args.every_s)
    except KeyboardInterrupt:
        pass
    finally:
        for w in writers.values():
            w.close()


if __name__ == '__main__':
    main()
