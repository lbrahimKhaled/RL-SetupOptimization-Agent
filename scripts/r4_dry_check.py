"""R4 dry check (docs/specs/R4_review.md req 4-5): build `--preset f1_r4` as train.py would, load the
init replay buffer in memory (adopt + speed relabel; nothing is written back), print the relabel
statistics (chains / anchors, unresolved fraction, F_v, entry / exit terms), the pre-entry / pre-exit
probe row counts and the lifetime step offset. Needs the sf70h_f1 simulator branch on PYTHONPATH and
~2 x the buffer size of free memory.

    PYTHONPATH=<ocd_first_lap-f1> .venv/bin/python scripts/r4_dry_check.py [--set key=value ...]
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


def main(argv):
    from ocd_lap.track import Track
    from stable_baselines3.common.save_util import load_from_pkl
    from rlsetup.driving import qss_profile
    from rlsetup.driving.agent import default_init_buffer
    from rlsetup.driving.buffers import BonusNStepReplayBuffer, pre_point_cells
    from rlsetup.driving.config import EnvConfig
    from rlsetup.driving.relabel import relabel_speed_profile
    from rlsetup.driving.train import parse_args, corner_table
    from rlsetup.driving.train_config import PRESETS, apply_overrides, config_from_dict
    from rlsetup.driving.reference import ReferenceLine, track_grid
    args = parse_args(['--preset', 'f1_r4', '--run-name', 'dry', *argv])
    cfg = replace(apply_overrides(PRESETS[args.preset](), args.overrides), seed=args.seed)
    buf_path = cfg.warm.init_buffer or default_init_buffer(cfg.init_from)
    src_dir = Path(cfg.init_from).resolve().parents[1]
    src_env = config_from_dict(EnvConfig, json.loads((src_dir / 'config.json').read_text())['env'])
    status = json.loads((src_dir / 'status.json').read_text()) if (src_dir / 'status.json').exists() else {}
    import zipfile
    with zipfile.ZipFile(cfg.init_from) as z:
        n_src = json.loads(z.read('data'))['num_timesteps']
    off = int(status.get('lifetime_steps_offset') or 0)
    print(f'preset f1_r4: init_from {cfg.init_from}\n  init_buffer {buf_path}\n  lifetime offset {off} + {n_src} = {off + n_src}')
    track = Track(cfg.env.track)
    qss = qss_profile.for_env(track, cfg.env)
    table = corner_table(cfg, track, qss)
    grid = track_grid(track, ReferenceLine.for_track(track, cfg.env.reference), cfg.env.obs.grid_m)
    print(f"corner table {table.meta.get('key')[:12]}")
    st = Path(buf_path).stat()
    t0 = time.time()
    rb = load_from_pkl(buf_path)
    print(f'loaded {type(rb).__name__} in {time.time() - t0:.1f} s: full {rb.full}, pos {rb.pos}, n_envs {rb.n_envs}')
    BonusNStepReplayBuffer.adopt(rb)
    t0 = time.time()
    stats, s = relabel_speed_profile(rb, src_env, cfg.env, table, grid, max_unresolved=cfg.warm.speed_relabel_max_unresolved)
    print(f'relabel in {time.time() - t0:.1f} s')
    print(json.dumps(stats, indent=1))
    L, before = float(table.length), cfg.warm.probe_before_m
    for name, pts in (('pre_entry', [u['start_m'] for u in table.units]), ('pre_exit', [u['exit_m'] for u in table.units])):
        print(f'{name} probe rows (<= {before:g} m): {len(pre_point_cells(s, pts, L, before)[0])}')
    st2 = Path(buf_path).stat()
    print(f'source file unchanged: {(st.st_size, st.st_mtime) == (st2.st_size, st2.st_mtime)}')


if __name__ == '__main__':
    main(sys.argv[1:])
