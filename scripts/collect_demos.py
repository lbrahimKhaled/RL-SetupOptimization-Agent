#!/usr/bin/env python3
"""Collect (or reuse) the D2 demonstration dataset for a training preset, with the gate
(spec §4 step 2): data/demos/demos_<key12>.npz + .json. One dataset is shared by all seeds.

    .venv/bin/python scripts/collect_demos.py --preset full [--set demos.workers=4 ...]
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


def main(argv=None):
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from rlsetup.driving.demos import ensure_demos, DemoGateFailed
    from rlsetup.driving.train_config import PRESETS, apply_overrides
    from rlsetup.driving.vec import track_sha256
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', default='full', choices=sorted(PRESETS))
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE')
    args = p.parse_args(argv)
    cfg = apply_overrides(PRESETS[args.preset](), args.overrides)
    track = Track(cfg.env.track)
    qss = qss_profile.for_env(track, cfg.env)
    try:
        path, meta, used = ensure_demos(cfg.env, cfg.demos, qss, track_sha256(track), cfg.demo_dir,
                                        start_method=cfg.vec_start_method)
    except DemoGateFailed as e:
        print(f'FAIL: {e}')
        return 1
    print(json.dumps({'path': str(path), 'sha256': meta['npz_sha256'], 'noise_std': list(used.noise_std),
                      'summary': meta['summary']}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
