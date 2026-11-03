"""Check that the scripted drivers stay on track along a preset's xi_ref (drive_line).

1. Settle check: the env's start placement + settle phase (demo driver, settle_s) at every
   `--step` m of Track.path, for each start-speed fraction f (v0 = f * min v_qss over the settle
   window, as DrivingEnv._sample_start), plus the evaluation start. A start that leaves the track
   or ends non-finite is a failure (DrivingEnv.reset raises for a fixed start).
2. Demonstrations (--demos): `demos.ensure_demos` with the preset's exact settings, i.e. the
   dataset the training run will load from the cache (same key), and its clean rate / gate.

    PYTHONPATH=<ocd_first_lap-f1> .venv/bin/python scripts/check_line_starts.py --preset f1_sha_w35 --demos

Writes JSON to --out (default: print only).
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocd_lap.track import Track                                          # noqa: E402
from rlsetup.driving import qss_profile                                  # noqa: E402
from rlsetup.driving.env import DrivingEnv                               # noqa: E402
from rlsetup.driving.train_config import PRESETS                         # noqa: E402


def settle_check(env, step_m, fractions):
    """[{s0, f, v0, ok, e_ref, s_ref, error}] for starts every step_m on Track.path."""
    cfg, L = env.cfg, env.track.length
    rows = []
    for s0 in np.arange(0., L, step_m):
        for f in fractions:
            window = f * float(env.qss.speed(s0)) * cfg.settle_s + cfg.settle_window_margin_m
            v0 = f * env.qss.min_speed(s0, window)
            row = {'s0': float(s0), 'f': f, 'v0': float(v0)}
            try:
                _, info = env.reset(seed=0, options={'start_s': float(s0), 'start_speed': float(v0)})
                row.update(ok=True, e_ref=float(info['e_ref']), s_ref=float(info['s_ref']),
                           track_position=env.track_position())
            except RuntimeError as e:
                row.update(ok=False, error=str(e))
            rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--preset', default='f1_sha_w35')
    ap.add_argument('--step', type=float, default=100., help='start spacing on Track.path [m]')
    ap.add_argument('--fractions', type=float, nargs='+', default=[.3, .5, .7])
    ap.add_argument('--demos', action='store_true', help='also collect / load the preset demo dataset')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    cfg = PRESETS[a.preset]()
    track = Track(cfg.env.track)
    qss = qss_profile.for_env(track, cfg.env)
    env = DrivingEnv(cfg.env, track=track, qss=qss)
    out = {'preset': a.preset, 'reference': cfg.env.reference, 'reference_digest': env.ref.digest,
           'reference_length_m': env.ref.length, 'drive_line_is_reference': env.drive_line is env.ref}
    t = time.time()
    rows = settle_check(env, a.step, a.fractions)
    bad = [r for r in rows if not r['ok']]
    ok = [r for r in rows if r['ok']]
    out['settle'] = {'starts': len(rows), 'failures': len(bad), 'failed': bad,
                     'max_abs_e_ref_at_handover_m': max(abs(r['e_ref']) for r in ok) if ok else None,
                     'max_abs_track_position': max(abs(r['track_position']) for r in ok) if ok else None,
                     'wall_s': time.time() - t}
    _, info = DrivingEnv(replace(cfg.env, evaluation=True), track=track, qss=qss).reset(seed=0)
    out['eval_start'] = {'s0': info['start']['s0'], 'v0': info['start']['v0'], 'e_ref': info['e_ref']}
    print(f"settle: {len(rows) - len(bad)}/{len(rows)} starts on track (every {a.step:g} m, f {a.fractions}); "
          f"max |e_ref| at hand-over {out['settle']['max_abs_e_ref_at_handover_m']:.3f} m; "
          f"max |track position| {out['settle']['max_abs_track_position']:.3f}; eval start e_ref {info['e_ref']:.3f} m")
    for r in bad:
        print(f"  FAIL s0 {r['s0']:.0f} m f {r['f']}: {r['error']}")
    if a.demos:
        from rlsetup.driving.demos import ensure_demos, DemoGateFailed
        from rlsetup.driving.vec import track_sha256
        t = time.time()
        try:
            path, meta, used = ensure_demos(cfg.env, cfg.demos, qss, track_sha256(track), cfg.demo_dir,
                                            start_method=cfg.vec_start_method)
            out['demos'] = {'path': str(path), 'summary': meta['summary'], 'cached': bool(meta.get('cached')),
                            'noise_std': list(used.noise_std), 'fallback_from': meta.get('fallback_from'),
                            'wall_s': time.time() - t}
            s = meta['summary']
            print(f"demos: {s['episodes']} episodes, clean {s['clean_fraction']:.3f} (gate "
                  f"{cfg.demos.min_clean_fraction}), causes {s['causes']}, mean speed fraction "
                  f"{s['mean_speed_fraction']:.3f}, noise {list(used.noise_std)}")
        except DemoGateFailed as e:
            out['demos'] = {'gate_failed': str(e)}
            print(f'demos: {e}')
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0 if not bad and 'gate_failed' not in out.get('demos', {}) else 1


if __name__ == '__main__':
    sys.exit(main())
