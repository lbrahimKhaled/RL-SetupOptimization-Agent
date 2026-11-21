"""R6 pre-launch checks (docs/specs/R6_throttle_remap.md §3), no training:
  (a) the new per-straight metrics (full_throttle_share_99, time_s, qss_time_s, time_lost_s) on R5's stored 7M
      deterministic lap (laps/step_7000000_det.npz) under the old map: the R6 baseline;
  (a') the same full evaluation (deterministic 2 laps + 10 stochastic laps) of the 7M actor under the old map,
       re-run with this code (comparable with (b) metric for metric);
  (b) the unchanged 7M actor under the R6 map (preset bcn_f1_r6's env).
Car sf70h_f1: run with OCD_FIRST_LAP=<ocd_first_lap-f1> PYTHONPATH=<ocd_first_lap-f1>:$PWD.

    .venv/bin/python scripts/r6_prelaunch.py [--out runs/mac/r6_prelaunch.json] [--skip-old-eval]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def straights_summary(lap):
    st = lap['straights']
    keys = ('full_throttle_share_99', 'full_throttle_share', 'brake_share', 'top_speed', 'time_s', 'qss_time_s',
            'time_lost_s')
    return {name: {k: v[k] for k in keys} for name, v in st.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(REPO / 'runs' / 'mac' / 'r6_prelaunch.json'))
    p.add_argument('--skip-old-eval', action='store_true')
    args = p.parse_args(argv)
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from rlsetup.driving.agent import StagedSAC
    from rlsetup.driving.evaluate import LapReport, TRACE_KEYS, lap_r4, evaluate_model
    from rlsetup.driving.train import corner_table, eval_envs
    from rlsetup.driving.train_config import PRESETS
    r5, r6 = PRESETS['bcn_f1_r5'](), PRESETS['bcn_f1_r6']()
    track = Track(r5.env.track)
    qss = qss_profile.for_env(track, r5.env)
    table = corner_table(r5, track, qss)
    out = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'init_from': r6.init_from,
           'throttle_remap': list(r6.env.throttle_remap), 'qss_lap_track_path_s': float(qss.lap_time_s)}
    # (a) stored 7M lap
    npz = np.load(Path(r6.init_from).parents[1] / 'laps' / 'step_7000000_det.npz')
    tr = {k: npz[k] for k in npz.files if k not in ('lap_times_s', 'clean', 'termination')}
    rep = LapReport(bool(npz['clean']), list(npz['lap_times_s']), [], str(npz['termination']), None,
                    len(tr['s']), 0., tr, None, None, [])
    _, lap = lap_r4(rep, table, r5.eval, pedal_deadzone=True, entry_exit=False)
    out['a_stored_7M_lap2_old_map'] = {
        'lap_times_s': [float(x) for x in npz['lap_times_s']], 'clean': bool(npz['clean']),
        'straights_time_s': lap['straights_time_s'], 'straights_qss_time_s': lap['straights_qss_time_s'],
        'straights_time_lost_s': lap['straights_time_lost_s'],
        'throttle_action_share_above_085': lap['throttle_action_share_above_sat'],
        'straights': straights_summary(lap)}
    print(json.dumps(out['a_stored_7M_lap2_old_map'], indent=1), flush=True)
    model = StagedSAC.load(r6.init_from, device='cpu')
    for key, cfg in ((None if args.skip_old_eval else 'a2_eval_7M_old_map', r5), ('b_eval_7M_new_map', r6)):
        if key is None:
            continue
        t0 = time.time()
        envs = eval_envs(cfg, track, qss, table)
        ev = evaluate_model(model, envs, cfg.eval)
        rec = {k: ev.get(k) for k in ('det_clean', 'det_termination', 'lap1_s', 'lap2_s', 'sto_clean', 'sto_n',
                                      'sto_clean_rate', 'sto_lap_mean_s', 'alpha', 'policy_entropy')}
        rec['sto_laps'] = ev['sto_laps']
        rec['sto_corners'] = ev.get('sto_corners')
        rec['q_vs_mc'] = {k: ev['q_vs_mc'].get(k) for k in ('n', 'mc_nstep_mean', 'q_min_mean', 'q_min_ratio_nstep')}
        r4 = ev.get('lap2_r4')
        if r4:
            rec.update(straights_time_lost_s=r4['straights_time_lost_s'], straights_time_s=r4['straights_time_s'],
                       throttle_action_share_above_sat=r4['throttle_action_share_above_sat'],
                       straights=straights_summary(r4))
        rec['wall_s'] = time.time() - t0
        out[key] = rec
        print(key, json.dumps({k: v for k, v in rec.items() if k not in ('straights', 'sto_laps', 'sto_corners')}),
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
