#!/usr/bin/env python3
"""D2 done-criterion report (docs/specs/D2_agent.md §8, DECISIONS #12).

    .venv/bin/python scripts/d2_report.py runs/D2/<run_name> [--no-reeval] [--strict]

1. All seed folders: identical config (except seed), rlsetup commit, ocd_lap code version.
2. Per seed, re-evaluate checkpoints/last.zip with the eval protocol (1 torch thread):
   deterministic 2-lap run + 10 stochastic 1-lap runs (torch seeds 0-9; clean = timed lap).
3. Re-eval lap 2 - last logged lap 2 (expect 0; nonzero = reproducibility flag).
4. Criteria per seed and overall: (i) clean deterministic flying lap; (ii) lap 2 <= 140.6 s;
   (iii) plateau: lap2(eval nearest 0.8 T) - min lap2 over evals in (0.8 T, T] < 0.2 s (plus the
   lap-2 range over the last 20 % as an oscillation diagnostic); (iv) sample std (ddof = 1) of the
   final lap 2 over >= 3 seeds <= 0.2 s; (v) >= 9/10 stochastic laps clean; (vi) env steps + wall
   time per phase logged (status.json 'phases' incl. 'rl').
5. Writes d2_report.json + d2_report.md in the run folder; prints PASS / FAIL per criterion.
--no-reeval uses the last logged evaluation instead of re-evaluating (criteria (i), (ii), (v)
then come from evals.jsonl). Exit code 0 unless --strict and a criterion fails.
A seed that cannot be re-evaluated (no checkpoints/last.zip, e.g. stopped at the demo / BC gate)
or that stopped at a gate is reported as FAIL with its status.json state and stop_reason
('fail_reason'); the other seeds are still re-evaluated and reported.
"""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

LAP2_MAX_S = 140.6          # 1.05 x 133.9 s QSS (DECISIONS #12)
PLATEAU_S = .2
SEED_STD_S = .2
MIN_SEEDS = 3
STO_LAPS, STO_MIN_CLEAN = 10, 9
CRITERIA = ('consistency', 'i_clean', 'ii_lap2', 'iii_plateau', 'iv_seed_std', 'v_stochastic', 'vi_logged')
GATE_STATES = ('demo_gate_failed', 'bc_gate_failed')   # train.py: stopped before RL


def _read_json(p):
    return json.loads(Path(p).read_text())


def _read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def reevaluate(seed_dir, n_sto=STO_LAPS):
    """Re-evaluate checkpoints/last.zip (same protocol as DrivingEvalCallback)."""
    import torch as th
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    from rlsetup.driving.agent import StagedSAC
    from rlsetup.driving.evaluate import flying_lap
    from rlsetup.driving.train import eval_envs
    from rlsetup.driving.train_config import train_config_from_dict
    cfg = train_config_from_dict(_read_json(seed_dir / 'config.json'))
    th.set_num_threads(1)
    track = Track(cfg.env.track)
    envs = eval_envs(cfg, track, qss_profile.for_env(track, cfg.env))
    model = StagedSAC.load(seed_dir / 'checkpoints' / 'last.zip', device='cpu')
    det = flying_lap(model, envs['eval_det'], deterministic=True, seed=cfg.eval.seed)
    sto = [flying_lap(model, envs['eval_sto'], deterministic=False, seed=k) for k in range(n_sto)]
    return {'det_clean': det.clean, 'lap2_s': det.lap2_s, 'det_termination': det.termination,
            'det_return': det.episode_return,
            'sto_clean': sum(s.clean for s in sto), 'sto_n': len(sto),
            'sto_lap_s': [s.lap1_s for s in sto]}


def plateau(evals):
    """(improvement [s] or None, lap-2 range over the last 20 % or None, note)."""
    if not evals:
        return None, None, 'no evaluations'
    T = evals[-1]['step']
    ref = min(evals, key=lambda r: abs(r['step'] - .8 * T))
    if ref is evals[-1]:
        return None, None, 'not enough evaluations (the eval nearest 0.8 T is the last one)'
    late = [r for r in evals if .8 * T < r['step'] <= T]
    late_laps = [r['lap2_s'] for r in late if r.get('det_clean') and r.get('lap2_s') is not None]
    rng = (max(late_laps) - min(late_laps)) if late_laps else None
    if not (ref.get('det_clean') and ref.get('lap2_s') is not None):
        return None, rng, f'eval at step {ref["step"]} (nearest 0.8 T) has no clean lap 2'
    if not late_laps:
        return None, rng, 'no clean evaluation in (0.8 T, T]'
    return ref['lap2_s'] - min(late_laps), rng, f'reference step {ref["step"]}, T = {T}'


def fail_reason(seed_dir, status, reeval):
    """Why a seed has no final policy to judge (None if it has one): stopped at a gate, or (with
    re-evaluation) no checkpoints/last.zip. Includes status.json state and stop_reason."""
    state, why = status.get('state'), status.get('stop_reason')
    if state in GATE_STATES:
        return f'state {state}: {why}'
    if reeval and not (seed_dir / 'checkpoints' / 'last.zip').exists():
        return f'no checkpoints/last.zip (state {state}: {why})'
    return None


def seed_report(seed_dir, reeval):
    evals = sorted(_read_jsonl(seed_dir / 'evals.jsonl'), key=lambda r: r['step'])
    sp = seed_dir / 'status.json'
    status = _read_json(sp) if sp.exists() else {}
    last = evals[-1] if evals else {}
    out = {'seed_dir': str(seed_dir), 'n_evals': len(evals), 'final_step': last.get('step'),
           'logged_lap2_s': last.get('lap2_s'), 'logged_det_clean': last.get('det_clean'),
           'state': status.get('state'), 'stop_reason': status.get('stop_reason'),
           'fail_reason': fail_reason(seed_dir, status, reeval)}
    if out['fail_reason'] is not None:
        out['reeval'] = None
        det_clean, lap2, sto_clean, sto_n = False, None, 0, 0
    elif reeval:
        r = reevaluate(seed_dir)
        out['reeval'] = r
        out['reeval_delta_s'] = (r['lap2_s'] - last['lap2_s']) if r['lap2_s'] is not None and \
            last.get('lap2_s') is not None else None
        # the deterministic return compares even when no lap 2 was completed
        out['reeval_return_delta'] = r['det_return'] - last['det_return'] if 'det_return' in last else None
        out['reproducibility_flag'] = any(d is not None and d != 0 for d in
                                          (out['reeval_delta_s'], out['reeval_return_delta']))
        det_clean, lap2, sto_clean, sto_n = r['det_clean'], r['lap2_s'], r['sto_clean'], r['sto_n']
    else:
        det_clean, lap2 = bool(last.get('det_clean')), last.get('lap2_s')
        sto_clean, sto_n = last.get('sto_clean', 0), last.get('sto_n', 0)
    imp, rng, note = plateau(evals)
    phases = status.get('phases') or {}
    logged = 'rl' in phases and all(isinstance(p, dict) and 'wall_s' in p and 'env_steps' in p
                                    for p in phases.values())
    out.update(final_lap2_s=lap2 if det_clean else None, det_clean=det_clean,
               plateau_improvement_s=imp, last20_lap2_range_s=rng, plateau_note=note,
               sto_clean=sto_clean, sto_n=sto_n, phases=phases)
    out['criteria'] = {
        'i_clean': bool(det_clean),
        'ii_lap2': bool(det_clean and lap2 is not None and lap2 <= LAP2_MAX_S),
        'iii_plateau': imp is not None and imp < PLATEAU_S,
        'v_stochastic': sto_n >= STO_LAPS and sto_clean >= STO_MIN_CLEAN * sto_n / STO_LAPS,
        'vi_logged': logged}
    return out


def consistency(seed_dirs):
    cfgs, commits, codes, problems = [], set(), set(), []
    for d in seed_dirs:
        c = _read_json(d / 'config.json')
        c.pop('seed', None)
        cfgs.append(c)
        if not (d / 'provenance.json').exists():
            problems.append(f'{d.name}: no provenance.json')
            continue
        p = _read_json(d / 'provenance.json')
        commits.add(p['rlsetup']['commit'])
        codes.add(p['ocd_lap_code_version'])
    if any(c != cfgs[0] for c in cfgs[1:]):
        problems.append('configs differ (beyond the seed)')
    if len(commits) > 1:
        problems.append(f'rlsetup commits differ: {sorted(commits)}')
    if len(codes) > 1:
        problems.append(f'ocd_lap code versions differ: {sorted(codes)}')
    return not problems, problems


def _std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def build_report(run_dir, *, reeval=True):
    run_dir = Path(run_dir)
    seed_dirs = sorted(p for p in run_dir.glob('seed*') if (p / 'config.json').exists())
    if not seed_dirs:
        raise FileNotFoundError(f'no seed folders with config.json in {run_dir}')
    ok, problems = consistency(seed_dirs)
    seeds = [seed_report(d, reeval) for d in seed_dirs]
    finals = [s['final_lap2_s'] for s in seeds]
    std = _std(finals) if len(finals) >= 2 and all(f is not None for f in finals) else None
    overall = {'consistency': ok,
               'iv_seed_std': len(seeds) >= MIN_SEEDS and std is not None and std <= SEED_STD_S}
    for k in ('i_clean', 'ii_lap2', 'iii_plateau', 'v_stochastic', 'vi_logged'):
        overall[k] = all(s['criteria'][k] for s in seeds)
    return {'run_dir': str(run_dir), 'reevaluated': reeval, 'n_seeds': len(seeds),
            'consistency_problems': problems, 'seed_lap2_std_s': std, 'seeds': seeds,
            'criteria': overall, 'pass': all(overall.values()),
            'thresholds': {'lap2_max_s': LAP2_MAX_S, 'plateau_s': PLATEAU_S, 'seed_std_s': SEED_STD_S,
                           'min_seeds': MIN_SEEDS, 'stochastic': f'{STO_MIN_CLEAN}/{STO_LAPS}'}}


def _fmt(x, nd=3):
    return '-' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def markdown(rep):
    pf = lambda b: 'PASS' if b else 'FAIL'
    lines = [f'# D2 report: {rep["run_dir"]}', '',
             f'Seeds: {rep["n_seeds"]}; re-evaluated: {rep["reevaluated"]}; seed std of final lap 2: '
             f'{_fmt(rep["seed_lap2_std_s"])} s; overall: **{pf(rep["pass"])}**', '',
             '| seed | final lap 2 [s] | (i) clean | (ii) <= 140.6 s | (iii) plateau impr. [s] | last-20% range [s] '
             '| (v) stochastic clean | (vi) logged | re-eval delta [s] |',
             '|---|---|---|---|---|---|---|---|---|']
    for s in rep['seeds']:
        c = s['criteria']
        lines.append(f'| {Path(s["seed_dir"]).name} | {_fmt(s["final_lap2_s"])} | {pf(c["i_clean"])} | '
                     f'{pf(c["ii_lap2"])} | {_fmt(s["plateau_improvement_s"])} {pf(c["iii_plateau"])} | '
                     f'{_fmt(s["last20_lap2_range_s"])} | {s["sto_clean"]}/{s["sto_n"]} {pf(c["v_stochastic"])} | '
                     f'{pf(c["vi_logged"])} | {_fmt(s.get("reeval_delta_s"))} |')
    failed = [(Path(s['seed_dir']).name, s['fail_reason']) for s in rep['seeds'] if s.get('fail_reason')]
    if failed:
        lines += ['', 'Seeds without a final policy (all their criteria FAIL):']
        lines += [f'- {name}: {why}' for name, why in failed]
    lines += ['', '| criterion | result |', '|---|---|']
    lines += [f'| {k} | {pf(rep["criteria"][k])} |' for k in CRITERIA]
    if rep['consistency_problems']:
        lines += ['', 'Consistency problems: ' + '; '.join(rep['consistency_problems'])]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--no-reeval', action='store_true')
    p.add_argument('--strict', action='store_true', help='exit 1 if a criterion fails')
    args = p.parse_args(argv)
    rep = build_report(args.run_dir, reeval=not args.no_reeval)
    run_dir = Path(args.run_dir)
    (run_dir / 'd2_report.json').write_text(json.dumps(rep, indent=1, default=str))
    (run_dir / 'd2_report.md').write_text(markdown(rep))
    for s in rep['seeds']:
        if s.get('fail_reason'):
            print(f'{Path(s["seed_dir"]).name}: FAIL ({s["fail_reason"]})')
    for k in CRITERIA:
        print(f'{k:14s} {"PASS" if rep["criteria"][k] else "FAIL"}')
    print(f'overall        {"PASS" if rep["pass"] else "FAIL"}   ({run_dir / "d2_report.md"})')
    return 1 if args.strict and not rep['pass'] else 0


if __name__ == '__main__':
    sys.exit(main())
