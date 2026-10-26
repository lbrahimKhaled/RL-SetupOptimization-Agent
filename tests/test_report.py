"""D2 §10: scripts/d2_report.py on synthetic evals.jsonl for 3 seeds -> PASS, and FAIL for each
criterion separately."""
import copy
import importlib.util
import json
import pytest
from rlsetup.provenance import REPO_ROOT

spec = importlib.util.spec_from_file_location('d2_report', REPO_ROOT / 'scripts' / 'd2_report.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

T = 1_000_000
CONFIG = {'preset': 'full', 'seed': 0, 'total_timesteps': T, 'agent': {'batch_size': 512}}
PHASES = {'demos': {'wall_s': 100., 'env_steps': 200_000}, 'bc': {'wall_s': 50., 'env_steps': 10_000},
          'online_warmup': {'wall_s': 60., 'env_steps': 20_000}, 'rl': {'wall_s': 3000., 'env_steps': 980_000}}


def evals(final, *, sto_clean=10, wiggle=.05):
    """lap 2 falls linearly to `final` at 0.6 T, then +-wiggle around it."""
    out = []
    for i, step in enumerate(range(0, T + 1, 50_000)):
        lap2 = final + 6. * max(0., .6 - step / T) + (wiggle if i % 2 else 0.)
        out.append({'step': step, 'det_clean': True, 'lap2_s': lap2, 'sto_clean': sto_clean, 'sto_n': 10})
    out[-1]['lap2_s'] = final
    return out


def write_run(root, finals=(139.0, 139.1, 139.05), mutate=None):
    root.mkdir(parents=True, exist_ok=True)
    for k, final in enumerate(finals):
        d = root / f'seed{k}'
        d.mkdir()
        files = {'config': dict(copy.deepcopy(CONFIG), seed=k),
                 'provenance': {'rlsetup': {'commit': 'a' * 40}, 'ocd_lap_code_version': 'b' * 12},
                 'status': {'state': 'completed', 'phases': copy.deepcopy(PHASES)}, 'evals': evals(final)}
        if mutate:
            mutate(k, files)
        for name in ('config', 'provenance', 'status'):
            (d / f'{name}.json').write_text(json.dumps(files[name]))
        (d / 'evals.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in files['evals']))
    return root


def criteria(tmp_path, **kw):
    rep = R.build_report(write_run(tmp_path / 'run', **kw), reeval=False)
    return rep, rep['criteria']


def test_all_pass(tmp_path):
    rep, c = criteria(tmp_path)
    assert all(c.values()) and rep['pass'], c
    assert rep['seed_lap2_std_s'] == pytest.approx(.05)
    s = rep['seeds'][0]
    assert s['plateau_improvement_s'] == pytest.approx(0.) and s['last20_lap2_range_s'] == pytest.approx(.05)
    assert R.main([str(tmp_path / 'run'), '--no-reeval', '--strict']) == 0
    assert (tmp_path / 'run' / 'd2_report.md').read_text().count('PASS') >= 7


def only_fails(c, *names):
    return {k for k, v in c.items() if not v} == set(names)


def test_consistency_fails(tmp_path):
    def m(k, f):
        if k == 2:
            f['config']['agent']['batch_size'] = 256
    _, c = criteria(tmp_path, mutate=m)
    assert only_fails(c, 'consistency')
    def m2(k, f):
        if k == 1:
            f['provenance']['ocd_lap_code_version'] = 'c' * 12
    _, c = criteria(tmp_path / 'b', mutate=m2)
    assert only_fails(c, 'consistency')


def test_clean_fails(tmp_path):
    def m(k, f):
        if k == 0:
            f['evals'][-1].update(det_clean=False, lap2_s=None)
    _, c = criteria(tmp_path, mutate=m)
    assert only_fails(c, 'i_clean', 'ii_lap2', 'iv_seed_std')   # no final lap -> no std either


def test_lap2_fails(tmp_path):
    _, c = criteria(tmp_path, finals=(140.65, 140.6, 140.55))
    assert only_fails(c, 'ii_lap2')


def test_plateau_fails(tmp_path):
    def m(k, f):
        if k == 1:                        # still improving: 0.3 s after the 0.8 T evaluation
            for r in f['evals']:
                if r['step'] == 800_000:
                    r['lap2_s'] = 139.4
    _, c = criteria(tmp_path, mutate=m)
    assert only_fails(c, 'iii_plateau')
    def m2(k, f):
        if k == 1:
            f['evals'] = [r for r in f['evals'] if r['step'] <= 800_000 or r['step'] == T]
            f['evals'][[r['step'] for r in f['evals']].index(800_000)].update(det_clean=False, lap2_s=None)
    _, c = criteria(tmp_path / 'b', mutate=m2)
    assert only_fails(c, 'iii_plateau')


def test_seed_std_fails(tmp_path):
    _, c = criteria(tmp_path, finals=(139.0, 139.5, 140.0))
    assert only_fails(c, 'iv_seed_std')
    _, c = criteria(tmp_path / 'b', finals=(139.0, 139.1))       # fewer than 3 seeds
    assert only_fails(c, 'iv_seed_std')


def test_stochastic_fails(tmp_path):
    def m(k, f):
        if k == 2:
            f['evals'][-1]['sto_clean'] = 8
    _, c = criteria(tmp_path, mutate=m)
    assert only_fails(c, 'v_stochastic')
    def m2(k, f):                          # 3/3 clean is not 9/10 (smoke preset logs 3 laps)
        f['evals'][-1].update(sto_clean=3, sto_n=3)
    _, c = criteria(tmp_path / 'b', mutate=m2)
    assert only_fails(c, 'v_stochastic')


def test_logged_fails(tmp_path):
    def m(k, f):
        if k == 0:
            del f['status']['phases']['rl']
    _, c = criteria(tmp_path, mutate=m)
    assert only_fails(c, 'vi_logged')
