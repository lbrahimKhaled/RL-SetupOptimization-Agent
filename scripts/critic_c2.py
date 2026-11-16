#!/usr/bin/env python
"""C2b: critics, training, evaluation (docs/specs/C2b_addendum.md, "Runs and budget" R0-R9).

Reads the data build of scripts/critic_c2_data.py (default: the latest under runs/critic_c2_data/),
writes runs/critic_c2/<UTC>/ (gitignored):
  bars.json      the addendum's bars, written (read-only) BEFORE any training
  folds.json     F_cell / F_person (3 salts) / F_hash keyed to the data hash (B6); asserted on resume
  config.json    full config, argv, provenance (run_provenance, data hash, style-rl commit, ...)
  fits/<run>/<fit>.{json,npz[,pt]}  one per fit: params hash, info, seconds, test-row predictions;
                 a fit whose json carries the same hash (data, rows, config, code, fit params) is skipped
  metrics/<run>.json and metrics.json (merged), timing.json
  bundles/{critic1,critic2,critic3}_final.{pt,json}  (R9); TD3's Q is always critic1_final (spec 12.21,
                 bundle role 'td3_q'); critic3_final is the style-free reference, never TD3's Q
No username is read or written: rows are positions in dataset.npz; persons / drivers are integers. The json /
npz outputs hold no person or driver id; the pickled fits (fits/*/*.pt, gitignored) hold the nuisance encoder,
whose level maps are keyed by integer driver ids (game x 1e9 + driver) with one mean date per driver (QA i1).
Resume safety (QA M1): the fit key includes the sha256 of dataset.npz + summary.json, the code digest of
rlsetup/critic/** + rlsetup/style/* + this script, the torch / numpy / scipy versions and --threads; folds.json
stores the build hash and a resume with another build is refused.

Runs: R0 smoke (F_cell fold 0, short training, timing + budget projection), R1 ridge / nuisance baselines,
R2 groups (PCA vs physics), R3 critic 3, R4 Barcelona heads + leave-out + Barcelona-only, R5 critic 1,
R6 critic 1 shuffle ablation, R7 critic 2, R8 sensitivity (Imola, Las Vegas dropped), R9 final bundles.
`--rows S` reruns R5-R7 on rows S (the B16 fallback; only after the trigger fired, or with --force-fallback).

Usage: .venv/bin/python scripts/critic_c2.py --runs R0 [--smoke] [--threads 1] [--data DIR] [--out DIR | --resume DIR]
"""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import warnings
import os
import stat
import subprocess
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlsetup import runs                                                              # noqa: E402
from rlsetup.critic import ridge                                                     # noqa: E402
from rlsetup.critic import train as T                                                # noqa: E402
from rlsetup.critic.bundle import CriticBundle, save_bundle                          # noqa: E402
from rlsetup.critic.c2_config import (ASSISTS_ABS_TC, BARCELONA, BARS, BRAKING_ASSIST, C2bConfig)  # noqa: E402
from rlsetup.critic.c2data import load_c2data                                        # noqa: E402
from rlsetup.critic.config import KNOB_ORDER, REPO_ROOT                              # noqa: E402
from rlsetup.critic.meta_prior import closeness, fit_meta_prior                      # noqa: E402
from rlsetup.critic.sources.f1_meta import published_meta_p, read_meta_rows         # noqa: E402
from rlsetup.critic.evaluate import Evaluator, stack_preds                           # noqa: E402
from rlsetup.critic.folds import load_folds, make_folds, write_folds                 # noqa: E402
from rlsetup.critic.groups import adjusted_rand, group_names, labels_of, pca_groups, physics_groups  # noqa: E402
from rlsetup.critic.metrics import singleton_rows, spearman, sst_w, strata_shuffle  # noqa: E402
from rlsetup.critic.provenance import library_versions                               # noqa: E402
from rlsetup.provenance import run_provenance                                       # noqa: E402
from rlsetup.style import STYLE_RL_COMMIT                                            # noqa: E402

RUNS = ('R0', 'R1', 'R2', 'R3', 'R3b', 'R3c', 'R3d', 'R4', 'R5', 'R5b', 'R6', 'R7', 'R8', 'R9')
SENS_DROP = ('imola', 'las_vegas')          # B3 sensitivity (R8)
STYLE_RL_REPO = Path(os.environ.get('STYLE_RL_REPO', Path.home() / 'Documents/style-rl'))
CODE_FILES = sorted(str(p.relative_to(REPO_ROOT)) for p in [*(REPO_ROOT / 'rlsetup/critic').rglob('*.py'),
                                                             *(REPO_ROOT / 'rlsetup/style').glob('*.py')]) + \
    ['scripts/critic_c2.py']                # QA M1; scripts/critic_c2_report.py is deliberately outside


def code_digest() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((REPO_ROOT / f).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------
# per-fit store (resumable)
# ---------------------------------------------------------------------------------------------------
class Store:
    def __init__(self, root: Path, base: dict):
        self.root, self.base = Path(root), base
        self.timing = []

    def _hash(self, params):
        return hashlib.sha256(json.dumps({**self.base, **params}, sort_keys=True, default=str).encode()).hexdigest()

    def _p(self, run, fid):
        return self.root / 'fits' / run / fid

    def get(self, run, fid, params):
        p = self._p(run, fid)
        j = p.with_suffix('.json')
        if not j.exists() or not p.with_suffix('.npz').exists():
            return None
        meta = json.loads(j.read_text())
        if meta.get('hash') != self._hash(params):
            return None
        z = np.load(p.with_suffix('.npz'))
        return {k: z[k] for k in z.files} | {'_info': meta['info'], '_seconds': meta['seconds']}

    def put(self, run, fid, params, preds: dict, info: dict, seconds: float, fitted=None):
        p = self._p(run, fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        if fitted is not None:
            torch.save(fitted, p.with_suffix('.pt'))
        np.savez_compressed(p.with_suffix('.npz'), **preds)
        runs.write_json(p.with_suffix('.json'), {'hash': self._hash(params), 'params': params, 'info': info,
                                                 'seconds': seconds, 'created_utc': runs.utc_now()})
        self.timing.append({'run': run, 'fit': fid, 'seconds': seconds})

    def fitted(self, run, fid):
        p = self._p(run, fid).with_suffix('.pt')
        return torch.load(p, weights_only=False) if p.exists() else None


# ---------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        self.args = args
        self.cfg = C2bConfig()
        self.d = load_c2data(args.data)
        d = self.d
        self.smoke = bool(args.smoke)
        self.repeats = int(args.repeats)
        self.priors, self._published = {}, None
        if args.resume:
            self.out = Path(args.resume)
        else:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%SZ')
            self.out = Path(args.out) / stamp
        self.out.mkdir(parents=True, exist_ok=True)
        self.write_bars()
        fpath = self.out / 'folds.json'
        if not fpath.exists():
            folds = make_folds(d.data_hash, np.array(d.games)[d.game], d.driver, d.track_key, d.person, d.setup_hash)
            write_folds(fpath, folds, d.data_hash, d.rows_sha256, d.build_sha256)
        self.folds, self.folds_sha256 = load_folds(fpath, d.data_hash, d.rows_sha256, d.build_sha256)
        self.code = code_digest()
        import scipy
        base = {'data_hash': d.data_hash, 'rows_sha256': d.rows_sha256, 'build_sha256': d.build_sha256,
                'config': asdict(self.cfg), 'code': self.code, 'smoke': self.smoke,
                'drop_quali': bool(getattr(args, 'drop_quali', False)), 'threads': int(args.threads),
                'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}}
        self.store = Store(self.out, base)
        self.ev = Evaluator(d, self.cfg.eval.n_boot, self.cfg.eval.boot_seed, self.cfg.eval.mde_factor)
        self.A = np.ones(d.n, bool)
        if getattr(args, 'drop_quali', False) and 'Quali' in d.sessions:     # B22.3 rule fired: rows A without Quali
            self.A &= d.session != list(d.sessions).index('Quali')
        self.E = d.exact & d.has_style                      # B5: exact rows with style (critics 1, 2)
        self.S = (d.cell >= 0) & d.has_style                # rows S with style (B16 fallback only)
        self.style_rows = self.S if args.rows == 'S' else self.E
        self.rows_tag = '_S' if args.rows == 'S' else ''
        cfg_doc = {'argv': sys.argv[1:], 'c2b_config': asdict(self.cfg), 'data_dir': d.provenance['data_dir'],
                   'data_hash': d.data_hash, 'rows_sha256': d.rows_sha256, 'folds_sha256': self.folds_sha256,
                   'code_sha256': self.code, 'code_files': CODE_FILES, 'threads': args.threads, 'repeats': self.repeats,
                   'smoke': self.smoke, 'rows': args.rows, 'counts': {'A': int(d.n), 'exact': int(d.exact.sum()),
                                                                      'exact_with_style': int(self.E.sum()),
                                                                      'exact_without_style': int((d.exact & ~d.has_style).sum()),
                                                                      'S_with_style': int(self.S.sum())},
                   'provenance': self.provenance()}
        runs.write_json(self.out / 'config.json', cfg_doc)

    def published_meta(self) -> dict:
        """{(game, track): p} of the published meta (B21.1 sensitivity only; never delivered)."""
        if self._published is None:
            self._published = published_meta_p(read_meta_rows([str(Path(self.args.meta_sources))]))
        return self._published

    def provenance(self):
        d = self.d
        rp = {k: ({kk: (Path(vv).name if kk == 'path' and vv else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
              for k, v in run_provenance().items()}          # absolute paths hold the account name: basename only
        return {'run_provenance': rp, 'libraries': library_versions() | {'torch': torch.__version__},
                'data': d.provenance, 'style_rl_commit': STYLE_RL_COMMIT, 'folds_sha256': getattr(self, 'folds_sha256', None)}

    def write_bars(self):
        """bars.json before any training; read-only afterwards. A resume checks it is unchanged."""
        p = self.out / 'bars.json'
        doc = {'source': 'docs/specs/C2b_addendum.md, section Bars (16 Nov 2026)', 'bars': BARS}
        text = runs.to_json(doc, indent=1)
        if p.exists():
            if json.loads(p.read_text()) != json.loads(text):
                raise RuntimeError('bars.json in the run directory differs from the addendum bars')
            return
        p.write_text(text + '\n')
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # ---- folds ----
    def fold_sets(self, sets=('F_cell', 'F_person'), repeats=None):
        R = self.repeats if repeats is None else repeats
        for fs in sets:
            for r in range(1 if fs == 'F_hash' else min(R, self.folds[fs].shape[0])):
                yield fs, r

    def fold_rows(self, fs, r, k, universe):
        f = self.folds[fs][r]
        return np.flatnonzero(universe & (f != k)), np.flatnonzero(universe & (f == k))

    def n_folds(self):
        return 1 if self.smoke else 5

    def sst(self, fs, universe, repeats):
        out = np.full((len(repeats), self.d.n), np.nan)
        for i, r in enumerate(repeats):
            for k in range(5):
                tr, te = self.fold_rows(fs, r, k, universe)
                if len(te) and len(tr):
                    out[i, te] = sst_w(self.d.gap, self.d.group_key(), np.isin(np.arange(self.d.n), tr), te)
        return out

    def log(self, *a):
        print(*a, flush=True)


def preds_of(rows, pr: dict, keys=('full', 'setup')):
    return {'rows': np.asarray(rows)} | {k: np.asarray(pr[k]) for k in keys}


def run_fit(c: Ctx, run, fid, params, fn):
    """Cached fit: `fn()` -> (preds dict, info dict, fitted or None)."""
    got = c.store.get(run, fid, params)
    if got is not None:
        return got
    t0 = time.time()
    preds, info, fitted = fn()
    dt = time.time() - t0
    c.store.put(run, fid, params, preds, info, dt, fitted)
    got = c.store.get(run, fid, params)
    c.log(f'  [{run}] {fid}: {dt:.1f} s')
    return got


def collect(c: Ctx, run, ids_by_repeat, key='full', rows_key='rows'):
    """(R, n) from cached fits: ids_by_repeat = [[fid, ...] per repeat]."""
    items = []
    for ids in ids_by_repeat:
        parts = []
        for fid in ids:
            p = c.store._p(run, fid).with_suffix('.npz')
            z = np.load(p)
            parts.append((z[rows_key], z[key]))
        items.append(parts)
    return stack_preds(c.d.n, items)


def ids(prefix, fs, r, nf, suffix=''):
    return [f'{prefix}_{fs}_r{r}_f{k}{suffix}' for k in range(nf)]


# ---------------------------------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------------------------------
def fold_groups(c: Ctx, kind, train_rows):
    if kind == 'physics':
        return physics_groups()
    d = c.d
    return pca_groups(d.p[train_rows], [f'{t}|{h}' for t, h in zip(d.track[train_rows], d.setup_hash[train_rows])],
                      c.cfg.eval.group_k)


def chosen_groups(c: Ctx):
    p = c.out / 'metrics' / 'R2.json'
    if not p.exists():
        if c.smoke:
            return 'pca'
        raise RuntimeError('R2 (groups) must run before R3 and later runs')
    return json.loads(p.read_text())['choice']['groups']


# ---------------------------------------------------------------------------------------------------
# critic-3 fit helper
# ---------------------------------------------------------------------------------------------------
def c3_fit(c: Ctx, run, fid, train, test, groups_kind, twin=None, guard=True, seed_key=None, extra_test=None,
           extra=None, prior=None, wd=None, session_dev=False):
    """One critic-3 fit (cached). `prior`: B21 MetaPrior with lambda (None = plain); `wd`: fixed weight decay
    (B21.3); `session_dev`: B22 S22b nuisance block. The same `seed_key` and twin give the same seed,
    bootstrap and inner split as the plain R3 fit, so only the objective differs."""
    d = c.d
    seed = T.seed_of('c3', seed_key or fid, twin)
    params = {'kind': 'critic3', 'groups': groups_kind, 'twin': twin, 'guard': guard, 'train_sha': _sha(train),
              'test_sha': _sha(test), 'seed': seed}
    if extra is not None:           # generic hook: same seed, folds and twins; only the objective differs
        params['extra_loss'] = extra.spec()
    if prior is not None:
        params['meta_prior'] = {'source': prior.source, 'c': prior.c, 'lam': prior.lam,
                                'w_sha': hashlib.sha256(np.ascontiguousarray(prior.w).tobytes()).hexdigest()[:16],
                                'm_sha': hashlib.sha256(np.ascontiguousarray(np.nan_to_num(prior.m, nan=-9.)).tobytes()).hexdigest()[:16]}
    if wd is not None:
        params['wd'] = wd
    if session_dev:
        params['session_dev'] = True

    def fn():
        grp = fold_groups(c, groups_kind, train)
        w = T.boot_weights(d.setup_hash[train], T.seed_of('boot_hash', seed_key or fid, twin)) if twin is not None else None
        f = T.pretrain(d, train, 'critic3', grp, c.cfg, seed=seed, weights=w, guard=guard, smoke=c.smoke, salt=seed_key or fid,
                       extra=extra, prior=prior, wd_grid=None if wd is None else (wd,), session_dev=session_dev)
        f.info['groups'] = [list(g) for g in grp]
        f.info['guard_stats'] = guard_stats(c, f, train, seed)
        rows = test if extra_test is None else np.union1d(test, extra_test)
        keys = ('full', 'setup', 'session_dev') if session_dev else ('full', 'setup')
        if session_dev:
            names = f.encoder.encode(d, train[:1]).dense_names
            coef = f.nuis.dense.detach().double().numpy()
            f.info['sdev_coef'] = {sn: [float(coef[names.index(f'sdev:{sn}:{j}')]) for j in range(d.p.shape[1])]
                                   for sn in c.cfg.session.s22b_sessions if f'sdev:{sn}:0' in names}
        if not len(rows):
            return {'rows': rows} | {k: np.zeros(0) for k in keys}, f.info, f
        return preds_of(rows, T.predict(f, d, rows), keys), f.info, f
    return run_fit(c, run, fid, params, fn)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()[:16]


def guard_stats(c: Ctx, f, train, seed, n=4096):
    """C1 amendment item 1: paired delta (sum h(p_corrupt) - sum h(p_source)) on the guard's own sampling
    distribution: mean and p5 / median / p95, for reporting only."""
    d = c.d
    z, g = T.context(d, train, f.scaler)
    gen = torch.Generator().manual_seed(seed)
    P, Z, G = T._t(d.p[train]), T._t(z), T._t(g)
    lo, hi = T._t(f.support[0]), T._t(f.support[1])
    idx = torch.randint(0, len(train), (n,), generator=gen)
    ps = P[idx]
    mask = torch.rand(ps.shape, generator=gen) < c.cfg.guard.prob
    pc = torch.where(mask, torch.rand(ps.shape, generator=gen) * 2 - 1, ps)
    out = ((pc < lo) | (pc > hi)).any(1)
    if not out.any():
        return None
    with torch.no_grad():
        delta = (f.model.setup_sum(pc[out], Z[idx][out], G[idx][out]) - f.model.setup_sum(ps[out], Z[idx][out], G[idx][out])).numpy()
    q = np.percentile(delta, [5, 50, 95])
    return {'n': int(out.sum()), 'mean': float(delta.mean()), 'p5': float(q[0]), 'median': float(q[1]), 'p95': float(q[2])}


# ---------------------------------------------------------------------------------------------------
# R0 smoke
# ---------------------------------------------------------------------------------------------------
def run_R0(c: Ctx):
    d, cfg = c.d, c.cfg
    if not c.smoke:
        raise RuntimeError('R0 is the smoke run: pass --smoke')
    fs, r, k = 'F_cell', 0, 0
    tr, te = c.fold_rows(fs, r, k, c.A)
    etr, ete = c.fold_rows(fs, r, k, c.E)
    timing = {}

    def timed(name, fn):
        t0 = time.time()
        out = fn()
        timing[name] = time.time() - t0
        c.log(f'  [R0] {name}: {timing[name]:.2f} s')
        return out
    nu = timed('nuisance_only_A', lambda: ridge.nuisance_only(d, tr, te, cfg))
    ra = timed('ridge_A', lambda: ridge.ridge_A(d, tr, te, cfg))
    nue = timed('nuisance_only_exact', lambda: ridge.nuisance_only(d, etr, ete, cfg, style=True))
    rex = timed('ridge_exact', lambda: ridge.ridge_exact(d, etr, ete, cfg))
    grp = fold_groups(c, 'pca', tr)
    seed = T.seed_of('R0', 'c3', 0)
    w = T.boot_weights(d.setup_hash[tr], T.seed_of('R0', 'boot', 0))
    f3 = timed('critic3_twin0_20epochs', lambda: T.pretrain(d, tr, 'critic3', grp, cfg, seed=seed, weights=w, smoke=True, salt='R0'))
    head, hinfo = timed('barcelona_head', lambda: T.fit_bcn_head(d, tr, f3, cfg.bcn, seed=seed, smoke=True, salt='R0'))
    f3.bcn = head
    wc = T.boot_weights(d.cell_key[etr], T.seed_of('R0', 'bootc', 0))
    f1 = timed('critic1_twin0_100steps', lambda: T.finetune(d, etr, f3, 'critic1', cfg, seed=seed, weights=wc, smoke=True, salt='R0'))
    c3e = timed('critic3_exact_comparator', lambda: T.finetune(d, etr, f3, 'c3_exact', cfg, smoke=True))
    f2p = timed('critic2_pretrain_20epochs', lambda: T.pretrain(d, tr, 'critic2', None, cfg, seed=seed, smoke=True, salt='R0c2'))
    f2 = timed('critic2_finetune_100steps', lambda: T.finetune(d, etr, f2p, 'critic2', cfg, seed=seed, smoke=True, salt='R0c2'))
    # B21-B23 wiring on the same fold (smoke budgets): meta prior, critic 3 with Pi (c = 0.1, wd of the c = 0 fit),
    # the S22b session-deviation critic 3, critic 1 at level L3 (B23 rung 2), critic 1 inheriting Pi
    pri = timed('meta_prior_fit', lambda: fit_meta_prior(d, tr, cfg))
    pri = pri.with_lambda(.1, float(np.std(T.predict(f3, d, tr, with_bcn=False, nuisance=False)['setup'])),
                          float(np.std(pri.pi_rows(d.p[tr], d.game[tr], d.track[tr]))))
    f3m = timed('critic3_b21_20epochs', lambda: T.pretrain(d, tr, 'critic3', grp, cfg, seed=seed, weights=w, smoke=True, salt='R0',
                                                          prior=pri, wd_grid=(f3.info['weight_decay'],)))
    f3s = timed('critic3_s22b_20epochs', lambda: T.pretrain(d, tr, 'critic3', grp, cfg, seed=seed, weights=w, smoke=True,
                                                           salt='R0', session_dev=True))
    f1l = timed('critic1_L3_100steps', lambda: T.finetune(d, etr, f3, 'critic1', cfg, seed=seed, weights=wc, smoke=True, salt='R0',
                                                         level=cfg.ladder.levels[-1]))
    f1m = timed('critic1_from_b21_100steps', lambda: T.finetune(d, etr, f3m, 'critic1', cfg, seed=seed, weights=wc, smoke=True,
                                                               salt='R0'))
    b21_check = {'lambda': pri.lam, 'prior_sd_train': float(np.std(pri.offset(d, tr))),
                 'pi_identical_after_critic1': bool(np.array_equal(T.predict(f1m, d, te)['prior'], T.predict(f3m, d, te)['prior'])),
                 'critic1_L3_e_gate': float(f1l.model.e_gate), 's22b_info': f3s.info['session_dev'],
                 'pooled_m': pri.info['pooled_m']}
    infos = {'critic3': f3.info, 'bcn': hinfo, 'critic1': f1.info, 'critic2_pre': f2p.info, 'critic2': f2.info,
             'critic3_b21': f3m.info, 'critic3_s22b': f3s.info, 'critic1_L3': f1l.info}
    # per-epoch / per-step times for the projection
    ep3 = sum(1 for _ in cfg.pretrain.weight_decay_grid) * f3.info['epochs_run']
    per_epoch_c3 = timing['critic3_twin0_20epochs'] / max(ep3, 1)
    per_epoch_c2 = timing['critic2_pretrain_20epochs'] / max(len(cfg.pretrain.weight_decay_grid) * f2p.info['epochs_run'], 1)
    per_step_c1 = timing['critic1_twin0_100steps'] / max(len(cfg.finetune.l2_grid) * f1.info['steps_run'], 1)
    proj = budget_projection(c, per_epoch_c3, per_epoch_c2, per_step_c1, timing)
    # metrics on the smoke fold (smoke: not results)
    def scatter(rows, v):
        o = np.full(d.n, np.nan)
        o[rows] = v
        return o[None]
    p3 = T.predict(f3, d, te)
    sst = np.full((1, d.n), np.nan)
    sst[0, te] = sst_w(d.gap, d.group_key(), np.isin(np.arange(d.n), tr), te)
    m = {'smoke': True, 'fold': f'{fs} r{r} f{k}', 'timing_s': timing, 'infos': infos, 'projection': proj, 'b21_b23_wiring': b21_check,
         'critic3_A': c.ev.block(scatter(te, p3['full']), scatter(te, p3['setup']), scatter(te, nu['full']), sst,
                                 np.isin(np.arange(d.n), te)),
         'ridge_A': c.ev.block(scatter(te, ra['full']), scatter(te, ra['setup']), scatter(te, nu['full']), sst,
                               np.isin(np.arange(d.n), te))}
    sste = np.full((1, d.n), np.nan)
    sste[0, ete] = sst_w(d.gap, d.group_key(), np.isin(np.arange(d.n), etr), ete)
    em = np.isin(np.arange(d.n), ete)
    p1, p3e, p2 = T.predict(f1, d, ete), T.predict(c3e, d, ete), T.predict(f2, d, ete)
    m['critic1_exact_dr2_vs_c3'] = c.ev.r2_diff(scatter(ete, p1['full']), scatter(ete, p3e['full']), sste, em)
    m['ridge_exact'] = c.ev.block(scatter(ete, rex['full']), scatter(ete, rex['setup']), scatter(ete, nue['full']), sste, em, pairs=False)
    m['critic2_exact_mae_minus_nuis'] = c.ev.mae_diff(scatter(ete, p2['full']), scatter(ete, nue['full']), em)
    m['critic1_e_clipped_frac'] = p1['clipped_frac']
    # smoke bundles: save -> load -> identical q
    bdir = c.out / 'bundles_smoke'
    tz = {t: c.d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(c.d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    s_tr = T.style_inputs(d, etr, f1.style_tf)[0]
    prov = c.provenance()
    checks = {}
    for kind, fits, st in (('critic3', [f3], None), ('critic1', [f1], s_tr), ('critic2', [f2], s_tr)):
        path = save_bundle(bdir / f'{kind}_smoke.pt', kind, fits, sources=d.games, track_z_raw=tz, z_provisional=zp,
                           style_train=st, provenance=prov)
        b = CriticBundle.load(path)
        row = ete[0]
        ctx = b.context(track=d.track_key[row], style=d.style_raw[row], source=d.games[d.game[row]])
        q_b = float(b.q(T._t(d.p[row]), ctx))
        pr = T.predict(fits[0], d, np.array([row]))
        checks[kind] = {'q_bundle': q_b, 'minus_setup_predict': float(-pr['setup'][0]), 'abs_diff': abs(q_b + float(pr['setup'][0]))}
    m['bundle_roundtrip'] = checks
    write_metrics(c, 'R0', m)
    c.log(f"R0 projection: {json.dumps(proj['summary'])}")


def budget_projection(c: Ctx, per_epoch_c3, per_epoch_c2, per_step_c1, timing):
    """Projected wall time of R1-R9 from R0's measured per-epoch / per-step times (single thread).
    Fit counts follow the addendum's run table with R repeats. 'upper': every fit runs the full epoch /
    step budget for every grid value; 'typical': 150 epochs and 300 steps per grid value (PLACEHOLDER;
    an exploratory full critic-3 fit on F_cell fold 0 stopped at 115 epochs)."""
    cfg = c.cfg
    R = c.repeats if not c.smoke else 3
    n_wd, n_l2 = len(cfg.pretrain.weight_decay_grid), len(cfg.finetune.l2_grid)
    counts = {'critic3': 20 + (20 * R + 10 + 10 + 5) + 7 + 5 + 2,        # R2; R3 (+F_hash, unguarded, strict); R4; R8; R9
              'critic2_pretrain': 10 * R + 1,                             # R7, R9
              'finetune_grid': 20 * R + 10 + 5 + 3 + 10 * R,             # R5 (+unguarded), R8, R9, R7 fine-tunes
              'finetune_fixed_l2': 100 * R + 50 * R,                      # R6 (2 schemes x 5 seeds, twin 0), R7 shuffles
              'barcelona_head': 20 * R + 10 * R + 3,                      # R4, R7, R9
              'ridge_fold': (2 * R * 5) * 4 + 5 * 2 + R * 5 * 6,         # R1
              'critic3_b21': 45 + 10 + 1,                                 # R3b (3 sets x 5 folds x 3 c + published 10), R9 c3B21final
              'critic3_s22': 20,                                          # R3d (S22a 10, S22b 10)
              'meta_prior': 15 + 10 + 1}                                  # R3b in-fold / published priors, final
    cond = {'critic3_b21_R3c': 20 * R + 10 + 5,                           # R3c (only if triggered or adopted)
            'finetune_level_R5b': 3 * 2 * R * 5 * 2 + 3 * 2}              # R5b (only if rung 0 fails) + R9 levels
    bcn_epochs = max(timing.get('barcelona_head', 0.) / max(len(cfg.bcn.l2_grid) * cfg.pretrain.smoke_epochs, 1), 0.)
    ridge_s = (timing['nuisance_only_A'] + timing['ridge_A'] + timing['nuisance_only_exact'] + timing['ridge_exact']) / 4
    out = {}
    for name, ep, st in (('upper', cfg.pretrain.max_epochs, cfg.finetune.max_steps), ('typical', 150, 300)):
        per = {'critic3': per_epoch_c3 * ep * n_wd, 'critic2_pretrain': per_epoch_c2 * ep * n_wd,
               'finetune_grid': per_step_c1 * st * n_l2, 'finetune_fixed_l2': per_step_c1 * st,
               'barcelona_head': bcn_epochs * ep * len(cfg.bcn.l2_grid), 'ridge_fold': ridge_s}
        per.update(critic3_b21=per_epoch_c3 * ep, critic3_s22=per_epoch_c3 * ep * n_wd,
                   meta_prior=timing.get('meta_prior_fit', 0.), critic3_b21_R3c=per_epoch_c3 * ep,
                   finetune_level_R5b=per_step_c1 * st)
        total = sum(per[k] * counts[k] for k in counts)
        total_cond = total + sum(per[k] * cond[k] for k in cond)
        out[name] = {'per_fit_s': per, 'total_h': total / 3600., 'total_with_conditional_h': total_cond / 3600.}
    out['fits'] = counts
    out['conditional_fits'] = cond
    out['repeats'] = R
    out['summary'] = {k: round(out[k]['total_h'], 2) for k in ('upper', 'typical')} | \
        {f'{k}_with_R3c_R5b': round(out[k]['total_with_conditional_h'], 2) for k in ('upper', 'typical')}
    out['rule'] = 'addendum: budget <= 4 h wall, hard stop 6 h; if R0 projects > 6 h cut repeats 3 -> 1 (never epochs or bars)'
    out['measured_per_epoch_s'] = {'critic3': per_epoch_c3, 'critic2': per_epoch_c2}
    out['measured_per_step_s'] = {'finetune': per_step_c1}
    return out


# ---------------------------------------------------------------------------------------------------
# R1 baselines
# ---------------------------------------------------------------------------------------------------
def run_R1(c: Ctx):
    d, cfg = c.d, c.cfg
    nf = c.n_folds()
    bcn = d.track_key == BARCELONA
    sens = np.isin(d.track_key, SENS_DROP)
    for fs, r in c.fold_sets(('F_cell', 'F_person', 'F_hash')):
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            run_fit(c, 'R1', f'nuis_{fs}_r{r}_f{k}', {'m': 'nuis', 'tr': _sha(tr)},
                    lambda: (preds_of(te, ridge.nuisance_only(d, tr, te, cfg), ('full',)), {}, None))
            run_fit(c, 'R1', f'ridgeA_{fs}_r{r}_f{k}', {'m': 'ridgeA', 'tr': _sha(tr)},
                    lambda: (preds_of(te, ridge.ridge_A(d, tr, te, cfg)), {}, None))
            if fs == 'F_hash':
                continue
            etr, ete = c.fold_rows(fs, r, k, c.style_rows)
            run_fit(c, 'R1', f'nuisE_{fs}_r{r}_f{k}{c.rows_tag}', {'m': 'nuisE', 'tr': _sha(etr)},
                    lambda: (preds_of(ete, ridge.nuisance_only(d, etr, ete, cfg, style=True), ('full',)), {}, None))
            run_fit(c, 'R1', f'ridgeE_{fs}_r{r}_f{k}{c.rows_tag}', {'m': 'ridgeE', 'tr': _sha(etr)},
                    lambda: (preds_of(ete, ridge.ridge_exact(d, etr, ete, cfg)), {}, None))
            if fs != 'F_cell':
                continue
            # E4 Barcelona ridge variants (cell folds)
            bte = te[bcn[te]]
            btr = tr[bcn[tr]]
            run_fit(c, 'R1', f'bcnOnly_{fs}_r{r}_f{k}', {'m': 'bcnOnly', 'tr': _sha(btr)},
                    lambda: (preds_of(bte, ridge.ridge_A(d, btr, bte, cfg, with_pz=False)), {}, None))
            ntr = tr[~bcn[tr]]
            run_fit(c, 'R1', f'bcnLeaveOut_{fs}_r{r}_f{k}', {'m': 'bcnLeaveOut', 'tr': _sha(ntr)},
                    lambda: (preds_of(bte, ridge.ridge_A(d, ntr, bte, cfg)), {}, None))
            for a in cfg.eval.bcn_delta_alphas:
                run_fit(c, 'R1', f'bcnDelta{a:g}_{fs}_r{r}_f{k}', {'m': 'bcnDelta', 'a': a, 'tr': _sha(tr)},
                        lambda a=a: (preds_of(te, ridge.ridge_A(d, tr, te, cfg, delta_alpha=a)), {}, None))
            str_ = tr[~sens[tr]]
            ste = te[~sens[te]]
            run_fit(c, 'R1', f'bcnNoImoLv_{fs}_r{r}_f{k}', {'m': 'noImoLv', 'tr': _sha(str_)},
                    lambda: (preds_of(ste, ridge.ridge_A(d, str_, ste, cfg)), {}, None))
    metrics_R1(c)


def metrics_R1(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    out = {'smoke': c.smoke}
    for fs in ('F_cell', 'F_person', 'F_hash'):
        reps = [r for f, r in c.fold_sets((fs,))]
        nu = collect(c, 'R1', [ids('nuis', fs, r, nf) for r in reps])
        ra = collect(c, 'R1', [ids('ridgeA', fs, r, nf) for r in reps])
        ras = collect(c, 'R1', [ids('ridgeA', fs, r, nf) for r in reps], 'setup')
        sst = c.sst(fs, c.A, reps)
        tested = np.isfinite(nu).any(0)
        out[f'ridge_A_{fs}'] = ev.block(ra, ras, nu, sst, tested, pairs=fs != 'F_hash')
        if fs == 'F_hash':
            continue
        nue = collect(c, 'R1', [ids('nuisE', fs, r, nf, c.rows_tag) for r in reps])
        rae = collect(c, 'R1', [ids('ridgeE', fs, r, nf, c.rows_tag) for r in reps])
        sste = c.sst(fs, c.style_rows, reps)
        te = np.isfinite(nue).any(0)
        out[f'ridge_exact_{fs}'] = ev.block(rae, rae, nue, sste, te, pairs=False)
    # Barcelona variants (F_cell)
    reps = [r for f, r in c.fold_sets(('F_cell',))]
    bcn = d.track_key == BARCELONA
    pooled = collect(c, 'R1', [ids('ridgeA', 'F_cell', r, nf) for r in reps], 'setup')
    var = {'barcelona_only': collect(c, 'R1', [ids('bcnOnly', 'F_cell', r, nf) for r in reps], 'setup'),
           'pooled': pooled,
           'leave_barcelona_out': collect(c, 'R1', [ids('bcnLeaveOut', 'F_cell', r, nf) for r in reps], 'setup'),
           'pooled_no_imola_lasvegas': collect(c, 'R1', [ids('bcnNoImoLv', 'F_cell', r, nf) for r in reps], 'setup')}
    for a in c.cfg.eval.bcn_delta_alphas:
        var[f'pooled_delta_{a:g}'] = collect(c, 'R1', [ids(f'bcnDelta{a:g}', 'F_cell', r, nf) for r in reps], 'setup')
    bm = {}
    for name, s in var.items():
        bm[name] = {'barcelona_pairs': ev.pair_acc(s, bcn), 'all_track_pairs': ev.pair_acc(s, np.isfinite(s).any(0))}
    bm['diff_pooled_minus_barcelona_only'] = ev.pair_diff(var['pooled'], var['barcelona_only'], bcn)
    bm['diff_pooled_delta10_minus_pooled'] = ev.pair_diff(var['pooled_delta_10'], var['pooled'], bcn)
    bm['diff_pooled_minus_leave_out'] = ev.pair_diff(var['pooled'], var['leave_barcelona_out'], bcn)
    out['barcelona_ridge_variants_F_cell'] = bm
    write_metrics(c, 'R1', out)


# ---------------------------------------------------------------------------------------------------
# R2 groups (C1-R12 on critic 3, rows A, F_cell repeat 0 + F_hash, 1 twin)
# ---------------------------------------------------------------------------------------------------
def run_R2(c: Ctx):
    nf = c.n_folds()
    labs = []
    for fs in ('F_cell', 'F_hash'):
        for k in range(nf):
            tr, te = c.fold_rows(fs, 0, k, c.A)
            for kind in ('pca', 'physics'):
                got = c3_fit(c, 'R2', f'c3{kind}_{fs}_r0_f{k}', tr, te, kind, twin=None)
                if kind == 'pca':
                    labs.append(labels_of(got['_info']['groups']))
    ev, d = c.ev, c.d
    out = {'smoke': c.smoke}
    res = {}
    for fs in ('F_cell', 'F_hash'):
        nu = collect(c, 'R1', [ids('nuis', fs, 0, nf)])
        sst = c.sst(fs, c.A, [0])
        for kind in ('pca', 'physics'):
            res[(fs, kind)] = (collect(c, 'R2', [ids(f'c3{kind}', fs, 0, nf)]), collect(c, 'R2', [ids(f'c3{kind}', fs, 0, nf)], 'setup'))
        tested = np.isfinite(nu).any(0)
        out[f'r2_gain_{fs}'] = {kind: ev.r2_gain(res[(fs, kind)][0], nu, sst, tested) for kind in ('pca', 'physics')}
    tested = np.isfinite(res[('F_cell', 'pca')][1]).any(0)
    out['pair_acc_F_cell'] = {kind: ev.pair_acc(res[('F_cell', kind)][1], tested) for kind in ('pca', 'physics')}
    diff = ev.pair_diff(res[('F_cell', 'pca')][1], res[('F_cell', 'physics')][1], tested)
    out['pair_diff_pca_minus_physics_F_cell'] = diff
    r2p, r2f = out['r2_gain_F_hash']['pca']['point'], out['r2_gain_F_hash']['physics']['point']
    ci_below = diff['ci'][1] is not None and diff['ci'][1] < 0
    r2_trails = r2p is not None and r2f is not None and (r2f - r2p) > BARS['groups']['r2_margin']
    out['choice'] = {'groups': 'physics' if (ci_below or r2_trails) else 'pca', 'pair_ci_entirely_below_0': bool(ci_below),
                     'r2_trails_physics_by_more_than_0.01': bool(r2_trails), 'rule': BARS['groups']['rule']}
    ari = [adjusted_rand(a, b) for i, a in enumerate(labs) for b in labs[i + 1:]]
    out['pca_groups_per_fold'] = [group_names(_groups_from_labels(l)) for l in labs]
    out['adjusted_rand_index'] = {'mean': float(np.mean(ari)) if ari else None, 'min': float(np.min(ari)) if ari else None,
                                  'n_pairs': len(ari)}
    write_metrics(c, 'R2', out)


def _groups_from_labels(lab):
    return tuple(tuple(int(j) for j in np.flatnonzero(lab == g)) for g in sorted(set(lab.tolist())))


# ---------------------------------------------------------------------------------------------------
# R3 critic 3
# ---------------------------------------------------------------------------------------------------
def run_R3(c: Ctx):
    d = c.d
    nf = c.n_folds()
    kind = chosen_groups(c)
    for fs, r in c.fold_sets(('F_cell', 'F_person', 'F_hash')):
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            for twin in (0, 1):
                c3_fit(c, 'R3', f'c3_{fs}_r{r}_f{k}_t{twin}', tr, te, kind, twin=twin, seed_key=f'c3_{fs}_r{r}_f{k}')
            if r == 0 and fs in ('F_cell', 'F_person'):
                c3_fit(c, 'R3', f'c3unguarded_{fs}_r0_f{k}_t0', tr, te, kind, twin=0, guard=False, seed_key=f'c3_{fs}_r0_f{k}')
            if r == 0 and fs == 'F_cell':                       # strict variant (C1-R4)
                strict = tr[~np.isin(d.setup_hash[tr], d.setup_hash[te])]
                c3_fit(c, 'R3', f'c3strict_{fs}_r0_f{k}_t0', strict, te, kind, twin=0, seed_key=f'c3strict_{fs}_r0_f{k}')
    metrics_R3(c)


def twin_mean(c, run, prefix, fs, reps, nf, key):
    a = collect(c, run, [ids(prefix, fs, r, nf, '_t0') for r in reps], key)
    b = collect(c, run, [ids(prefix, fs, r, nf, '_t1') for r in reps], key)
    return (a + b) / 2, a, b


def metrics_R3(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    out = {'smoke': c.smoke, 'groups': chosen_groups(c)}
    for fs in ('F_cell', 'F_person', 'F_hash'):
        reps = [r for f, r in c.fold_sets((fs,))]
        full, f0, f1 = twin_mean(c, 'R3', 'c3', fs, reps, nf, 'full')
        setup, s0, s1 = twin_mean(c, 'R3', 'c3', fs, reps, nf, 'setup')
        nu = collect(c, 'R1', [ids('nuis', fs, r, nf) for r in reps])
        sst = c.sst(fs, c.A, reps)
        tested = np.isfinite(full).any(0)
        blk = ev.block(full, setup, nu, sst, tested, pairs=fs != 'F_hash')
        out[f'critic3_{fs}'] = blk
        out[f'critic3_twin0_{fs}'] = ev.block(f0, s0, nu, sst, tested, pairs=fs != 'F_hash')
        if fs != 'F_hash':
            ras = collect(c, 'R1', [ids('ridgeA', fs, r, nf) for r in reps], 'setup')
            out[f'critic3_minus_ridge_pairs_{fs}'] = ev.pair_diff(setup, ras, tested)
            out[f'ridge_pairs_{fs}'] = ev.pair_acc(ras, tested)
    # strict
    st = collect(c, 'R3', [ids('c3strict', 'F_cell', 0, nf, '_t0')], 'setup')
    out['critic3_strict_F_cell_r0'] = ev.pair_acc(st, np.isfinite(st).any(0))
    # guard comparison (C1 amendment item 4) on common on-support held-out rows / pairs
    gc = {}
    for fs in ('F_cell', 'F_person'):
        g_full = collect(c, 'R3', [ids('c3', fs, 0, nf, '_t0')])
        g_set = collect(c, 'R3', [ids('c3', fs, 0, nf, '_t0')], 'setup')
        u_full = collect(c, 'R3', [ids('c3unguarded', fs, 0, nf, '_t0')])
        u_set = collect(c, 'R3', [ids('c3unguarded', fs, 0, nf, '_t0')], 'setup')
        on = on_support_mask(c, 'R3', [f'c3_{fs}_r0_f{k}_t0' for k in range(nf)]) & np.isfinite(g_full[0])
        sst = c.sst(fs, c.A, [0])
        pd_ = ev.pair_diff(g_set, u_set, on)
        rd = ev.r2_diff(g_full, u_full, sst, on)
        blocks = (pd_['ci'][1] is not None and pd_['ci'][1] < 0) or (rd['ci'][1] is not None and rd['ci'][1] < 0) \
            or pd_['ci'][0] is None or rd['ci'][0] is None
        gc[fs] = {'pair_acc_guarded_minus_unguarded': pd_, 'r2_guarded_minus_unguarded': rd, 'blocks_readiness': bool(blocks),
                  'n_on_support_rows': int(on.sum())}
    out['guard_comparison'] = gc
    infos = [json.loads(p.read_text())['info'] for p in sorted((c.out / 'fits' / 'R3').glob('c3_*.json'))]
    rb = r11_ranking_bars(c, out)
    fh = out['critic3_F_hash']['pooled']['r2_gain']['point']
    out['bars_judged'] = {'r11_ranking': rb, 'F_hash_r2_gain_gt_0': bool(fh is not None and fh > BARS['critic3_A_pooled']
                                                                           ['heldout_setup_r2_gain_gt']),
                          'all_R11_met': bool(rb['all_met'] and fh is not None and fh > 0.)}
    out['stalled_fits'] = int(sum(i.get('stalled', False) for i in infos))
    out['weight_decay_chosen'] = {str(w): int(sum(i.get('weight_decay') == w for i in infos)) for w in c.cfg.pretrain.weight_decay_grid}
    out['guard_stats'] = [i.get('guard_stats') for i in infos[:6]]
    write_metrics(c, 'R3', out)


# ---------------------------------------------------------------------------------------------------
# B21 meta prior: R3b (always), R3c (only if triggered or adopted); delivered critic 3
# ---------------------------------------------------------------------------------------------------
def fold_prior(c: Ctx, key, train, source='in_fold'):
    """B21.1-2 prior of one fold's training rows (lambda = 0), cached per run of the script."""
    k = (key, source, _sha(train))
    if k not in c.priors:
        c.priors[k] = fit_meta_prior(c.d, train, c.cfg, published=c.published_meta() if source == 'published' else None)
    return c.priors[k]


def with_lambda(c: Ctx, prior, cval, base_run, base_fid, train):
    """lambda = c SD_train(setup part of the c = 0 fit) / SD_train(Pi) (B21.3); twins share it (twin-0 base)."""
    f0 = c.store.fitted(base_run, base_fid)
    if f0 is None:
        raise RuntimeError(f'{base_run}/{base_fid} (the c = 0 fit) must exist before B21 fits')
    sd_setup = float(np.std(T.predict(f0, c.d, train, with_bcn=False, nuisance=False)['setup']))
    sd_pi = float(np.std(prior.pi_rows(c.d.p[train], c.d.game[train], c.d.track[train])))
    return prior.with_lambda(cval, sd_setup, sd_pi)


def _mode_c(values):
    """Most frequent c; a tie goes to the smaller c (B21.3)."""
    vals, cnt = np.unique(np.asarray(values, float), return_counts=True)
    return float(vals[cnt == cnt.max()].min()) if len(vals) else 0.


R3B_SETS = (('F_cell', 0), ('F_person', 0), ('F_hash', 0))


def run_R3b(c: Ctx):
    d, mc = c.d, c.cfg.meta
    nf = c.n_folds()
    kind = chosen_groups(c)
    for fs, r in R3B_SETS:
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            base = f'c3_{fs}_r{r}_f{k}_t0'
            f0 = c.store.fitted('R3', base)
            if f0 is None:
                raise RuntimeError('R3 must run before R3b')
            prior = fold_prior(c, f'{fs}_r{r}_f{k}', tr)
            for cv in mc.c_grid:
                if cv == 0.:
                    continue                                # c = 0 is the R3 fit (B21.3)
                c3_fit(c, 'R3b', f'c3meta_c{cv:g}_{fs}_r{r}_f{k}_t0', tr, te, kind, twin=0, seed_key=f'c3_{fs}_r{r}_f{k}',
                       prior=with_lambda(c, prior, cv, 'R3', base, tr), wd=f0.info['weight_decay'])
    sel = r3b_selection(c)
    for fs in ('F_cell', 'F_person'):                       # published-meta sensitivity at the modal selected c
        cv = sel['published_c'][fs]
        for k in range(nf):
            tr, te = c.fold_rows(fs, 0, k, c.A)
            base = f'c3_{fs}_r0_f{k}_t0'
            prior = fold_prior(c, f'{fs}_r0_f{k}', tr, 'published')
            c3_fit(c, 'R3b', f'c3pub_{fs}_r0_f{k}_t0', tr, te, kind, twin=0, seed_key=f'c3_{fs}_r0_f{k}',
                   prior=with_lambda(c, prior, cv, 'R3', base, tr), wd=c.store.fitted('R3', base).info['weight_decay'])
    metrics_R3b(c)


def r3b_selection(c: Ctx) -> dict:
    """Selected c per fold (inner-validation MSE of the c = 0 R3 fit and the R3b fits; tie -> smaller c) and the
    modal c per fold set (R3c; the published sensitivity uses the modal c, or the smallest non-zero c if 0)."""
    nf = c.n_folds()
    out = {'per_fold': {}, 'modal': {}, 'published_c': {}}
    for fs, r in R3B_SETS:
        chosen = []
        for k in range(nf):
            mse = {0.: c.store.fitted('R3', f'c3_{fs}_r{r}_f{k}_t0').info['inner_val_mse']}
            for cv in c.cfg.meta.c_grid:
                if cv:
                    mse[cv] = json.loads(c.store._p('R3b', f'c3meta_c{cv:g}_{fs}_r{r}_f{k}_t0').with_suffix('.json')
                                         .read_text())['info']['inner_val_mse']
            best = min(mse.values())
            chosen.append(min(cv for cv, v in mse.items() if v == best))
        out['per_fold'][fs] = chosen
        out['modal'][fs] = _mode_c(chosen)
        nz = [cv for cv in c.cfg.meta.c_grid if cv]
        out['published_c'][fs] = out['modal'][fs] if out['modal'][fs] else min(nz)
    out['final_c'] = _mode_c(sum(out['per_fold'].values(), []))
    return out


def _r3b_pred(c: Ctx, fs, sel, key):
    """(1, n) predictions of the per-fold selected c (c = 0 -> the R3 twin-0 fit)."""
    nf = c.n_folds()
    parts = []
    for k, cv in enumerate(sel['per_fold'][fs]):
        run, fid = ('R3', f'c3_{fs}_r0_f{k}_t0') if cv == 0. else ('R3b', f'c3meta_c{cv:g}_{fs}_r0_f{k}_t0')
        z = np.load(c.store._p(run, fid).with_suffix('.npz'))
        parts.append((z['rows'], z[key]))
    return stack_preds(c.d.n, [parts])


def r11_ranking_bars(c: Ctx, m3: dict, prefix='critic3') -> dict:
    """R11 ranking bars on A (B21.5) per fold set (F_cell, F_person); strict on F_cell repeat 0."""
    b = BARS['critic3_A_pooled']
    out = {}
    for fs in ('F_cell', 'F_person'):
        pa = m3[f'{prefix}_{fs}']['pooled']['pair_acc']
        vr = m3.get(f'{prefix}_minus_ridge_pairs_{fs}', {}).get('point')
        out[fs] = {'pair_acc_min': pa['point'] is not None and pa['point'] >= b['pair_acc_min'],
                   'pair_acc_ci_low_gt': pa['ci'][0] is not None and pa['ci'][0] > b['pair_acc_ci_low_gt'],
                   'vs_ridge_margin': vr is not None and vr >= b['vs_ridge_margin']}
    st = m3.get(f'{prefix}_strict_F_cell_r0', {}).get('point')
    out['strict_pair_acc_min'] = st is not None and st >= b['strict_pair_acc_min']
    out['all_met'] = bool(all(all(v.values()) for v in (out['F_cell'], out['F_person'])) and out['strict_pair_acc_min'])
    return out


def metrics_R3b(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    bcn = d.track_key == BARCELONA
    sel = r3b_selection(c)
    out = {'smoke': c.smoke, 'selection': sel, 'c_grid': list(c.cfg.meta.c_grid)}
    res = {}
    for fs, r in R3B_SETS:
        plain_s = collect(c, 'R3', [ids('c3', fs, 0, nf, '_t0')], 'setup')
        plain_f = collect(c, 'R3', [ids('c3', fs, 0, nf, '_t0')], 'full')
        sel_s, sel_f = _r3b_pred(c, fs, sel, 'setup'), _r3b_pred(c, fs, sel, 'full')
        tested = np.isfinite(plain_f).any(0)
        x = {}
        if fs != 'F_hash':
            x['delta_pair_acc_A'] = ev.pair_diff(sel_s, plain_s, tested)
            x['delta_pair_acc_barcelona'] = ev.pair_diff(sel_s, plain_s, tested & bcn)
            x['pair_acc_barcelona_b21'] = ev.pair_acc(sel_s, tested & bcn)
            x['pair_acc_barcelona_plain'] = ev.pair_acc(plain_s, tested & bcn)
            x['per_session_delta'] = {sn: ev.pair_diff(sel_s, plain_s, tested & (d.session == si))
                                      for si, sn in enumerate(d.sessions)}
            x['per_c'] = {}
            for cv in c.cfg.meta.c_grid:
                if cv:
                    s_c = collect(c, 'R3b', [ids(f'c3meta_c{cv:g}', fs, 0, nf, '_t0')], 'setup')
                    x['per_c'][f'{cv:g}'] = {'delta_pair_acc_A': ev.pair_diff(s_c, plain_s, tested),
                                             'delta_pair_acc_barcelona': ev.pair_diff(s_c, plain_s, tested & bcn)}
            pub = collect(c, 'R3b', [ids('c3pub', fs, 0, nf, '_t0')], 'setup')
            x['published'] = {'c': sel['published_c'][fs], 'delta_pair_acc_A_vs_plain': ev.pair_diff(pub, plain_s, tested),
                              'delta_pair_acc_A_vs_in_fold': ev.pair_diff(pub, sel_s, tested),
                              'delta_pair_acc_barcelona_vs_plain': ev.pair_diff(pub, plain_s, tested & bcn)}
        sst = c.sst(fs, c.A, [0])
        x['delta_r2_b21_minus_plain'] = ev.r2_diff(sel_f, plain_f, sst, tested)
        res[fs] = x
    out['b21_vs_plain'] = res
    # w table (F_cell repeat 0 folds, mean), Barcelona first; closeness of in-fold and published m (E21.3)
    ws, rb, close = [], [], []
    for k in range(nf):
        tr, _ = c.fold_rows('F_cell', 0, k, c.A)
        pr = fold_prior(c, f'F_cell_r0_f{k}', tr)
        pp = fold_prior(c, f'F_cell_r0_f{k}', tr, 'published')
        ws.append(pr.w)
        rb.append(pr.r_bar)
        close.append(closeness(pr.m, pp.m, d.p[tr].std(0)))
    w = np.mean(ws, 0)
    order = [d.track_names.index(BARCELONA)] + [i for i in range(len(d.track_names)) if d.track_names[i] != BARCELONA] \
        if BARCELONA in d.track_names else list(range(len(d.track_names)))
    out['w_table'] = {'knobs': list(KNOB_ORDER), 'rows': [[d.track_names[i]] + w[i].round(4).tolist() for i in order],
                      'r_bar': np.mean(rb, 0).round(4).tolist(), 'pooled_m': pr.info['pooled_m']}
    out['in_fold_vs_published_m'] = {'mean_distance_per_sd': float(np.mean([x['mean_distance_per_sd'] for x in close
                                                                            if 'mean_distance_per_sd' in x] or [np.nan])),
                                     'share_equal': float(np.mean([x['share_equal'] for x in close if 'share_equal' in x] or [np.nan]))}
    out['published_meta_note'] = ('sensitivity only, never delivered: f1laps publishes its meta as the top 100 setups '
                                  'ranked by lap time per track and game (half-steps consistent with a per-parameter '
                                  'median), i.e. computed from the labels incl. held-out folds (leak)')
    # B21.4 adoption, B21.5 trigger
    ba = BARS['b21_adoption']
    pt = lambda m: m.get('point')
    lo = lambda m: m.get('ci', [None])[0]
    a1 = lo(res['F_person']['delta_pair_acc_A']) is not None and lo(res['F_person']['delta_pair_acc_A']) > ba['F_person_delta_pair_ci_low_gt']
    a2 = pt(res['F_cell']['delta_pair_acc_A']) is not None and pt(res['F_cell']['delta_pair_acc_A']) >= ba['F_cell_delta_pair_point_ge']
    bdel = pt(res['F_cell']['delta_pair_acc_barcelona'])
    a3_evaluable = bdel is not None                 # QA m14: no Barcelona pairs -> not evaluable, treated as pass
    a3 = (not a3_evaluable) or bdel >= -ba['barcelona_max_drop_point']
    hdel = pt(res['F_hash']['delta_r2_b21_minus_plain'])
    a4 = hdel is not None and hdel >= -ba['F_hash_r2_gain_max_drop']
    out['adoption'] = {'F_person_ci_low_gt_0': bool(a1), 'F_cell_point_ge_0': bool(a2), 'barcelona_within_0.01': bool(a3),
                       'barcelona_item_evaluable': bool(a3_evaluable),
                       'F_hash_r2_within_0.005': bool(a4), 'adopted': bool(a1 and a2 and a3 and a4)}
    m3 = c.out / 'metrics' / 'R3.json'
    bars = r11_ranking_bars(c, json.loads(m3.read_text())) if m3.exists() else None
    out['trigger'] = {'r11_bars_plain': bars, 'triggered': bool(bars is not None and not bars['all_met'])}
    out['needs_R3c'] = bool(out['adoption']['adopted'] or out['trigger']['triggered'])
    if not out['needs_R3c']:
        out['delivered_critic3'] = 'plain'
    write_metrics(c, 'R3b', out)


def run_R3c(c: Ctx):
    """B21 at the modal selected c on the full R3 grid (only if B21 is triggered or adopted)."""
    m3b = c.out / 'metrics' / 'R3b.json'
    if not m3b.exists():
        raise RuntimeError('R3b must run before R3c')
    b = json.loads(m3b.read_text())
    if not b['needs_R3c']:
        write_metrics(c, 'R3c', {'smoke': c.smoke, 'needed': False, 'delivered_critic3': 'plain'})
        return
    nf = c.n_folds()
    kind = chosen_groups(c)
    for fs, r in c.fold_sets(('F_cell', 'F_person', 'F_hash')):
        cv = b['selection']['modal'][fs]
        if cv == 0.:
            continue                                        # B21 at c = 0 is the R3 fit (bit for bit)
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            prior = with_lambda(c, fold_prior(c, f'{fs}_r{r}_f{k}', tr), cv, 'R3', f'c3_{fs}_r{r}_f{k}_t0', tr)
            for twin in (0, 1):
                f0 = c.store.fitted('R3', f'c3_{fs}_r{r}_f{k}_t{twin}')
                c3_fit(c, 'R3c', f'c3B21_{fs}_r{r}_f{k}_t{twin}', tr, te, kind, twin=twin, seed_key=f'c3_{fs}_r{r}_f{k}',
                       prior=prior, wd=f0.info['weight_decay'])
            if r == 0 and fs == 'F_cell':                   # strict variant with the prior (R11 bars of both versions)
                strict = tr[~np.isin(c.d.setup_hash[tr], c.d.setup_hash[te])]
                f0 = c.store.fitted('R3', f'c3strict_{fs}_r0_f{k}_t0')
                ps = with_lambda(c, fold_prior(c, f'strict_{fs}_r0_f{k}', strict), cv, 'R3', f'c3strict_{fs}_r0_f{k}_t0', strict)
                c3_fit(c, 'R3c', f'c3B21strict_{fs}_r0_f{k}_t0', strict, te, kind, twin=0, seed_key=f'c3strict_{fs}_r0_f{k}',
                       prior=ps, wd=f0.info['weight_decay'])
    metrics_R3c(c, b)


def c3_source(c: Ctx, fs: str):
    """(run, prefix) of the delivered critic-3 fits of a fold set: R3c 'c3B21' iff B21 is delivered and its
    modal c for the fold set is non-zero; otherwise R3 'c3'. Smoke runs without R3b use plain."""
    m = c.out / 'metrics'
    dec = None
    for name in ('R3c', 'R3b'):
        if (m / f'{name}.json').exists():
            dec = json.loads((m / f'{name}.json').read_text()).get('delivered_critic3')
            if dec is not None:
                break
    if dec is None:
        if c.smoke or not (m / 'R3.json').exists():
            return 'R3', 'c3'
        raise RuntimeError('the delivered critic 3 is not decided yet: run R3b (and R3c) after R3')
    if dec == 'B21':
        cv = json.loads((m / 'R3b.json').read_text())['selection']['modal'].get(fs, 0.)
        if cv:
            return 'R3c', 'c3B21'
    return 'R3', 'c3'


def metrics_R3c(c: Ctx, b: dict):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    bcn = d.track_key == BARCELONA
    out = {'smoke': c.smoke, 'needed': True, 'modal_c': b['selection']['modal'], 'triggered': b['trigger']['triggered'],
           'adopted': b['adoption']['adopted']}
    tab = {}
    for fs in ('F_cell', 'F_person', 'F_hash'):
        reps = [r for f, r in c.fold_sets((fs,))]
        cv = b['selection']['modal'][fs]
        run, pre = ('R3c', 'c3B21') if cv else ('R3', 'c3')
        full, _, _ = twin_mean(c, run, pre, fs, reps, nf, 'full')
        setup, _, _ = twin_mean(c, run, pre, fs, reps, nf, 'setup')
        pfull, _, _ = twin_mean(c, 'R3', 'c3', fs, reps, nf, 'full')
        psetup, _, _ = twin_mean(c, 'R3', 'c3', fs, reps, nf, 'setup')
        nu = collect(c, 'R1', [ids('nuis', fs, r, nf) for r in reps])
        sst = c.sst(fs, c.A, reps)
        tested = np.isfinite(full).any(0)
        tab[f'critic3_{fs}'] = ev.block(full, setup, nu, sst, tested, pairs=fs != 'F_hash')
        if fs != 'F_hash':
            ras = collect(c, 'R1', [ids('ridgeA', fs, r, nf) for r in reps], 'setup')
            tab[f'critic3_minus_ridge_pairs_{fs}'] = ev.pair_diff(setup, ras, tested)
            tab[f'delta_pair_acc_A_{fs}'] = ev.pair_diff(setup, psetup, tested)
            tab[f'delta_pair_acc_barcelona_{fs}'] = ev.pair_diff(setup, psetup, tested & bcn)
        tab[f'delta_r2_{fs}'] = ev.r2_diff(full, pfull, sst, tested)
    if b['selection']['modal']['F_cell']:
        st = collect(c, 'R3c', [ids('c3B21strict', 'F_cell', 0, nf, '_t0')], 'setup')
    else:
        st = collect(c, 'R3', [ids('c3strict', 'F_cell', 0, nf, '_t0')], 'setup')
    tab['critic3_strict_F_cell_r0'] = ev.pair_acc(st, np.isfinite(st).any(0))
    out['b21_table'] = tab
    out['r11_bars'] = {'b21': r11_ranking_bars(c, tab), 'plain': b['trigger']['r11_bars_plain']}
    if b['trigger']['triggered']:
        bt = BARS['b21_triggered_delivery']
        lo = tab['delta_pair_acc_A_F_person'].get('ci', [None])[0]
        bp = tab['delta_pair_acc_barcelona_F_cell'].get('point')
        ok = lo is not None and lo > bt['F_person_delta_pair_ci_low_gt'] and (bp is None or bp >= -bt['barcelona_max_drop_point'])
        out['delivered_critic3'] = 'B21' if ok else 'plain'
        plain_ok = bool(out['r11_bars']['plain'] and out['r11_bars']['plain']['all_met'])
        b21_ok = bool(out['r11_bars']['b21']['all_met'])
        out['statement'] = ('passes with the in-fold meta prior' if (b21_ok and not plain_ok and ok) else
                            'critic 3 does not grade setups at the R11 level on game data' if not (b21_ok or plain_ok) else
                            'see r11_bars')
    else:
        out['delivered_critic3'] = 'B21'                    # adopted (B21.4)
    write_metrics(c, 'R3c', out)


# ---------------------------------------------------------------------------------------------------
# B22 sessions: R3d (S22a TT+Race-only critic 3, S22b TT-referenced session deviation)
# ---------------------------------------------------------------------------------------------------
def s22a_train(d, train, drop='Quali'):
    """S22a: the training rows without `drop` sessions (evaluation rows unchanged)."""
    train = np.asarray(train)
    return train[d.session[train] != list(d.sessions).index(drop)] if drop in d.sessions else train


def run_R3d(c: Ctx):
    d = c.d
    nf = c.n_folds()
    kind = chosen_groups(c)
    for fs in ('F_cell', 'F_person'):
        for k in range(nf):
            tr, te = c.fold_rows(fs, 0, k, c.A)
            c3_fit(c, 'R3d', f'c3noQ_{fs}_r0_f{k}_t0', s22a_train(d, tr, c.cfg.session.s22a_drop), te, kind, twin=0,
                   seed_key=f'c3_{fs}_r0_f{k}')
            c3_fit(c, 'R3d', f'c3sdev_{fs}_r0_f{k}_t0', tr, te, kind, twin=0, seed_key=f'c3_{fs}_r0_f{k}', session_dev=True)
    metrics_R3d(c)


def metrics_R3d(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    quali = d.session == list(d.sessions).index('Quali') if 'Quali' in d.sessions else np.zeros(d.n, bool)
    tt = d.session == list(d.sessions).index('TT') if 'TT' in d.sessions else np.zeros(d.n, bool)
    out = {'smoke': c.smoke, 'S22a': {}, 'S22b': {}}
    for fs in ('F_cell', 'F_person'):
        plain = collect(c, 'R3', [ids('c3', fs, 0, nf, '_t0')], 'setup')
        noq = collect(c, 'R3d', [ids('c3noQ', fs, 0, nf, '_t0')], 'setup')
        sdev = collect(c, 'R3d', [ids('c3sdev', fs, 0, nf, '_t0')], 'setup')
        sdev_dev = collect(c, 'R3d', [ids('c3sdev', fs, 0, nf, '_t0')], 'session_dev')
        tested = np.isfinite(plain).any(0)
        out['S22a'][fs] = {'delta_TT_Race_pairs': ev.pair_diff(noq, plain, tested & ~quali),
                           'delta_all_pairs': ev.pair_diff(noq, plain, tested),
                           'per_session': {sn: ev.pair_diff(noq, plain, tested & (d.session == si)) for si, sn in enumerate(d.sessions)}}
        out['S22b'][fs] = {'delta_TT_pairs_qTT_minus_plain': ev.pair_diff(sdev, plain, tested & tt),
                           'pair_acc_TT_qTT': ev.pair_acc(sdev, tested & tt),
                           'pair_acc_all_session_specific': ev.pair_acc(sdev + sdev_dev, tested),
                           'pair_acc_all_plain': ev.pair_acc(plain, tested)}
    a_p, a_c = out['S22a']['F_person']['delta_TT_Race_pairs'], out['S22a']['F_cell']['delta_TT_Race_pairs']
    out['S22a']['drop_quali'] = bool(a_p['ci'][0] is not None and a_p['ci'][0] > 0 and a_c['point'] is not None and a_c['point'] > 0)
    b_p = out['S22b']['F_person']['delta_TT_pairs_qTT_minus_plain']
    out['S22b']['escalate_TT_reference_decision'] = bool(b_p['ci'][0] is not None and b_p['ci'][0] > 0)
    infos = [json.loads(p.read_text())['info'] for p in sorted((c.out / 'fits' / 'R3d').glob('c3sdev_*.json'))]
    coefs = {sn: np.median([i['sdev_coef'][sn] for i in infos if sn in i.get('sdev_coef', {})], axis=0).tolist()
             for sn in c.cfg.session.s22b_sessions if infos}
    out['S22b']['deviation_coef_median'] = {'knobs': list(KNOB_ORDER), **coefs}
    out['session_facts'] = 'E22: Quali - TT level shift -1.2 % (median, same driver), Quali / Race lap noise ~2.3x TT in SD'
    write_metrics(c, 'R3d', out)


def on_support_mask(c: Ctx, run, fids):
    """Rows whose p lies inside the training support box of the fit that predicted them."""
    m = np.zeros(c.d.n, bool)
    for fid in fids:
        f = c.store.fitted(run, fid)
        z = np.load(c.store._p(run, fid).with_suffix('.npz'))
        rows = z['rows']
        lo, hi = f.support
        m[rows] = ((c.d.p[rows] >= lo) & (c.d.p[rows] <= hi)).all(1)
    return m


# ---------------------------------------------------------------------------------------------------
# R4 Barcelona heads, leave-out, Barcelona-only
# ---------------------------------------------------------------------------------------------------
def run_R4(c: Ctx):
    d, cfg = c.d, c.cfg
    nf = c.n_folds()
    bcn = d.track_key == BARCELONA
    kind = chosen_groups(c)
    for fs, r in c.fold_sets(('F_cell', 'F_person')):
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            for twin in (0, 1):
                run3, pre3 = c3_source(c, fs)
                base_id = f'{pre3}_{fs}_r{r}_f{k}_t{twin}'
                fid = f'bcn_{fs}_r{r}_f{k}_t{twin}'
                seed = T.seed_of('bcn', base_id)

                def fn(base_id=base_id, tr=tr, te=te, seed=seed, run3=run3):
                    f = c.store.fitted(run3, base_id)
                    head, info = T.fit_bcn_head(d, tr, f, cfg.bcn, seed=seed, smoke=c.smoke, salt=base_id)
                    f.bcn = head
                    rows = te[bcn[te]]
                    return preds_of(rows, T.predict(f, d, rows)), info, head
                run_fit(c, 'R4', fid, {'base': base_id, 'base_run': run3, 'seed': seed}, fn)
    # leave Barcelona out: all non-Barcelona rows A -> all Barcelona rows
    for twin in (0, 1):
        c3_fit(c, 'R4', f'c3leaveout_all_t{twin}', np.flatnonzero(~bcn), np.flatnonzero(bcn), kind, twin=twin, seed_key='c3leaveout')
    # Barcelona-only critic 3 (report): F_cell repeat 0, 1 twin
    for k in range(nf):
        tr, te = c.fold_rows('F_cell', 0, k, bcn)
        c3_fit(c, 'R4', f'c3bcnonly_F_cell_r0_f{k}_t0', tr, te, kind, twin=0, seed_key=f'c3bcnonly_{k}')
    metrics_R4(c)


def metrics_R4(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    bcn = d.track_key == BARCELONA
    out = {'smoke': c.smoke}
    for fs in ('F_cell', 'F_person'):
        reps = [r for f, r in c.fold_sets((fs,))]
        pooled, _, _ = twin_mean(c, *c3_source(c, fs), fs, reps, nf, 'setup')
        head, _, _ = twin_mean(c, 'R4', 'bcn', fs, reps, nf, 'setup')
        res = {'pooled': ev.pair_acc(pooled, bcn), 'pooled_plus_head': ev.pair_acc(head, bcn),
               'head_minus_pooled': ev.pair_diff(head, pooled, bcn)}
        if fs == 'F_cell':
            lo = (collect(c, 'R4', [['c3leaveout_all_t0']], 'setup') + collect(c, 'R4', [['c3leaveout_all_t1']], 'setup')) / 2
            only = collect(c, 'R4', [ids('c3bcnonly', 'F_cell', 0, nf, '_t0')], 'setup')
            res['leave_barcelona_out'] = ev.pair_acc(lo, bcn)
            res['barcelona_only_r0'] = ev.pair_acc(only, bcn)
            res['pooled_minus_leave_out'] = ev.pair_diff(pooled, lo, bcn)
            pooled_r0, _, _ = twin_mean(c, *c3_source(c, fs), fs, [0], nf, 'setup')
            res['pooled_r0_minus_barcelona_only_r0'] = ev.pair_diff(pooled_r0, only, bcn)
            hp, pp = res['pooled_plus_head']['point'], res['pooled']['point']
            deliver = 'pooled' if (hp is not None and pp is not None and pp - hp > BARS['barcelona']['head_vs_pooled_max_drop']) \
                else 'pooled_plus_head'
            res['delivered'] = deliver
            res['delivered_pair_acc'] = res['pooled_plus_head' if deliver == 'pooled_plus_head' else 'pooled']
            dp = res['delivered_pair_acc']['point']
            res['bars_judged'] = {'barcelona_pair_acc_ge_0.55': bool(dp is not None and dp >= BARS['barcelona']['pair_acc_min'])}
        out[f'barcelona_{fs}'] = res
    infos = [json.loads(p.read_text())['info'] for p in sorted((c.out / 'fits' / 'R4').glob('bcn_*.json'))]
    out['head_l2_chosen'] = {str(l): int(sum(i.get('l2') == l for i in infos)) for l in c.cfg.bcn.l2_grid}
    out['head_sd_pct_median'] = float(np.median([i['head_sd_pct'] for i in infos])) if infos else None
    write_metrics(c, 'R4', out)


# ---------------------------------------------------------------------------------------------------
# R5 critic 1, R6 ablation, R7 critic 2
# ---------------------------------------------------------------------------------------------------
def base_c3(c: Ctx, fs, r, k, twin):
    run3, pre3 = c3_source(c, fs)
    f = c.store.fitted(run3, f'{pre3}_{fs}_r{r}_f{k}_t{twin}')
    if f is None:
        raise RuntimeError('R3 (and the delivered critic 3) must run before R5 / R6 / R8')
    head = c.store.fitted('R4', f'bcn_{fs}_r{r}_f{k}_t{twin}') if barcelona_delivery(c) == 'pooled_plus_head' else None
    f.bcn = head                    # QA m1: the head only when R4 delivers it (as R9 does)
    return f, head is not None


def barcelona_delivery(c: Ctx) -> str:
    """R4's Barcelona decision ('pooled_plus_head' | 'pooled'); 'pooled_plus_head' before R4 has run."""
    r4 = c.out / 'metrics' / 'R4.json'
    return json.loads(r4.read_text())['barcelona_F_cell']['delivered'] if r4.exists() else 'pooled_plus_head'


def run_R5(c: Ctx):
    d, cfg = c.d, c.cfg
    nf = c.n_folds()
    tag = c.rows_tag
    for fs, r in c.fold_sets(('F_cell', 'F_person')):
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            etr, ete = c.fold_rows(fs, r, k, c.style_rows)
            for twin in (0, 1):
                f3, has_head = base_c3(c, fs, r, k, twin)
                fid = f'c1_{fs}_r{r}_f{k}_t{twin}{tag}'
                seed = T.seed_of('c1', fid)
                params = {'base': '{1}_{0}'.format(f'{fs}_r{r}_f{k}_t{twin}', c3_source(c, fs)[1]), 'head': has_head,
                          'seed': seed, 'tr': _sha(etr)}

                def fn(f3=f3, etr=etr, ete=ete, te=te, seed=seed, fid=fid, twin=twin, guard=True):
                    w = T.boot_weights(d.cell_key[etr], T.seed_of('boot_cell', fid))
                    f1 = T.finetune(d, etr, f3, 'critic1', cfg, seed=seed, weights=w, guard=guard, smoke=c.smoke, salt=fid)
                    pr = T.predict(f1, d, ete)
                    pa = T.predict(f1, d, te, e_zero=True, nuisance=False)          # B16 (iii): critic 1 at e = 0 on A
                    p3 = T.predict(f3, d, te, nuisance=False)       # QA M2: critic 3 with the same (delivered) head
                    f1.info['clipped_frac_test'] = pr['clipped_frac']
                    f1.info['m_style_coef_refit'] = f1.nuis.dense.detach().numpy()[-9:].tolist()
                    preds = {'rows': ete, 'full': pr['full'], 'setup': pr['setup'], 'rowsA': te, 'setupA_e0': pa['setup'],
                             'setupA_c3': p3['setup']}
                    return preds, f1.info, f1
                run_fit(c, 'R5', fid, params, fn)
                run_fit(c, 'R5', f'c3E_{fs}_r{r}_f{k}_t{twin}{tag}', {'base': params['base'], 'head': has_head, 'tr': _sha(etr)},
                        lambda f3=f3, etr=etr, ete=ete: (preds_of(ete, T.predict(T.finetune(d, etr, f3, 'c3_exact', cfg), d, ete)),
                                                         {}, None))
                if r == 0 and twin == 0:                         # unguarded critic-1 fine-tune (B12)
                    fid_u = f'c1unguarded_{fs}_r0_f{k}_t0{tag}'
                    run_fit(c, 'R5', fid_u, params | {'guard': False},
                            lambda f3=f3, etr=etr, ete=ete, seed=seed, fid=fid: _c1_unguarded(c, f3, etr, ete, seed, fid))
    metrics_R5(c)


def _c1_unguarded(c, f3, etr, ete, seed, fid):
    d = c.d
    w = T.boot_weights(d.cell_key[etr], T.seed_of('boot_cell', fid))
    f1 = T.finetune(d, etr, f3, 'critic1', c.cfg, seed=seed, weights=w, guard=False, smoke=c.smoke, salt=fid)
    return preds_of(ete, T.predict(f1, d, ete)), f1.info, None


# ---------------------------------------------------------------------------------------------------
# B23 remedy ladder: rung 2 (R5b, shrunk style path on exact rows) and the ladder decision
# ---------------------------------------------------------------------------------------------------
def critic1_bars(m5: dict, key='critic1') -> dict:
    """The 'Critic 1' bars on an R5-like metrics dict: delta R2 (c1 - c3) CI low > -0.02 on F_person and F_cell;
    |e = 0 pair accuracy on A - critic 3| <= 0.01 on both."""
    b = BARS['critic1_exact']
    out = {}
    for fs in b['fold_sets']:
        r = m5.get(f'{key}_{fs}')
        if r is None:
            return {'evaluated': False}
        lo = r['pooled']['delta_r2_c1_minus_c3']['ci'][0]
        e0 = r['e0_pair_acc_A_minus_critic3']['point']
        out[fs] = {'delta_r2_ci_low_ok': lo is not None and lo > b['delta_r2_ci_low_gt'],
                   'e0_ok': e0 is not None and abs(e0) <= b['e0_pair_acc_A_max_abs_diff']}
    out['evaluated'] = True
    out['passes'] = bool(all(v['delta_r2_ci_low_ok'] and v['e0_ok'] for k, v in out.items() if isinstance(v, dict)))
    return out


def remedy_ladder(m: dict) -> dict:
    """B23 (spec 12.21): TD3's Q is critic 1 at every rung; the first rung that passes the critic-1 bars is
    delivered, in the fixed order rung 0 (B9, exact rows) -> rung 1 (rows S, only if the B16 trigger fires)
    -> rung 2 (selected shrinkage level) -> rung 2 L3. L3 failing = implementation fault (stop, report)."""
    out = {'td3_q': 'critic1', 'order': list(BARS['b23']['rung_order']), 'tested': []}
    r5 = m.get('R5')
    if r5 is None:
        return out | {'rung': 'pending', 'needs': 'R5'}
    b0 = critic1_bars(r5)
    out['tested'].append({'rung0_B9_exact': b0})
    style_null = not (m.get('R6') or {}).get('style_helps_claim', False)
    note = ('style path present; effect not detectable on game data; to be fine-tuned on OCD/AC laps with '
            'controlled styles')
    if b0.get('passes'):
        return out | {'rung': 'rung0_B9_exact', 'status': note if style_null else 'style helps (F_person)'}
    trig = fallback_trigger(m)
    if trig.get('fires'):
        if m.get('R5_S') is None:
            return out | {'rung': 'pending', 'needs': 'R5 --rows S (B16 trigger fired)'}
        b1 = critic1_bars(m['R5_S'])
        out['tested'].append({'rung1_rows_S': b1})
        if b1.get('passes'):
            return out | {'rung': 'rung1_rows_S', 'status': note}
    r5b = m.get('R5b')
    if r5b is None or not r5b.get('needed', True):
        return out | {'rung': 'pending', 'needs': 'R5b'}
    bs, b3 = r5b['bars_selected'], r5b['bars_L3']
    out['tested'] += [{'rung2_selected_level': bs}, {'rung2_L3': b3}]
    if bs.get('passes'):
        return out | {'rung': 'rung2_selected_level', 'status': note}
    if b3.get('passes'):
        return out | {'rung': 'rung2_L3', 'status': note}
    return out | {'rung': 'implementation_fault',
                  'status': 'L3 fails the critic-1 bars: implementation fault (B23); stop and report. Critic 3 is never Q.'}


LEVEL_ORDER = ('L1', 'L2', 'L3')        # rung-2 selection set (architect ruling 30): B9 is never a rung-2 candidate


def run_R5b(c: Ctx):
    d, cfg = c.d, c.cfg
    m5 = c.out / 'metrics' / 'R5.json'
    if not m5.exists():
        raise RuntimeError('R5 must run before R5b')
    if critic1_bars(json.loads(m5.read_text())).get('passes'):
        write_metrics(c, 'R5b', {'smoke': c.smoke, 'needed': False, 'reason': 'rung 0 passes'})
        return
    nf = c.n_folds()
    for fs, r in c.fold_sets(('F_cell', 'F_person')):
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            etr, ete = c.fold_rows(fs, r, k, c.E)
            for twin in (0, 1):
                f3, has_head = base_c3(c, fs, r, k, twin)
                rid = f'c1_{fs}_r{r}_f{k}_t{twin}'          # the rung-0 fit: same seed, bootstrap, inner split
                base = '{1}_{0}'.format(f'{fs}_r{r}_f{k}_t{twin}', c3_source(c, fs)[1])
                for lv in cfg.ladder.levels:
                    fid = f'c1{lv[0]}_{fs}_r{r}_f{k}_t{twin}'
                    params = {'base': base, 'head': has_head, 'level': list(lv), 'tr': _sha(etr), 'rung0': rid}

                    def fn(f3=f3, etr=etr, ete=ete, te=te, rid=rid, lv=lv):
                        w = T.boot_weights(d.cell_key[etr], T.seed_of('boot_cell', rid))
                        f1 = T.finetune(d, etr, f3, 'critic1', cfg, seed=T.seed_of('c1', rid), weights=w, smoke=c.smoke,
                                        salt=rid, level=lv)
                        pr = T.predict(f1, d, ete)
                        pa = T.predict(f1, d, te, e_zero=True, nuisance=False)
                        p3 = T.predict(f3, d, te, nuisance=False)
                        return {'rows': ete, 'full': pr['full'], 'setup': pr['setup'], 'rowsA': te, 'setupA_e0': pa['setup'],
                                'setupA_c3': p3['setup']}, f1.info, f1
                    run_fit(c, 'R5b', fid, params, fn)
                    # QA M3 / architect must-change 1: the matched comparator (critic 3 + exact-row nuisance at the
                    # level's alpha of m(s~)); the alpha = 10 comparator of R5 stays report-only
                    run_fit(c, 'R5b', f'c3E{lv[0]}_{fs}_r{r}_f{k}_t{twin}', {'base': base, 'head': has_head, 'tr': _sha(etr),
                                                                              'alpha_m_s': lv[3]},
                            lambda f3=f3, etr=etr, ete=ete, lv=lv: _c3E_level(c, f3, etr, ete, lv))
    metrics_R5b(c)


def _c3E_level(c, f3, etr, ete, lv):
    cfg = replace(c.cfg, nuisance=replace(c.cfg.nuisance, alpha_style=float(lv[3])))
    f = T.finetune(c.d, etr, f3, 'c3_exact', cfg)
    return preds_of(ete, T.predict(f, c.d, ete)), {'mode': 'c3_exact', 'level': lv[0], 'alpha_m_s': float(lv[3])}, None


def _level_fit(c, name, fs, r, k, twin):
    return 'R5b', f'c1{name}_{fs}_r{r}_f{k}_t{twin}'


def r5b_selection(c: Ctx) -> dict:
    """Per fit: the level among (L1, L2, L3) with the lowest inner-validation MSE (tie -> earlier level)."""
    nf = c.n_folds()
    out = {}
    for fs, r in c.fold_sets(('F_cell', 'F_person')):
        for k in range(nf):
            for twin in (0, 1):
                mse = [json.loads(c.store._p(*_level_fit(c, n, fs, r, k, twin)).with_suffix('.json').read_text())['info']
                       ['inner_val_mse'] for n in LEVEL_ORDER]
                out[f'{fs}_r{r}_f{k}_t{twin}'] = LEVEL_ORDER[int(np.argmin(mse))]
    return out


def _collect_levels(c, fs, reps, choose, key, rows_key='rows', comparator=False):
    """Twin-mean (1 x R, n) predictions of the per-fit chosen level (`comparator`: its matched c3E{L})."""
    nf = c.n_folds()
    tw = []
    for twin in (0, 1):
        items = []
        for r in reps:
            parts = []
            for k in range(nf):
                name = choose(fs, r, k, twin)
                fid = f'c3E{name}_{fs}_r{r}_f{k}_t{twin}' if comparator else _level_fit(c, name, fs, r, k, twin)[1]
                z = np.load(c.store._p('R5b', fid).with_suffix('.npz'))
                parts.append((z[rows_key], z[key]))
            items.append(parts)
        tw.append(stack_preds(c.d.n, items))
    return (tw[0] + tw[1]) / 2


def metrics_R5b(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    sel = r5b_selection(c)
    out = {'smoke': c.smoke, 'needed': True, 'levels': [list(lv) for lv in c.cfg.ladder.levels],
           'selection_set': list(LEVEL_ORDER), 'selected_counts': {n: int(sum(v == n for v in sel.values())) for n in LEVEL_ORDER},
           'comparator': BARS['b23']['comparator']}
    variants = {'selected': lambda fs, r, k, t: sel[f'{fs}_r{r}_f{k}_t{t}'], 'L1': lambda *a: 'L1', 'L2': lambda *a: 'L2',
                'L3': lambda *a: 'L3'}
    blocks, alpha10 = {}, {}
    for vname, choose in variants.items():
        res, res10 = {}, {}
        for fs in ('F_cell', 'F_person'):
            reps = [r for f, r in c.fold_sets((fs,))]
            c1 = _collect_levels(c, fs, reps, choose, 'full')
            c3m = _collect_levels(c, fs, reps, choose, 'full', comparator=True)        # matched alpha of m(s~)
            c3_10 = (collect(c, 'R5', [ids('c3E', fs, r, nf, '_t0') for r in reps]) +
                     collect(c, 'R5', [ids('c3E', fs, r, nf, '_t1') for r in reps])) / 2
            sste = c.sst(fs, c.E, reps)
            te = np.isfinite(c1).any(0)
            e0 = _collect_levels(c, fs, reps, choose, 'setupA_e0', rows_key='rowsA')
            c3a = _collect_levels(c, fs, reps, choose, 'setupA_c3', rows_key='rowsA')
            res[f'critic1_{fs}'] = {'pooled': {'delta_r2_c1_minus_c3': ev.r2_diff(c1, c3m, sste, te)},
                                    'e0_pair_acc_A_minus_critic3': ev.pair_diff(e0, c3a, np.isfinite(e0).any(0))}
            res10[fs] = {'delta_r2_c1_minus_c3_alpha10': ev.r2_diff(c1, c3_10, sste, te),
                         'credit_of_the_alpha_change': ev.r2_diff(c3m, c3_10, sste, te)}
        blocks[vname], alpha10[vname] = res, res10
    out['variants'] = blocks
    out['report_only_vs_alpha10_comparator'] = alpha10
    out['bars_selected'] = critic1_bars(blocks['selected'])
    out['bars_L3'] = critic1_bars(blocks['L3'])
    out['delta_r2_trend_L1_to_L3'] = {v: {fs: blocks[v][f'critic1_{fs}']['pooled']['delta_r2_c1_minus_c3']['point']
                                          for fs in ('F_cell', 'F_person')} for v in ('L1', 'L2', 'L3')}
    out['critic2_note'] = ('architect ruling 29 (B23 amended): in C2b critic 2 gets rungs 0-1 only; its rung 2 is deferred '
                           'to the OCD/AC stage; if it fails rungs 0-1 it is a negative result and screening uses critic 1')
    write_metrics(c, 'R5b', out)


def assist_free(d):
    """F1 26 rows with ABS off and TC off (B14 / R5 report)."""
    a = d.assists
    off = lambda name: np.array([lv == 'Off' for lv in np.array(d.assist_levels[name], object)[a[name]]])
    g26 = np.array([g == 'F1 26' for g in np.array(d.games, object)[d.game]])
    return g26 & off('anti_lock_brakes') & off('traction_control')


def metrics_R5(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    tag = c.rows_tag
    out = {'smoke': c.smoke, 'rows': c.args.rows,
           'exact_without_style_excluded': int((d.exact & ~d.has_style).sum())}
    af = assist_free(d)
    fallback = {}
    for fs in ('F_cell', 'F_person'):
        reps = [r for f, r in c.fold_sets((fs,))]
        c1 = (collect(c, 'R5', [ids('c1', fs, r, nf, f'_t0{tag}') for r in reps]) +
              collect(c, 'R5', [ids('c1', fs, r, nf, f'_t1{tag}') for r in reps])) / 2
        c3 = (collect(c, 'R5', [ids('c3E', fs, r, nf, f'_t0{tag}') for r in reps]) +
              collect(c, 'R5', [ids('c3E', fs, r, nf, f'_t1{tag}') for r in reps])) / 2
        nue = collect(c, 'R1', [ids('nuisE', fs, r, nf, tag) for r in reps])
        sste = c.sst(fs, c.style_rows, reps)
        te = np.isfinite(c1).any(0)
        res = {}
        for name, m in c.ev.subsets(te).items():
            if m.any():
                res[name] = {'delta_r2_c1_minus_c3': ev.r2_diff(c1, c3, sste, m), 'r2_gain_c1': ev.r2_gain(c1, nue, sste, m),
                             'r2_gain_c3': ev.r2_gain(c3, nue, sste, m), 'mae_c1': ev.mae(c1, m), 'mae_c3': ev.mae(c3, m),
                             'mean_residual_c1': ev.ratio(_avg_rows(d.gap - c1)[m], np.ones(m.sum()), d.person[m]),
                             'mean_residual_c3': ev.ratio(_avg_rows(d.gap - c3)[m], np.ones(m.sum()), d.person[m])}
        if (te & af).any():
            res['assist_free_F1_26'] = {'delta_r2_c1_minus_c3': ev.r2_diff(c1, c3, sste, te & af), 'n_rows': int((te & af).sum())}
        if c.args.rows == 'S':
            c1s = (collect(c, 'R5', [ids('c1', fs, r, nf, f'_t0{tag}') for r in reps], 'setup') +
                   collect(c, 'R5', [ids('c1', fs, r, nf, f'_t1{tag}') for r in reps], 'setup')) / 2
            res['pair_acc_S'] = ev.pair_acc(c1s, te)
        # B16 (iii): critic 1 at e = 0 vs critic 3 on rows A pairs (twin-wise, then mean)
        e0 = (collect(c, 'R5', [ids('c1', fs, r, nf, f'_t0{tag}') for r in reps], 'setupA_e0', rows_key='rowsA') +
              collect(c, 'R5', [ids('c1', fs, r, nf, f'_t1{tag}') for r in reps], 'setupA_e0', rows_key='rowsA')) / 2
        c3a = (collect(c, 'R5', [ids('c1', fs, r, nf, f'_t0{tag}') for r in reps], 'setupA_c3', rows_key='rowsA') +
               collect(c, 'R5', [ids('c1', fs, r, nf, f'_t1{tag}') for r in reps], 'setupA_c3', rows_key='rowsA')) / 2
        teA = np.isfinite(e0).any(0)
        res['e0_pair_acc_A_minus_critic3'] = ev.pair_diff(e0, c3a, teA)
        out[f'critic1_{fs}'] = res
        dr2 = res['pooled']['delta_r2_c1_minus_c3']
        fallback[fs] = {'i_delta_r2_point': dr2['point'], 'iii_e0_pair_diff': res['e0_pair_acc_A_minus_critic3']['point']}
    # guard comparison for critic 1 (F_cell / F_person repeat 0, twin 0)
    gc = {}
    for fs in ('F_cell', 'F_person'):
        g = collect(c, 'R5', [ids('c1', fs, 0, nf, f'_t0{tag}')])
        u = collect(c, 'R5', [ids('c1unguarded', fs, 0, nf, f'_t0{tag}')])
        sste = c.sst(fs, c.style_rows, [0])
        gc[fs] = {'r2_guarded_minus_unguarded': ev.r2_diff(g, u, sste, np.isfinite(g).any(0))}
    out['guard_comparison'] = gc
    infos = [json.loads(p.read_text())['info'] for p in sorted((c.out / 'fits' / 'R5').glob(f'c1_*{tag}.json'))]
    out['l2_chosen'] = {str(l): int(sum(i.get('l2') == l for i in infos)) for l in c.cfg.finetune.l2_grid}
    out['W1e_norm_median'] = float(np.median([i['W1e_norm'] for i in infos])) if infos else None
    out['e_clipped_frac_test_mean'] = float(np.mean([i['clipped_frac_test'] for i in infos])) if infos else None
    out['m_style_coef_median'] = np.median([i['m_style_coef_refit'] for i in infos], axis=0).tolist() if infos else None
    out['fallback_inputs'] = fallback
    write_metrics(c, 'R5' + tag, out)


def _avg_rows(a):
    a = np.atleast_2d(a)
    with warnings.catch_warnings():         # all-NaN columns (a metric undefined in every repeat) stay NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=0)


def strata_codes(d, scheme: str) -> np.ndarray:
    """B14 shuffle strata per row. 'track': game x track x controller x ABS x TC x braking-assist on/off
    (F1 26), game x track x controller (F1 25). 'cross': the same without the track."""
    a = d.assists
    abs_, tc = a[ASSISTS_ABS_TC[0]], a[ASSISTS_ABS_TC[1]]
    levels = np.array(d.assist_levels[BRAKING_ASSIST], object)
    brk = np.array([0 if lv == 'unknown' else (1 if lv == 'Off' else 2) for lv in levels[a[BRAKING_ASSIST]]])
    f126 = np.array([g == 'F1 26' for g in np.array(d.games, object)[d.game]])
    track = d.track if scheme == 'track' else np.zeros(d.n, int)
    zero = np.zeros(d.n, int)
    return np.stack([d.game, track, d.controller, np.where(f126, abs_, zero), np.where(f126, tc, zero),
                     np.where(f126, brk, zero)], 1)


def run_R6(c: Ctx):
    d, cfg = c.d, c.cfg
    nf = c.n_folds()
    tag = c.rows_tag
    for scheme in ('track', 'cross'):
        strata = strata_codes(d, scheme)
        for fs, r in c.fold_sets(('F_cell', 'F_person')):
            for k in range(nf):
                etr, ete = c.fold_rows(fs, r, k, c.style_rows)
                real_id = f'c1_{fs}_r{r}_f{k}_t0{tag}'
                f_real = c.store.fitted('R5', real_id)
                if f_real is None:
                    raise RuntimeError('R5 must run before R6')
                lam = f_real.info['l2']
                f3, has_head = base_c3(c, fs, r, k, 0)
                for sd in cfg.eval.shuffle_seeds:
                    fid = f'c1shuf{scheme}_{fs}_r{r}_f{k}_s{sd}{tag}'
                    params = {'real': real_id, 'l2': lam, 'scheme': scheme, 'seed': sd, 'head': has_head,
                              'base_src': c3_source(c, fs)[1]}
                    run_fit(c, 'R6', fid, params, lambda f3=f3, etr=etr, ete=ete, sd=sd, lam=lam, real_id=real_id, strata=strata:
                            _c1_shuffled(c, f3, etr, ete, strata, sd, lam, real_id))
    metrics_R6(c)


def _shuffled_s(c, rows, f_style, strata, seed):
    d = c.d
    s = np.zeros((d.n, 9))
    s[rows] = T.style_inputs(d, rows, f_style)[0]
    sh = strata_shuffle(s, strata, rows, seed)
    return sh[rows]


def _c1_shuffled(c, f3, etr, ete, strata, sd, lam, real_id):
    d = c.d
    from rlsetup.critic.style_transform import TrackStyleTransform
    tf = TrackStyleTransform().fit(d.style_raw, d.track_key, etr)
    s_tr = _shuffled_s(c, etr, tf, strata, T.seed_of('shuf_tr', real_id, sd))
    s_te = _shuffled_s(c, ete, tf, strata, T.seed_of('shuf_te', real_id, sd))
    w = T.boot_weights(d.cell_key[etr], T.seed_of('boot_cell', real_id))           # the real twin 0's bootstrap
    f = T.finetune(d, etr, f3, 'critic1', c.cfg, seed=T.seed_of('c1', real_id), weights=w, smoke=c.smoke, salt=real_id,
                   s_head=s_tr, l2=lam)
    pr = T.predict(f, d, ete, s_head=s_te)
    info = f.info | {'singleton_rows_train': singleton_rows(strata, etr), 'singleton_rows_test': singleton_rows(strata, ete)}
    return preds_of(ete, pr), info, None


def metrics_R6(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    tag = c.rows_tag
    out = {'smoke': c.smoke, 'mde_note': 'MDE about 0.04 (F_person) and 0.02 (F_cell) per the addendum; a null result is expected'}
    for scheme in ('track', 'cross'):
        strata = strata_codes(d, scheme)
        res = {'rows_in_singleton_strata': singleton_rows(strata, np.flatnonzero(c.style_rows))}
        for fs in ('F_cell', 'F_person'):
            reps = [r for f, r in c.fold_sets((fs,))]
            real = collect(c, 'R5', [ids('c1', fs, r, nf, f'_t0{tag}') for r in reps])
            sse_sh = np.mean([c.ev.sq(collect(c, 'R6', [ids(f'c1shuf{scheme}', fs, r, nf, f'_s{s}{tag}') for r in reps]))
                              for s in c.cfg.eval.shuffle_seeds], axis=0)        # loss averaged over the shuffle seeds
            sste = c.sst(fs, c.style_rows, reps)
            te = np.isfinite(real).any(0)
            r2 = {}
            for name, m in ev.subsets(te).items():
                if m.any():
                    num = _avg_rows(sse_sh - c.ev.sq(real))[m]
                    r2[name] = ev.ratio(num, _avg_rows(sste)[m], d.person[m])
            res[fs] = {'delta_r2_real_minus_shuffle': r2}
        out[scheme] = res
    pt = out['track']['F_person']['delta_r2_real_minus_shuffle'].get('pooled', {})
    out['style_helps_claim'] = bool(pt.get('ci', [None])[0] is not None and pt['ci'][0] > BARS['style_helps']['claim_if_ci_low_gt'])
    write_metrics(c, 'R6' + tag, out)


def run_R7(c: Ctx):
    d, cfg = c.d, c.cfg
    nf = c.n_folds()
    tag = c.rows_tag
    strata = strata_codes(d, 'track')
    for fs, r in c.fold_sets(('F_cell', 'F_person')):
        for k in range(nf):
            tr, te = c.fold_rows(fs, r, k, c.A)
            etr, ete = c.fold_rows(fs, r, k, c.style_rows)
            pid = f'c2pre_{fs}_r{r}_f{k}'
            seed = T.seed_of('c2', pid)

            pri = delivered_prior(c, fs, r, k)              # B21.5: critic 2 inherits the delivered Pi frozen

            def fn_pre(tr=tr, te=te, seed=seed, pid=pid, pri=pri):
                f = T.pretrain(d, tr, 'critic2', None, cfg, seed=seed, smoke=c.smoke, salt=pid, prior=pri)
                head, hinfo = T.fit_bcn_head(d, tr, f, cfg.bcn, seed=seed, smoke=c.smoke, salt=pid)
                f.bcn = head
                f.info['bcn_head'] = hinfo
                return preds_of(te, T.predict(f, d, te)), f.info, f
            run_fit(c, 'R7', pid, {'seed': seed, 'tr': _sha(tr), 'prior': None if pri is None else [pri.c, pri.lam]}, fn_pre)
            f2p = c.store.fitted('R7', pid)
            run_fit(c, 'R7', f'c2_{fs}_r{r}_f{k}{tag}', {'base': pid, 'tr': _sha(etr)},
                    lambda f2p=f2p, etr=etr, ete=ete, seed=seed, pid=pid: _c2_ft(c, f2p, etr, ete, seed, pid))
            run_fit(c, 'R7', f'c2preE_{fs}_r{r}_f{k}{tag}', {'base': pid, 'tr': _sha(etr)},
                    lambda f2p=f2p, etr=etr, ete=ete: (preds_of(ete, T.predict(T.finetune(d, etr, f2p, 'c2pre_exact', cfg), d, ete)),
                                                       {}, None))
            for sd in cfg.eval.shuffle_seeds:
                real_id = f'c2_{fs}_r{r}_f{k}{tag}'
                run_fit(c, 'R7', f'c2shuf_{fs}_r{r}_f{k}_s{sd}{tag}', {'base': pid, 'seed': sd},
                        lambda f2p=f2p, etr=etr, ete=ete, sd=sd, real_id=real_id: _c2_shuffled(c, f2p, etr, ete, strata, sd, real_id))
    metrics_R7(c)


def _c2_ft(c, f2p, etr, ete, seed, pid):
    d = c.d
    f = T.finetune(d, etr, f2p, 'critic2', c.cfg, seed=seed, smoke=c.smoke, salt=pid)
    return preds_of(ete, T.predict(f, d, ete)), f.info, f


def _c2_shuffled(c, f2p, etr, ete, strata, sd, real_id):
    d = c.d
    from rlsetup.critic.style_transform import TrackStyleTransform
    f_real = c.store.fitted('R7', real_id)
    tf = TrackStyleTransform().fit(d.style_raw, d.track_key, etr)
    s_tr = _shuffled_s(c, etr, tf, strata, T.seed_of('shuf_tr', real_id, sd))
    s_te = _shuffled_s(c, ete, tf, strata, T.seed_of('shuf_te', real_id, sd))
    f = T.finetune(d, etr, f2p, 'critic2', c.cfg, seed=T.seed_of('c2', real_id), smoke=c.smoke, salt=real_id, s_head=s_tr,
                   l2=f_real.info['l2'])
    return preds_of(ete, T.predict(f, d, ete, s_head=s_te)), f.info, None


def metrics_R7(c: Ctx):
    d, ev = c.d, c.ev
    nf = c.n_folds()
    tag = c.rows_tag
    out = {'smoke': c.smoke, 'F_person_miss_expected': True}
    fallback = {}
    for fs in ('F_cell', 'F_person'):
        reps = [r for f, r in c.fold_sets((fs,))]
        c2 = collect(c, 'R7', [ids('c2', fs, r, nf, tag) for r in reps])
        c2pe = collect(c, 'R7', [ids('c2preE', fs, r, nf, tag) for r in reps])
        nue = collect(c, 'R1', [ids('nuisE', fs, r, nf, tag) for r in reps])
        sste = c.sst(fs, c.style_rows, reps)
        te = np.isfinite(c2).any(0)
        res = {}
        for name, m in ev.subsets(te).items():
            if m.any():
                res[name] = {'mae_c2': ev.mae(c2, m), 'mae_c2_s': ev.mae(c2, m, True), 'mae_nuis': ev.mae(nue, m),
                             'mae_c2_minus_nuis': ev.mae_diff(c2, nue, m), 'r2_gain_c2': ev.r2_gain(c2, nue, sste, m),
                             'delta_r2_c2_minus_c2pre': ev.r2_diff(c2, c2pe, sste, m), 'spearman_within': ev.spearman(c2, m),
                             'mean_residual_c2': ev.ratio(_avg_rows(d.gap - c2)[m], np.ones(m.sum()), d.person[m])}
        sse_sh = np.mean([ev.sq(collect(c, 'R7', [ids('c2shuf', fs, r, nf, f'_s{s}{tag}') for r in reps]))
                          for s in c.cfg.eval.shuffle_seeds], axis=0)
        res['delta_r2_real_minus_shuffle'] = ev.ratio(_avg_rows(sse_sh - ev.sq(c2))[te], _avg_rows(sste)[te], d.person[te])
        # grouping cost: A-pretrained critic 2 vs critic 3 on A (report only)
        c2a = collect(c, 'R7', [ids('c2pre', fs, r, nf) for r in reps])
        c2as = collect(c, 'R7', [ids('c2pre', fs, r, nf) for r in reps], 'setup')
        c3a, c3a0, _ = twin_mean(c, *c3_source(c, fs), fs, reps, nf, 'full')
        c3s, c3s0, _ = twin_mean(c, *c3_source(c, fs), fs, reps, nf, 'setup')
        nu = collect(c, 'R1', [ids('nuis', fs, r, nf) for r in reps])
        sst = c.sst(fs, c.A, reps)
        teA = np.isfinite(c2a).any(0)
        res['grouping_cost_A'] = {'critic2_pre': ev.block(c2a, c2as, nu, sst, teA),
                                  'c2pre_minus_c3_twin_mean_pairs': ev.pair_diff(c2as, c3s, teA),
                                  'c2pre_minus_c3_twin0_pairs': ev.pair_diff(c2as, c3s0, teA),
                                  'c2pre_minus_c3_twin_mean_r2': ev.r2_diff(c2a, c3a, sst, teA)}
        hi = res['pooled']['mae_c2_minus_nuis']['ci'][1]
        res['bars_judged'] = {'mae_below_nuisance_ci_excludes_0': bool(hi is not None and hi < 0)}   # QA m3
        out[f'critic2_{fs}'] = res
        fallback[fs] = {'ii_delta_r2_point': res['pooled']['delta_r2_c2_minus_c2pre']['point']}
    out['fallback_inputs'] = fallback
    write_metrics(c, 'R7' + tag, out)


# ---------------------------------------------------------------------------------------------------
# R8 sensitivity: Imola and Las Vegas dropped
# ---------------------------------------------------------------------------------------------------
def run_R8(c: Ctx):
    d, cfg = c.d, c.cfg
    nf = c.n_folds()
    keep = ~np.isin(d.track_key, SENS_DROP)
    kind = chosen_groups(c)
    for k in range(nf):
        tr, te = c.fold_rows('F_cell', 0, k, c.A & keep)
        c3_fit(c, 'R8', f'c3sens_F_cell_r0_f{k}_t0', tr, te, kind, twin=0, seed_key=f'c3_F_cell_r0_f{k}')
        f3 = c.store.fitted('R8', f'c3sens_F_cell_r0_f{k}_t0')
        etr, ete = c.fold_rows('F_cell', 0, k, c.E & keep)
        fid = f'c1sens_F_cell_r0_f{k}_t0'

        def fn(f3=f3, etr=etr, ete=ete, fid=fid):
            f1 = T.finetune(d, etr, f3, 'critic1', cfg, seed=T.seed_of('c1', fid), smoke=c.smoke, salt=fid)
            c3e = T.finetune(d, etr, f3, 'c3_exact', cfg)
            p1, p3 = T.predict(f1, d, ete), T.predict(c3e, d, ete)
            return {'rows': ete, 'full': p1['full'], 'setup': p1['setup'], 'c3full': p3['full']}, f1.info, None
        run_fit(c, 'R8', fid, {'base': f'c3sens_F_cell_r0_f{k}_t0'}, fn)
    ev = c.ev
    s3 = collect(c, 'R8', [ids('c3sens', 'F_cell', 0, nf, '_t0')], 'setup')
    ref = collect(c, 'R3', [ids('c3', 'F_cell', 0, nf, '_t0')], 'setup')
    m = np.isfinite(s3).any(0)
    c1 = collect(c, 'R8', [ids('c1sens', 'F_cell', 0, nf, '_t0')])
    c3e = collect(c, 'R8', [ids('c1sens', 'F_cell', 0, nf, '_t0')], 'c3full')
    sste = c.sst('F_cell', c.E & keep, [0])
    out = {'smoke': c.smoke, 'dropped_tracks': list(SENS_DROP),
           'critic3_pairs_dropped': {n: ev.pair_acc(s3, mm) for n, mm in ev.subsets(m).items() if mm.any()},
           'critic3_pairs_reference_same_rows': {n: ev.pair_acc(ref, mm) for n, mm in ev.subsets(m).items() if mm.any()},
           'critic1_delta_r2_c1_minus_c3': ev.r2_diff(c1, c3e, sste, np.isfinite(c1).any(0))}
    write_metrics(c, 'R8', out)


# ---------------------------------------------------------------------------------------------------
# R9 final fits -> bundles
# ---------------------------------------------------------------------------------------------------
def ocd_driver_styles():
    """style-rl's OCD driver styles (targets/style12.json at the pinned commit), or None."""
    r = subprocess.run(['git', '-C', str(STYLE_RL_REPO), 'show', f'{STYLE_RL_COMMIT}:targets/style12.json'],
                       capture_output=True, check=False)
    if r.returncode:
        return None, None
    doc = json.loads(r.stdout)
    return doc['drivers'], hashlib.sha256(r.stdout).hexdigest()


def run_R9(c: Ctx):
    d, cfg = c.d, c.cfg
    allA = np.flatnonzero(c.A)
    allE = np.flatnonzero(c.E)
    kind = chosen_groups(c)
    r4 = c.out / 'metrics' / 'R4.json'
    deliver = json.loads(r4.read_text())['barcelona_F_cell']['delivered'] if r4.exists() else 'pooled_plus_head'
    variant = delivered_variant(c)
    for twin in (0, 1):
        c3_fit(c, 'R9', f'c3final_t{twin}', allA, allA[:0], kind, twin=twin, seed_key='c3final')
    pre = 'c3final'
    final_c = None
    if variant == 'B21':                                    # B21: the in-fold prior on all rows A (B21.1)
        final_c = json.loads((c.out / 'metrics' / 'R3b.json').read_text())['selection']['final_c']
        if final_c:
            prior = with_lambda(c, fold_prior(c, 'final', allA), final_c, 'R9', 'c3final_t0', allA)
            for twin in (0, 1):
                c3_fit(c, 'R9', f'c3B21final_t{twin}', allA, allA[:0], kind, twin=twin, seed_key='c3final', prior=prior,
                       wd=c.store.fitted('R9', f'c3final_t{twin}').info['weight_decay'])
            pre = 'c3B21final'
    c3s = []
    for twin in (0, 1):
        f = c.store.fitted('R9', f'{pre}_t{twin}')
        if deliver == 'pooled_plus_head':
            run_fit(c, 'R9', f'bcnfinal_t{twin}', {'base': f'{pre}_t{twin}'},
                    lambda f=f, twin=twin: (preds_of(allA[:0], {'full': [], 'setup': []}),
                                            *_head_only(c, allA, f, T.seed_of('bcnfinal', twin))))
            f.bcn = c.store.fitted('R9', f'bcnfinal_t{twin}')
        c3s.append(f)
    ladder = remedy_ladder({p.stem: json.loads(p.read_text()) for p in (c.out / 'metrics').glob('*.json')})
    rung = ladder['rung']
    if rung in ('pending', 'implementation_fault'):
        if not c.smoke:
            raise RuntimeError(f"B23 remedy ladder: {rung} ({ladder.get('needs') or ladder.get('status')}); "
                               'R9 delivers no TD3 Q until it is resolved')
        rung = 'rung0_B9_exact'                             # smoke only: wiring check
    rows_c1 = np.flatnonzero(c.S) if rung == 'rung1_rows_S' else allE
    names = {'rung0_B9_exact': ('B9',), 'rung1_rows_S': ('B9',), 'rung2_selected_level': LEVEL_ORDER,
             'rung2_L3': ('L3',)}[rung]
    levels = {lv[0]: lv for lv in cfg.ladder.levels}
    fits = {}
    for name in names:
        for twin in (0, 1):
            fid = f'c1final_t{twin}' if name == 'B9' and rung != 'rung1_rows_S' else f'c1final{name}{"_S" if rung == "rung1_rows_S" else ""}_t{twin}'
            run_fit(c, 'R9', fid, {'base': f'{pre}_t{twin}', 'deliver': deliver, 'level': name, 'rows': _sha(rows_c1)},
                    lambda twin=twin, fid=fid, name=name: (preds_of(allE[:0], {'full': [], 'setup': []}),
                                                           *_c1_final(c, rows_c1, c3s[twin], f'c1final_t{twin}', levels.get(name))))
            fits[(name, twin)] = c.store.fitted('R9', fid)
    mse = {n: np.mean([fits[(n, t)].info['inner_val_mse'] for t in (0, 1)]) for n in names}
    chosen = min(names, key=lambda n: (mse[n], names.index(n)))
    c1s = [fits[(chosen, 0)], fits[(chosen, 1)]]
    p3 = c3s[0].prior
    run_fit(c, 'R9', 'c2prefinal', {'deliver': deliver, 'prior': None if p3 is None else [p3.c, p3.lam]},
            lambda: (preds_of(allA[:0], {'full': [], 'setup': []}), *_c2pre_final(c, allA, deliver, p3)))
    f2p = c.store.fitted('R9', 'c2prefinal')
    run_fit(c, 'R9', 'c2final', {'base': 'c2prefinal'}, lambda: (preds_of(allE[:0], {'full': [], 'setup': []}),
                                                                 *_c2_final(c, allE, f2p)))
    f2 = c.store.fitted('R9', 'c2final')
    tz = {t: d.z_raw[i] for i, t in enumerate(d.track_names)}
    zp = {t: bool(d.z_provisional[i]) for i, t in enumerate(d.track_names)}
    prov = c.provenance() | {'groups': kind, 'barcelona_delivered': deliver, 'smoke': c.smoke}
    bdir = c.out / 'bundles'
    s_tr = T.style_inputs(d, rows_c1, c1s[0].style_tf)[0]
    paths = {'critic3': save_bundle(bdir / 'critic3_final.pt', 'critic3', c3s, sources=d.games, track_z_raw=tz, z_provisional=zp,
                                    provenance=prov),
             'critic1': save_bundle(bdir / 'critic1_final.pt', 'critic1', c1s, sources=d.games, track_z_raw=tz, z_provisional=zp,
                                    style_train=s_tr, provenance=prov),
             'critic2': save_bundle(bdir / 'critic2_final.pt', 'critic2', [f2], sources=d.games, track_z_raw=tz, z_provisional=zp,
                                    style_train=s_tr, provenance=prov)}
    out = {'smoke': c.smoke, 'bundles': {k: str(v.relative_to(c.out)) for k, v in paths.items()}, 'groups': kind,
           'td3_q': {'bundle': str(paths['critic1'].relative_to(c.out)), 'kind': 'critic1',
                     'style_rows': 'S' if rung == 'rung1_rows_S' else 'exact', 'rung': rung, 'level': chosen,
                     'level_inner_val_mse': mse, 'e_gate': [float(f.model.e_gate) for f in c1s], 'ladder': ladder,
                     'rule': "spec 12.21 / B23: TD3's Q is always critic 1; critic 3 is a separate style-free bundle"},
           'critic3_delivered': {'variant': variant, 'final_c': final_c, 'fits': pre},
           'barcelona_delivered': deliver, 'r8_disagreement': r8_disagreement(c, paths)}
    out['barcelona_style_reference'] = bcn_style_reference(c, paths['critic1'])
    write_metrics(c, 'R9', out)


def _head_only(c, rows, f, seed):
    head, info = T.fit_bcn_head(c.d, rows, f, c.cfg.bcn, seed=seed, smoke=c.smoke, salt='final')
    return info, head


def _c1_final(c, allE, f3, fid, level=None):
    d = c.d
    w = T.boot_weights(d.cell_key[allE], T.seed_of('boot_cell', fid))
    f1 = T.finetune(d, allE, f3, 'critic1', c.cfg, seed=T.seed_of('c1', fid), weights=w, smoke=c.smoke, salt=fid, level=level)
    return f1.info, f1


def delivered_prior(c: Ctx, fs, r, k):
    """The B21 prior (lambda, m, w) of the delivered critic 3 of a fold (twin 0), or None (plain)."""
    run3, pre3 = c3_source(c, fs)
    if pre3 == 'c3':
        return None
    return c.store.fitted(run3, f'{pre3}_{fs}_r{r}_f{k}_t0').prior


def delivered_variant(c: Ctx) -> str:
    """'B21' or 'plain' (the delivered critic 3, B21.4 / B21.5); plain in smoke runs without R3b."""
    for name in ('R3c', 'R3b'):
        p = c.out / 'metrics' / f'{name}.json'
        if p.exists():
            dec = json.loads(p.read_text()).get('delivered_critic3')
            if dec is not None:
                return dec
    if c.smoke:
        return 'plain'
    raise RuntimeError('the delivered critic 3 is not decided: run R3b (and R3c)')


def _c2pre_final(c, allA, deliver, prior=None):
    d = c.d
    f = T.pretrain(d, allA, 'critic2', None, c.cfg, seed=T.seed_of('c2', 'final'), smoke=c.smoke, salt='c2final', prior=prior)
    if deliver == 'pooled_plus_head':
        f.bcn, f.info['bcn_head'] = T.fit_bcn_head(d, allA, f, c.cfg.bcn, seed=T.seed_of('bcn2final'), smoke=c.smoke, salt='c2final')
    return f.info, f


def _c2_final(c, allE, f2p):
    f = T.finetune(c.d, allE, f2p, 'critic2', c.cfg, seed=T.seed_of('c2ft', 'final'), smoke=c.smoke, salt='c2ftfinal')
    return f.info, f


def r8_disagreement(c: Ctx, paths):
    """B14: Spearman of J over 2,000 support-box setups between g = F1 25 and g = F1 26, per track."""
    rng = np.random.default_rng(T.seed_of('r8'))
    out = {}
    for kind in ('critic3', 'critic1'):
        b = CriticBundle.load(paths[kind])
        lo, hi = b.support_box()
        P = torch.as_tensor(lo + rng.random((c.cfg.eval.r8_setups, 15)) * (hi - lo), dtype=torch.float32)
        per = {}
        for t in c.d.track_names:
            js = [b.J(P, b.context(track=t, source=g)).numpy() for g in b.sources]
            per[t] = spearman(js[0], js[1]) if len(js) == 2 else None
        vals = [v for v in per.values() if v is not None]
        out[kind] = {'barcelona': per.get(BARCELONA), 'mean_over_tracks': float(np.mean(vals)) if vals else None,
                     'min_over_tracks': float(np.min(vals)) if vals else None, 'per_track': per}
    return out


def bcn_style_reference(c: Ctx, c1_path):
    """The Barcelona style cells (all cells with style on `spain`) as the reference population, and where
    style-rl's OCD drivers fall in it: clipped e per twin, Mahalanobis to the Barcelona cells (s~)."""
    d = c.d
    b = CriticBundle.load(c1_path)
    t_idx = d.track_names.index(BARCELONA)
    cm = (d.cells[:, 2] == t_idx) & np.isfinite(d.cell_style).any(1)
    raw = d.cell_style[cm]
    st, where = b.style_scaler.transform(raw, [BARCELONA] * len(raw))
    mu, cov = st.mean(0), np.cov(st.T) + 1e-6 * np.eye(st.shape[1])
    ci = np.linalg.inv(cov)
    out = {'n_cells': int(cm.sum()), 'centre': sorted(set(where.tolist())), 'drivers': {}}
    drivers, sha = ocd_driver_styles()
    out['style_rl_targets_sha256'] = sha
    if drivers is None:
        out['note'] = 'style-rl targets/style12.json not available at the pinned commit'
        return out
    from rlsetup.style import KEYS
    for name, v in drivers.items():
        x = np.array([v[k] for k in KEYS], float)
        s, _ = b.style_scaler.transform(x[None], [BARCELONA])
        dlt = s[0] - mu
        rep = b.style_report(x, BARCELONA)
        out['drivers'][name] = {'mahalanobis_to_barcelona_cells': float(np.sqrt(max(dlt @ ci @ dlt, 0.))),
                                'mahalanobis_to_training_cells': rep['mahalanobis'],
                                'e_clipped_per_twin': rep['clipped'],
                                's_tilde_clipped_dims': int((np.abs(s[0]) >= b.style_scaler.cfg.z_clip - 1e-12).sum())}
    return out


# ---------------------------------------------------------------------------------------------------
def write_metrics(c: Ctx, run, m):
    runs.write_json(c.out / 'metrics' / f'{run}.json', m)
    merged = {p.stem: json.loads(p.read_text()) for p in sorted((c.out / 'metrics').glob('*.json'))}
    runs.write_json(c.out / 'metrics.json', merged | {'fallback_trigger': fallback_trigger(merged),
                                                      'remedy_ladder': remedy_ladder(merged)})
    runs.write_json(c.out / 'timing.json', c.store.timing)


def fallback_trigger(m) -> dict:
    """B16: fires iff (i) critic 1's delta R2 (c1 - c3) point < -0.02 on F_person or F_cell, (ii) the same
    for the fine-tuned critic 2 vs its A-pretrained version, (iii) critic 1 at e = 0 moves A pair accuracy
    by more than 0.01 vs critic 3. None of a null style effect, critic 2 missing its bar, or a missed
    Barcelona bar triggers it. The trigger never makes critic 3 TD3's Q (spec 12.21: TD3's Q is always
    critic 1; the B16 'neither passes -> critic 3' clause is void, its replacement B23 is pending)."""
    bar = BARS['fallback_trigger']
    r5, r7 = m.get('R5'), m.get('R7')
    if r5 is None or r7 is None:
        return {'evaluated': False, 'needs': [r for r in ('R5', 'R7') if m.get(r) is None]}
    fi = r5['fallback_inputs']
    f7 = r7['fallback_inputs']
    i = any(v['i_delta_r2_point'] is not None and v['i_delta_r2_point'] < bar['delta_r2_point_lt'] for v in fi.values())
    ii = any(v['ii_delta_r2_point'] is not None and v['ii_delta_r2_point'] < bar['delta_r2_point_lt'] for v in f7.values())
    iii = any(v['iii_e0_pair_diff'] is not None and abs(v['iii_e0_pair_diff']) > bar['e0_pair_acc_A_abs_gt'] for v in fi.values())
    return {'evaluated': True, 'fires': bool(i or ii or iii), 'i': bool(i), 'ii': bool(ii), 'iii': bool(iii),
            'td3_q': 'critic1'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default='R1,R2,R3,R3b,R3c,R3d,R4,R5,R5b,R6,R7,R8,R9', help='comma list of ' + ', '.join(RUNS))
    ap.add_argument('--smoke', action='store_true', help='R0 semantics: F_cell fold 0 only, 20 epochs / 100 steps')
    ap.add_argument('--threads', type=int, default=1, help='torch.set_num_threads (another agent uses this machine)')
    ap.add_argument('--data', default=None, help='data build dir (default: latest runs/critic_c2_data/*)')
    ap.add_argument('--out', default=str(REPO_ROOT / 'runs/critic_c2'))
    ap.add_argument('--resume', default=None, help='existing run dir: fits with the same hash are skipped')
    ap.add_argument('--repeats', type=int, default=3, help='F_cell / F_person repeats (budget rule: 3 -> 1 only if R0 '
                                                           'projects > 6 h)')
    ap.add_argument('--rows', choices=('exact', 'S'), default='exact', help='style rows of R5-R7 (S = B16 fallback)')
    ap.add_argument('--force-fallback', action='store_true', help='allow --rows S without a fired trigger')
    ap.add_argument('--drop-quali', action='store_true', help='rows A without Quali (only if the B22.3 S22a rule fired)')
    ap.add_argument('--meta-sources', default=str(REPO_ROOT / 'setup_data' / '*'),
                    help='exports with meta_setups_wide.csv (B21 published-meta sensitivity only)')
    args = ap.parse_args(argv)
    T.set_determinism(args.threads)
    want = [r.strip() for r in args.runs.split(',') if r.strip()]
    bad = [r for r in want if r not in RUNS]
    if bad:
        raise SystemExit(f'unknown runs {bad}')
    if args.smoke:
        args.repeats = 1
    c = Ctx(args)
    if args.rows == 'S' and not args.force_fallback:
        trig = json.loads((c.out / 'metrics.json').read_text()).get('fallback_trigger', {}) if (c.out / 'metrics.json').exists() else {}
        if not trig.get('fires'):
            raise SystemExit('B16 fallback has not fired (metrics.json fallback_trigger); use --force-fallback to override')
    shown = c.out.relative_to(REPO_ROOT) if c.out.is_relative_to(REPO_ROOT) else c.out.name   # no account path in logs
    c.log(f'run dir: {shown}  data hash {c.d.data_hash[:12]}  threads {args.threads}  repeats {c.repeats}  smoke {c.smoke}')
    for r in RUNS:
        if r in want:
            t0 = time.time()
            c.log(f'== {r} ==')
            globals()[f'run_{r}'](c)
            c.log(f'== {r} done in {time.time() - t0:.1f} s')
    return c.out


if __name__ == '__main__':
    main()
