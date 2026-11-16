"""Post-hoc ensemble check (lead, 16 Nov; NOT pre-registered). Reads saved per-fold test predictions only
(no fitting). Usage: .venv/bin/python scripts/critic_c2_ensemble_check.py [RUN_DIR]
(default runs/critic_c2/20261116_081601Z)."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlsetup.critic.c2data import load_c2data
from rlsetup.critic.evaluate import Evaluator
R = sys.argv[1].rstrip('/') if len(sys.argv) > 1 else 'runs/critic_c2/20261116_081601Z'
cfg = json.load(open(R + '/config.json'))
d = load_c2data(Path(__file__).resolve().parents[1] / 'runs/critic_c2_data' / cfg['data_dir'])   # the run's own build
assert d.data_hash == cfg['data_hash'], 'data build does not match the run'
ev = Evaluator(d, 2000, 0, 2.8)
bcn = np.array([str(t) for t in d.track_key]) == 'spain' if hasattr(d, 'track_key') else None

def load(run, fid, z_score):
    z = np.load(f'{R}/fits/{run}/{fid}.npz'); s = z['setup'].astype(float)
    if z_score: s = (s - s.mean()) / (s.std() + 1e-12)
    return z['rows'], s

def stack(run, prefix, fs, twin, z_score):
    out = np.full((3, d.n), np.nan)
    for r in range(3):
        for k in range(5):
            fid = f'{prefix}_{fs}_r{r}_f{k}' + (f'_t{twin}' if twin is not None else '')
            rows, s = load(run, fid, z_score)
            out[r, rows] = s
    return out

for fs in ('F_cell', 'F_person'):
    c3 = (stack('R3', 'c3', fs, 0, True) + stack('R3', 'c3', fs, 1, True)) / 2
    rg = stack('R1', 'ridgeA', fs, None, True)
    mask = np.isfinite(c3).any(0)
    base = ev.pair_acc(c3, mask)['point']
    print(fs, 'critic3', round(base, 4), 'ridge', round(ev.pair_acc(rg, mask)['point'], 4))
    for w in (0.25, 0.5, 0.75):
        bl = (1 - w) * c3 + w * rg
        dd = ev.pair_diff(bl, c3, mask)
        print(f'  blend w_ridge={w}: {ev.pair_acc(bl, mask)["point"]:.4f}  diff vs critic3 {dd["point"]:+.4f} CI [{dd["ci"][0]:+.4f}, {dd["ci"][1]:+.4f}]')
        if bcn is not None:
            print(f'     barcelona: blend {ev.pair_acc(bl, mask & bcn)["point"]:.4f} vs critic3 {ev.pair_acc(c3, mask & bcn)["point"]:.4f}')
print('--- 3-way: critic3 + ridge + critic2-pretrain (equal weights)')
import glob, os
for fs in ('F_cell', 'F_person'):
    c3 = (stack('R3', 'c3', fs, 0, True) + stack('R3', 'c3', fs, 1, True)) / 2
    rg = stack('R1', 'ridgeA', fs, None, True)
    c2 = stack('R7', 'c2pre', fs, None, True)
    mask = np.isfinite(c3).any(0)
    bl = (c3 + rg + c2) / 3
    dd = ev.pair_diff(bl, c3, mask)
    print(fs, f'c2pre alone {ev.pair_acc(c2, mask)["point"]:.4f}; 3-way {ev.pair_acc(bl, mask)["point"]:.4f} diff vs critic3 {dd["point"]:+.4f} CI [{dd["ci"][0]:+.4f}, {dd["ci"][1]:+.4f}]; barcelona {ev.pair_acc(bl, mask & bcn)["point"]:.4f}')
print('--- 50/50 critic 3 + critic-2-pretrain (ensemble TD3 Q offline)')
for fs in ('F_cell', 'F_person'):
    c3 = (stack('R3', 'c3', fs, 0, True) + stack('R3', 'c3', fs, 1, True)) / 2
    c2 = stack('R7', 'c2pre', fs, None, True)
    mask = np.isfinite(c3).any(0)
    bl = (c3 + c2) / 2
    dd = ev.pair_diff(bl, c3, mask)
    print(fs, f'critic3 {ev.pair_acc(c3, mask)["point"]:.4f}; c2pre {ev.pair_acc(c2, mask)["point"]:.4f}; '
          f'50/50 {ev.pair_acc(bl, mask)["point"]:.4f} diff vs critic3 {dd["point"]:+.4f} CI [{dd["ci"][0]:+.4f}, {dd["ci"][1]:+.4f}]; '
          f'barcelona critic3 {ev.pair_acc(c3, mask & bcn)["point"]:.4f} c2pre {ev.pair_acc(c2, mask & bcn)["point"]:.4f} '
          f'50/50 {ev.pair_acc(bl, mask & bcn)["point"]:.4f}')
