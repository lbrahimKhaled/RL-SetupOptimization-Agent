"""Figures for the Nov 2026 progress deck. Numbers are copied from the committed records
(branch sf70h-f1 / comparison-tum-vs-rl); learning curves are read from the evals.jsonl /
SEEDS.md files given on the command line (optional)."""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / 'fig'
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})
CAT = {'ref': (BLUE, 'External reference (real car / published)'),
       'sim': (ORANGE, 'Our physics bound (QSS / OCP)'),
       'rl': (AQUA, 'Our RL driver')}


def hbars(rows, fname, xlim, title, vline=None):
    fig, ax = plt.subplots(figsize=(6.4, 0.40 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, (label, val, cat) in zip(y, rows):
        ax.barh(yi, val - xlim[0], left=xlim[0], height=0.62, color=CAT[cat][0],
                edgecolor='white', linewidth=2)
        ax.text(val + 0.012 * (xlim[1] - xlim[0]), yi, f'{val:.2f} s', va='center', fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_xlabel('lap time [s]  (axis does not start at 0)')
    ax.grid(axis='x', color='#e4e3df', linewidth=0.8)
    ax.set_axisbelow(True)
    if vline:
        ax.axvline(vline[0], color=INK2, linestyle=':', linewidth=1)
    used = []
    for k in ('ref', 'sim', 'rl'):
        if any(r[2] == k for r in rows):
            used.append(plt.Rectangle((0, 0), 1, 1, color=CAT[k][0], label=CAT[k][1]))
    fig.legend(handles=used, loc='lower center', ncol=1 if len(rows) > 6 else 3, fontsize=7.5, frameon=False)
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.17 if len(rows) > 6 else 0.08, 1, 1))
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)


hbars([
    ('Perantoni & Limebeer 2014 OCP (published)', 82.43, 'ref'),
    ('PL14 OCP, mesh-converged (published)', 82.57, 'ref'),
    ('Masouleh & Limebeer 2016 OCP, w/ suspension', 84.55, 'ref'),
    ('Our PL14-style OCP, Limebeer car', 82.34, 'sim'),
    ('Our OCD QSS, sf70h_tum + bcn_best, safe line', 82.89, 'sim'),
    ('Our RL driver, sf70h_tum + bcn_best', 83.97, 'rl'),
], 'barcelona_bars.png', (80, 85.6), 'Barcelona-Catalunya')

hbars([
    ('Fastest ever, Piastri 2025 Q3', 90.641, 'ref'),
    ('Vettel, Ferrari SF70H, 2017 Q3', 91.864, 'ref'),
    ('TUM QSS, ERS + DRS', 91.24, 'ref'),
    ('TUM QSS, no ERS / DRS', 94.89, 'ref'),
    ('Our PL14-style OCP, SF70H-TUM values', 94.25, 'sim'),
    ('Our OCD QSS, TUM raceline', 95.73, 'sim'),
    ('Our OCD QSS, safe line (p1_TUM)', 103.56, 'sim'),
    ('Our RL driver, p1_TUM', 101.27, 'rl'),
    ('Our RL driver, p0 (sf70h_mac_r1)', 103.24, 'rl'),
], 'shanghai_bars.png', (88, 106.5), 'Shanghai International Circuit')

# Shanghai progression of the RL driver
fig, ax = plt.subplots(figsize=(7.6, 3.2))
stages = ['AV-21 car\nold reward\n(full1)', 'AV-21 car\nnew reward\n(r1_warm)',
          'sf70h_tum car\nnew reward\n(sf70h_mac_r1)', 'sf70h_tum car\n+ p1_TUM setup\n(p1_TUM)']
vals = [122.85, 123.04, 103.24, 101.27]
x = np.arange(len(stages))
ax.bar(x, np.array(vals) - 85, bottom=85, width=0.55, color=AQUA, edgecolor='white', linewidth=2)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.6, f'{v:.2f} s', ha='center', fontsize=9, color=INK)
ax.axhline(91.864, color=BLUE, linestyle='--', linewidth=1.2)
ax.text(3.45, 92.4, 'real SF70H 91.86 s', ha='right', fontsize=8, color=INK2)
ax.set_xticks(x, stages, fontsize=8)
ax.set_ylim(85, 128)
ax.set_ylabel('best clean lap 2 [s]')
ax.grid(axis='y', color='#e4e3df', linewidth=0.8); ax.set_axisbelow(True)
ax.set_title('Shanghai: RL driver, best clean deterministic lap', loc='left', fontsize=11)
fig.tight_layout(); fig.savefig(OUT / 'shanghai_progression.png', dpi=200); plt.close(fig)

# Where TUM's lap loses time when driven in OCD (closed loop, f = 1.00)
fig, ax = plt.subplots(figsize=(5.0, 3.0))
cause = ['traction-limited exits', 'braking / hairpin entry', 'power (carried speed)',
         'low-speed lateral grip', 'controller']
dt = [4.40, 2.40, 1.00, 0.53, 0.47]
y = np.arange(len(cause))[::-1]
ax.barh(y, dt, color=ORANGE, height=0.6, edgecolor='white', linewidth=2)
for yi, v in zip(y, dt):
    ax.text(v + 0.06, yi, f'{v:.2f} s', va='center', fontsize=9)
ax.set_yticks(y, cause, fontsize=9); ax.set_xlim(0, 5.2)
ax.set_xlabel('time lost vs TUM (total 8.79 s)')
ax.grid(axis='x', color='#e4e3df', linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / 'gap_breakdown.png', dpi=200); plt.close(fig)

# Penalty shape: c e^2 vs k |e|
e = np.linspace(-3, 3, 400)
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(e, 0.1 * e ** 2, color=ORANGE, linewidth=2, label=r'old: $c\,e^2$ ($c$=0.1)')
ax.plot(e, 1.0 * np.abs(e), color=BLUE, linewidth=2, label=r'new: $\Phi=-k|e|$ ($k$=1/m)')
ax.set_xlabel('lateral offset e to ξref [m]'); ax.set_ylabel('penalty / potential magnitude')
ax.legend(fontsize=8, frameon=False, loc='upper center')
ax.grid(color='#e4e3df', linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / 'penalty_shape.png', dpi=200); plt.close(fig)


# Learning curves
def from_seeds(path):
    pts = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\|\s*([\d,]+)\s*\|\s*clean\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|', line)
        if m:
            pts.append((int(m.group(1).replace(',', '')), float(m.group(2))))
    return pts


def from_evals(path):
    pts = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if d.get('det_clean') and d.get('lap2_s'):
            pts.append((d['step'], d['lap2_s']))
    return pts


if len(sys.argv) == 3:
    curves = [(from_seeds(sys.argv[1]), 'Barcelona (bcn)', [(82.89, 'our QSS 82.89 s', ORANGE), (82.57, 'Limebeer OCP 82.57 s', BLUE)], (81, 92)),
              (from_evals(sys.argv[2]), 'Shanghai (p1_TUM)', [(103.56, 'our QSS, safe line 103.56 s', ORANGE), (94.89, 'TUM QSS, no ERS/DRS 94.89 s', BLUE)], (93, 112))]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.3))
    for ax, (pts, name, refs, ylim) in zip(axs, curves):
        s, t = zip(*[p for p in pts if p[0] > 0])
        ax.plot(np.array(s) / 1e6, t, color=AQUA, linewidth=2, marker='o', markersize=4,
                markeredgecolor='white', label='RL, clean det. lap 2')
        best = np.minimum.accumulate(t)
        ax.plot(np.array(s) / 1e6, best, color=INK2, linewidth=1, drawstyle='steps-post', label='best so far')
        for v, lab, col in refs:
            ax.axhline(v, color=col, linestyle='--', linewidth=1.2, label=lab)
        ax.set_ylim(*ylim); ax.set_xlabel('env steps [M]'); ax.set_ylabel('lap 2 [s]')
        ax.set_title(name, loc='left', fontsize=11)
        ax.grid(color='#e4e3df', linewidth=0.8); ax.set_axisbelow(True)
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(OUT / 'learning_curves.png', dpi=200); plt.close(fig)
print('figures written to', OUT)
