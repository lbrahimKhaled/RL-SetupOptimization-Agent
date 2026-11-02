"""Dynamics figures for the deck: MF5.2 tyre curves of the three cars (mf52.py, a copy of OCD's
formulas) and the g-g-v limits from ocd_first_lap docs/SF70H_TUM_CAR.md."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mf52

OUT = Path(__file__).parent / 'fig'
GRAY, ORANGE, BLUE, INK, INK2, GRID = '#8a8984', '#eb6834', '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})
CARS = [('AV-21 (shipped)', lambda ax: mf52.AV21, GRAY, ':'),
        ('sf70h_tum', mf52.sf70h_tum, ORANGE, '--'),
        ('sf70h_f1', mf52.sf70h_f1, BLUE, '-')]


def style(ax):
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)


# Longitudinal: Fx/Fz vs kappa, rear axle, 3 kN and 6 kN
k = np.linspace(0, 0.4, 4001)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
for ax, fz in zip(axs, (3000., 6000.)):
    for name, fn, col, ls in CARS:
        f = mf52.fx0(fn('rear'), fz, k) / fz
        i = f.argmax()
        ax.plot(k, f, color=col, linestyle=ls, linewidth=2, label=name)
        ax.plot(k[i], f[i], 'o', color=col, markersize=7, markeredgecolor='white')
        ax.annotate(f'κ={k[i]:.3f}', (k[i], f[i]), textcoords='offset points', xytext=(4, 6),
                    fontsize=8, color=INK)
    ax.axvspan(0.08, 0.11, color=BLUE, alpha=0.08)
    ax.text(0.095, 0.15, 'F1 band\n(PL14)', ha='center', fontsize=7.5, color=INK2)
    ax.set_title(f'rear tyre, Fz = {fz/1000:.0f} kN', loc='left', fontsize=10.5)
    ax.set_xlabel('longitudinal slip κ [–]'); style(ax); ax.set_ylim(0, 2.3)
axs[0].set_ylabel('Fx / Fz  (pure slip)')
axs[1].legend(fontsize=8, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(OUT / 'tyre_fx.png', dpi=200); plt.close(fig)

# Lateral: |Fy|/Fz vs alpha (mean of both slip directions, as the committed sweep reports)
a = np.radians(np.linspace(0, 20, 2001))
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
for ax, fz in zip(axs, (3000., 6000.)):
    for name, fn, col, ls in CARS:
        p = fn('rear')
        f = (mf52.fy0(p, fz, -a) - mf52.fy0(p, fz, a)) / 2 / fz
        i = f.argmax()
        ax.plot(np.degrees(a), f, color=col, linestyle=ls, linewidth=2, label=name)
        ax.plot(np.degrees(a[i]), f[i], 'o', color=col, markersize=7, markeredgecolor='white')
        ax.annotate(f'{np.degrees(a[i]):.1f}°', (np.degrees(a[i]), f[i]), textcoords='offset points',
                    xytext=(4, -16) if col == BLUE else (4, 6), fontsize=8, color=INK)
    ax.set_title(f'rear tyre, Fz = {fz/1000:.0f} kN', loc='left', fontsize=10.5)
    ax.set_xlabel('slip angle α [deg]'); style(ax); ax.set_ylim(0, 2.5)
axs[0].set_ylabel('|Fy| / Fz  (pure slip)')
axs[1].legend(fontsize=8, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(OUT / 'tyre_fy.png', dpi=200); plt.close(fig)

# Combined slip on a corner exit: rear tyre, 6 kN, slip angle 3 deg, kappa sweep
k = np.linspace(0, 0.35, 3501)
alpha = np.radians(3.)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
for name, fn, col, ls in CARS[1:]:
    fx, fy = mf52.combined(fn('rear'), 6000., k, alpha)
    fx, fy = fx / 6000., np.abs(fy) / 6000.
    axs[0].plot(k, fx, color=col, linestyle=ls, linewidth=2, label=f'{name}: Fx')
    axs[0].plot(k, fy, color=col, linestyle=ls, linewidth=1.2, alpha=0.6, label=f'{name}: |Fy|')
    i = fx.argmax()
    axs[0].plot(k[i], fx[i], 'o', color=col, markersize=7, markeredgecolor='white')
    axs[0].annotate(f'Fx peak κ={k[i]:.2f}\n|Fy| left {fy[i]:.2f}', (k[i], fx[i]), textcoords='offset points',
                    xytext=(10, -44) if col == BLUE else (-20, -36), fontsize=7.5, color=INK)
    axs[1].plot(fx, fy, color=col, linestyle=ls, linewidth=2, label=name)
axs[0].set_xlabel('longitudinal slip κ [–]'); axs[0].set_ylabel('force / Fz')
axs[0].set_title('rear, 6 kN, α = 3°: force vs drive slip', loc='left', fontsize=10.5)
axs[0].legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); style(axs[0])
axs[1].set_xlabel('Fx / Fz'); axs[1].set_ylabel('|Fy| / Fz'); axs[1].set_aspect('equal', adjustable='datalim')
axs[1].set_title('same sweep in the (Fx, Fy) plane', loc='left', fontsize=10.5)
axs[1].legend(fontsize=8, frameon=False, loc='lower left'); style(axs[1])
fig.tight_layout(); fig.savefig(OUT / 'tyre_combined.png', dpi=200); plt.close(fig)

# g-g-v: lateral limit vs speed (SF70H_TUM_CAR.md, measured ramp steer; TUM from its car.py)
v = [15, 30, 45, 60, 75, 90]
ay = {'AV-21 (shipped)': ([1.17, 1.45, 1.87, 2.45, 3.20, np.nan], GRAY, ':'),
      'sf70h_tum (full car)': ([1.73, 2.22, 2.88, 3.77, 4.87, 6.32], ORANGE, '--'),
      "TUM's own model": ([2.02, 2.48, 3.20, 4.14, 5.23, 6.41], BLUE, '-')}
br = {'AV-21 (shipped)': [1.33, 1.62, 2.19, 2.81, 3.48, np.nan],
      'sf70h_tum (full car)': [1.54, 1.96, 2.54, 3.16, 3.88, 4.68],
      "TUM's own model": [1.97, 2.43, 3.16, 4.12, 5.25, 6.48]}
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for name, (vals, col, ls) in ay.items():
    axs[0].plot(v, vals, color=col, linestyle=ls, linewidth=2, marker='o', markersize=5, markeredgecolor='white', label=name)
    axs[1].plot(v, br[name], color=col, linestyle=ls, linewidth=2, marker='o', markersize=5, markeredgecolor='white', label=name)
axs[0].set_title('lateral limit ay_max', loc='left', fontsize=10.5)
axs[1].set_title('braking limit (tyre only)', loc='left', fontsize=10.5)
for ax in axs:
    ax.set_xlabel('speed [m/s]'); style(ax)
axs[0].set_ylabel('[g]'); axs[0].legend(fontsize=8, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(OUT / 'ggv_limits.png', dpi=200); plt.close(fig)
print('ok')
