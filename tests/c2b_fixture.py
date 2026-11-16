"""Synthetic C2b data build (dataset.npz + summary.json in the layout of scripts/critic_c2_data.py) for the
C2b tests. No real data, no usernames: persons and drivers are integers from the start.

Six tracks (incl. 'spain' = Barcelona and the B3 provisional 'imola' / 'las_vegas'), two games, persons
of whom some play both games. Each driver has rows on a few tracks in all three sessions, and one exact
TT row (with style) per cell for most cells. Label: a planted setup effect (quadratic in a few knobs)
+ driver skill + session offset + noise, then the B1 robust label per (game, track, session).
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from rlsetup.critic.config import KNOB_ORDER, TRACK_FEATURES
from rlsetup.critic.dataset import robust_label
from rlsetup.critic.knobs import setup_hash
from rlsetup.critic.sources.f1laps import ASSIST_LEVELS
from rlsetup.style import KEYS

GAMES = ('F1 25', 'F1 26')
TRACKS = ('austria', 'imola', 'las_vegas', 'monza', 'silverstone', 'spain')
SESSIONS = ('Race', 'Quali', 'TT')
CONTROLLERS = ('Wheel', 'Controller', 'blank')
BASE_LAP = {'austria': 64., 'imola': 75., 'las_vegas': 93., 'monza': 79., 'silverstone': 86., 'spain': 72.}


def setup_effect(p):
    """Planted % effect of the setup (lower = faster)."""
    return .6 * (p[:, 6] - .2) ** 2 + .4 * (p[:, 0] + .3) ** 2 + .3 * p[:, 8] * p[:, 9] + .2 * p[:, 13]


def write_c2b_dataset(root, n_persons: int = 48, seed: int = 0, style_interaction: float = 0.) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    drivers = {}
    cells = {}
    for person in range(n_persons):
        games = [0] if person % 5 == 0 else ([1] if person % 5 == 1 else ([0, 1] if person % 7 == 0 else [person % 2]))
        skill = rng.normal(0, .5)
        base_style = rng.normal(0, 1, len(KEYS))
        ctrl = int(rng.integers(0, 3))
        for g in games:
            drv = drivers.setdefault((g, person), len(drivers))
            tracks = rng.choice(len(TRACKS), size=int(rng.integers(2, 5)), replace=False)
            for t in tracks:
                n = int(rng.integers(4, 9))
                cell = cells.setdefault((g, drv, int(t)), (len(cells), base_style + rng.normal(0, .3, len(KEYS))))
                exact_row = int(rng.integers(0, n)) if rng.random() < .8 else -1
                for k in range(n):
                    p = np.clip(rng.normal(0, .45, 15), -1, 1)
                    sess = 2 if k == exact_row else int(rng.integers(0, 3))
                    rows.append(dict(p=p, game=g, track=int(t), driver=drv, person=person, session=sess, controller=ctrl,
                                     skill=skill, cell=cell[0], exact=k == exact_row, style=cell[1], date=19700 + int(rng.integers(0, 300)),
                                     team=int(rng.integers(0, 3)), abs=int(rng.integers(1, 3)) if g == 1 else 0,
                                     tc=int(rng.integers(1, 4)) if g == 1 else 0, brk=int(rng.integers(1, 5)) if g == 1 else 0))
    n = len(rows)
    P = np.array([r['p'] for r in rows])
    style = np.array([r['style'] for r in rows])
    eff = setup_effect(P) + style_interaction * style[:, 0] * P[:, 6]
    sess_off = np.array([1.4, -1.35, 0.])[[r['session'] for r in rows]]
    game = np.array([r['game'] for r in rows])
    track = np.array([r['track'] for r in rows])
    base = np.array([BASE_LAP[TRACKS[t]] for t in track]) * np.where(game == 1, 1.03, 1.)
    gap_true = eff + np.array([r['skill'] for r in rows]) + sess_off + rng.normal(0, .3, n)
    lap = base * (1 + gap_true / 100)
    session = np.array([r['session'] for r in rows])
    group = (game * len(TRACKS) + track) * 3 + session
    keep, gap, med, _ = robust_label(lap, group)
    sel = np.flatnonzero(keep)
    knobs_hash = setup_hash(np.round(P, 3))
    # cell styles: positive raw numbers in the style-rl ranges, NaN geometry numbers on imola / las_vegas
    n_cells = len(cells)
    cell_style = np.zeros((n_cells, len(KEYS)))
    cell_keys = np.zeros((n_cells, 3), int)
    for (g, drv, t), (ci, st) in cells.items():
        cell_keys[ci] = (g, drv, t)
        v = np.exp(.3 * st) * np.array([.6, .85, 5.8, 1.3, .85, 2.6, 6.2, 18.5, 49., 2.2, 1., 3.5])
        v[:2] = np.clip(v[:2], 0, 1)
        if TRACKS[t] in ('imola', 'las_vegas'):
            v[8:] = np.nan
        cell_style[ci] = v
    n_laps = np.ones(n_cells, int)
    n_laps[::17] = 0                                     # a few cells without style
    cell_style[n_laps == 0] = np.nan
    cell = np.array([r['cell'] for r in rows])
    exact = np.array([r['exact'] for r in rows])
    z = np.array([[10, 669, .645, .033, .192], [14, 1179, .526, .026, .286], [14, 1782, .656, .037, .178],
                  [13, 1174, .719, .023, .154], [17, 717, .555, .041, .193], [12, 1010, .412, .034, .277]], float)
    assists = {a: np.zeros(n, int) for a in ASSIST_LEVELS}
    assists['anti_lock_brakes'] = np.array([r['abs'] for r in rows])
    assists['traction_control'] = np.array([r['tc'] for r in rows])
    assists['braking_assist'] = np.array([r['brk'] for r in rows])
    arrays = dict(knob_order=np.array(KNOB_ORDER), knobs=P[sel], p=P[sel], gap=gap[sel], lap_s=lap[sel], game=game[sel],
                  track=track[sel], driver=np.array([r['driver'] for r in rows])[sel],
                  team=np.array([r['team'] for r in rows])[sel], session=session[sel],
                  controller=np.array([r['controller'] for r in rows])[sel],
                  date_days=np.array([r['date'] for r in rows])[sel], setup_hash=knobs_hash[sel].astype(str),
                  cell=cell[sel], exact_lap=np.where(exact, np.arange(n), -1)[sel],
                  person=np.array([r['person'] for r in rows])[sel], label_median_s=med[sel],
                  games=np.array(GAMES), track_names=np.array(TRACKS), team_names=np.array(['A', 'B', 'C']),
                  sessions=np.array(SESSIONS), controllers=np.array(CONTROLLERS), cells=cell_keys,
                  style_keys=np.array(KEYS), cell_style=np.nan_to_num(cell_style, nan=np.nan), cell_style_n_laps=n_laps,
                  track_features=z, track_feature_names=np.array(TRACK_FEATURES),
                  z_provisional=np.array([t in ('imola', 'las_vegas') for t in TRACKS]),
                  assist_names=np.array(list(ASSIST_LEVELS)), **{f'assist_{a}': v[sel] for a, v in assists.items()})
    np.savez_compressed(root / 'dataset.npz', **arrays)
    files = {f'synthetic/{k}': hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest() for k in ('setups.csv', 'laps.csv')}
    from rlsetup.critic.folds import data_hash_of
    summ = {'created_utc': 'synthetic', 'vocabularies': {'sessions': SESSIONS, 'controllers': CONTROLLERS,
                                                         'assist_levels': {a: list(v) for a, v in ASSIST_LEVELS.items()}},
            'data_hash': data_hash_of(files),
            'provenance': {'input_files_sha256': files, 'style_rl_commit': 'synthetic', 'geometry_sha256': {}}}
    (root / 'summary.json').write_text(json.dumps(summ))
    write_meta_export(root / 'meta_export')
    return root


def write_meta_export(root) -> Path:
    """An export folder with an empty setups.csv and a meta_setups_wide.csv for every (game, track) (B21
    published-meta sensitivity); game-unit values like the f1laps file."""
    from rlsetup.critic.sources.f1_game_map import GAME_COLUMNS
    root = Path(root) / 'data'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'setups.csv').write_text('setup_id\n')
    base = dict(zip(GAME_COLUMNS, (25., 25., 70., 40., -3.5, -2., 0., .1, 37., 16., 12., 9., 22., 48., 56.)))
    lines = ['game,track,conditions,' + ','.join(GAME_COLUMNS)]
    for gi, g in enumerate(GAMES):
        for ti, t in enumerate(TRACKS):
            v = dict(base, front_wing=20. + 3 * ti, rear_wing=18. + 2 * ti + gi)
            lines.append(f'{g},{t},Dry,' + ','.join(f'{v[c]:g}' for c in GAME_COLUMNS))
    (root / 'meta_setups_wide.csv').write_text('\n'.join(lines) + '\n')
    return root.parent


__all__ = ['GAMES', 'SESSIONS', 'TRACKS', 'setup_effect', 'write_c2b_dataset', 'write_meta_export']
