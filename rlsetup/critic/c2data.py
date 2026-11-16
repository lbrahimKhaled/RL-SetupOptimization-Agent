"""C2b input: the data build of scripts/critic_c2_data.py (dataset.npz + summary.json) as one container.

Source-independent: everything is integer codes, names of tracks / games / vocabularies, and the
style numbers. No username is in the files (C2a), and none is read here.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import REPO_ROOT, TRACK_FEATURES
from .folds import data_hash_of, rows_digest


@dataclass
class C2Data:
    """Rows A in dataset.npz order. `style_raw` is the cell style (12 style-rl numbers, NaN where the
    row is not a style row or its cell has no style)."""
    p: np.ndarray                   # (n, 15) normalised knobs, KNOB_ORDER
    gap: np.ndarray                 # % gap to the (game, track, session) median (B1), lower = faster
    lap_s: np.ndarray
    label_median_s: np.ndarray      # B1 final median of the row's group [s] (MAE in seconds)
    game: np.ndarray
    track: np.ndarray
    driver: np.ndarray              # (game, user) id
    person: np.ndarray              # cross-game person id (B6)
    team: np.ndarray
    session: np.ndarray
    controller: np.ndarray
    assists: dict                   # name -> (n,) level index (0 = unknown)
    date_days: np.ndarray           # -1 = unknown
    setup_hash: np.ndarray          # (n,) str
    cell: np.ndarray                # -1 = not a style row
    exact: np.ndarray               # (n,) bool: exact telemetry row
    style_raw: np.ndarray           # (n, 12)
    has_style: np.ndarray           # (n,) bool: the row's cell has style numbers
    games: tuple
    track_names: tuple
    sessions: tuple
    controllers: tuple
    team_names: tuple
    assist_levels: dict             # name -> levels
    z_raw: np.ndarray               # (n_tracks, 5) TRACK_FEATURES, NaN = none
    z_provisional: np.ndarray       # (n_tracks,) bool (B3)
    data_hash: str
    rows_sha256: str
    cells: np.ndarray = None        # (n_cells, 3) (game, driver, track) of every telemetry cell
    cell_style: np.ndarray = None   # (n_cells, 12) cell style, NaN where none
    source_dir: str = ''
    provenance: dict = field(default_factory=dict)
    build_sha256: str = ''          # sha256 over dataset.npz + summary.json bytes (QA M1: every column of the build)

    def __len__(self):
        return len(self.gap)

    @property
    def n(self):
        return len(self.gap)

    @property
    def cell_key(self) -> np.ndarray:
        """(n,) int cell code (game, driver, track) for every row A (also rows without telemetry)."""
        return (self.game.astype(np.int64) * 10_000_000 + self.driver) * 1000 + self.track

    @property
    def track_key(self) -> np.ndarray:
        return np.array(self.track_names, object)[self.track]

    def group_key(self) -> np.ndarray:
        """(game, track, session) code per row (B1 label group, B7 fixed effect)."""
        return (self.game * len(self.track_names) + self.track) * len(self.sessions) + self.session


def latest_data_dir(root=REPO_ROOT / 'runs/critic_c2_data') -> Path:
    dirs = sorted(d for d in Path(root).iterdir() if (d / 'dataset.npz').exists() and (d / 'summary.json').exists())
    if not dirs:
        raise FileNotFoundError(f'no C2 data build under {root}; run scripts/critic_c2_data.py')
    return dirs[-1]


def load_c2data(data_dir=None) -> C2Data:
    d = Path(data_dir) if data_dir is not None else latest_data_dir()
    z = np.load(d / 'dataset.npz', allow_pickle=False)
    summ = json.loads((d / 'summary.json').read_text())
    if 'person' not in z.files or 'z_provisional' not in z.files:
        raise ValueError(f'{d} is a C2a build without person / z_provisional: rebuild with scripts/critic_c2_data.py')
    if list(z['track_feature_names']) != list(TRACK_FEATURES):
        raise ValueError('track feature names differ from config.TRACK_FEATURES')
    cell = z['cell']
    cs = z['cell_style']
    style = np.full((len(cell), cs.shape[1]), np.nan)
    ok = cell >= 0
    style[ok] = cs[cell[ok]]
    has = np.zeros(len(cell), bool)
    has[ok] = z['cell_style_n_laps'][cell[ok]] > 0
    names = [str(a) for a in z['assist_names']]
    levels = summ['vocabularies']['assist_levels']
    dhash = data_hash_of(summ['provenance']['input_files_sha256'])
    if summ.get('data_hash') not in (None, dhash):
        raise ValueError('summary.json data_hash differs from the recomputed hash of its input files')
    setup_hash = z['setup_hash'].astype(str)
    hb = hashlib.sha256()
    for name in ('dataset.npz', 'summary.json'):
        hb.update(name.encode())
        hb.update((d / name).read_bytes())
    return C2Data(p=z['p'].astype(float), gap=z['gap'].astype(float), lap_s=z['lap_s'].astype(float),
                  label_median_s=z['label_median_s'].astype(float), game=z['game'].astype(int),
                  track=z['track'].astype(int), driver=z['driver'].astype(int), person=z['person'].astype(int),
                  team=z['team'].astype(int), session=z['session'].astype(int), controller=z['controller'].astype(int),
                  assists={a: z[f'assist_{a}'].astype(int) for a in names}, date_days=z['date_days'].astype(int),
                  setup_hash=setup_hash, cell=cell.astype(int), exact=z['exact_lap'] >= 0, style_raw=style, has_style=has,
                  games=tuple(map(str, z['games'])), track_names=tuple(map(str, z['track_names'])),
                  sessions=tuple(map(str, z['sessions'])), controllers=tuple(map(str, z['controllers'])),
                  team_names=tuple(map(str, z['team_names'])), assist_levels={a: tuple(levels[a]) for a in names},
                  z_raw=z['track_features'].astype(float), z_provisional=z['z_provisional'].astype(bool),
                  data_hash=dhash, rows_sha256=rows_digest(setup_hash, z['lap_s']), cells=z['cells'].astype(int),
                  cell_style=np.where((z['cell_style_n_laps'] > 0)[:, None], cs, np.nan), source_dir=str(d),
                  provenance={'data_dir': d.name, 'data_hash': dhash,
                              'style_rl_commit': summ['provenance'].get('style_rl_commit'),
                              'geometry_sha256': summ['provenance'].get('geometry_sha256'),
                              'data_summary_created_utc': summ.get('created_utc'), 'build_sha256': hb.hexdigest()},
                  build_sha256=hb.hexdigest())


__all__ = ['C2Data', 'latest_data_dir', 'load_c2data']
