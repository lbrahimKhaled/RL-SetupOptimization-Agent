"""C2b_addendum B6 (QA I5): fold sets keyed to the data hash, and the inner splits.

- data_hash = sha256 over the sorted sha256 values of the input files (summary.json
  `provenance.input_files_sha256`), joined with newlines. Driver / person ids are renumbered when
  exports are added; keying every fold to the data hash makes an assignment valid for one data set only.
- F_cell:   fold = sha256(data_hash | salt | game | driver | track) mod 5, salts 0, 1, 2 (a cell =
  (game, driver, track): every session of a driver on a track is in one fold).
- F_person: fold = sha256(data_hash | salt | person) mod 5, salts 0, 1, 2 (a person across games).
- F_hash:   fold = sha256(data_hash | setup hash) mod 5, once (critic 3's held-out-setup R2; rows A).
The same assignment covers rows A and the exact rows (exact rows are rows A). `folds.json` stores the
data hash; `load_folds` raises when it does not match. Persons are integers; folds.json maps row
positions (dataset.npz order) to folds and never stores a person or driver id.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

N_FOLDS = 5
REPEATS = 3
FOLD_SETS = ('F_cell', 'F_person', 'F_hash')


def data_hash_of(input_files_sha256: dict) -> str:
    """sha256 over the sorted input-file sha256 values (B6)."""
    return hashlib.sha256('\n'.join(sorted(input_files_sha256.values())).encode()).hexdigest()


def _h(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def _fold(keys, n_folds, *prefix) -> np.ndarray:
    cache = {}
    out = np.empty(len(keys), int)
    for i, k in enumerate(keys):
        if k not in cache:
            cache[k] = _h(*prefix, *(k if isinstance(k, tuple) else (k,))) % n_folds
        out[i] = cache[k]
    return out


def rows_digest(setup_hash, lap_s) -> str:
    """sha256 identifying the row order of a dataset (setup hash and lap time per row)."""
    h = hashlib.sha256()
    h.update('\n'.join(map(str, setup_hash)).encode())
    h.update(np.ascontiguousarray(np.asarray(lap_s, float)).tobytes())
    return h.hexdigest()


def make_folds(data_hash: str, game_names, driver, track_names, person, setup_hash, n_folds: int = N_FOLDS,
               repeats: int = REPEATS) -> dict:
    """{'F_cell': (repeats, n), 'F_person': (repeats, n), 'F_hash': (1, n)} int fold per row.
    `game_names` / `track_names`: the source name of each row (strings, not indices)."""
    cell_keys = list(zip(map(str, game_names), map(int, driver), map(str, track_names)))
    return {'F_cell': np.stack([_fold(cell_keys, n_folds, data_hash, salt) for salt in range(repeats)]),
            'F_person': np.stack([_fold([int(p) for p in person], n_folds, data_hash, salt) for salt in range(repeats)]),
            'F_hash': _fold([str(h) for h in setup_hash], n_folds, data_hash)[None]}


def inner_split(keys, frac: float, data_hash: str, salt) -> np.ndarray:
    """Deterministic inner-validation mask: True for the keys whose hash falls in the first `frac`
    (keys = cell or person per row; the same key is always on the same side)."""
    cache = {}
    out = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        if k not in cache:
            cache[k] = (_h(data_hash, 'inner', salt, *(k if isinstance(k, tuple) else (k,))) % 100_000) < frac * 100_000
        out[i] = cache[k]
    return out


def write_folds(path, folds: dict, data_hash: str, rows_sha256: str, build_sha256: str = None) -> str:
    """folds.json; returns its sha256 (the folds hash recorded in bundle provenance). `build_sha256`: the
    sha256 of the whole data build (dataset.npz + summary.json; QA M1), asserted by `load_folds`."""
    doc = {'data_hash': data_hash, 'rows_sha256': rows_sha256, 'build_sha256': build_sha256, 'n_folds': N_FOLDS,
           'n_rows': int(next(iter(folds.values())).shape[1]),
           'keys': {'F_cell': 'sha256(data_hash|salt|game|driver|track) mod 5, salts 0..2',
                    'F_person': 'sha256(data_hash|salt|person) mod 5, salts 0..2',
                    'F_hash': 'sha256(data_hash|setup_hash) mod 5, once (rows A)'},
           'sets': {k: np.asarray(v).tolist() for k, v in folds.items()}}
    text = json.dumps(doc, separators=(',', ':'))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_folds(path, data_hash: str, rows_sha256: str = None, build_sha256: str = None):
    """(folds {name: (r, n) int array}, folds sha256). Raises ValueError when the stored data hash, row
    digest or build hash differs from the given one."""
    text = Path(path).read_text()
    doc = json.loads(text)
    if doc['data_hash'] != data_hash:
        raise ValueError(f"folds.json data hash {doc['data_hash'][:12]} != data {data_hash[:12]}")
    if rows_sha256 is not None and doc['rows_sha256'] != rows_sha256:
        raise ValueError('folds.json row digest does not match the dataset rows')
    if build_sha256 is not None and doc.get('build_sha256') != build_sha256:
        raise ValueError('folds.json build hash does not match this data build (dataset.npz / summary.json changed): '
                         'start a fresh run directory')
    return {k: np.asarray(v, int) for k, v in doc['sets'].items()}, hashlib.sha256(text.encode()).hexdigest()


__all__ = ['FOLD_SETS', 'N_FOLDS', 'REPEATS', 'data_hash_of', 'inner_split', 'load_folds', 'make_folds', 'rows_digest',
           'write_folds']
