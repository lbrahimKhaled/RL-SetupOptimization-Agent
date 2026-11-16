"""F1 25 / F1 26 published meta setups (`meta_setups_wide.csv` of the f1laps exports) -> p per (game, track).

B21.1: sensitivity only, never delivered. The published meta is an aggregate of the community's setups and
probably their lap times, held-out folds included (a leak). Only the 15 GAME_COLUMNS are mapped
(`game_to_knobs` + `normalise`); brake pressure, tyre pressures and engine braking have no OCD knob.
Read in place from the export folders / zips (like the setups); the first export that lists a
(game, track) wins. No usernames are in these files.
"""
import csv
import io
import numpy as np
from ..knobs import normalise
from .f1_game_map import GAME_COLUMNS, game_to_knobs
from .f1laps import discover, expand_sources

FILE = 'meta_setups_wide.csv'


def read_meta_rows(sources) -> dict:
    """{(game, track): row dict} from every export under `sources` (paths / globs)."""
    out = {}
    for src in expand_sources(sources):
        for exp in discover(src):
            raw = exp.read(FILE)
            if raw is None:
                continue
            for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
                if (r.get('conditions') or 'Dry') != 'Dry':
                    continue
                out.setdefault((r['game'], r['track']), r)
    return out


def published_meta_p(rows: dict) -> dict:
    """{(game, track): p (15,)} box-normalised, KNOB_ORDER."""
    out = {}
    for key, r in sorted(rows.items()):
        try:
            vals = {c: np.array([float(r[c])]) for c in GAME_COLUMNS}
        except (KeyError, ValueError):
            continue
        out[key] = normalise(game_to_knobs(vals))[0]
    return out


__all__ = ['FILE', 'published_meta_p', 'read_meta_rows']
