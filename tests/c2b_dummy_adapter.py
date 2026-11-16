"""A minimal non-F1 data-source adapter (a stand-in for the later Assetto Corsa adapter) for the B17
test. It imports numpy and the source-independent knob mapping only: no F1 module, no torch, no critic
model, train or bundle code. What it provides is what a bundle query needs from a source: a canonical
track key (or None for a track the critic has never seen), its centre line, normalised knobs p and the
12 style-rl numbers of a lap (here constants).
"""
import numpy as np
from rlsetup.critic.knobs import denormalise, normalise

SOURCE = 'dummy_sim'                     # never a training source of the critic
TRACKS = {'dummy_barcelona': 'spain', 'dummy_ring': None}


def canonical_track(track_id: str):
    return TRACKS.get(track_id)


def centre_line(track_id: str) -> np.ndarray:
    """(N, 2) closed centre line [m] (a 3-lobe star, ~4.3 km)."""
    th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    r = 650. * (1 + .25 * np.cos(3 * th) + .08 * np.sin(5 * th))
    return np.column_stack((r * np.cos(th), r * np.sin(th)))


def to_p(physical_knobs) -> np.ndarray:
    """Physical OCD knobs (KNOB_ORDER, SI units) -> normalised p in [-1, 1]."""
    return normalise(np.asarray(physical_knobs, float))


def baseline_physical(n: int = 1) -> np.ndarray:
    return denormalise(np.zeros((n, 15)))


def style_numbers() -> np.ndarray:
    """12 raw style-rl numbers (KEYS order); the two geometry numbers missing (NaN)."""
    v = np.array([.6, .85, 5.8, 1.3, .85, 2.6, 6.2, 18.5, 49., 2.2, 1., 3.5])
    v[8:10] = np.nan
    return v
