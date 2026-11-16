"""The 12-number driver style vector of style-rl, as a numpy-only port.

Source: `style_rl/dstyle/features.py` and `style_rl/dstyle/track.py` of the style-rl repository
(branch `style12-whole-lap`) at commit STYLE_RL_COMMIT. The logic is copied verbatim; only the pandas
DataFrames are replaced (a lap is a mapping of arrays, corners are a list of `Corner` tuples, a track
is built from arrays). Definitions, thresholds and the per (driver, track) median are style-rl's and
must not be changed here (C2 Decision 2): re-port from a newer commit instead, update the constants
below and let `tests/test_c2_style.py` (port equality) compare against the pinned original.

features.py   style_vector(lap, track, q=...): one lap -> the 12 numbers
track.py      Track: position on the road and corner detection from the curvature
"""
STYLE_RL_COMMIT = 'ea37559005b4758e0c03bc3ced2359bab9894781'
# sha256 of the two ported files at STYLE_RL_COMMIT (`git show <commit>:<path> | shasum -a 256`).
STYLE_RL_SOURCES_SHA256 = {
    'style_rl/dstyle/features.py': 'bc16b00cd71346680f4b17a36b47107575f9f6e424141d8be33276d8d7d90f1a',
    'style_rl/dstyle/track.py': '1af3ae81f6e8c2a5cacecb9f4e5ed1f09d9fef5c2f067e10038704816224b491',
}

from .features import KEYS, NAMES, STYLE, style_table, style_vector    # noqa: E402
from .track import Corner, Track, load_track                           # noqa: E402
