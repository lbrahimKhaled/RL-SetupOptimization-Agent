#!/usr/bin/env python3
"""(Re)build the cached p0 QSS speed profile: data/qss_profile_<track>_p0.npz."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from ocd_lap.track import Track
from rlsetup.driving import qss_profile

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', default='Shanghai')
    args = p.parse_args()
    prof = qss_profile.load_or_build(Track(args.track), rebuild=True)
    print(f'{qss_profile.default_path(args.track)}: lap {prof.lap_time_s:.3f} s, '
          f'v {prof.v.min():.2f}-{prof.v.max():.2f} m/s')
