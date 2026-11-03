# Shanghai, sf70h_f1: line vs setup (102.5 s QSS on the safe line vs 94.7 s on TUM's raceline)

Started 3 Nov 2026. Simulator: worktree `/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1`
(commit ee86683) via PYTHONPATH. QSS = `ocd_lap.qss_lap` (15 kN m brake ramp, ERS/DRS off, car's
ellipse exponent 2). f1_sha training runs in parallel: every job here is 1 process, OMP_NUM_THREADS=1.

## Progress log
- step 0: context read; plan = (1) lines by width allowance, (2) setup sweep, (3) report.
- step 1a (`scripts/sha_line_setup.py lines`, lines.json): track.py's single-QP opt_min_curv call
  rebuilt at w_veh 4.5 reproduces Track.path exactly (max |diff| 0). QSS p1_TUM_f1 (g-g-v 15-90 m/s,
  all rows valid): safe 102.532; w4.0 102.288; w3.5 102.145; w3.0 101.965 (100 % fit, +0.07 m);
  w2.8 101.846 (99.98 %); w2.6 101.900 (99.98 %, -0.12 m, 1 m outside); w2.4 101.773;
  raceline 94.847 (68.15 % fit, 1701 m outside, -0.82 m); OCP line 101.160 (94.1 % fit; its 0.5 m
  points give noisy curvature, |k|max 2.9, so this QSS is not meaningful as is).
  **Finding: the width allowance is worth only ~0.6 s.** The raceline's curvature is much lower
  (|k|max 0.076 vs 0.115, int k^2 0.62 vs 0.74), so the gap is the optimiser/preprocessing, not w_veh.
  Next: GRO-style preprocessing (smoothed, resampled centre line) + iterative QP (tph.iqp_handler).
- step 1b (experiment /tmp, to be folded into the script): **the single linearised QP is the
  culprit, not the width.** Same optimiser, iterated (tph.iqp_handler, as TUM's
  global_racetrajectory_optimization 'mincurv_iqp'; iters_min 3, curv error 0.01, 3 m re-interp.):
  raw centre line iqp w2.6 **95.507 s** (fit 99.88 %, -0.12 m), w3.0 95.975 (99.87 %, -0.34 m);
  GRO-style smoothed centre line (splprep s=10, 3 m) single QP w2.6 100.231 / w3.0 100.419 (100 %),
  iqp w2.6 95.601 (99.69 %) / w3.0 95.520 (99.89 %). tph 0.80's own spline_approximation breaks on
  this scipy/numpy (fmin returns an array) -> reimplemented the smoothing; GRO (opt_mintime) not
  installed -> skipped.
- step 1c (`sha_line_setup.py lines`, iterated QP added; lines.json, line_*.csv):

  | line | QSS p1_TUM_f1 [s] | footprint fit | min clear [m] | length [m] | abs k max | int k^2 ds |
  |---|---|---|---|---|---|---|
  | safe = QP w4.5 (Track.path) | 102.532 | 100 % | +0.89 | 5388.8 | 0.110 | 0.762 |
  | QP w3.5 | 102.145 | 100 % | +0.32 | 5378.1 | 0.113 | 0.751 |
  | QP w3.0 | 101.965 | 100 % | +0.07 | 5372.9 | 0.114 | 0.744 |
  | QP w2.6 | 101.900 | 99.98 % | -0.12 | 5368.7 | 0.115 | 0.740 |
  | IQP w4.5 | 96.609 | 100 % | +0.70 | 5375.1 | 0.097 | 0.685 |
  | IQP w4.0 | 96.469 | 99.93 % | -0.00 | 5365.1 | 0.083 | 0.673 |
  | **IQP w3.5** | **96.010** | **100 %** | **+0.10** | 5364.7 | 0.095 | 0.667 |
  | IQP w3.0 | 95.975 | 99.87 % | -0.34 | 5355.4 | 0.075 | 0.654 |
  | **IQP w2.6** | **95.507** | 99.88 % | -0.12 | 5355.8 | 0.093 | 0.651 |
  | TUM raceline (Track.reference) | 94.847 | 68.15 % | -0.82 | 5340.8 | 0.076 | 0.624 |
  | OCP sf70h_f1 line (2 m) | 99.573* | 94.44 % | -0.47 | 5373.5 | 2.1* | 4.8* |

  *OCP x_car/y_car points have jitter -> curvature noise; QSS on it is not meaningful.
  IQP w2.6/w3.0/w4.0 leave the footprint at one spot only, s ~ 4714-4726 m (hairpin T14), 4-7 m long.
  Footprint check assumes zero body slip (yaw = line tangent).
  **Line accounts for 102.53 -> 95.51 s (7.0 s), almost all from iterating the QP (6.5 s at equal w).**
  Best fitting line (>= 99.5 %) = IQP w2.6 -> sweep target; IQP w3.5 also reported (fully inside).
- step 2a (first sweep, 37 g-g-v): admissibility excluded ramp-steer "spun" rows, which removed the
  big aero gains (aero_front_share 0.5384 -93.43 s on IQP w2.6, aero_level 0.7 -94.68). "spun" in
  handling.ramp_steer = |beta| > 15 deg at the end of the ramp, i.e. limit oversteer, not a fault
  (bcn_best has it). Real faults seen: aero_level 0.9 / 1.0 -> the 90 m/s settle collapses
  (v_traction -17 / -59 m/s; front ride height +0.5 / +0.3 cm already at 85 m/s; no floor contact
  in OCD). Rerun with: spin flagged only; exclude invalid g-g-v rows and car below ground in a
  90 m/s hold; lowering the car excluded (bcn rule); final probe halfway to excluded values.
  Single-knob results (IQP w2.6, base 95.507): aero_front_share .48 94.419 / .5092 93.696 / .5384
  93.432; aero_level .7 94.681; ride_height_front .025 95.265 (lowers); rake .07 95.403 / .09 95.311
  / .10 95.259; brake bias .54 95.365; roll front share .40 95.393; roll stiffness 4e5 95.445;
  lsd_lock_drive .5 95.447; heave front 10 Hz 95.486; final drive, heave rear, heave damping, ride
  freq rear, LSD preload: < 0.01 s.
- step 2b (`sha_line_setup.py sweep`, sweep.json, exactly 40 g-g-v): greedy kept aero_front_share
  0.5384 (93.432), roll_stiffness_front_share 0.40 (93.387), roll_stiffness 4e5 (93.304),
  heave_freq_front 10 Hz (93.244). Rejected: aero_level 0.7 on top (92.684, front -0.11 cm at
  90 m/s), rake 0.10 (slower combined), brake bias 0.54 (slower combined: 93.552), lsd 0.5;
  probe aero_level 0.75 (92.439, front -0.7 cm). Excluded upfront: aero_level >= 0.9 (90 m/s settle
  collapses), lower front ride height (bcn rule).
- step 2c: **sha_f1_best = greedy result without heave_freq_front 10 Hz** (that 0.06 s came only
  from the front sitting 0.09 cm off the ground at 90 m/s). Added to rlsetup/driving/setups.py;
  `setups.get('sha_f1_best')` reproduces the sweep row exactly (re-measured: same laps). Translator
  / lift warnings: none. QSS v_max on Shanghai (ERS off) is 83.6 m/s, so the 90 m/s check is a
  margin. Ride heights (front / rear, cm): sha_f1_best +0.85 / +4.9 at 84 m/s, +0.43 / +4.4 at 90;
  sha_f1_best + aero_level 0.7: +0.39 at 84, -0.11 at 90.

## Result

| line | fit | p1_TUM_f1 | sha_f1_best |
|---|---|---|---|
| safe (Track.path, QP w4.5) | 100 % | 102.532 | 100.162 |
| QP w2.6 (track.py call, smaller w) | 99.98 % | 101.900 | - |
| IQP w4.5 | 100 % | 96.609 | - |
| IQP w3.5 (fully inside, +0.10 m) | 100 % | 96.010 | 93.825 |
| IQP w2.6 (best fitting) | 99.88 % | 95.507 | 93.304 |
| TUM raceline (does not fit) | 68 % | 94.847 | 92.656 |

Split of 102.53 -> 95: **line ~7.0 s** (safe -> IQP w2.6; 6.4 s of it is iterating the QP, 0.6 s
the smaller width allowance), **setup ~2.2 s** (on IQP w2.6; 2.4 s on the safe line). Line + setup
= 93.30 s, i.e. below 95 in QSS. The rest of the gap to the raceline (0.66 s at either setup) is
corners where the raceline leaves the footprint. Caveats: QSS has no transients and no yaw
stability; the aero gain is the PLACEHOLDER aero map's balance; sha_f1_best is neutral to
slightly oversteering at 60-90 m/s (limit spin in the ramp steer) where p1_TUM_f1 understeers
(+1.6 to +2.2 deg/g); min-curvature != min-time (PL14 OCP of p1_TUM_f1: 94.62 s, different model).

## Recommended next RL run (Ibrahim to decide)
f1_sha with xi_ref = IQP w3.5 (`line_iqp_w3.5.csv`, fully inside; 0.5 s slower in QSS than
IQP w2.6, which is 0.12 m outside at T14 and would put the settle/demo driver off track) and
setup sha_f1_best (QSS 93.83 s on that line). Code needed (not written):
1. `ReferenceLine.for_track`: accept e.g. `'csv:<repo-relative path>'` -> `from_csv` (a string,
   so vector-env workers get it through the config dict); copy the line to `data/lines/`.
2. `DrivingEnv`: `drive_line = self.ref` also for csv lines (now only for 'path'), so settle and
   demos follow the new line.
3. QSS profile stays on Track.path (start speeds only); record the QSS lap on the new line
   (`qss_lap.lap_time(path=ref.path)`) in the run metadata / keeper threshold.
4. Preset `f1_sha_line` (reference + setup), tests (loads, closed, fits the footprint).
