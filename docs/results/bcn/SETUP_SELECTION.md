# Barcelona (Catalunya): setup selection and run `bcn`

Car `sf70h_tum` unchanged, ERS / DRS off. Recipe = preset p1_TUM (time_pbrs from scratch, demos f in (0.5, 0.8), BC gate not required, no step/wall cap) on track `Catalunya` with setup `bcn_best` and 15 kN m brakes (preset `bcn`).

## QSS sweep (scripts/bcn_setup_sweep.py -> qss_sweep.json, 39 g-g-v evaluations, ~23 s)
QSS lap [s] on Track.path (safe line, 4602.6 m) / Track.reference (TUM raceline, 4572.5 m). rh@vmax = static front/rear ride height at the QSS top speed (OCD has no floor contact).

| Setup | Brakes | Safe line | Raceline | rh@vmax front/rear [cm] |
|---|---|---|---|---|
| p0 | 12 | 89.233 | 83.371 | -2.7 / -2.0 |
| p0 | 15 | 89.031 | 83.165 | -2.7 / -2.0 |
| p1_TUM | 12 | 87.899 | 81.928 | -2.8 / -2.0 |
| p1_TUM (base) | 15 | 87.884 | 81.906 | -2.8 / -2.0 |
| aero_level 0.3 / 0.7 / 0.9 / **1.0 (bound)** | 15 | 89.486 / 86.539 / 85.529 / 85.171 | 83.446 / 80.533 / 79.681 / 79.449 | down to -4.4 / -4.8 |
| aero_front_share 0.42 / 0.48 / 0.509 / 0.538 | 15 | 88.715 / 87.086 / 86.499 / 86.411 | 82.768 / 81.043 / 80.463 / 80.459 | |
| ride_height_front 0.025 / 0.045 / 0.02 (bound) | 15 | 87.407 / 88.315 / 87.263 | 81.413 / 82.364 / 81.258 | lowers the car: not admitted |
| rake 0.03 / 0.07 / 0.01 | 15 | 87.699 / 88.017 / 87.703 | 81.721 / 82.054 / 81.715 | lowers the car: not admitted |
| brake_bias_front 0.46 / 0.52 / 0.55 / 0.58 | 15 | 88.088 / 87.594 / 87.464 / 87.595 | 82.110 / 81.666 / 81.535 / 81.667 | |
| final_drive 0.95 / 1.05 / 1.1 | 15 | 87.913 / 87.886 / 88.059 | 81.929 / 81.916 / 82.102 | |
| roll_stiffness_front_share 0.40 / 0.50 / 0.55 / 0.35 (bound) | 15 | 87.793 / 87.981 / 88.191 / 87.718 | | |
| roll_stiffness 2e5 / 4e5 / 5.24e5 N m/rad | 15 | 88.215 / 87.844 / 87.892 | | |
| lsd_lock_drive 0.5 / lsd_preload 100 N m | 15 | 87.943 / 87.884 | | |
| combo: aero 1.0 + front share 0.5384 | 15 | 83.310 | 77.755 | -6.2 / -2.1 |
| **+ brake bias 0.55 = bcn_best** | **15** | **82.888** | **77.396** | -6.2 / -2.1 |
| + roll share 0.35 (bound) / + roll 4e5 | 15 | 82.865 / 82.872 | | |
| bcn_best | 12 | 83.081 | 77.581 | |

## Chosen: `bcn_best` + 15 kN m
p1_TUM with aero_level 1.0, aero_front_share 0.5384, brake_bias_front 0.55 (lsd_lock_drive 1.0, roll share 0.4533 as p1_TUM). QSS 82.888 s safe line (p1_TUM 87.884: -5.0 s).
Flags:
- aero_level at its upper bound (1.0). The aero map is PLACEHOLDER (CzA x1.4, CxA x1.3 at level 1).
- Understeer gradient [deg/g] at 15..90 m/s: 0.27, 0.11, 0.03, 0.01, -0.02, -0.19 (p1_TUM 0.33 ... 0.12, 0.44): mildly oversteer at 75-90 m/s. QSS knows nothing about yaw stability.
- Front ride height at v_max -6.2 cm (p1_TUM -2.8): already negative for every setup (soft springs vs F1 downforce, no floor contact in OCD).
- Not taken: roll share 0.35 (bound, -0.023 s; front bar already clamped to 0 by the translator at 0.4533), lower ride height / rake (QSS gain only from the PLACEHOLDER 2 %/cm map with the car already below the ground).
- Ramp steer reports "spun" for every setup incl. p0 (ramp runs to the limit), so it does not discriminate.

## Part B: QSS lap of our car on Catalunya [s]
| Setup | Brakes | Safe line | TUM raceline |
|---|---|---|---|
| p0 | 12 | 89.233 | 83.371 |
| p1_TUM | 15 | 87.884 | 81.906 |
| bcn_best | 15 | 82.888 | 77.396 |

## Run `bcn` (Mac CPU, launched 31 Oct 19:11)
```sh
caffeinate -dimsu nohup platforms/mac/train.sh --preset bcn --seed 0 --run-name bcn --allow-dirty > runs/mac/bcn_stdout.log 2>&1 &
nohup runs/mac/keeper_nostop.sh bcn > runs/mac/keeper_bcn.log 2>&1 &
.venv/bin/tensorboard --logdir_spec "bcn:runs/mac/D2/bcn/seed0,p1_TUM:runs/mac/D2/p1_TUM/seed0" --port 6006 --host 127.0.0.1 --reload_interval 30
```
`--allow-dirty`: launched from an uncommitted tree (setups.py, train_config.py, tests); exact diff in `runs/mac/D2/bcn/seed0/uncommitted_at_launch.diff`. Training PID 79627, keeper PID 79963 (never auto-stops; resumes after a crash up to 5 times with --allow-dirty --allow-code-change).
Stop: `pkill -f "keeper_nostop.sh bcn"; pkill -INT -f "run-name bcn"`.
Resume by hand: `caffeinate -dimsu nohup platforms/mac/train.sh --resume --preset bcn --seed 0 --run-name bcn --allow-dirty --allow-code-change >> runs/mac/bcn_stdout.log 2>&1 &`

## STATUS 31 Oct 19:13: run stopped at the DEMO gate (not the BC gate), no RL step taken
`runs/mac/D2/bcn/seed0/status.json`: state `demo_gate_failed`. The demos (pure pursuit on the safe line at f in (0.5, 0.8)) were 0.50 clean at steering noise 0.05 (off_track 89, wrong_way 14, of 204 episodes) and 0.76 clean after the automatic 0.025 fallback (off_track 35, wrong_way 2, of 155). The gate needs 0.8 (`demos.min_clean_fraction`). p1_TUM's recipe only waives the BC gate, not this one. On Shanghai, p1_TUM passed at 0.92 after the fallback. The off-track starts are spread over the whole lap. The keeper was stopped (pkill) so it would not loop resumes. TensorBoard is still running on 6006.
Options (Ibrahim / lead): (a) relaunch with `--set demos.min_clean_fraction=0.75` (demos are cached, so ~1 min to reach BC); (b) a less oversteery setup (e.g. aero_front_share 0.48, 87.086 s single-knob) and re-check the demos; (c) a lower demo fraction range. A relaunch needs a new run name or removing `runs/mac/D2/bcn`.
