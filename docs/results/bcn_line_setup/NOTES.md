# Barcelona (Catalunya), sf70h_f1: line and setup for run bcn_f1_r4 (17 Nov 2026, Windows)

Simulator: `ocd_first_lap` branch `sf70h-f1` (ee86683). QSS = `ocd_lap.qss_lap`, 15 kN m brakes, ERS/DRS off.
Script: `scripts/sha_line_setup.py --track Catalunya` (the Shanghai procedure, NOTES of shanghai_line_setup).

## 1. Lines (`lines`, lines.json, line_*.csv), QSS with p1_TUM_f1

| line | QSS [s] | footprint fit | min clearance [m] | outside |
|---|---|---|---|---|
| safe (Track.path = QP w4.5) | 87.142 | 100 % | +1.04 | — |
| QP w3.0 | 86.386 | 100 % | +0.20 | — |
| IQP w3.5 | 82.744 | 100 % | +0.26 | — |
| **IQP w3.0 (chosen)** | **82.409** | **100 %** | **+0.01** | — |
| IQP w2.6 | 82.192 | 99.83 % | −0.05 | 2 stretches, 8 m |
| IQP w2.0 | 81.972 | 83.12 % | −0.31 | 55 stretches, 772 m |
| IQP w1.7 | 81.848 | 71.57 % | −0.46 | 44 stretches, 1300 m |
| TUM raceline (Track.reference) | 81.426 | 66.32 % | −0.94 | 40 stretches, 1540 m |
| OCP sf70h_f1 (2 m) | 83.403 | 91.81 % | −0.37 | 83 stretches, 377 m |

Ibrahim first asked for IQP w1.7. The env ends an episode when any corner of the car's footprint leaves the
track (`Track.contains_car`), so on that line the car would be off track for 28 % of the lap, and so would the
scripted demos that follow it. He chose **IQP w3.0** instead: the tightest IQP line that fits everywhere.

## 2. Setup sweep (`sweep --line iqp_w3`, sweep.json): 40 g-g-v measurements around p1_TUM_f1

Base 82.409 s. Greedy combination kept: aero_front_share 0.5384, roll_stiffness_front_share 0.40,
roll_stiffness_Nm_per_rad 4.0e5, heave_freq_front_hz 10 → **80.111 s** (IQP w3.5 80.449, safe line 84.742).
This is sha_f1_best + front heave 10 Hz (the heave adds 0.016 s). Named setup **`bcn_f1_best`** (Ibrahim's choice).
- All g-g-v rows valid; front ride height in a 90 m/s hold +0.1 cm (front heave 10 Hz lowers it by 0.3 cm).
- Ramp-steer spin (limit oversteer) flagged for every candidate past the base, as for sha_f1_best.
- Not taken: aero_level 0.7 / 0.75 (79.08 / 78.77 s, front below the ground at 90 m/s: OCD has no floor contact);
  lower front ride height and more rake (SENSIBLE rule: no lowering of the car).
- Caveats: the aero gain rests on the PLACEHOLDER aero map; QSS's acceleration model is optimistic
  (docs/results/realistic_reference: ~3 s on Shanghai), so 80.1 s is not a realistic target.

## 3. Corner table on IQP w3.0 / bcn_f1_best (key 6313f2151e24; Shanghai detection parameters)

8 units, windows 2269.9 m (49.3 % of 4602.6 m): U1 [737, 990] (T1-2), U2 [1132, 1317] (T3), U3 [1611, 1864]
(T4), U4 [2013, 2261] (T5), U5 [2446, 2684] (T7-8), U6 [2827, 3033] (T9), U7 [3353, 3857] (T10-13, 3 apexes),
U8 [3908, 4291] (T14-16, 3 apexes). The turn names are my reading of the s positions, not checked against a map.

## 4. Corner line penalty size

k = R4's maximum corner terms per lap / window length = (10 + 10) × 8 / 2269.9 m = **0.0705** per m driven per m
past the 0.5 m free band (1 m past the band in every window costs 160 = 3.5 % of a lap's progress); cap 0.5 per m.
