# Realistic reference check (9 Nov 2026, lead)

Question (Ibrahim): is the 95 s target unrealistic, is it the setup, or the driver?
Car sf70h_f1, setup sha_f1_best, IQP w3.5 line, Shanghai, ERS/DRS off. Scripts in this folder
(scratch copies; run with `PYTHONPATH=<ocd_first_lap-f1>`), sim commit ee86683.

## 1. The car's real straight-line acceleration (real1b_envelope.py)
Open loop in OCDBackend (powertrain 'engine', as the env), straight road.
- Full pedal from 15 m/s spins the rears (slip 1.0, gearbox jumps to 8th): below ~55 m/s the pedal must
  be modulated. With a slip controller (target 0.06): 12 -> 75 m/s in 9.9 s / 498 m.
- Above 40 m/s, drive force and aero drag match the QSS model, but the sim has an extra resistance of
  ~1.1–1.5 m/s^2 (~0.8–1.1 kN; rotating inertia + rolling/wheel losses) that QSS leaves out.
- Real terminal speed **80.6 m/s**; QSS assumes the car reaches 82.6 / 83.6 m/s at the U1 / U6 entries.

| v [m/s] | 50 | 60 | 70 | 75 | 80 |
|---|---|---|---|---|---|
| QSS net accel [m/s^2] | 10.31 | 7.26 | 4.44 | 2.98 | 1.35 |
| sim net accel (full pedal) | 8.86 | 5.31 | 3.01 | 1.73 | 0.18 |

## 2. Realistic reference lap (real2_lap.py)
Same tph solver, same g-g-v (braking + lateral), only the acceleration limit replaced:
- A QSS as used: **93.83 s** (v_max 83.6)
- C QSS + measured extra resistance only: **95.42 s** (v_max 80.6)
- B real accel (sim, incl. low-speed traction with a simple slip controller): **96.85 s** (v_max 80.5)
B's low-speed part depends on the slip controller; the realistic reference is 95.4–96.9 s, best guess ~96.9 s.
The 95 s stop rule (1.0125 x 93.83) is below these references, so it looks out of reach on this line, but
they are not hard bounds: QSS is a model, the RL can leave the line, and it peaks at up to 107 % of the
g-g-v lateral limit. The source of the extra resistance was not identified (at 80 m/s the net accel is
~0.2 m/s^2, so rotating inertia cannot explain it there).

## 3. Where the RL loses time (best lap f1_r4 6.95M, 101.25 s) vs B (96.85 s): +4.40 s
| | U1 | str | U2 | str | U3 | str | U4 | str | U5 | back str | U6 | str | U7 | str→U1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RL − B [s] | +.33 | +.30 | +.50 | +.24 | +.25 | +.13 | +.29 | +.28 | +.49 | **+.89** | +.45 | +.08 | −.01 | +.17 |
Straights ~2.1 s, corner windows (braking to exit) ~2.3 s.

## 4. Driver behaviour (real3_driver.py)
- **Rarely full throttle**: pedal >= 0.99 for 0.3–0.5 % of lap 2 on average (all clean det laps of
  f1_best_w35, f1_r2, f1_r3, f1_r4; max 4 %), mean 0.93–0.96 on straights, lifts to 0.66–0.76.
  This is NOT a hard ceiling of the pedal map: the driver reaches up to 0.9999, and the median lap max is
  0.992. Accel achieved 77–86 % of the car's at 40–76 m/s, 20–42 % at 76–81 m/s.
- Back straight (51 m/s, 1274 m), sim open loop: pedal 1.0 17.18 s, 0.989 17.24, 0.96 17.40, 0.93
  17.58; RL 17.80 s. Holding 0.989 instead of 1.0 costs only 0.06 s here, so a pedal-map change alone
  would gain little. The remaining ~0.4 s (lifts, weaving, line) was not separated. Steer action std
  0.14–0.30, lateral offset range ~14 m (part of it may be positioning for U6).
- **Braking**: mean decel 50–73 % of the g-g-v braking limit, brake pedal mean 0.3–0.5 (peaks reach
  the limit briefly). The mean includes the trail-braking release, so it overstates the softness; the
  2.3 s lost in the corner windows is not proven to be braking (it also holds the lower entry speed and
  the different driven line).
- **Cornering at the grip limit**: peak |ay| 90–107 % of the g-g-v lateral limit at every apex.

## 5. Setup
Not tested. The QSS sweep that chose sha_f1_best used the optimistic acceleration model; parameters
that do not change straight-line performance (balance, roll stiffness) should keep their ranking,
drag/downforce-related ones (aero_level) may not.

## Reproduce
From the repo root, with `PYTHONPATH=<ocd_first_lap sf70h-f1 worktree>`: `python docs/results/realistic_reference/real1b_envelope.py docs/results/realistic_reference/real1b.npz`, then `real2_lap.py` (writes real2.npz next to it), then `real3_driver.py` (reads the Mac run folders under `runs/mac/D2/`; adjust for another machine). The .npz outputs are not tracked.
