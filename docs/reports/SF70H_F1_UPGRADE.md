# SF70H F1 upgrade: car `sf70h_f1` (1 Nov 2026)

New car on ocd_first_lap branch `sf70h-f1` (worktree `../ocd_first_lap-f1`, commit `ee86683`). Existing cars (`av21`, `sf70h_tum`) are bit-identical (fingerprints, 85 unit tests, Shanghai lap.csv sha `d47063de…`). Evidence: `docs/research/f1_car_parameters.md` (representative F1 values from cited papers, mostly the Oxford/Limebeer 2013-era car; not SF70H data). Numbers: `docs/results/sf70h_f1/results.json`.

## 1. Replaced (old → new)
| Item | Old | New | Source | Confidence |
|---|---|---|---|---|
| Tyre PKX1/2/3 | 30.7 / 0.27 / 0.13 | front 42.17 / −0.96 / 0; rear 49.84 / −0.38 / 0 | §2 option B (PL14 peak slip, TUM μ) | medium-low |
| PCX1, PCY1 | 1.6, 1.5 | 1.9, 1.9 | §2 (PL14 Q) | medium-low |
| PEX1–4, PEY1–5 | AV-21 | 0 | §2 | medium-low |
| Steering lock | 0.6 rad | 0.30 rad (rate 1.5 kept) | §3 | medium |
| Heave element | off | 1.0 MN/m per front wheel (12.95 Hz), 0.25 MN/m rear (5.85 Hz); front knob range to 16 Hz (PLACEHOLDER bound) | §5/§9 | low-medium |
| Brake bias default | 0.5 | 0.58 | §6 | medium |
| Brake torque | 12 kN·m | 15 kN·m car default | §6 | low-medium |
| Footprint | 4.6 × 2.0 m | 5.006 × 2.07 m (x ∈ [−2.342, 2.664], ±1.035 m) | §1 (FIA 2017 width; length low) | width high |
| Iz (variant `sf70h_f1_iz560`) | 1159.6 | 560.3 | §4 | medium-low |
Unchanged: peak μ (TUM), PKY, ride heights (2013 front 0.014 m is below the knob range; the knobs are the PLACEHOLDER aero map's reference).

## 2. Still PLACEHOLDER (no F1 source)
Ix, Iy; wheel masses and spin inertias; tyre vertical stiffness; combined-slip R* and offsets; relaxation lengths (hard-coded in OCD C++); roll/pitch centres; dampers incl. heave damping (default 0) and packers; steering rate and T_PT1.

## 3. Results (p1_TUM knobs, 15 kN·m, ERS off)
- **Tyre:** Fx peak moves from κ 0.14–0.27 to 0.075–0.081; lateral peak 7.1° / 8.2° at 3 kN (was 9.3° / 10.8°); peak μ unchanged.
- **Understeer gradient** unchanged (0.07–0.29 vs 0.12–0.28 deg/g).
- **QSS Shanghai (safe line / TUM raceline):** sf70h_tum 103.56 / 95.65 s; **sf70h_f1 102.40 / 94.67 s**; bias 0.49 103.13 / 95.35; + heave damping 0.3 102.53 / 94.85; iz560 102.51 / 94.79.
- **QSS Barcelona (bcn_best, g-g-v to 75 m/s):** sf70h_tum 82.89 / 77.40; **sf70h_f1 82.44 / 77.01**; iz560 82.67 / 77.18. With g-g-v to 90 m/s the F1 car's settle collapses (porpoising) → invalid.
- **Porpoising (new finding):** the approved heave springs with zero heave damping leave a 4.5 Hz heave mode that grows with speed through the ride-height aero map (front ride-height std 9.8 mm at 90 m/s vs 3.0 mm old car; not an integrator artefact). Heave damping 0.3 (diagnostic) cuts it ≈ 10×.
- **Closed-loop TUM lap in OCD (lap 2, f 1.00 / 0.96):** debug car 103.68 / 102.99; sf70h_tum 102.87 / 102.45; sf70h_f1 with the old controller spins (its TC target 0.08 sits on the new tyre's peak); **sf70h_f1, tyre-scaled controller 101.84 / 102.56**; + heave damping 0.3 **101.29** / 101.63; iz560 + heave damping 100.72 / 101.58. TC target alone moves the lap ≈ 2 s, so controller tuning is as large as the car effect.
- **PL14-style OCP (0.5 m adaptive):** Shanghai old mapping 94.25 → **sf70h_f1 94.62** (iz560 94.69); Barcelona 78.76 → **79.38** (iz560 79.43). Slower by 0.4–0.6 s from the wider lateral peak slip and the wider car (narrower corridor); the 0.30 rad lock does not bind (max 0.27 rad at T14).

## 4. Iz sensitivity
Small and method-dependent: QSS +0.10 / +0.23 s, closed loop −0.4 to −0.6 s, OCP +0.05 to +0.08 s. Keep 1159.6 until better data.

## 5. Decisions needed
1. **Heave:** add a heave-damping default for this car (no sourced value) or soften the heave springs — as approved, the car porpoises at high downforce.
2. Iz: keep 1159.6 (recommended) or 560.
3. For RL on this car: env steering scaling to 0.30 rad (env scales to 0.6), and re-check demo-driver traction-control targets for the peakier tyre.

## Reproduce
- Car tests: `cd ../ocd_first_lap-f1 && ../ocd_first_lap/.venv/bin/python -m unittest discover -s tests`
- Checks: `.venv/bin/python scripts/sf70h_f1_checks.py [params tyres qss_sha qss_bcn handling closed_loop]`
- Closed loop: `.venv/bin/python scripts/comparison/tum_closed_loop.py --ocd-root ../ocd_first_lap-f1 --car sf70h_f1 --f 1.0 0.96 --all [--gain tc_slip 0.06 --gain abs_slip 0.06 --gain alpha_f_max_rad 0.125,0.125]`
- OCP: `OMP_NUM_THREADS=2 .venv/bin/python scripts/ocp_sf70h_f1.py sha_f1 …`
