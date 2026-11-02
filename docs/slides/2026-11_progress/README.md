# Progress deck, 2 Nov 2026 (Beamer, 15 pages)

`progress_2026-11.pdf`:
1. Title
2–3. Reward engineering (time_pbrs: no jitter term, linear |e| as potential-based shaping) and its effect
4–11. Vehicle dynamics: AV-21 → `sf70h_tum` → `sf70h_f1` (TUM values, MF5.2 peak-friction mapping, slip-stiffness problem and PL14 fix, tyre curves, combined slip, heave/brakes/steering/footprint, porpoising, lap-time and controller effect)
12. Limebeer-style OCP verification
13–14. Barcelona vs Limebeer, Shanghai vs real world / QSS
15. Summary and decisions

Rebuild:
```sh
python3 make_figures.py <bcn SEEDS.md> <p1_TUM evals.jsonl>   # needs matplotlib
python3 make_dynamics_figures.py                              # tyre curves via mf52.py
pdflatex progress_2026-11.tex && pdflatex progress_2026-11.tex  # metropolis theme
```
`fig/` also holds charts from the longer earlier version (g-g-v, lateral tyre curve, Shanghai progression, training curves) that the 15-page deck no longer shows.

`mf52.py` is a Python copy of Open-Car-Dynamics' MF5.2 (`lib/ocd_tire_models_cpp/src/mf_52.cpp`, zero camber). It reproduces the peak slips and peak μ in `docs/results/sf70h_f1/results.json` (branch `sf70h-f1`).

## Sources (numbers in the deck)
Most of these are on branches `comparison-tum-vs-rl` / `sf70h-f1`, not `main`.
- Reward: `docs/specs/R1_time_optimal.md`, `docs/research/reward_redesign.md`, `rlsetup/driving/config.py`, DECISIONS #13, #19
- Runs: `docs/results/{D2_full1, R1_warm, sf70h_mac_r1, bcn}`, `comparison/training_stop_20261029_094402Z/p1_TUM`
- AV-21 diagnosis: `comparison/tum_ocd_debug_20261031_144747Z/FINDINGS.md`, `docs/research/f1_car_parameters.md`
- New car: `docs/reports/SF70H_F1_UPGRADE.md`, `docs/results/sf70h_f1/results.json`; ocd_first_lap branch `sf70h-f1`: `ocd_lap/cars.py`, `ocd_lap/aero_map.py`, `docs/SF70H_TUM_CAR.md`
- OCP: `docs/results/limebeer_ocp/REPORT.md`, `docs/research/limebeer_lap_times.md`
- Real laps: `docs/research/shanghai_lap_records.md`, `docs/research/shanghai_sim_and_car_reference.md`
