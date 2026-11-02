# Progress deck, 2 Nov 2026 (Beamer, 10 pages)

`progress_2026-11.pdf`:
1. Title
2–3. Reward engineering (time_pbrs) and its effect
4. The three cars (av21, sf70h_tum, sf70h_f1)
5. sf70h_tum: the F1-like car from TUM's data (ERS/DRS modelled but off in all RL runs)
6. TUM's optimal lap driven in OCD by pure pursuit
7. sf70h_f1: the next car (no RL run yet)
8. Limebeer's optimal-control laps and ours
9–10. Barcelona and Shanghai lap-time charts

Rebuild:
```sh
python3 make_figures.py <bcn SEEDS.md> <p1_TUM evals.jsonl>   # needs matplotlib
python3 make_dynamics_figures.py                              # tyre curves via mf52.py
pdflatex progress_2026-11.tex && pdflatex progress_2026-11.tex  # metropolis theme
```
`fig/` also holds charts from the longer earlier versions (tyre curves, combined slip, g-g-v, closed-loop gap breakdown, Shanghai progression, training curves) that this deck no longer shows.

`mf52.py` is a Python copy of Open-Car-Dynamics' MF5.2 (`lib/ocd_tire_models_cpp/src/mf_52.cpp`, zero camber). It reproduces the peak slips and peak μ in `docs/results/sf70h_f1/results.json` (branch `sf70h-f1`).

## Sources (numbers in the deck)
Most of these are on branches `comparison-tum-vs-rl` / `sf70h-f1`, not `main`.
- Reward: `docs/specs/R1_time_optimal.md`, `docs/research/reward_redesign.md`, `rlsetup/driving/config.py`, DECISIONS #13, #19
- Runs: `docs/results/{D2_full1, R1_warm, sf70h_mac_r1, bcn}`, `comparison/training_stop_20261029_094402Z/p1_TUM`
- AV-21 diagnosis: `comparison/tum_ocd_debug_20261031_144747Z/FINDINGS.md`, `docs/research/f1_car_parameters.md`
- New car: `docs/reports/SF70H_F1_UPGRADE.md`, `docs/results/sf70h_f1/results.json`; ocd_first_lap branch `sf70h-f1`: `ocd_lap/cars.py`, `ocd_lap/aero_map.py`, `docs/SF70H_TUM_CAR.md`
- OCP: `docs/results/limebeer_ocp/REPORT.md`, `docs/research/limebeer_lap_times.md`
- Real laps: `docs/research/shanghai_lap_records.md`, `docs/research/shanghai_sim_and_car_reference.md`
