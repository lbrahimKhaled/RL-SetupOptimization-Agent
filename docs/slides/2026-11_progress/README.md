# Progress deck, 2 Nov 2026 (Beamer)

`progress_2026-11.pdf`: reward engineering (time_pbrs), the vehicle-dynamics changes parameter by parameter (AV-21 → `sf70h_tum` → `sf70h_f1`: tyre MF5.2 mapping, slip stiffness, heave, brakes, steering, footprint, porpoising), the Limebeer-style OCP check, and the Barcelona / Shanghai lap-time comparisons.

Numbers come from the committed records on branches `comparison-tum-vs-rl` and `sf70h-f1` (the last slide lists the files).

Rebuild:
```sh
python3 make_figures.py <bcn SEEDS.md> <p1_TUM evals.jsonl>   # needs matplotlib
python3 make_dynamics_figures.py                              # tyre curves via mf52.py
pdflatex progress_2026-11.tex && pdflatex progress_2026-11.tex  # metropolis theme
```

`mf52.py` is a Python copy of Open-Car-Dynamics' MF5.2 (`lib/ocd_tire_models_cpp/src/mf_52.cpp`, zero camber). It reproduces the peak slips and peak μ in `docs/results/sf70h_f1/results.json` (branch `sf70h-f1`).
