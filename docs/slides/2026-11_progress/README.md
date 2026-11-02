# Progress deck, 2 Nov 2026 (Beamer)

`progress_2026-11.pdf`: reward engineering (time_pbrs), AV-21 leftovers in the F1 car and the `sf70h_f1` fix, the Limebeer-style OCP check, and the Barcelona / Shanghai lap-time comparisons.

Numbers come from the committed records on branches `comparison-tum-vs-rl` and `sf70h-f1` (the last slide lists the files).

Rebuild:
```sh
python3 make_figures.py <bcn SEEDS.md> <p1_TUM evals.jsonl>   # needs matplotlib
pdflatex progress_2026-11.tex && pdflatex progress_2026-11.tex  # metropolis theme
```
