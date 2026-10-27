# r1_warm — time-optimal reward (time_pbrs), warm start from full1 best (122.85 s)

Stopped by Ibrahim on 27 Oct at 2.7M steps (of 3M). Safe min-curvature line as ξref. Settings: DECISIONS #19, #20.

| Metric | full1 (old reward) | r1_warm (new reward) |
|---|---|---|
| Best clean deterministic lap 2 | **122.85 s** | 123.04 s (step 300k) |
| Median of last 5 clean det. laps | — | 123.89 s |
| Det. lap clean (post-unfreeze evals) | — | 8 / 55 |
| Best stochastic mean lap | 124.77 s (at 2.6M) | 125.05 s |
| Stochastic clean rate, last 10 evals | 0.8–1.0 | 0.59 |
| Mean \|e\| on clean laps | 0.43 m | 0.56–1.03 m |

Result: the freed driver left the line (as intended) but did not get faster and crashed more often. The warm-started policy did not improve on 122.85 s; `best.zip` of this run is the step-0 init. Per-eval table and plot: `SEEDS.md`, `lap2_vs_steps.png`.
