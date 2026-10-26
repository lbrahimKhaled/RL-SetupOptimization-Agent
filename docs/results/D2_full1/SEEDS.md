# D2 run `full1` — per-seed progress

Generated 2026-10-26 11:09 (local) by `scripts/seed_summary.py`. Done criterion (DECISIONS #12): clean deterministic flying lap 2 ≤ 140.6 s on all 3 seeds, plateau < 0.2 s over the last 20 %, seed std ≤ 0.2 s, ≥ 9/10 stochastic laps clean. Evaluations every 50k steps; "det." = deterministic policy, flying lap 2 after a run-in lap.

## seed0

- **State:** phase `online_warmup`, state `running`, rollbacks 0, updated 2026-10-26T09:04:56+00:00 UTC
- **BC (instructive init):** epochs 18, best val NLL -2.426, gate lap 2 340.58 s (PASS, clean)
- **Best clean deterministic lap 2:** 340.58 s at step 0 (target ≤ 140.6 s, QSS 133.9 s)
- **Latest eval:** step 0; deterministic clean 340.58 s; stochastic clean 0/10; train wall 0.00 h; — env steps/s

| step | det. result | lap 1 [s] | lap 2 [s] | stoch. clean | stoch. mean lap [s] | α | Q_min / MC | failure causes (training, cumulative) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | clean | 340.19 | 340.58 | 0/10 | — | 0.100 | 1.44 |  |

![lap 2 vs steps](lap2_vs_steps.png)
