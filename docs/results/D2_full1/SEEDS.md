# D2 run `full1` — per-seed progress

Generated 2026-10-26 11:19 (local) by `scripts/seed_summary.py`. Done criterion (DECISIONS #12): clean deterministic flying lap 2 ≤ 140.6 s on all 3 seeds, plateau < 0.2 s over the last 20 %, seed std ≤ 0.2 s, ≥ 9/10 stochastic laps clean. Evaluations every 50k steps; "det." = deterministic policy, flying lap 2 after a run-in lap.

## seed0

- **State:** phase `rl`, state `running`, rollbacks 0, updated 2026-10-26T09:16:10+00:00 UTC
- **BC (instructive init):** epochs 18, best val NLL -2.426, gate lap 2 340.58 s (PASS, clean)
- event `unfreeze` at step 20000 (2026-10-26T09:10:11)
- event `det_regression` at step 50000 (2026-10-26T09:12:22)
- **Best clean deterministic lap 2:** 340.58 s at step 0 (target ≤ 140.6 s, QSS 133.9 s)
- **Latest eval:** step 100,000; deterministic wrong_way; stochastic clean 0/10; train wall 0.11 h; 237 env steps/s

| step | det. result | lap 1 [s] | lap 2 [s] | stoch. clean | stoch. mean lap [s] | α | Q_min / MC | failure causes (training, cumulative) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | clean | 340.19 | 340.58 | 0/10 | — | 0.100 | 1.44 |  |
| 50,000 | off_track | — | — | 0/10 | — | 0.066 | 1.57 | stall 15, off_track 62, time_limit 5, wrong_way 6 |
| 100,000 | wrong_way | — | — | 0/10 | — | 0.072 | 1.23 | stall 21, off_track 114, time_limit 15, wrong_way 8 |

![lap 2 vs steps](lap2_vs_steps.png)
