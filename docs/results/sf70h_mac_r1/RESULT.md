# sf70h_mac_r1 — sf70h_tum car, new reward (time_pbrs), from scratch, Mac (CPU)

Stopped by Ibrahim on 28 Oct at 9.7M steps (saturated). Safe line as ξref, ERS off, no DRS. BC gate failed on the Mac (2 seeds; passed on Windows with the same settings), so the run continued past it (`bc.gate_required=False`).

| Metric | Value |
|---|---|
| Best clean deterministic lap 2 | **103.24 s** (step 6.95M, `best.zip`) |
| QSS reference, this car, safe line, no ERS/DRS | 104.78 s |
| QSS, TUM raceline, ERS + DRS | 94.83 s |
| TUM laptime-sim (ERS + DRS, raceline) | 91.24 s |
| Det. lap clean | 48 / 194 evaluations |
| Stochastic median lap, last 1M steps | ≈ 105.4–105.6 s, 60–85 % clean |

Saturation: best det. lap per 500k window 104.9 (1M) → 103.3 (4.5M) → 103.2–103.8 (4.5–9.5M). Next: lap analysis (time loss vs QSS profile, crash corner), then a racing line sized for the car and ERS on.
