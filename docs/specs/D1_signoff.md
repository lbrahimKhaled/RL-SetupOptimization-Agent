# D1 sign-off — rl-architect (25 Oct 2026)

Evidence labels: [F] formulation, [C] code / own read-only measurement, [P] paper (keys in docs/research/driving_agent_sota.md), [J] judgement.

New measurements (read-only): demo flying laps f=0.5 → 268.36/268.38 s, f=0.7 → 192.01/192.03 s; per-feature p99 |obs|: actual steer 0.061, heading error 0.032, v_y 0.009, yaw rate 0.41–0.57, slips 0.11–0.28, preview 1.07. `Track` is NOT picklable (`PicklingError: Prepared geometries cannot be pickled`); cached build 21 ms. `backend.reset(None, …)` skips `translate` → shipped car silently.

**Overall: APPROVE-WITH-NOTES. Two fixes required before D2: A5 (obs scales) and N1 (setup=None guard).**

**A1. Demo driver traction control — APPROVE-WITH-NOTES.** [F] Driver outputs only steer/throttle/brake; env has no TC; formulation untouched — TC lives only in the demo/settle controller. [C] Rear slip ratios (×5) are in the obs, so the BC actor can represent the TC law. [C] Settle is outside episode and buffer; in eval it precedes the reported lap 2 by ≥ 166 m + a lap, so TC can't affect setup scores. Notes: log in DECISIONS as technical deviation; `traction_control` goes into the demo-dataset cache key; BC report plots throttle vs max rear slip (should show cut-off near 0.08–0.20); first suspect if the BC gate fails on corner exits.

**A2. DrivingEnv defaults to Setup() — APPROVE** (DECISIONS #4).
**N1 — CHANGES-REQUIRED (one line):** `reset(options={'setup': None})` passes None to `backend.reset`, which skips `translate` → shipped car, zero downforce [C backend.py:122–130]. D3 sends setups through options; a stray None would silently score the wrong car. Fix: `setup = options.get('setup'); setup = self.setup if setup is None else setup`. Test: `reset(options={'setup': None})` state == reset without options. The R2.5 equivalence test must pass `Setup()` explicitly to both envs (it already does, test_env.py:45).

**A3. Track loaded once per env — APPROVE-WITH-NOTES.** [C] Track can't be pickled, so SubprocVecEnv workers must build their own anyway. D2 rule: main process builds `Track(name)` first (validates/rewrites cache single-process), computes `sha256(track.path)`, passes it to the worker factory; each worker asserts its sha matches.

**A4. obs_gear only current tick — APPROVE** (moot, #11 = off). Keep flag + test that obs size is 159 when off.

**A5. Steer angle scaled by π — CHANGES-REQUIRED (config only, before any D2 training).** [C] steer/π p99 0.061, max 0.092; heading/π p99 0.032; v_y/80 p99 0.009. [J] Fixed scales should put typical values ~0.1–1, and steer should be in action units so `last_action[0] − steer_obs` is the PT1 lag. Fix: separate `ObsConfig` fields `steer_scale_rad = 0.6` (MAX_STEER_RAD), `heading_scale_rad = 0.5` (wrong-way at π/2 → 3.14, inside ±10 clip), `vy_scale_mps = 5.0`. Re-run clip-fraction test (< 1 %). **Obs layout is frozen from D2 on** (changing it invalidates every trained driver).

**A6. Random-action stalls; min start 3.05 m/s — APPROVE, no change.** [C] Uniform random pedals brake 75 % of the time (deadzone) and 12 kN·m beats throttle — stalling is correct. D2 never takes uniform random actions (learning_starts = 0, BC actor acts from step one). Stall needs > 2 s continuously below 3 m/s, so a 3.05 m/s hand-over isn't an instant stall. Watch: stall share among episodes with v0 < 5 m/s (train_stats).

**A7. Readings — all APPROVE.** (a) s0 uniform on Track.path, car on ξref abeam — same coordinate for every ξref. (b) Settle window from f·v_qss(s0) — avoids circularity, conservative. (c) Settle at 25 Hz with 50 Hz gains — OK given 200/200; add ΣFz-at-hand-over statistic (within 5 % of 0.5 s mean, R9) to slow-test output if missing. (d) Eval hand-over ~166 m before line — lap 2 starts at the agent's own line speed. (e) Crossing tick split proportionally — time component per lap is exactly b·lap time. (f) progress = −1e7 — OK. (g) P on non-finite uses last finite e — OK; count `nonfinite` separately everywhere, it should be 0, any occurrence → sim-runner (simulator issue, not driver).
