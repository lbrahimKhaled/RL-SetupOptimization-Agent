# D1 review — rl-architect (25 Oct 2026)

Binding on the D1 implementation: where this review and `D1_driving_env.md` conflict, **this review wins**. Labels: [F] formulation, [C] code / measurement, [P] paper / research note, [J] judgement. Items marked "Ibrahim decides" are configurable flags until he answers (see DECISIONS.md).

Verdicts: CHANGES-REQUIRED — compliance/subclassing, action, reset/settle, termination. APPROVE-WITH-NOTES — layout, observation, reward, info, tests.

## R1. Code layout — APPROVE-WITH-NOTES
- All numbers in `config.py`. Freeze as one "driver definition": control rate, γ, (a, b, W, c), P0, κ, T_rec. c's trade-off vs speed is fixed per 20 ms tick; in steady state it's γ-independent (progress and e² both scale with 1/(1−γ)) but not rate-independent. [J]
- Add `rlsetup/driving/qss_profile.py` (v(s) for p0, cached as a file in this repo) and `demo_driver.py` (R5).
- Build `Track` once in the main process before SubprocVecEnv workers start, so ocd_first_lap's safe-line cache is never rewritten concurrently. [C track.py:71–82 rewrites a stale cache]

## R2. Compliance / subclassing — CHANGES-REQUIRED
Works [C]: `step` uses `self._line_errors()[0]` (env.py:147) and `self._observation()` (env.py:157); the nonfinite zero-obs uses the subclass's `observation_space.shape`; `self.to_inputs` resolves to an override.
1. **Reset:** do NOT call `OCDRacingEnv.reset` (second `backend.reset`, wrong pose/s: path[0] at 5 m/s). Call `gym.Env.reset(self, seed=seed)`, run settle, then set `state, s` (via `Track.match`), `progress, steps=0, done=False, physics_steps=0, last_action` = settle's last action (else first u̇ spikes from [0, −1, −1]). Settle steps go straight through `backend.step` and are not counted.
2. **ξref match runs twice per step** (reward, then obs): cache keyed on `physics_steps`. Keep an s_ref hint as separate state (for Jad's lines).
3. **Action repeat 2 (25 Hz decisions):** override/wrapper calling `super().step` twice, summing rewards; stop early if the first sub-step terminates. Second sub-step has u̇ = 0 automatically.
4. **Flying lap:** base env terminates at `progress >= L`. For evaluation set `progress = −(run_in + L)` so the episode covers run-in + 2 laps; time line crossings yourself (interpolated, as `drive()` does).
5. **Equivalence test:** with ξref = `Track.path`, start s = 0, v = 5, no settle, no repeat → `DrivingEnv` rewards equal `OCDRacingEnv` exactly. Catches upstream changes to private fields.

`Track.match` cost: ~23 µs with hint, ~25 µs without; wrap at s = L handled correctly [C]. Set `self.s` from a match after settle.

## R3. Action — CHANGES-REQUIRED (pedal deadzone: Ibrahim decides)
[C] At 12 kN·m, tanh-noise residual brake is a big drag: 2.5 % pedal = 300 N·m ≈ 1 kN ≈ 1.25 m/s². Pushing the mean to −1 saturates tanh (dead gradient).
Fix: override `to_inputs`: pedal = clip((a + 0.5)/1.5, 0, 1) for both pedals; steering a × 0.6 rad. Still 3 absolute outputs. Demo inverse: a = −0.5 + 1.5·pedal for pedal > 0; a = −0.8 for pedal = 0. Flag `pedal_deadzone` (default on pending Ibrahim).

## R4. Observation — APPROVE-WITH-NOTES
1. **Accelerations (required fix):** a_x = Δv_x/Δt − r·v_y, a_y = Δv_y/Δt + r·v_x (or difference world-frame velocity and rotate). Test: steady circle a_y ≈ v·r.
2. **Preview approved:** 20 stations at 300·(k/N)^1.5 m, points only (SOPHY22 Fig. 2b: course points beat rays+curvature [P]). Precompute on a 1 m grid of `Track.path` s: ξref xy (via an s_path→s_ref map), left edge, right edge; vectorised interpolation at runtime. Scale station k by max(d_k, 10 m) (fixed constants, no running normalisation).
3. **Gear:** gear/8 from public `backend.gear` (cheap). Wheelspin causes upshifts so gear isn't a function of speed alone [C README]. Leak concern (gear at a speed hints final drive) → flag `obs_gear` (Ibrahim decides).
4. **Front slip:** rear only for v1 (front kappa needs a `signals()` call). Monitor front lock in evaluation.
5. **History:** keep 3-tick stack of ego + actions (ACGYM24 [P]); also helps implicit setup inference when fine-tuning [J].

## R5. Reset / settle — CHANGES-REQUIRED
[C] Measured after a reset at speed (constant-speed PI, p0, 12 kN·m): 30 m/s → ΣFz within 5 % at 1.14 s, 2 % at 1.78 s; 60 m/s → 16.4 vs 19.4 kN at 1.0 s, 5 % at 1.70 s, 2 % at 2.36 s. Gearbox 2nd → 6th takes 0.6 s (one shift / 0.1 s, powertrain.py:68).
- **Settle 2.0 s.**
- **Start:** s_start ~ U[0, L) on ξref with ξref heading; settle at constant v0 = f · min v_qss over [s_start, s_start + v0·2 s + 50 m], f ~ U(0.3, 0.7). No braking needed in settle; hand-over position stays uniform.
- **Settle controller = demo driver:** pure-pursuit steering (`ScriptedDriver` geometry); speed PI on pedals with `manoeuvres._speed_pedals` gains (0.4/0.2) — held 60 m/s in test. NOT the `ScriptedDriver` PI (caps drive torque ~270 N·m). Brake pedal capped at 0.3 (3.6 kN·m < 3.65 kN·m no-aero lock). Resample start if the car leaves the track.
- **v(s):** `qss_lap.lap_time` returns scalars only. Recompute the QSS profile in this repo from public pieces (`measure_ggv`, `machine_limit`, `tph.calc_vel_profile`, qss_lap.py lines 94–108). Test: its `calc_t_profile` equals `qss_lap.lap_time(..., ggv_data=same)['lap_time_s']`. Compute once for p0, use for every setup (same start distribution per setup = fairer).
- **Evaluation:** fixed start, settle at f = 0.5, 200 m run-in, 2 timed laps; report lap 2; |lap1 − lap2| as diagnostic.
- `ScriptedPolicy` is unusable for warm-start data at 12 kN·m: its `ScriptedDriver` uses the 360 N·m calibration (hairpin request → pedal 1.0 → full 12 kN·m → lock) and its PI caps ~270 N·m drive torque (can't hold > ~25–30 m/s).

## R6. Reward — APPROVE-WITH-NOTES
1. Reward (1) as written [F]. No PBRS ξref term (NG99: removes the driver at the optimum [P]). No c curriculum.
2. At a fixed rate b·Δt is a per-step constant; speed incentive comes from discounted progress. **The setup agent must score the undiscounted evaluation-lap cost** Σ b·Δt(1 + u̇ᵀWu̇) + Σ c·e², never the training return or P. Log these components per lap in `info`.
3. **Off-track penalty** (25 Hz, γ = 0.992, a = 1, b = 1, c = 0.1, base tick 0.02 s): per-decision r ≈ 0.04·v − 0.04 − c(e₁² + e₂²) − smoothness; negative when 0.04·v < 0.04 + 0.2·e² (e = 2 m: v < 21 m/s; e = 3 m: v < 46 m/s). On track, ξref-to-edge up to 14.8 m [C] → e up to ~13.8 m. Linear recovery from e₀ over T_rec costs (c/0.02)·e₀²·T_rec/3 ≈ 380 at e₀ = 13.8 m, T_rec = 1.2 s. A constant P would need ~600 everywhere → over-cautious where ξref hugs an edge.
   **P = P0 + κ·(c/Δt_tick)·(T_rec/3)·e_exit² = P0 + 30·c·e_exit²**, κ = 1.5, T_rec = 1.2 s, P0 = 50 (covers slow-speed living cost b·0.04/(1−γ) = 5 plus slow recovery ~20). c = 0.1: P = 62 at e = 2 m, 621 at 13.8 m. **No speed scaling** (termination already forfeits value ∝ v, ~190 at 40 m/s; the trap is at low speed). P scales with c automatically in the D3 sweep.
4. u̇ in action units/s. Under repeat, a jump Δu costs 50·b·w·Δu² (steering 0.5·Δu²) — small vs progress ~1.6/decision at 40 m/s. Monitor.

## R7. Termination — CHANGES-REQUIRED
- Add **stall / wrong-way** termination (failure, same P): speed < 3 m/s for > 2 s, or |heading error| > 90° (ACGYM24 < 5 km/h for 2 s; LINESIGHT no checkpoint in 2 s [P]).
- Off-track, nonfinite → `terminated`; time limit → `truncated`.
- 60 s < L/v_max ≈ 67 s: lap completion never fires in training. Keep `max_episode_s` < 67 s or handle explicitly.

## R8. Info — APPROVE-WITH-NOTES
Per-wheel Fz needs a 27-name `signals()` call (single-name call was 33 % of step time). Read telemetry only when `record_telemetry=True` (evaluation). Add termination cause and lap-cost components.

## R9. Tests — APPROVE-WITH-NOTES
Add: R2.5 equivalence; a_y ≈ v·r on a circle; QSS profile reproduces lap time; same seed → same (s0, v0) for different setups; ≥ 95 % of 200 resets settle, ΣFz within 5 % of its 0.5 s mean at hand-over; P formula; stall termination; flying-lap timing with two crossings; obs clip fraction < 1 % over a demo lap; `check_env`.

## R10. D2 (agent) recommendation
- **Library:** SB3 2.9.0 SAC, CPU [P §5]. A/B 1: TQC (sb3-contrib; distributional like SOPHY22 QR-SAC / TRI24 DSAC). A/B 2: CrossQ.
- **Core:** γ = 0.992 per 25 Hz decision (~5 s, ACGYM24); n-step 3 (A/B 5; ACGYM24, HILD25); 3×256 ReLU actor + twin critics, LayerNorm in critic (RLPD) via small custom policy; lr 3e-4; batch 512; buffer 1e6 float32; learning starts after warm-up.
- **Throughput:** 4 SubprocVecEnv (forkserver, `__main__` guard), gradient_steps = 2 per vector step (UTD 0.5); torch threads + workers ≤ performance cores. sim-runner benchmarks env vs gradient steps/s first; pick UTD ∈ {0.25, 0.5, 1} by lap time per wall-clock hour.
- **α:** `ent_coef="auto_0.1"`, target −3 (default 1.0 would blow the BC actor off the demos).
- **Instructive initialization (in order):**
  1. *Output bias* (fallback + ablation): mean-head weights ×0.01, bias = atanh([0, −0.275, −0.8]) (straight, 15 % throttle, brake inside deadzone); log_std head bias −1 (σ ≈ 0.37).
  2. *Demo data:* ~200k transitions from the demo driver (R5), f ~ U(0.35, 0.65), random starts with settle, N(0, 0.1) noise on executed+recorded action, env's own reward (valid off-policy). Not `ScriptedPolicy`.
  3. *BC:* fit actor by max-likelihood with squashed-Gaussian log-prob, validation split. Gate: deterministic BC actor completes a lap.
  4. *Critic warm-up:* prefill buffer with demos; ~20k critic-only gradient steps (actor + α frozen — needs a SAC subclass since SB3 resets lrs each `train()`); then ~20k online steps with the frozen BC actor, critic-only updates (WSRL [P]).
  5. *Unfreeze:* demos dilute naturally. Collapse guard: eval every 10k for the first 200k; if progress < 50 % of BC actor's, roll back and drop actor lr to 1e-4.
- **Eval every 50k:** deterministic flying-lap-2 time; lap-cost components; mean/max |e|; min edge margin; front-lock fraction, wheelspin upshifts; 10 stochastic laps (clean-lap rate); Q vs MC return; α; termination causes.
- **Done (proposed, Ibrahim decides):** 3 seeds each with a clean deterministic flying lap on p0; lap 2 ≤ 1.05 × 133.9 = 140.6 s; < 0.2 s improvement over the last 20 % of steps; seed std ≤ 0.2 s (needed to resolve knob effects of 0.28–1.05 s, README Stage 6); ≥ 9/10 stochastic laps clean; env steps + wall time logged.

## Risks (metric)
1. Termination exploitation (off-track fraction at large |e| / low v rising with return). 2. BC collapse (eval progress vs BC actor; Q spikes). 3. Overestimation (Q vs MC return). 4. Front lock / gear hunting (κ_front < −0.12 fraction; upshifts with slip > 0.12). 5. Residual settle transient (ΣFz deviation at hand-over). 6. Chatter (u̇ᵀWu̇ share of lap cost). 7. α collapse. 8. Seed variance (lap std across seeds = setup agent's noise floor). 9. Obs saturation (clip fraction). 10. Compute (env vs grad steps/s). 11. §11.8 (Ibrahim): OCD and eval policy are deterministic, so B repeated laps add no information; the variance that matters is across fine-tune seeds.

## To verify after SB3 install
1. SB3 2.9.0 n-step buffer: correct `truncated` (timeout bootstrapping) handling with SubprocVecEnv; supported by sb3-contrib TQC/CrossQ? 2. `target_entropy="auto"` = −dim(A)? 3. `SquashedDiagGaussianDistribution.log_prob` tanh inversion with clipping?

### Verified after install (lead, 25 Oct; from installed source)
1. `stable_baselines3/common/buffers.py:850–905` `NStepReplayBuffer(n_steps, gamma)` accumulates n-step returns "properly handling episode termination and timeouts" (uses `handle_timeout_termination` / `timeouts`). sb3-contrib 2.9.0 TQC (`tqc.py:99`) and CrossQ (`crossq.py:92`) both take `n_steps` and use γ^n except on early termination.
2. `sac.py:169–171`: `target_entropy="auto"` = −prod(action_space.shape) = −3 for us.
3. `distributions.py:209–241`: `SquashedDiagGaussianDistribution.log_prob` inverts tanh with clipping (comment l. 234) and adds the `log(1 − a² + 1e-6)` correction.
