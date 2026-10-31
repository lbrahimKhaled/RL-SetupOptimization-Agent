# Minimum-lap-time optimal control in the style of Perantoni & Limebeer 2014 (PL14)

31 Oct 2026. Every lap time is IPOPT `Solve_Succeeded` and mesh-converged unless stated. Code: `rlsetup/ocp/`, `scripts/ocp_pl14.py`; tests `tests/test_ocp_*.py` (16 pass). Data: `results.json`, solution CSVs (s, t, C, n, xi, u, v, omega, delta, T, kappa×4, N×4, power_W, x_car, y_car, n_min, n_max), log `PROGRESS.md`.

## Headline
| Case | Lap [s] | Reference |
|---|---|---|
| **D-check: Barcelona, PL14 car** | **82.34** | PL14 mesh-converged 82.57 s → **−0.28 %, PASS** (band ±1.5 %) |
| Shanghai, PL14 car | 97.23 | — |
| **Shanghai, PL14 model with SF70H-TUM values** | **94.25** | TUM laptime-sim `none` 94.89 s; OCD RL p1_TUM 101.27 s |

## 1. Model
Source `docs/research/limebeer_ocp_model.md` (PL14; LPR14 and ML16 where PL14 is silent). Planar body, states n, ξ, u, v, ω in the distance domain s. Equations: (14)–(16), (20)–(22) kinematics; (17)–(19) body; (23)–(26) quasi-static wheel loads (algebraic in the NLP); (32)–(33) aero; A.1–A.18 tyre with Table 3; (34) front brakes; (35) LSD kd = 10.47; power limit (LPR14); cost ∫ S_f ds (the printed S_f⁻¹ is a typo). PL14's SAE frame (y right, z down, n > 0 right, C > 0 right turns), pinned by unit tests.

Deviations from PL14:
1. **Tyre signs:** printed signs conflict for Fz ≤ 0, so N = −Fz, κ = RΩ/u_w − 1, α = atan(v_w/u_w) (as A.17), Fx = +μx N κn/ρ, Fy = −μy N αn/ρ. Tests: zero force at free rolling, drive/brake signs, side force opposing slide, L/R symmetry, peak μ at both reference loads, force inside the friction ellipse.
2. **Peak slip:** A.13 as printed puts the force peak at ρ = 0.750 (peak slip 0.75·κ_max); kept as printed, peak μ exact.
3. Zero-slip smoothing ρ = √(αn² + κn² + 1e-6).
4. No light-wheel clipping (eqs 28, 31); wheel load ≥ 50 N instead, inactive in all reported solutions (min 472 N Barcelona, 441 N Shanghai PL14, 198 N Shanghai SF70H).
5. **Controls** (not listed in PL14): δ and one longitudinal demand T; drive max(T,0) to the rear through the diff; brake min(T,0) split β front / 1−β rear; smoothing per eq. 48, ε = 1e-3 (Mg)². Front forces always equal (no lock release of eq. 34).
6. Power at the rear wheel spin: max(T,0)·R·(Ω_rr + Ω_rl)/2 ≤ Pmax.
7. Control-rate regularisation (ML16 eq. 52 style) 0.1 (dδ/ds)² + 1e-3 (d(T/Mg)/ds)²: costs 0.016–0.02 s, excluded from quoted lap times.

## 2. Assumptions (labelled ASSUMPTION in code)
- Engine power 560 kW (LPR14 Table 2).
- Brake balance 0.55 / 0.60 / 0.65 front (none published).
- Corridor: mass centre within ±(half-width − 0.9 m) (2013 max car width 1.8 m); sensitivity ±4 m (ML16).
- Track: racetrack-database Catalunya (4649.8 m; PL14 ≈ 4655 m) and Shanghai (5445.2 m). Curvature from the PL14 auxiliary OCP with r = 1e4 m⁶ (fit error 0.037 m RMS / 0.47 m max Catalunya, 0.054 / 0.53 m Shanghai); r = 1e6 (PL14, no units) as sensitivity.
- g = 9.81 m/s².
- **SF70H-TUM mapping** (ocd_first_lap `docs/SF70H_TUM_CAR.md`): 733 kg; wheelbase 3.6 m, a = 1.968 m; h = 0.335 m; wf = wr = 0.8 m; Iz 1159.6 kg m² (PLACEHOLDER in source); ρ 1.18; ClA 4.88 m² at aA = 1.977 m; CdA 1.56 m²; 575 kW; peak μ per axle from TUM's μ0 − 5e-5 (N − 3000) at 2000 / 6000 N (front x/y 1.65/1.85, rear 1.95/2.15). Kept at PL14 values: peak slips, Q shape factors, kd, D_roll, R = 0.33 m, half-width 0.9 m.

## 3. Transcription and solver
Direct collocation in s, periodic, trapezoidal (as PL14); states, controls and algebraic variables at nodes. PL14 scaling (unit mass, unit length = wheelbase, g = 1). CasADi 3.8.2 (MX, mapped), IPOPT + MUMPS, exact Hessian, tol 1e-6, adaptive barrier. Initial guess: point-mass QSS speed on the centreline (μ = 1.5). Meshes: uniform 4 → 2 → 1 → 0.5 m warm-started; **speed-adapted** h(s) = ds·clip((u/30)², 0.1, 1), u_min 3 m/s (the stiff lateral/yaw mode's s-domain eigenvalue grows like 1/u², so slow hairpins need finer h). Solve times 11–30 s (Barcelona 1–4 m), 60–120 s (0.5 m), up to 5 min (Shanghai 0.5 m), single thread.

## 4. Verification
- **KKT:** all quoted laps `Solve_Succeeded`; max scaled equality residual 3e-8 to 5e-7.
- **Mesh convergence, Barcelona (β 0.60):** uniform 4/2/1/0.5 m 82.2325 / 82.2950 / 82.3347 / 82.3382 s; adaptive 82.3408 / 82.3073 / 82.3273 / 82.3339 s; agree within 0.005 s → **82.34 s**. As in PL14 Fig. 11, lap time rises with refinement.
- **Shanghai PL14 car:** adaptive 97.2813 / 97.2169 / 97.2253 / 97.2279 s (Δ +0.008, +0.003 s). Uniform 97.69 / 98.09 / 98.31 / 98.41 s, first-order only (pinned at u ≥ 10 m/s at the s ≈ 4800 m hairpin, radius 7 m, δ/ω sawtooth): **rejected**.
- **Shanghai SF70H:** homotopy (4 parameter steps from PL14) 1 m 94.2458 → 0.5 m 94.2491 s (Δ +0.003 s). Direct solves were `Infeasible_Problem_Detected` (uniform 1 m; adaptive 4 m from QSS; adaptive 2 m warm from PL14, twice); their coarse 4 m values (94.68, 94.51 s) are not comparable.
- **Initial guesses (Barcelona 4 m):** QSS μ 1.5 and μ 1.0 → identical 82.2325 s. Constant 40 m/s → **different local optimum 82.7819 s** (82.98 s at 1 m; loses 0.31 s at s ≈ 3400–3600 m, 0.14 s at ≈ 4200 m; lines differ by up to 0.8 m). A circular test track also shows a non-steady local optimum (+0.3 %).
- **A-posteriori re-integration** per interval with IDAS (rtol/atol 1e-10, linear controls, exact algebraic solve). IDAS fails on 4 % of intervals (195/4650 Barcelona 1 m, 216/5675 Shanghai 1 m), all in threshold braking with the fronts exactly at the force peak (dFx/dκ = 0 → singular algebraic Jacobian): an integrator limitation. On the rest:

| Case | max \|err\| n | max \|err\| u | max \|err\| v | max \|err\| ω | RMS v |
|---|---|---|---|---|---|
| Barcelona 1 m | 1.2 mm | 0.9 mm/s | 3.8 cm/s | 0.038 rad/s | 0.8 mm/s |
| Shanghai 1 m adaptive | 1.0 mm | 6.2 mm/s | 1.3 cm/s | 0.066 rad/s | 0.5 mm/s |
| Barcelona 4 m | 5.7 cm | — | 0.22 m/s | — | — |

  Re-integrated lap times 82.3355 vs 82.3347 s (Barcelona), 97.2297 vs 97.2291 s (Shanghai). Chained 50 m open-loop windows: median n error < 0.1 mm (62/92 Barcelona, 82/109 Shanghai windows completed; the rest hit an IDAS failure).

## 5. Results
- **Barcelona, PL14 car:** β 0.60 **82.34 s (−0.28 % vs 82.57 s, PASS)**. At 1 m: β 0.55 82.3724, 0.60 82.3347, 0.65 82.4411 s (spread 0.11 s). ±4 m corridor 83.2518 s (+0.83 %). r = 1e6 smoothing 82.1282 s (−0.21 s). 88.5–314.9 km/h, full power on 53 % of nodes, max sideslip 5.4°.
- **Shanghai, PL14 car (β 0.60): 97.23 s**, 58.6–315.9 km/h.
- **Shanghai, PL14 model with SF70H-TUM values (β 0.60): 94.25 s**, 60.6–306.0 km/h, full power on 65 % of nodes, max sideslip 8.3°; longitudinal slip at its ±0.6 bound on 0.7 % of nodes; inner wheels down to 198 N.

## 6. Limitations
- Local optima only (a different guess at Barcelona gave +0.55 s).
- Not OCD: no suspension, no wheel-spin dynamics, a different tyre, idealised powertrain. These are not OCD or RL lap times.
- Track data differ from PL14's private GPS set; smoothing weight r alone moves the lap by 0.2 s.
- Simplifications: no light-wheel clipping, no lock release (eq. 34), single steer angle, chosen brake ratio.
- SF70H mapping partial (Iz PLACEHOLDER; slip/shape parameters from PL14); its solution sits near wheel lift and slip bounds and needed a homotopy.
- PL14's 82.43 s on their unstated fixed grid cannot be reproduced exactly.
- PROJECT_CONTEXT §12: "OCP/NLP as the solver" is superseded; this OCP is an external reference only, not the setup solver.

## 7. Open choices
Speed-adapted mesh as Shanghai default (PL14 used uniform); equal front braking always (may contribute to SF70H infeasibility); r = 1e4 m⁶; brake-balance sweep only at Barcelona.

## Reproduce
`OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python scripts/ocp_pl14.py <case>` (cases in the script docstring). Install: `uv pip install --python .venv/bin/python casadi==3.8.2` (extra `[ocp]`).
