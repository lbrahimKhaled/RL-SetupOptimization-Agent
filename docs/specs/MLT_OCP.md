# MLT-OCP: minimum-lap-time optimal control for OCD `sf70h_tum` at p1_TUM

Status: architect spec, 31 Oct 2026, NOT started (Ibrahim deciding between this and a Limebeer-benchmark first). Labels: [F] formulation, [C] read in code, [P] paper to verify, [J] judgement. PLACEHOLDER = revisit with data.

## 0. Verdict
- OCP as the "best possible lap" reference: allowed as a baseline only ([F §2]); never the setup solver (§12). Does not settle OPEN §11.5.
- **C1. "QSS ≤ OCP ≤ driven laps" is not a valid sandwich.** A fixed-line QSS is not a lower bound for a free-line OCP, nor a relaxation of the transients; a driven lap is an upper bound only with the same car, setup and track rule. Valid upper bounds: RL p1_TUM lap (same car/setup/footprint rule) and the OCP-tracked lap in real OCD (V7). QSS = diagnostic only. No rigorous lower-bound claim (IPOPT = local optimum).
- **C2.** The closed-loop debug lap (102.99 s) and its QSS (95.73 s) use `sf70h_tum_debug` (constant aero, TUM r(v), CoG rule): initial guess / generic excitation only; target-car QSS must be recomputed (P0).

## 1. Facts [C]
- Integrator: fixed-step Dormand–Prince 5, h = 2 ms; `ode()` uses the previous evaluation's wheel speeds (one-stage lag; zero after reset).
- 33 states: VD 28 (x, y, z, ψ, φ, θ, v_x, v_y, v_z, ψ̇, φ̇, θ̇, z_w×4, v_zw×4, κ×4, α×4), DT 4 (ω_FL, ω_FR, ω_axle, ω_diff; rear L/R = ω_axle ± ω_diff), steering 1.
- Tyre MF5.2 on relaxed κ, α; Fz ≥ 0.001 N clamp; right wheels mirrored; camber −camber_out; τ_κ = 0.3/clamp(|v_x|,10,100), τ_α = 0.2/clamp(…) (2–3 ms at 90 m/s, stiff); κ_raw = clamp((rω − v_x)/max(3, v_x), ±1), α_raw = −atan2(v_y, max(1, v_x)); copysign in PEX4/PEY3 (jump only in 3rd derivative).
- p1_TUM effective parameters: camber = toe = 0; four-way dampers exactly linear; heave off; packers gap 0.10 m, 2 MN/m; pitch centres 0.2 m (anti-dive switch inert); front ARB 0; aero c_l,F = 2.354 − 4.40 rh_F, c_l,R = 3.136 − 5.36 rh_R ([0,5] clamp inactive for |rh| < 0.5 m; negative rh extrapolated); rolling resistance only in wheel load torque; T_PT1 0.06 s, |δ| ≤ 0.6 rad, |δ̇| ≤ 1.5 rad/s.
- LSD at p1_TUM: exactly M = tanh(ω_diff)·T_in (preload 0, lock_drive 1, lock_coast 0, no engine braking). ω ≥ 0 via derivative clamps.
- Powertrain (Python, 2 ms ZOH): T_axle = throttle·η·r_g·T_e(rpm); T_e = max(0, P(clip(n, 7875, 14640)))/ω_e; automatic gearbox (upshift > n_shift[g]; downshift < n_shift[g−1] − 1000; ≥ 0.1 s between shifts; instantaneous); gear-dependent engine inertia 1.4–7.8 kg m²; upshift map monotone.
- Brakes 15 kN·m × b, bias 0.49 (3675 N·m per front, 3825 per rear wheel).
- Track: `Track.polygon` from the racetrack-database centreline (1090 pts, L = 5445 m, width 4.8–9.0 m); footprint corners (2.468, ±1.0), (−2.132, ±1.0); max (w_inner·|κ|) ≈ 0.90 at s ≈ 4806 m (T14): near-singular for curvilinear coordinates.
- RL reference: `comparison/p1_tum_ghost_20261029_081152Z/rl/` (lap 2 = 101.26992540585894 s; no vertical/wheel states logged; front_lock_fraction 0.081, 17 spin upshifts).
- M4 16 GB RAM; casadi not installed.

## 2. Model: CasADi replica (`rlsetup/mlt/replica.py`)
- Parameters never typed by hand: built from the dict `OCDBackend.reset(p1_TUM)` passes to `Car.set_parameters` + `Powertrain` data; a test fails on any unread/missing key; dict sha256 in provenance.
- Treatments: **V** verbatim; **I** inactive clamp → path constraint keeping the trajectory where the clamp is inactive (audited, V5); **S** smooth equivalent with stated error (smax/smin with ε, tanh sign, logistic blend), tested in V4.

| Sub-model | Treatment |
|---|---|
| Body 3 DOF + heave/roll/pitch, aero torque, CoG terms | V |
| Wheel vertical, tyre spring | V; Fz clamp → I (Fz ≥ 50 N) |
| Springs, linear dampers, virtual ARB, jacking | V (assert parameter equalities) |
| Packers | I: bump ≤ gap − 2 mm |
| Heave | off, asserted |
| Kinematics, steer | V |
| Raw slip | V; clamp κ → I (|κ_raw| ≤ 0.9); max(3,v_x), max(1,v_x) → I (v_x,tyre ≥ 5 m/s) |
| Relaxation PT1 | V; clamp(|v_x|,10,100) → S (ε = 0.1 m/s, τ error ≤ 0.5 %) |
| MF5.2 pure + combined, all coefficients | V; copysign → S tanh(·/1e-4) |
| Aero ride-height polynomial, drag, pitch torque | V; [0,5] clamp → I; v_x > 0 (I) |
| Drivetrain + LSD | V (assert preload = 0, lock_coast = 0); low-speed inertia → I (v ≥ 8 m/s); ω ≥ 0 → I (ω ≥ 1 rad/s, no lock) |
| Steering PT1 with clamps | I, lossless: δ̇ = (δ_cmd − δ)/T, |δ̇| ≤ 1.5, |δ|, |δ_cmd| ≤ 0.6 |
| Engine map | S (clip ε = 50 rpm, max(0,P) ε = 1 kW); error table |
| Gearbox | S: static upshift map g(ω_axle), torque/inertia blended across shift points (≈ 20 rpm); "free gear" as sensitivity; P5b iteration if V7 shows > 0.1 s from gear mismatch |
| Brakes | V |
| ERS/DRS | off, asserted |

Two builds from the same equations: `replica.py` (continuous, smooth, for the OCP) and `replica_discrete.py` (OCD's exact step map: DoPri at h, wheel-speed lag, original clamps, real `Powertrain` logic; validation only). `ocd_harness.py` drives `ocd_native.Car` + `Powertrain` directly with full initial state; bit-identical to `OCDBackend.step` over 10 s. No ocd_first_lap edits.

## 3. Replica validation (P3 gate, before any OCP)
- **Level A (equation equivalence)** `replica_discrete` vs OCD@2 ms: A1 lockstep one-step error ≤ 1e-10 (≥ 20 000 steps), A2 2 s free run ≤ 1e-6 relative, A3 sub-model signals ≤ 1e-9. Any miss = transcription bug; blocks P4.
- **Level B (smooth replica vs real OCD)** CVODES (rtol 1e-10), same 2 ms ZOH inputs and gears, vs OCD@0.25 ms (and reported vs @2 ms). Excitations: E1 full throttle 15→85 m/s; E2 braking b 0.5/0.7 from 85 m/s; E3 step steer 0.01/0.03 rad at 30/50/70 m/s; E4 sine sweep 0.2→4 Hz ±0.02 rad at 50 m/s; E5 100 × 1 s windows of the RL lap from full logged states; E6 trail braking.
- Thresholds (PLACEHOLDER, only the architect may revise, with numbers shown to Ibrahim): |Δv_x| 0.02 m/s, |Δv_y| 0.01 m/s, |Δψ̇| 0.005 rad/s, |Δx|,|Δy| 0.05 m, |ΔFz| 1 % + 20 N, |Δκ| 2e-3, |Δα| 1e-3 rad, |Δω| 0.5 rad/s. Pass vs @0.25 ms: E1–E4, E6 all; E5 ≥ 95 % windows (all within 3×). B2: smooth gear map instead of imposed gears, reported.

## 4. OCP formulation (`rlsetup/mlt/ocp.py`)
- Spatial domain s ∈ [0, L] on a smoothed periodic C³ centreline (fallback: smoothed TUM raceline if T14 fails min(1 − nκ) ≥ 0.1 or (w_inner − 1)|κ| ≤ 0.8). Edges by ray-casting onto `Track.polygon` every 0.5 m, made conservative (±2.5 m window min, periodic B-spline, inset by fit error; tested inside the polygon).
- States (32): n, ξ = ψ − ψ_ref, v_x, v_y, ψ̇, z, v_z, φ, φ̇, θ, θ̇, z_w×4, v_zw×4, κ×4, α×4, ω_FL, ω_FR, ω_axle, ω_diff, δ; time as quadrature. Controls: δ_cmd, throttle, brake (overlap allowed, reported).
- ṡ = (v_x cos ξ − v_y sin ξ)/(1 − nκ); n′ = (v_x sin ξ + v_y cos ξ)/ṡ; ξ′ = ψ̇/ṡ − κ; other x′ = f(x,u)/ṡ.
- **Footprint constraint (default, Q1):** for each corner c, Δs_c = x_c cos ξ − y_c sin ξ, n_c = n + x_c sin ξ + y_c cos ξ − ½κΔs_c²/(1 − κn); −n_right(s + Δs_c) + 0.05 ≤ n_c ≤ n_left(s + Δs_c) − 0.05. Approximation error ≤ 0.02 m over 10⁴ random poses, else ε grows. Arbiter: `Track.contains_car` every 0.1 m on the solution and on V7. Secondary variant: CoG rule.
- Bounds: steering as §2, throttle/brake ∈ [0,1], I-constraints, v_x ≥ 8 m/s. Periodicity x(0) = x(L), u(0) = u(L) (flying lap).
- Objective J = t(L)/100 + R, R = w_δ Σ(Δδ_cmd)² + w_p Σ(Δthrottle² + Δbrake²) with R ≤ 1e-4 J.

## 5. Transcription and solver
- Legendre–Gauss–Radau (Radau IIA) d = 3 (L-stable for the 2–3 ms tyre/wheel modes) [P].
- Mesh N0 = 1100 intervals (≈ 5 m); refinements 2N0 (h), d = 5 at 2N0 (p); 4N0 optional on the lab PC (RAM).
- Fixed nominal scaling (v 50, ω 200, z 0.01, κ 0.1, α 0.05, n 5, ξ 0.1, δ 0.1, Fz 1e4).
- IPOPT via casadi (pinned, expand, optional codegen): mumps (mem 1000 %), exact Hessian, tol/constr_viol/compl_inf 1e-8, adaptive μ, max_iter 5000, gradient-based scaling; warm starts with bound push 1e-9, μ_init 1e-5. Only `Solve_Succeeded` counts.
- Two stages per guess: A feasibility (track-to-guess with L1 track slacks), B minimum time. P4a: 600 m open segment (T14 approach) first.
- Guesses: G1 RL p1_TUM lap 2 (re-logged with full state, P0); G2 target-car QSS on TUM raceline; G3 target-car QSS on Track.path; G4 debug lap geometry; G5 centreline at 0.9 × G3 speed.

## 6. Verification protocol (P5)
- **V0** Stage A on G1 gives a feasible replica lap of 101.27 ± 0.5 s.
- **V1** KKT: Solve_Succeeded; defects ≤ 1e-6; path violation ≤ 1e-6; report dual infeasibility/complementarity; no inertia correction in final iterations ⇒ evidence of a strict local minimiser (SOSC) [P], else "first-order point only".
- **V2** Mesh convergence: |T(2N0,d3) − T(N0,d3)| ≤ 20 ms, |T(2N0,d5) − T(2N0,d3)| ≤ 5 ms; a-posteriori CVODES interval error ≤ 1e-3 [P Betts].
- **V3** Multi-start: ≥ 4 of G1–G5 converge, all within 0.10 s of the best; otherwise "best of K local optima" with Δt per corner.
- **V4** Smoothing widths ÷ 10: |ΔT| ≤ 5 ms; regularisation ÷ 10: |ΔT| ≤ 1 ms.
- **V5** Inactive-clamp audit: margins and multipliers; any active one discussed with its time cost.
- **V6** Segment re-simulation in real OCD@2 ms (0.5 s windows from x*): ≥ 95 % within 2× Level B thresholds; stitched |T − T*| ≤ 0.05 s.
- **V7** Closed-loop tracked lap in real OCD (certified number): `OCDBackend` p1_TUM, 2 ms, footprint check; s-indexed feedforward u*(s) + small feedback (k_n 0.02 rad/m, k_ξ 0.3, k_r 0.02, k_v 0.1 PLACEHOLDER, ≤ 10 tuning runs). Accept: clean lap, T_ach − T* ≤ 0.30 s, feedback RMS ≤ 10 % of feedforward steering (≤ 0.05 pedals), max |n − n*| ≤ 0.3 m. Also 25 Hz ZOH variant.
- **V8** Diagnostics: target-car QSS on TUM line, safe line, OCP line (expect QSS(OCP line) ≤ T*); expect T* ≤ 101.27 s (blocks the report otherwise).

## 7. Phases (≈ 16 working days + 30 % buffer)
| Phase | Work | Effort | Acceptance |
|---|---|---|---|
| P0 Freeze | casadi extra `[mlt]`; versions/hashes; full-state replay of the RL lap at 2 ms; target-car QSS | 0.5 d | replay lap 2 = 101.26992540585894 s ± 1e-9 |
| P1 Track | reference spline, conservative edges, corner function, singularity check | 1.5 d | edges inside polygon; corner error ≤ 0.02 m |
| P2 Replica | params, replica, replica_discrete, smooth powertrain, harness, tests | 3 d | key test; harness bit-identical; error tables |
| P3 Validation | Level A + B | 2 d | §3 criteria; `docs/results/mlt/replica_validation.json` |
| P4 OCP | P4a segment, full lap N0 from G1 | 4 d (main risk) | Solve_Succeeded; V0; T* ≤ 101.27 |
| P5 Verification | V1–V8, G2–G5, refinements (+P5b) | 3 d (+1) | §6 criteria |
| P6 Report | `docs/reports/MLT_OCP_report.md` + figures | 2 d | lead review |
Code: `rlsetup/mlt/*`, `scripts/mlt/*`, `tests/test_mlt_*.py`; large artefacts `runs/mlt/<UTC>/` (git-ignored); summaries/solution CSV/logs in `docs/results/mlt/`. Fallback if P4 fails: quasi-steady tyre relaxation (no longer verbatim; Ibrahim's call, Q3).

## 8. Report structure
Problem; model (V/I/S table, PLACEHOLDERs); track model; transcription and solver; replica validation; results (T*, T_ach, line, speed, g-g use, Δt per corner vs RL, inputs, gears, ride height, pedal overlap); verification V0–V8; limitations (local optimum only, smoothed gearbox, continuous vs 2 ms, placeholders, model exploits); comparison table (T*, T_ach, RL 101.27 like-for-like; debug 102.99, QSS, TUM 94.886 as context); reproduction commands.

## 9. Risks (metric)
Stiffness/IPOPT stalls (iterations, restoration, inf_pr); T14 singularity (min(1 − nκ)); model exploits (min ride height < 0, |Fx_RL − Fx_RR|, pedal overlap, active I-constraints); gearbox mismatch (V7 share, B2 Δv); feedback masking infeasibility (V7 RMS share, V6 ΔT); smoothing (V4); multiple local optima (V3 spread); memory (MUMPS RSS > 12 GB → lab PC); edge conservativeness (ε, inset; CoG variant).

## 10. Questions for Ibrahim
1. Track rule for the headline: footprint (like-for-like with RL 101.27 s, recommended), CoG, or both (+1 d)?
2. Gearbox meaning: OCD's automatic gearbox (recommended) or free gear choice (as sensitivity)?
3. Effort ≈ 16 days + buffer: approve full scope? Fallback to quasi-steady tyre relaxation if P4 does not converge?
4. 15 kN·m brakes PLACEHOLDER (OPEN §11.6): proceed with results labelled conditional?
5. Negative ride height (extrapolated aero map, no ground contact): leave unconstrained and report (recommended), or add rh ≥ 0?

## 11. To verify (web-researcher)
TUMFTM `opt_mintime_traj` transcription/options; Massaro & Limebeer (VSD ≈ 2021) verification practice; Perantoni & Limebeer 2014 transcription and mesh convergence; Dal Bianco, Lot & Gadola 2018 (tyre relaxation/suspension stiffness handling); IPOPT inertia-correction ⇒ SOSC conditions (Wächter & Biegler 2006; Nocedal & Wright Thm 12.6); Radau IIA L-stability vs Hermite–Simpson (Hairer & Wanner II; Biegler 2010; Kelly 2017); casadi macOS arm64 / Python 3.12 wheel with IPOPT + MUMPS.
