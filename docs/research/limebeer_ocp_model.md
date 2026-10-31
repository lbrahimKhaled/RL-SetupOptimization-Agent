# Limebeer-group minimum-lap-time OCP: model, parameters, solver (web-researcher, 31 Oct 2026)

Status: complete for what could be accessed. Items marked NOT VERIFIED could not be read.
Notation: "SAYS" = the source says (quoted or transcribed from the PDF text); "READING" = my interpretation.

## 0. Sources actually read (full text)

| Key | Source | Access |
|---|---|---|
| **PL14** | G. Perantoni & D.J.N. Limebeer, "Optimal control for a Formula One car with variable parameters", *Vehicle System Dynamics* 52(5):653–678, 2014, DOI 10.1080/00423114.2014.889315. Accepted manuscript dated 20 Jan 2014, 37 pp. | ORA: https://ora.ox.ac.uk/objects/uuid:ce1a7106-0a2c-41af-8449-41541220809f/files/m776aa23411ad9d78c36f96620a0e0f0b (read in full) |
| **LPR14** | D.J.N. Limebeer, G. Perantoni & A.V. Rao, "Optimal control of Formula One car energy recovery systems", *Int. J. Control* 87(10), 2014. ORA manuscript dated 7 Mar 2016, 39 pp. | ORA: https://ora.ox.ac.uk/objects/uuid:db0437cc-0b4f-495d-96d5-524ea4bfda5e/files/m0745125c68504d40553beb40568db35a (read in full) |
| **ML16** | M. Imani Masouleh & D.J.N. Limebeer, "Optimizing the aero-suspension interactions in a Formula One car", *IEEE Trans. Control Syst. Technol.* 24(3), 2016. ORA manuscript, 16 pp. | ORA: https://ora.ox.ac.uk/objects/uuid:856dd2b0-3df9-4daa-a4f7-9462dc7cd2f8/files/m3fb7226495d64be875ddfdc868095771 (read in full) |
| **TUM** | TUMFTM `global_racetrajectory_optimization`, `opt_mintime_traj/src/opt_mintime.py` (last commit touching it 6e76953, 2021-04-01) and `params/racecar.ini` (branch master) | https://github.com/TUMFTM/global_racetrajectory_optimization (read via GitHub page; summary, not line-by-line) |
| **PyPI** | casadi 3.8.2 release JSON | https://pypi.org/pypi/casadi/3.8.2/json |

Correction to `docs/research/limebeer_lap_times.md`: in the PL14 manuscript, the **vehicle/tyre parameters are Appendix B, Tables 3–4**; Tables 1–2 are the setup-optimisation results. Equation/table numbers below are those of the ORA manuscript; the typeset VSD version may number differently (NOT VERIFIED, publisher PDF not accessible).

## 1. PL14 model (the D-check target)

### 1.1 What the model is (SAYS, §1, §2.2)
"The car model is described in Section 2.2 and has longitudinal, lateral and yaw freedoms. The model has a combined-slip magic-formula-like tyre description, longitudinal and lateral load transfer effects, four-wheel braking and a limited-slip differential." Quasi-static load transfer (algebraic), no wheel-spin dynamics, no suspension/roll/pitch/heave states (§2.3.2: "the model being used here has no pitch, roll or heave freedoms").

### 1.2 Track model, curvilinear coordinates (SAYS, §2.1, eqs 1–8)
- s = centre-line arc length; n = lateral offset of the mass centre from the centre line; θ = track heading; C = 1/R curvature; ψ = car yaw; ξ = ψ − θ (car heading relative to track); N = track half-width.
- (1) dx = ds cos θ; (2) dy = ds sin θ; (4) C = dθ/ds.
- (6) ṡ = (u cos ξ − v sin ξ) / (1 − nC)
- (7) ṅ = u sin ξ + v cos ξ
- (8) ξ̇ = ψ̇ − C ṡ
- Track limit: "we can constrain the vehicle to drive on the track by simply enforcing |n| ≤ N" with footnote a: "The real track boundary constraints apply to the positions of the vehicle's tyres with respect to the track edges. In order to simplify the problem the boundary constraints are enforced on the vehicle's centre of mass, with the track half-width reduced by the vehicle half-width." PL14 does not give the numeric N used.
- ML16 §V (same group, same Barcelona track) gives the number: "the car's mass centre is constrained to remain within ±4 m of the track centre. The track width is approximately 12 m on almost all parts of the track. This constraint ensures that the car's front wheels remain on the track whatever the vehicle's orientation". READING: ±4 m is the best available value for the PL14 replication too, but PL14 itself does not state it (NOT VERIFIED for PL14).

### 1.3 Independent variable s (SAYS, §2.1.2, eqs 14–16, 20–22)
- (14) S_f = (ds/dt)^−1 = (1 − nC) / (u cos ξ − v sin ξ)
- (15) dn/ds = S_f (u sin ξ + v cos ξ)
- (16) dξ/ds = S_f ω − C, with ω = ψ̇
- (20–22) du/ds = S_f u̇, dv/ds = S_f v̇, dω/ds = S_f ω̇
- §3.1: "f(·) contains equations (15), (16), (20), (21) and (22), with the associated states given by n, ξ, u, v and ω." Cost: "The Lagrange cost function is l(·) = S_f^−1" — READING: as printed this is inconsistent; lap time = ∫ S_f ds, so the stage cost in s must be S_f. Implement l = S_f.
- Boundary: "the track closure condition x̃(s_f) − x̃(s_0) given in g_b(·)" (periodic states).

### 1.4 Body equations (SAYS, §2.2, eqs 17–19)
```
M du/dt = M ω v + Fx
M dv/dt = −M ω u + Fy
Iz dω/dt = a (cos δ (Ffry + Ffly) + sin δ (Ffrx + Fflx))
         + wf (sin δ Ffry − cos δ Ffrx) − wr Frrx
         + wf (cos δ Fflx − sin δ Ffly) + wr Frlx − b (Frry + Frly)          (17)
Fx = cos δ (Ffrx + Fflx) − sin δ (Ffry + Ffly) + (Frrx + Frlx) + Fax          (18)
Fy = cos δ (Ffry + Ffly) + sin δ (Ffrx + Fflx) + (Frry + Frly)                (19)
```
"The rear-wheel tyre forces are expressed in the vehicle's body-fixed reference frame, while the front tyre forces are expressed in a steered reference frame" (§2.3). Both front wheels have the same steer δ (Fig. 4).

### 1.5 Load transfer, quasi-static (SAYS, §2.3.1–2.3.2, eqs 23–31)
```
0 = Frrz + Frlz + Ffrz + Fflz + M g + Faz                                     (23)
0 = wr (Frlz − Frrz) + wf (Fflz − Ffrz) + h Fy                                (24)
0 = b (Frrz + Frlz) − a (Ffrz + Fflz) + h Fx + (aA − a) Faz                   (25)
Ffrz − Fflz = D (Ffrz + Frrz − Fflz − Frlz),  D ∈ [0,1]  (roll balance)       (26)
```
- Sign: tyre normal forces are non-positive (z down): "(28) Fz = min(F̄z, 0)"; positive values are "non-physical 'forces of attraction'".
- Light-wheel handling (31): mechanics (23)–(25) enforced on the clipped loads Fz; the roll balance (26) on the unclipped F̄z: `[A1 0; 0 A2][Fz; F̄z] = [c; 0]`, solved inside the NLP. "There is a circular dependency between tyre loads and tyre forces ... treated as a nonlinear set of algebraic equations that is solved as part of the optimising NLP".
- Fx, Fy in (24)–(25) are the totals of (18)–(19) ("the longitudinal inertial force acting on the car's mass centre; see (18)"). Downforce acts at the centre of pressure, aA behind the front axle; it enters the pitch balance only through (aA − a) Faz in (25).

### 1.6 Aerodynamics (SAYS, §2.4, eqs 32–33)
"(32) Faz = 0.5 Cl ρ A u²" and "(33) Fax = −0.5 Cd ρ A u²", "which act in the car's body-fixed z- and x-axis directions respectively"; applied "at the centre of pressure"; "aerodynamic side forces, yawing moments and pitching effects have been neglected." Uses u (body longitudinal speed).

### 1.7 Brakes, differential, engine (SAYS, §2.5, eqs 34–35)
- "(34) 0 = max(ωfr, 0) max(ωfl, 0)(Ffrx − Fflx)" (equal front braking forces unless a wheel locks).
- "(35) R(Flrx − Frrx) = −kd(ωlr − ωrr)" limited-slip differential; "The special cases of an open- and a locked-differential correspond to kd = 0 and kd arbitrarily large".
- §2.5 mentions "braking pressures between the front and rear axles satisfying some design ratio" but **no brake-ratio value is given** in PL14 or LPR14. NOT VERIFIED.
- **Engine power limit: not described anywhere in the PL14 manuscript.** LPR14 (same car, Appendix B Table 2) gives "Pmax Peak Engine Power 560/440* kW" (2013 / 2014 car). READING: PL14 most likely used 560 kW, but PL14 does not say so (NOT VERIFIED for PL14). LPR14 §4.2.1: without KERS the car "can operate only along the horizontal axis as far to the right as P_IC^max; the engine's peak power limit", constraint P_IC^max + Pkers − Pm ≥ 0 (29), Pm = "power delivered to the rear wheels".
- **Control vector: not listed explicitly in PL14.** The slip equations use wheel spin speeds ω_ij and steer δ. READING: controls are δ plus wheel-torque/slip quantities, with (34), (35) and the power limit as path constraints; exact choice NOT VERIFIED (perhaps in Perantoni's 2013 DPhil thesis, ORA uuid:b75fa69c-ebe0-4812-98e9-984d93476d37, which I could not read: served as octet-stream, text not extractable with my tools).

### 1.8 Tyre model (SAYS, Appendix A, eqs A.1–A.18)
```
κ = −(1 + R ωw / uw)                                       (A.1)
tan α = −vw / uw                                           (A.2)
µx,max = (Fz − Fz1)(µx2 − µx1)/(Fz2 − Fz1) + µx1           (A.3)
µy,max = (Fz − Fz1)(µy2 − µy1)/(Fz2 − Fz1) + µy1           (A.4)
κmax   = (Fz − Fz1)(κ2 − κ1)/(Fz2 − Fz1) + κ1              (A.5)
αmax   = (Fz − Fz1)(α2 − α1)/(Fz2 − Fz1) + α1              (A.6)
κn = κ/κmax ;  αn = α/αmax                                  (A.7–A.8)
ρ = sqrt(αn² + κn²)                                         (A.10)
µx = µx,max sin(Qx arctan(Sx ρ)) ;  µy = µy,max sin(Qy arctan(Sy ρ))   (A.11–A.12)
Sx = π / (2 arctan(Qx)) ;  Sy = π / (2 arctan(Qy))          (A.13–A.14)
Fx = µx Fz κn/ρ ;  Fy = µy Fz αn/ρ                          (A.15–A.16)
```
Wheel slip angles (A.17) and longitudinal slips (A.18), transcribed:
```
αrr = arctan( (v − ψ̇ b) / (u − ψ̇ wr) )
αrl = arctan( (v − ψ̇ b) / (u + ψ̇ wr) )
αfr = arctan( (sin δ(ψ̇ wf − u) + cos δ(ψ̇ a + v)) / (cos δ(u − ψ̇ wf) + sin δ(ψ̇ a + v)) )
αfl = arctan( (cos δ(ψ̇ a + v) − sin δ(ψ̇ wf + u)) / (cos δ(ψ̇ wf + u) + sin δ(ψ̇ a + v)) )
κrr = −(1 + R ωrr / (u − ψ̇ wr))
κrl = −(1 + R ωrl / (u + ψ̇ wr))
κfr = −(1 + R ωfr / (cos δ(u − ψ̇ wf) + sin δ(ψ̇ a + v)))
κfl = −(1 + R ωfl / (cos δ(u + ψ̇ wf) + sin δ(ψ̇ a + v)))
```
"Due to a paucity of publicly available Formula One tyre data, we have used the same tyre model on the front and rear wheels." Load interpolation is from Kelly's 2008 Cranfield thesis [11]. LPR14 Appendix A has identical equations (36)–(53) and identical Table 3.

READING (sign-convention warnings; settle by unit tests, do not copy blindly):
1. (A.2) has a minus sign, (A.17) does not; (A.1) gives κ = −2 at free rolling if ω > 0, yet (34) says a locked wheel has "non-positive" ω. The printed conventions are not mutually consistent. Pick one (SAE, z down, Fz ≤ 0) and test: free rolling → Fx = 0; drive slip → forward force; slip angle → force opposing lateral sliding.
2. Fz is signed (≤ 0) but Fz1 = 2000 N, Fz2 = 6000 N are positive: interpolate (A.3)–(A.6) with |Fz|.
3. ρ → 0 at zero slip: κn/ρ, αn/ρ need smoothing (e.g. ρ = sqrt(αn² + κn² + ε)).

### 1.9 PL14 parameters (SAYS, Appendix B, Table 3 tyre; Table 4 vehicle)
Table 3 (tyre):

| Symbol | Description | Value |
|---|---|---|
| Fz1 | reference load 1 | 2000 N |
| Fz2 | reference load 2 | 6000 N |
| µx1 | peak longitudinal friction coefficient at load 1 | 1.75 |
| µx2 | peak longitudinal friction coefficient at load 2 | 1.40 |
| κ1 | slip coefficient for the friction peak at load 1 | 0.11 |
| κ2 | slip coefficient for the friction peak at load 2 | 0.10 |
| µy1 | peak lateral friction coefficient at load 1 | 1.80 |
| µy2 | peak lateral friction coefficient at load 2 | 1.45 |
| α1 | slip angle for the friction peak at load 1 | 9° |
| α2 | slip angle for the friction peak at load 2 | 8° |
| Qx | longitudinal shape factor | 1.9 |
| Qy | lateral shape factor | 1.9 |

Table 4 (vehicle):

| Symbol | Description | Value |
|---|---|---|
| M | vehicle mass | 660 kg |
| Iz | moment of inertia about the z-axis | 450 kg m² |
| w | wheelbase | 3.4 m |
| a | distance of the mass centre from the front axle | 1.8 m |
| b | distance of the mass centre from the rear axle | w − a |
| h | centre of mass height | 0.3 m |
| Droll | roll moment distribution (fraction at the front axle) | 0.5 |
| wf | front wheel to car centre line distance | 0.73 m |
| wr | rear wheel to car centre line distance | 0.73 m |
| R | wheel radius | 0.33 m |
| kd | differential friction coefficient | 10.47 N m s/rad |
| Cd | drag coefficient | 0.9 |
| Cl | downforce coefficient | 3.0 |
| A | frontal area | 1.5 m² |
| ρ | air density | 1.2 kg/m³ |
| aA | centre of pressure to front axle distance | 1.9 m |
| bA | centre of pressure to rear axle distance | w − aA |

Not in PL14: engine power (LPR14 Table 2: 560 kW, 2013 car), brake ratio (not found anywhere), g (§3.3.1 says "approximately g=9.8 ms−2"), track half-width N (ML16: ±4 m on the mass centre).
LPR14 Table 2 repeats every PL14 Table 4 value above except kd, given as "100 N/rpm" (READING: unit ambiguous; if N·m per rpm, ≈ 955 N m s/rad), and adds Pmax 560/440 kW, P_kers^max 60/120 kW, M 660/710 kg; ρ = 1.2, A = 1.5 m² in the Appendix B text.

### 1.10 Transcription, solver, Barcelona result (SAYS, §3.2–3.4, §4.1, §4.3)
- "the MATLAB-based optimal control transcription toolbox ICLOCS [18] was used with trapezoidal integration to convert the optimal control problem into a NLP problem ... (IPOPT [19])". Analytical derivatives via Maple, compiled C.
- Trapezoidal NLP: cost ≈ Σ (s_{i+1} − s_i)(l_i + l_{i+1})/2 (44); defects (T1 ⊗ In)X = (diag(Sd)T2 ⊗ In)F (46).
- Scaling (§3.3.1): "All the mass- and length-related quantities are scaled so that the vehicle has unit mass and unit length. Time is scaled so that the acceleration due to gravity is unity too".
- Smoothing (48): max(x,0) ≈ (x + sqrt(x² + ε))/2, min(x,0) ≈ −(−x + sqrt(x² + ε))/2, "values in the range 10^−4 ≤ ε ≤ 10^−2".
- Track curvature (§2.1.1, eqs 9–13): auxiliary OCP, control ũ = dC/ds, states C, θ, x, y; minimise ∫(r ũ² + (xc − x)² + (yc − y)²) ds with closure enforced; xc, yc = "given GPS data for the track centre line". §4.3: "The centre-line curvature estimate used here corresponds to the case r = 10^6". The GPS data source is not named and not published with the paper.
- **Barcelona (§4.3)**: "The computed lap time is 82.43 s, which was obtained using a fixed uniform grid that will be used for all the calculations in this section." Fig. 11: "the predicted optimal lap time approaches an asymptotic value of 82.57 s"; "the number of samples is approximately N = 4655/∆s"; "t_NLP ≈ 5·10^−3 N^1.5"; "Once the mesh length falls below 0.5 m, solution times of the order of hours can be expected." §5: "less than fifteen minutes ... using a sampling distance of approximately 2 m". The spacing behind 82.43 s is not stated. READING: on Fig. 11's curve, 82.43 s sits around ∆s ≈ 2–3 m; the mesh-converged 82.57 s is the cleaner target.
- Setup study (Table 1, one at a time): a 1.816 m → 82.425 s; aA 2.025 m → 82.212 s; Droll 0.629 → 81.947 s; kd 2552.3 N m s/rad ("essentially locked") → 81.353 s; mass ≈ 0.035 s/kg. Table 2 (joint): a 1.9 m (box-constrained), aA 2.1789 m, Droll 0.4537, kd 2218.1 → 80.94 s.

## 2. Related Limebeer-group numbers (same base car)
- **LPR14 (GPOPS-II, LGR, hp-adaptive)**: speed-dependent CD(u), CL(u), aA(u) given only as a plot (Fig. 4, not tabulated). "The predicted lap time in this case is 80.50 s ... These results are in broad agreement with those given in [6], with exact agreement achieved when the simple constant-coefficient aerodynamic model given in [6] is used." KERS 2013 car: 80.23 s. 2014 car (710 kg, 440 kW, ERS): race lap 82.43 s; qualifying 82.06 s; 75 kg fuel 85.23 s. Initial mesh "twenty segments with four mesh points per segment"; "fifteen grid refinement steps". READING: 82.43 s appears in both PL14 (2013 car, constant aero) and LPR14 (2014 ERS car); do not confuse them.
- **ML16 (GPOPS-II; downforce split per axle, applied at the axle centres, drag at the mass centre; roll constraint φf = φr replaces Droll)**: "Nominal values for the car and tire parameters can be found in [12]" (= PL14). Table III: (A) rigid suspension, no aero sensitivities 83.10 s; (B) with sensitivities 83.54 s (text: 83.45 s); (C) suspension baseline 84.55 s; (I) all parameters 82.75 s; (J) point-wise aero balance 82.49 s. Table II nominal: CfL 1.2000, CrL 0.9494, CD 0.9455. Regularisation (52): J = ∫(1 + Σ εi ui²) dt, "the computed 'true' lap time is typically 0.05 s smaller". ε range "10^−5 ≤ ε ≤ 10^−2". Scaling: car length, car mass, g = 1.

## 3. Open-source cross-check: TUMFTM `opt_mintime` (SAYS via GitHub page summary; repo master, file last changed 6e76953 on 2021-04-01)
- Docstring: minimum-lap-time OCP, direct collocation + IPOPT, curvilinear abscissa, "double-track vehicle model with quasi-steady-state tyre loads".
- States: v, β, ω_z, n, ξ (+ optional powertrain thermal states, SOC). Controls: δ, f_drive, f_brake, γ_y (lateral load transfer as a control). s-domain: `sf = (1.0 - n*kappa) / (v*cos(xi + beta))`.
- Tyre: simplified Pacejka with load sensitivity: `mue * f_z * (1 + eps*f_z/f_z0) * sin(C*atan(B*alpha - E*(B*alpha - atan(B*alpha))))`; Kamm circle per wheel as a path constraint (no combined-slip formula).
- Loads: static + aero lift per axle + longitudinal transfer (cog_z) + γ_y split by k_roll.
- Power: `v * f_drive ≤ power_max`; drive/brake exclusivity via `f_drive*f_brake ≤ small negative`; brake split `k_brake_front`; first-order actuator lags t_delta, t_drive, t_brake.
- Collocation: degree d = 3, `ca.collocation_points(d, 'legendre')`; IPOPT `max_iter 2000`, `tol 1e-7`, `expand True`; objective = lap time + penalties on Δδ and ΔF.
- `params/racecar.ini` defaults (a 1200 kg electric car, not F1): wheelbase_front 1.6, wheelbase_rear 1.4, track widths 1.6 m, cog_z 0.38, I_z 1200, liftcoeff_front 0.45, liftcoeff_rear 0.75, dragcoeff 0.75, k_brake_front 0.6, k_roll 0.5, power_max 230 kW, f_drive_max 7000 N, f_brake_max 20000 N, delta_max 0.35; tyre B 10, C 2.5, E 1.0, eps −0.1, f_z0 3000 N, c_roll 0.013; optim width_opt 3.4, penalty_delta 10, penalty_F 0.01, mue 1.0.
- READING: TUM is a CasADi template for the s-domain double-track OCP but its tyre (no κ, no wheel slip) and parameters differ from PL14; use it for code structure, not for numbers.

## 4. casadi on macOS arm64 / Python 3.12
- SAYS (PyPI JSON): latest **casadi 3.8.2**, uploaded 2026-10-27; requires_dist numpy. No cp312-specific wheel; there is `casadi-3.8.2-cp311-abi3-macosx_11_0_arm64.whl` (55.3 MB).
- READING: a cp311-abi3 wheel uses the stable ABI and installs on CPython ≥ 3.11, so it serves Python 3.12 on Apple silicon. 3.8.2 is four days old; pinning `casadi==3.8.2` is fine if it imports and solves a toy NLP with IPOPT/MUMPS on our machine.
- IPOPT/MUMPS bundling: CasADi wiki (InstallationInstructions, last edited 11 Mar 2023): "The binaries include the solvers that are distributed with the tool (Sundials, qpOASES, CSparse) as well as IPOPT (shipped with the MUMPS linear solver; HSL is available through IPOPT's dynamic loader)." YAPSS docs (third party) say the pip IPOPT build uses SPRAL as default on Windows/Linux, MUMPS selectable via `linear_solver`. NOT VERIFIED for the 3.8.2 macOS arm64 wheel specifically; check locally with `casadi.nlpsol('s','ipopt',...)` and `ipopt.linear_solver: mumps`.

## 5. NOT VERIFIED / not accessible
- Leyton-Romero & Limebeer, "Laptime-driven optimisation of vehicle suspension systems", VSD 2026, DOI 10.1080/00423114.2026.2674200 (published 15 May 2026 per search metadata): T&F HTML and PDF both HTTP 403; Semantic Scholar has no abstract. Search snippet only: it optimises car parameters for minimum laptime following Perantoni & Limebeer and Masouleh & Limebeer, with an initial setup from an H2 approach on "a single-track vehicle model". Model level, parameters, lap times: NOT VERIFIED.
- Massaro & Limebeer, "Minimum-lap-time optimisation and simulation", VSD 59(7):1069–1113, 2021: Padova postprint URLs return 403. NOT VERIFIED.
- Perantoni 2013 DPhil thesis (ORA uuid:b75fa69c-…): downloaded but not text-extractable with my tools. NOT VERIFIED (likely source for controls, engine limit and brake ratio).
- Tremlett & Limebeer 2016 and Limebeer & Perantoni 2015 JDSMC Part 2: not read.
- Scholar Gateway connector was not reachable ("not connected") during this task.

## 6. Implications for us (interpretation, not from the sources)
1. PL14 gives everything needed for the planar model except three items: engine power (use LPR14's 560 kW and label it as an assumption), brake ratio (none published; options: free per-wheel braking with eq. 34-style equal-per-axle constraint only, or a fixed ratio chosen by us, decision for Ibrahim), and track half-width (ML16's ±4 m on the mass centre).
2. The D-check cannot be bit-exact: PL14's Barcelona centre line is a private GPS set smoothed with r = 10^6; we will use a different public centre line and width. A realistic acceptance band is needed (e.g. within ~1 % of 82.57 s), set by Ibrahim.
3. Replicate the PL14 numerics where cheap: s-domain, ~2 m mesh first then refine to ~0.5–1 m to show convergence, unit-mass/unit-length/g = 1 scaling, sqrt-smoothing with ε ≈ 1e-4–1e-2, periodic boundary conditions, l = S_f. CasADi + IPOPT (MUMPS) replaces ICLOCS; trapezoidal or low-degree Legendre/Radau collocation both fit.
4. Write unit tests for the tyre sign conventions before the OCP (free rolling → zero force; symmetric left/right; Fz interpolation with |Fz|), since the printed equations conflict.
5. Shanghai has no published Limebeer lap time, so the Barcelona D-check is the only external anchor; the TUM repo is a code-structure reference only.
