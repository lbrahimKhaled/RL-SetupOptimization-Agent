# Leyton-Romero & Limebeer (VSD 2026), "Laptime-driven optimisation of vehicle suspension systems": assessment

Read 1 Nov 2026 from the local PDF (`~/Desktop/RESEARCH/Leyton-Romero and Limebeer - 2026 - ….pdf`, 20 pages; text extracted, some Greek symbols lost, figures not seen). Page numbers = printed (PDF = printed + 1). [J] = our inference.

## Short answers
1. **Compare:** only relative setup gain (% of lap), trends and reporting format. Absolute lap times are not comparable: **NASCAR at Sonoma**, not F1.
2. **Equations into OCD:** OCD already has the paper's suspension structure (and more). The paper gives **no heave damper**, so it does not fix the sf70h_f1 porpoising. Useful: load-variation metric eq. (34) and axle-centre track-limit rule eq. (29), both Python-only in our repo.

## 1. Paper summary
- **Problem:** suspension setup minimising lap time on a 3D track (OCP), vs a road-holding baseline from H2 minimisation. Generic NASCAR, Sonoma "chute" (3.2 km, 12 turns, clockwise), LiDAR 1 m grid, ribbon surface eq. (1).
- **Model (§3, p.4–9):** 15 DOF; chassis (yaw, planar, roll, pitch, heave) + 4 unsprung heave; corner spring + damper eq. (4)–(6); dampers with separate compression/rebound eqs. (7)–(8); wheel rate k* = k/ν² eq. (9); ARB from unsprung displacement difference eq. (12); multibody Newton–Euler loads (13)–(25); tyre load F_z = k_t z_t eq. (30). No roll/pitch centres, anti-dive/squat, camber, compliance or heave element. Tyre: Kelly's empirical model (no camber/pressure). Aero: polynomial in roll, sideslip, front/rear ride height (eq. 2, Fig. 5), coefficients not printed (refers to ML16 [16] and [20]).
- **Optimisation (§4):** cost eq. (28) J = ∮(1 + R_i U_i²)(1/ṡ) ds; track limits on front and rear axle positions eq. (29); power (31); open diff (32)–(33); Gaussian-quadrature hp collocation, ADiGator, IPOPT. Two sequential 8-parameter OCPs: (1) 4 corner springs + 4 tyre stiffnesses (dampers nominal), (2) 4 compression + 4 rebound dampers. Per-corner (asymmetric). Bounds: springs 100–400 N/mm, tyres 200–400 N/mm; damper bounds not stated.
- **Baseline Car 1 (Appendix, Table A1):** H2 norm on a linear single-track model → softest feasible settings (springs 100 N/mm, dampers 7 Ns/mm, tyres 200 N/mm).
- **Results (Car 1 → Car 2):**

| Case | Car 1 [s] | Car 2 [s] | Δ | Car 2 parameters |
|---|---|---|---|---|
| Turn 1 (Table 1, p.12) | 4.281 | 4.259 | −0.022 (−0.51 %) | springs lf/rf/lr/rr 100/400/100/400; tyres 200/200/400/200; load RMS higher at every corner |
| Turn 11 (Table 2, p.13) | 14.499 | 14.334 | −0.165 (−1.14 %) | springs 400/270/100/258; tyres 258/312/200/206; compression 38.5 all; rebound 7.02/8.45/16.5/70 |
| Full lap (Table 3, p.15) | 76.046 | 75.857 | −0.189 (−0.25 %) | springs 191/400/100/400; tyres 269/200/200/200; compression 38.5 all; rebound 15.7/70/28/70; load RMS lower at all corners |

  Trend: stiff right springs and high right rebound on a clockwise track. Load-variation metric eq. (34): RMS(F_z − mean)/mean. No validation against measured data (only "comparable with 2025 qualifying", no number).

## 2. What we can compare
- **Size of the setup gain (relative only):** L-R&L suspension-only −0.189 s (0.25 %, vs a weak lower-bound baseline). Ours, Barcelona QSS suspension knobs: roll front share 0.35 → −0.17 s (0.19 %); spread over roll settings 0.47 s (0.54 %). Same group F1/Barcelona: PL14 −1.49 s (1.8 %), ML16 −1.80 s (2.1 %) from balance/aero/diff. Wording: "suspension-only changes are worth a few tenths of a percent (L-R&L 2026: 0.25 %; our QSS 0.2–0.5 %); balance/aero/diff about 2 % (PL14, ML16)". QSS has no damper/transient effect and our aero map is a PLACEHOLDER.
- **Trend:** stiffer springs in aero-dominated fast corners loosely matches our QSS preference for higher roll stiffness — qualitative only [J].
- **Reporting format:** Car 1 vs Car 2 with Δlap + per-corner eq. (34) RMS maps directly onto p0 vs optimised setup.
- **Cost link:** eq. (28) has the structure of our reward's time + smoothness terms (supports PROJECT_CONTEXT §4).
- **F(p):** the setup loss's F(p) is essentially eq. (34).
- **Not comparable:** absolute lap times; their 15-DOF hp-collocation OCP vs our planar OCP; asymmetric per-corner parameters (we keep L = R), tyre stiffness as a pressure proxy (out of scope), no ARB/ride-height/aero knobs; open-loop optimal driver vs our RL driver; not reproducible (car data in [20], LiDAR not public [J]).
- **Thesis support:** lap time over road holding as objective (H2 setup slower; Table 1 lap optimum has higher load variation); model-free motivation (their method needs an explicit differentiable model, sequential spring/damper steps [J]). OCP stays a reference, never our setup solver (PROJECT_CONTEXT §12).
- **Consequence [J]:** suspension effects ≈ 0.2 s, so the setup agent needs driven-lap noise well below that. Not yet: controller TC target moves a closed-loop lap ≈ 2 s; RL is 1.08 s above QSS on Barcelona; 26/142 clean deterministic Barcelona evals.

## 3. Equations/data for OCD
OCD already has: double-track with chassis heave/roll/pitch + 4 unsprung heave; linear corner springs per axle; linear or four-way dampers with bump/rebound (covers eqs. 7–8); ARB; heave spring and damper; packers; roll/pitch-centre jacking (anti-dive/squat); tyre spring with lift-off; MF5.2 with camber. Lacks: aero in roll/sideslip (ride height only, constant drag), left/right asymmetry, 3D-track gravity terms.

| # | Candidate | Source | Mapping | Effort | Value |
|---|---|---|---|---|---|
| 1 | Normalised RMS tyre-load variation | eq. (34) | Python, our repo (eval/analysis; F(p)) | 1–2 h | High |
| 2 | Axle-centre track limit | eq. (29) | Python (env / track) | hours + re-baselining all laps | Medium; Ibrahim's call |
| 3 | Heave damping (porpoising fix) | not in paper | `heave_damping_ratio` default on sf70h_f1 | 1 h + sensitivity | High, but value still PLACEHOLDER |
| 4 | Asymmetric compression/rebound | eqs. (7)–(8) | already in OCD | 0 | confirms rebound/bump as a knob (OPEN #1) |
| 5 | Motion ratio k* = k/ν² | eq. (9) | Python, AC translator only | < 1 h | low–medium (AC phase) |
| 6 | Aero map in roll/sideslip | eq. (2); coefficients in ML16/[20] | ride-height fit Python if ML16 prints a map; roll/sideslip needs C++ | hours / days | high only with real data |
| 7 | 3D ribbon track + gravity | eq. (1), (20)–(22) | C++ + elevation data | high | low for scope |
Do not port: their ARB form, Kelly tyre, full Newton–Euler rewrite, asymmetric setups.

## 4. Limits and ambiguities
Car data missing (mass, inertia, geometry, unsprung masses, motion ratios, ARB, aero, tyre) → not reproducible; spring units (k vs k*) unclear; damper bounds and the nominal dampers in OCP 1 not stated; compression 38.5 at every corner and rebound 70 repeated (bound saturation?) [J]; Table 3 has two "lr" rows; unclear whether lap times include regularisation (ML16: ≈ 0.05 s); turn segments not additive; baseline at lower bounds (weak); no validation, single track/car, local optima; **no heave-damping value**.

## Questions for Ibrahim
1. OPEN §11.5: cite as closest related work and copy its Car 1 / Car 2 + eq. (34) reporting (not a numeric baseline)?
2. Heave-damping default for sf70h_f1: PLACEHOLDER (e.g. 0.3) or source it first?
3. Adopt eq. (29) as the off-track rule (changes every lap time)?
4. OPEN §11.8 λ: F(p) for reporting/tie-breaks only (λ = 0 for ranking)?
5. Aero: get ML16 aero-map coefficients; C++ roll/sideslip aero patch worth it?

## To research
ML16 aero-map coefficients; Limebeer et al. 2023 VSD 61(5) NASCAR masses/motion ratios/damper bounds; any published F1 heave-damper value; NASCAR Cup 2025 Sonoma pole time.
