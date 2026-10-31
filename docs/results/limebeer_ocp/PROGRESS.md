# PL14 minimum-lap-time OCP: running progress log

Spec: lead task of 31 Oct 2026; model source: docs/research/limebeer_ocp_model.md only.

## M0 (31 Oct) setup
- casadi 3.8.2 installed with `uv pip install --python .venv/bin/python casadi==3.8.2` (venv has no pip).
  IPOPT + MUMPS verified on a toy NLP: Solve_Succeeded.
- pyproject: optional extra `[ocp]` = casadi==3.8.2, scipy.
- Design (frame = PL14's SAE frame: x fwd, y right, z down, yaw CW+, n>0 right of centreline, C>0 right turn):
  states n, xi, u, v, omega; controls delta, T (total longitudinal demand: drive = max(T,0) on rear,
  brake = min(T,0) split beta front / 1-beta rear); algebraic kappa_ij (4 slips), N_ij (4 positive normal loads).
  Algebraic eqs: (23)-(26) loads, front Ffrx=Fflx (34), front sum, rear sum, diff (35).
  Power: max(T,0) * R * mean rear wheel spin <= Pmax.
- Next: write rlsetup/ocp/pl14_model.py + tyre tests.

## M1 (31 Oct) model + track
- rlsetup/ocp/pl14_model.py + tests/test_ocp_model.py: 12 tests pass (free rolling 0, signs, symmetry,
  peak mu exact at both loads, combined slip inside ellipse, straight-line equilibrium, turn signs, diff/brake split).
- FINDING: printed A.13 S = pi/(2 atan Q) puts the force peak at rho = 0.750 (not 1): peak slip = 0.75*kappa_max.
  Kept as printed; peak mu exact. Pinned in test.
- rlsetup/ocp/track.py: PL14 auxiliary OCP smoothing. Sweep of r [m^6] (fit RMS / max [m], Cmax [1/m]):
  Catalunya L=4649.8 m (PL14: ~4655): r=1e3 0.016/0.20/0.089; 1e4 0.037/0.47/0.076; 1e5 0.086/0.65/0.067; 1e6 0.245/1.91/0.057; 1.5e9 4.3/17.3/0.025
  Shanghai L=5445.2 m: 1e3 0.020/0.22/0.161; 1e4 0.054/0.53/0.144; 1e5 0.146/1.37/0.120; 1e6 0.34/2.91/0.093
  Chosen r = 1e4 m^6 (ASSUMPTION). Min half-widths: Catalunya right 4.36 / left 4.22 m; Shanghai 5.16/4.82 m.
- Next: solve.py (trapezoidal, scaled, periodic), first 4 m Barcelona solve.

## M2 (31 Oct) first solve works
- rlsetup/ocp/solve.py (trapezoidal, PL14 scaling, MX+map, IPOPT/MUMPS, 2 threads env).
- Barcelona, PL14 car, beta 0.60, corridor = half-width - 0.9 m, ds 4 m (1162 nodes), QSS guess (96.5 s):
  Solve_Succeeded, 118 iters, 14.3 s, lap 82.233 s (penalty 0.019), max scaled eq residual 2.4e-7.
  vs PL14 82.57: -0.41 %.
- Next: driver script scripts/ocp_pl14.py (mesh 4->2->1, 3 betas, +-4 m, guesses, re-integration, outputs).

## M3 (31 Oct) driver + verify + tests
- scripts/ocp_pl14.py CASE... (cases: bcn_beta055/060/065, bcn_pm4, bcn_guesses, bcn_reint, sha_pl14, sha_sf70h,
  sha_{pl14,sf70h}_betaXXX). Appends to results.json per mesh; writes solution_<case>_1m.csv and _warm_<case>.npz.
- rlsetup/ocp/verify.py: IDAS re-integration per interval + 50 m chained windows.
- tests/test_ocp_solve.py (circle track: curvature sign, corridor, steady-cornering OCP) -> 14 OCP tests pass in 0.5 s.
- Observed at 4 m: node-to-node ringing in delta/omega (trapezoid, fast yaw mode h*lambda ~ 2.6 at 4 m);
  check whether it vanishes at 1 m.
- Running: bcn_beta060 chain (4 m done: 82.233 s).

## M4 (31 Oct) Barcelona beta 0.60 mesh chain DONE
- 4 m 82.2325 s (118 it, 14.9 s) -> 2 m 82.2950 (62 it, 15.4 s) -> 1 m 82.3347 (54 it, 26.1 s); all Solve_Succeeded.
  1 m vs PL14 82.57: -0.29 %. Mesh deltas +0.062, +0.040 s.
- Ringing at 1 m is local (s ~ 4168 m, 8-13 nodes in delta/omega); max sideslip 5.4 deg.
- Next (queued): bcn_beta055 bcn_beta065 bcn_pm4 bcn_guesses bcn_half bcn_reint sha_pl14 sha_sf70h.
- NOTE: harness blocked writing REPORT.md from the subagent; the REPORT text goes in the handback message.

## M5 (31 Oct) Barcelona all cases (all Solve_Succeeded)
- beta 0.55: 82.269 / 82.3325 / 82.3724 (4/2/1 m); beta 0.65: 82.3379 / 82.4008 / 82.4411; beta 0.60: 82.2325/82.2950/82.3347
- beta 0.60 at 0.5 m (warm from 1 m): 82.3382 (53 it, 61 s): 1->0.5 m delta +0.0035 s -> mesh-converged ~82.34 s (-0.28 % vs 82.57).
- +-4 m corridor (ML16): 83.1797 / 83.2385 / 83.2518 (+0.92 s vs half-width corridor; +0.83 % vs PL14).
- Initial guesses at 4 m: QSS mu1.5 82.2325, QSS mu1.0 82.2325 (same), constant 40 m/s 82.7819 (DIFFERENT local optimum, +0.55 s).
- Running: bcn_reint, sha_pl14, sha_sf70h.

## M6 (31 Oct) verification
- const-40 guess local optimum: loses 0.31 s at s~3400-3600 and 0.14 s at s~4200 (different line, n differs up to 0.8 m);
  continued to 1 m it gives 82.98 s (vs 82.33). Two QSS guesses agree to 1e-4 s.
- Re-integration (IDAS rtol/atol 1e-10, per interval from collocated nodes, linear controls): IDAS fails on 100/1162 (4 m)
  and 195/4650 (1 m) intervals: all in threshold braking with the front tyres exactly at the force peak
  (dFx/dkappa = 0 -> singular algebraic Jacobian; not a solution defect). On the rest, 1 m max |err|: n 1.2 mm,
  xi 4.4e-4 rad, u 0.9 mm/s, v 3.8 cm/s, omega 0.038 rad/s; RMS v 0.8 mm/s. 4 m: max n 5.7 cm, v 0.22 m/s.
  Lap time re-integrated (trapezoid on failed intervals) 82.3355 vs 82.3347 s at 1 m; 82.2427 vs 82.2326 at 4 m.
- 50 m chained windows: 62/92 complete (1 m); median |err| n 0.1 mm, u 1.7e-5 m/s.

## M7 (31 Oct) sensitivity + Shanghai started
- bcn_r1e6 (curvature smoothing r = 1e6 m^6, PL14's number read as SI): 82.1126 / 82.1199 / 82.1282 (4/2/1 m), all Succeeded; -0.21 s vs r=1e4.
- sha_pl14 4 m: 97.6934 s (227 it, 46.6 s). sha_pl14 2/1 m and sha_sf70h running (queue2, /tmp/ocp/queue2.log).
- sha_pl14: 97.6934 / 98.0917 / 98.3058 (4/2/1 m; 227/355/110 it; 47/110/74 s), all Succeeded. Mesh deltas +0.40, +0.21 -> NOT converged at 1 m;
  running 0.5 m (case sha_pl14_half). sha_sf70h chain running.
- sha_pl14 0.5 m: 98.4086 s (213 it, 325 s). Deltas 0.40/0.21/0.10: FIRST-ORDER convergence (trapezoid is 2nd order).
  Cause found: Shanghai hairpin (s~4800, centre-line C=0.144, R 7 m) -> u hits the u_min = 10 m/s bound at s~4817 and
  delta/omega saw-tooth there (stiff yaw mode: s-domain eigenvalue ~1/u^2, h*lambda >> 1 at 10 m/s).
  Penalty grows with refinement (0.018 -> 0.070 s). Barcelona min speed 24.6 m/s, much milder.
- Fix in progress: speed-adapted non-uniform mesh (h ~ ds * clip(u/u_ref)), lower u_min; keep uniform results too.
- Added speed-adapted mesh option (SolveSettings.mesh_u_ref, h = ds*clip((u/u_ref)^2, 0.1, 1)); verify.py handles variable h. 16 tests pass.
- Circle probe: steady optimum 9.6978 s for ds 6/3/1.5 m, but ds 3 m with default weights stopped at a non-steady local optimum 9.7307 -> local optima exist even on a circle.
