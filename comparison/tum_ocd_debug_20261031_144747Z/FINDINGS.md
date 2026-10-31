# Why OCD does not drive TUM's 94.9 s lap: closed-loop debug (31 Oct 2026)

**Target.** TUM laptime-sim `none` (no ERS/DRS/recuperation): **94.886 s**. OCD's own QSS on TUM's raceline: 95.65 s (cached) and **95.73 s** for the debug car below (`qss_debug_car.json`).

**Result.** Best driven OCD lap **102.99 s** (f = 0.96); stop-rule lap **103.68 s** (f = 1.00): 8.1–8.8 s to TUM. The car's quasi-static limits are not the cause (QSS within 0.84 s of TUM); the time is lost when the car is driven through transients: traction-limited exits, hairpin braking/entries, and a locked-LSD artefact.

## Setup
- **Debug OCD:** temporary branch `tum-debug` of ocd_first_lap, worktree `../ocd_first_lap-tumdebug` (Python only; `vendor/` and `ocd_native` symlinked from the main checkout, same C++ source). `sf70h-tum-car` untouched.
- **Car `sf70h_tum_debug`:** `sf70h_tum`, ERS off, constant aero after setup translation (CzA 2.20 / 2.68, CxA 1.56, no ride-height terms), TUM rolling radius r(v) = 2.073/2π·(1 + (3.6v − 60)·0.045/200). Setup p1_TUM, 15 kN·m brakes.
- **Track `Track('Shanghai', tum_mode=True)`:** path = TUM raceline; off-track = CoG outside the track edges; the 4.6 × 2.0 m footprint check is logged only.
- **Controller** (`scripts/comparison/tum_closed_loop.py`, 50 Hz, references indexed by raceline arc length s): steering = feed-forward atan(L κ_TUM(s + 0.1 v)) + understeer term + feedback on CoG offset and course-angle error + yaw-rate damping + front slip-angle cap (11.5° → 7.5°, 25 → 60 m/s). Pedals = feed-forward r(v)(m e_i f² ax_TUM + drag + rolling) + PI on f·v_TUM(s) + traction control (rear slip target 0.08) + per-wheel ABS (−0.10). Gains are PLACEHOLDERs (about 10 hand-tuned development runs). Flying lap from 200 m before the line, lap 2 reported. Deterministic (rerun byte-identical).

## Lap 2 per speed scale (`runs.json`, `cl_f*.csv`)
| f | Lap 2 [s] | vs TUM | CoG on track | Footprint on track | max / RMS \|e\| [m] | v_max [km/h] |
|---|---|---|---|---|---|---|
| **1.00** (stop rule) | **103.680** | +8.79 | 92.9 % | 69.3 % | 16.0 / 3.29 | 284.7 |
| 0.98 | 103.495 | +8.61 | 95.7 % | 70.3 % | 12.9 / 2.54 | 278.9 |
| **0.96** | **102.985** | +8.10 | 98.1 % | 72.6 % | 10.3 / 1.54 | 273.2 |
| 0.94 | 103.696 | +8.81 | 98.5 % | 70.0 % | 7.8 / 1.08 | 267.5 |
| 0.92 | 104.627 | +9.74 | 100 % | 69.6 % | 3.3 / 0.42 | 261.8 |

f = 0.98–0.92 are diagnostics. If the CoG must stay on track the whole lap, the first qualifying f is 0.92 (104.63 s). TUM's line itself keeps the 4.6 × 2.0 m footprint inside the track on only 70.1 % of its length (CoG: 100 %), so a footprint track check cannot be met on TUM's line.

## Where the time goes (f = 1.00, `analysis.json`)
Δt(s) = t_OCD(s) − t_TUM(s); segments cut at TUM's braking points; each segment's Δt split by where the speed deficit grows: braking (brake on), traction (TC cutting), power (full throttle), lateral grip (an axle ≥ 0.9 of TUM's potential), controller (rest).

| s-range [m] | Δt [s] | v min TUM / OCD [m/s] | max \|e\| [m] | Binding (s) |
|---|---|---|---|---|
| 2988–4606 (corner at 3075, onto back straight) | 2.52 | 22.1 / 19.2 | 12.7 | traction 1.29, controller 0.48, braking 0.43, power 0.21 |
| 4606–5011 (T14 hairpin) | 2.43 | 16.0 / 10.2 | 16.0 | braking 1.23, traction 1.03 |
| 1464–2218 (T6 hairpin) | 1.47 | 22.6 / 17.5 | 7.7 | braking 0.59, traction 0.44, power 0.38 |
| 834–1464 (corner at 865 + exit) | 0.84 | 22.1 / 21.9 | 1.4 | traction 0.58, power 0.23 |
| 2423–2988 (corner at 2495 + exit) | 0.69 | 30.4 / 30.3 | 1.9 | traction 0.55 |

Whole lap: traction 4.40 s, braking 2.40 s, power 1.00 s, lateral grip 0.53 s, controller 0.47 s (total 8.79 s). Straights are fine (4000–4500 m: OCD 78.9–79.1 m/s, TUM 78.7–79.0).

Straight-line capability at equal speed (|ay| < 5 m/s²): full-throttle ax OCD / TUM [m/s²] 11.8/12.3 (35 m/s), 8.3/10.8 (45), 8.4/7.8 (50), 4.7/4.0 (60), 2.1/1.7 (70), 1.3/0.4 (75). Peak braking OCD / TUM: −33.0/−36.8 (45), −35.7/−39.6 (50), −44.7/−48.0 (60), −56.9/−56.7 (70), −51.7/−60.6 (75): OCD 5–15 % below TUM.

## Tyres (`tyre_check.json`) and cornering (`lateral_check.json`)
- Peak μx matches TUM exactly; peak μy matches on average (±2 % split from the shipped zero-slip offset).
- Combined slip: close to TUM's ellipse (exponent 2) when driving; inside it when braking while cornering (exponent 1.6–1.8, 4–8 % short at 45°).
- **Peak slip positions are the bigger problem:** rear longitudinal peak at 13–33 % slip (AV-21 slip stiffness carrying a ~30 % higher μ), while lateral force already drops steeply at a few percent slip, so the rear cannot use its traction peak in a corner exit.
- Steady cornering ay_max OCD / TUM: 0.83 (15 m/s), 0.91 (25), 0.93 (35), 0.95 (50), 0.95 (70). Hypothesis (not verified): cornering drag of a 19.5° steered front at low speed, cancelled by rear drive force, costs rear side force through combined slip; TUM's point mass has no such term.

## Causes of the gap, ranked
1. **Traction-limited exits: 4.4 s.** TC active on 14 % of the lap; raising its slip target (0.12 / 0.16) spins the car. Stiffer longitudinal slip stiffness (LKX × 1.6) gives **100.99 s (−2.69 s)**, QSS unchanged (95.68 s). Fix: real F1 slip stiffness (PKX/PKY or LKX/LKY scaled with μ), combined-slip shape closer to TUM's ellipse, TC target scheduled on lateral demand.
2. **Braking and hairpin entries: 2.4 s** (mostly T14, T6). OCD decelerates 5–15 % less; lateral limit 9–17 % lower in hairpins, so it runs wide (16 m at T14). Contributors: fixed brake bias 0.49, braking-while-cornering envelope, light inside wheels locking. Fix: per-corner speed target from OCD's own limits, tyre changes from item 1, brake balance closer to load split.
3. **Low-speed lateral grip 5–17 % below TUM: 0.5 s** direct, plus part of item 2. Check LKY, cornering drag, roll-centre PLACEHOLDER (0.2 m) and roll split.
4. **Power: 1.0 s**, mostly speed carried from the exits (4–24 % less acceleration at 35–45 m/s; equal or better ≥ 50 m/s).
5. **Controller: 0.5 s** direct, plus the uniform-f target (f = 0.96 is 0.7 s better than 1.00).
6. **QSS optimism:** OCD's QSS assumes ideal transients and an exponent-2 ellipse on measured pure-slip limits, and omits TUM's e_i, rolling resistance, cornering margin and r(v) (≈ −3.1 s, docs/SF70H_TUM_CAR.md), which hides a ≈ +3.9 s car-limit deficit. The 95.65 / 95.73 s QSS is not reachable for a driven OCD lap with this tyre model.

## Artefacts
- **Locked LSD (p1_TUM `lsd_lock_drive` 1.0):** OCD's LSD torque is tanh(ω_diff)·lock·input torque; locked, the rear wheels fight each other on throttle in corners (median |Fx_RL − Fx_RR| 3.8 kN vs 0.47 kN open). Still better than open (`diag_lsd_open/`: 106.04 s, +2.36 s). TUM's ideal LSD cannot be represented; intermediate locks not tested.
- **Logged `ax_mps2` with the LSD locked** differs from dv/dt by a median −7.2 m/s² on throttle straights (open LSD: 0.08). Checked on the debug car only; existing pure-pursuit and RL CSVs agree with dv/dt within 0.5 m/s². Do not use `ax_mps2` from p1_TUM laps; `analyse()` uses dv/dt.

## Files and how to run
- `cl_f{1.00..0.92}.csv` + `.meta.json`, `runs.json`, `analysis.json`, `tyre_check.json`, `lateral_check.json`, `qss_debug_car.json`; diagnostics at f = 1.00: `diag_lsd_open/`, `diag_LKX1.6/`, `diag_tc0.12/`, `diag_tc0.16/`.
- `.venv/bin/python scripts/comparison/tum_closed_loop.py --all` (about 1 min). Diagnostics: `--f 1.0` with `--lsd-lock 0`, `--tyre-scale LKX 1.6` or `--gain tc_slip 0.12`.
