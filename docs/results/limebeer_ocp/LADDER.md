# OCP ladder: PL14 OCP made closer to OCD's car, one feature per step

Updated 2026-11-01 00:24. Code: rlsetup/ocp/{ocd_car,ladder_model,ladder_solve}.py, scripts/ocp_ladder.py. Every step keeps all earlier ones. Lap times exclude the control-rate regularisation; only Solve_Succeeded laps are listed. Mesh = speed-adapted, nominal 1 m / 0.5 m.

## Shanghai (p1_TUM)

References: OCD RL driver (p1_TUM) 101.27 s; OCD QSS on TUM raceline (debug car) 95.73 s; TUM laptime-sim none 94.886 s

| Step | Feature added | 1 m [s] | 0.5 m [s] | mesh Δ [s] | Δ vs prev [s] | route | v min / max [km/h] | notes |
|---|---|---|---|---|---|---|---|---|
| L0 | PL14 model, SF70H-TUM mapping (beta 0.60, 575 kW, kd diff, PL14 loads, CoG corridor w - 0.9 m) | 94.216 | – | – | – | direct | 60.8 / 306.0 | drive-lim 65%; brake cap 0%; N_min 201 N |
| L1 | OCD MF5.2 tyre (car's coefficients, combined slip, |kappa| <= 1), R = 0.3299 m | 93.986 | – | – | -0.230 | direct | 53.0 / 305.7 | drive-lim 70%; brake cap 0%; N_min 192 N |
| L2 | OCD brakes: 15 kN m total, bias 0.49 (setup), per-wheel torques, wheel-spin equilibrium | 93.498 | – | – | -0.488 | direct | 54.0 / 305.8 | drive-lim 71%; brake cap 0%; N_min 195 N |
| L2b | OCD rolling resistance 0.03 N in the wheel torque balances | 94.884 | – | – | +1.386 | direct | 53.9 / 293.9 | drive-lim 71%; brake cap 0%; N_min 207 N |

Largest per-segment changes vs the previous step (segments split halfway between corner apexes; apexes at 620, 888, 1045, 1609, 2368, 2574, 3148, 3354, 4803, 5180 m):

- **L1** vs L0: 3251–4078 m -0.294 s (v_min 128→126 km/h); 966–1327 m -0.158 s (v_min 110→119 km/h); 177–754 m +0.133 s (v_min 130→125 km/h); 1988–2471 m +0.123 s (v_min 196→187 km/h)
- **L2** vs L1: 4078–4992 m -0.199 s (v_min 53→54 km/h); 2861–3251 m -0.102 s (v_min 79→79 km/h); 1327–1988 m -0.100 s (v_min 84→83 km/h); 177–754 m -0.036 s (v_min 125→125 km/h)
- **L2b** vs L2: 4078–4992 m +0.323 s (v_min 54→54 km/h); 3251–4078 m +0.248 s (v_min 126→125 km/h); 1327–1988 m +0.150 s (v_min 83→83 km/h); 177–754 m +0.139 s (v_min 125→125 km/h)

