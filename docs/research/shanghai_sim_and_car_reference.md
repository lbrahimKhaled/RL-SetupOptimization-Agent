# Shanghai: simulator lap times and physical reference numbers for a 2017 F1 car (SF70H)

web-researcher, 2026-10-27. All pages were opened on this date unless marked NOT VERIFIED. Real-car lap references are in `shanghai_lap_records.md`. This file adds simulator and game times, TUM laptime-sim inputs, and physical numbers.

## 1. Simulator / game lap times at Shanghai

| Sim | Best time found | Details | Source | Status |
|---|---|---|---|---|
| EA F1 24, Time Trial | **1:30.293** | Th3L1ght (Ferrari), 16 Jun 2024, dry. Then 1:30.627 (Paivaa, Mercedes), 1:30.629, 1:30.630, 1:30.782 | https://www.f1laps.com/f1-24/leaderboard/china/ | Verified page. This is the F1Laps **community** board ("ten fastest, valid Time Trial laps that include telemetry data set by the F1Laps community"), not the official EA world record. |
| EA F1 25, Time Trial | **1:32.174** | dezzali (McLaren), 12 Apr 2026, dry, wheel. Then 1:32.408 (Ferrari, 2 Jan 2026), 1:32.419, 1:32.610, 1:32.709 | https://www.f1laps.com/f1-25/leaderboard/china/ | Verified page. Same caveat: community board, not official. |
| EA F1 24, "TT record" | 1:31.847 (claimed) | Claimed by a setup site; no source given | https://simracingsetup.com/setups/f1-24/china-gp-setups/ | Search snippet only. Conflicts with F1Laps, so do not use. |
| Official EA/Codemasters TT world records | NOT FOUND | No public official leaderboard page found for Shanghai | — | NOT VERIFIED |
| iRacing | **Track not available** | Shanghai is not on iRacing's track list. The only Asian tracks listed are Japanese (Suzuka, Fuji, Motegi, Okayama, Tsukuba). | https://www.iracing.com/tracks/ | Verified 2026-10-27 |
| Assetto Corsa (incl. RSS Formula Hybrid 2017 / VRC / SF70H mods) | NOT FOUND | No Shanghai F1-circuit leaderboard or hotlap found. The one Shanghai AC mod found is the 2010 DTM Pudong street circuit, not the F1 track. | https://overtake.gg/news/dtms-short-holiday-in-china-shanghais-street-circuit-in-assetto-corsa.2678 | NOT VERIFIED (search only) |
| rFactor 2 | NOT FOUND | No official Shanghai content or leaderboard found | — | NOT VERIFIED |

Caveat (my reading): the F1 games' car models are year-specific: F1 24 uses the 2024 cars and F1 25 uses the 2025 cars, both with ERS and DRS in Time Trial. Treat them only as "human-in-sim" references for modern hybrid F1 cars.

### Published lap-time simulation: TUMFTM laptime-simulation (F1_Shanghai)
- Repo: https://github.com/TUMFTM/laptime-simulation, branch `master` (7 commits; LGPL-3.0). Paper: Heilmeier, Geisslinger, Betz, "A Quasi-Steady-State Lap Time Simulation for Electrified Race Cars", EVER 2019, DOI 10.1109/EVER.2019.8813646.
- **Published Shanghai reference time: NOT FOUND.** The README gives no Shanghai lap time. I could not open the paper: the TUM mediaTUM copy (https://mediatum.ub.tum.de/1520504) was blocked by a bot wall, and IEEE is paywalled. The abstract says only that F1 and FE results are shown "on different racetracks". The 91.24 s in `shanghai_lap_records.md` was produced by running the repo here; it is not a published number.
- Default run config, from `main_laptimesim.py` on master: `trackname="Shanghai"`, `vehicle="F1_Shanghai.ini"`, `series="F1"`, `mu_weather=1.0`, `use_drs1=True`, `use_drs2=True`, `use_pit=False`, `find_v_start=True`, `em_strategy="FCFB"` ("First Come First Boost"), `initial_energy=4.0e6` J, `use_recuperation=True`. **So the default Shanghai run models ERS deployment (4 MJ initial energy, FCFB) and DRS.**
- `track_pars.ini`, Shanghai entry: `mu_mean=1.0`, DRS zones 3930–4590 m and 5165–450 m (the second zone wraps across the start/finish line).

## 2. Physical reference numbers

### (d) TUMFTM `F1_Shanghai.ini`
Source: https://raw.githubusercontent.com/TUMFTM/laptime-simulation/master/laptimesim/input/vehicles/F1_Shanghai.ini (master)
- Powertrain `type: hybrid`, RWD. `pow_max = 575 kW`, defined in the header as "[W] maximum power". It sits in the engine block alongside the engine rpm range (10,500–12,200 rpm) and fuel flow `be_max = 100 kg/h`. Electric power is a **separate** parameter: `pow_e_motor = 120 kW`, defined as "total electric motor power (after efficiency losses)", with `eta_e_motor = 0.90`, recuperation efficiencies 0.15 (motor) and 0.10 (turbo), minimum e-motor speed 27.777 m/s, max torque 200 Nm.
  - My reading: **575 kW is ICE only. ERS adds 120 kW on top, for about 695 kW peak combined** when the EM strategy deploys it.
- Tyres: the header says "tire data should be normalized to mu = 1.0". `mux`/`muy` are the "coefficient of friction at nominal tire load (fz == fz_0)":
  - Front: `mux = 1.65`, `muy = 1.85`
  - Rear: `mux = 1.95`, `muy = 2.15`
  - `fz_0 = 3000 N`, `dmux_dfz = dmuy_dfz = -5.0e-5` per N ("reduction of force potential with rising tire load"), `tire_model_exp = 2.0` (friction ellipse/circle shape).
  - My arithmetic, assuming a linear form mu = mu0 + dmu_dfz·(Fz − Fz0): at Fz = 6000 N, front muy ≈ 1.70 and rear muy ≈ 2.00. At 8000 N they are ≈ 1.60 and 1.90.
- Other values: `m = 733 kg` (with driver, without fuel), `c_w_a = 1.56 m²`, `c_z_a_f = 2.20 m²`, `c_z_a_r = 2.68 m²` (sum 4.88 m²), rho 1.18 kg/m³, DRS reduction factor 0.17, `f_roll = 0.03`, h_cog 0.335 m, lf/lr 1.968/1.632 m, gearbox efficiency 0.96.

### (a) 2017 F1 power
- MGU-K is capped at **120 kW** (the cap applies to the 2014–2021 hybrid regulations). Source: Honda, https://global.honda/en/tech/motorsports/Formula-1/Powertrain_MGU-H_MGU-K/ (search snippet; page not opened).
- ICE power: **no official 2017 figure exists**, because manufacturers do not publish it. Secondary estimates put the ICE at roughly 600–700+ hp and the whole PU at about 800–1,000 hp (≈600–745 kW): https://www.drivencarguide.co.nz/news/f1-engines-from-ice-to-mgu-k/ and f1technical forum threads. These are low-confidence estimates.
- The TUM file uses 575 kW ICE + 120 kW ERS ≈ 695 kW (≈930 hp), which falls inside that estimate band. (My reading.)

### (b) F1 slick peak friction coefficient
- The only primary source verified here is the TUM file above: mu ≈ 1.65–1.95 longitudinal and 1.85–2.15 lateral at 3 kN, falling by 0.05 per kN. The README and the file do not say whether these are fitted or tuned to lap times; I did not check.
- Load sensitivity concept (mu falls with Fz, peak force still rises): https://en.wikipedia.org/wiki/Tire_load_sensitivity. This is a secondary source.
- Pirelli: the 2017 tyres are 25% wider (front 245→305 mm, rear 325→405 mm, 13-inch rims), to cope with about 20% more downforce: https://www.pirelli.com/global/en-ww/race/racingspot/formula-1/the-same-tyres-for-everyone-at-the-first-five-grands-prix-of-2017-here-s-why--50362/ (search snippet).
- **NOT VERIFIED:** the academic optimal-control F1 tyre parameters (Perantoni & Limebeer, VSD 2014; West & Limebeer 2022). The ORA PDF returned HTTP 500, and the UP repository PDF could not be text-extracted here. I am not quoting values from memory.

### (c) 2017 F1 at Shanghai: speed, lateral g, braking g
- **Speed trap, 2017 Chinese GP qualifying** (dry). Source: FIA PDF "Qualifying Session Speed Trap", https://www.fia.com/file/55177/download. Magnussen top at 329.7 km/h, Bottas 328.8, Hamilton 328.0. **Vettel (SF70H) 325.3, Räikkönen (SF70H) 323.9**, Verstappen slowest at 314.4. The trap location is not stated on the sheet.
- **Race speed trap** (damp start, then drying; slipstream and DRS possible). Source: https://www.fia.com/sites/default/files/speed_trap_45.pdf. Massa 336.4, Bottas 336.0, **Vettel 335.1, Räikkönen 332.7** km/h.
- **Braking at Turn 14** (end of the 1.2 km back straight), from Brembo-derived press. Speed drops from 330 to 66 km/h in about 3 s. Distance is about 129 m and deceleration is **4.7 g** (Paddock Magazine, undated: https://paddockmagazine.com/?p=6496). Motorbox (2017 Brembo data) gives 330→66 km/h and ranks the stop 4th-hardest of 2017: https://www.motorbox.com/auto/sport/f1/news/brembo-tutti-i-dati-delle-frenate-del-mondiale-di-f1-2017. Brembo's 2014 card gives 4.77 g. These are search snippets; the year of the 4.7 g figure is not confirmed.
- **Peak lateral g in 2017**: about 6.5 g for Hamilton at Albert Park T11, with drivers above 6 g in T1 and T11. Source: Formula1.com video analysis, https://www.formula1.com/en/latest/features/2017/3/f1-video-2017-vs-2016-g-force-comparison.html, reported by https://www.thedrive.com/accelerator/8734/2017s-formula-one-cars-are-pulling-1-3gs-more-than-before. Pirelli (Isola) puts loads at up to 35–40% higher than in 2016 in the fastest corners: https://f1technical.net/news/21476. **No Shanghai-specific 2017 lateral-g figure was found.** These are search snippets.
- **2017 qualifying lap times** (Wikipedia, https://en.wikipedia.org/wiki/2017_Chinese_Grand_Prix): Hamilton 1:31.678 (pole and new circuit record), Vettel 1:31.864, Räikkönen 1:32.140. Fastest race lap: Hamilton 1:35.378.

## Implications for us (interpretation, not from the sources)
1. **Top speed.** Our 279 km/h is about 45 km/h below the SF70H qualifying trap (323.9–325.3 km/h). That gap is large, but it is plausible for a car with 575 kW and no ERS. TUM's run has the extra 120 kW and DRS (−17% drag) on the back straight. ERS alone is 17% less power, so the top speed this car can reach is lower.
2. **Grip.** Our MF5.2 PDY1·LMUY ≈ 1.16 and PDX1·LMUX ≈ 1.46 sit well below TUM's muy 1.85–2.15 and mux 1.65–1.95 at 3 kN. Because of load sensitivity, the effective mu at F1 loads is lower than the D-coefficient. Our measured p99 lateral of 2.4 g against about 5–6.5 g real is consistent with a much lower-grip tyre. That probably explains most of the lap-time gap: 122.8 s vs 91.9 s for Vettel, and vs 91.24 s for TUM QSS run locally.
3. **Braking.** Our 3.4 g peak vs about 4.7 g real at T14 is consistent with the lower PDX1 and a slower entry speed.
4. **Reference times.** For the thesis, quote the SF70H 1:31.864 (2017 Q3) as the real reference, and the TUM QSS time (run locally, with ERS and DRS on by default) as the physics upper bound. Game TT times are 2024/2025 cars and are not comparable to a 2017 car with no ERS.
5. **Open choice for Ibrahim.** If he wants the simulated car closer to the SF70H, ERS and the tyre mu scaling are the two biggest levers. Any change of that kind is his decision (PROJECT_CONTEXT §11).
