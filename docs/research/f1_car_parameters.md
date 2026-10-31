# Evidence-based F1 values for the AV-21 leftovers in `sf70h_tum` (web-researcher, 2026-11-01)

Status: complete for what could be accessed. "SAYS" = the source says (quoted/transcribed); "READING" = my interpretation or arithmetic. NOT VERIFIED = I could not open the source; no value is filled from memory. Summary table: §9. Implications: §10.

Access notes for this session: Scholar Gateway connector "not connected". Local PDF rendering unavailable (no poppler), so PDFs were read through the text proxy `https://r.jina.ai/<url>` (worked for the FIA 2017 regulations, ORA ML16 and Veneri & Massaro; later HTTP 429 / Cloudflare) or through DSpace extracted-text bitstreams (West 2020). Blocked (403 / bot walls): Cranfield DSpace (Kelly 2008, Casanova 2000, Tremlett et al. 2015), CORE, Southampton eprints (Dal Bianco/Lot GP2 papers), Padova research.unipd.it (Dal Bianco PhD thesis), academia.edu (Perantoni DPhil).

## 0. Inventory of AV-21 leftovers (read from code, 2026-11-01)

Sources read: `ocd_first_lap/docs/SF70H_TUM_CAR.md`, `ocd_lap/cars.py`, `ocd_lap/translator.py`, `ocd_lap/car_setup.py`, `ocd_lap/backend.py`, shipped JSON `vendor/Open-Car-Dynamics/python3/ocd_vehicle_models_py/config/OCD_Vehicle_RWD_LSD__PT1__DOUBLE_TRACK__MF52__RIDE_HEIGHT.json`, and `vendor/Open-Car-Dynamics/lib/ocd_vehicle_dynamics_double_track_cpp/include/ocd_vehicle_dynamics_double_track_cpp/vehicle_dynamics_equations_impl.hpp`.

| Group | OCD key / place | AV-21 value now in `sf70h_tum` |
|---|---|---|
| Body inertia | `moment_of_inertia_kgpm2.x/y/z` | 97.6 / 579.8 / 1159.6 kg m² (AV-21 100/500/1000 scaled by m, L², t²) |
| Wheel mass | `mass_wheel_kg.front/rear` | 15 / 15 kg |
| Wheel spin inertia | `drivetrain.I_wheel_front_kgm2`, `I_wheel_rear_kgm2` | 1.15 / 1.35 kg m² |
| Tyre vertical stiffness | `tire.spring_stiffness_Npm.front/rear` | 250 000 / 250 000 N/m |
| Roll centre | `suspension.roll_center_height_m` (one value, both axles) | 0.2 m |
| Pitch centre | `suspension.pitch_center_height_m.accel/decel` | 0.2 / 0.2 m |
| Springs / dampers / bars | setup knobs (`car_setup.py`): ride freq 4.23/3.37 Hz (range 2.5–8), ζ 0.144/0.106 (0.1–1), roll stiffness 275 835 N m/rad (1.5e5–1.5e6), front share 0.613 (0.35–0.8) | AV-21-derived defaults |
| Heave / packers | knobs heave freq 0 (0–8 Hz), packer gap 0.10 m (0.005–0.10); `PACKER_STIFFNESS_NPM` 2e6 N/m, `DAMPER_KNEE_MPS` 0.05 m/s (translator.py) | PLACEHOLDER constants |
| Ride height / rake | knobs ride_height_front 0.035 m (0.02–0.08), rake 0.05 m (0–0.10) | aero-map reference |
| Motion ratio | `translator.CarGeometry.motion_ratio_*` = 1 (forces act at wheel) | n/a in OCD |
| Steering | `steering_actuator.angle_max_rad` 0.6, `angle_rate_max_radps` 1.5 (backend override; JSON 0.3 / 0.5), `T_PT1` 0.06 s | AV-21 / backend |
| Brakes | `max_brake_torque_Nm` (backend default 360; RL repo uses 12 000 N m total), knob `brake_bias_front` 0.45–0.70 (default 0.5) | |
| Differential | `drivetrain.LSD_preload` (0–300 N m), `LSD_lock_drive`, `LSD_lock_coast` (0–1) | knobs, default 0 (open) |
| Footprint | `CarSpec.footprint` lf+0.5 / −lr−0.5, ±1.0 m | 4.6 × 2.0 m |
| Tyre MF5.2 (beyond peak μ) | PKX1 30.7, PKX2 0.27, PKX3 0.13; PKY1 −75.5, PKY2 4.65, PKY3 −0.6, PKY4 2.0; PCX1 1.6, PCY1 1.5; PEX1 0.7, PEX2 −0.17, PEX3 0.023, PEX4 −0.14; PEY1 0.1, PEY2 −1.2, PEY3 0.05, PEY4 −8.1; RBX1 17.4, RBX2 12.9, RCX1 1.1; RBY1 20.6, RBY2 −23.5, RBY3 0.001, RCY1 1.0; RVY1 0.16, RVY2 0.03, RVY3 27.5, RVY4 −29.7, RVY5 0.03; PVY1 0.04, PVY2 −0.02, PVY3 −0.97, PVY4 −0.41; PHY1 0.003, PHY2 −0.0006, PHY3 0.004; PDX3/PDY3 0.1; LMUX/LMUY 0.97; FNOMIN 3000 N | AV-21 |
| Relaxation | **hard-coded**, not a parameter: `pt1_T_relax_long = 0.3 / clamp(|vx|,10,100)`, `pt1_T_relax_lat = 0.2 / clamp(|vx|,10,100)` (double-track `vehicle_dynamics_equations_impl.hpp` l.728–729), i.e. relaxation lengths 0.3 m (long.) and 0.2 m (lat.) | changing needs a vendor patch |
| Rolling radius | already TUM (0.3299 m) | speed growth not modelled |
| Engine inertia, gear ratios | already TUM (`powertrain.py`, `e_i`, `i_trans`) | not AV-21 |

## 1. Regulations, dimensions, footprint, wheels and tyres (2017)

### Sources
- **[FIA17]** FIA, *2017 Formula One Technical Regulations*, version dated 30 April 2017 (footer "© 2017 FIA"), https://www.fia.com/file/56348/download/18819 (read via r.jina.ai text proxy, 2026-11-01).
- **[F1-PZ]** Formula1.com, "Pirelli reveal wider, faster 2017 tyres", 28 May 2016, https://www.formula1.com/en/latest/article/pirelli-reveal-wider-faster-2017-tyres.5eQCKChfRIx709KvI2LO3S
- **[ESPN-PZ]** ESPN, "Pirelli unveils 2017 tyre", 28 May 2016 (quotes Mario Isola, Pirelli), https://africa.espn.com/f1/story/_/id/15782991/pirelli-unveils-2017-tyre
- **[F1-WB]** Formula1.com, "Wheelbase: the design characteristic that might hurt Mercedes in Monaco", 22 May 2017, https://www.formula1.com/en/latest/article/wheelbase-the-design-characteristic-that-might-hurt-mercedes-in-monaco.DrrNIPZ1ksEC0I20OKISe
- **[MSM17]** M. Hughes, "Mercedes vs Ferrari", *Motor Sport* December 2017 issue, https://www.motorsportmagazine.com/archive/article/december-2017/72/mercedes-vs-ferrari/
- **[WIKI-SF70H]** Wikipedia, "Ferrari SF70H" infobox, https://en.wikipedia.org/wiki/Ferrari_SF70H (secondary; no citation on length/wheelbase rows).

### SAYS
- [FIA17] Art. 3.2.2: "The overall width of the car, excluding tyres, must not exceed 2000mm". Art. 3.2.3 (overhangs): "No part of the car may be more than 710mm behind the rear wheel centre line"; no part more than 1200 mm ahead of the front wheel centre line.
- [FIA17] Art. 4.1: "The weight of the car, without fuel, must not be less than 728kg at all times during the Event."
- [FIA17] Art. 12.4.1: complete wheel width 370–385 mm front, 455–470 mm rear. Art. 12.4.2: complete wheel diameter ≤ 670 mm with dry tyres (680 mm wet). Art. 12.4.4: rim outer lip diameter 358 mm (±1).
- [F1-PZ]: "the new sizes for slick tyres will be 305/670-13 at the front and 405/670-13 at the rear"; "the total diameter will go from the current 660mm to 670mm"; "Wheel rim diameter remains unchanged at 13 inches."
- [ESPN-PZ] (Isola, May 2016, prototype estimates): "the front is currently 8.5kg and will increase by 1kg"; "The rear tyre is now 10kg and it will be around 11.5kg, depending on the final version of the prototype." (ESPN also prints a 325 mm front width, which conflicts with [F1-PZ] 305 mm; 305 is the size in Pirelli's 305/670-13 designation.)
- [F1-WB]: Ferrari 2017 wheelbase "approximately 3,594mm"; Mercedes W08 "about 3,760mm" (other teams' analysis). [MSM17]: W08 "3738mm", "143mm longer than the Ferrari".
- [WIKI-SF70H]: length 5000 mm, width 2000 mm, height 950 mm, wheelbase 3,594 mm (no row-level citation).

### Per-parameter findings

| Parameter | F1 value | Source | Car/year | Conf. | Mapping to OCD |
|---|---|---|---|---|---|
| Wheelbase (check of TUM 3.6 m) | ≈ 3.594–3.595 m | [F1-WB], [MSM17] | SF70H 2017 | medium (rival-team / journalist estimates, two agree) | TUM's 3.600 m is within 6 mm; keep. |
| Overall width (bodywork) | ≤ 2.000 m excluding tyres | [FIA17] 3.2.2 | 2017 regs | high | Footprint half-width. READING: with TUM's 1.6 m track and 0.455–0.470 m rear complete-wheel width, the outer rear tyre edge is at 0.8 + 0.235 ≈ **1.035 m** from the centre line (front 0.8 + 0.1925 ≈ 0.99 m). Use ±1.035 m if the footprint must contain the tyres, ±1.0 m for bodywork. |
| Front overhang | ≤ 1.200 m ahead of front axle | [FIA17] 3.2.3 | 2017 regs | high (limit, not SF70H actual) | footprint x_front = lf + overhang. |
| Rear overhang | ≤ 0.710 m behind rear axle | [FIA17] 3.2.3 | 2017 regs | high (limit) | footprint x_rear = −(lr + overhang). |
| Overall length | 5.000 m | [WIKI-SF70H] | SF70H | low (uncited) | READING: 5.000 − 3.594 = 1.406 m total overhang, inside the regulatory 1.91 m. Any front/rear split of the 1.406 m is unsourced. Conservative: regulatory maxima (front 1.2, rear 0.71 → 5.50 m box: x ∈ [−2.342, +3.168] m about the CoG). Wikipedia-length option with the full 0.71 m rear: front 0.696 m → x ∈ [−2.342, +2.664]. |
| Minimum mass | 728 kg without fuel, with driver | [FIA17] 4.1 | 2017 | high | TUM's 733 kg is consistent (+5 kg). Keep TUM. |
| Tyre size | 305/670-13 front, 405/670-13 rear (dry) | [F1-PZ], [FIA17] 12.4 | 2017 Pirelli P Zero | high | Unloaded radius 0.335 m. TUM circ_ref 2.073 m → 0.3299 m, ~5 mm under the free radius: consistent with a loaded radius. |
| Tyre mass (tyre only) | ≈ 9.5 kg front, ≈ 11.5 kg rear (prototype estimates) | [ESPN-PZ] (Isola) | 2017 prototypes | medium-low | Part of OCD `mass_wheel_kg` (§4). |

## 2. Tyre model beyond peak μ (MF5.2 slip stiffness, shape, combined slip, relaxation, radius, vertical stiffness)

### Sources
- **[PL14]** Perantoni & Limebeer, "Optimal control for a Formula One car with variable parameters", *Vehicle System Dynamics* 52(5):653–678, 2014, DOI 10.1080/00423114.2014.889315; ORA accepted manuscript, Appendix A eqs A.1–A.16 and Appendix B Table 3. Transcribed in `docs/research/limebeer_ocp_model.md` §1.8–1.9. PL14 takes its tyre data from Kelly 2008 [their ref 11] and says: "Due to a paucity of publicly available Formula One tyre data, we have used the same tyre model on the front and rear wheels."
- **[West20]** W.J. West, *Optimal Tyre Management for a Formula One Car*, MEng dissertation, University of Pretoria, 2020 (supervisors D.J.N. Limebeer, I. Craig). Read via the repository's extracted-text bitstream https://repository.up.ac.za/server/api/core/bitstreams/5e267d09-ab64-4041-b1c5-0bed1f86c63e/content (item https://repository.up.ac.za/items/e6e24303-5b62-43c0-a749-9f96d2e43bed). Addendum B, Tables B.1–B.3; every table cites ref [14] = Tremlett & Limebeer, "Optimal tyre usage for a Formula One car", VSD 54(10):1448–1473, 2016 (TL16 itself NOT read).
- **[ML16]** Imani Masouleh & Limebeer, "Optimizing the aero-suspension interactions in a Formula One car", *IEEE Trans. Control Syst. Technol.* 24(3), 2016; ORA manuscript https://ora.ox.ac.uk/objects/uuid:856dd2b0-3df9-4daa-a4f7-9462dc7cd2f8/files/m3fb7226495d64be875ddfdc868095771 (read via r.jina.ai), eq. (17), §II-A, Table I. ML16: "Nominal values for the car and tire parameters can be found in [12]" (= PL14).
- **[VM19]** Veneri & Massaro, "A free-trajectory quasi-steady-state optimal-control method for minimum lap-time of race vehicles", VSD (accepted manuscript, May 2019), Table 1, https://research.unipd.it/bitstream/11577/3297111/2/2019%20-%20A%20free-trajectory%20quasi-steady-state%20optimal-control%20method%20for%20minimum%20lap-time%20of%20race%20vehicles%20-%20MM.pdf (read via r.jina.ai). **Not an F1 car** (1300 kg, 415 kW, GT-like): racing-slick MF cross-check only.
- NOT VERIFIED (blocked): Kelly, *Lap time simulation with transient vehicle and tyre dynamics*, PhD, Cranfield 2008, https://dspace.lib.cranfield.ac.uk/handle/1826/4791; TL16 full text; Casanova 2000 PhD (Cranfield, handle 1826/1091).

### SAYS
- [PL14] Table 3 (= [West20] Table B.3): Fz1 2000 N, Fz2 6000 N; μx 1.75 / 1.40; μy 1.80 / 1.45; κ "slip coefficient for the friction peak" 0.11 / 0.10; α "slip angle for the friction peak" 9° / 8°; Qx = Qy = 1.9. Model: μ = μmax sin(Q arctan(S ρ)), S = π/(2 arctan Q), ρ = sqrt(αn² + κn²), κn = κ/κmax, αn = α/αmax, Fx = μx Fz κn/ρ, Fy = μy Fz αn/ρ.
- [West20] Table B.1 (from TL16): tyre carcass mass 9.5 kg front / 11.5 kg rear, tread mass 0.5 kg; tyre diameter 0.66 m; tyre width 0.355 / 0.380 m; contact patch width 0.20 / 0.22 m. Table B.2: R 0.33 m, M 660 kg, Iz 450 kg m², kd "100 N/rpm".
- [ML16] eq. (17): "The loaded tire radius characteristic used here is given by" R_fw = 326.0 + 4.00×10⁻³ F_TL + 4.075×10⁻⁸ F_TL², "in which F_TL is the normal tire load expressed in Newtons; this impirical formula is based on tire-squash measurements." Units of R and the sign of F are not stated; ML16 §III treats tyre normal forces as "non-positive".
- [VM19] Table 1 (GT-class): pCx1 1.6935, pDx1 1.8757, pDx2 −0.127, pEx1 0.07708, pKx1 30.5, pKx3 0.2766, λμx 0.93; pCy1 1.733, pDy1 1.8217, pDy2 −0.4388, pEy1 0.29446, pKy1 44.2, pKy2 2.5977, λμy 0.84; N0 3500 N; δmax 20°.

### READING: what PL14 implies for MF5.2 (my arithmetic, done by hand)
1. PL14's curve is an MF sine curve in normalised slip with **C = Q = 1.9, E = 0**, B = S/κmax (S = π/(2·arctan 1.9) = 1.4459). In MF terms B = K/(C·D), so PL14 ⇔ **K/Fz = Q·S·μ/κmax = 2.747·μ/κmax** (same for α with αmax in rad).
2. PL14 small-slip stiffnesses: Kx/Fz = 43.7 (2 kN), 42.5 (3 kN, interpolated), 38.5 (6 kN). Ky = 63.0 kN/rad (2 kN), 92.4 kN/rad (3 kN), 171 kN/rad (6 kN).
3. AV-21 MF5.2 at γ = 0 (Ky = PKY1·FNOMIN·sin(2·atan(Fz/(PKY2·FNOMIN))); Kx = Fz(PKX1 + PKX2·dfz)·exp(PKX3·dfz)): **Ky = 63.6 / 93.1 / 164 kN/rad at 2 / 3 / 6 kN, within 1–4 % of PL14.** Kx/Fz = 29.3 / 30.7 / 35.3: **25–33 % softer than PL14 at 2–3 kN**, 8 % softer at 6 kN. That matches SF70H_TUM_CAR.md, where the Fx peak sits at 14–27 % slip (PL14: ~10 %).
4. Caveat on PL14's printed formula: with S = π/(2 arctan Q), the maximum of sin(Q·arctan(Sρ)) is at ρ = tan(π/(2Q))/S = 0.751. So with the formulas exactly as printed the force peak is at 0.75·κmax (κ ≈ 0.08, α ≈ 6–7°); taking "slip at the friction peak" literally it is 0.10–0.11 / 8–9°. Either way the F1 band is **κ_peak ≈ 0.08–0.11, α_peak ≈ 6–9°**, falling slightly with load.

### Per-parameter findings (tyre)

| OCD parameter | AV-21 now | F1 value / method | Source | Year/car | Conf. | Mapping |
|---|---|---|---|---|---|---|
| PKX1, PKX2, PKX3 (Kx/Fz) | 30.7, 0.27, 0.13 | **Option A (PL14 stiffness): PKX1 ≈ 42.5, PKX2 ≈ −4.0, PKX3 = 0** (fits 43.7/42.5/38.5 at 2/3/6 kN within 0.1). **Option B (PL14 peak slip kept with TUM μ):** Kx/Fz = 2.747·μ_TUM(Fz)/κmax(Fz) → front PKX1 ≈ 42.2, PKX2 ≈ −1.0; rear PKX1 ≈ 49.8, PKX2 ≈ −0.35; PKX3 = 0 | [PL14] Table 3 + READING 1–2 | 2013-era F1 (PL14 car); data via Kelly 2008 | medium-low (one academic source; not 2017; front = rear) | `z.tire_model.<wheel>.PKX*`, FNOMIN 3000 N (dfz = (Fz−3000)/3000). Verify with the MF sweep in `sf70h_checks.py`: the Fx peak should land at κ ≈ 0.08–0.11. |
| PKY1, PKY2 (Ky) | −75.5, 4.65 | **Keep** (matches PL14 within 4 % at 2–6 kN). Option B (keep PL14 α_peak with TUM's higher μy): LKY ≈ μy_TUM/μy_PL14 at 3 kN → **front ≈ 1.08, rear ≈ 1.26** | [PL14] + READING 3 | as above | medium-low | `PKY*` unchanged, or `LKY` per axle. |
| PCX1, PCY1 (shape C) | 1.6, 1.5 | **1.9** (= PL14 Qx = Qy) | [PL14] Table 3 | as above | medium-low | MF C ≡ PL14 Q. Consistent only together with E = 0. |
| PEX1–4, PEY1–5 (curvature E) | 0.7, −0.17, 0.023, −0.14 / 0.1, −1.2, 0.05, −8.1, 0 | **0** (PL14 has no E term) | [PL14] A.11–A.14 | as above | medium-low | C = 1.9 with E = 0 reproduces PL14's pure-slip shape (READING 1). Cross-check [VM19] (GT, not F1): Cx 1.69, Cy 1.73, Ex 0.077, Ey 0.29 — also small E. |
| RBX*, RCX1, RBY*, RCY1, RVY* (combined slip) | AV-21 | **No published F1 MF combined-slip coefficients found.** PL14 combines in normalised slip, ρ = sqrt(κn² + αn²); TUM uses a friction ellipse (exponent 2) | [PL14] A.10, A.15–16; TUM ini `tire_model_exp` | – | none (for MF values) | Keep AV-21 R*, or fit R* so the MF combined-slip curves approximate PL14's normalised-slip ellipse (estimate method; a sim-runner job). |
| PVY*, PHY*, RVY* (offsets / slip-induced side force) | 0.04, … | **No F1 source.** PL14's tyre is symmetric (no offsets) | [PL14] | – | – | Setting them to 0 removes the ±2 % μy split noted in SF70H_TUM_CAR.md; decision for Ibrahim. |
| Relaxation length | 0.3 m long. / 0.2 m lat. (hard-coded in OCD C++) | **No F1 source found.** PL14/ML16 are quasi-static (no relaxation). Generic tyres 0.12–0.45 m (Wikipedia "Relaxation length", secondary). Kelly 2008 has transient tyre dynamics but is NOT VERIFIED | – | – | none | Keep OCD's hard-coded values (changing them needs a vendor patch). |
| Loaded radius / squash | 0.3299 m fixed (TUM circ) | R = 326.0 + 4.00e-3·F + 4.075e-8·F² (mm, F in N, F ≤ 0 compressive): 326 mm at 0, ≈ 314 mm at −3 kN, ≈ 304 mm at −6 kN | [ML16] eq. 17 ("based on tire-squash measurements") | 2013-era F1, 660 mm tyre | medium (units/sign inferred) | OCD rolling radius is constant (+ speed scaling). For 2017 (670 mm) TUM's 0.330 m is consistent. |
| Tyre vertical stiffness | 250 kN/m | **≈ 250 kN/m at zero load, ≈ 266 kN/m at 3 kN, ≈ 285 kN/m at 6 kN** (= 1/(dR/dF), my derivative of [ML16] eq. 17, assuming R in mm, F in N, F ≤ 0) | [ML16] eq. 17 + READING | 2013-era F1 | medium-low (derived; pre-2017 tyre; no 2017 Pirelli figure found) | `tire.spring_stiffness_Npm.front/rear` ≈ **2.6–2.7e5 N/m** at typical loads. The AV-21's 2.5e5 is at the low end of the band. |
| Rolling-radius speed growth | not modelled | TUM: +4.5 % per 200 km/h (already documented) | TUM `car.py` | – | – | unchanged |

## 3. Steering

### Sources
- **[F1-INBOX19]** Formula1.com, "F1 INBOX after the Monaco Grand Prix…", 1 Jun 2019, answered by Rob Smedley (former Ferrari/Williams head of vehicle performance), https://www.formula1.com/en/latest/article/f1-inbox-your-questions-on-hamilton-tyresand-leclercs-home-race-monaco-grand-2019.4zHk6V0eVvQ5g4yIl3KHk
- **[MS10]** Motorsport.com blog, "Monaco Grand Prix – The latest Tech stories", 17 May 2010 (no named author), https://www.motorsport.com/f1/news/monaco-grand-prix-the-latest-tech-stories/3217923/ (read via r.jina.ai).
- **[WA21]** Wheatley & Ali, "Designing an upper stage steering system for a Formula FSAE car", *Sci. J. Silesian Univ. Technol. Ser. Transport* 113:205–218, 2021, https://researchonline.jcu.edu.au/72597/ — secondary; cites Milliken & Milliken 1995 [their ref 9], which I did NOT open.
- [FIA17] Art. 10.4.

### SAYS
- [F1-INBOX19]: "you need around 21-22 degrees of steer to get around Loews Hairpin"; "In Monaco, with the wheelbase on F1 cars around 3.1-3.3 metres"; teams fit a Monaco-specific configuration with more lock.
- [MS10]: "The main one is the steering lock, which needs to be 21 degrees in order to get round the Loews hairpin." "Normally an F1 car has a steering lock of 17 degrees."
- [WA21]: "In contrast, Formula 1 cars (10:1) [9]" (ref 9 = Milliken & Milliken, *Race Car Vehicle Dynamics*, SAE 1995).
- [FIA17] 10.4.1: no steering system may realign more than two wheels (no rear steer); 10.4.2: power steering may not be electronically controlled or electrically powered.
- NOT VERIFIED (search snippet only): NHTSA DOT HS 809 875 (Forkenbrock & Elsasser, June 2005), "An Assessment of Human Driver Steering Capability", https://www.nhtsa.gov/sites/nhtsa.gov/files/forkenbrock_0.pdf — peak hand-wheel rates of order 1000–1800 °/s in emergency double lane changes (road-car drivers).

| OCD parameter | AV-21 now | F1 value | Source | Year | Conf. | Mapping |
|---|---|---|---|---|---|---|
| `steering_actuator.angle_max_rad` (road-wheel angle) | 0.6 rad (34.4°) | **≈ 17° = 0.297 rad normal; 21–22° = 0.37–0.38 rad Monaco spec** | [MS10], [F1-INBOX19] | 2010 / 2019 (journalistic; no 2017-specific value) | medium | Road-wheel angle limit; for Shanghai 0.30 rad. READING: OCD's own JSON value (0.3 rad) is already the F1 value; the backend's 0.6 rad override is about twice F1 lock. Check that the Shanghai hairpins (T6, T14) are drivable with 0.30 rad on the chosen line. |
| Steering ratio | n/a (OCD steers the road wheel directly) | ≈ 10:1 | [WA21] citing Milliken 1995 | 1990s F1 | low (secondary) | Only needed to convert hand-wheel rates: 17° road wheel × 10 ≈ ±170° at the hand wheel. |
| `steering_actuator.angle_rate_max_radps` | 1.5 rad/s | **No F1 source found.** Estimate: human hand-wheel rates of several hundred to ~1000 °/s (NHTSA, NOT VERIFIED) ÷ 10:1 → ~0.9–1.7 rad/s at the road wheel | estimate | – | low | Keep 1.5 rad/s (inside the estimate band) or treat it as an RL-agent design choice. |
| `steering_actuator.T_PT1` | 0.06 s | **No F1 source.** | – | – | none | Keep. |

## 4. Masses, wheel masses, wheel spin inertia, body inertias

### Sources
- [PL14] Table 4 / [West20] Table B.2: M 660 kg, Iz 450 kg m², wheelbase 3.4 m, a 1.8 m, h 0.3 m, wf = wr 0.73 m, R 0.33 m (2013-era F1).
- [West20] Table B.1 (from TL16): tyre carcass 9.5 / 11.5 kg + tread 0.5 kg (front / rear).
- [ESPN-PZ]: 2017 tyre ≈ 9.5 / 11.5 kg (estimate).
- **[MITO]** "Mitostile Prototipo – Calculating vehicle inertia", https://sites.google.com/site/mitostile/vehicle-dynamics-powertrain-basics/technical-articles/chassis-suspension-brakes/calculating-vehicle-inertia — uncited blog, unlabeled columns: race-car(?) yaw inertia "900-1150" kg m² for 450–750 kg dry mass. Low quality; listed only because it is the only other number found.
- NOT VERIFIED: Casanova 2000 thesis (F1 yaw-inertia sensitivity study); Casanova, Sharp & Symonds, "Sensitivity to mass variations of the fastest possible lap of a Formula One car", VSD 2001 (ResearchGate snippet: yaw inertia has "a surprisingly small influence"); Dal Bianco/Lot/Gadola GP2 multibody data (Proc IMechE D 2018; Padova PhD thesis) — all blocked.

### Findings

| OCD parameter | AV-21 / now | F1 value | Source | Year/car | Conf. | Mapping |
|---|---|---|---|---|---|---|
| `moment_of_inertia_kgpm2.z` (yaw) | 1159.6 | **PL14: 450 kg m² at 660 kg, L 3.4 m → scaled to TUM car (I ∝ m L²): 450 × (733/660) × (3.6/3.4)² ≈ 560 kg m²** | [PL14] Table 4 + READING | 2013-era F1 | medium-low (single source; does not say whether wheels are included) | Current value is ~2× the PL14-scaled value. READING (plausibility only): four wheel assemblies of ~25–30 kg at (±1.8, ±0.8) m alone contribute ≈ 400–470 kg m², so 450 kg m² looks low for a whole car; [MITO]'s 900–1150 (uncited) is at the other end. No 2017 or SF70H value found. Recommend a lap-time sensitivity run at 560 / 860 / 1160 before choosing (decision for Ibrahim). |
| `moment_of_inertia_kgpm2.x` (roll), `.y` (pitch) | 97.6, 579.8 | **No F1 source found.** | – | – | none | Keep the scaled AV-21 values, or, if Iz changes, keep Iy/Iz and Ix/Iz ratios (estimate method). |
| `mass_wheel_kg.front/rear` (unsprung per corner) | 15 / 15 | **No F1 unsprung-mass source found.** Known components: tyre ≈ 10 kg front / 12 kg rear (carcass + tread, [West20] B.1; [ESPN-PZ]); rim, upright, brake disc/caliper, hub, half the links: no source | [West20], [ESPN-PZ] | 2014–2017 | tyre: medium; total: none | READING: 15 kg is below the tyre-plus-rim mass alone; a whole corner is likely 20–30 kg (estimate, unsourced). Changing it also changes the sprung corner masses in `translator.py`, hence every knob→rate mapping. |
| `drivetrain.I_wheel_front_kgm2`, `I_wheel_rear_kgm2` | 1.15 / 1.35 | **No F1 source found.** Estimate (READING): tyre as an annulus between the bead (r ≈ 0.179 m, from the 358 mm rim lip, [FIA17] 12.4.4) and the tread (0.335 m): I = m(r_o² + r_i²)/2 ≈ 0.72 kg m² front (10 kg), 0.87 rear (12 kg); all mass at the tread gives 1.12 / 1.35. Rim and disc add ~0.05–0.15 | estimate | 2017 sizes | low | AV-21 values sit at the top of the estimate band; keep, or use ~0.9 / 1.1 kg m². The rear also carries driveline inertia, which `powertrain.py` already handles via TUM's `e_i`. |

## 5. Suspension: ride/roll rates, heave, damping, roll/pitch centres, ride heights, rake

### Sources
- [ML16] §II-A, Fig. 5, Table I. SAYS: the suspension model "and the tire squash, are replaced by a metamodel"; "Since the suspension deflection is dominated by the normal tire load"; "As the tire loads increase, the car moves towards the road (in the positive nz-axis direction)"; "In its nominal configuration the front ride height is hf0 = −14 mm, while the rear ride height is hr0 = −74.1 mm" (footnote: car stationary on the garage floor); rear ride height "varies between -88 mm and -27 mm", roll "± 5°"; front "varies between -24 mm and 7 mm"; loads swept "from zero to -8 kN at the rear axle" and "from zero to -6 kN" at the front. Table I caption: "The left- and right-wheel normal loads Fl and Fr, respectively, are given in (kN)"; ride heights in mm, roll in rad. Coefficients (columns 1, Fl, Fr, Fl², FlFr, Fr²):
  - hf: −21.52, −2.513, −2.513, −0.08324, 0.1126, −0.08324
  - hr: −89.04, −4.1305, −4.1305, −0.09450, 0.1119, −0.09450
  - φf: 0, 0.00851, −0.00851, 0.000256, 0, −0.000256
  - φr: 0, 0.0112, −0.0112, 6.1e-5, 0, −6.1e-5
  - ML16 describes the car: "A typical Formula One car deploys a double wishbone suspension system with a push rod"; "The suspension rockers are also connected to a parallel heave spring and damper combination". No spring, damper, ARB or motion-ratio values are given.
- [MSM17]: SF70H "typically running at about 1.55 degrees (Red Bull 1.58, Mercedes 0.9)" of rake.
- Motorsport.com (Spanish edition) rake table, https://es.motorsport.com/f1/news/rake-angulo-inclinacion-entre-ejes/3174315/ — search snippet only (Ferrari 2017 1.28°, Red Bull 1.9°): NOT VERIFIED, conflicts with [MSM17].

### READING: rates implied by ML16 Table I (my arithmetic; ML16 car = PL14 car, track 2 × 0.73 m)
- Check: at static wheel loads ≈ −1.6 kN front, hf = −13.6 mm, matching hf0 = −14 mm, so the table reads as stated.
- Heave (both wheels of an axle loaded equally), derivative at typical loads: front 4.85–4.70 mm per kN per wheel → **ride rate incl. tyre ≈ 206–213 kN/m per front wheel**; rear 7.95–7.64 mm/kN → **≈ 126–131 kN/m per rear wheel**.
- Removing the tyre in series (k_t ≈ 258–266 kN/m from §2): **suspension-only heave rate ≈ 0.9–1.15 MN/m per front wheel (very stiff; mostly the heave element), ≈ 245 kN/m per rear wheel.**
- Roll: φ = 2ΔF (a + 2bF0) for a load transfer ±ΔF → roll stiffness incl. tyres ≈ **95 kN m/rad front, 67 kN m/rad rear** (front share 0.59). Removing tyre roll stiffness (k_t·t²/2 ≈ 275 kN m/rad per axle): **suspension-only roll stiffness ≈ 145 kN m/rad front, 88 kN m/rad rear, total ≈ 232 kN m/rad, front share ≈ 0.62.**
- Nominal static ride heights (magnitudes): ≈ 14 mm front, ≈ 74 mm rear → ≈ 60 mm difference over 3.4 m ≈ 1.0° (2013-era car, static). [MSM17] SF70H rake 1.55° → over 3.594 m ≈ 97 mm rear-minus-front at the axles (READING, assumes the angle is measured between axle-line ride heights).

### Findings

| OCD parameter / knob | AV-21 default | F1 value | Source | Year | Conf. | Mapping |
|---|---|---|---|---|---|---|
| `roll_stiffness_Nm_per_rad` (springs + bars, at the wheel) | 275 835 (range 1.5e5–1.5e6) | **≈ 2.3e5 N m/rad** (suspension only) | [ML16] Table I + READING | 2013-era F1 (660 kg) | medium-low (derived) | Default is within 20 %; keep, or set ≈ 2.3e5 and scale with mass if wanted. Range covers it. |
| `roll_stiffness_front_share` | 0.613 (0.35–0.8) | **≈ 0.62** (suspension only), 0.59 incl. tyres | [ML16] + READING | 2013-era F1 | medium-low | Default is already consistent; keep. (PL14's QSS uses D_roll 0.5; its optimum was 0.45–0.63.) |
| `ride_freq_front_hz` / `ride_freq_rear_hz` (corner springs) and `heave_freq_*_hz` (heave element) | 4.23 / 3.37 Hz; heave 0 (range 0–8) | **Heave rate per wheel ≈ 1.0 MN/m front, ≈ 245 kN/m rear (suspension only)**. With the TUM car's corner sprung masses (151 / 185 kg with 15 kg wheels): if corner springs carry only the roll stiffness above (no bars), front corner ≈ 2·145e3/t², rear ≈ 2·88e3/t²; the heave element supplies the rest. Total heave alone ⇒ ≈ 13 Hz front, ≈ 5.8 Hz rear on the suspension; incl. tyres ≈ 5.9 Hz front, 4.2 Hz rear | [ML16] + READING | 2013-era F1 | low-medium (derived via two fits) | READING: an F1-like setup needs a **non-zero heave element**, very stiff at the front; the front value is above the 8 Hz knob limit. Extending the heave-knob range is a decision for Ibrahim. |
| `damping_ratio_*`, `rebound_bump_ratio`, `fast_slow_ratio`, `heave_damping_ratio`, `DAMPER_KNEE_MPS` | 0.144 / 0.106, 1, 1, 0, 0.05 m/s | **No F1 source found.** | – | – | none | Keep (setup knobs). |
| `roll_center_height_m`, `pitch_center_height_m.accel/decel` | 0.2 / 0.2 / 0.2 m | **No F1 source found** (no roll-centre heights or anti-dive/anti-squat percentages published; forum opinion only) | – | – | none | Keep; flag. |
| `ride_height_front_m` | 0.035 (0.02–0.08) | ≈ 0.014 m static (2013-era), range ≈ −0.007 to 0.024 m with load | [ML16] §II-A | 2013-era F1 | medium (sign convention inferred) | Ride-height reference plane differs from OCD's aero-map reference; READING: F1 static front ride height is below the knob's 0.02 m minimum. Only matters with a ride-height aero map (Stage 2 PLACEHOLDER). |
| `rake_m` (rear − front) | 0.05 (0–0.10) | ≈ 0.06 m static (2013-era, ML16); SF70H ≈ 1.55° ≈ 0.097 m over the wheelbase | [ML16]; [MSM17] + READING | 2013 / 2017 | medium-low | SF70H rake is at the top of the knob range; keep range. |
| `PACKER_STIFFNESS_NPM`, packer gaps | 2e6 N/m; 0.005–0.10 m | **No F1 source found.** | – | – | none | Keep. |
| Motion ratios | 1 (OCD acts at the wheel) | not needed: ML16 rates above are already at the wheel | [ML16] | – | – | – |

## 6. Brakes

### Sources
- [FIA17] Art. 11: one system, two circuits (front/rear), one pedal (11.1.1); calipers aluminium (E ≤ 80 GPa), one per wheel, ≤ 6 pistons (11.2); discs ≤ 32 mm thick, ≤ 278 mm outside diameter, ≤ 2 pads per wheel (11.3); powered rear brake control allowed under conditions (11.7).
- **[MS-BR17]** Motorsport.com, "Analysis: Teams split as F1 2017 braking forces go up by 25 percent", 30 Jan 2017, quoting Mauro Piccoli (Brembo), https://www.motorsport.com/f1/news/analysis-teams-split-as-f1-2017-braking-forces-go-up-by-25-percent-869021/ (read via r.jina.ai). SAYS: "torque forces predicted to be around 25 percent higher this year than they were in 2016"; disc "maximum thickness going up from 28mm to 32mm"; "the current split is 60/40" (28 vs 32 mm); "Piccoli expects braking G-forces to exceed 6G this year." No absolute torque.
- **[MB-BR]** Mercedes-AMG F1, "Formula One brake systems explained", https://mercedesamgf1.com/news/formula-one-brake-systems-explained (undated). SAYS: "The cars decelerate at around 5G"; pedal "well over 100kg of force"; discs "can reach 1,000°C or more"; "Inside each calliper, six pistons clamp pads against the disc"; front/rear master cylinders; rear via brake-by-wire with the MGU-K.
- Brake bias: no primary source with a number found (fan glossaries quote 55–60 % front; formula1-dictionary.net checked and does not give a range). NOT VERIFIED.

| OCD parameter | Now | F1 value | Source | Year | Conf. | Mapping |
|---|---|---|---|---|---|---|
| `max_brake_torque_Nm` (total, 4 wheels) | 360 backend default; 12 000 in the RL repo | **No published absolute F1 brake torque.** Required (READING): tyre-limited deceleration at 90 m/s needs ≈ 14–15 kN m total (TUM μ, downforce, r 0.33; consistent with SF70H_TUM_CAR.md's 14.3 kN m lock torque); Brembo's ">6G" (incl. ≈ 1 g drag at 90 m/s) gives ≈ 12 kN m | [MS-BR17], [MB-BR] + READING | 2017 | low-medium (derived) | Set ≥ 15 kN m so the tyres, not the brakes, limit (real F1 brakes can lock the wheels at any speed). |
| `brake_bias_front` (knob 0.45–0.70) | 0.5 default | **Ideal bias for the TUM car ≈ 0.58–0.60 front** (equal-μ axles: front load share at 2–5 g with 45 % aero balance and 45.3 % static front, h 0.335, L 3.6; my calculation at 30/50/80 m/s gives 0.59/0.60/0.58) | READING (physics) | – | medium (derivation) | Default 0.5 is rear-biased vs ideal; range 0.45–0.70 covers F1 practice. Changing the default is a setup decision for Ibrahim. |
| Disc / caliper limits | not modelled | ≤ 278 mm × 32 mm carbon discs, ≤ 6-piston Al calipers; discs ~1000 °C | [FIA17] 11.2–11.3; [MB-BR] | 2017 | high | Nothing to map (OCD has no brake thermal model). |

## 7. Differential

- [FIA17] 9.9: devices that transfer torque from a slower to a faster wheel are prohibited (no torque vectoring); 9.1: at most two driven wheels.
- **[GA21]** Gary Anderson, "Differentials are F1's unsung set-up weapon", The Race, 9 Apr 2021, https://www.the-race.com/formula-1/how-f1-differentials-work.md. SAYS: "The differential in an F1 car is a bit like a clutch in that it's a multi-plate differential."; "Then you have an actuator that puts hydraulic pressure on those plates."; "So the completely free differential is low pressure, with the locked differential being high pressure." No numbers.
- [PL14] (viscous LSD model, eq. 35): baseline kd 10.47 N m s/rad; one-at-a-time optimum 2552.3 ("essentially locked"); joint optimum 2218.1 N m s/rad; LPR14/[West20] list kd as "100 N/rpm".

| OCD parameter | Now | F1 value | Conf. | Mapping |
|---|---|---|---|---|
| `LSD_preload`, `LSD_lock_drive`, `LSD_lock_coast` | knobs 0–300 N m, 0–1, 0–1 (default open) | **No published F1 preload or locking-fraction numbers.** F1 = hydraulically actuated multi-plate LSD spanning fully open to locked [GA21]; PL14's optimum is near-locked | none for numbers | Keep the knob ranges (they already span open→locked); defaults are setup choices. |

## 8. Powertrain items already from TUM (check only)

- [FIA17] 9.6.1: 8 forward ratios, nominated to the FIA (9.6.2). TUM's `i_trans` has 8 gears: consistent. No SF70H ratio set is public (not searched further).
- Engine inertia: derived from TUM's `e_i` in `powertrain.py`; no F1 crank-inertia source looked for.

## 9. Summary table

| Item | AV-21 value now | Proposed F1 value | Source (short) | Conf. |
|---|---|---|---|---|
| Yaw inertia Iz | 1159.6 kg m² | 560 (PL14 450 scaled to 733 kg, 3.6 m); test 560–1160 | PL14 Tab. 4 | medium-low |
| Roll / pitch inertia | 97.6 / 579.8 | none found; keep | – | none |
| Wheel (unsprung) mass | 15 kg | tyre alone 10 / 12 kg; corner total unsourced (est. 20–30) | West20 B.1 (TL16); Isola 2016 | tyre medium; total none |
| Wheel spin inertia | 1.15 / 1.35 | est. 0.8–1.2 / 1.0–1.4 (keep) | estimate from tyre mass + 2017 sizes | low |
| Tyre vertical stiffness | 250 kN/m | 2.5e5 (0 load) → 2.66e5 (3 kN) → 2.85e5 (6 kN) | ML16 eq. 17 (derived) | medium-low |
| Loaded radius | 0.3299 m | keep (ML16: 326 → 304 mm with load, older tyre) | ML16 eq. 17 | medium |
| PKX1/2/3 | 30.7 / 0.27 / 0.13 | 42.5 / −4.0 / 0 (or per-axle 42.2 / 49.8 to keep κ_peak with TUM μ) | PL14 Tab. 3 (derived) | medium-low |
| PKY1/2 | −75.5 / 4.65 | keep (= PL14 within 4 %); optional LKY 1.08 F / 1.26 R | PL14 (derived) | medium-low |
| PCX1, PCY1 | 1.6, 1.5 | 1.9, 1.9 | PL14 Qx, Qy | medium-low |
| PEX*, PEY* | 0.7…, 0.1… | 0 | PL14 (no E) | medium-low |
| Combined slip R* | AV-21 | none found; keep or fit to PL14 ellipse | – | none |
| Offsets PVY/PHY/RVY | AV-21 | none found; 0 = symmetric (PL14-like) | PL14 | none |
| Relaxation lengths | 0.3 / 0.2 m (hard-coded) | none found; keep | – | none |
| Max road-wheel steer | 0.6 rad | 0.30 rad (17°); Monaco 0.37–0.38 | Motorsport.com 2010; F1.com 2019 | medium |
| Steer rate | 1.5 rad/s | none found; est. 0.9–1.7 rad/s; keep | estimate | low |
| Steering ratio | n/a | ≈ 10:1 | Wheatley & Ali 2021 citing Milliken | low |
| Max brake torque (total) | 12 kN m (RL repo) | ≥ 15 kN m (tyre-limited at 90 m/s needs 14–15) | derived; Brembo ">6G" | low-medium |
| Brake bias range / ideal | 0.45–0.70, default 0.5 | ideal ≈ 0.58–0.60 | physics derivation | medium |
| Roll stiffness / front share | 275 835 N m/rad / 0.613 | ≈ 2.3e5 / ≈ 0.62 (keep) | ML16 Tab. I (derived) | medium-low |
| Heave element | off | needed: ≈ 1.0 MN/m per front wheel, ≈ 0.25 MN/m rear (suspension only) | ML16 Tab. I (derived) | low-medium |
| Ride frequencies incl. tyre | – | ≈ 5.9 Hz F / 4.2 Hz R | ML16 (derived) | low-medium |
| Damping, packers, knee | AV-21 | none found; keep | – | none |
| Roll / pitch centre heights | 0.2 m | none found; keep | – | none |
| Static ride height F / rake | 0.035 / 0.05 m | ≈ 0.014 m / 0.06 m (2013); SF70H rake 1.55° ≈ 0.097 m | ML16 §II-A; Motor Sport 2017 | medium-low |
| Footprint | 4.6 × 2.0 m | width ±1.035 m over tyres (±1.0 bodywork); overhang ≤ 1.2 m front, ≤ 0.71 m rear; SF70H length 5.0 m (uncited) | FIA 2017 Art. 3.2.2–3.2.3; Wikipedia | high (limits) / low (length) |
| LSD | open default | no numbers; hydraulic multi-plate open→locked | Anderson 2021; PL14 kd | none (numbers) |
| Mass check | 733 kg | 728 kg minimum (TUM consistent) | FIA 2017 Art. 4.1 | high |
| Wheelbase check | 3.600 m | ≈ 3.594 m | F1.com May 2017; Motor Sport | medium |

## 10. Implications for us (interpretation, not from the sources)

1. **Biggest evidence-backed changes:** steering lock 0.6 → 0.30 rad; yaw inertia (1160 → ~560 is what PL14 supports, but it is one source and looks low on a mass-distribution check, so run a sensitivity test first); longitudinal slip stiffness up ~40 % plus C = 1.9, E = 0 so the Fx peak moves from 14–27 % to ~8–11 % slip; a non-zero, stiff front heave element; brake torque ≥ 15 kN m.
2. **AV-21 values that are already F1-consistent:** lateral cornering stiffness (PKY), tyre vertical stiffness (within ~10 %), roll stiffness and its front share, wheel spin inertias, steer rate (no evidence against it).
3. **No credible F1 source exists** for MF combined-slip coefficients, tyre offsets, relaxation lengths, roll/pitch centre heights, damping, packers, roll/pitch inertias, unsprung corner mass, LSD numbers or brake torque. Keep the AV-21 values and label them PLACEHOLDER, or use the estimate methods above.
4. Nearly all tyre and suspension numbers are from **one research group's 2013-era car** (PL14/ML16/TL16, tyre data traced to Kelly 2008), not the 2017 SF70H. 2017 tyres are 25 % wider, so stiffnesses were probably higher. Quote them in the thesis as "representative F1 values (Perantoni & Limebeer 2014; Masouleh & Limebeer 2016)", not as SF70H data.
5. Which options to adopt (A/B for PKX/LKY, offsets to zero, Iz choice, extending the heave-knob range above 8 Hz, brake-bias default) are decisions for Ibrahim (PROJECT_CONTEXT §11). All the tyre changes can be checked cheaply with the existing MF sweep in `sf70h_checks.py` before any training run.
