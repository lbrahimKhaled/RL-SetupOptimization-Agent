# Assetto Corsa: suspension stiffness and frequency in a car setup

Scope: Assetto Corsa (PC, Kunos), not Competizione. Researched 9 Nov 2026.

**How to read this file.** "Source" marks what a document says. "Reading" marks inference from those documents. Most AC evidence is community material: modder files, forum posts and community documentation of the Custom Shaders Patch (CSP). No Kunos-authored specification of these files was found. Forum quotes come from extracted page text, so check the wording against the page before quoting it in the thesis. The OptimumG PDFs and the Rouelle article were read in full. Access status for every source is in the last section.

## 1. Setup knobs for springs and anti-roll bars

**Source**

- `setup.ini` defines each knob as its own block with `MIN`, `MAX`, `STEP` and display keys. Example from a student-team AC car mod ([MUR 2019 `setup.ini`, commit 3eff3dd](https://gitlab.eng.unimelb.edu.au/fwongkwok/mur-assettocorsa/-/raw/3eff3dd3bb0c795fd25163a424f9258fb702b3ac/mur_2019/data/setup.ini)):

| Block | NAME | MIN | MAX | STEP | SHOW_CLICKS | HELP |
|---|---|---|---|---|---|---|
| `[SPRING_RATE_LR]` | Wheel rate LR | 70 | 136 | 1 | 0 | `HELP_LR_WHEELRATE` |
| `[SPRING_RATE_HF]` | Wheel rate HR (sic) | 115 | 217 | 17 | 0 | `HELP_HF_WHEELRATE` |
| `[ARB_FRONT]` | ARB Front | 32072 | 117024 | 2072 | 2 | `HELP_FRONT_ARB` |
| `[ARB_REAR]` | ARB Rear | 489 | 1647 | 579 | 2 | `HELP_REAR_ARB` |

- That file also has `ROD_LENGTH_*` and per-corner `PACKER_RANGE_LF/RF/LR/RR` blocks. It has no `SPRING_RATE_LF/RF`; the front uses the single `HF` element.
- A `[SPRING_RATE_LF]` block with `NAME=Wheel rate LF`, `MIN=80`, `MAX=160`, `STEP=5`, `HELP=HELP_LF_WHEELRATE` appears in a [Scribd copy of a `setup.ini`](https://www.scribd.com/document/709471331/setup-ini). This was seen only as a search excerpt, and the car is unidentified.
- The in-game knob is labelled wheel rate. "in this sim we don't see the spring rate, only the wheel rate" (FezzantPlucka, 19 Feb 2017, [Kunos forum, p. 2](https://www.assettocorsa.net/forum/index.php?threads/working-out-spring-rate-in-assetto-corsa.43186/page-2)). The same poster gives the BMW Z4 GT3 range as "front WR of 150-240 Nmm and a rear WR of 55-90 Nmm" ([p. 1, post 12](https://www.assettocorsa.net/forum/index.php?threads/working-out-spring-rate-in-assetto-corsa.43186/)) and later writes "150 N/mm".
- `suspensions.ini` stores a base value per axle. From the [MUR `suspensions - Copy.ini`, commit 182280d](https://gitlab.eng.unimelb.edu.au/fwongkwok/mur-assettocorsa/-/raw/182280d711fb69e465344f306efef2d0b7613fe6/mur_2019/data/suspensions%20-%20Copy.ini): `SPRING_RATE=38800 ; Wheel rate stifness in Nm. Do not use spring value but calculate wheel rate`, and `[ARB] FRONT=122400 ; Front antiroll bar stifness. in Nm`. A second modder's file carries the same comment ([assettocorsamods.net thread](https://assettocorsamods.net/threads/debugging-car-handling-and-suspension.1847/)).
- Unit correction: the [CSP-derived docs](https://docs.assetto.cn/en/car/physics/suspension) write `FRONT = 100000 ; Front anti-roll bar stiffness in N/m`.
- Extended cars only: CSP adds `[SPRING_LF]` ("replaces old SPRING_RATE_LF entry") and `[ARB_F]` ("replaces old vanilla AC entry"), with `LUT` values in `name|rate` format (same docs). CSP "cosmic" cars differ from vanilla: "no more wheel rates, spring rates are the rates of the springs themselves" ([CSP cosmic suspension docs](https://cup.acstuff.club/docs/csp/cars/cosmic-suspension)).

**Reading**

- The player edits wheel rate, not coil rate.
- The stiffness inside `suspensions.ini` is N/m despite the "Nm" comment, because the comment's magnitudes and the CSP docs only make sense as a linear rate.
- Scale differs between knobs. In the MUR file springs are around 10² (N/mm scale, matching the forum's N/mm) while ARBs are around 10⁴ to 10⁵ (N/m scale). This rests on one student mod, so it must be checked on the SF70H file.
- `SHOW_CLICKS` semantics are not documented in anything found.

## 2. Ride frequency, natural frequency and damping ratio

**Source**

- No `setup.ini` block examined defines a frequency knob. Knobs are stiffness, length or range values (section 1).
- The read-out is in the developer "Car Engineer" app:
  - "Can't you use the Hz to setup the car? It's displayed on the car engineer app" (Leonardo Ratafia, 16 Feb 2017, [Kunos forum](https://www.assettocorsa.net/forum/index.php?threads/working-out-spring-rate-in-assetto-corsa.43186/)).
  - It is enabled with `ENABLE_DEV_APPS=1` in `assetto_corsa.ini` (same thread).
  - A Steam user says it shows "suspension frequencies and damping rates", "only for the slow dampers, sadly", plus "sprung and unsprung mass" (jason, 30 May 2015, [Steam thread](https://steamcommunity.com/app/244210/discussions/0/618463446155043156/)).
  - A user reported 3.5 Hz front and 2.9 Hz rear on the BMW Z4 GT3 from these apps ([p. 2](https://www.assettocorsa.net/forum/index.php?threads/working-out-spring-rate-in-assetto-corsa.43186/page-2)).
- Reliability warning: "There are many bugs/glitches in these apps as they haven't been updated in ages" (ling.speed, same Steam thread).
- Whether the app's Hz includes heave elements was asked and not answered: "Is this because the car engineer app doesn't take into account the heave dampers and with those values it'd be >4hz?" (Timberwind, 27 Jun 2017, [Kunos forum](https://www.assettocorsa.net/forum/index.php?threads/damping-ratio-ride-frequency-questions.46113/)).
- General formula, not AC-specific: f = (1/2π)·√(K/M), with K the wheel rate (N/m) and M the suspended mass (kg), and K = K_s/MR² with MR = wheel travel / spring travel (Rouelle, section 5).

**Not found**

- The formula the Car Engineer app uses.
- Any damping-ratio read-out described as such.
- Any Content Manager display of frequency. This is a negative result from searches and one file of its source, not an audit of Content Manager.

**Reading**

- Frequency is a read-out only, in a dev app, and never a knob.
- On vanilla cars the value in the file is already a wheel rate, so a corner frequency needs only that rate and the sprung corner mass, with no motion ratio. Heave elements, ARBs and tyre series stiffness are not captured by this one-corner formula. Rouelle states the first two exclusions and separately notes the tyre is in series.

## 3. Coupling or feasibility constraints

**Source**

- Spring keys are per corner: `[SPRING_LF] ; LF, RF, RR, LR` and `[PACKER_RANGE_LF] ; LF, RF, LR, RR` ([CSP-derived docs](https://docs.assetto.cn/en/car/physics/suspension)). The MUR file has separate `LR` and `RR` blocks. ARBs are per axle (`ARB_FRONT`, `ARB_REAR`).
- No source states that springs parametrise ARBs, or the reverse, or that front and rear are tied.
- Knob-level interactions that are reported:
  - "ride height also varies depending on the wheel rate values" ([Steam setup guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2231125504)).
  - On helper-spring cars, "The Wheelrate RL and RR values you adjust with BMW change both helper spring and mainspring stiffness" (Kade, 19 Feb 2017, [p. 2](https://www.assettocorsa.net/forum/index.php?threads/working-out-spring-rate-in-assetto-corsa.43186/page-2)).
- One structural constraint, on heave length rather than stiffness: Content Manager's repair text says AC "might crash if there is ROD_LENGTH_HF or ROD_LENGTH_HR section in setup.ini" when corner suspension is present ([actools `RodLengthHeaveRepair.cs`, master, not pinned; the class is commented out](https://github.com/gro-ove/actools/blob/master/AcManager.ContentRepair/Repairs/RodLengthHeaveRepair.cs)).

**Not found**

- Whether the in-game screen offers a left/right link or symmetric option.

**Reading**

- In the file format, springs and ARBs are independent numbers limited by per-knob MIN/MAX/STEP. Front/rear and left/right are independent at file level. This is inferred from the format and the absence of any contrary statement; it is not documented.

## 4. Ferrari SF70H (`ks_ferrari_sf70h`) specifically

**Not found.** No opened source gives SF70H spring or ARB ranges, or its list of setup items, or confirms heave springs, packers or rod length on this car. Kunos car data is shipped in `data.acd`, and a modder reports that edits need the original renamed or removed ([Overtake excerpt](https://www.overtake.gg/threads/assetto-corsa-suspension-ini-changes-not-applying-in-game.281803/)).

Attempts that returned nothing usable:

- The [Patacuack car page](https://assetto.patacuack.net/car/ks_ferrari_sf70h) has no suspension data.
- A YouTube description excerpt lists ARB "26 front, 3 rear" for the SF70H ([video](https://www.youtube.com/watch?v=tLrPORgmGho)). The page itself did not load the description, and those numbers could be clicks or raw values.
- Overtake and simracingsystem SF70H setup threads could not be opened.

**Indirect evidence, another Kunos F1 car (F138).** Timberwind: "Based on the F138 default setup it seems to be about 2-3x more than a single wheel". Kade: "I ended up running softest front corner springs and maximum front heave stiffness" (28 Jun 2017, [Kunos forum](https://www.assettocorsa.net/forum/index.php?threads/damping-ratio-ride-frequency-questions.46113/)).

**Reading.** Kunos F1 cars expose heave stiffness, so the SF70H probably does too, but this is unconfirmed. A dump of the car's `setup.ini` from a PC install, or reading the setup screen min/max values on the PS4 copy, would settle it.

## 5. Race-engineering guidance on ride frequency

**Source (OptimumG, read in full)**

- Method: "The first step in choosing spring stiffness is to choose your desired ride frequencies, front and rear." K_s = 4π²f²·m_sm·MR² (N/m, MR = wheel/spring travel). Ranges: 0.5–1.5 Hz passenger; 1.5–2.0 Hz sedan racecars and moderate-downforce formula cars; **3.0–5.0+ Hz high-downforce racecars** ([Giaraffa, Tech Tip Part One](https://optimumg.com/wp-content/uploads/2020/01/SpringsDampers_Tech_Tip_1.pdf)).
- Front/rear split, same tip: "A common split is 10 – 20% front to rear", with a higher rear for flat ride. That theory comes from passenger cars. Higher damping leads "some racecars" to run a higher front, for corner-entry response, less front ride-height variation, and rear traction. It "should be chosen based on which is more important".
- Roll, after ride: choose a roll gradient (**0.2–0.7 deg/g for stiff higher-downforce cars**), compute the springs' contribution, then use ARBs to reach the target. Distribution is the "Magic Number" ("Milliken calls it Total Lateral Load Transfer Distribution"): "As a baseline, use 5% higher Magic Number than the static front weight distribution" ([Tech Tip Part Two](https://optimumg.com/wp-content/uploads/2020/01/SpringsDampers_Tech_Tip_2.pdf)). OptimumG also lists 0.25–0.75 deg/g for high-downforce race cars and the same order of choices ([Bar Talk](https://optimumg.com/bar-talk/)).
- Third spring: acts in ride but not single-wheel bump, allowing "a lower frequency in single wheel bump than overall ride". Example: 1.5 Hz ride with 1.0 Hz single wheel. Ride damping ratio baseline for racecars is 0.65–0.70 ([Tech Tip Part Three](https://optimumg.com/wp-content/uploads/2020/01/SpringsDampers_Tech_Tip_3.pdf)).

**Source (Rouelle, "Of springs and dampers", [Racecar Engineering, Nov 2020, pp. 50–52](https://optimumg.com/wp-content/uploads/2021/10/racecar-2020_11.pdf))**

- "for aerodynamic racecars – 3.0-7.0Hz". The 7 Hz end applies to the "front suspension of aerodynamic racecars for which front ride height control is critical", often with "an additional third (heave) spring, and most probably the use of bump stops".
- Limits of the formula: it "is for one corner of the car only. Third (heave) springs are not taken into consideration here, nor are anti-roll bars."
- Frequency's value: it "links both masses, kinematics of the motion ratio and suspension stiffness", so it is "a common denominator" for comparing cars.

**Other**

- "High-downforce prototype and formula cars can run 3.0 Hz and above"; a common start is rear 10–20% above front ([Penske, 12 May 2026](https://www.penskeshocks.com/blog/natural-frequency-ride-frequency-cpm-in-race-suspension)).

**Milliken (RCVD) could not be opened.**

- Suspension Secrets, as seen in a search excerpt only (the page returned 403), lists passenger 30–50, sports 70–90 and open-wheel without ground effect 95–120 cycles/min, and attributes its equations to RCVD ([page](https://suspensionsecrets.co.uk/calculating-ideal-spring-and-roll-bar-rates/)). Do not cite this as Milliken without checking the book.
- The Chapter 16 "Ride and Roll Rates" content was not verified.

**Not found.** Any published 2017 F1 ride or heave frequency.

## Implications for us (interpretation)

1. Frequency is our parametrisation, not an AC knob. Moving a setup to AC means f → corner wheel rate → snap to the car's MIN/MAX/STEP, and an MR is not needed for vanilla AC cars.
2. The sourced corner-frequency prior for a high-downforce car is 3–7 Hz (Rouelle), with higher values at the front and with heave and bump stops carrying part of the load. Our `sf70h_f1` front heave of 12.95 Hz (`docs/reports/SF70H_F1_UPGRADE.md`) is outside that range, but Rouelle's figure is not defined to include heave, so the two are not directly comparable.
3. The constraint `K_b ≥ 0` in PROJECT_CONTEXT §5 is ours. Nothing found suggests AC enforces a spring-ARB relation, so its only limits should be per-knob ranges.
4. Before fixing any AC-side bounds, the actual `ks_ferrari_sf70h` setup items are needed. This remains unresolved and is a decision for Ibrahim.

## Source access log

| Source | Status |
|---|---|
| OptimumG Tech Tips 1–3 (PDF metadata 2005–2010), Rouelle 2020 | Read in full |
| Kunos forum threads 43186 (pp. 1–2), 46113; Steam 618463446155043156 | Opened (forum posts, mostly 2015–2017; 46113 not read to the end) |
| MUR `setup.ini` and `suspensions - Copy.ini` raw files | Opened (student mod, not Kunos) |
| docs.assetto.cn (AssettoCN, "Built on CSP Official Wiki", community); cup.acstuff.club CSP docs; actools source file | Opened |
| OptimumG Bar Talk; Penske blog; Steam setup guide 2231125504; assettocorsamods thread 1847; Patacuack | Opened |
| Scribd `setup.ini`; Overtake `suspensions.ini` thread; Suspension Secrets; SF70H YouTube description | Search excerpt only |
| Overtake SF70H threads; simracingsystem SF70H thread; RCVD text; Autosport Milliken thread | Could not open (403, certificate error, or no accessible text) |
