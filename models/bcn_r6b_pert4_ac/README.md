# Barcelona SF70H AC warm start from R6b / pert4

This is the SAC driving policy from the **83.437 s selection flying lap**
and **83.447 s reporting flying lap**, under OCD with the `pert4` setup.
The starting model was `bcn_f1_r6b` `best_aided.zip` at +1M R6b steps.
The exact exported weights came from the +20k A2 driver fine-tune on
seed 13. The later fresh-confirmation record of 83.692 s used seed 16.
The setup actor's weights are a separate model.

The AC learner can initialize the driver actor, source critics, target
critics, and entropy coefficient (0.06073243) from these files. AC starts
fresh optimizer state and replay. `r6b_pert4_seed13_ac_init.zip` is a
source archive for the AC learner's `--critic-init` and the AC actor
exporter. Use the included `actor/` directory as `--actor-bundle`.
The runtime preserves the original R6 throttle map. The included
Barcelona reference has the exact OCD `pert4` speed profile and corner
targets registered to the existing Barcelona GP geometry.

## Windows launch

On the other PC, fetch and check out this branch:

```powershell
git fetch origin ac-r6b-pert4
git switch --track origin/ac-r6b-pert4
```

Assumes the one-PC AC environment is already installed. The
[Windows AC instructions](../../platforms/windows/README_FIRST_onepc.md)
cover Python, vJoy, sensors_par, CSP and the teleport app. Load SF70H
on Barcelona GP in Hotlap, prepare the setup from the sheet below, and
save it as **bcn_r6b_pert4**. Calibrate on that PC with the game on track,
the car stopped in neutral, pedals released, and someone at the controls:

```powershell
powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_calibrate_f1r5.ps1 -SetupId bcn_r6b_pert4 -Output runs\windows\ac\steering-r6b-pert4.json
```

Start the learner, then start the worker in another terminal. The learner
and worker run on the Windows PC and connect over 127.0.0.1 on port 2352.

```powershell
powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_server_f1r5.ps1 -Port 2352 -WorkDir runs\windows\ac\tcp-r6b-pert4-001 -ActorBundle models\bcn_r6b_pert4_ac\actor -CriticInit models\bcn_r6b_pert4_ac\r6b_pert4_seed13_ac_init.zip
powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_worker_f1r5.ps1 -Port 2352 -WorkDir runs\windows\ac\tcp-worker-r6b-pert4-001 -SetupId bcn_r6b_pert4 -Calibration runs\windows\ac\steering-r6b-pert4.json -Reference models\bcn_r6b_pert4_ac\barcelona_reference.json
```

Reuse those AC work directories to resume the AC run. The local
initialization and model inference were verified. The Windows AC game,
steering calibration, and on-track behavior require the other PC.

## SF70H setup sheet

Copy your saved `bcn_f1_best.ini`, then save a new
`bcn_r6b_pert4.ini` so the baseline is available for comparison. The
previous handoff places it in
`Documents\Assetto Corsa\setups\ks_ferrari_sf70h\ks_barcelona\`
on the Lenovo. The exact prior save was unavailable on the Mac.
Left/right values in the table are symmetric.

| AC setting | Closest available `pert4` target |
|---|---|
| Front brake bias | **54%** in the setup menu, from an OCD target of 54.406% |
| Front wheel rate LF/RF | **4 clicks each**, 80 N/mm |
| Rear wheel rate LR/RR | **7 clicks each**, 50 N/mm |
| Front / rear heave wheel rate HF/HR | **10 / 8 clicks**, 180 / 150 N/mm |
| Front / rear anti-roll bar | **0 / 12 clicks**, 30 / 29 kN/m. Use the rear maximum if AC exposes 30 kN/m. |
| Front corner dampers | **1 / 6 / 1 / 6** on each side |
| Rear corner dampers | **0 / 2 / 0 / 2** on each side |
| Front heave dampers | **19 / 24 / 19 / 24** |
| Rear heave dampers | **11 / 24 / 11 / 24** |

Damper order is **bump / fast bump / rebound / fast rebound**. These
clicks match each rate independently as closely as the AC menu permits.
The suspension targets exceed several AC limits. Combined roll
balance and driving behavior will differ. The click ranges come from
the game's [SF70H setup data](../../data/F1MODELL/setup.ini).

- **Gears and differential.** AC fixes gears 1–8 at
  **2.9688 / 2.3943 / 2.0411 / 1.7155 / 1.4800 / 1.2800 / 1.1400 / 1.0400**
  and the final drive at **4.42**. The AC differential file lists
  **15% power / 30% coast / 150 N·m preload**. Keep those fixed values.
  The OCD final-drive multiplier was 1.000 and its ratios differ.
  [AC car data](../../data/F1MODELL/drivetrain.ini)
- **Alignment and height.** OCD targets were camber **−3.57° front /
  −2.06° rear**, toe-in **0.047° front / 0.112° rear**, and static heights
  **20.0 mm front / 32.2 mm rear**. These are physical readings.
  Retain the existing ride-height and toe clicks until you can check
  live geometry. Use the camber targets to check the live readings.
- **Wings, brake power, packers, tyres and fuel.** Keep the entries
  from `bcn_f1_best` for the first AC test. The previous screenshots
  showed **17 psi cold** on all four tyres. Wing clicks have no verified
  inverse from the OCD aero model. OCD ran with ERS off and DRS closed.
  For an AC transfer with the existing electronics setup, record the
  resulting ERS mode alongside the test.
- **Earlier Barcelona screenshot settings** for comparison. Corner
  dampers front **13 / 5 / 15 / 8** and rear **21 / 8 / 19 / 13**,
  wheel rates **0 / 3** front/rear, heave **4 / 5** front/rear, ARBs
  **17 / 5** front/rear, and height menu **250 / 0** front/rear.
  These are the photographed settings and cover part of the save.

[Exact OCD setup](ocd_pert4_setup.json) · [Checkpoint and lap provenance](provenance.json)
