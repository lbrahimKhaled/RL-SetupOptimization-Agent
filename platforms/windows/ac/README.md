# Assetto Corsa on this PC (game and learner on one machine)

Two stacks:
- **`rlsetup.assetto` (main):** the code that ran on the Lenovo (worker, bundle v11 from `ac-lenovo-tcp`, hashes in `data/ac/lenovo_v11_BUILD.json`) with the learner on the Mac. Here, both run on this PC over 127.0.0.1. SF70H (`ks_ferrari_sf70h`) on Barcelona GP (`ks_barcelona`, `layout_gp`). The source actor is an OCD SAC policy (sf70h_tum on Catalunya, 159 obs, 3 actions). Runs in `ocdenv`.
- **Upstream `assetto_corsa_gym` trainer (optional):** `train_ac.ps1` in `acenv`. It's only used as a cross-check. Its in-game plugin (`sensors_par`) is shared by both stacks.

## How it fits on one PC
```
acs.exe (High priority, 100 FPS cap)              ocdenv python
  apps/python/sensors_par (AC's Python 3.3)
    ego server    127.0.0.1:2345 <-- UDP 25 Hz -->  worker (ac_worker.ps1): actor inference, CPU, 1 thread
    management    127.0.0.1:2347 <-- TCP ------->     reset, static info
    drives vJoy itself, 100 Hz tick                     | chunks (NPZ) / new actor (zip), token-authenticated
                                                        v
                                                    learner (ac_server.ps1), 127.0.0.1:2350  [learn.py: Mac]
```
- `install_ac_plugin.ps1` patches the installed plugin. The servers bind `127.0.0.1` (upstream binds `0.0.0.0`). `vjoy_executed_by_server = True` at 100 Hz, which `rlsetup.assetto` requires (`SteeringMap.verify`: "server-side vJoy controls required"; `Parked`: 100 / 25 Hz).
- The worker raises `acs.exe` to High priority, so the learner's update bursts can't drop AC below its 100 Hz tick.
- The TCP token is local: `runs\windows\ac\tcp-session.key` is created on first use and git-ignored. The Lenovo's key belonged to the Mac pairing, so it isn't reused here.

## Walkthrough
State on 17 Nov 2026: AC is installed, CSP is installed, the SF70H and Barcelona are present, and the plugin is installed (step 3). `check_ac.py` shows what's left at any time:
`C:\Users\imk14\ocdenv\python.exe platforms\windows\ac\check_ac.py` (add `--live` while a session runs).

1. **vJoy (needs admin).** Run `Downloads\vJoySetup.exe` (vJoy 2.x) and accept the driver prompt. Then open **Configure vJoy**: device 1 must be enabled with axes **X, Y, Z** and at least **2 buttons** (the default device 1 has these). `Vjoy.ini` maps X = steering, Y = throttle, Z = brake, button 1 = gear up, button 2 = gear down.
2. **Custom Shaders Patch:** done. The plugin's car reset needs it.
   **64-bit AC without admin:** the plugin only works in 64-bit `acs.exe` (its `system\x64` libs; `launcher.ini` must have `X86=0`). 64-bit `acs.exe` needs the VC++ 2013 x64 runtime, which Steam can't install without admin (the admin prompt on every launch). Without it, AC quits in about 2 s with `0xC000007B`. Fix used here: `msvcp120.dll` and `msvcr120.dll` (x64, Microsoft-signed, extracted from Steam's `_CommonRedist\vcredist\2013\vcredist_x64.exe`) are in `%LOCALAPPDATA%\ac-vcruntime-x64`, and that folder is on the user PATH. Restart Steam and Content Manager after changing PATH. With admin, running that installer does the same thing.
3. **Plugin, socket libs, controller presets:** done by `install_ac_plugin.ps1`. It copied `sensors_par` into `assettocorsa\apps\python`, the socket DLLs and Lib into `system\x64`, and `Vjoy.ini` / `WASD.ini` into `Documents\Assetto Corsa\cfg\controllers\savedsetups`. On the Lenovo this was done by hand: dragging the files into Content Manager and the AC folders. To redo it, close AC and run it again.
4. **Content Manager settings:**
   - Settings > Assetto Corsa > Video: **framerate limit 100** (the plugin ticks at 100 Hz). Windowed or borderless makes it easier to keep the window focused.
   - Settings > Assetto Corsa > Apps (Python apps): enable **sensors_par**.
   - Settings > Assetto Corsa > Controls: **Load preset > Vjoy** (from `savedsetups`). The input should show "vJoy Device" for steer, throttle and brake.
   - Assists: automatic gearbox ON (the actor has no gear output), auto clutch ON, TC / ABS / stability OFF, damage OFF, fuel and tyre wear OFF, tyre blankets ON (upstream `INSTALL.md` §2.5).
5. **Start a session.** Drive > Hotlap, car **Ferrari SF70H**, track **Barcelona, layout GP**, then Go. In the game, open the right-hand app bar and turn on `sensors_par`. Then run `check_ac.py --live`: it should print `car=ks_ferrari_sf70h track=ks_barcelona`.
6. **Steering calibration for this PC** (about 10 s): car stationary, **N** visible, you at the keyboard. Run `powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_calibrate.ps1`, which writes `runs\windows\ac\steering.json`. The Lenovo map is kept in `data\ac\steering_sf70h_barcelona_lenovo.json` for comparison: about ±0.300 rad of road-wheel angle at full lock, reached by command ±0.5.
7. **Train** (two windows, AC in front):
   ```powershell
   platforms\windows\ac\ac_server.ps1 -WorkDir runs\windows\ac\server-001 -ActorBundle <source actor dir> --critic-updates C --updates U --batch-size B ...
   platforms\windows\ac\ac_worker.ps1 -WorkDir runs\windows\ac\worker-001
   ```
   The worker settings are the Lenovo's (`START_LENOVO_FULL.cmd` v11): `ocd_path` reference (`data\ac\ocd_path_reference_barcelona.json`), 3750 steps / 180 s per drive, exploration 0.3, an evaluation drive every 10th, and 5 consecutive failures allowed. Ctrl+C stops it; run it again to continue.

## Learner code (from the Mac branch `ac-barcelona-transfer`, 095d6d8)
`learn.py`, the newer `distributed.py` (adds `--critic-init`; worker side unchanged), the `tests/test_ac_*` suite, `scripts/ac_*` and `docs/AC_QUICKSTART.md` / `docs/handoff/AC_*.md` are taken from that branch as-is (the rest of the branch, slides and C2 work, is not merged). On this PC the 257 AC tests pass, and so does `scripts/ac_loopback_smoke.py` (real learner and worker processes against a simulated 25 Hz AC, with a random stand-in actor via `--actor-bundle`). The harness needed one Windows fix: ignore `ConnectionResetError` on UDP `recvfrom`, as the real plugin does.

## Still missing from the Mac (blocks step 7)
Model files are git-ignored, so copy them over by hand (USB or cloud), keeping the paths:
1. `runs/ac/source_actor/` (`actor.pt`, `config.json`, `manifest.json`; manifest sha256 `e4cdf256…` in `data/ac/lenovo_v11_BUILD.json`).
2. `runs/mac/D2/bcn/seed0/checkpoints/best.zip` and `runs/mac/D2/bcn/seed0/config.json`. These are for `--critic-init`, which accepts only that exact checkpoint. With them, (1) can also be re-exported here: `python -m rlsetup.assetto.actor export --checkpoint ...best.zip --config ...config.json --output runs\ac\source_actor`.
3. `runs/ac/tcp-launch/START_MAC_FULL.command`, for the exact `--updates` and `--batch-size` (the rest of the schedule is in `ac_server.ps1`).
4. Optional, to continue the Mac run rather than start fresh: the Mac server work dir `runs/ac/tcp-full-002`.

On the Mac:
```bash
cd <repo> && zip -r ac_for_windows.zip runs/ac/source_actor runs/mac/D2/bcn/seed0/checkpoints/best.zip \
    runs/mac/D2/bcn/seed0/config.json runs/ac/tcp-launch/START_MAC_FULL.command
```
Unzip it into the repo root here.

## Upstream trainer (optional)
`setup_acenv.ps1` builds `acenv` (a clone of ocdenv plus `requirements-ac.txt`, numpy < 2.5 for numba's `cross2d`). `train_ac.ps1` runs upstream `train.py` with `ac_local.overrides`. Its car configs cover only `dallara_f317` (a mod), `bmw_z4_gt3` and `ks_mazda_miata`, not the SF70H. Output goes to `..\assetto_corsa_gym\outputs\<timestamp>\` (upstream crashes if `work_dir` is set).
