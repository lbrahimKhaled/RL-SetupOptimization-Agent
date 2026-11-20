# Lenovo only: AC fine-tuning of bcn_f1_r5 (7M) and OCD training on one PC

Everything runs on the Lenovo: Assetto Corsa, the AC worker and the learner talk over 127.0.0.1 (no Mac).
Scripts were written on the Mac and are **untested on Windows**; report the first error you see.
Commands run in PowerShell from the `RL-SetupOptimization-Agent` folder, as
`powershell -ExecutionPolicy Bypass -File <script>.ps1` (no admin needed).

## Bring from other machines / folders
1. **`ocd_native.cp312-win_amd64.pyd`** (OCD training only; AC does not need it): copy it from the i9 PC's
   `ocd_first_lap` folder into this bundle's `ocd_first_lap\`. Static MinGW runtime, no extra DLLs. It cannot be
   built on the Mac; to build it here instead, follow `platforms\windows\README.md` ("ocd_native": WinLibs MinGW
   GCC + cmake/ninja/eigen/boost from conda; `vendor\` is already in the bundle).
2. **Steering map**: `steering-f1r4.json` from the existing `ac-lenovo-tcp` folder (setup bcn_f1_best, valid for
   F1R5). Optional: `ac-lenovo-tcp\tcp-worker-f1r4-001\heights.npz` (teleport starts work from the first drive).
3. **AC setup `bcn_f1_best`**: already on the Lenovo in `Documents\Assetto Corsa\setups\ks_ferrari_sf70h\ks_barcelona\`.
   It is not in the repo or this bundle; for any other PC copy that .ini from the Lenovo.

## Folder layout (unzip as is)
```
<folder>\README_FIRST.md
<folder>\RL-SetupOptimization-Agent\    this repo with .git (branch sf70h-f1 + uncommitted R5 AC changes)
<folder>\ocd_first_lap\                 ocd_first_lap branch sf70h-f1 (ee86683) incl. vendor\
<folder>\assetto_corsa_gym\             upstream at ef0fa50 (plugin source for install_ac_plugin.ps1)
<folder>\ocd_first_lap_sf70h-f1.bundle  git bundle of ocd_first_lap branch sf70h-f1
```
Keep the three folders side by side. Use a short path, e.g. `C:\Users\<you>\rl` (not inside `ac-lenovo-tcp`).

## Steps
1. **Already done on the Lenovo, skip (only for a fresh PC):** Steam + AC + Ferrari SF70H (Ferrari 70th
   Anniversary pack) + Barcelona; Content Manager (https://assettocorsa.club/content-manager.html); CSP
   (Content Manager > Settings > Custom Shaders Patch); vJoy 2.x (admin; device 1 with X, Y, Z and 2 buttons);
   sensors_par plugin and `apps\lua\rl_teleport`; Content Manager settings (100 FPS cap, sensors_par enabled,
   Vjoy controls preset, auto gearbox + auto clutch ON, TC/ABS/stability/damage/fuel/wear OFF, blankets ON).
   To redo the plugin/teleport part: AC closed, `$env:AC_INSTALL='D:\SteamLibrary\steamapps\common\assettocorsa'`
   then `powershell -ExecutionPolicy Bypass -File platforms\windows\ac\install_ac_plugin.ps1` (same window).
2. **Python 3.12** (if `py -3.12` does not work): https://www.python.org/downloads/windows/, 64-bit, "Install for
   current user". The `.venv` inside `ac-lenovo-tcp` has only numpy + torch; the learner and OCD need the full env:
3. **Python env**: `powershell -ExecutionPolicy Bypass -File platforms\windows\setup_pc.ps1`
   Creates a venv at `%USERPROFILE%\ocdenv`; torch 2.11.0 CUDA 12.8 build if `nvidia-smi` finds an NVIDIA GPU,
   else CPU (force with `-Torch cpu` / `-Torch cuda`); ocd_first_lap's requirements; SB3 / sb3-contrib 2.9.0,
   gymnasium 1.4.0, tensorboard, pytest; `ocd_first_lap.pth`; then `platforms\windows\check_setup.py`.
   Needs internet. Safe to re-run.
4. **ocd_native** (for OCD training): copy the .pyd into `ocd_first_lap\`, re-run setup_pc.ps1; all checks OK.
5. **Steering map**: copy `ac-lenovo-tcp\steering-f1r4.json` to
   `RL-SetupOptimization-Agent\runs\windows\ac\steering-f1r5.json` (create the folder). Only if you do not have
   it: AC on track with bcn_f1_best, car stopped, N visible, pedals released, you at the keyboard:
   `powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_calibrate_f1r5.ps1`.
   Optional: copy `ac-lenovo-tcp\tcp-worker-f1r4-001\heights.npz` to `runs\windows\ac\tcp-worker-f1r5-001\heights.npz`.
6. **Stop the old worker**: Ctrl+C in the START_LENOVO_F1R4 window (and any other worker).
7. **AC**: Hotlap, Ferrari SF70H, Barcelona GP, load setup **bcn_f1_best**, sensors_par on.
   Check: `%USERPROFILE%\ocdenv\Scripts\python.exe platforms\windows\ac\check_ac.py --live`
   (with `$env:AC_INSTALL='D:\SteamLibrary\steamapps\common\assettocorsa'` set in that window first).
8. **Learner** (window 1): `powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_server_f1r5.ps1`
   Work dir `runs\windows\ac\tcp-f1r5-001`, actor `runs\ac\source_actor_bcn_f1_r5_7m`, critics/alpha from
   `models\bcn_f1_r5_step7000k\seed0\checkpoints\last.zip`, the Mac's START_MAC_F1R5 schedule, bound to 127.0.0.1.
   Creates the local token `runs\windows\ac\tcp-session.key` (not the ac-lenovo-tcp one).
9. **Worker** (window 2, AC in front): `powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_worker_f1r5.ps1`
   Work dir `runs\windows\ac\tcp-worker-f1r5-001`, reference `runs\ac\ocd_line_reference_bcn_f1_r5_7m.json`,
   teleport dir `D:\SteamLibrary\steamapps\common\assettocorsa\apps\lua\rl_teleport` (default when that folder
   exists; else set `AC_INSTALL`). Same drive settings as START_LENOVO_F1R5.cmd.
   Windows power: Sleep and screen off **Never**, no screen saver/lock (the scripts keep the PC awake, but a lock
   screen can still stop AC reading vJoy).

## Progress
- `runs\windows\ac\tcp-f1r5-001\progress.jsonl`: one line per drive.
- TensorBoard: `%USERPROFILE%\ocdenv\Scripts\python.exe scripts\ac_progress_to_tb.py runs\windows\ac\tcp-f1r5-001 --follow`
  in a third window, then `%USERPROFILE%\ocdenv\Scripts\tensorboard.exe --logdir runs\windows\ac\tcp-f1r5-001\tb --host 127.0.0.1 --port 6007`.

## Stop and resume
Stop: Ctrl+C in the worker window, then in the learner window. Resume: start the learner, then the worker, with
the same scripts (same work dirs); learner continues from its last committed cycle, worker from the last actor.
Keep the work dirs and `tcp-session.key`.

## OCD training (rlsetup.driving.train)
Needs the .pyd. Foreground `platforms\windows\train.ps1 ...`; long runs detached:
`platforms\windows\start_detached.ps1 -Log runs\windows\NAME_stdout.log platforms\windows\train.ps1 ...`
(`platforms\windows\README.md`). Fresh runs get `train.overrides` (device=cuda, 16 envs); without CUDA add
`--set device=cpu`. Do not train OCD while AC fine-tuning runs (both share the CPU).

Generic resume of bcn_f1_r5 from 6M (bundle: `runs\mac\D2\bcn_f1_r5\seed0` with step_6000000 / step_7000000 /
best_aided, no replay buffer):
```powershell
New-Item -ItemType Directory -Force runs\windows\D2\bcn_f1_r5 | Out-Null
Copy-Item -Recurse runs\mac\D2\bcn_f1_r5\seed0 runs\windows\D2\bcn_f1_r5\seed0
Copy-Item runs\windows\D2\bcn_f1_r5\seed0\checkpoints\step_6000000.zip runs\windows\D2\bcn_f1_r5\seed0\checkpoints\last.zip
# edit runs\windows\D2\bcn_f1_r5\seed0\config.json:  "vec_start_method": "forkserver"  ->  "spawn"
platforms\windows\train.ps1 --resume --preset bcn_f1_r5 --seed 0 --run-name bcn_f1_r5 --allow-code-change --allow-dirty
```
- The Mac config has `vec_start_method: forkserver` (no such thing on Windows) and resume accepts no `--set` for
  it, so without the edit the resume fails when it starts the env workers. It keeps the Mac's device cpu, 4 envs.
- No replay buffer: continues with an empty buffer (warning; no updates until it refills).
- `evals.jsonl` / `status.json` cover the whole Mac run (to 10.35M), not cut at 6M.
- `--allow-code-change`: Windows ocd_native differs from the Mac's; `--allow-dirty`: uncommitted changes.
- `models\bcn_f1_r5_step7000k\seed0` is a clean resumable folder at 7M.
- **The speed branch from 6M is not decided yet**; its preset comes later.

## Optional: make ocd_first_lap a real git repo (Git needed)
```powershell
cd ..\ocd_first_lap
git init
git config core.autocrlf false
git fetch ..\ocd_first_lap_sf70h-f1.bundle sf70h-f1:sf70h-f1
git symbolic-ref HEAD refs/heads/sf70h-f1
git reset                 # index only; files untouched
git status                # should be clean
git remote add origin https://github.com/lbrahimKhaled/ocd_first_lap.git
```

## Troubleshooting
- **AC quits after ~2 s, `0xC000007B`**: 64-bit acs.exe needs the VC++ 2013 x64 runtime (`msvcp120.dll`,
  `msvcr120.dll`): with admin run Steam's `_CommonRedist\vcredist\2013\vcredist_x64.exe`; without, put the two
  DLLs in `%LOCALAPPDATA%\ac-vcruntime-x64` on the user PATH and restart Steam/Content Manager
  (`platforms\windows\ac\README.md` step 2). `Documents\Assetto Corsa\cfg\launcher.ini` must have `X86=0`.
- **Car freezes when AC loses focus**: CSP `[CONTROL] INACTIVE_ONLY_WHEN_COVERED=1` in
  `Documents\Assetto Corsa\cfg\extension\general.ini` (install_ac_plugin.ps1 sets it; restart AC). A locked
  screen may still count as covered (untested).
- **Firewall**: not needed, everything is on 127.0.0.1.
- **CRLF**: Git for Windows defaults to `core.autocrlf=true`, which changes every source hash. The repo's
  `.gitattributes` turns it off; for ocd_first_lap run `git config core.autocrlf false` (setup_pc.ps1 does if git
  is installed); if it was checked out with CRLF: `git rm -r -q --cached . ; git reset --hard`.
- **Worker: steering map / setup id mismatch**: the map must be for bcn_f1_best (step 5).
- **Worker: "CSP teleport app not found"**: set `AC_INSTALL` or re-run install_ac_plugin.ps1.
- **Learner: "different settings/source"**: that work dir belongs to other arguments; use a new `-WorkDir`.
