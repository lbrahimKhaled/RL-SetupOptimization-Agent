# Assetto Corsa on this PC (game and learner on one machine)

Interface: [assetto_corsa_gym](https://github.com/dasGringuen/assetto_corsa_gym) (Remonda et al., NeurIPS 2024, MIT), checked out next to this repo at `ef0fa50`. Target: SF70H (`ks_ferrari_sf70h`) on Shanghai (`china-gp`, a mod; assetto_corsa_gym ships its centre line and racing line).

## How it fits on one PC
```
acs.exe (High priority, 50 FPS cap)                 acenv python (learner, torch on CUDA)
  apps/python/sensors_par  (AC's Python 3.3)          AssettoCorsaEnv client, 25 Hz
    ego server        127.0.0.1:2345  <-- UDP -->       observations / actions
    opponents server  127.0.0.1:2346
    sim management    127.0.0.1:2347  <-- TCP -->       reset, static info
  reads the vJoy device  <------------------------     vJoyInterface.dll (client drives vJoy)
```
- Servers bind `127.0.0.1`, not upstream's `0.0.0.0` (`install_ac_plugin.ps1` patches the installed copy): nothing is exposed on the LAN and no firewall prompt appears.
- The client uses `127.0.0.1`, not `localhost` (`ac_local.overrides`): with a hostname, every UDP `sendto` re-resolves it.
- `vjoy_executed_by_server = False` with a 50 Hz plugin tick: upstream's local mode. The 100 Hz server-side vJoy mode is only for two machines.
- Physics and control are real time. `train_ac.ps1` raises `acs.exe` to High priority and gives the learner 4 torch threads (of 32), so learner bursts can't drop AC below 50 FPS. Keep the AC window in front: AC ignores controller input when it's not focused.
- Python envs: `ocdenv` (OCD training) is unchanged. `acenv` is a clone of it plus `requirements-ac.txt`, so the AC client's pins (gym, numba) can't change OCD numerics. Upstream pins (gym 0.21, numpy 1.23, Python 3.9) don't install on 3.12; the nearest working versions are used.

## Setup (in order)
| # | Step | Admin? | State on 17 Nov 2026 |
|---|---|---|---|
| 1 | Finish the Assetto Corsa download in Steam | no | **231 MB of 10.2 GB, paused** |
| 2 | Install [vJoy](https://sourceforge.net/projects/vjoystick/) 2.x (driver, default path `C:\Program Files\vJoy`) | **yes** (ask AUB IT) | missing |
| 3 | Content Manager (installed) > Settings > Custom Shaders Patch > Install (needed for resets) | no | after 1 |
| 4 | SF70H DLC (`ks_ferrari_sf70h`) in Steam; the `china-gp` Shanghai track mod into `content\tracks` | no | after 1 |
| 5 | `powershell -ExecutionPolicy Bypass -File platforms\windows\ac\setup_acenv.ps1` (assetto_corsa_gym clone + `acenv`) | no | **done** |
| 6 | `powershell -ExecutionPolicy Bypass -File platforms\windows\ac\install_ac_plugin.ps1` (AC closed) | no | after 1 |
| 7 | In AC: UI Modules > enable `sensors_par`; Controls > load `Vjoy`; Video > 50 FPS cap; assists as in upstream `INSTALL.md` §2.5 | no | after 6 |
| 8 | SF70H control maps for the gym: `AssettoCorsaConfigs\cars\ks_ferrari_sf70h\steer_map.csv` (normalised action at full lock and the lock in degrees, same format as `dallara_f317`) and `brake_map.csv` | no | after 7 |
| 9 | `acenv\python platforms\windows\ac\check_ac.py --live` with a Hotlap session running | no | after 8 |

`check_ac.py` (no `--live`) can be run any time; it prints what is still missing.

## Train
```powershell
platforms\windows\ac\train_ac.ps1                       # SF70H on china-gp, assetto_corsa_gym SAC, CUDA
platforms\windows\ac\train_ac.ps1 -Extra 'AssettoCorsa.car=dallara_f317','AssettoCorsa.track=ks_barcelona-layout_gp'   # upstream's tested pair
```
Output goes to `..\assetto_corsa_gym\outputs\<timestamp>\`. Upstream `train.py` crashes if `work_dir` is set (`args.work_dir` and top-level `config.track` don't exist at `ef0fa50`), so the launcher doesn't set it.

Not done yet: our own SAC driver (`rlsetup.driving`) on AC. It needs an adapter from `AssettoCorsaEnv` (gym 0.26 API, its own observation) to our 159-dim observation and 3 actions. That waits until the plugin runs on this PC (steps 1 to 9).
