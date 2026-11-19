@echo off
rem AC training worker for the F1 driver bcn_f1_r4. Runs until Ctrl+C. Run it again to continue
rem from the last committed actor (same tcp-worker-f1r4-001 folder, same Mac run tcp-f1r4-001).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -c "import numpy, torch" >nul 2>&1
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install numpy torch
  if errorlevel 1 goto failed
)
if not exist steering-f1r4.json (
  echo No steering-f1r4.json yet: run CALIBRATE_F1R4.cmd first.
  goto failed
)
if "%AC_LEARNER_HOST%"=="" set "AC_LEARNER_HOST=192.168.1.249"
if "%AC_TELEPORT_DIR%"=="" set "AC_TELEPORT_DIR=D:\SteamLibrary\steamapps\common\assettocorsa\apps\lua\rl_teleport"
.venv\Scripts\python.exe -m rlsetup.assetto.distributed worker --server "%AC_LEARNER_HOST%" --port 2350 --token-file tcp-session.key --calibration steering-f1r4.json --setup-id bcn_f1_best --work-dir tcp-worker-f1r4-001 --ac-host 127.0.0.1 --reference ocd_path --reference-file ocd_line_reference_bcn_f1_r4.json --max-steps 3750 --max-wall-s 180 --explore-std-scale 0.3 --eval-every 10 --reconnect-attempts 20 --max-consecutive-failures 5 --teleport-dir "%AC_TELEPORT_DIR%" --execute
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo The training loop stopped. Read the message above. Run this file again to continue.
pause
exit /b 1
