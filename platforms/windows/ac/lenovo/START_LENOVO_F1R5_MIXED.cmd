@echo off
rem Run from the existing v15 ac-lenovo-tcp folder. Fresh work-dir requests fork actor via hello.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe goto failed
.venv\Scripts\python.exe -c "import numpy, torch" >nul 2>&1
if errorlevel 1 goto failed
if not exist steering-f1r4.json goto failed
if not exist tcp-worker-f1r5-mixed-001 mkdir tcp-worker-f1r5-mixed-001
if not exist tcp-worker-f1r5-mixed-001\heights.npz if exist tcp-worker-f1r5-001\heights.npz copy /y tcp-worker-f1r5-001\heights.npz tcp-worker-f1r5-mixed-001\heights.npz >nul
if "%AC_LEARNER_HOST%"=="" (
  echo Set AC_LEARNER_HOST to the Mac's current IP in this command window first.
  goto failed
)
if "%AC_TELEPORT_DIR%"=="" set "AC_TELEPORT_DIR=D:\SteamLibrary\steamapps\common\assettocorsa\apps\lua\rl_teleport"
.venv\Scripts\python.exe -m rlsetup.assetto.distributed worker --server "%AC_LEARNER_HOST%" --port 2351 --token-file tcp-session.key --calibration steering-f1r4.json --setup-id bcn_f1_best --work-dir tcp-worker-f1r5-mixed-001 --ac-host 127.0.0.1 --reference ocd_path --reference-file ocd_line_reference_bcn_f1_r5_7m.json --max-steps 3750 --max-wall-s 180 --explore-std-scale 0.3 --eval-every 10 --reconnect-attempts 20 --max-consecutive-failures 5 --teleport-dir "%AC_TELEPORT_DIR%" --execute
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo The mixed training loop stopped. Check the message and existing v15 environment.
pause
exit /b 1
