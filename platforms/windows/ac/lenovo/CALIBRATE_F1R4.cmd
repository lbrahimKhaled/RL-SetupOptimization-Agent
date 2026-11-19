@echo off
rem One-time steering calibration for the bcn_f1_best setup (about 10 s).
rem Before running: AC on track with the new setup loaded, car stopped, gear N visible, pedals released.
rem Stay at the controls ready to stop AC; the steering moves through its full range.
cd /d "%~dp0"
if exist steering-f1r4.json (
  echo steering-f1r4.json already exists. Delete or rename it to calibrate again.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m rlsetup.assetto.calibration --full-range --host 127.0.0.1 --setup-id bcn_f1_best --output steering-f1r4.json --trace steering-f1r4-trace.jsonl --execute --neutral-confirmed --at-controls --speed-mps 0.2 --motion-m 0.1
pause
