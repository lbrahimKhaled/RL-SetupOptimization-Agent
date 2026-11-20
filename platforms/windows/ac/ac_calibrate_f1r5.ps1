# One-time steering calibration of THIS PC for setup bcn_f1_best (vJoy command -> SF70H road-wheel
# angle, about 10 s), for ac_worker_f1r5.ps1. Same arguments as the Lenovo's CALIBRATE_F1R4.cmd.
# The map depends on this PC's vJoy and AC control settings, so it must be measured here; the
# Lenovo's map is kept in data\ac\steering_sf70h_barcelona_bcn_f1_best.json for comparison only.
# Before: AC on track with bcn_f1_best loaded, car stopped, gear N visible, pedals released, you at the
# keyboard ready to stop AC (the steering moves through its full range).
#   powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_calibrate_f1r5.ps1
# Output: runs\windows\ac\steering-f1r5.json (never overwritten; delete or rename it to measure again).
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
Set-Location $Repo
$dir = 'runs\windows\ac'
$out = "$dir\steering-f1r5.json"
if (Test-Path $out) { throw "$out already exists. Delete or rename it to calibrate again." }
if (-not (Get-Process acs -ErrorAction SilentlyContinue)) { throw 'Assetto Corsa (acs.exe) is not running.' }
New-Item -ItemType Directory -Force $dir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$ErrorActionPreference = 'Continue'
& $OcdPython -m rlsetup.assetto.calibration --full-range --host 127.0.0.1 --setup-id bcn_f1_best `
    --output $out --trace "$dir\steering-f1r5-trace_$stamp.jsonl" --execute --neutral-confirmed `
    --at-controls --speed-mps 0.2 --motion-m 0.1
exit $LASTEXITCODE
