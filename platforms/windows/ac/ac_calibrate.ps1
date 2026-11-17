# Measures THIS PC's steering map (vJoy command -> SF70H road-wheel angle), ~10 s, car stationary.
# The Lenovo map (data\ac\steering_sf70h_barcelona_lenovo.json) depends on that PC's vJoy and AC
# control settings, so measure again here. Preconditions (calibration.py asks you to confirm them):
#   AC running SF70H / Barcelona GP Hotlap, car stopped in the pits or on track, gear N visible,
#   you at the keyboard to stop AC if the car moves.
# Output: runs\windows\ac\steering.json (never overwritten; delete or rename to measure again).
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
Set-Location $Repo
$dir = 'runs\windows\ac'
New-Item -ItemType Directory -Force $dir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
& $OcdPython -m rlsetup.assetto.calibration --host 127.0.0.1 --full-range --execute --neutral-confirmed `
    --at-controls --output "$dir\steering.json" --trace "$dir\steering_trace_$stamp.jsonl"
exit $LASTEXITCODE
