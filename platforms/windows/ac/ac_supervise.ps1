# Unattended AC training (weekends): runs ac_supervisor.py, which keeps the worker going and restarts
# Assetto Corsa, reloads the setup and clicks Drive whenever the worker stops (e.g. AC lost vJoy after a
# domain policy refresh). Works with the PC locked. Defaults: the R6b pert4 run (learner on 2352).
# Before: the learner is running; AC was set up once by hand (Content Manager left race.ini on SF70H,
# Barcelona GP, Hotlap; the setup is saved as setups\ks_ferrari_sf70h\ks_barcelona\<Setup>.ini).
# A worker already writing to -WorkDir is adopted, not restarted. Stop: close this window (or kill the
# supervisor's python, pid in runs\windows\ac\logs\supervisor.pid); the worker keeps running.
#   powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_supervise.ps1
param(
    [string]$Setup = 'r6b',
    [string]$SetupId = 'bcn_r6b_pert4',
    [int]$Port = 2352,
    [string]$WorkDir = 'runs\windows\ac\tcp-worker-r6b-pert4-001',
    [string]$Progress = 'runs\windows\ac\tcp-r6b-pert4-001\progress.jsonl',
    [string]$Calibration = 'runs\windows\ac\steering-r6b-pert4.json',
    [string]$Reference = 'models\bcn_r6b_pert4_ac\barcelona_reference.json',
    [string]$TeleportDir = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
Set-Location $Repo
if (-not $TeleportDir) {
    $ac = if ($env:AC_INSTALL) { $env:AC_INSTALL } else { 'C:\Program Files (x86)\Steam\steamapps\common\assettocorsa' }
    $TeleportDir = Join-Path $ac 'apps\lua\rl_teleport'
}
& $OcdPython (Join-Path $PSScriptRoot 'ac_supervisor.py') --setup $Setup --port $Port --work-dir $WorkDir `
    --progress $Progress -- -Port $Port -WorkDir $WorkDir -SetupId $SetupId -Calibration $Calibration `
    -Reference $Reference -TeleportDir $TeleportDir
exit $LASTEXITCODE
