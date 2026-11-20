# AC worker for the F1 driver bcn_f1_r5 on THIS PC: the Lenovo's START_LENOVO_F1R5.cmd with the learner
# on 127.0.0.1, this PC's own steering calibration and the local token (token.ps1).
# Before: AC on track (Ferrari SF70H, Barcelona GP, Hotlap, setup bcn_f1_best loaded, Vjoy controls,
# 100 FPS cap, sensors_par on), ac_calibrate_f1r5.ps1 done once, ac_server_f1r5.ps1 running in another
# window. Runs until Ctrl+C; run it again (same -WorkDir) to continue from the last committed actor.
#   powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_worker_f1r5.ps1
#   AC_INSTALL  Assetto Corsa folder (default: D:\SteamLibrary\... if present, else Steam's default);
#               the teleport app lives in its apps\lua\rl_teleport
param(
    [string]$WorkDir = 'runs\windows\ac\tcp-worker-f1r5-001',
    [int]$Port = 2350,
    [string]$Calibration = 'runs\windows\ac\steering-f1r5.json',
    [string]$Reference = 'runs\ac\ocd_line_reference_bcn_f1_r5_7m.json',
    [string]$TeleportDir = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
. (Join-Path $PSScriptRoot 'keep_awake.ps1')
Set-Location $Repo
# AC_INSTALL, else the Lenovo's D:\SteamLibrary install if present, else Steam's default library.
$AcInstall = $env:AC_INSTALL
if (-not $AcInstall) {
    $AcInstall = 'D:\SteamLibrary\steamapps\common\assettocorsa'
    if (-not (Test-Path $AcInstall)) { $AcInstall = 'C:\Program Files (x86)\Steam\steamapps\common\assettocorsa' }
}
if (-not $TeleportDir) { $TeleportDir = Join-Path $AcInstall 'apps\lua\rl_teleport' }
if (-not (Test-Path $OcdPython)) { throw "no Python env at $OcdEnv (run platforms\windows\setup_pc.ps1 first)" }
if (-not (Test-Path $Calibration)) { throw "no steering calibration at $Calibration (run ac_calibrate_f1r5.ps1 first)" }
if (-not (Test-Path $Reference)) { throw "reference line missing: $Reference" }
if (-not (Test-Path (Join-Path $TeleportDir 'rl_teleport.lua'))) {
    throw "CSP teleport app not found in $TeleportDir (run install_ac_plugin.ps1, or set AC_INSTALL)"
}
$acs = Get-Process acs -ErrorAction SilentlyContinue
if (-not $acs) { throw 'Assetto Corsa (acs.exe) is not running.' }
# One PC: keep AC's 100 Hz physics/vJoy tick ahead of the learner's update bursts (High needs no admin).
foreach ($p in $acs) { try { $p.PriorityClass = 'High' } catch { Write-Warning "could not raise acs.exe priority: $_" } }
$token = . (Join-Path $PSScriptRoot 'token.ps1')
Set-KeepAwake -Display
$ErrorActionPreference = 'Continue'
& $OcdPython -m rlsetup.assetto.distributed worker --server 127.0.0.1 --port $Port --token-file $token `
    --calibration $Calibration --setup-id bcn_f1_best --work-dir $WorkDir --ac-host 127.0.0.1 `
    --reference ocd_path --reference-file $Reference --max-steps 3750 --max-wall-s 180 `
    --explore-std-scale 0.3 --eval-every 10 --reconnect-attempts 20 --max-consecutive-failures 5 `
    --teleport-dir $TeleportDir --execute
exit $LASTEXITCODE
