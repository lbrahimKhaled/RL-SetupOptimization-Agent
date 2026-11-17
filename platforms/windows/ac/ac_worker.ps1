# rlsetup.assetto AC worker on THIS PC: START_LENOVO_FULL.cmd (v11) with the learner on 127.0.0.1.
# Start AC first (SF70H, Barcelona GP, Hotlap, Vjoy controls, 100 FPS cap) and ac_server.ps1 in a
# second window. Runs until Ctrl+C; run it again to continue from the last committed actor.
#   platforms\windows\ac\ac_worker.ps1 -WorkDir runs\windows\ac\worker-001
param(
    [Parameter(Mandatory)] [string]$WorkDir,
    [int]$Port = 2350,
    [string]$Calibration = 'runs\windows\ac\steering.json',
    [Parameter(ValueFromRemainingArguments)] [string[]]$Extra = @()
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
Set-Location $Repo
if (-not (Test-Path $Calibration)) { throw "no steering calibration at $Calibration (README walkthrough step 6)" }
$acs = Get-Process acs -ErrorAction SilentlyContinue
if (-not $acs) { throw 'Assetto Corsa (acs.exe) is not running.' }
# One PC: keep AC's 100 Hz physics/vJoy tick ahead of the learner's update bursts (High needs no admin).
foreach ($p in $acs) { try { $p.PriorityClass = 'High' } catch { Write-Warning "could not raise acs.exe priority: $_" } }
$token = . (Join-Path $PSScriptRoot 'token.ps1')
# Same settings as START_LENOVO_FULL.cmd (v11); override any of them through -Extra.
& $OcdPython -m rlsetup.assetto.distributed worker --server 127.0.0.1 --port $Port --token-file $token `
    --calibration $Calibration --work-dir $WorkDir --ac-host 127.0.0.1 `
    --reference ocd_path --reference-file data\ac\ocd_path_reference_barcelona.json `
    --max-steps 3750 --max-wall-s 180 --explore-std-scale 0.3 --eval-every 10 `
    --reconnect-attempts 20 --max-consecutive-failures 5 --execute @Extra
exit $LASTEXITCODE
