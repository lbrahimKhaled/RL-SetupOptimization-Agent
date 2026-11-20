# AC learner for the F1 driver bcn_f1_r5 (7M, R5 reward) on THIS PC: game, worker and learner on one
# machine over 127.0.0.1. Same settings as the Mac's runs/ac/tcp-launch/START_MAC_F1R5.command apart
# from --bind 127.0.0.1 (no LAN exposure, no firewall prompt), the local token (token.ps1) and the
# Windows work dir. Runs until Ctrl+C; run it again (same -WorkDir) to continue from the last
# committed cycle. --critic-init only seeds a NEW learner; a resumed work dir ignores it.
#   powershell -ExecutionPolicy Bypass -File platforms\windows\ac\ac_server_f1r5.ps1
# Start this first, then ac_worker_f1r5.ps1 in a second window. Only one learner can listen on a port.
param(
    [string]$WorkDir = 'runs\windows\ac\tcp-f1r5-001',
    [string]$ActorBundle = 'runs\ac\source_actor_bcn_f1_r5_7m',
    [string]$CriticInit = 'models\bcn_f1_r5_step7000k\seed0\checkpoints\last.zip',
    [int]$Port = 2350
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
. (Join-Path $PSScriptRoot 'keep_awake.ps1')
Set-Location $Repo
if (-not (Test-Path $OcdPython)) { throw "no Python env at $OcdEnv (run platforms\windows\setup_pc.ps1 first)" }
if (-not (Test-Path (Join-Path $ActorBundle 'manifest.json'))) { throw "actor bundle missing: $ActorBundle" }
if (-not (Test-Path $CriticInit)) { throw "critic-init checkpoint missing: $CriticInit" }
$token = . (Join-Path $PSScriptRoot 'token.ps1')
Set-KeepAwake
Write-Host "learner: 127.0.0.1:$Port, work dir $WorkDir (progress: $WorkDir\progress.jsonl)"
$ErrorActionPreference = 'Continue'
& $OcdPython -m rlsetup.assetto.distributed server --bind 127.0.0.1 --port $Port --token-file $token `
    --actor-bundle $ActorBundle --work-dir $WorkDir --batch-size 256 --buffer-size 300000 `
    --warmup-transitions 5000 --critic-updates 1000 --transitions-per-update 4 --updates 0 `
    --critic-init $CriticInit --keep-learners 2 --max-no-update 20
exit $LASTEXITCODE
