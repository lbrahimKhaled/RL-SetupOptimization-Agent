# rlsetup.assetto learner server on THIS PC (the Mac's role in the Lenovo setup), bound to 127.0.0.1.
# Runs in ocdenv (SB3 / torch). Runs until Ctrl+C; run it again with the same -WorkDir to continue.
#   platforms\windows\ac\ac_server.ps1                         # Mac run tcp-full-002 settings, new work dir below
#   platforms\windows\ac\ac_server.ps1 -WorkDir <dir> <server args...>   # any other settings, passed unchanged
# Default settings = the Mac's runs/ac/tcp-launch/START_MAC_FULL.command (tcp-full-002) apart from
# --bind 127.0.0.1 (single PC) and Windows paths. Needs the files copied from the Mac (README
# "Model files from the Mac"): runs\ac\source_actor and runs\mac\D2\bcn\seed0\checkpoints\best.zip.
param(
    [string]$WorkDir = 'runs\windows\ac\server-001',
    [string]$ActorBundle = 'runs\ac\source_actor',
    [int]$Port = 2350,
    [Parameter(ValueFromRemainingArguments)] [string[]]$ServerArgs = @()
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
Set-Location $Repo
if (-not $ServerArgs) {
    $ServerArgs = @('--batch-size', '256', '--buffer-size', '300000', '--warmup-transitions', '5000',
                    '--critic-updates', '1000', '--transitions-per-update', '4', '--updates', '0',
                    '--critic-init', 'runs\mac\D2\bcn\seed0\checkpoints\best.zip', '--keep-learners', '2',
                    '--max-no-update', '20')
}
$token = . (Join-Path $PSScriptRoot 'token.ps1')
& $OcdPython -m rlsetup.assetto.distributed server --bind 127.0.0.1 --port $Port --token-file $token `
    --work-dir $WorkDir --actor-bundle $ActorBundle @ServerArgs
exit $LASTEXITCODE
