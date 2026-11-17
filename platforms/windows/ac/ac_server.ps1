# rlsetup.assetto learner server on THIS PC (the Mac's role in the Lenovo setup), bound to 127.0.0.1.
# Runs in ocdenv (SB3 / torch). Everything after the named parameters goes to
# `rlsetup.assetto.distributed server` unchanged: use the Mac's training settings, e.g.
#   platforms\windows\ac\ac_server.ps1 -WorkDir runs\windows\ac\server-001 -ActorBundle <source actor dir> `
#       --critic-updates C --updates U --batch-size B [--warmup-transitions W --transitions-per-update K --keep-learners N]
# Needs rlsetup\assetto\learn.py (only on the Mac so far; README "Missing from the Mac").
param(
    [Parameter(Mandatory)] [string]$WorkDir,
    [Parameter(Mandatory)] [string]$ActorBundle,
    [int]$Port = 2350,
    [Parameter(ValueFromRemainingArguments)] [string[]]$ServerArgs = @()
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
Set-Location $Repo
if (-not (Test-Path 'rlsetup\assetto\learn.py')) {
    throw 'rlsetup\assetto\learn.py is missing: it was never committed from the Mac (README "Missing from the Mac").'
}
$token = . (Join-Path $PSScriptRoot 'token.ps1')
& $OcdPython -m rlsetup.assetto.distributed server --bind 127.0.0.1 --port $Port --token-file $token `
    --work-dir $WorkDir --actor-bundle $ActorBundle @ServerArgs
exit $LASTEXITCODE
