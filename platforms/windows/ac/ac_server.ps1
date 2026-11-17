# rlsetup.assetto learner server on THIS PC (the Mac's role in the Lenovo setup), bound to 127.0.0.1.
# Runs in ocdenv (SB3 / torch). Everything after the named parameters goes to
# `rlsetup.assetto.distributed server` unchanged. The Mac run tcp-full-002 (docs/AC_QUICKSTART.md,
# "Full training") used the ratio schedule below; --updates / --batch-size are in the Mac's
# runs/ac/tcp-launch/START_MAC_FULL.command (not committed):
#   platforms\windows\ac\ac_server.ps1 -WorkDir runs\windows\ac\server-001 -ActorBundle runs\ac\source_actor `
#       --critic-updates 1000 --updates U --batch-size B --warmup-transitions 5000 --transitions-per-update 4 `
#       --keep-learners 2 --critic-init runs\mac\D2\bcn\seed0\checkpoints\best.zip
param(
    [Parameter(Mandatory)] [string]$WorkDir,
    [Parameter(Mandatory)] [string]$ActorBundle,
    [int]$Port = 2350,
    [Parameter(ValueFromRemainingArguments)] [string[]]$ServerArgs = @()
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env.ps1')
Set-Location $Repo
$token = . (Join-Path $PSScriptRoot 'token.ps1')
& $OcdPython -m rlsetup.assetto.distributed server --bind 127.0.0.1 --port $Port --token-file $token `
    --work-dir $WorkDir --actor-bundle $ActorBundle @ServerArgs
exit $LASTEXITCODE
