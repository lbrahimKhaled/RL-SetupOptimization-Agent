# python -m rlsetup.driving.train with the Windows defaults (platforms\README.md).
#
#   platforms\windows\train.ps1 --preset full --seed 0 --run-name NAME [--set k=v ...]
#   platforms\windows\train.ps1 --resume --preset full --seed 0 --run-name NAME
#
# Fresh run: --set runs_root=$env:RUNS_ROOT, then every line of train.overrides, then the caller's
# arguments (a later --set wins). Resume: only runs_root (config.json keeps the rest; train.py
# refuses other overrides on resume). RUNS_ROOT defaults to <repo>\runs\windows\D2.
# Runs in the foreground; for a long run use start_detached.ps1 (README.md).
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Repo
$Py = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { $OcdPython }
$Root = if ($env:RUNS_ROOT) { $env:RUNS_ROOT } else { Join-Path $Repo 'runs\windows\D2' }
$Sets = @('--set', "runs_root=$Root")
if ($args -notcontains '--resume') { $Sets += Read-Overrides (Join-Path $PSScriptRoot 'train.overrides') }
$ErrorActionPreference = 'Continue'
& $Py -m rlsetup.driving.train @Sets @args
exit $LASTEXITCODE
