# One-time setup of the Assetto Corsa Python side (no admin needed):
#   1. assetto_corsa_gym checkout next to this repo, at the pinned commit
#   2. conda env acenv = clone of ocdenv (same torch / SB3 / rlsetup stack) + requirements-ac.txt
#   3. check_ac.py (imports, vJoy DLL, AC install, plugin) -- prints what is still missing
# Usage: powershell -ExecutionPolicy Bypass -File platforms\windows\ac\setup_acenv.ps1
param([string]$OcdEnv = (Join-Path $HOME 'ocdenv'))
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env_ac.ps1')
$AcGymCommit = 'ef0fa50'   # assetto_corsa_gym master, 2026-11-17

if (-not (Test-Path (Join-Path $AcGym '.git'))) {
    git clone https://github.com/dasGringuen/assetto_corsa_gym.git $AcGym
}
$head = (git -C $AcGym rev-parse --short HEAD).Trim()
if ($head -ne $AcGymCommit) { Write-Warning "assetto_corsa_gym is at $head, this setup was tested at $AcGymCommit" }

$conda = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
if (-not $conda) { $conda = 'C:\ProgramData\miniconda3\Scripts\conda.exe' }
if (-not (Test-Path $AcPython)) {
    & $conda create -y -p $AcEnv --clone $OcdEnv
    if ($LASTEXITCODE) { throw "conda clone of $OcdEnv failed" }
}
& $AcPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements-ac.txt')
if ($LASTEXITCODE) { throw 'pip install failed' }
& $AcPython (Join-Path $PSScriptRoot 'check_ac.py')
