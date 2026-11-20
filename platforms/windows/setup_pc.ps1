# One-time Python setup of a NEW Windows PC that runs everything itself: Assetto Corsa fine-tuning
# (game, worker and learner on 127.0.0.1) and OCD training (rlsetup.driving.train). No admin needed.
#   powershell -ExecutionPolicy Bypass -File platforms\windows\setup_pc.ps1 [-Torch auto|cuda|cpu] [-Python C:\...\python.exe]
#
# What it does (README_FIRST.md at the top of the bundle explains the whole PC setup):
#   1. Checks the side-by-side layout: ..\ocd_first_lap (OCD_FIRST_LAP) and ..\assetto_corsa_gym.
#   2. Creates a venv at $HOME\ocdenv (OCD_ENV) from a per-user Python 3.12 (python.org installer,
#      "Install for current user"; found with `py -3.12` or in %LOCALAPPDATA%\Programs\Python\Python312).
#      A venv, not conda: one installer, no admin, and env.ps1 / every launcher find Scripts\python.exe.
#      An existing env at that path (conda or venv) is reused as it is.
#   3. torch 2.11.0: the CUDA 12.8 build if `nvidia-smi -L` lists a GPU, else the CPU build (-Torch forces one).
#   4. ocd_first_lap\requirements.txt; trajectory-planning-helpers from ocd_first_lap\vendor (same commit
#      aa950f6 as that file pins, so git is not needed); then requirements-win.txt (SB3 2.9.0, ...).
#   5. ocd_first_lap.pth in the venv, so `import ocd_lap` also works outside the launchers.
#   6. core.autocrlf false in each git checkout (CRLF changes every source hash).
#   7. check_setup.py: imports and files (without ocd_native if its .pyd is not there yet).
# Needs internet (pip). Safe to run again: pip skips what is already installed.
param(
    [ValidateSet('auto', 'cuda', 'cpu')] [string]$Torch = 'auto',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Repo
$TorchVersion = '2.11.0'

function Run([string]$Exe, [string[]]$ArgList) {
    Write-Host ">> $Exe $($ArgList -join ' ')"
    $ErrorActionPreference = 'Continue'
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $Exe $($ArgList -join ' ')" }
}

function PyVersion([string]$Exe) {
    # 312 for Python 3.12; 0 if $Exe does not run (e.g. the Microsoft Store python.exe alias)
    $ErrorActionPreference = 'Continue'
    try { $v = & $Exe -c 'import sys; print(sys.version_info[0] * 100 + sys.version_info[1])' 2>$null } catch { return 0 }
    if ($LASTEXITCODE -ne 0 -or -not $v) { return 0 }
    return [int]($v | Select-Object -Last 1)
}

# ---- 1. layout
Write-Host "repo          $Repo"
Write-Host "ocd_first_lap $OcdFirstLap"
Write-Host "python env    $OcdEnv"
if (-not (Test-Path (Join-Path $OcdFirstLap 'ocd_lap\__init__.py'))) {
    throw "ocd_first_lap not found at $OcdFirstLap (unzip it next to this repo, or set OCD_FIRST_LAP)"
}
$OcdFirstLap = (Resolve-Path $OcdFirstLap).Path
$AcGym = Join-Path (Split-Path $Repo -Parent) 'assetto_corsa_gym'
if (-not (Test-Path (Join-Path $AcGym 'assetto_corsa_gym'))) {
    Write-Warning "assetto_corsa_gym not found at $AcGym (install_ac_plugin.ps1 needs it)"
}

# ---- 2. Python 3.12 venv
if (Test-Path $OcdPython) {
    Write-Host "reusing the existing env: $OcdPython"
} else {
    if (-not $Python) {
        if (Get-Command py.exe -ErrorAction SilentlyContinue) {
            $ErrorActionPreference = 'Continue'
            $found = & py.exe -3.12 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $Python = ($found | Select-Object -Last 1).Trim() }
            $ErrorActionPreference = 'Stop'
        }
    }
    if (-not $Python) {
        $guess = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
        if (Test-Path $guess) { $Python = $guess }
    }
    if (-not $Python -or (PyVersion $Python) -ne 312) {
        throw ('No Python 3.12 found. Install it from https://www.python.org/downloads/windows/ ' +
               '(64-bit, "Install for current user", py launcher on), or pass -Python <path to python.exe>.')
    }
    Write-Host "base Python 3.12: $Python"
    Run $Python @('-m', 'venv', $OcdEnv)
    $OcdPython = Join-Path $OcdEnv 'Scripts\python.exe'
}
if ((PyVersion $OcdPython) -ne 312) { throw "$OcdPython is not Python 3.12" }
Run $OcdPython @('-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', 'pip')

# ---- 3. torch (CUDA if an NVIDIA GPU answers, else CPU)
if ($Torch -eq 'auto') {
    $Torch = 'cpu'
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smi) {
        $ErrorActionPreference = 'Continue'
        $gpus = & $smi.Source -L 2>$null
        if ($LASTEXITCODE -eq 0 -and ($gpus -match 'GPU')) { $Torch = 'cuda' }
        $ErrorActionPreference = 'Stop'
    }
    Write-Host "torch build: $Torch (nvidia-smi $(if ($smi) { 'found' } else { 'not found' }))"
}
$tag = if ($Torch -eq 'cuda') { 'cu128' } else { 'cpu' }
Run $OcdPython @('-m', 'pip', 'install', '--disable-pip-version-check', "torch==$TorchVersion+$tag",
                 '--index-url', "https://download.pytorch.org/whl/$tag")

# ---- 4. ocd_first_lap requirements, then this repo's
$req = Join-Path $OcdFirstLap 'requirements.txt'
$tph = Join-Path $OcdFirstLap 'vendor\trajectory_planning_helpers'
if (Test-Path (Join-Path $tph 'setup.py')) {
    $tmp = Join-Path $env:TEMP 'ocd_first_lap_requirements_nogit.txt'
    Get-Content $req | Where-Object { $_ -notmatch '^\s*trajectory-planning-helpers' } | Set-Content -Encoding Ascii $tmp
    Run $OcdPython @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $tmp)
    Run $OcdPython @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps', $tph)
} else {
    Write-Warning "no $tph, so installing trajectory-planning-helpers from GitHub (needs git on PATH)"
    Run $OcdPython @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $req)
}
Run $OcdPython @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', (Join-Path $PSScriptRoot 'requirements-win.txt'))

# ---- 5. ocd_first_lap on the env's path (env.ps1 also sets PYTHONPATH for the launchers)
$ErrorActionPreference = 'Continue'
$site = (& $OcdPython -c 'import sysconfig; print(sysconfig.get_path(''purelib''))' | Select-Object -Last 1).Trim()
$ErrorActionPreference = 'Stop'
[IO.File]::WriteAllText((Join-Path $site 'ocd_first_lap.pth'), "$OcdFirstLap`r`n")
Write-Host "wrote $site\ocd_first_lap.pth -> $OcdFirstLap"

# ---- 6. no CRLF conversion in the git checkouts
if (Get-Command git.exe -ErrorAction SilentlyContinue) {
    foreach ($d in $Repo, $OcdFirstLap, $AcGym) {
        if (Test-Path (Join-Path $d '.git')) { Run 'git.exe' @('-C', $d, 'config', 'core.autocrlf', 'false') }
    }
}

# ---- 7. checks
$pyd = @(Get-ChildItem -Path $OcdFirstLap -Filter 'ocd_native*.pyd' -ErrorAction SilentlyContinue)
$checkArgs = @()
if ($pyd.Count -eq 0) {
    Write-Warning ("ocd_native*.pyd is not in $OcdFirstLap yet. AC fine-tuning does not need it; OCD training " +
                   'does. Copy ocd_native.cp312-win_amd64.pyd from the i9 PC (README_FIRST.md), then run this again.')
    $checkArgs = @('--no-ocd')
}
Run $OcdPython (@((Join-Path $PSScriptRoot 'check_setup.py')) + $checkArgs)
Write-Host ''
Write-Host "Done. Python env: $OcdPython (torch $TorchVersion+$tag)."
Write-Host 'Next: README_FIRST.md (AC plugin, calibration, learner + worker; OCD training).'
