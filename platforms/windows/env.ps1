# Windows environment for this repo; dot-sourced by the launchers in this folder.
# Sets $Repo, $OcdEnv, $OcdFirstLap, $OcdPython and puts the Python env on PATH and ocd_first_lap on
# PYTHONPATH (this process and its children only).
#   OCD_ENV        conda env or venv (default $HOME\ocdenv). Conda layout: python.exe in the env folder
#                  (i9 PC). venv layout: Scripts\python.exe (setup_pc.ps1 on a new PC).
#   OCD_FIRST_LAP  ocd_first_lap checkout (default: next to this repo, ..\ocd_first_lap)
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$OcdEnv = if ($env:OCD_ENV) { $env:OCD_ENV } else { Join-Path $HOME 'ocdenv' }
$OcdFirstLap = if ($env:OCD_FIRST_LAP) { $env:OCD_FIRST_LAP } else { Join-Path (Split-Path $Repo -Parent) 'ocd_first_lap' }
$OcdPython = Join-Path $OcdEnv 'python.exe'
if (-not (Test-Path $OcdPython) -and (Test-Path (Join-Path $OcdEnv 'Scripts\python.exe'))) {
    $OcdPython = Join-Path $OcdEnv 'Scripts\python.exe'
}
$env:PATH = "$OcdEnv;$(Join-Path $OcdEnv 'Library\bin');$(Join-Path $OcdEnv 'Scripts');$env:PATH"
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$OcdFirstLap;$env:PYTHONPATH" } else { $env:PYTHONPATH = $OcdFirstLap }
$env:PYTHONUNBUFFERED = '1'

# Platform overrides file -> @('--set', 'k=v', ...) ('#' comments and blank lines skipped)
function Read-Overrides([string]$Path) {
    $out = @()
    foreach ($line in Get-Content -LiteralPath $Path) {
        $kv = ($line -replace '#.*$', '').Trim()
        if ($kv) { $out += @('--set', $kv) }
    }
    return ,$out
}
