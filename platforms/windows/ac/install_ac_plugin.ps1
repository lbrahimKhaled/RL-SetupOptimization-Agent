# Installs the assetto_corsa_gym in-game plugin for SINGLE-PC use (game and learner on this machine).
# Run after Assetto Corsa has finished installing in Steam, with AC closed. No admin needed if the
# Steam library is user-writable (the default); vJoy itself needs admin (see README).
#   - apps\python\sensors_par            <- AssettoCorsaPlugin\plugins\sensors_par
#       config.py patched: the three servers bind 127.0.0.1 instead of 0.0.0.0 (no LAN exposure,
#       no firewall prompt); vjoy_executed_by_server = True with a 100 Hz tick, the mode
#       rlsetup.assetto requires (SteeringMap.verify, Parked: 100 / 25 Hz, plugin drives vJoy).
#       This is the Lenovo setup; AC's framerate limit must then be 100 FPS.
#   - system\x64\DLLs, system\x64\Lib    <- windows-libs (socket library for AC's Python 3.3)
#   - Documents\Assetto Corsa\cfg\controllers\savedsetups\{Vjoy,WASD}.ini
# Usage: powershell -ExecutionPolicy Bypass -File platforms\windows\ac\install_ac_plugin.ps1 [-WhatIf]
[CmdletBinding(SupportsShouldProcess)]
param()
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env_ac.ps1')

$plugin = Join-Path $AcGym 'assetto_corsa_gym\AssettoCorsaPlugin\plugins\sensors_par'
$libs = Join-Path $AcGym 'assetto_corsa_gym\AssettoCorsaPlugin\windows-libs'
if (-not (Test-Path $plugin)) { throw "assetto_corsa_gym not found at $AcGym (run setup_acenv.ps1 first)" }
if (-not (Test-Path (Join-Path $AcInstall 'acs.exe'))) {
    throw "Assetto Corsa is not (fully) installed at $AcInstall (acs.exe missing). Finish the Steam download first."
}

$dst = Join-Path $AcInstall 'apps\python\sensors_par'
if ($PSCmdlet.ShouldProcess($dst, 'copy plugin')) {
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item -Recurse -Force (Join-Path $plugin '*') $dst
    $cfg = Join-Path $dst 'config.py'
    $text = [IO.File]::ReadAllText($cfg)
    $text = $text -replace '(_host_name\s*=\s*)"0\.0\.0\.0"', '$1"127.0.0.1"'
    $text = $text -replace '(self\.vjoy_executed_by_server\s*=\s*)False', '$1True'
    $text = $text -replace '(self\.sampling_freq\s*=\s*)\d+', '${1}100'
    [IO.File]::WriteAllText($cfg, $text)
    if ($text -match '"0\.0\.0\.0"' -or $text -notmatch 'self\.vjoy_executed_by_server\s*=\s*True' -or
        $text -notmatch 'self\.sampling_freq\s*=\s*100\b') { throw "could not patch $cfg" }
}

foreach ($d in 'DLLs', 'Lib') {
    $to = Join-Path $AcInstall "system\x64\$d"
    if ($PSCmdlet.ShouldProcess($to, "copy windows-libs\$d")) {
        New-Item -ItemType Directory -Force $to | Out-Null
        Copy-Item -Recurse -Force (Join-Path $libs "$d\*") $to
    }
}

$saved = Join-Path $AcDocs 'cfg\controllers\savedsetups'
if ($PSCmdlet.ShouldProcess($saved, 'copy Vjoy.ini, WASD.ini')) {
    New-Item -ItemType Directory -Force $saved | Out-Null
    Copy-Item -Force (Join-Path $libs 'Vjoy.ini'), (Join-Path $libs 'WASD.ini') $saved
}

Write-Host "Plugin installed (servers on 127.0.0.1:2345-2347, plugin drives vJoy at 100 Hz)."
Write-Host "Next: platforms\windows\ac\README.md, 'Walkthrough' (controls, 100 FPS cap, Hotlap session)."
