# Trains in Assetto Corsa on this PC with assetto_corsa_gym's train.py (SAC / DisCor), game and
# learner on the same machine. Start AC first (Hotlap session, Vjoy controls, 50 FPS cap) and keep
# its window in front; then run, e.g.:
#   platforms\windows\ac\train_ac.ps1
#   platforms\windows\ac\train_ac.ps1 -Extra 'AssettoCorsa.car=dallara_f317','AssettoCorsa.track=ks_barcelona-layout_gp'
# Single-PC measures:
#   - acs.exe is raised to High priority, so a learner update burst cannot drop AC below 50 FPS
#     (the physics and the 25 Hz control loop are real time). High needs no admin rights.
#   - the learner gets $TorchThreads intra-op threads (default 4 of 32), leaving cores to AC.
#   - the plugin and client talk over 127.0.0.1 (ac_local.overrides, install_ac_plugin.ps1).
param(
    [int]$TorchThreads = 4,
    [string[]]$Extra = @()
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env_ac.ps1')

$acs = Get-Process acs -ErrorAction SilentlyContinue
if (-not $acs) { throw 'Assetto Corsa (acs.exe) is not running: start a Hotlap session first.' }
foreach ($p in $acs) { try { $p.PriorityClass = 'High' } catch { Write-Warning "could not raise acs.exe priority: $_" } }

$env:OMP_NUM_THREADS = "$TorchThreads"
$env:MKL_NUM_THREADS = "$TorchThreads"
$overrides = @()
foreach ($line in Get-Content -LiteralPath (Join-Path $PSScriptRoot 'ac_local.overrides')) {
    $kv = ($line -replace '#.*$', '').Trim()
    if ($kv) { $overrides += $kv }
}
& $AcPython (Join-Path $PSScriptRoot 'check_ac.py') --live
if ($LASTEXITCODE) { throw 'preflight failed (see above)' }

Push-Location $AcGym   # train.py resolves ./assetto_corsa_gym and ./algorithm relative to the cwd
try {
    # Output: <AC_GYM>\outputs\<timestamp>\ (log.log, checkpoints). Do not pass work_dir: upstream
    # train.py reads args.work_dir / config.track there, which do not exist (ef0fa50).
    & $AcPython train.py --config config.yml @overrides @Extra
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
