# D2 full run, unattended, Windows port of scripts/run_d2_full.sh (same decisions, lock file and
# report): collect the demo dataset once, train seeds 0, 1, 2 sequentially, then
# scripts/d2_report.py. Windows PowerShell 5.1 or later.
#
#   platforms\windows\start_detached.ps1 -Log runs\windows\full1_stdout.log platforms\windows\run_d2_full.ps1
#   powershell -ExecutionPolicy Bypass -File platforms\windows\run_d2_full.ps1 --dry-run   # print the commands, run nothing
#
# Environment (defaults):
#   RUN_NAME=full1         run folder $RUNS_ROOT\$RUN_NAME\seed<k>
#   MAX_WALL_H=3           wall-clock cap per seed and per invocation, h (a resumed seed gets a
#                          fresh cap: the training code counts it from the start of the process)
#   TOTAL_STEPS=2000000    total env steps per seed
#   RUNS_ROOT              run root (default <repo>\runs\windows\D2; passed as runs_root)
#   OVERRIDES_FILE         platform overrides (default platforms\windows\train.overrides): passed
#                          as --set to collect_demos.py and to fresh starts only (a resume keeps
#                          the run's config.json)
#   PYTHON_BIN             python for the pipeline commands (default $OCD_ENV\python.exe, env.ps1)
#
# Per seed: status.json phase 'done' (completed / wall_clock_limit / collapsed) -> skip;
# state bc_gate_failed / demo_gate_failed -> skip, counted as a failure (rerunning is
# deterministic; move the folder away to retry); checkpoints\last.zip exists -> --resume;
# otherwise a fresh start (a stale folder without last.zip is first moved to
# $RUNS_ROOT\$RUN_NAME\_aborted\, never deleted). A failing seed does not stop the next one; the
# script exits 1 if any step failed. Lock: $RUNS_ROOT\$RUN_NAME\.run_d2_full.lock holds the PID;
# a lock of a dead process is taken over.
$ErrorActionPreference = 'Continue'
Set-StrictMode -Version 2
$global:LASTEXITCODE = 0
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Repo

function EnvOr([string]$Name, [string]$Default) {
    $v = [Environment]::GetEnvironmentVariable($Name)
    if ($v) { return $v } else { return $Default }
}
$Py = EnvOr 'PYTHON_BIN' $OcdPython
$RunName = EnvOr 'RUN_NAME' 'full1'
$MaxWallH = EnvOr 'MAX_WALL_H' '3'
$TotalSteps = EnvOr 'TOTAL_STEPS' '2000000'
$Root = EnvOr 'RUNS_ROOT' (Join-Path $Repo 'runs\windows\D2')
$OverridesFile = EnvOr 'OVERRIDES_FILE' (Join-Path $PSScriptRoot 'train.overrides')
$Seeds = @(0, 1, 2)

$Dry = $false
foreach ($a in $args) {
    switch ($a) {
        '--dry-run' { $Dry = $true }
        { $_ -in '-h', '--help' } { Get-Content $PSCommandPath | Select-Object -First 26; exit 0 }
        default { [Console]::Error.WriteLine("unknown argument: $a"); exit 2 }
    }
}

if (-not (Test-Path -LiteralPath $OverridesFile)) { [Console]::Error.WriteLine("missing overrides file: $OverridesFile"); exit 2 }
$PlatSet = Read-Overrides $OverridesFile
$RunDir = Join-Path $Root $RunName
$Common = @('--preset', 'full', '--run-name', $RunName, '--set', "max_wall_h=$MaxWallH",
            '--set', "total_timesteps=$TotalSteps", '--set', "runs_root=$Root")

function Log([string]$Msg) { Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] $Msg" }

# Invoke-Step CMD...: print it; execute unless --dry-run; exit code in $script:Rc
function Invoke-Step([string[]]$Cmd) {
    Log ('+ ' + ($Cmd -join ' '))
    $script:Rc = 0
    if ($Dry) { return }
    try {
        $rest = @($Cmd | Select-Object -Skip 1)
        & $Cmd[0] @rest
        $script:Rc = $LASTEXITCODE
    } catch {
        [Console]::Error.WriteLine("$($Cmd[0]): $_")
        $script:Rc = 127
    }
}

# Status-Field FILE KEY: top-level string field of a status.json ('' if missing / unreadable)
function Status-Field([string]$File, [string]$Key) {
    try {
        $j = Get-Content -LiteralPath $File -Raw | ConvertFrom-Json
        $p = $j.PSObject.Properties[$Key]
        if ($p -and $p.Value -is [string]) { return $p.Value }
    } catch { }
    return ''
}

if (-not (Get-Command $Py -ErrorAction SilentlyContinue)) { [Console]::Error.WriteLine("missing python: $Py"); exit 1 }

# one invocation per run name at a time (stale lock of a dead process is taken over)
$Lock = Join-Path $RunDir '.run_d2_full.lock'
if (-not $Dry) {
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    if (Test-Path -LiteralPath $Lock) {
        $holder = (Get-Content -LiteralPath $Lock -Raw).Trim()
        if ($holder -match '^\d+$' -and (Get-Process -Id ([int]$holder) -ErrorAction SilentlyContinue)) {
            Log "another run_d2_full (pid $holder) is running $RunName; exiting"
            exit 1
        }
    }
    Set-Content -LiteralPath $Lock -Value $PID -Encoding Ascii
}

$Failed = 0
$Summary = @()
try {
    Log "run_d2_full: repo $Repo, run $RunDir, seeds $($Seeds -join ' '), max_wall_h $MaxWallH, total_steps $TotalSteps, dry_run $([int]$Dry)"
    Log "platform overrides: $OverridesFile $($PlatSet -join ' ')"
    $head = (git -C $Repo rev-parse --short HEAD 2>$null)
    $changes = @(git -C $Repo status --porcelain --untracked-files=no 2>$null).Count
    Log "git: $head $changes tracked changes"

    # --- demos once (cached per dataset key; the seeds reuse the file) ---
    $t0 = [Diagnostics.Stopwatch]::StartNew()
    Invoke-Step (@($Py, 'scripts/collect_demos.py', '--preset', 'full') + $PlatSet)
    $DemoRc = $script:Rc
    Log "demos: exit $DemoRc ($([int]$t0.Elapsed.TotalSeconds) s)"
    if ($DemoRc -ne 0) { $Failed = 1 }
    $Summary += "  demos: exit $DemoRc"

    # --- seeds ---
    foreach ($s in $Seeds) {
        $SD = Join-Path $RunDir "seed$s"
        $t0 = [Diagnostics.Stopwatch]::StartNew()
        Log "seed ${s}: start ($SD)"
        $phase = ''; $state = ''
        $statusFile = Join-Path $SD 'status.json'
        if (Test-Path -LiteralPath $statusFile) {
            $phase = Status-Field $statusFile 'phase'
            $state = Status-Field $statusFile 'state'
        }
        $shownState = if ($state) { $state } else { '?' }
        $rc = 0
        if ($phase -eq 'done') {
            $action = "skip (finished: $state)"
            Log "seed ${s}: $action"
        } elseif ($state -eq 'bc_gate_failed' -or $state -eq 'demo_gate_failed') {
            $action = "skip ($state; move $SD away to retry)"
            $rc = 1
            Log "seed ${s}: $action"
        } elseif (Test-Path -LiteralPath (Join-Path $SD 'checkpoints\last.zip')) {
            $action = "resume (state $shownState)"
            Log "seed ${s}: $action"
            Invoke-Step (@($Py, '-m', 'rlsetup.driving.train', '--resume', '--seed', "$s") + $Common)
            $rc = $script:Rc
        } elseif ($DemoRc -ne 0) {
            $action = 'skip fresh start (demo collection failed)'
            $rc = 1
            Log "seed ${s}: $action"
        } else {
            if (Test-Path -LiteralPath $SD) {
                $dest = Join-Path $RunDir "_aborted\seed$s.$(Get-Date -Format 'yyyyMMddTHHmmss')"
                $was = if ($state) { $state } else { 'none' }
                Log "seed ${s}: no last.zip (state $was); moving the stale folder to $dest"
                Log "+ Move-Item $SD $dest"
                if (-not $Dry) {
                    New-Item -ItemType Directory -Force -Path (Join-Path $RunDir '_aborted') | Out-Null
                    Move-Item -LiteralPath $SD -Destination $dest
                }
            }
            $action = 'fresh'
            Invoke-Step (@($Py, '-m', 'rlsetup.driving.train', '--seed', "$s") + $Common + $PlatSet)
            $rc = $script:Rc
        }
        if ($rc -ne 0) { $Failed = 1 }
        $secs = [int]$t0.Elapsed.TotalSeconds
        Log "seed ${s}: end, $action, exit $rc, $secs s"
        $Summary += "  seed ${s}: $action, exit $rc, $secs s"
    }

    # --- report ---
    $t0 = [Diagnostics.Stopwatch]::StartNew()
    Invoke-Step @($Py, 'scripts/d2_report.py', $RunDir)
    $ReportRc = $script:Rc
    Log "report: exit $ReportRc ($([int]$t0.Elapsed.TotalSeconds) s)"
    if ($ReportRc -ne 0) { $Failed = 1 }
    $Summary += "  report: exit $ReportRc"

    Log "run_d2_full: summary`n$($Summary -join "`n")"
    Log "run_d2_full: done, exit $Failed"
} finally {
    if (-not $Dry) { Remove-Item -LiteralPath $Lock -Force -ErrorAction SilentlyContinue }
}
exit $Failed
