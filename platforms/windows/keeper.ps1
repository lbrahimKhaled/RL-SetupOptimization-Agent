# Windows keeper for one training run (port of the Mac keeper_f1.sh logic, without the auto-stop):
#   every 3 min: mirror evals.jsonl into TensorBoard (scripts/evals_to_tb.py);
#   if the training process is gone: resume it from last.zip (detached, at most -MaxResumes times),
#   unless the run ended itself (status.json stop_reason set and state not 'error': completed, collapsed,
#   plateau, wall-clock limit, demo / BC gate) or the stop flag <seed_dir>\STOP_KEEPER exists.
# Never stops a run by itself: stop it with stop_run.ps1 (flag + wait for a fresh checkpoint save + kill).
#
#   platforms\windows\start_detached.ps1 -Log runs\windows\keeper_NAME.log platforms\windows\keeper.ps1 NAME [MAX_RESUMES] [PRIORITY]
#
# PRIORITY (e.g. BelowNormal, so a real-time job such as Assetto Corsa keeps the CPU): every 30 s the keeper sets
# this priority class on the training process and all its descendants (env workers start at different times).
#
# Run name = preset name (the resume passes --preset NAME). RUNS_ROOT as train.ps1 (default runs\windows\D2).
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Repo
if ($args.Count -lt 1) { Write-Error 'usage: keeper.ps1 RUN_NAME [MAX_RESUMES]'; exit 2 }
$Run = $args[0]
$MaxResumes = if ($args.Count -ge 2) { [int]$args[1] } else { 5 }
$Priority = if ($args.Count -ge 3) { [string]$args[2] } else { '' }
$Root = if ($env:RUNS_ROOT) { $env:RUNS_ROOT } else { Join-Path $Repo 'runs\windows\D2' }
$Seed = Join-Path $Root "$Run\seed0"
$Flag = Join-Path $Seed 'STOP_KEEPER'
$Pattern = "rlsetup\.driving\.train .*--run-name $([regex]::Escape($Run))( |`"|$)"
$resumes = 0

function Log([string]$m) { Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
function Set-RunPriority {
    if (-not $Priority) { return }
    $all = @(Get-CimInstance Win32_Process)
    $ids = @($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match $Pattern } | ForEach-Object { $_.ProcessId })
    $tree = @($ids)
    while ($ids.Count) {
        $ids = @($all | Where-Object { $ids -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId })
        $tree += $ids
    }
    foreach ($id in $tree) {
        try { $p = Get-Process -Id $id -ErrorAction Stop; if ("$($p.PriorityClass)" -ne $Priority) { $p.PriorityClass = $Priority } } catch {}
    }
}
function Training-Alive {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $Pattern }).Count -gt 0
}

Log "keeper for $Run ($Seed), max resumes $MaxResumes, priority '$Priority'"
while ($true) {
    if (Test-Path $Flag) { Log "stop flag $Flag present: keeper exits"; exit 0 }
    Set-RunPriority
    if (Test-Path (Join-Path $Seed 'evals.jsonl')) {
        & $OcdPython scripts\evals_to_tb.py $Seed 2>&1 | Out-Null
    }
    if (-not (Training-Alive)) {
        Start-Sleep -Seconds 20                   # a resume / launch may be starting up
        if (Training-Alive) { continue }
        if (Test-Path $Flag) { Log "stop flag present: keeper exits"; exit 0 }
        $st = $null
        try { $st = Get-Content (Join-Path $Seed 'status.json') -Raw | ConvertFrom-Json } catch {}
        if ($st -and $st.stop_reason -and $st.state -ne 'error') {
            Log "run ended itself (state $($st.state), stop_reason $($st.stop_reason)): keeper exits"; exit 0
        }
        if (-not (Test-Path (Join-Path $Seed 'checkpoints\last.zip'))) {
            Log 'training not running and no checkpoints\last.zip to resume from: keeper exits'; exit 1
        }
        if ($resumes -ge $MaxResumes) { Log "training died again; $MaxResumes resumes used: keeper exits"; exit 1 }
        $resumes++
        Log "training not running (state $($st.state)): resume $resumes of $MaxResumes"
        & (Join-Path $PSScriptRoot 'start_detached.ps1') -Log "runs\windows\$($Run)_stdout_resume$resumes.log" `
            (Join-Path $PSScriptRoot 'train.ps1') --resume --preset $Run --seed 0 --run-name $Run --allow-code-change --allow-dirty
        for ($i = 0; $i -lt 4; $i++) { Start-Sleep -Seconds 30; Set-RunPriority }
        continue
    }
    for ($i = 0; $i -lt 6; $i++) { Set-RunPriority; Start-Sleep -Seconds 30 }
}
