# Stop a training run cleanly (resumable): set the keeper's stop flag, wait for the next checkpoint +
# replay-buffer save (replay_buffer.pkl gets a newer mtime), then stop the training process and its workers.
#
#   powershell -ExecutionPolicy Bypass -File platforms\windows\stop_run.ps1 NAME [-Now]
#
# -Now: do not wait for a save (loses the steps since the last checkpoint). Resume later with
#   platforms\windows\start_detached.ps1 -Log runs\windows\NAME_stdout.log platforms\windows\train.ps1 --resume --preset NAME --seed 0 --run-name NAME --allow-code-change --allow-dirty
# (delete <seed_dir>\STOP_KEEPER before restarting the keeper).
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
if ($args.Count -lt 1) { Write-Error 'usage: stop_run.ps1 RUN_NAME [-Now]' }
$Run = $args[0]
$Now = $args -contains '-Now'
$Root = if ($env:RUNS_ROOT) { $env:RUNS_ROOT } else { Join-Path $Repo 'runs\windows\D2' }
$Seed = Join-Path $Root "$Run\seed0"
New-Item -ItemType File -Force (Join-Path $Seed 'STOP_KEEPER') | Out-Null
Write-Output "stop flag set ($Seed\STOP_KEEPER): the keeper will not resume $Run"
$Pattern = "rlsetup\.driving\.train .*--run-name $([regex]::Escape($Run))( |`"|$)"
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $Pattern })
if (-not $procs) { Write-Output 'no training process running'; exit 0 }
if (-not $Now) {
    $buf = Join-Path $Seed 'replay_buffer.pkl'
    $t0 = if (Test-Path $buf) { (Get-Item $buf).LastWriteTime } else { [datetime]::MinValue }
    Write-Output "waiting for the next replay_buffer.pkl save (last $t0) ..."
    while ($true) {
        Start-Sleep -Seconds 15
        if ((Test-Path $buf) -and (Get-Item $buf).LastWriteTime -gt $t0) {
            $size = -1
            while ((Get-Item $buf).Length -ne $size) { $size = (Get-Item $buf).Length; Start-Sleep -Seconds 10 }
            break
        }
    }
    Write-Output "saved at $((Get-Item $buf).LastWriteTime)"
}
$ids = $procs.ProcessId
$children = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $ids -contains $_.ParentProcessId })
foreach ($p in @($procs) + $children) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Output "stopped $($procs.Count) training process(es) and $($children.Count) worker(s) of $Run"
