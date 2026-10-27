# Windows throughput benchmarks for choosing n_envs / gradient_steps / torch_threads / device at
# UTD 0.5 (README.md "Tuning"). Runs scripts\bench\*.py one after the other at low CPU priority
# (this process and its children), so a live training run keeps the CPU first.
#
#   powershell -ExecutionPolicy Bypass -File platforms\windows\bench.ps1                 # all steps
#   powershell -ExecutionPolicy Bypass -File platforms\windows\bench.ps1 -Steps learner,vec
#
# CPU priority does not apply to the GPU. While a training run is live (an rlsetup.driving.train
# process exists), the CUDA steps are skipped unless -AllowGpuWhileTraining is given: they would
# share the GPU with the run and slow both. Any number measured during a live run is contended;
# meta.json records the live training processes, so such numbers can be flagged.
#
# Output: runs\windows\bench\<yyyyMMdd-HHmmss>\ meta.json, <step>.json / .jsonl, <step>.log / .err
param(
    [string[]]$Steps = @('learner', 'vec', 'e2e'),
    [ValidateSet('Idle', 'BelowNormal', 'Normal')][string]$Priority = 'BelowNormal',
    [int[]]$CpuThreads = @(1, 2, 4, 8),
    [int[]]$CudaThreads = @(1, 2, 4),
    [int[]]$Workers = @(4, 8, 12, 16, 20, 24),
    [int[]]$E2eWorkers = @(8, 12, 16, 24),     # gradient_steps = workers / 2 (UTD 0.5)
    [int]$E2eEnvSteps = 16000,
    [int]$E2eCpuThreads = 4,
    [int]$E2eCudaThreads = 2,
    [switch]$AllowGpuWhileTraining
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Repo
(Get-Process -Id $PID).PriorityClass = $Priority      # children inherit Idle / BelowNormal

$Out = Join-Path $Repo "runs\windows\bench\$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$rel = $Out.Substring($Repo.Length + 1)

$live = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
          Where-Object { $_.CommandLine -match 'rlsetup\.driving\.train' } |
          ForEach-Object { @{ pid = $_.ProcessId; command_line = $_.CommandLine } })
$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader) -join '; '
}
$meta = [ordered]@{
    utc_start = (Get-Date).ToUniversalTime().ToString('s') + 'Z'
    computer = $env:COMPUTERNAME
    cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    logical_processors = [Environment]::ProcessorCount
    gpu_at_start = $gpu
    priority = $Priority
    live_training_processes = $live
    contended = ($live.Count -gt 0)
    python = $OcdPython
    steps = $Steps
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Out 'meta.json') -Encoding Ascii
$cudaOk = ($live.Count -eq 0) -or $AllowGpuWhileTraining
if ($live.Count -gt 0) {
    Write-Output "live training processes: $(($live | ForEach-Object { $_.pid }) -join ', '); results are contended"
    if (-not $cudaOk) { Write-Output 'CUDA steps skipped (pass -AllowGpuWhileTraining to run them anyway)' }
}

function Bench([string]$Name, [string[]]$PyArgs) {
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] $Name : python $($PyArgs -join ' ')"
    $p = Start-Process -FilePath $OcdPython -ArgumentList $PyArgs -WorkingDirectory $Repo -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $Out "$Name.log") -RedirectStandardError (Join-Path $Out "$Name.err")
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] $Name : exit $($p.ExitCode)"
    Get-Content (Join-Path $Out "$Name.log") | Where-Object { $_ -like '{*' }
}

if ($Steps -contains 'learner') {
    Bench 'learner_cpu' (@('scripts/bench/sac_train_speed.py', '--device', 'cpu', '--json', "$rel\learner_cpu.json", '--threads') + $CpuThreads)
    if ($cudaOk) {
        Bench 'learner_cuda' (@('scripts/bench/sac_train_speed.py', '--device', 'cuda', '--json', "$rel\learner_cuda.json", '--threads') + $CudaThreads)
    }
}
if ($Steps -contains 'vec') {
    Bench 'vec_spawn' (@('scripts/bench/vec_throughput.py', '--start-method', 'spawn', '--json', "$rel\vec_spawn.json", '--workers') + $Workers)
}
if ($Steps -contains 'e2e') {
    foreach ($n in $E2eWorkers) {
        $g = [int]($n / 2)
        Bench "e2e_cpu_t${E2eCpuThreads}_w$n" @('scripts/bench/sac_e2e.py', '--workers', "$n", '--grad', "$g", '--env-steps', "$E2eEnvSteps",
                                               '--device', 'cpu', '--threads', "$E2eCpuThreads", '--start-method', 'spawn')
        if ($cudaOk) {
            Bench "e2e_cuda_t${E2eCudaThreads}_w$n" @('scripts/bench/sac_e2e.py', '--workers', "$n", '--grad', "$g", '--env-steps', "$E2eEnvSteps",
                                                     '--device', 'cuda', '--threads', "$E2eCudaThreads", '--start-method', 'spawn')
        }
    }
}
Write-Output "results in $Out"
