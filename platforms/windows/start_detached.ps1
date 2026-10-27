# Start a long-running PowerShell script hidden and detached, with stdout / stderr in files, so it
# survives the terminal (or tool call) that started it. A process started with `nohup ... &` from
# Git Bash dies with its parent; this one does not.
#
#   platforms\windows\start_detached.ps1 -Log runs\windows\full1_stdout.log platforms\windows\run_d2_full.ps1
#   platforms\windows\start_detached.ps1 -Log runs\windows\x.log platforms\windows\train.ps1 --resume --preset full --seed 0 --run-name x
#
# -Log is relative to the repo root unless absolute. stdout goes to -Log and stderr to -Log + '.err'
# (Start-Process cannot send both to one file). Prints the PID of the hidden powershell.exe.
# Stop: Stop-Process -Id <pid>, then its python children:
#   Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? CommandLine -match 'rlsetup.driving.train' | % { Stop-Process -Id $_.ProcessId }
# (no param block, so --flags for the script pass through $args untouched)
$ErrorActionPreference = 'Stop'
if ($args.Count -lt 3 -or $args[0] -ne '-Log') { Write-Error 'usage: start_detached.ps1 -Log FILE SCRIPT.ps1 [ARGS...]' }
$Log, $Script = $args[1], $args[2]
$ScriptArgs = @($args | Select-Object -Skip 3)
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Repo
$LogPath = [IO.Path]::GetFullPath((Join-Path $Repo $Log))
if ([IO.Path]::IsPathRooted($Log)) { $LogPath = [IO.Path]::GetFullPath($Log) }
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null
$ScriptPath = (Resolve-Path $Script).Path
function Quote([string]$s) { if ($s -match '[\s"]') { '"' + ($s -replace '"', '\"') + '"' } else { $s } }
$argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Quote $ScriptPath)) + @($ScriptArgs | ForEach-Object { Quote $_ })
$p = Start-Process -FilePath 'powershell.exe' -ArgumentList $argList -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.err" -PassThru
Write-Output "started pid $($p.Id): $ScriptPath $($ScriptArgs -join ' ')"
Write-Output "stdout $LogPath, stderr $LogPath.err"
