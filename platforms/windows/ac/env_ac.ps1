# Assetto Corsa environment for this repo; dot-sourced by the scripts in this folder.
# Sets $Repo, $AcEnv, $AcPython, $AcGym, $AcInstall, $AcDocs and puts acenv on PATH and
# assetto_corsa_gym (+ its package folder) and this repo on PYTHONPATH (this process and children).
#   AC_ENV      conda env for the AC client / learner (default $HOME\acenv)
#   AC_GYM      assetto_corsa_gym checkout (default: next to this repo, ..\assetto_corsa_gym)
#   AC_INSTALL  Assetto Corsa install folder (default: Steam's default library)
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$AcEnv = if ($env:AC_ENV) { $env:AC_ENV } else { Join-Path $HOME 'acenv' }
$AcGym = if ($env:AC_GYM) { $env:AC_GYM } else { Join-Path (Split-Path $Repo -Parent) 'assetto_corsa_gym' }
$AcInstall = if ($env:AC_INSTALL) { $env:AC_INSTALL } else { 'C:\Program Files (x86)\Steam\steamapps\common\assettocorsa' }
$AcDocs = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Assetto Corsa'
$AcPython = Join-Path $AcEnv 'python.exe'
$env:PATH = "$AcEnv;$(Join-Path $AcEnv 'Library\bin');$(Join-Path $AcEnv 'Scripts');$env:PATH"
$pp = @($AcGym, (Join-Path $AcGym 'assetto_corsa_gym'), $Repo) -join ';'
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$pp;$env:PYTHONPATH" } else { $env:PYTHONPATH = $pp }
$env:PYTHONUNBUFFERED = '1'
