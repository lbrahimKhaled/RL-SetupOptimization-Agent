# Keeps Windows from sleeping while the calling PowerShell process runs (the Windows counterpart of
# the Mac's `caffeinate`). With -Display it also keeps the screen on, so an idle display timeout
# cannot turn it off or lock the PC under AC. It does NOT stop a screen saver (Microsoft docs for
# SetThreadExecutionState); switch that off in Windows settings. Released when the process exits.
# Dot-source this file, then call:  Set-KeepAwake [-Display]
function Set-KeepAwake([switch]$Display) {
    try {
        if (-not ('RlSetup.Power' -as [type])) {
            Add-Type -Namespace RlSetup -Name Power -MemberDefinition `
                '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
        }
        $flags = [uint32]2147483649                      # ES_CONTINUOUS 0x80000000 | ES_SYSTEM_REQUIRED 0x1
        if ($Display) { $flags = [uint32]2147483651 }    # ... | ES_DISPLAY_REQUIRED 0x2
        [void][RlSetup.Power]::SetThreadExecutionState($flags)
    } catch {
        Write-Warning "could not keep the PC awake ($_); set Power > Sleep and Screen to Never instead"
    }
}
