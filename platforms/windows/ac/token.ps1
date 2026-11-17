# Returns the path of this PC's shared TCP token (runs\windows\ac\tcp-session.key; *.key is git-ignored),
# creating a random 64-hex-character one if missing. Server and worker on this PC read the same file.
# The Lenovo's tcp-session.key belongs to the Mac pairing; it is not reused here.
$dir = Join-Path $Repo 'runs\windows\ac'
$key = Join-Path $dir 'tcp-session.key'
if (-not (Test-Path $key)) {
    New-Item -ItemType Directory -Force $dir | Out-Null
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    [IO.File]::WriteAllText($key, (($bytes | ForEach-Object { $_.ToString('x2') }) -join ''))
}
$key
