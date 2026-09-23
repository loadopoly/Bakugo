<#
.SYNOPSIS
  Run one phase of the Bakugo private trainer in Docker.

.DESCRIPTION
  Phases follow the QUIPU circadian split:
    day   -> sync (network) then day (offline): ingest, split, pseudo-label, queue
    night -> night (offline): fit candidates, evaluate against the champion
    dawn  -> dawn (offline): one promotion decision per candidate

  Schedule with Task Scheduler, e.g.:
    schtasks /Create /TN "Bakugo trainer day"   /SC DAILY /ST 12:00 /TR "pwsh -File \"%CD%\trainer\Run-Trainer.ps1\" -Phase day"
    schtasks /Create /TN "Bakugo trainer night" /SC DAILY /ST 01:00 /TR "pwsh -File \"%CD%\trainer\Run-Trainer.ps1\" -Phase night"
    schtasks /Create /TN "Bakugo trainer dawn"  /SC DAILY /ST 05:30 /TR "pwsh -File \"%CD%\trainer\Run-Trainer.ps1\" -Phase dawn"

  Required user environment variables (setx), none of them stored in the repo:
    BAKUGO_PRIVATE_ROOT_HOST   e.g. %LOCALAPPDATA%\Bakugo\private
    BAKUGO_RCLONE_CONFIG_HOST  e.g. %LOCALAPPDATA%\Bakugo\rclone.conf
    BAKUGO_DRIVE_FOLDER        the Drive folder path inside the remote
    BAKUGO_RCLONE_REMOTE       optional, default bakugo-drive
    BAKUGO_VAULT_DIR_HOST      folder holding supabase_vault.duckdb (outside the repo)
    BAKUGO_VAULT_NAME          optional, default supabase_vault.duckdb
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('day', 'night', 'dawn', 'status')][string]$Phase
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Assert-OutsideRepo([string]$name) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if (-not $value) { throw "$name is not set" }
    if (-not (Test-Path $value)) { throw "$name points at '$value', which does not exist" }
    $full = (Resolve-Path $value).Path.TrimEnd('\') + '\'
    if ($full.StartsWith($repo.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "$name ($full) is inside the repository ($repo). Move it outside."
    }
    if ($env:OneDrive -and $full.StartsWith($env:OneDrive.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "$name ($full) is inside OneDrive; private data and tokens must not sync."
    }
}

Assert-OutsideRepo 'BAKUGO_PRIVATE_ROOT_HOST'
# The folder holding supabase_vault.duckdb. The trainer's `trainer` schema in it
# holds private collection data, and Docker needs the folder (DuckDB writes a
# .wal beside the file), so it must not be the repository.
Assert-OutsideRepo 'BAKUGO_VAULT_DIR_HOST'
$compose = @('compose', '-f', (Join-Path $repo 'docker-compose.yml'), '--profile', 'trainer')
$log = Join-Path $env:BAKUGO_PRIVATE_ROOT_HOST 'logs'
New-Item -ItemType Directory -Force -Path $log | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

function Invoke-Trainer([string[]]$runArgs) {
    # docker writes progress to stderr; in Windows PowerShell 5.1 that becomes a
    # terminating error under Stop, so rely on the exit code instead.
    $ErrorActionPreference = 'Continue'
    & docker @compose run --rm @runArgs 2>&1 | Tee-Object -FilePath (Join-Path $log "run-$Phase-$stamp.txt") -Append
    if ($LASTEXITCODE -ne 0) { throw "docker compose run $($runArgs -join ' ') failed ($LASTEXITCODE)" }
}

switch ($Phase) {
    'day' {
        Assert-OutsideRepo 'BAKUGO_RCLONE_CONFIG_HOST'
        if (-not $env:BAKUGO_DRIVE_FOLDER) { throw 'BAKUGO_DRIVE_FOLDER is not set' }
        Invoke-Trainer @('trainer-sync')
        Invoke-Trainer @('trainer', 'day')
    }
    default { Invoke-Trainer @('trainer', $Phase) }
}
