<#
.SYNOPSIS
  Rebuild and restart Bakugo, then verify the running version.

.DESCRIPTION
  docker compose up -d --build, wait for the health endpoint, and compare the
  version it reports with the one in pyproject.toml. Exits non-zero if the
  running version is not the built one, so a half-finished rebuild does not
  look like a success.

  Run from anywhere:  pwsh -File deploy\Rebuild.ps1
  Skip the tunnel check:  pwsh -File deploy\Rebuild.ps1 -SkipTunnel

  WHICH STACK: the server that is actually running may belong to a parent
  compose project (on the hideout PC it is `bakugo-app` in the hideout stack,
  which builds ./Bakugo). Running this repo's compose file there collides with
  that stack's container names and deploys nothing. So the script finds the
  running bakugo:dev container, reads which compose file and service own it,
  and rebuilds that one service in that project. Override with -ComposeFile
  and -Service; with nothing running it falls back to this repo's compose.
#>
[CmdletBinding()]
param(
    [string]$LocalUrl = 'http://127.0.0.1:8765',
    [string]$TunnelUrl = 'https://bakugo.loadopoly.com',
    [switch]$SkipTunnel,
    [int]$TimeoutSeconds = 180,
    [string]$ComposeFile = '',
    [string]$Service = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$expected = (Select-String -Path (Join-Path $repo 'pyproject.toml') -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
Write-Host "Building cardcenter $expected in $repo" -ForegroundColor Cyan

if (-not $ComposeFile) {
    $ids = @(& docker ps -q)
    if ($ids.Count -gt 0) {
        # Assign before filtering: Windows PowerShell 5.1's ConvertFrom-Json
        # emits a JSON array as ONE object, so piping it straight into
        # Where-Object would test the whole array at once.
        $all = @(& docker inspect $ids | Out-String | ConvertFrom-Json)
        $owner = $all | ForEach-Object { $_ } |
            Where-Object { $_.Config.Image -eq 'bakugo:dev' -and $_.Config.Labels.'com.docker.compose.project.config_files' } |
            Select-Object -First 1
        if ($owner) {
            $ComposeFile = ($owner.Config.Labels.'com.docker.compose.project.config_files' -split ',')[0]
            if (-not $Service) { $Service = $owner.Config.Labels.'com.docker.compose.service' }
            Write-Host "Live server: $($owner.Name.TrimStart('/')) = service '$Service' in $ComposeFile" -ForegroundColor Cyan
        }
    }
}
if (-not $ComposeFile) {
    $ComposeFile = Join-Path $repo 'docker-compose.yml'
    Write-Host "No running bakugo:dev container; using $ComposeFile" -ForegroundColor Cyan
}
if (-not $Service) { $Service = 'bakugo' }
$composeDir = Split-Path -Parent $ComposeFile

Push-Location $composeDir
try {
    $ErrorActionPreference = 'Continue'   # docker writes progress to stderr
    & docker compose -f $ComposeFile build $Service 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "docker compose build failed ($LASTEXITCODE)" }
    # --no-deps: restart only this service, not the rest of a shared stack.
    & docker compose -f $ComposeFile up -d --no-deps $Service 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed ($LASTEXITCODE)" }
    $ErrorActionPreference = 'Stop'
} finally {
    Pop-Location
}

function Get-Version([string]$url) {
    try {
        $r = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 5
        return $r.version
    } catch {
        return $null
    }
}

Write-Host "Waiting for $LocalUrl/health ..." -NoNewline
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$running = $null
while ((Get-Date) -lt $deadline) {
    $running = Get-Version $LocalUrl
    if ($running) { break }
    Write-Host '.' -NoNewline
    Start-Sleep -Seconds 3
}
Write-Host ''

if (-not $running) {
    Write-Host "No answer from $LocalUrl/health. Container logs:" -ForegroundColor Red
    Push-Location $composeDir
    try { & docker compose -f $ComposeFile logs --tail 40 $Service } finally { Pop-Location }
    exit 1
}

if ($running -ne $expected) {
    Write-Host "Running version is $running, expected $expected." -ForegroundColor Red
    Write-Host "The image did not rebuild. Try: docker compose -f `"$ComposeFile`" build --no-cache $Service" -ForegroundColor Yellow
    exit 1
}
Write-Host "Local OK: $running" -ForegroundColor Green

if (-not $SkipTunnel) {
    $viaTunnel = Get-Version $TunnelUrl
    if ($viaTunnel -eq $expected) {
        Write-Host "Tunnel OK: $TunnelUrl serving $viaTunnel" -ForegroundColor Green
    } elseif ($viaTunnel) {
        Write-Host "Tunnel still serving $viaTunnel (cloudflared may need a moment)." -ForegroundColor Yellow
    } else {
        Write-Host "Tunnel did not answer. Is cloudflared running?" -ForegroundColor Yellow
    }
}

Write-Host ''
Write-Host "Hard-reload the page on your phone; the pill should read v$expected." -ForegroundColor Cyan
