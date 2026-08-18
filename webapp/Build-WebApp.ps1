# Rebuild the cardcenter wheel for the Bakugo web app and refresh its manifest.
# Run from anywhere:  pwsh Bakugo/webapp/Build-WebApp.ps1 [-Dest <portal bakugo dir>]
# Public URL is loadopoly.com/bakugo — the Python package name stays cardcenter.
param(
    [string]$Dest = (Join-Path $PSScriptRoot "..\..\Supply-Chain-Brain\Loadopoly-Portal\bakugo")
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$Dest = (New-Item -ItemType Directory -Force -Path $Dest).FullName

# Old wheels out, so the folder never serves two engine versions.
Get-ChildItem $Dest -Filter "cardcenter-*.whl" -ErrorAction SilentlyContinue | Remove-Item

Push-Location $repo
try {
    python -m pip wheel . --no-deps -w $Dest | Out-Null
} finally {
    Pop-Location
}

$wheel = Get-ChildItem $Dest -Filter "cardcenter-*.whl" | Select-Object -First 1
if (-not $wheel) { throw "wheel build produced no cardcenter-*.whl in $Dest" }
$version = ($wheel.Name -split "-")[1]

@{
    version = $version
    wheel   = $wheel.Name
    pyodide = "v314.0.5"
} | ConvertTo-Json | Set-Content (Join-Path $Dest "manifest.json") -Encoding utf8

Write-Host "built $($wheel.Name) -> $Dest"
Write-Host "commit and push the Loadopoly-Portal repo to deploy to loadopoly.com"
