# Build-Mobile.ps1 — Build and sync the Bakugo Mobile Native App
param(
    [switch]$OpenAndroid,
    [switch]$BuildApk
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $ScriptDir
try {
    Write-Host "📦 1. Preparing mobile web bundle..." -ForegroundColor Cyan
    node build_assets.js

    if (Test-Path "node_modules") {
        Write-Host "⚡ 2. Syncing Capacitor native platform..." -ForegroundColor Cyan
        npx cap sync
    } else {
        Write-Host "💡 Capacitor dependencies not yet installed. Run 'npm install' in ./mobile first if building native APK." -ForegroundColor Yellow
    }

    if ($OpenAndroid) {
        Write-Host "🚀 3. Opening Android Studio project..." -ForegroundColor Green
        npx cap open android
    }
} finally {
    Pop-Location
}
