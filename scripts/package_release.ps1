# Build GUI and create a versioned release zip with INSTALL.txt (Windows).
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Version = (Get-Content -Raw (Join-Path $Root "VERSION")).Trim()
if (-not $Version) {
    throw "VERSION file is empty"
}

& (Join-Path $Root "scripts\build_gui.bat")
if ($LASTEXITCODE -ne 0) {
    throw "build_gui.bat failed with exit code $LASTEXITCODE"
}

$Platform = "windows"
$Arch = $env:PROCESSOR_ARCHITECTURE
if ($Arch -eq "AMD64") { $Arch = "x86_64" }
elseif ($Arch -eq "ARM64") { $Arch = "arm64" }

$StagingRoot = Join-Path $Root "dist\release-staging"
$StagingApp = Join-Path $StagingRoot "transcode_gui"
$ZipName = "transcode-gui-v$Version-$Platform-$Arch.zip"
$ZipPath = Join-Path $Root "dist\$ZipName"

if (Test-Path $StagingRoot) {
    Remove-Item -Recurse -Force $StagingRoot
}
New-Item -ItemType Directory -Path $StagingApp | Out-Null
Copy-Item -Recurse -Force (Join-Path $Root "dist\transcode_gui\*") $StagingApp
Copy-Item -Force (Join-Path $Root "packaging\INSTALL.txt") (Join-Path $StagingRoot "INSTALL.txt")

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}
Compress-Archive -Path (Join-Path $StagingRoot "transcode_gui"), (Join-Path $StagingRoot "INSTALL.txt") -DestinationPath $ZipPath

Set-Content -Path (Join-Path $Root "dist\release-asset-name.txt") -Value $ZipName -NoNewline
Write-Host "Release zip: $ZipPath"
