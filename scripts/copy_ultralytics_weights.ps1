# Copy weights from local Ultralytics cache, Docker image, or partial build cache.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

$MinBytes = 100000
$Files = @("yolov8n.pt", "yolov8n-pose.pt", "FastSAM-s.pt")

function Test-WeightOk($Path) {
    return (Test-Path $Path) -and ((Get-Item $Path).Length -ge $MinBytes)
}

function Copy-IfOk($Src, $Dest, $Label) {
    if (-not (Test-Path $Src)) { return $false }
    $Len = (Get-Item $Src).Length
    if ($Len -lt $MinBytes) { return $false }
    Copy-Item -Force $Src $Dest
    Write-Host "Copied $Label -> $Dest ($Len bytes)"
    return $true
}

Write-Host "Looking for existing weights..."
$ErrorActionPreference = "SilentlyContinue"
docker rm -f wt_extract 2>$null | Out-Null
$ErrorActionPreference = "Stop"

# 1) Ultralytics default cache (Windows / Linux)
$CacheRoots = @(
    (Join-Path $env:USERPROFILE ".ultralytics\weights"),
    (Join-Path $env:USERPROFILE ".cache\ultralytics"),
    "/app/.ultralytics/weights"
)
foreach ($Name in $Files) {
    $Dest = Join-Path $Models $Name
    if (Test-WeightOk $Dest) { continue }
    foreach ($Cache in $CacheRoots) {
        if (Copy-IfOk (Join-Path $Cache $Name) $Dest "cache $Cache") { break }
    }
    if (-not (Test-WeightOk $Dest)) {
        $Cwd = Join-Path $Root $Name
        if (Test-Path $Cwd) { Copy-IfOk $Cwd $Dest "project root" | Out-Null }
    }
}

# 2) Docker image (if app was built before)
$ImageNames = @(
    "drpinfotech/camera-intelligence:develop",
    "drpinfotech/camera-intelligence:latest",
    "drp_computer_vision-app",
    "drp-computer_vision-app"
)
foreach ($Name in $Files) {
    $Dest = Join-Path $Models $Name
    if (Test-WeightOk $Dest) { continue }
    foreach ($Img in $ImageNames) {
        $Id = docker image ls -q $Img 2>$null
        if (-not $Id) { continue }
        Write-Host "Trying docker image $Img for $Name ..."
        $Tmp = Join-Path $env:TEMP "wt-$Name"
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        docker rm wt_extract 2>$null | Out-Null
        docker create --name wt_extract $Img 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            docker cp "wt_extract:/app/$Name" $Tmp 2>$null
        }
        docker rm wt_extract 2>$null | Out-Null
        $ErrorActionPreference = $prevEap
        if (Test-Path $Tmp) {
            Copy-IfOk $Tmp $Dest "docker $Img" | Out-Null
            Remove-Item $Tmp -Force -ErrorAction SilentlyContinue
        }
        if (Test-WeightOk $Dest) { break }
    }
}

$Missing = @()
foreach ($Name in $Files) {
    $Dest = Join-Path $Models $Name
    if (Test-WeightOk $Dest) {
        Write-Host "OK: $Name ($((Get-Item $Dest).Length) bytes)"
    } else {
        $Missing += $Name
    }
}

if ($Missing.Count -eq 0) {
    Write-Host ""
    Write-Host "All weights ready in models\"
    exit 0
}

Write-Host ""
Write-Host "Still missing: $($Missing -join ', ')"
Write-Host "Run download-weights.bat (uses HuggingFace + GitHub mirrors) or download manually in a browser."
exit 1
