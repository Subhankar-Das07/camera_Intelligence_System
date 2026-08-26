# Download YOLO + FastSAM + optional OSNet weights into models/ (cache first, then mirrors).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

$MinBytes = 100000
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

# Try copy from cache / old Docker image first (no internet)
$ErrorActionPreference = "Continue"
& (Join-Path $Root "scripts\copy_ultralytics_weights.ps1")
$copyExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($copyExit -eq 0) { exit 0 }

$WeightSources = @{
    "yolov8n.pt" = @(
        "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
    )
    "yolov8n-pose.pt" = @(
        "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n-pose.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt"
    )
    "FastSAM-s.pt" = @(
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt",
        "https://huggingface.co/Ultralytics/FastSAM/resolve/main/FastSAM-s.pt"
    )
    "osnet_x0_25.onnx" = @(
        "https://huggingface.co/onnxmodelzoo/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx"
    )
}
$OptionalWeights = @("osnet_x0_25.onnx")

function Test-WeightOk($Path) {
    if (-not (Test-Path $Path)) { return $false }
    $min = if ($Path -like "*.onnx") { 10000 } else { $MinBytes }
    return ((Get-Item $Path).Length -ge $min)
}

function Download-Url($Url, $Out) {
    $Tmp = "$Out.part"
    if (Test-Path $Tmp) { Remove-Item $Tmp -Force -ErrorAction SilentlyContinue }
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & curl.exe -fL --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 900 -o $Tmp $Url
        if ($LASTEXITCODE -eq 0 -and (Test-Path $Tmp)) {
            Move-Item -Force $Tmp $Out
            return $true
        }
    }
    Invoke-WebRequest -Uri $Url -OutFile $Tmp -UseBasicParsing -TimeoutSec 900
    Move-Item -Force $Tmp $Out
    return $true
}

foreach ($Name in $WeightSources.Keys) {
    $Out = Join-Path $Models $Name
    if (Test-WeightOk $Out) {
        Write-Host "Skip (exists): $Name ($((Get-Item $Out).Length) bytes)"
        continue
    }
    $Ok = $false
    foreach ($Url in $WeightSources[$Name]) {
        try {
            Write-Host "Downloading $Name from $Url ..."
            Download-Url $Url $Out
            if (Test-WeightOk $Out) {
                Write-Host "Saved $Out ($((Get-Item $Out).Length) bytes)"
                $Ok = $true
                break
            }
            Remove-Item $Out -Force -ErrorAction SilentlyContinue
            Write-Host "File too small, trying next mirror..."
        } catch {
            Write-Host "Mirror failed: $_"
        }
    }
    if (-not $Ok) {
        if ($OptionalWeights -contains $Name) {
            Write-Host "Optional weight missing (OK): $Name — OpenCV Re-ID fallback will be used"
            continue
        }
        Write-Host ""
        Write-Host "FAILED: $Name - all mirrors unreachable."
        Write-Host "Open in browser: https://huggingface.co/Ultralytics/YOLOv8/tree/main"
        Write-Host "Save files into the models folder, then run docker-rebuild.bat"
        exit 1
    }
}

Write-Host ""
Write-Host "All weights ready in models/"
Write-Host "Run docker-rebuild.bat next."
