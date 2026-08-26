@echo off

setlocal EnableDelayedExpansion

cd /d "%~dp0"



echo ========================================

echo  Docker Rebuild - build with layer cache

echo ========================================

echo.

echo Use when requirements.txt or Dockerfile changed.

echo See docs\DOCKER_DEV.md for which bat to use.

echo.



where docker >nul 2>&1

if errorlevel 1 (

  echo ERROR: Docker is not installed or not on PATH.

  echo Install Docker Desktop and try again.

  pause

  exit /b 1

)



echo [0/3] Ensuring model weights in models\ ...

set "MISSING=0"

for %%F in (yolov8n.pt yolov8n-pose.pt FastSAM-s.pt) do (

  set "FOUND=0"

  if exist "models\%%F" (

    for %%A in ("models\%%F") do if %%~zA GEQ 100000 set "FOUND=1"

  )

  if "!FOUND!"=="0" set "MISSING=1"

)

if "%MISSING%"=="1" (

  echo Weights missing or incomplete — downloading on host ^(not inside Docker^)...

  powershell -ExecutionPolicy Bypass -File scripts\download_ultralytics_weights.ps1

  if errorlevel 1 (

    echo ERROR: Could not download weights. Check internet / VPN / firewall.

    echo Or run download-weights.bat manually, then retry.

    pause

    exit /b 1

  )

)



set "MISSING=0"

for %%F in (yolov8n.pt yolov8n-pose.pt FastSAM-s.pt) do (

  set "FOUND=0"

  if exist "models\%%F" (

    for %%A in ("models\%%F") do if %%~zA GEQ 100000 set "FOUND=1"

  )

  if "!FOUND!"=="0" set "MISSING=1"

)

if "%MISSING%"=="1" (

  echo ERROR: models\ still missing valid .pt files after download.

  pause

  exit /b 1

)

echo Weights OK in models\



echo [1/3] Building images ^(no --pull, uses cache^)...

docker compose build

if errorlevel 1 (

  echo ERROR: docker compose build failed.

  pause

  exit /b 1

)



echo [2/3] Starting containers...

docker compose up -d

if errorlevel 1 (

  echo ERROR: docker compose up failed.

  pause

  exit /b 1

)



echo [3/3] Status:

docker compose ps

echo.

echo Open http://localhost:8000

echo.

pause

endlocal

