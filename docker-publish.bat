@echo off
setlocal
cd /d "%~dp0"

set IMAGE=drpinfotech/camera-intelligence
set TAG_VERSION=0.1.0
set TAG_BRANCH=develop

echo ========================================
echo  Docker Publish - %IMAGE%
echo  Tags: %TAG_VERSION% and %TAG_BRANCH%
echo ========================================
echo.

where docker >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker is not on PATH.
  pause
  exit /b 1
)

echo [1/4] Building image...
docker build -t %IMAGE%:%TAG_VERSION% -t %IMAGE%:%TAG_BRANCH% .
if errorlevel 1 (
  echo ERROR: docker build failed.
  pause
  exit /b 1
)

echo [2/4] Checking Docker Hub login (drpinfotech)...
docker info >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker daemon not running.
  pause
  exit /b 1
)

echo If push fails with auth error, run: docker login
echo.

echo [3/4] Pushing %IMAGE%:%TAG_VERSION% ...
docker push %IMAGE%:%TAG_VERSION%
if errorlevel 1 (
  echo ERROR: push %TAG_VERSION% failed. Run "docker login" as drpinfotech and retry.
  pause
  exit /b 1
)

echo [4/4] Pushing %IMAGE%:%TAG_BRANCH% ...
docker push %IMAGE%:%TAG_BRANCH%
if errorlevel 1 (
  echo ERROR: push %TAG_BRANCH% failed.
  pause
  exit /b 1
)

echo.
echo Published:
echo   https://hub.docker.com/r/drpinfotech/camera-intelligence
echo   %IMAGE%:%TAG_VERSION%
echo   %IMAGE%:%TAG_BRANCH%
echo.
echo Teammates can run:
echo   docker pull %IMAGE%:%TAG_BRANCH%
echo   docker compose -f docker-compose.yml -f docker-compose.hub.yml up -d
echo.
pause
endlocal
