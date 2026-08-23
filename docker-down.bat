@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Docker Down - stop project containers
echo ========================================
echo.
echo See docs\DOCKER_DEV.md for which bat to use.
echo.

where docker >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker is not installed or not on PATH.
  echo Install Docker Desktop and try again.
  pause
  exit /b 1
)

echo [1/1] Stopping containers...
docker compose down
if errorlevel 1 (
  echo ERROR: docker compose down failed.
  pause
  exit /b 1
)

echo.
echo Containers stopped. Volumes ^(redis_data, app_storage, face_data^) kept.
echo.
pause
endlocal
