@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Docker Up - start stack ^(no rebuild^)
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

echo [1/1] Starting containers...
docker compose up -d
if errorlevel 1 (
  echo ERROR: docker compose up failed.
  pause
  exit /b 1
)

echo.
echo Status:
docker compose ps
echo.
echo Open http://localhost:8000
echo.
pause
endlocal
