@echo off
setlocal
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

echo [1/2] Building images ^(no --pull, uses cache^)...
docker compose build
if errorlevel 1 (
  echo ERROR: docker compose build failed.
  pause
  exit /b 1
)

echo [2/2] Starting containers...
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
