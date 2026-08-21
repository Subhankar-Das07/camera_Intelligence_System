@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Docker Refresh - rebuild and redeploy
echo ========================================
echo.

where docker >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker is not installed or not on PATH.
  echo Install Docker Desktop and try again.
  pause
  exit /b 1
)

echo [1/4] Stopping existing containers...
docker compose down
if errorlevel 1 (
  echo WARNING: docker compose down reported an error. Continuing...
)

echo [2/4] Building images from current project files...
docker compose build --pull
if errorlevel 1 (
  echo ERROR: docker compose build failed.
  pause
  exit /b 1
)

echo [3/4] Starting containers...
docker compose up -d --force-recreate --remove-orphans
if errorlevel 1 (
  echo ERROR: docker compose up failed.
  pause
  exit /b 1
)

echo [4/4] Status:
docker compose ps
echo.
echo Open http://localhost:8000
echo Redis DB volume: redis_data
echo.
pause
endlocal
