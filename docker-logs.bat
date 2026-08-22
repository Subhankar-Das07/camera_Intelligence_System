@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Docker Logs - tail app ^(Ctrl+C to stop^)
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

docker compose logs -f --tail=200 app
echo.
pause
endlocal
