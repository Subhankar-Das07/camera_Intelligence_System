@echo off
echo ============================================================
echo   Smart Vision Assistant — Web Interface Launcher
echo ============================================================
echo.

:: ── Install Python dependencies ──────────────────────────────────────────────
echo [1/3] Installing Python dependencies...
pip install fastapi uvicorn[standard] python-multipart --quiet
if errorlevel 1 (
    echo ERROR: Failed to install Python packages.
    pause
    exit /b 1
)
echo       Done.

:: ── Install Node.js dependencies ─────────────────────────────────────────────
echo.
echo [2/3] Installing frontend dependencies...
cd frontend
call npm install --silent
if errorlevel 1 (
    echo ERROR: npm install failed. Is Node.js installed?
    cd ..
    pause
    exit /b 1
)
cd ..
echo       Done.

:: ── Start servers ─────────────────────────────────────────────────────────────
echo.
echo [3/3] Starting servers...
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo.
echo Press Ctrl+C in each window to stop.
echo ============================================================

:: Start FastAPI in a new window
start "Smart Vision — FastAPI Backend" cmd /k "uvicorn web_server:app --host 0.0.0.0 --port 8000"

:: Wait 3 seconds for backend to initialize
timeout /t 3 /nobreak > nul

:: Start Vite in a new window
start "Smart Vision — React Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers are starting in separate windows.
echo Open http://localhost:5173 in your browser.
pause
