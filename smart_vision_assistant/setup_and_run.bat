@echo off
setlocal

echo ====================================================
echo    Smart Vision Assistant — Visual Scene Engine
echo ====================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

echo.
echo [2/3] Installing dependencies...
pip install opencv-python ultralytics Pillow numpy google-genai python-dotenv psutil
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check internet connection.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Smart Vision Assistant...
echo       Console reports every %REPORT_INTERVAL% seconds (default: 10s)
echo.
python main.py

pause
