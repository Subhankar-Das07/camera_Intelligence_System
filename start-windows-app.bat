@echo off
echo =======================================================
echo Starting Camera Intelligence System
echo =======================================================

:: Start Redis in a hidden background window so it doesn't clutter
echo [1/2] Starting local Redis Database...
start /b "" ".\Redis\redis-server.exe" >nul 2>&1

:: Start the Python server
echo [2/2] Starting Python Server...
set REDIS_URL=redis://localhost:6379/0
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
