@echo off
setlocal
cd /d "%~dp0"
echo Ensuring YOLO + FastSAM weights in models\ ...
echo Step 1: check local cache / Docker image (no internet)...
powershell -ExecutionPolicy Bypass -File scripts\copy_ultralytics_weights.ps1
if not errorlevel 1 goto :ok
echo Step 2: download from HuggingFace / GitHub mirrors...
powershell -ExecutionPolicy Bypass -File scripts\download_ultralytics_weights.ps1
if errorlevel 1 exit /b 1
:ok
echo Done.
endlocal
