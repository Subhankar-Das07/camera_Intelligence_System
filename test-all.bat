@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Camera Intelligence System ? Automated Test Suite
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found. Activate your virtual environment first.
  pause
  exit /b 1
)

echo [1/3] Running full test suite with coverage...
echo.
python -m pytest tests/ -v --tb=short --cov=core --cov=pipelines --cov-report=term-missing --html=tests/report.html --self-contained-html

set EXIT=%ERRORLEVEL%

echo.
echo ============================================================
if %EXIT%==0 (
  echo  ALL TESTS PASSED - Safe to deploy!
) else (
  echo  TESTS FAILED - DO NOT deploy until issues are fixed!
)
echo ============================================================
echo.
echo [2/3] HTML report saved to: tests\report.html
echo [3/3] Open report in browser? (type y to open)
choice /c yn /n /m ""
if errorlevel 2 goto done
if errorlevel 1 start tests\report.html

:done
echo.
exit /b %EXIT%
