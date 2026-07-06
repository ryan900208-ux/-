@echo off
cd /d "%~dp0"

set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo Starting Quality Momentum API Server...
echo API dashboard will be available at: http://localhost:8000/api/dashboard
echo Launching browser...
echo Press Ctrl+C to stop the server.

:: Wait 2 seconds and launch default browser to local server address
start /b cmd /c "timeout /t 2 >nul && start http://localhost:8000/"

python -m uvicorn api_app:app --host 0.0.0.0 --port 8000 --log-level info

pause
