@echo off
setlocal
title World Cup Forecast Launcher

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
set "BACKEND_URL=http://127.0.0.1:8000"
set "FRONTEND_URL=http://127.0.0.1:5173"

cd /d "%ROOT%"

echo ============================================
echo  World Cup Forecast one-click launcher
echo ============================================
echo.

if not exist "%PYTHON_EXE%" (
  echo [1/5] Creating Python virtual environment...
  python -m venv .venv
  if errorlevel 1 goto fail
) else (
  echo [1/5] Python virtual environment found.
)

echo [2/5] Installing backend dependencies...
call "%PYTHON_EXE%" -m pip install -e ".[dev]"
if errorlevel 1 goto fail

echo [3/5] Installing frontend dependencies...
if not exist "apps\web\node_modules" (
  pushd "apps\web"
  call npm install
  if errorlevel 1 goto fail_pop
  popd
) else (
  echo Frontend dependencies found.
)

echo [4/5] Starting backend and frontend...
start "World Cup Forecast Backend" /D "%ROOT%" cmd /k call "%PYTHON_EXE%" -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
start "World Cup Forecast Frontend" /D "%ROOT%apps\web" cmd /k call npm run dev -- --host 0.0.0.0 --port 5173

echo [5/5] Waiting for services, then opening browser...
ping 127.0.0.1 -n 9 >nul
start "" "%FRONTEND_URL%"

echo.
echo Frontend: %FRONTEND_URL%
echo Backend:  %BACKEND_URL%
echo API docs: %BACKEND_URL%/docs
echo.
echo Services are running in separate windows. Close those windows to stop them.
pause
exit /b 0

:fail_pop
popd

:fail
echo.
echo Startup failed. Please check the error above.
pause
exit /b 1
