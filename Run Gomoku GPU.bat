@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project GPU environment not found: .venv\Scripts\python.exe
    echo Ask Codex to recreate the local CUDA environment.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "Gomoku AI player V1.0.py"
if errorlevel 1 pause
