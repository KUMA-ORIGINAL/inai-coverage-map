@echo off
rem One command for Windows: creates .venv, installs requirements, runs main.py.
rem Usage:  run.bat [data_folder] [--open] [other main.py options]
setlocal
set "ROOT=%~dp0"
set "VPY=%ROOT%.venv\Scripts\python.exe"

if not exist "%VPY%" (
    echo [setup] Creating virtual environment .venv ...
    py -3 -m venv "%ROOT%.venv" 2>nul || python -m venv "%ROOT%.venv"
    if not exist "%VPY%" (
        echo Python 3.10+ not found. Install it from https://www.python.org and run again.
        exit /b 3
    )
)

"%VPY%" -m pip install -q --disable-pip-version-check -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo Failed to install requirements.
    exit /b 1
)

"%VPY%" "%ROOT%main.py" %*
exit /b %errorlevel%
