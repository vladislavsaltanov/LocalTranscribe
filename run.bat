@echo off
chcp 65001 >nul
set PYTHONUTF8=1
if "%~1"=="" (
    echo Usage: run.bat video.mp4 [-o out.txt] [--lang en] [--speakers N]
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe_interview.py" %*
