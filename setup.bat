@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem GigaAM longform extra pins torch 2.10.*, and plain PyPI gives a CPU-only torch on Windows.
rem So GigaAM goes first and a CUDA build of the 2.10 line is installed last.
rem Override the CUDA build with: set CUDA_TAG=cu126
if "%CUDA_TAG%"=="" set CUDA_TAG=cu128

where git >nul 2>nul
if errorlevel 1 (
    echo [!] git not found - install it, it is needed to clone GigaAM
    exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 echo [!] ffmpeg not found in PATH - install it before running transcription

if not exist .venv (
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip

if not exist GigaAM git clone https://github.com/salute-developers/GigaAM.git
pip install -e ".\GigaAM[torch,longform]"
if errorlevel 1 exit /b 1
pip install -r requirements.txt

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo Installing CUDA build of torch 2.10 from the %CUDA_TAG% index...
    pip install "torch==2.10.*" "torchaudio==2.10.*" --index-url https://download.pytorch.org/whl/%CUDA_TAG% --force-reinstall --no-deps
    if errorlevel 1 exit /b 1
)

echo.
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo If CUDA available is False, check the driver with nvidia-smi and try another CUDA_TAG.
echo Next: set HF_TOKEN once with  setx HF_TOKEN hf_xxx  and reopen the terminal.
