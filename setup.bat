@echo off
echo Installing Python requirements...

where python >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3 and ensure it is in your PATH.
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Done!
pause