@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -m venv venv
    if errorlevel 1 goto :error
    call "venv\Scripts\activate.bat"
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 goto :error
) else (
    call "venv\Scripts\activate.bat"
)
echo.
echo Starting PitchIQ EPL Predictor...
echo Open http://127.0.0.1:8000 in your browser.
echo Stop the server with Ctrl+C.
echo.
python run.py
goto :eof
:error
echo Setup failed. Check that Python is installed and available as the py command.
pause
exit /b 1
