@echo off
REM Full Fairness Experiment Runner for Windows
REM This script runs the comprehensive fairness analysis

echo ============================================================
echo FULL FAIRNESS EXPERIMENT
echo ============================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Load environment variables
if exist ".env" (
    echo Loading environment variables from .env...
    python -c "from dotenv import load_dotenv; load_dotenv()"
)

REM Show experiment configuration
echo.
echo Experiment Configuration:
echo -------------------------
type full_experiment_config.yaml | findstr /R "^K: ^repeats: ^models: ^max_cost:"
echo.

REM Prompt for confirmation
set /p confirm="Do you want to start the full experiment? (yes/no): "
if /i not "%confirm%"=="yes" (
    echo Experiment cancelled.
    pause
    exit /b 0
)

REM Run the experiment
echo.
echo Starting experiment...
echo This will take several hours. You can safely interrupt with Ctrl+C.
echo Partial results will be saved and can be resumed.
echo.

python run_full_experiment.py --threads 5

echo.
echo ============================================================
echo EXPERIMENT COMPLETE
echo ============================================================
pause