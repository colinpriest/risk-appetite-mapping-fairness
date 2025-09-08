@echo off
REM Launch the unified workflow manager
REM This provides an interactive interface for all experiment tasks

echo Starting Risk Fairness Experiment Workflow Manager...
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the workflow manager
python workflow_manager.py

pause