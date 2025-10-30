@echo off
echo ================================================
echo Wetlands ML GeoAI - Environment Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Setting up virtual environment...
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...

REM Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo Installing/updating pip...
python -m pip install --upgrade pip

echo.
echo Installing required packages...
echo This may take a while, especially for PyTorch and geospatial packages...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ================================================
echo Setup completed successfully!
echo ================================================
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To start Jupyter Notebook, run:
echo   jupyter notebook
echo.
echo To run the training script:
echo   python train_unet.py
echo.
echo To run the test script:
echo   python test_unet.py
echo.
pause
