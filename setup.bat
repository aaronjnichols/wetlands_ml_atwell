@echo off
echo ================================================
echo Wetlands ML Atwell - Environment Setup
echo ================================================
echo.

REM Check if Python 3.12 is available (preferred for geospatial package compatibility)
py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.12
    echo Python 3.12 found. Using for best package compatibility.
) else (
    REM Fall back to default python
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.10+ and try again
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
    echo Python found. Setting up virtual environment...
)
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv312" (
    echo Creating virtual environment 'venv312'...
    %PYTHON_CMD% -m venv venv312
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment 'venv312' already exists.
)

echo.
echo Activating virtual environment...

REM Activate virtual environment
call venv312\Scripts\activate.bat

echo.
echo Installing/updating pip...
python -m pip install --upgrade pip

echo.
echo Installing package in editable mode with all dependencies...
echo This may take a while, especially for PyTorch and geospatial packages...
python -m pip install -e ".[all]"

if %errorlevel% neq 0 (
    echo ERROR: Failed to install package
    echo.
    echo Trying without optional dependencies...
    python -m pip install -e .
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install base package
        pause
        exit /b 1
    )
    echo.
    echo Base package installed. Optional dependencies skipped.
)

echo.
echo ================================================
echo Setup completed successfully!
echo ================================================
echo.
echo To activate the environment in the future, run:
echo   venv312\Scripts\activate.bat
echo.
echo Available commands after activation:
echo   wetlands-train    - Train a UNet model
echo   wetlands-infer    - Run inference on raster data
echo   wetlands-sentinel2 - Process Sentinel-2 composites
echo.
echo Or use the module interface:
echo   python -m wetlands_ml_atwell.train_unet --help
echo   python -m wetlands_ml_atwell.test_unet --help
echo   python -m wetlands_ml_atwell.sentinel2.cli --help
echo.
echo To run tests:
echo   pytest tests/unit/
echo.
pause
