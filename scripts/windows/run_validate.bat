@echo off
setlocal

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

set "SEASON_RASTER=data\small_test\s2\s2_spr_median_7band.tif"
set "SEASON_LABEL=SPR"
set "YEARS=2022 2023"
set "OUTPUT_CSV=data\small_test\validation\spr_corner_observations.csv"
set "PIXELS=lower_left lower_right upper_left upper_right"
set "STAC_URL=https://earth-search.aws.element84.com/v1"
set "CLOUD_COVER=60"
set "LOG_LEVEL=INFO"

if "%SEASON_RASTER%"=="" (
    echo [ERROR] Set SEASON_RASTER to a seasonal median GeoTIFF path.
    pause
    exit /b 1
)

if not exist "%SEASON_RASTER%" (
    echo [ERROR] Seasonal raster not found: %SEASON_RASTER%
    pause
    exit /b 1
)

if "%SEASON_LABEL%"=="" (
    echo [ERROR] Set SEASON_LABEL, for example SPR.
    pause
    exit /b 1
)

if "%YEARS%"=="" (
    echo [ERROR] YEARS must list the composite years, such as 2022 2023.
    pause
    exit /b 1
)

if not exist "venv312\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "venv312\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
set "PYTHONPATH=src"

python -m wetlands_ml_atwell.validate_seasonal_pixels ^
    --season-raster "%SEASON_RASTER%" ^
    --season-label %SEASON_LABEL% ^
    --years %YEARS% ^
    --output-csv "%OUTPUT_CSV%" ^
    --pixels %PIXELS% ^
    --stac-url "%STAC_URL%" ^
    --cloud-cover %CLOUD_COVER% ^
    --log-level %LOG_LEVEL%

if errorlevel 1 (
    echo [ERROR] Validation failed.
    pause
    exit /b 1
)

echo [INFO] Validation CSV written to %OUTPUT_CSV%
pause
exit /b 0
