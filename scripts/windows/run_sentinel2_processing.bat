@echo off
setlocal enabledelayedexpansion

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

REM ------------------------------------------------------------------
REM Edit the values below to match your project before running.
REM ------------------------------------------------------------------
set "AOI_PATH=data\20251201_MI_NWI_Small_Test\train\aoi_train.gpkg"
set "YEARS=2021 2022 2023"
set "OUTPUT_DIR=data\20251201_MI_NWI_Small_Test\train\s2_tmp"
set "SEASONS=SPR SUM FAL"
set "NAIP_PATH=data\20251201_MI_NWI_Small_Test\train\rasters\naip.tif"
set "CLOUD_COVER=60"
set "MIN_CLEAR_OBS=3"
set "MASK_DILATION=0"
set "STAC_URL=https://earth-search.aws.element84.com/v1"
set "LOG_LEVEL=DEBUG"

REM Auto-download configuration (set true/false)
set "AUTO_DOWNLOAD_NAIP=false"
set "AUTO_DOWNLOAD_NAIP_YEAR=2022"
set "AUTO_DOWNLOAD_NAIP_MAX_ITEMS=1"
set "AUTO_DOWNLOAD_NAIP_OVERWRITE=false"
set "AUTO_DOWNLOAD_NAIP_PREVIEW=false"
set "NAIP_TARGET_RESOLUTION="

set "AUTO_DOWNLOAD_WETLANDS=false"
set "WETLANDS_OUTPUT="
set "WETLANDS_OVERWRITE=false"

set "AUTO_DOWNLOAD_TOPOGRAPHY=true"
set "TOPOGRAPHY_BUFFER_METERS="
set "TOPOGRAPHY_TPI_SMALL="
set "TOPOGRAPHY_TPI_LARGE="
set "TOPOGRAPHY_CACHE_DIR="
set "TOPOGRAPHY_DEM_DIR=data\20251201_MI_NWI_Small_Test\train\rasters\dem"

if not exist "venv312\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "venv312\Scripts\activate.bat"
set "PYTHONPATH=src"

REM --- Argument Construction ---

set "ARGS=--aoi "%AOI_PATH%" --years %YEARS% --output-dir "%OUTPUT_DIR%" --seasons %SEASONS%"
set "ARGS=%ARGS% --cloud-cover %CLOUD_COVER% --min-clear-obs %MIN_CLEAR_OBS% --mask-dilation %MASK_DILATION%"
set "ARGS=%ARGS% --stac-url "%STAC_URL%" --log-level %LOG_LEVEL%"

if not "%NAIP_PATH%"=="" set "ARGS=%ARGS% --naip-path "%NAIP_PATH%""
if not "%NAIP_TARGET_RESOLUTION%"=="" set "ARGS=%ARGS% --naip-target-resolution %NAIP_TARGET_RESOLUTION%"

if /I "%AUTO_DOWNLOAD_NAIP%"=="true" (
    set "ARGS=!ARGS! --auto-download-naip"
    if not "%AUTO_DOWNLOAD_NAIP_YEAR%"=="" set "ARGS=!ARGS! --auto-download-naip-year %AUTO_DOWNLOAD_NAIP_YEAR%"
    if not "%AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"=="" set "ARGS=!ARGS! --auto-download-naip-max-items %AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"
    if /I "%AUTO_DOWNLOAD_NAIP_OVERWRITE%"=="true" set "ARGS=!ARGS! --auto-download-naip-overwrite"
    if /I "%AUTO_DOWNLOAD_NAIP_PREVIEW%"=="true" set "ARGS=!ARGS! --auto-download-naip-preview"
)

if /I "%AUTO_DOWNLOAD_WETLANDS%"=="true" (
    set "ARGS=!ARGS! --auto-download-wetlands"
    if not "%WETLANDS_OUTPUT%"=="" set "ARGS=!ARGS! --wetlands-output-path "%WETLANDS_OUTPUT%""
    if /I "%WETLANDS_OVERWRITE%"=="true" set "ARGS=!ARGS! --wetlands-overwrite"
)

if /I "%AUTO_DOWNLOAD_TOPOGRAPHY%"=="true" (
    set "ARGS=!ARGS! --auto-download-topography"
    if not "%TOPOGRAPHY_BUFFER_METERS%"=="" set "ARGS=!ARGS! --topography-buffer-meters %TOPOGRAPHY_BUFFER_METERS%"
    if not "%TOPOGRAPHY_TPI_SMALL%"=="" set "ARGS=!ARGS! --topography-tpi-small %TOPOGRAPHY_TPI_SMALL%"
    if not "%TOPOGRAPHY_TPI_LARGE%"=="" set "ARGS=!ARGS! --topography-tpi-large %TOPOGRAPHY_TPI_LARGE%"
    if not "%TOPOGRAPHY_CACHE_DIR%"=="" set "ARGS=!ARGS! --topography-cache-dir "%TOPOGRAPHY_CACHE_DIR%""
    if not "%TOPOGRAPHY_DEM_DIR%"=="" set "ARGS=!ARGS! --topography-dem-dir "%TOPOGRAPHY_DEM_DIR%""
)

REM --- Execution ---

echo Running Sentinel-2 Processing...
python -m wetlands_ml_atwell.sentinel2.cli %ARGS%

if errorlevel 1 (
    echo [ERROR] Sentinel-2 processing failed.
    pause
    exit /b 1
)

echo [INFO] Sentinel-2 seasonal products and stack manifest generated in %OUTPUT_DIR%
pause
