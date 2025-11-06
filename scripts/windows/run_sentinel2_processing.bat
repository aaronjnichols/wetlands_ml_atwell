@echo off
setlocal enabledelayedexpansion

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

REM ------------------------------------------------------------------
REM Edit the values below to match your project before running.
REM ------------------------------------------------------------------
set "AOI_PATH=data\00_small_model_train_test\s2_test_topo\test_aoi.gpkg"
set "YEARS=2021 2022 2023"
set "OUTPUT_DIR=data\00_small_model_train_test\s2_test_topo"
set "SEASONS=SPR SUM FAL"
set "NAIP_PATH=data\00_small_model_train_test\s2_test_topo\m_4308404_nw_16_060_20220810.tif"
set "CLOUD_COVER=60"
set "MIN_CLEAR_OBS=3"
set "MASK_DILATION=0"
set "STAC_URL=https://earth-search.aws.element84.com/v1"
set "LOG_LEVEL=DEBUG"

REM Auto-download configuration (set false to disable)
set "AUTO_DOWNLOAD_NAIP=false"
set "AUTO_DOWNLOAD_NAIP_YEAR=2022"
set "AUTO_DOWNLOAD_NAIP_MAX_ITEMS=1"
set "AUTO_DOWNLOAD_NAIP_OVERWRITE=false"
set "AUTO_DOWNLOAD_NAIP_PREVIEW=false"
set "AUTO_DOWNLOAD_WETLANDS=false"
set "WETLANDS_OUTPUT="
set "WETLANDS_OVERWRITE=false"
set "NAIP_TARGET_RESOLUTION="
set "AUTO_DOWNLOAD_TOPOGRAPHY=true"
set "TOPOGRAPHY_BUFFER_METERS="
set "TOPOGRAPHY_TPI_SMALL="
set "TOPOGRAPHY_TPI_LARGE="
set "TOPOGRAPHY_CACHE_DIR="
set "TOPOGRAPHY_DEM_DIR=data\00_small_model_train_test\s2_test_topo\dem"

if not exist "venv312\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "venv312\Scripts\activate.bat"

set "NAIP_ARG="
if not "%NAIP_PATH%"=="" set "NAIP_ARG=--naip-path "%NAIP_PATH%""

set "AUTO_NAIP_FLAG="
set "AUTO_NAIP_YEAR_ARG="
set "AUTO_NAIP_MAX_ITEMS_ARG="
set "AUTO_NAIP_OVERWRITE_FLAG="
set "AUTO_NAIP_PREVIEW_FLAG="
if /I "%AUTO_DOWNLOAD_NAIP%"=="true" (
    set "AUTO_NAIP_FLAG=--auto-download-naip"
    if not "%AUTO_DOWNLOAD_NAIP_YEAR%"=="" set "AUTO_NAIP_YEAR_ARG=--auto-download-naip-year %AUTO_DOWNLOAD_NAIP_YEAR%"
    if not "%AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"=="" set "AUTO_NAIP_MAX_ITEMS_ARG=--auto-download-naip-max-items %AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"
    if /I "%AUTO_DOWNLOAD_NAIP_OVERWRITE%"=="true" set "AUTO_NAIP_OVERWRITE_FLAG=--auto-download-naip-overwrite"
    if /I "%AUTO_DOWNLOAD_NAIP_PREVIEW%"=="true" set "AUTO_NAIP_PREVIEW_FLAG=--auto-download-naip-preview"
)

set "AUTO_WETLANDS_FLAG="
set "WETLANDS_OUTPUT_ARG="
set "WETLANDS_OVERWRITE_FLAG="
if /I "%AUTO_DOWNLOAD_WETLANDS%"=="true" (
    set "AUTO_WETLANDS_FLAG=--auto-download-wetlands"
    if not "%WETLANDS_OUTPUT%"=="" set "WETLANDS_OUTPUT_ARG=--wetlands-output-path "%WETLANDS_OUTPUT%""
    if /I "%WETLANDS_OVERWRITE%"=="true" set "WETLANDS_OVERWRITE_FLAG=--wetlands-overwrite"
)

set "NAIP_RESOLUTION_ARG="
if not "%NAIP_TARGET_RESOLUTION%"=="" set "NAIP_RESOLUTION_ARG=--naip-target-resolution %NAIP_TARGET_RESOLUTION%"

set "AUTO_TOPO_FLAG="
set "TOPO_BUFFER_ARG="
set "TOPO_SMALL_ARG="
set "TOPO_LARGE_ARG="
set "TOPO_CACHE_ARG="
set "TOPO_DEM_ARG="
if /I "%AUTO_DOWNLOAD_TOPOGRAPHY%"=="true" (
    set "AUTO_TOPO_FLAG=--auto-download-topography"
    if not "%TOPOGRAPHY_BUFFER_METERS%"=="" set "TOPO_BUFFER_ARG=--topography-buffer-meters %TOPOGRAPHY_BUFFER_METERS%"
    if not "%TOPOGRAPHY_TPI_SMALL%"=="" set "TOPO_SMALL_ARG=--topography-tpi-small %TOPOGRAPHY_TPI_SMALL%"
    if not "%TOPOGRAPHY_TPI_LARGE%"=="" set "TOPO_LARGE_ARG=--topography-tpi-large %TOPOGRAPHY_TPI_LARGE%"
    if not "%TOPOGRAPHY_CACHE_DIR%"=="" set "TOPO_CACHE_ARG=--topography-cache-dir "%TOPOGRAPHY_CACHE_DIR%""
    if not "%TOPOGRAPHY_DEM_DIR%"=="" set "TOPO_DEM_ARG=--topography-dem-dir "%TOPOGRAPHY_DEM_DIR%""
)

python "sentinel2_processing.py" ^
    --aoi "%AOI_PATH%" ^
    --years %YEARS% ^
    --output-dir "%OUTPUT_DIR%" ^
    --seasons %SEASONS% ^
    %NAIP_ARG% ^
    %AUTO_NAIP_FLAG% ^
    %AUTO_NAIP_YEAR_ARG% ^
    %AUTO_NAIP_MAX_ITEMS_ARG% ^
    %AUTO_NAIP_OVERWRITE_FLAG% ^
    %AUTO_NAIP_PREVIEW_FLAG% ^
    %AUTO_WETLANDS_FLAG% ^
    %WETLANDS_OUTPUT_ARG% ^
    %WETLANDS_OVERWRITE_FLAG% ^
    %NAIP_RESOLUTION_ARG% ^
    %AUTO_TOPO_FLAG% ^
    %TOPO_BUFFER_ARG% ^
    %TOPO_SMALL_ARG% ^
    %TOPO_LARGE_ARG% ^
    %TOPO_CACHE_ARG% ^
    %TOPO_DEM_ARG% ^
    --cloud-cover %CLOUD_COVER% ^
    --min-clear-obs %MIN_CLEAR_OBS% ^
    --mask-dilation %MASK_DILATION% ^
    --stac-url "%STAC_URL%" ^
    --log-level %LOG_LEVEL%

if errorlevel 1 (
    echo [ERROR] Sentinel-2 processing failed.
    pause
)

echo [INFO] Sentinel-2 seasonal products and stack manifest generated in %OUTPUT_DIR%
pause
