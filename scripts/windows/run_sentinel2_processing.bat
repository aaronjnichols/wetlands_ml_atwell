@echo off
setlocal enabledelayedexpansion

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

REM ------------------------------------------------------------------
REM Edit the values below to match your project before running.
REM ------------------------------------------------------------------

REM AOI Configuration: Use EITHER labels-path (recommended) OR aoi-path
REM   - LABELS_PATH: Path to wetland labels; AOI auto-generated from extent + buffer
REM   - AOI_PATH: Explicit AOI file (only used if LABELS_PATH is empty)
set "LABELS_PATH="
set "LABELS_BUFFER="
set "AOI_PATH=data\MI_Atwell\base_small\aois_base_small_train.gpkg"

set "YEARS=2021 2022 2023"
set "OUTPUT_DIR=data\MI_Atwell\base_small\s2"
set "SEASONS=SPR SUM FAL"
set "NAIP_PATH=data\MI_Atwell\base_small\naip"
set "CLOUD_COVER=60"
set "MIN_CLEAR_OBS=3"
set "MASK_DILATION=0"
set "STAC_URL=https://earth-search.aws.element84.com/v1"
set "LOG_LEVEL=DEBUG"

REM Chunk size for dask parallelization (auto, integer, or empty for default)
REM   - "auto": Compute optimal chunk size based on AOI dimensions (recommended)
REM   - integer: Explicit chunk size in pixels (e.g., 512)
REM   - empty: Use default (2048)
set "CHUNK_SIZE=auto"

REM Performance optimization settings
REM   - MAX_SCENES_PER_SEASON: Limit scenes to N clearest by cloud cover (empty = all)
REM   - PARALLEL_FETCH: Use parallel download strategy (true/false) - 5-10x faster
REM   - FETCH_WORKERS: Number of parallel download workers (default 24)
set "MAX_SCENES_PER_SEASON="
set "PARALLEL_FETCH=true"
set "FETCH_WORKERS=24"

REM Target CRS for all outputs. EPSG:5070 (NAD83 Conus Albers) recommended for CONUS.
REM Prevents CRS mismatches when AOIs span UTM zone boundaries.
set "TARGET_CRS=EPSG:5070"

REM Auto-download configuration (set true/false)
set "AUTO_DOWNLOAD_NAIP=false"
set "AUTO_DOWNLOAD_NAIP_YEAR=2022"
set "AUTO_DOWNLOAD_NAIP_MAX_ITEMS=100"
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
set "TOPOGRAPHY_DEM_DIR="
REM DEM resolution: "1m" (default, highest detail), "10m" (faster), "30m" (fastest)
REM Use "10m" when your stack is 10m resolution (e.g., resampled NAIP + Sentinel-2)
set "DEM_RESOLUTION=10m"
if not exist "venv312\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "venv312\Scripts\activate.bat"
set "PYTHONPATH=src"

REM --- Argument Construction ---

set "ARGS=--years %YEARS% --output-dir "%OUTPUT_DIR%" --seasons %SEASONS%"

REM Use labels-path if set, otherwise use aoi-path
if not "%LABELS_PATH%"=="" (
    set "ARGS=!ARGS! --labels-path "%LABELS_PATH%" --labels-buffer %LABELS_BUFFER%"
) else if not "%AOI_PATH%"=="" (
    set "ARGS=!ARGS! --aoi "%AOI_PATH%""
) else (
    echo [ERROR] Either LABELS_PATH or AOI_PATH must be set.
    pause
    exit /b 1
)
set "ARGS=%ARGS% --cloud-cover %CLOUD_COVER% --min-clear-obs %MIN_CLEAR_OBS% --mask-dilation %MASK_DILATION%"
set "ARGS=%ARGS% --stac-url "%STAC_URL%" --log-level %LOG_LEVEL%"

if not "%CHUNK_SIZE%"=="" set "ARGS=%ARGS% --chunk-size %CHUNK_SIZE%"

REM Performance optimization arguments
if not "%MAX_SCENES_PER_SEASON%"=="" set "ARGS=%ARGS% --max-scenes-per-season %MAX_SCENES_PER_SEASON%"
if /I "%PARALLEL_FETCH%"=="true" set "ARGS=%ARGS% --parallel-fetch"
if not "%FETCH_WORKERS%"=="" set "ARGS=%ARGS% --fetch-workers %FETCH_WORKERS%"

if not "%NAIP_PATH%"=="" set "ARGS=%ARGS% --naip-path "%NAIP_PATH%""
if not "%NAIP_TARGET_RESOLUTION%"=="" set "ARGS=%ARGS% --naip-target-resolution %NAIP_TARGET_RESOLUTION%"
if not "%TARGET_CRS%"=="" set "ARGS=%ARGS% --target-crs %TARGET_CRS%"

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
    if not "%DEM_RESOLUTION%"=="" set "ARGS=!ARGS! --dem-resolution %DEM_RESOLUTION%"
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
