@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0..\.."

REM ==================================================================
REM FULL PIPELINE CONFIGURATION
REM ==================================================================
REM This script runs both stack creation and model training.
REM Edit the values below before running.

REM --- Pipeline Control ---
REM Set to "true" to run each stage, "false" to skip
set "RUN_STACK_CREATION=true"
set "RUN_TRAINING=true"

REM --- Common Paths ---
REM AOI: Use EITHER labels-path OR aoi-path
set "LABELS_PATH="
set "LABELS_BUFFER="
set "AOI_PATH=data\MI_Atwell\base\aois_base_train.gpkg"

REM Training labels (wetlands polygons)
set "TRAINING_LABELS=data\NWI\MI_Wetlands_Geopackage.gpkg"

REM Master output directory - all outputs go here
set "OUTPUT_DIR=data\MI_Atwell\base\s2"

REM ==================================================================
REM SENTINEL-2 / STACK CREATION SETTINGS
REM ==================================================================

set "YEARS=2021 2022 2023"
set "SEASONS=SPR SUM FAL"
set "CLOUD_COVER=60"
set "MIN_CLEAR_OBS=3"
set "MASK_DILATION=0"
set "STAC_URL=https://earth-search.aws.element84.com/v1"
set "CHUNK_SIZE=auto"
set "TARGET_CRS=EPSG:5070"
set "S2_LOG_LEVEL=INFO"

REM NAIP auto-download
set "AUTO_DOWNLOAD_NAIP=true"
set "AUTO_DOWNLOAD_NAIP_YEAR=2022"
set "AUTO_DOWNLOAD_NAIP_MAX_ITEMS=100"
set "NAIP_TARGET_RESOLUTION=0.6"

REM Topography auto-download
set "AUTO_DOWNLOAD_TOPOGRAPHY=true"
set "DEM_RESOLUTION=10m"

REM Performance
set "PARALLEL_FETCH=true"
set "FETCH_WORKERS=24"

REM ==================================================================
REM TRAINING SETTINGS
REM ==================================================================

REM Tiling
set "TILE_SIZE=512"
set "STRIDE=512"
set "BUFFER=0"

REM Model architecture
set "ARCHITECTURE=unet"
set "ENCODER_NAME=resnet34"
set "ENCODER_WEIGHTS=imagenet"

REM Hyperparameters
set "BATCH_SIZE=16"
set "EPOCHS=100"
set "LEARNING_RATE=0.0001"
set "WEIGHT_DECAY=0.0001"
set "VAL_SPLIT=0.2"
set "SEED=42"

REM Balanced sampling (recommended)
set "BALANCED_SAMPLING=false"
set "NWI_PATH="
set "POSITIVE_NEGATIVE_RATIO=1.0"
set "SAFE_ZONE_BUFFER=100.0"

REM Runtime
set "NUM_WORKERS=8"
set "PLOT_CURVES=true"
set "SAVE_BEST_ONLY=true"
set "TRAIN_LOG_LEVEL=INFO"

REM Fine-tuning (optional)
set "CHECKPOINT_PATH="
set "RESUME_TRAINING=false"
set "SKIP_TILING=false"

REM ==================================================================
REM VALIDATION
REM ==================================================================

if "%LABELS_PATH%"=="" if "%AOI_PATH%"=="" (
    echo [ERROR] Either LABELS_PATH or AOI_PATH must be set.
    pause
    exit /b 1
)

if /I "%RUN_TRAINING%"=="true" (
    if "%TRAINING_LABELS%"=="" (
        echo [ERROR] TRAINING_LABELS must be set for training.
        pause
        exit /b 1
    )
)

if not exist "venv312\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "venv312\Scripts\activate.bat"
set "PYTHONPATH=src"

REM ==================================================================
REM DEBUG
REM ==================================================================
echo.
echo [DEBUG] RUN_STACK_CREATION=%RUN_STACK_CREATION%
echo [DEBUG] RUN_TRAINING=%RUN_TRAINING%
echo.

REM ==================================================================
REM STAGE 1: STACK CREATION
REM ==================================================================

if /I "%RUN_STACK_CREATION%" NEQ "true" (
    echo [INFO] Skipping stack creation
    goto training
)

echo.
echo ================================================================
echo STAGE 1: SENTINEL-2 STACK CREATION
echo Started: %date% %time%
echo ================================================================
echo.

set "S2_ARGS=--years %YEARS% --output-dir %OUTPUT_DIR% --seasons %SEASONS%"

if not "%LABELS_PATH%"=="" (
    set "S2_ARGS=!S2_ARGS! --labels-path %LABELS_PATH% --labels-buffer %LABELS_BUFFER%"
) else (
    set "S2_ARGS=!S2_ARGS! --aoi %AOI_PATH%"
)

set "S2_ARGS=%S2_ARGS% --cloud-cover %CLOUD_COVER% --min-clear-obs %MIN_CLEAR_OBS%"
set "S2_ARGS=%S2_ARGS% --mask-dilation %MASK_DILATION% --stac-url %STAC_URL%"
set "S2_ARGS=%S2_ARGS% --chunk-size %CHUNK_SIZE% --target-crs %TARGET_CRS%"
set "S2_ARGS=%S2_ARGS% --log-level %S2_LOG_LEVEL%"

if /I "%PARALLEL_FETCH%"=="true" set "S2_ARGS=%S2_ARGS% --parallel-fetch"
if not "%FETCH_WORKERS%"=="" set "S2_ARGS=%S2_ARGS% --fetch-workers %FETCH_WORKERS%"

if /I "%AUTO_DOWNLOAD_NAIP%"=="true" (
    set "S2_ARGS=!S2_ARGS! --auto-download-naip"
    if not "%AUTO_DOWNLOAD_NAIP_YEAR%"=="" set "S2_ARGS=!S2_ARGS! --auto-download-naip-year %AUTO_DOWNLOAD_NAIP_YEAR%"
    if not "%AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"=="" set "S2_ARGS=!S2_ARGS! --auto-download-naip-max-items %AUTO_DOWNLOAD_NAIP_MAX_ITEMS%"
    if not "%NAIP_TARGET_RESOLUTION%"=="" set "S2_ARGS=!S2_ARGS! --naip-target-resolution %NAIP_TARGET_RESOLUTION%"
)

if /I "%AUTO_DOWNLOAD_TOPOGRAPHY%"=="true" (
    set "S2_ARGS=!S2_ARGS! --auto-download-topography"
    if not "%DEM_RESOLUTION%"=="" set "S2_ARGS=!S2_ARGS! --dem-resolution %DEM_RESOLUTION%"
)

echo [DEBUG] S2_ARGS=%S2_ARGS%
echo.

python -m wetlands_ml_atwell.sentinel2.cli %S2_ARGS%

if errorlevel 1 (
    echo.
    echo [ERROR] Stack creation failed at %date% %time%
    echo [ERROR] Training will not proceed.
    pause
    exit /b 1
)

echo.
echo [INFO] Stack creation completed at %date% %time%
echo.

REM ==================================================================
REM STAGE 2: MODEL TRAINING
REM ==================================================================

:training
if /I "%RUN_TRAINING%" NEQ "true" (
    echo [INFO] Skipping training
    goto done
)

echo.
echo ================================================================
echo STAGE 2: UNET MODEL TRAINING
echo Started: %date% %time%
echo ================================================================
echo.

REM Derived paths from OUTPUT_DIR
set "STACK_MANIFEST=%OUTPUT_DIR%"
set "TILES_DIR=%OUTPUT_DIR%\tiles"
set "MODELS_DIR=%OUTPUT_DIR%\models"

set "TRAIN_ARGS=--stack-manifest %STACK_MANIFEST% --labels %TRAINING_LABELS%"
set "TRAIN_ARGS=%TRAIN_ARGS% --tiles-dir %TILES_DIR% --models-dir %MODELS_DIR%"
set "TRAIN_ARGS=%TRAIN_ARGS% --tile-size %TILE_SIZE% --stride %STRIDE% --buffer %BUFFER%"
set "TRAIN_ARGS=%TRAIN_ARGS% --batch-size %BATCH_SIZE% --epochs %EPOCHS%"
set "TRAIN_ARGS=%TRAIN_ARGS% --learning-rate %LEARNING_RATE% --weight-decay %WEIGHT_DECAY%"
set "TRAIN_ARGS=%TRAIN_ARGS% --val-split %VAL_SPLIT% --seed %SEED%"
set "TRAIN_ARGS=%TRAIN_ARGS% --architecture %ARCHITECTURE% --encoder-name %ENCODER_NAME%"
set "TRAIN_ARGS=%TRAIN_ARGS% --encoder-weights %ENCODER_WEIGHTS%"
set "TRAIN_ARGS=%TRAIN_ARGS% --num-workers %NUM_WORKERS% --log-level %TRAIN_LOG_LEVEL%"

if /I "%PLOT_CURVES%"=="true" set "TRAIN_ARGS=%TRAIN_ARGS% --plot-curves"
if /I "%SAVE_BEST_ONLY%"=="false" set "TRAIN_ARGS=%TRAIN_ARGS% --save-all-checkpoints"

if /I "%BALANCED_SAMPLING%"=="true" (
    set "TRAIN_ARGS=!TRAIN_ARGS! --balanced-sampling"
    if not "%NWI_PATH%"=="" set "TRAIN_ARGS=!TRAIN_ARGS! --nwi-path %NWI_PATH%"
    if not "%POSITIVE_NEGATIVE_RATIO%"=="" set "TRAIN_ARGS=!TRAIN_ARGS! --positive-negative-ratio %POSITIVE_NEGATIVE_RATIO%"
    if not "%SAFE_ZONE_BUFFER%"=="" set "TRAIN_ARGS=!TRAIN_ARGS! --safe-zone-buffer %SAFE_ZONE_BUFFER%"
)

if not "%CHECKPOINT_PATH%"=="" set "TRAIN_ARGS=%TRAIN_ARGS% --checkpoint-path %CHECKPOINT_PATH%"
if /I "%RESUME_TRAINING%"=="true" set "TRAIN_ARGS=%TRAIN_ARGS% --resume-training"
if /I "%SKIP_TILING%"=="true" set "TRAIN_ARGS=%TRAIN_ARGS% --skip-tiling"

echo [DEBUG] TRAIN_ARGS=%TRAIN_ARGS%
echo.

python -m wetlands_ml_atwell.train_unet %TRAIN_ARGS%

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed at %date% %time%
    pause
    exit /b 1
)

echo.
echo [INFO] Training completed at %date% %time%
echo.

:done
echo ================================================================
echo PIPELINE COMPLETE
echo Finished: %date% %time%
echo ================================================================
echo.
echo Output directory: %OUTPUT_DIR%
echo.
pause
exit /b 0
