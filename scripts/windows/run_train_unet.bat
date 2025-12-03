@echo off
setlocal enabledelayedexpansion

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

REM ------------------------------------------------------------------
REM Training Configuration
REM ------------------------------------------------------------------
set "STACK_MANIFEST=data\20251201_MI_NWI_Small_Test\train\s2_refactor_test\aoi_01\stack_manifest.json"
set "TRAIN_RASTER="
set "LABELS=data\mi_nwi_wetlands.gpkg"
set "TILES_DIR=data\20251201_MI_NWI_Small_Test\train\tiles_refactor_test"
set "MODELS_DIR=data\20251201_MI_NWI_Small_Test\train\models_refactor_test"

REM Tiling
set "TILE_SIZE=512"
set "STRIDE=256"
set "BUFFER=0"

REM Model
set "ARCHITECTURE=unet"
set "ENCODER_NAME=resnet34"
set "ENCODER_WEIGHTS=imagenet"
set "USE_ENCODER_WEIGHTS=true"
set "NUM_CHANNELS="
set "NUM_CLASSES="

REM Hyperparameters
set "BATCH_SIZE=16"
set "EPOCHS=3"
set "LEARNING_RATE=0.0001"
set "WEIGHT_DECAY=0.0001"
set "VAL_SPLIT=0.2"
set "SEED=42"

REM Runtime
set "TARGET_SIZE="
set "RESIZE_MODE=resize"
set "NUM_WORKERS="
set "PLOT_CURVES=true"
set "SAVE_BEST_ONLY=true"
set "CHECKPOINT_PATH="
set "RESUME_TRAINING=false"
set "LOG_LEVEL=INFO"

REM ------------------------------------------------------------------
REM Validation
REM ------------------------------------------------------------------

if "%LABELS%"=="" (
    echo [ERROR] LABELS must point to your wetlands training data.
    pause
    exit /b 1
)

if "%STACK_MANIFEST%"=="" if "%TRAIN_RASTER%"=="" (
    echo [ERROR] Provide STACK_MANIFEST or TRAIN_RASTER.
    pause
    exit /b 1
)

if not exist "venv312\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "venv312\Scripts\activate.bat"
set "PYTHONPATH=src"

REM ------------------------------------------------------------------
REM Argument Construction
REM ------------------------------------------------------------------

set "ARGS=--labels "%LABELS%" --tiles-dir "%TILES_DIR%" --models-dir "%MODELS_DIR%""
set "ARGS=%ARGS% --tile-size %TILE_SIZE% --stride %STRIDE% --buffer %BUFFER%"
set "ARGS=%ARGS% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE%"
set "ARGS=%ARGS% --weight-decay %WEIGHT_DECAY% --val-split %VAL_SPLIT% --seed %SEED%"
set "ARGS=%ARGS% --architecture %ARCHITECTURE% --encoder-name %ENCODER_NAME% --resize-mode %RESIZE_MODE%"
set "ARGS=%ARGS% --log-level %LOG_LEVEL%"

if not "%STACK_MANIFEST%"=="" set "ARGS=%ARGS% --stack-manifest "%STACK_MANIFEST%""
if not "%TRAIN_RASTER%"=="" set "ARGS=%ARGS% --train-raster "%TRAIN_RASTER%""

if /I "%USE_ENCODER_WEIGHTS%"=="false" (
    set "ARGS=%ARGS% --no-encoder-weights"
) else (
    if not "%ENCODER_WEIGHTS%"=="" set "ARGS=%ARGS% --encoder-weights %ENCODER_WEIGHTS%"
)

if not "%NUM_CHANNELS%"=="" set "ARGS=%ARGS% --num-channels %NUM_CHANNELS%"
if not "%NUM_CLASSES%"=="" set "ARGS=%ARGS% --num-classes %NUM_CLASSES%"
if not "%TARGET_SIZE%"=="" set "ARGS=%ARGS% --target-size "%TARGET_SIZE%""
if not "%NUM_WORKERS%"=="" set "ARGS=%ARGS% --num-workers %NUM_WORKERS%"
if not "%CHECKPOINT_PATH%"=="" set "ARGS=%ARGS% --checkpoint-path "%CHECKPOINT_PATH%""

if /I "%PLOT_CURVES%"=="true" set "ARGS=%ARGS% --plot-curves"
if /I "%SAVE_BEST_ONLY%"=="false" set "ARGS=%ARGS% --save-all-checkpoints"
if /I "%RESUME_TRAINING%"=="true" set "ARGS=%ARGS% --resume-training"

REM ------------------------------------------------------------------
REM Execution
REM ------------------------------------------------------------------

echo Running UNet Training...
python -m wetlands_ml_atwell.train_unet %ARGS%

if errorlevel 1 (
    echo [ERROR] UNet training failed.
    pause
    exit /b 1
)

echo [INFO] Training complete. Models saved to %MODELS_DIR%
pause
exit /b 0
