@echo off
setlocal enabledelayedexpansion

REM Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."

REM ------------------------------------------------------------------
REM Edit the values below to match your training setup before running.
REM ------------------------------------------------------------------
set "STACK_MANIFEST=C:\Users\anichols\OneDrive - Atwell LLC\Desktop\_Atwell_AI\Projects\Wetlands_ML\Prelim_Test_MI\data\mi_model_large\s2_train"
set "TRAIN_RASTER="
set "LABELS=C:\Users\anichols\Downloads\MI_geopackage_wetlands\MI_Wetlands_Geopackage.gpkg"
set "TILES_DIR=C:\Users\anichols\OneDrive - Atwell LLC\Desktop\_Atwell_AI\Projects\Wetlands_ML\Prelim_Test_MI\data\mi_model_large\tiles"
set "MODELS_DIR=C:\Users\anichols\OneDrive - Atwell LLC\Desktop\_Atwell_AI\Projects\Wetlands_ML\Prelim_Test_MI\data\mi_model_large\models"
set "TILE_SIZE=512"
set "STRIDE=256"
set "BUFFER=0"
set "BATCH_SIZE=4"
set "EPOCHS=20"
set "LEARNING_RATE=0.0001"
set "WEIGHT_DECAY=0.0001"
set "VAL_SPLIT=0.2"
set "NUM_CHANNELS="
set "ARCHITECTURE=unet"
set "ENCODER_NAME=resnet34"
set "ENCODER_WEIGHTS=imagenet"
set "USE_ENCODER_WEIGHTS=true"
set "SEED=42"
set "TARGET_SIZE="
set "RESIZE_MODE=resize"
set "NUM_WORKERS="
set "PLOT_CURVES=true"
set "SAVE_BEST_ONLY=true"
set "CHECKPOINT_PATH="
set "RESUME_TRAINING=false"
set "LOG_LEVEL=INFO"

if "%LABELS%"=="" (
    echo [ERROR] LABELS must point to your wetlands training data.
    pause
    exit /b 1
)

if "%STACK_MANIFEST%"=="" if "%TRAIN_RASTER%"=="" (
    echo [ERROR] Provide STACK_MANIFEST or TRAIN_RASTER before running.
    pause
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

set "STACK_ARG="
if not "%STACK_MANIFEST%"=="" set "STACK_ARG=--stack-manifest ""%STACK_MANIFEST%"""

set "RASTER_ARG="
if not "%TRAIN_RASTER%"=="" set "RASTER_ARG=--train-raster ""%TRAIN_RASTER%"""

set "NUM_CHANNELS_ARG="
if not "%NUM_CHANNELS%"=="" set "NUM_CHANNELS_ARG=--num-channels %NUM_CHANNELS%"

set "TARGET_SIZE_ARG="
if not "%TARGET_SIZE%"=="" set "TARGET_SIZE_ARG=--target-size ""%TARGET_SIZE%"""

set "NUM_WORKERS_ARG="
if not "%NUM_WORKERS%"=="" set "NUM_WORKERS_ARG=--num-workers %NUM_WORKERS%"

set "PLOT_CURVES_FLAG="
if /I "%PLOT_CURVES%"=="true" set "PLOT_CURVES_FLAG=--plot-curves"

set "SAVE_CHECKPOINT_FLAG="
if /I "%SAVE_BEST_ONLY%"=="false" (
    set "SAVE_CHECKPOINT_FLAG=--save-all-checkpoints"
)

set "ENCODER_FLAG="
if /I "%USE_ENCODER_WEIGHTS%"=="false" (
    set "ENCODER_FLAG=--no-encoder-weights"
) else if not "%ENCODER_WEIGHTS%"=="" (
    set "ENCODER_FLAG=--encoder-weights ""%ENCODER_WEIGHTS%"""
)

set "CHECKPOINT_ARG="
if not "%CHECKPOINT_PATH%"=="" set "CHECKPOINT_ARG=--checkpoint-path ""%CHECKPOINT_PATH%"""

set "RESUME_FLAG="
if /I "%RESUME_TRAINING%"=="true" set "RESUME_FLAG=--resume-training"

set "ENCODER_ARG="
if not "%ENCODER_NAME%"=="" set "ENCODER_ARG=--encoder-name %ENCODER_NAME%"

set "ARCH_ARG="
if not "%ARCHITECTURE%"=="" set "ARCH_ARG=--architecture %ARCHITECTURE%"

set "NUM_CLASSES_ARG="
if not "%NUM_CLASSES%"=="" set "NUM_CLASSES_ARG=--num-classes %NUM_CLASSES%"

python train_unet.py ^
    %STACK_ARG% %RASTER_ARG% ^
    --labels "%LABELS%" ^
    --tiles-dir "%TILES_DIR%" ^
    --models-dir "%MODELS_DIR%" ^
    --tile-size %TILE_SIZE% ^
    --stride %STRIDE% ^
    --buffer %BUFFER% ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --learning-rate %LEARNING_RATE% ^
    --weight-decay %WEIGHT_DECAY% ^
    --val-split %VAL_SPLIT% ^
    %NUM_CHANNELS_ARG% ^
    %ARCH_ARG% ^
    %ENCODER_ARG% ^
    %ENCODER_FLAG% ^
    %NUM_CLASSES_ARG% ^
    --seed %SEED% ^
    %TARGET_SIZE_ARG% ^
    --resize-mode %RESIZE_MODE% ^
    %NUM_WORKERS_ARG% ^
    %PLOT_CURVES_FLAG% ^
    %SAVE_CHECKPOINT_FLAG% ^
    %CHECKPOINT_ARG% ^
    %RESUME_FLAG% ^
    --log-level %LOG_LEVEL%

if errorlevel 1 (
    echo [ERROR] UNet training failed.
    pause
    exit /b 1
)

echo [INFO] UNet training complete. Models saved to %MODELS_DIR%
pause
exit /b 0


