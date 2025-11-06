@echo off
setlocal enabledelayedexpansion

rem Relaunch with persistent window when double-clicked
if "%~1"=="" (
    start "" cmd /k "%~f0" --stay-open
    exit /b
)
if /i "%~1"=="--stay-open" (
    set "_stay_open=1"
    shift
)

rem Change to project root directory (2 levels up from scripts\windows)
cd /d "%~dp0..\.."
if errorlevel 1 goto :fail

set "STACK_MANIFEST=data\00_small_model_train_test\s2_test_topo\aoi_01\stack_manifest.json"
set "TEST_RASTER="
set "MODEL_PATH=C:\_code\python\wetlands_ml_codex\data00_small_model_train_test\models\best_model.pth"
set "OUTPUT_DIR=data\00_small_model_train_test\s2_test_topo"
set "MASK_PATH="
set "VECTOR_PATH="
set "WINDOW_SIZE=512"
set "OVERLAP=256"
set "BATCH_SIZE=4"
set "NUM_CHANNELS="
set "NUM_CLASSES=2"
set "ARCHITECTURE=unet"
set "ENCODER_NAME=resnet34"
set "MIN_AREA=100"
set "SIMPLIFY=1.0"
set "PROB_THRESHOLD="
set "LOG_LEVEL=INFO"

if "%MODEL_PATH%"=="" goto :missing_model
if "%STACK_MANIFEST%"=="" if "%TEST_RASTER%"=="" goto :missing_inputs

if not exist "venv312\Scripts\activate.bat" goto :missing_venv

call "venv312\Scripts\activate.bat"
if errorlevel 1 goto :fail

set "STACK_ARG="
if not "%STACK_MANIFEST%"=="" set "STACK_ARG=--stack-manifest ^"%STACK_MANIFEST%^""

set "RASTER_ARG="
if not "%TEST_RASTER%"=="" set "RASTER_ARG=--test-raster ^"%TEST_RASTER%^""

set "MASK_ARG="
if not "%MASK_PATH%"=="" set "MASK_ARG=--masks ^"%MASK_PATH%^""

set "VECTOR_ARG="
if not "%VECTOR_PATH%"=="" set "VECTOR_ARG=--vectors ^"%VECTOR_PATH%^""

set "NUM_CHANNELS_ARG="
if not "%NUM_CHANNELS%"=="" set "NUM_CHANNELS_ARG=--num-channels %NUM_CHANNELS%"

set "NUM_CLASSES_ARG="
if not "%NUM_CLASSES%"=="" set "NUM_CLASSES_ARG=--num-classes %NUM_CLASSES%"

set "ARCH_ARG="
if not "%ARCHITECTURE%"=="" set "ARCH_ARG=--architecture %ARCHITECTURE%"

set "ENCODER_ARG="
if not "%ENCODER_NAME%"=="" set "ENCODER_ARG=--encoder-name %ENCODER_NAME%"

set "WINDOW_ARG="
if not "%WINDOW_SIZE%"=="" set "WINDOW_ARG=--window-size %WINDOW_SIZE%"

set "OVERLAP_ARG="
if not "%OVERLAP%"=="" set "OVERLAP_ARG=--overlap %OVERLAP%"

set "BATCH_ARG="
if not "%BATCH_SIZE%"=="" set "BATCH_ARG=--batch-size %BATCH_SIZE%"

set "MIN_AREA_ARG="
if not "%MIN_AREA%"=="" set "MIN_AREA_ARG=--min-area %MIN_AREA%"

set "SIMPLIFY_ARG="
if not "%SIMPLIFY%"=="" set "SIMPLIFY_ARG=--simplify-tolerance %SIMPLIFY%"

set "PROB_THRESHOLD_ARG="
if not "%PROB_THRESHOLD%"=="" set "PROB_THRESHOLD_ARG=--probability-threshold %PROB_THRESHOLD%"

set "LOG_LEVEL_ARG="
if not "%LOG_LEVEL%"=="" set "LOG_LEVEL_ARG=--log-level %LOG_LEVEL%"

python test_unet.py %STACK_ARG% %RASTER_ARG% --model-path "%MODEL_PATH%" --output-dir "%OUTPUT_DIR%" %MASK_ARG% %VECTOR_ARG% %WINDOW_ARG% %OVERLAP_ARG% %BATCH_ARG% %NUM_CHANNELS_ARG% %NUM_CLASSES_ARG% %ARCH_ARG% %ENCODER_ARG% %MIN_AREA_ARG% %SIMPLIFY_ARG% %PROB_THRESHOLD_ARG% %LOG_LEVEL_ARG%
if errorlevel 1 goto :py_fail

echo [INFO] UNet inference complete. Outputs saved to %OUTPUT_DIR%
goto :success

:missing_model
echo [ERROR] MODEL_PATH must point to a trained UNet checkpoint (.pth).
goto :fail

:missing_inputs
echo [ERROR] Provide STACK_MANIFEST or TEST_RASTER before running.
goto :fail

:missing_venv
echo [ERROR] Python virtual environment not found. Run setup.bat first.
goto :fail

:py_fail
echo [ERROR] UNet inference failed.
goto :fail

:fail
if defined _stay_open pause
exit /b 1

:success
if defined _stay_open pause
exit /b 0
