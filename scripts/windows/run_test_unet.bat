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

REM ------------------------------------------------------------------
REM Inference Configuration
REM ------------------------------------------------------------------
set "STACK_MANIFEST=data\MI_Atwell\big\test\check_point_6epochs\s2\aoi_01\stack_manifest.json"
set "TEST_RASTER="
set "MODEL_PATH=data\MI_Atwell\small\20251204\train\s2\models\best_model.pth"
set "OUTPUT_DIR=data\MI_Atwell\big\test\check_point_6epochs"
set "MASK_PATH="
set "VECTOR_PATH="

REM Windowing
set "WINDOW_SIZE=512"
set "OVERLAP=0"
set "BATCH_SIZE=16"

REM Model Params
set "NUM_CHANNELS="
set "NUM_CLASSES=2"
set "ARCHITECTURE=unet"
set "ENCODER_NAME=resnet34"

REM Post-Processing
set "MIN_AREA=10"
set "SIMPLIFY=2.0"
set "PROB_THRESHOLD=0.5"
set "LOG_LEVEL=INFO"

REM ------------------------------------------------------------------
REM Validation
REM ------------------------------------------------------------------

if "%MODEL_PATH%"=="" (
    echo [ERROR] MODEL_PATH must point to a trained UNet checkpoint.
    goto :fail
)

if "%STACK_MANIFEST%"=="" if "%TEST_RASTER%"=="" (
    echo [ERROR] Provide STACK_MANIFEST or TEST_RASTER.
    goto :fail
)

if not exist "venv312\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat first.
    goto :fail
)

call "venv312\Scripts\activate.bat"
set "PYTHONPATH=src"

REM ------------------------------------------------------------------
REM Argument Construction
REM ------------------------------------------------------------------

set "ARGS=--model-path "%MODEL_PATH%" --output-dir "%OUTPUT_DIR%""
set "ARGS=%ARGS% --window-size %WINDOW_SIZE% --overlap %OVERLAP% --batch-size %BATCH_SIZE%"
set "ARGS=%ARGS% --architecture %ARCHITECTURE% --encoder-name %ENCODER_NAME%"
set "ARGS=%ARGS% --min-area %MIN_AREA% --simplify-tolerance %SIMPLIFY% --probability-threshold %PROB_THRESHOLD%"
set "ARGS=%ARGS% --log-level %LOG_LEVEL%"

if not "%STACK_MANIFEST%"=="" set "ARGS=%ARGS% --stack-manifest "%STACK_MANIFEST%""
if not "%TEST_RASTER%"=="" set "ARGS=%ARGS% --test-raster "%TEST_RASTER%""
if not "%MASK_PATH%"=="" set "ARGS=%ARGS% --masks "%MASK_PATH%""
if not "%VECTOR_PATH%"=="" set "ARGS=%ARGS% --vectors "%VECTOR_PATH%""

if not "%NUM_CHANNELS%"=="" set "ARGS=%ARGS% --num-channels %NUM_CHANNELS%"
if not "%NUM_CLASSES%"=="" set "ARGS=%ARGS% --num-classes %NUM_CLASSES%"

REM ------------------------------------------------------------------
REM Execution
REM ------------------------------------------------------------------

echo Running UNet Inference...
python -m wetlands_ml_atwell.test_unet %ARGS%

if errorlevel 1 (
    echo [ERROR] Inference failed.
    goto :fail
)

echo [INFO] Inference complete. Outputs in %OUTPUT_DIR%
goto :success

:fail
if defined _stay_open pause
exit /b 1

:success
if defined _stay_open pause
exit /b 0
