# Wetlands ML Atwell

Core pipelines for generating Sentinel-2 seasonal composites, stacking with NAIP imagery, training UNet semantic segmentation models, and running inference.

## Project Layout

- `src/wetlands_ml_atwell/` - Reusable Python package.
    - `config/` - Centralized configuration schemas and CLI parsing.
    - `services/` - Unified data acquisition services (NAIP, Wetlands, Topography).
    - `sentinel2/` - Sentinel-2 seasonal compositing and stack generation.
    - `training/` - UNet training logic and tile management.
    - `inference/` - Sliding-window inference pipeline.
- `train_unet.py`, `test_unet.py`, `sentinel2_processing.py` - Thin CLI wrappers.
- `tools/` - Utility scripts (e.g., NAIP download helpers).
- `docs/` - Detailed project documentation and architectural plans.

## Quick Start

### Environment Setup
```bash
python setup.bat
```

### 1. Data Acquisition & Compositing
Use `sentinel2_processing.py` to generate multi-source training stacks.

```bash
# Minimal example with auto-download enabled
python sentinel2_processing.py \
    --aoi "path/to/aoi.geojson" \
    --years 2022 2023 \
    --output-dir "data/stacks" \
    --auto-download-naip \
    --auto-download-wetlands \
    --auto-download-topography
```

### 2. Training
Train a UNet model using the consolidated configuration system.

```bash
# Using CLI arguments
python train_unet.py \
    --train-raster "data/stacks/aoi_01/stack.tif" \
    --labels "data/labels.gpkg" \
    --epochs 50 \
    --batch-size 8

# Using a YAML configuration file (Recommended)
python train_unet.py --config configs/train_v1.yaml
```

### 3. Inference
Run inference on new imagery.

```bash
python test_unet.py \
    --test-raster "data/new_aoi/stack.tif" \
    --model-path "tiles/models_unet/best_model.pth" \
    --output-dir "data/predictions"
```

## Architecture

This project follows a service-oriented architecture:

*   **Configuration**: All workflows use typed configuration objects (`TrainingConfig`, `InferenceConfig`) defined in `src/wetlands_ml_atwell/config/`.
*   **Services**: External data interactions are handled by `NaipService`, `WetlandsService`, and `TopographyService` in `src/wetlands_ml_atwell/services/`.
*   **Normalization**: Data is normalized consistently to `[0, 1]` (float32) or `[0, 255]` (uint8) across training and inference to ensure model stability.

See `docs/architecture/refactoring/` for detailed design notes.
