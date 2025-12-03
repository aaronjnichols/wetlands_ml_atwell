# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wetlands ML Atwell is a Python package for wetland semantic segmentation using satellite imagery. It provides pipelines for Sentinel-2 seasonal compositing, NAIP imagery integration, topography feature extraction, UNet model training, and sliding-window inference.

## Common Commands

### Environment Setup
```bash
# Windows: creates venv and installs dependencies
setup.bat
```

### Running the Main Workflows
```bash
# Data acquisition and compositing
python -m wetlands_ml_atwell.sentinel2 --aoi path/to/aoi.geojson --years 2022 2023 --output-dir data/stacks

# Training (CLI or YAML config)
python -m wetlands_ml_atwell.train_unet --train-raster data/stack.tif --labels data/labels.gpkg
python -m wetlands_ml_atwell.train_unet --config configs/train.yaml

# Inference
python -m wetlands_ml_atwell.test_unet --test-raster data/stack.tif --model-path models/best_model.pth
```

### Testing
```bash
# Run all tests (pytest configured in pytest.ini)
pytest

# Run only unit tests (faster)
pytest tests/unit/

# Run a single test file
pytest tests/unit/test_normalization.py

# Skip slow tests
pytest -m "not slow"

# Skip integration tests
pytest -m "not integration"
```

## Architecture

### Core Pipeline Stages

1. **Data Acquisition** (`services/download.py`): `NaipService`, `WetlandsService`, `TopographyService` handle external data downloads
2. **Sentinel-2 Compositing** (`sentinel2/compositing.py`, `seasonal.py`): Creates cloud-masked seasonal median composites via STAC API
3. **Stack Assembly** (`stacking.py`): `StackManifest` and `RasterStack` classes virtualize multi-source rasters (NAIP, Sentinel-2, topography) for streaming access
4. **Training** (`training/unet.py`): Tile extraction, geoai integration, UNet training orchestration
5. **Inference** (`inference/unet_stream.py`): Sliding-window inference with batch processing

### Configuration System

All workflows use typed dataclasses in `config/models.py`:
- `TrainingConfig`: training parameters, tiling, model architecture
- `InferenceConfig`: inference parameters, output paths
- Configs can be built from CLI args or YAML files (`config/yaml_loader.py`)

### Normalization Pipeline (Critical)

The `normalization.py` module handles data format conversion required for geoai compatibility:

- geoai's training dataset **always divides by 255**, assuming uint8 input
- All data is normalized to float32 [0-1] internally, then converted to uint8 [0-255] for tiles
- `to_geoai_format()`: Converts [0-1] float → [0-255] uint8 for tile export
- `prepare_for_model()`: Handles both legacy float32 and new uint8 tiles during inference

### Stack Manifest Format

Manifests (`stack_manifest.json`) define virtual multi-source rasters:
```json
{
  "grid": {"crs": "EPSG:32610", "transform": [...], "width": 5000, "height": 5000},
  "sources": [
    {"type": "naip", "path": "naip.tif", "band_labels": ["R","G","B","NIR"], "scale_max": 255},
    {"type": "sentinel", "path": "sentinel.tif", "band_labels": ["spring_B02",...]}
  ]
}
```

### Key Dependencies

- `geoai-py`: Training orchestration, model architectures (wraps segmentation_models_pytorch)
- `rasterio`/`rioxarray`: Raster I/O and windowed reading
- `pystac-client`/`stackstac`: STAC catalog access for Sentinel-2
- `torch`: PyTorch backend

## Code Conventions

- Entry points are thin CLI wrappers (`train_unet.py`, `test_unet.py`) that delegate to module functions
- Use `TrainingConfig`/`InferenceConfig` dataclasses rather than passing many individual parameters
- Raster data flows as float32 [0-1] internally; convert to uint8 only at tile export boundaries
- Per-band scaling for topography features uses `band_scaling` dict in manifests
