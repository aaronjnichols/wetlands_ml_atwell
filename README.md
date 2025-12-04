# Wetlands ML Atwell

Core pipelines for generating Sentinel-2 seasonal composites, stacking with NAIP imagery, training UNet semantic segmentation models, and running inference.

## Project Layout

- `src/wetlands_ml_atwell/` - Reusable Python package.
    - `config/` - Centralized configuration schemas and CLI parsing.
    - `services/` - Unified data acquisition services (NAIP, Wetlands, Topography).
    - `sentinel2/` - Sentinel-2 seasonal compositing and stack generation.
    - `training/` - UNet training logic and tile management.
    - `inference/` - Sliding-window inference pipeline.
- `scripts/windows/` - Windows batch scripts for common workflows.
- `tools/` - Utility scripts (e.g., NAIP download helpers).
- `docs/` - Detailed project documentation and architectural plans.

## Installation

### Quick Setup (Windows)

```cmd
setup.bat
```

This creates a `venv312` virtual environment and installs the package with all dependencies.

### Manual Installation

```bash
# Create and activate virtual environment
python -m venv venv312
venv312\Scripts\activate  # Windows
# source venv312/bin/activate  # Linux/Mac

# Install in editable mode with all dependencies
pip install -e ".[all]"

# Or install just core dependencies
pip install -e .

# Or install with specific extras
pip install -e ".[dev]"      # Development/testing
pip install -e ".[viz]"      # Visualization tools
pip install -e ".[notebook]" # Jupyter notebook support
```

## Quick Start

### 1. Data Acquisition & Compositing

Generate multi-source training stacks from Sentinel-2 imagery.

```bash
# Using the CLI command (after installation)
wetlands-sentinel2 \
    --aoi "path/to/aoi.geojson" \
    --years 2022 2023 \
    --output-dir "data/stacks" \
    --auto-download-naip \
    --auto-download-wetlands \
    --auto-download-topography

# Or using the module interface
python -m wetlands_ml_atwell.sentinel2.cli --help
```

### 2. Training

Train a UNet model using the consolidated configuration system.

```bash
# Using CLI command
wetlands-train \
    --train-raster "data/stacks/aoi_01/stack.tif" \
    --labels "data/labels.gpkg" \
    --epochs 50 \
    --batch-size 8

# Using a YAML configuration file (Recommended)
wetlands-train --config configs/train_v1.yaml

# Or using the module interface
python -m wetlands_ml_atwell.train_unet --help
```

### 3. Inference

Run inference on new imagery.

```bash
# Using CLI command
wetlands-infer \
    --test-raster "data/new_aoi/stack.tif" \
    --model-path "tiles/models_unet/best_model.pth" \
    --output-dir "data/predictions"

# Or using the module interface
python -m wetlands_ml_atwell.test_unet --help
```

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (faster)
pytest tests/unit/

# Skip slow or integration tests
pytest -m "not slow"
pytest -m "not integration"
```

## Architecture

This project follows a service-oriented architecture:

*   **Configuration**: All workflows use typed configuration objects (`TrainingConfig`, `InferenceConfig`) defined in `src/wetlands_ml_atwell/config/`.
*   **Services**: External data interactions are handled by `NaipService`, `WetlandsService`, and `TopographyService` in `src/wetlands_ml_atwell/services/`.
*   **Normalization**: Data is normalized consistently to `[0, 1]` (float32) or `[0, 255]` (uint8) across training and inference to ensure model stability.

See `docs/architecture/refactoring/` for detailed design notes.
