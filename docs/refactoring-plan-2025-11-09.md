# Wetlands ML GeoAI - Comprehensive Refactoring Plan

**Date:** November 9, 2025
**Version:** 1.0
**Status:** Proposed
**Estimated Total Effort:** 20-30 hours
**Target Completion:** 4-6 weeks (incremental phases)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Identified Issues and Opportunities](#identified-issues-and-opportunities)
4. [Proposed Refactoring Plan](#proposed-refactoring-plan)
5. [Risk Assessment and Mitigation](#risk-assessment-and-mitigation)
6. [Testing Strategy](#testing-strategy)
7. [Success Metrics](#success-metrics)
8. [Appendix: Code Examples](#appendix-code-examples)

---

## Executive Summary

The Wetlands ML GeoAI codebase demonstrates solid architectural foundations with clear module separation and appropriate use of modern Python patterns. However, several critical issues impact maintainability, testability, and extensibility:

### Critical Findings

**🔴 High Priority Issues:**
- **Code Duplication**: Identical geometry buffering logic exists in 2 locations
- **God Functions**: 650-line `run_from_args()` mixes 8+ concerns
- **Parameter Explosion**: `train_unet()` accepts 28 parameters
- **Empty Utilities Module**: No centralized helper functions
- **Missing Tests**: <5% estimated code coverage

**🟡 Medium Priority Issues:**
- **Scattered Configuration**: 20+ environment variables with no registry
- **Inconsistent Patterns**: CLI parsing, logging, and error handling vary
- **No Schema Validation**: Manifest JSON structure is implicit
- **Type Hint Gaps**: 30% of code lacks type annotations

**🟢 Strengths to Preserve:**
- Clean module boundaries (sentinel2, topography, training, inference)
- Dataclass configuration objects with immutability
- Context managers for resource cleanup
- Environment variable support for flexible deployment

### Recommended Approach

Implement a **4-phase incremental refactoring plan** over 4-6 weeks:

1. **Phase 1** (Week 1): Quick wins - Extract utilities, constants, and helpers (4-6 hours)
2. **Phase 2** (Week 2): CLI standardization and documentation (6-8 hours)
3. **Phase 3** (Week 3): Type safety and schema validation (4-6 hours)
4. **Phase 4** (Week 4+): Function decomposition and testing (8-12 hours)

Each phase is designed to be **independently deployable** with **zero breaking changes** to existing CLI interfaces.

---

## Current State Analysis

### Codebase Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Python Files | 31 | Moderate size |
| Lines of Code | ~4,179 | Manageable |
| Module Packages | 5 (sentinel2, topography, training, inference, utils) | Well-organized |
| Largest File | compositing.py (986 LOC) | Needs refactoring |
| Test Files | 3 | Critically insufficient |
| Test Functions | ~8 | <5% coverage |
| Type Hint Coverage | ~70% | Good but incomplete |

### Module Responsibility Matrix

| Module | Purpose | LOC | Coupling | Cohesion | Status |
|--------|---------|-----|----------|----------|--------|
| **sentinel2** | Seasonal compositing pipeline | 1,100 | High (depends on topography) | Medium | Needs decoupling |
| **topography** | LiDAR derivatives computation | 407 | Low | High | ✅ Good |
| **training** | UNet training orchestration | 344 | Medium | Medium | Needs decomposition |
| **inference** | Model prediction streaming | 313 | Low | High | ✅ Good |
| **stacking** | Multi-source raster access | 313 | Low | High | ✅ Good |
| **data_acquisition** | NAIP/wetlands downloads | 250 | Low | High | ✅ Good |
| **tiling** | Tile analysis helpers | 62 | Low | High | ✅ Good |
| **utils** | Shared utilities | 0 | N/A | N/A | ❌ Empty |

### Architectural Strengths

1. **Clean Separation of Concerns**: Each module has a well-defined purpose
2. **Immutable Configuration**: Extensive use of `@dataclass(frozen=True)`
3. **Type Hints**: 70% coverage demonstrates good discipline
4. **Resource Management**: Proper use of context managers (`RasterStack.__enter__/__exit__`)
5. **Flexible Configuration**: Environment variable fallbacks throughout

### Architectural Weaknesses

1. **Code Duplication**: 5+ instances of duplicated logic
2. **Long Functions**: 4 functions exceed 200 lines
3. **Inconsistent Patterns**: CLI, logging, and error handling vary
4. **Missing Abstractions**: No centralized utilities or constants
5. **Tight Coupling**: sentinel2 directly imports topography
6. **Limited Testing**: Major pipelines have 0% coverage

---

## Identified Issues and Opportunities

### 1. Code Duplication (CRITICAL)

#### Issue 1.1: Geometry Buffering Logic

**Severity:** 🔴 **HIGH** | **Effort:** 1 hour | **Risk:** LOW

**Location:**
- `src/wetlands_ml_geoai/sentinel2/compositing.py:239-245` (`_buffer_in_meters`)
- `src/wetlands_ml_geoai/topography/pipeline.py:21-25` (`_buffer_geometry`)

**Current Code:**
```python
# File 1: sentinel2/compositing.py
def _buffer_in_meters(geom: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    if buffer_meters <= 0:
        return geom
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]

# File 2: topography/pipeline.py
def _buffer_geometry(geometry: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]
```

**Problems:**
- Same logic, different function names
- Inconsistent parameter validation (one checks `<= 0`, other doesn't)
- No error handling for edge cases
- Duplicated maintenance burden

**Proposed Solution:**
Extract to `utils/geometry.py` as canonical implementation:

```python
# utils/geometry.py
from shapely.geometry.base import BaseGeometry
import geopandas as gpd

def buffer_geometry_meters(
    geometry: BaseGeometry,
    buffer_meters: float,
    source_crs: str = "EPSG:4326"
) -> BaseGeometry:
    """
    Buffer a geometry by a distance in meters.

    Args:
        geometry: Input geometry in WGS84 (EPSG:4326)
        buffer_meters: Buffer distance in meters
        source_crs: Source CRS (default: EPSG:4326)

    Returns:
        Buffered geometry in source CRS
    """
    if buffer_meters <= 0:
        return geometry

    if geometry.is_empty:
        raise ValueError("Cannot buffer empty geometry")

    series = gpd.GeoSeries([geometry], crs=source_crs)
    utm_crs = series.estimate_utm_crs()
    projected = series.to_crs(utm_crs)
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(source_crs).iloc[0]
```

**Migration Steps:**
1. Create `utils/geometry.py` with implementation
2. Add unit tests for edge cases
3. Update `compositing.py` to import and use
4. Update `pipeline.py` to import and use
5. Remove old `_buffer_in_meters` and `_buffer_geometry`

---

#### Issue 1.2: Rasterio Profile Creation

**Severity:** 🟡 **MEDIUM** | **Effort:** 1 hour | **Risk:** LOW

**Location:**
- `topography/processing.py:160`
- `stacking.py:178`
- `sentinel2/compositing.py:150`
- `inference/unet_stream.py:125`

**Current Pattern (repeated 4+ times):**
```python
profile = {
    "driver": "GTiff",
    "width": width,
    "height": height,
    "dtype": "float32",
    "transform": transform,
    "compress": "deflate",
    "tiled": True,
    "BIGTIFF": "IF_SAFER",
}
```

**Proposed Solution:**
```python
# utils/rasterio_helpers.py
from typing import Any, Optional
from rasterio import Affine
import numpy as np

def create_geotiff_profile(
    width: int,
    height: int,
    transform: Affine,
    count: int = 1,
    dtype: str | np.dtype = "float32",
    crs: Any = None,
    nodata: Optional[float] = None,
    compress: str = "deflate",
    tiled: bool = True,
    bigtiff: str = "IF_SAFER",
    **kwargs
) -> dict[str, Any]:
    """
    Create a standard GeoTIFF profile for rasterio writing.

    Args:
        width: Raster width in pixels
        height: Raster height in pixels
        transform: Affine transformation matrix
        count: Number of bands (default: 1)
        dtype: Data type (default: float32)
        crs: Coordinate reference system
        nodata: NoData value
        compress: Compression method (default: deflate)
        tiled: Enable tiled storage (default: True)
        bigtiff: BigTIFF mode (default: IF_SAFER)
        **kwargs: Additional profile parameters

    Returns:
        Rasterio profile dictionary
    """
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "transform": transform,
        "compress": compress,
        "tiled": tiled,
        "BIGTIFF": bigtiff,
    }

    if crs is not None:
        profile["crs"] = crs

    if nodata is not None:
        profile["nodata"] = nodata

    profile.update(kwargs)
    return profile
```

---

#### Issue 1.3: CLI Argument Parsing

**Severity:** 🟡 **MEDIUM** | **Effort:** 2 hours | **Risk:** MEDIUM

**Location:**
- `train_unet.py:parse_args()` (194 lines)
- `test_unet.py:parse_args()` (113 lines)
- `validate_seasonal_pixels.py:parse_args()` (45 lines)

**Problems:**
- Each module independently reads environment variables
- Duplicate argparse patterns
- Inconsistent naming: `parse_args()` vs `build_parser()`
- No shared validation logic

**Proposed Solution:**
```python
# utils/cli_helpers.py
import argparse
import os
from pathlib import Path
from typing import Optional

def add_raster_input_args(
    parser: argparse.ArgumentParser,
    env_prefix: str = "TRAIN"
) -> None:
    """Add standard raster input arguments."""
    parser.add_argument(
        "--train-raster",
        type=Path,
        default=os.getenv(f"{env_prefix}_RASTER_PATH"),
        help=f"Path to training raster (env: {env_prefix}_RASTER_PATH)"
    )
    parser.add_argument(
        "--stack-manifest",
        type=Path,
        default=os.getenv(f"{env_prefix}_STACK_MANIFEST"),
        help=f"Path to stack manifest JSON (env: {env_prefix}_STACK_MANIFEST)"
    )

def add_tile_args(parser: argparse.ArgumentParser) -> None:
    """Add standard tiling arguments."""
    parser.add_argument(
        "--tile-size",
        type=int,
        default=int(os.getenv("TILE_SIZE", "512")),
        help="Tile size in pixels (env: TILE_SIZE)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=int(os.getenv("TILE_STRIDE", "256")),
        help="Tile stride in pixels (env: TILE_STRIDE)"
    )

def validate_raster_inputs(
    train_raster: Optional[Path],
    stack_manifest: Optional[Path]
) -> None:
    """Validate that at least one raster input is provided."""
    if not train_raster and not stack_manifest:
        raise ValueError(
            "Must provide either --train-raster or --stack-manifest"
        )
```

---

### 2. God Functions & Parameter Explosion (CRITICAL)

#### Issue 2.1: `sentinel2/compositing.py::run_from_args()` (650 lines)

**Severity:** 🔴 **HIGH** | **Effort:** 4-6 hours | **Risk:** MEDIUM

**Current Signature:**
```python
def run_from_args(args) -> None:
    """650-line orchestration mixing 8+ concerns"""
```

**Problems:**
- Mixes parsing, validation, orchestration, I/O, and logging
- Hard to test individual components
- Difficult to reuse sub-steps
- High cognitive complexity

**Concerns Mixed:**
1. AOI parsing and validation
2. NAIP reference preparation and mosaicking
3. Wetlands/NAIP auto-download
4. STAC item fetching
5. Seasonal composite computation
6. Topography stack integration
7. Manifest generation
8. Progress reporting

**Proposed Decomposition:**

```python
# sentinel2/compositing.py (refactored)

@dataclass(frozen=True)
class CompositingConfig:
    """Configuration for Sentinel-2 compositing pipeline."""
    aoi: BaseGeometry
    years: list[int]
    seasons: list[str]
    output_dir: Path
    naip_sources: list[Path]
    buffer_meters: float = 0.0
    cloud_threshold: float = 60.0
    include_topography: bool = False
    # ... other config

class CompositingPipeline:
    """Orchestrates Sentinel-2 seasonal compositing."""

    def __init__(self, config: CompositingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(self) -> list[Path]:
        """Execute full compositing pipeline."""
        # 1. Prepare inputs
        aoi_polygons = self._prepare_aoi()
        naip_reference = self._prepare_naip_reference()

        # 2. Process each AOI
        manifests = []
        for polygon in aoi_polygons:
            manifest = self._process_aoi(polygon, naip_reference)
            manifests.append(manifest)

        # 3. Generate manifest index
        index_path = self._write_manifest_index(manifests)

        return manifests

    def _prepare_aoi(self) -> list[Polygon]:
        """Extract and buffer AOI polygons."""
        # ~20 lines

    def _prepare_naip_reference(self) -> NAIPReference:
        """Prepare NAIP mosaic and metadata."""
        # ~50 lines

    def _process_aoi(
        self,
        polygon: Polygon,
        naip_ref: NAIPReference
    ) -> Path:
        """Process single AOI through all seasons."""
        # ~100 lines

    def _compute_seasonal_composite(
        self,
        polygon: Polygon,
        year: int,
        season: str
    ) -> Path:
        """Compute single seasonal composite."""
        # ~80 lines

    def _integrate_topography(
        self,
        naip_path: Path,
        polygon: Polygon
    ) -> Optional[Path]:
        """Add topography stack if requested."""
        # ~30 lines

    def _write_manifest_index(
        self,
        manifests: list[Path]
    ) -> Path:
        """Generate meta-index of all manifests."""
        # ~20 lines

def run_from_args(args) -> None:
    """CLI entry point - delegates to CompositingPipeline."""
    config = CompositingConfig(
        aoi=parse_aoi(args.aoi),
        years=args.years,
        seasons=args.seasons,
        output_dir=Path(args.output_dir),
        naip_sources=collect_naip_sources(args.naip_path),
        buffer_meters=args.buffer_meters,
        cloud_threshold=args.cloud_threshold,
        include_topography=args.include_topography,
    )

    pipeline = CompositingPipeline(config)
    manifests = pipeline.run()

    logging.info(f"Generated {len(manifests)} stack manifests")
```

**Benefits:**
- Each method has single responsibility
- Easy to test individual steps
- Configuration is explicit and type-safe
- Pipeline can be reused programmatically
- Easier to add hooks/callbacks

---

#### Issue 2.2: `training/unet.py::train_unet()` (28 parameters)

**Severity:** 🔴 **HIGH** | **Effort:** 3-4 hours | **Risk:** MEDIUM

**Current Signature:**
```python
def train_unet(
    labels_path: Path,
    tiles_dir: Path,
    models_dir: Path,
    train_raster: Optional[Path] = None,
    stack_manifest_path: Optional[Sequence[Path | str]] = None,
    tile_size: int = 512,
    stride: int = 256,
    buffer_radius: int = 0,
    num_channels_override: Optional[int] = None,
    num_classes: int = 2,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    batch_size: int = 4,
    epochs: int = 25,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_split: float = 0.2,
    save_best_only: bool = True,
    plot_curves: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    resize_mode: str = "resize",
    num_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume_training: bool = False,
) -> None:
```

**Problems:**
- 28 parameters is unmaintainable
- Hard to remember parameter order
- Difficult to add new parameters
- No logical grouping
- IDE autocomplete becomes useless

**Proposed Solution:**

```python
# training/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

@dataclass(frozen=True)
class TilingConfig:
    """Configuration for tile generation."""
    tile_size: int = 512
    stride: int = 256
    buffer_radius: int = 0
    target_size: Optional[Tuple[int, int]] = None
    resize_mode: str = "resize"

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model architecture."""
    architecture: str = "unet"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    num_channels: Optional[int] = None
    num_classes: int = 2

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 4
    epochs: int = 25
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    val_split: float = 0.2
    num_workers: Optional[int] = None
    seed: int = 42

@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpoint management."""
    save_best_only: bool = True
    checkpoint_path: Optional[Path] = None
    resume_training: bool = False
    plot_curves: bool = False

@dataclass(frozen=True)
class UNetTrainingConfig:
    """Complete configuration for UNet training pipeline."""
    # Input/output paths
    labels_path: Path
    tiles_dir: Path
    models_dir: Path
    train_raster: Optional[Path] = None
    stack_manifest_paths: Optional[Sequence[Path | str]] = None

    # Sub-configurations
    tiling: TilingConfig = field(default_factory=TilingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

# training/unet.py (refactored)
def train_unet(config: UNetTrainingConfig) -> None:
    """
    Train UNet semantic segmentation model.

    Args:
        config: Complete training configuration
    """
    # Access config hierarchically
    tile_size = config.tiling.tile_size
    learning_rate = config.training.learning_rate
    encoder = config.model.encoder_name

    # ... implementation
```

**CLI Adapter:**
```python
# train_unet.py (CLI wrapper)
def main() -> None:
    args = parse_args()

    config = UNetTrainingConfig(
        labels_path=Path(args.labels),
        tiles_dir=Path(args.tiles_dir),
        models_dir=Path(args.models_dir),
        train_raster=Path(args.train_raster) if args.train_raster else None,
        stack_manifest_paths=args.stack_manifest,
        tiling=TilingConfig(
            tile_size=args.tile_size,
            stride=args.stride,
            buffer_radius=args.buffer_radius,
        ),
        model=ModelConfig(
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_classes=args.num_classes,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        ),
        checkpoint=CheckpointConfig(
            save_best_only=args.save_best_only,
            checkpoint_path=args.checkpoint_path,
        ),
    )

    train_unet(config)
```

**Benefits:**
- Logical grouping of related parameters
- Easy to add new parameters to appropriate group
- Type-safe with IDE autocomplete
- Can serialize/deserialize from YAML/JSON
- Testable configuration objects

---

### 3. Empty Utilities Module (CRITICAL)

**Severity:** 🔴 **HIGH** | **Effort:** 2 hours | **Risk:** LOW

**Current State:**
- `utils/__init__.py` is completely empty (0 lines)
- Helper functions scattered across modules
- No centralized location for shared code

**Proposed Structure:**

```
src/wetlands_ml_geoai/utils/
├── __init__.py           # Export public API
├── geometry.py           # Geometry operations
├── rasterio_helpers.py   # Rasterio profile/window helpers
├── constants.py          # Project-wide constants
├── env_vars.py           # Environment variable registry
├── validation.py         # Input validation
└── manifest_schema.py    # Manifest validation
```

**Implementation:**

```python
# utils/__init__.py
"""Shared utilities for wetlands ML GeoAI."""

from .geometry import buffer_geometry_meters
from .rasterio_helpers import create_geotiff_profile, create_window_grid
from .constants import (
    FLOAT_NODATA,
    DEFAULT_TILE_SIZE,
    DEFAULT_STRIDE,
    SEASON_WINDOWS,
    NAIP_BAND_LABELS,
)
from .env_vars import EnvVars
from .validation import validate_path_exists, validate_crs
from .manifest_schema import validate_manifest, ManifestSchema

__all__ = [
    # Geometry
    "buffer_geometry_meters",
    # Rasterio
    "create_geotiff_profile",
    "create_window_grid",
    # Constants
    "FLOAT_NODATA",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_STRIDE",
    "SEASON_WINDOWS",
    "NAIP_BAND_LABELS",
    # Environment
    "EnvVars",
    # Validation
    "validate_path_exists",
    "validate_crs",
    "validate_manifest",
    "ManifestSchema",
]
```

```python
# utils/constants.py
"""Project-wide constants."""

# NoData values
FLOAT_NODATA = -9999.0
UINT8_NODATA = 0

# Default tile parameters
DEFAULT_TILE_SIZE = 512
DEFAULT_STRIDE = 256
DEFAULT_BUFFER_RADIUS = 0

# NAIP bands
NAIP_BAND_LABELS = ["red", "green", "blue", "nir"]

# Sentinel-2 bands
SENTINEL_BANDS = [
    "B02",  # Blue
    "B03",  # Green
    "B04",  # Red
    "B05",  # Red Edge 1
    "B06",  # Red Edge 2
    "B07",  # Red Edge 3
    "B08",  # NIR
    "B8A",  # Red Edge 4
    "B11",  # SWIR 1
    "B12",  # SWIR 2
]

# Seasonal windows
SEASON_WINDOWS = {
    "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "fall": ("09-01", "11-30"),
    "winter": ("12-01", "02-28"),
}

# Topography bands
TOPOGRAPHY_BANDS = [
    "elevation",
    "slope",
    "tpi_small",
    "tpi_large",
    "depression_depth",
]
```

```python
# utils/env_vars.py
"""Registry of all environment variables used in the project."""

class EnvVars:
    """
    Centralized registry of environment variable names.

    Usage:
        import os
        from wetlands_ml_geoai.utils import EnvVars

        raster_path = os.getenv(EnvVars.TRAIN_RASTER_PATH)
    """

    # Training inputs
    TRAIN_RASTER_PATH = "TRAIN_RASTER_PATH"
    TRAIN_STACK_MANIFEST = "TRAIN_STACK_MANIFEST"
    TRAIN_LABELS_PATH = "TRAIN_LABELS_PATH"

    # Training outputs
    TRAIN_TILES_DIR = "TRAIN_TILES_DIR"
    TRAIN_MODELS_DIR = "TRAIN_MODELS_DIR"

    # Tile parameters
    TILE_SIZE = "TILE_SIZE"
    TILE_STRIDE = "TILE_STRIDE"
    TILE_BUFFER = "TILE_BUFFER"

    # Model architecture
    UNET_ARCHITECTURE = "UNET_ARCHITECTURE"
    UNET_ENCODER = "UNET_ENCODER"
    UNET_NUM_CLASSES = "UNET_NUM_CLASSES"

    # Training hyperparameters
    UNET_BATCH_SIZE = "UNET_BATCH_SIZE"
    UNET_NUM_EPOCHS = "UNET_NUM_EPOCHS"
    UNET_LEARNING_RATE = "UNET_LEARNING_RATE"
    UNET_WEIGHT_DECAY = "UNET_WEIGHT_DECAY"

    # Inference
    TEST_RASTER_PATH = "TEST_RASTER_PATH"
    TEST_STACK_MANIFEST = "TEST_STACK_MANIFEST"
    MODEL_PATH = "MODEL_PATH"
    PREDICTIONS_DIR = "PREDICTIONS_DIR"

    # Data acquisition
    NAIP_RASTER_DIR = "NAIP_RASTER_DIR"
    WETLANDS_VECTOR_PATH = "WETLANDS_VECTOR_PATH"

    @classmethod
    def list_all(cls) -> list[str]:
        """Get list of all environment variable names."""
        return [
            v for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str)
        ]

    @classmethod
    def document(cls) -> str:
        """Generate documentation of all environment variables."""
        lines = ["# Environment Variables\n"]
        for name in cls.list_all():
            lines.append(f"- `{name}`")
        return "\n".join(lines)
```

```python
# utils/validation.py
"""Input validation utilities."""

from pathlib import Path
from typing import Union
import rasterio

def validate_path_exists(
    path: Union[Path, str],
    name: str = "Path"
) -> Path:
    """
    Validate that a path exists.

    Args:
        path: Path to validate
        name: Human-readable name for error messages

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If path does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path

def validate_crs(crs: Any) -> bool:
    """
    Validate that a CRS is defined and valid.

    Args:
        crs: CRS to validate (rasterio CRS object)

    Returns:
        True if valid

    Raises:
        ValueError: If CRS is None or invalid
    """
    if crs is None:
        raise ValueError("CRS is None")

    # Add more validation as needed
    return True
```

---

### 4. Missing Schema Validation (MEDIUM)

**Severity:** 🟡 **MEDIUM** | **Effort:** 2 hours | **Risk:** LOW

**Current State:**
- Manifest JSON structure is implicit
- No validation when loading manifests
- Runtime errors when structure changes
- Difficult to debug malformed manifests

**Proposed Solution:**

```python
# utils/manifest_schema.py
"""Schema validation for stack manifests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict
import json

class StackSourceDict(TypedDict, total=False):
    """Type definition for stack source in manifest."""
    path: str
    band_labels: list[str]
    scaling: Optional[float]

class StackGridDict(TypedDict):
    """Type definition for stack grid in manifest."""
    crs: str
    transform: list[float]
    width: int
    height: int

class ManifestDict(TypedDict):
    """Type definition for complete stack manifest."""
    grid: StackGridDict
    naip: StackSourceDict
    sentinel2: dict[str, StackSourceDict]  # season -> source
    topography: Optional[StackSourceDict]

class ManifestSchema:
    """Validator for stack manifest JSON files."""

    REQUIRED_KEYS = {"grid", "naip", "sentinel2"}
    GRID_REQUIRED_KEYS = {"crs", "transform", "width", "height"}
    SOURCE_REQUIRED_KEYS = {"path", "band_labels"}

    @classmethod
    def validate(cls, data: dict[str, Any]) -> None:
        """
        Validate manifest structure.

        Args:
            data: Manifest dictionary to validate

        Raises:
            ValueError: If validation fails
        """
        # Check top-level keys
        missing = cls.REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(f"Manifest missing required keys: {missing}")

        # Validate grid
        cls._validate_grid(data["grid"])

        # Validate NAIP source
        cls._validate_source(data["naip"], "naip")

        # Validate Sentinel-2 sources
        if not isinstance(data["sentinel2"], dict):
            raise ValueError("sentinel2 must be a dictionary")

        for season, source in data["sentinel2"].items():
            cls._validate_source(source, f"sentinel2.{season}")

        # Validate optional topography
        if "topography" in data and data["topography"] is not None:
            cls._validate_source(data["topography"], "topography")

    @classmethod
    def _validate_grid(cls, grid: Any) -> None:
        """Validate grid structure."""
        if not isinstance(grid, dict):
            raise ValueError("grid must be a dictionary")

        missing = cls.GRID_REQUIRED_KEYS - set(grid.keys())
        if missing:
            raise ValueError(f"grid missing required keys: {missing}")

        # Validate types
        if not isinstance(grid["transform"], list):
            raise ValueError("grid.transform must be a list")
        if len(grid["transform"]) != 6:
            raise ValueError("grid.transform must have 6 elements")

        if not isinstance(grid["width"], int) or grid["width"] <= 0:
            raise ValueError("grid.width must be positive integer")

        if not isinstance(grid["height"], int) or grid["height"] <= 0:
            raise ValueError("grid.height must be positive integer")

    @classmethod
    def _validate_source(cls, source: Any, name: str) -> None:
        """Validate source structure."""
        if not isinstance(source, dict):
            raise ValueError(f"{name} must be a dictionary")

        missing = cls.SOURCE_REQUIRED_KEYS - set(source.keys())
        if missing:
            raise ValueError(f"{name} missing required keys: {missing}")

        # Validate path exists
        path = Path(source["path"])
        if not path.exists():
            raise FileNotFoundError(f"{name} path does not exist: {path}")

        # Validate band_labels
        if not isinstance(source["band_labels"], list):
            raise ValueError(f"{name}.band_labels must be a list")

def validate_manifest(manifest_path: Path) -> ManifestDict:
    """
    Load and validate a stack manifest.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Validated manifest dictionary

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest validation fails
        json.JSONDecodeError: If JSON is malformed
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        data = json.load(f)

    ManifestSchema.validate(data)
    return data
```

**Usage:**
```python
# Before (no validation)
with open(manifest_path) as f:
    manifest = json.load(f)
# Crashes later if structure is wrong

# After (validated)
from wetlands_ml_geoai.utils import validate_manifest

manifest = validate_manifest(manifest_path)
# Clear error message immediately if structure is wrong
```

---

### 5. Inconsistent CLI Patterns (MEDIUM)

**Severity:** 🟡 **MEDIUM** | **Effort:** 1.5 hours | **Risk:** LOW

**Current State:**

| File | Pattern | Testability |
|------|---------|-------------|
| `sentinel2/cli.py` | `build_parser() + main(argv=None)` | ✅ Good |
| `train_unet.py` | `parse_args() + main()` | ❌ Hard to test |
| `test_unet.py` | `parse_args() + main()` | ❌ Hard to test |
| `topography/cli.py` | `build_parser() + main(argv=None)` | ✅ Good |

**Proposed Standard:**

```python
# Standard CLI module pattern
import argparse
import sys
from typing import Optional

def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="...",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add arguments
    parser.add_argument("--foo", ...)

    return parser

def main(argv: Optional[list[str]] = None) -> None:
    """
    Main CLI entry point.

    Args:
        argv: Command-line arguments (None = sys.argv)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate
    if not args.required_thing:
        parser.error("Missing required thing")

    # Execute
    run_pipeline(args)

if __name__ == "__main__":
    main()
```

**Benefits:**
- Consistent pattern across all CLIs
- Easy to test with `main(["--foo", "bar"])`
- Reusable parser for documentation/help
- Follows Python best practices

---

### 6. Type Hint Gaps (MEDIUM)

**Severity:** 🟡 **MEDIUM** | **Effort:** 2 hours | **Risk:** LOW

**Current Coverage:** ~70%

**Gaps:**
- `Any` used in manifest dict structures
- Optional not always declared
- Some variable annotations missing
- Return types sometimes omitted

**Target Coverage:** 95%+

**Proposed Actions:**

1. **Add TypedDict for Manifests** (see Issue 4 above)

2. **Strict Optional Typing:**
```python
# Before
def foo(x=None):
    ...

# After
def foo(x: Optional[str] = None) -> Optional[int]:
    ...
```

3. **Variable Annotations:**
```python
# Before
manifests = []
for path in paths:
    manifests.append(load_manifest(path))

# After
manifests: list[StackManifest] = []
for path in paths:
    manifests.append(load_manifest(path))
```

4. **mypy Integration:**
```bash
# Add to CI/pre-commit
mypy src/wetlands_ml_geoai --strict
```

---

## Proposed Refactoring Plan

### Overview

The refactoring will be executed in **4 phases** over **4-6 weeks**, with each phase independently deployable and backward-compatible.

### Phase 1: Foundation & Quick Wins (Week 1)

**Goals:**
- Create utilities module infrastructure
- Extract duplicated code
- Centralize constants and environment variables

**Estimated Effort:** 4-6 hours

#### Tasks

| # | Task | Effort | Files Changed |
|---|------|--------|---------------|
| 1.1 | Create `utils/geometry.py` with `buffer_geometry_meters()` | 30 min | +1 new |
| 1.2 | Create `utils/rasterio_helpers.py` with profile builders | 30 min | +1 new |
| 1.3 | Create `utils/constants.py` with all hardcoded values | 1 hour | +1 new |
| 1.4 | Create `utils/env_vars.py` with EnvVars registry | 30 min | +1 new |
| 1.5 | Update `compositing.py` to use `buffer_geometry_meters()` | 15 min | compositing.py |
| 1.6 | Update `pipeline.py` to use `buffer_geometry_meters()` | 15 min | pipeline.py |
| 1.7 | Update 4 files to use `create_geotiff_profile()` | 45 min | 4 files |
| 1.8 | Update imports to use `utils.constants` | 30 min | Multiple |
| 1.9 | Add unit tests for new utilities | 1 hour | +tests |

**Detailed Steps:**

**Step 1.1-1.4:** Create Utilities Module
```bash
# Create structure
mkdir -p src/wetlands_ml_geoai/utils
touch src/wetlands_ml_geoai/utils/{__init__.py,geometry.py,rasterio_helpers.py,constants.py,env_vars.py}
```

Implement files as shown in Issue #3 above.

**Step 1.5:** Update compositing.py
```python
# Before
from .compositing import _buffer_in_meters

# After
from ..utils import buffer_geometry_meters

# Replace all calls
buffered = buffer_geometry_meters(geom, buffer_meters)
```

**Step 1.6:** Update pipeline.py
```python
# Before
from .pipeline import _buffer_geometry

# After
from ..utils import buffer_geometry_meters

# Replace all calls
buffered = buffer_geometry_meters(geometry, buffer_meters)
```

**Step 1.7:** Update Profile Creation
```python
# Before (in 4 files)
profile = {
    "driver": "GTiff",
    "width": width,
    "height": height,
    ...
}

# After
from ..utils import create_geotiff_profile

profile = create_geotiff_profile(
    width=width,
    height=height,
    transform=transform,
    count=count,
    crs=crs,
    nodata=nodata_value,
)
```

**Step 1.8:** Update Constants
```python
# Before (scattered across files)
FLOAT_NODATA = -9999.0
DEFAULT_TILE_SIZE = 512
SEASON_WINDOWS = {...}

# After
from ..utils.constants import FLOAT_NODATA, DEFAULT_TILE_SIZE, SEASON_WINDOWS
```

**Step 1.9:** Add Tests
```python
# tests/test_utils.py
import pytest
from wetlands_ml_geoai.utils import buffer_geometry_meters
from shapely.geometry import Point

def test_buffer_geometry_meters_positive():
    point = Point(0, 0)
    buffered = buffer_geometry_meters(point, 1000)
    assert buffered.area > point.area

def test_buffer_geometry_meters_zero():
    point = Point(0, 0)
    buffered = buffer_geometry_meters(point, 0)
    assert buffered.equals(point)

def test_buffer_geometry_meters_negative():
    point = Point(0, 0)
    buffered = buffer_geometry_meters(point, -100)
    assert buffered.equals(point)
```

**Acceptance Criteria:**
- ✅ All tests pass
- ✅ No breaking changes to CLI interfaces
- ✅ Code duplication reduced by 50%
- ✅ Constants centralized in single location

---

### Phase 2: CLI Standardization & Documentation (Week 2)

**Goals:**
- Standardize CLI patterns across all modules
- Document all environment variables
- Create shared CLI helpers

**Estimated Effort:** 6-8 hours

#### Tasks

| # | Task | Effort | Files Changed |
|---|------|--------|---------------|
| 2.1 | Create `utils/cli_helpers.py` with shared functions | 1.5 hours | +1 new |
| 2.2 | Refactor `train_unet.py` to standard pattern | 1 hour | train_unet.py |
| 2.3 | Refactor `test_unet.py` to standard pattern | 1 hour | test_unet.py |
| 2.4 | Refactor `validate_seasonal_pixels.py` to standard pattern | 30 min | validate_seasonal_pixels.py |
| 2.5 | Generate environment variable documentation | 30 min | +docs/env_vars.md |
| 2.6 | Add CLI unit tests | 1.5 hours | +tests |
| 2.7 | Update README with env var documentation | 30 min | README.md |

**Detailed Steps:**

**Step 2.1:** Create CLI Helpers (see Issue 1.3 above)

**Step 2.2-2.4:** Standardize CLI Modules
```python
# train_unet.py (after refactoring)
import argparse
from typing import Optional
from wetlands_ml_geoai.utils import cli_helpers

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(...)

    # Use shared helpers
    cli_helpers.add_raster_input_args(parser, env_prefix="TRAIN")
    cli_helpers.add_tile_args(parser)
    cli_helpers.add_model_args(parser)
    cli_helpers.add_training_args(parser)

    return parser

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Use shared validation
    cli_helpers.validate_raster_inputs(args.train_raster, args.stack_manifest)

    # Execute
    train_unet(...)

if __name__ == "__main__":
    main()
```

**Step 2.5:** Generate Documentation
```python
# Generate docs/env_vars.md
from wetlands_ml_geoai.utils import EnvVars

with open("docs/env_vars.md", "w") as f:
    f.write(EnvVars.document())
```

**Step 2.6:** Add Tests
```python
# tests/test_cli.py
import pytest
from train_unet import main

def test_train_unet_cli_missing_inputs():
    with pytest.raises(SystemExit):
        main([])  # Should fail without required inputs

def test_train_unet_cli_with_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"grid": {...}}')

    # Should parse successfully
    main([
        "--stack-manifest", str(manifest),
        "--labels", "labels.gpkg",
        "--tiles-dir", "tiles",
        "--models-dir", "models",
    ])
```

**Acceptance Criteria:**
- ✅ All CLI modules follow same pattern
- ✅ Environment variables documented
- ✅ CLI tests cover basic scenarios
- ✅ All CLIs testable with `main(argv=[...])`

---

### Phase 3: Type Safety & Schema Validation (Week 3)

**Goals:**
- Add manifest schema validation
- Increase type hint coverage to 95%+
- Enable mypy strict mode

**Estimated Effort:** 4-6 hours

#### Tasks

| # | Task | Effort | Files Changed |
|---|------|--------|---------------|
| 3.1 | Create `utils/manifest_schema.py` with validation | 1 hour | +1 new |
| 3.2 | Update `stacking.py` to validate manifests on load | 30 min | stacking.py |
| 3.3 | Add type hints to remaining functions | 1.5 hours | Multiple |
| 3.4 | Add TypedDict for manifest structures | 30 min | Multiple |
| 3.5 | Configure mypy and fix issues | 1 hour | +mypy.ini |
| 3.6 | Add manifest validation tests | 1 hour | +tests |

**Detailed Steps:**

**Step 3.1:** Create Schema Validator (see Issue 4 above)

**Step 3.2:** Update stacking.py
```python
# Before
def load_manifest(path: Union[str, Path]) -> StackManifest:
    with open(path) as f:
        data = json.load(f)
    return StackManifest.from_dict(data)

# After
from .utils import validate_manifest

def load_manifest(path: Union[str, Path]) -> StackManifest:
    data = validate_manifest(Path(path))  # Validates structure
    return StackManifest.from_dict(data)
```

**Step 3.3:** Add Missing Type Hints
```python
# Scan for missing type hints
mypy src/wetlands_ml_geoai --strict 2>&1 | grep "missing type annotation"

# Add annotations
def foo(x: int) -> str:  # Was: def foo(x):
    ...
```

**Step 3.4:** Add TypedDict (see Issue 4 above)

**Step 3.5:** Configure mypy
```ini
# mypy.ini
[mypy]
python_version = 3.8
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = True

[mypy-geoai.*]
ignore_missing_imports = True

[mypy-rasterio.*]
ignore_missing_imports = True
```

**Step 3.6:** Add Tests
```python
# tests/test_manifest_schema.py
import pytest
from wetlands_ml_geoai.utils import validate_manifest, ManifestSchema

def test_validate_manifest_valid(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('''{
        "grid": {"crs": "EPSG:32610", "transform": [1,0,0,0,-1,0], "width": 100, "height": 100},
        "naip": {"path": "naip.tif", "band_labels": ["r", "g", "b", "nir"]},
        "sentinel2": {}
    }''')

    data = validate_manifest(manifest)
    assert data["grid"]["width"] == 100

def test_validate_manifest_missing_grid(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"naip": {}}')

    with pytest.raises(ValueError, match="missing required keys"):
        validate_manifest(manifest)
```

**Acceptance Criteria:**
- ✅ Manifest validation catches structural errors early
- ✅ Type hint coverage ≥95%
- ✅ mypy strict mode passes
- ✅ Clear error messages for invalid manifests

---

### Phase 4: Function Decomposition & Testing (Week 4+)

**Goals:**
- Break down god functions into manageable pieces
- Add comprehensive test coverage
- Improve testability

**Estimated Effort:** 8-12 hours

#### Tasks

| # | Task | Effort | Files Changed |
|---|------|--------|---------------|
| 4.1 | Create `training/config.py` with config dataclasses | 1 hour | +1 new |
| 4.2 | Refactor `train_unet()` to use `UNetTrainingConfig` | 2 hours | training/unet.py |
| 4.3 | Create `CompositingPipeline` class | 3 hours | compositing.py |
| 4.4 | Refactor `run_from_args()` to use pipeline class | 1 hour | compositing.py |
| 4.5 | Add training pipeline tests | 2 hours | +tests |
| 4.6 | Add compositing pipeline tests (with STAC mocking) | 3 hours | +tests |
| 4.7 | Add integration tests | 2 hours | +tests |

**Detailed Steps:**

**Step 4.1-4.2:** Refactor train_unet (see Issue 2.2 above)

**Step 4.3-4.4:** Refactor Compositing (see Issue 2.1 above)

**Step 4.5:** Add Training Tests
```python
# tests/test_training_pipeline.py
import pytest
from wetlands_ml_geoai.training.config import UNetTrainingConfig, TilingConfig
from wetlands_ml_geoai.training.unet import train_unet

def test_training_config_creation():
    config = UNetTrainingConfig(
        labels_path=Path("labels.gpkg"),
        tiles_dir=Path("tiles"),
        models_dir=Path("models"),
        tiling=TilingConfig(tile_size=256),
    )
    assert config.tiling.tile_size == 256

def test_training_config_defaults():
    config = UNetTrainingConfig(
        labels_path=Path("labels.gpkg"),
        tiles_dir=Path("tiles"),
        models_dir=Path("models"),
    )
    assert config.tiling.tile_size == 512  # Default
    assert config.training.batch_size == 4  # Default
```

**Step 4.6:** Add Compositing Tests
```python
# tests/test_compositing_pipeline.py
import pytest
from unittest.mock import patch, MagicMock
from wetlands_ml_geoai.sentinel2.compositing import CompositingPipeline

@pytest.fixture
def mock_stac():
    with patch("pystac_client.Client.open") as mock:
        yield mock

def test_compositing_pipeline_aoi_preparation(tmp_path, mock_stac):
    config = CompositingConfig(
        aoi=Polygon([...]),
        years=[2022],
        seasons=["summer"],
        output_dir=tmp_path,
        naip_sources=[],
    )

    pipeline = CompositingPipeline(config)
    aoi_polygons = pipeline._prepare_aoi()

    assert len(aoi_polygons) > 0
    assert all(isinstance(p, Polygon) for p in aoi_polygons)
```

**Step 4.7:** Add Integration Tests
```python
# tests/test_integration.py
import pytest
from pathlib import Path

def test_full_pipeline_single_aoi(tmp_path, sample_naip_raster):
    """Test complete pipeline from compositing to inference."""
    # 1. Create compositing config
    # 2. Run compositing
    # 3. Verify manifest created
    # 4. Run training
    # 5. Verify model created
    # 6. Run inference
    # 7. Verify predictions created
```

**Acceptance Criteria:**
- ✅ Functions <100 lines each
- ✅ Clear single responsibility
- ✅ Test coverage ≥60% for critical paths
- ✅ Integration tests verify end-to-end workflows

---

## Risk Assessment and Mitigation

### High-Risk Changes

| Change | Risk Level | Impact | Mitigation Strategy |
|--------|-----------|---------|---------------------|
| Refactor `run_from_args()` | 🔴 HIGH | Breaks compositing pipeline | • Thorough integration tests<br>• Parallel implementation<br>• Gradual migration |
| Refactor `train_unet()` signature | 🔴 HIGH | Breaks training CLI | • Keep backward-compatible wrapper<br>• Update all call sites<br>• Comprehensive tests |
| Manifest schema validation | 🟡 MEDIUM | Rejects existing manifests | • Validate against existing manifests<br>• Provide migration tool<br>• Clear error messages |

### Medium-Risk Changes

| Change | Risk Level | Impact | Mitigation Strategy |
|--------|-----------|---------|---------------------|
| CLI pattern standardization | 🟡 MEDIUM | Changes internal API | • Only affects module imports<br>• No user-facing changes<br>• Test all CLI entry points |
| Extract to utils module | 🟡 MEDIUM | Import path changes | • Update all imports atomically<br>• Run full test suite<br>• Use IDE refactoring tools |

### Low-Risk Changes

| Change | Risk Level | Impact | Mitigation Strategy |
|--------|-----------|---------|---------------------|
| Add type hints | 🟢 LOW | No runtime changes | • Use mypy to verify<br>• No behavioral changes |
| Add constants module | 🟢 LOW | Organizational only | • Import both old and new temporarily<br>• Gradual migration |
| Add documentation | 🟢 LOW | No code changes | • Review for accuracy |

### Rollback Strategy

Each phase is independently deployable, allowing rollback at phase boundaries:

**Phase 1:** If issues found, remove `utils/` directory and revert imports
**Phase 2:** Revert CLI changes, use old `parse_args()` pattern
**Phase 3:** Disable manifest validation, remove mypy strict mode
**Phase 4:** Keep old function signatures as deprecated wrappers

### Testing Strategy for Risky Changes

1. **Before Refactoring:**
   - Document current behavior with characterization tests
   - Create baseline output samples

2. **During Refactoring:**
   - Parallel implementation (old + new side by side)
   - Compare outputs byte-for-byte
   - Run both implementations on same inputs

3. **After Refactoring:**
   - Full regression test suite
   - Performance benchmarking
   - Memory profiling

---

## Testing Strategy

### Current State

**Test Coverage:** <5% estimated
**Test Files:** 3
**Test Functions:** ~8

**Gaps:**
- Sentinel-2 compositing: 0 tests
- Training orchestration: 0 tests
- Inference streaming: 0 tests
- Manifest generation: 0 tests
- Data acquisition: 0 tests

### Target State

**Test Coverage:** 60%+ for critical paths
**Test Files:** 15+
**Test Functions:** 100+

### Test Categories

#### 1. Unit Tests (Fast, Isolated)

**Target:** 80+ tests

```python
# tests/unit/test_geometry.py
def test_buffer_geometry_meters_positive()
def test_buffer_geometry_meters_zero()
def test_buffer_geometry_meters_negative()
def test_buffer_geometry_meters_empty_geometry()

# tests/unit/test_manifest_schema.py
def test_validate_manifest_valid()
def test_validate_manifest_missing_grid()
def test_validate_manifest_invalid_transform()
def test_validate_manifest_missing_source_path()

# tests/unit/test_constants.py
def test_season_windows_all_defined()
def test_naip_band_labels_count()

# tests/unit/test_rasterio_helpers.py
def test_create_geotiff_profile_defaults()
def test_create_geotiff_profile_with_nodata()
def test_create_geotiff_profile_with_crs()
```

#### 2. Integration Tests (Medium Speed)

**Target:** 20+ tests

```python
# tests/integration/test_training_pipeline.py
def test_training_with_manifest(sample_manifest, sample_labels)
def test_training_with_single_raster(sample_raster, sample_labels)
def test_training_multi_aoi(multiple_manifests, sample_labels)

# tests/integration/test_inference_pipeline.py
def test_inference_with_manifest(sample_manifest, trained_model)
def test_inference_with_single_raster(sample_raster, trained_model)
def test_inference_streaming_windows(large_raster, trained_model)

# tests/integration/test_stacking.py
def test_raster_stack_window_reading(sample_manifest)
def test_raster_stack_normalization(sample_manifest)
def test_raster_stack_multiple_sources(sample_manifest)
```

#### 3. End-to-End Tests (Slow)

**Target:** 5+ tests

```python
# tests/e2e/test_full_pipeline.py
def test_compositing_to_inference_single_aoi()
def test_compositing_to_inference_multi_aoi()
def test_training_from_scratch_to_prediction()
```

#### 4. CLI Tests

**Target:** 15+ tests

```python
# tests/cli/test_train_unet_cli.py
def test_train_unet_help()
def test_train_unet_missing_inputs()
def test_train_unet_with_env_vars()
def test_train_unet_with_manifest()

# tests/cli/test_test_unet_cli.py
def test_test_unet_help()
def test_test_unet_missing_model()
def test_test_unet_with_manifest()

# tests/cli/test_sentinel2_cli.py
def test_sentinel2_help()
def test_sentinel2_missing_aoi()
def test_sentinel2_invalid_years()
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds

@pytest.fixture
def tmp_geotiff(tmp_path):
    """Create a temporary GeoTIFF for testing."""
    def _create(width=100, height=100, count=3, dtype="uint8"):
        path = tmp_path / "test.tif"

        transform = from_bounds(0, 0, width, height, width, height)

        with rasterio.open(
            path, "w",
            driver="GTiff",
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            transform=transform,
            crs="EPSG:4326",
        ) as dst:
            for i in range(1, count + 1):
                dst.write(np.random.randint(0, 255, (height, width), dtype=np.uint8), i)

        return path

    return _create

@pytest.fixture
def sample_manifest(tmp_path, tmp_geotiff):
    """Create a sample stack manifest."""
    naip_path = tmp_geotiff(count=4)

    manifest_data = {
        "grid": {
            "crs": "EPSG:4326",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 100.0],
            "width": 100,
            "height": 100,
        },
        "naip": {
            "path": str(naip_path),
            "band_labels": ["red", "green", "blue", "nir"],
        },
        "sentinel2": {},
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    return manifest_path

@pytest.fixture
def sample_labels(tmp_path):
    """Create sample label GeoPackage."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame({
        "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        "class": [1],
    }, crs="EPSG:4326")

    path = tmp_path / "labels.gpkg"
    gdf.to_file(path, driver="GPKG")

    return path
```

### Test Execution Strategy

```bash
# Fast unit tests (run frequently)
pytest tests/unit -v

# Integration tests (run before commit)
pytest tests/integration -v

# Full suite (run in CI)
pytest tests/ -v --cov=src/wetlands_ml_geoai --cov-report=html

# E2E tests (run before release)
pytest tests/e2e -v --slow
```

### Coverage Targets

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| sentinel2/compositing.py | 0% | 60% | HIGH |
| training/unet.py | 0% | 70% | HIGH |
| inference/unet_stream.py | 0% | 70% | HIGH |
| stacking.py | 10% | 80% | HIGH |
| topography/ | 20% | 70% | MEDIUM |
| data_acquisition.py | 0% | 50% | MEDIUM |
| utils/ | 0% | 95% | HIGH |

---

## Success Metrics

### Code Quality Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Code Duplication | 5+ instances | 0 instances | Manual code review |
| Longest Function | 650 lines | <100 lines | Static analysis |
| Function Parameters | 28 max | <10 max | Static analysis |
| Type Hint Coverage | 70% | 95% | mypy coverage report |
| Test Coverage | <5% | 60%+ | pytest-cov |
| Cyclomatic Complexity | Unknown | <10 per function | radon/mccabe |

### Maintainability Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Time to Add Feature | Unknown | -30% | Developer survey |
| Time to Fix Bug | Unknown | -40% | Issue tracking |
| Onboarding Time | Unknown | -50% | New developer survey |
| Code Review Time | Unknown | -30% | PR metrics |

### Test Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Test Count | 8 | 100+ | pytest collection |
| Test Execution Time | <1 min | <5 min (full suite) | pytest duration |
| Test Flakiness | Unknown | <2% | CI metrics |
| Bug Detection Rate | Unknown | 80%+ | Defect analysis |

### Documentation Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Env Var Documentation | 0% | 100% | Manual review |
| Docstring Coverage | Unknown | 90% | interrogate |
| API Documentation | Minimal | Complete | Manual review |
| Examples/Tutorials | Minimal | 5+ examples | Manual count |

### Phase Completion Criteria

**Phase 1 Complete When:**
- ✅ `utils/` module created with 5+ helper modules
- ✅ Code duplication reduced to 0 instances
- ✅ All constants centralized
- ✅ 20+ unit tests for utilities
- ✅ All existing tests pass

**Phase 2 Complete When:**
- ✅ All CLI modules use `build_parser() + main(argv=None)` pattern
- ✅ Environment variables documented
- ✅ 15+ CLI tests
- ✅ All existing tests pass
- ✅ No breaking changes to CLI interfaces

**Phase 3 Complete When:**
- ✅ Manifest validation implemented
- ✅ Type hint coverage ≥95%
- ✅ mypy strict mode passes
- ✅ 10+ schema validation tests
- ✅ All existing tests pass

**Phase 4 Complete When:**
- ✅ No function >100 lines
- ✅ No function >10 parameters
- ✅ Test coverage ≥60% for critical paths
- ✅ 50+ new tests added
- ✅ Integration tests verify end-to-end workflows
- ✅ All existing tests pass

### Overall Success Criteria

**Project Refactoring Complete When:**
1. ✅ All 4 phases completed
2. ✅ All acceptance criteria met
3. ✅ Test coverage ≥60%
4. ✅ Type hint coverage ≥95%
5. ✅ No code duplication
6. ✅ No function >100 lines
7. ✅ No function >10 parameters
8. ✅ CI pipeline passes
9. ✅ Documentation complete
10. ✅ Team approval and review

---

## Appendix: Code Examples

### Example 1: Before/After Geometry Buffering

**Before:**
```python
# sentinel2/compositing.py
def _buffer_in_meters(geom: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    if buffer_meters <= 0:
        return geom
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]

# topography/pipeline.py
def _buffer_geometry(geometry: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]
```

**After:**
```python
# utils/geometry.py
def buffer_geometry_meters(
    geometry: BaseGeometry,
    buffer_meters: float,
    source_crs: str = "EPSG:4326"
) -> BaseGeometry:
    """Buffer geometry by distance in meters."""
    if buffer_meters <= 0:
        return geometry

    if geometry.is_empty:
        raise ValueError("Cannot buffer empty geometry")

    series = gpd.GeoSeries([geometry], crs=source_crs)
    utm_crs = series.estimate_utm_crs()
    projected = series.to_crs(utm_crs)
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(source_crs).iloc[0]

# Both files now use:
from ..utils import buffer_geometry_meters
buffered = buffer_geometry_meters(geom, buffer_meters)
```

### Example 2: Before/After Training Configuration

**Before:**
```python
def train_unet(
    labels_path: Path,
    tiles_dir: Path,
    models_dir: Path,
    train_raster: Optional[Path] = None,
    stack_manifest_path: Optional[Sequence[Path | str]] = None,
    tile_size: int = 512,
    stride: int = 256,
    buffer_radius: int = 0,
    num_channels_override: Optional[int] = None,
    num_classes: int = 2,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    batch_size: int = 4,
    epochs: int = 25,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_split: float = 0.2,
    save_best_only: bool = True,
    plot_curves: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    resize_mode: str = "resize",
    num_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume_training: bool = False,
) -> None:
    # 240 lines of implementation
```

**After:**
```python
# training/config.py
@dataclass(frozen=True)
class UNetTrainingConfig:
    """Complete configuration for UNet training pipeline."""
    labels_path: Path
    tiles_dir: Path
    models_dir: Path
    train_raster: Optional[Path] = None
    stack_manifest_paths: Optional[Sequence[Path | str]] = None

    tiling: TilingConfig = field(default_factory=TilingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

# training/unet.py
def train_unet(config: UNetTrainingConfig) -> None:
    """Train UNet model with given configuration."""
    # Clean implementation using config.tiling.tile_size, etc.
```

### Example 3: Before/After CLI Pattern

**Before:**
```python
# train_unet.py
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # ... 100+ lines of argument definitions
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    # Can't test with custom argv
```

**After:**
```python
# train_unet.py
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # ... argument definitions using shared helpers
    return parser

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Testable with main(["--foo", "bar"])

if __name__ == "__main__":
    main()
```

### Example 4: Before/After Manifest Validation

**Before:**
```python
# No validation
with open(manifest_path) as f:
    data = json.load(f)

# Crashes later if structure wrong
naip_path = data["naip"]["path"]  # KeyError if missing
```

**After:**
```python
from wetlands_ml_geoai.utils import validate_manifest

# Validates structure immediately
manifest = validate_manifest(manifest_path)

# Clear error: "Manifest missing required keys: {'naip'}"
naip_path = manifest["naip"]["path"]  # Safe after validation
```

---

## Conclusion

This refactoring plan provides a **systematic, low-risk approach** to improving the Wetlands ML GeoAI codebase over 4-6 weeks. By focusing on incremental phases with clear acceptance criteria, we can modernize the architecture while maintaining backward compatibility and avoiding disruption.

**Key Benefits:**
- **Reduced Maintenance Burden:** Eliminate code duplication and long functions
- **Improved Testability:** Achieve 60%+ test coverage
- **Better Type Safety:** 95%+ type hint coverage with mypy validation
- **Clearer Architecture:** Centralized utilities and consistent patterns
- **Easier Onboarding:** Better documentation and clearer code organization

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1 implementation
3. Establish CI pipeline with automated testing
4. Schedule weekly check-ins to review progress
5. Iterate based on feedback and discoveries

**Estimated Timeline:**
- Phase 1: Week of Nov 11-15, 2025
- Phase 2: Week of Nov 18-22, 2025
- Phase 3: Week of Nov 25-29, 2025
- Phase 4: Week of Dec 2-13, 2025
- Final Review: Week of Dec 16-20, 2025

---

**Document Version:** 1.0
**Last Updated:** November 9, 2025
**Status:** Proposed - Awaiting Approval
