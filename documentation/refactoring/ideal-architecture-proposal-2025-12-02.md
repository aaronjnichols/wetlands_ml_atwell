# Ideal Architecture: Wetlands ML Pipeline

**Date:** 2025-12-02
**Type:** Greenfield Architecture Proposal

---

## Design Principles

1. **Single Source of Truth** - One place for each concept (normalization, config, schemas)
2. **Dependency Injection** - Components receive dependencies, not create them
3. **Configuration over Code** - Behavior controlled by config files, not hardcoded
4. **Protocol-Based Abstractions** - Define contracts via protocols/interfaces
5. **Immutable Data Flow** - Data transforms create new objects, don't mutate
6. **Fail Fast with Clear Errors** - Validate early, provide actionable messages
7. **Test-First Design** - Every component designed for testability

---

## Proposed Project Structure

```
wetlands_ml/
├── pyproject.toml                    # Modern Python packaging (PEP 621)
├── configs/
│   ├── default.yaml                  # Base configuration
│   ├── sentinel2.yaml                # Sentinel-2 specific settings
│   ├── training.yaml                 # Training hyperparameters
│   └── inference.yaml                # Inference settings
│
├── src/
│   └── wetlands_ml/
│       ├── __init__.py
│       ├── py.typed                  # PEP 561 marker
│       │
│       ├── core/                     # Foundation layer - no external deps
│       │   ├── __init__.py
│       │   ├── config.py             # Configuration loading & validation
│       │   ├── types.py              # Core type definitions
│       │   ├── errors.py             # Custom exception hierarchy
│       │   ├── protocols.py          # Abstract interfaces (Protocol classes)
│       │   └── constants.py          # Named constants with documentation
│       │
│       ├── domain/                   # Domain models - pure Python
│       │   ├── __init__.py
│       │   ├── geometry.py           # AOI, bounding boxes, transforms
│       │   ├── raster.py             # RasterMetadata, BandInfo, Window
│       │   ├── manifest.py           # StackManifest, StackSource (Pydantic)
│       │   ├── normalization.py      # NormalizationConfig, scaling strategies
│       │   └── prediction.py         # PredictionResult, confidence maps
│       │
│       ├── io/                       # I/O adapters - external world boundary
│       │   ├── __init__.py
│       │   ├── readers/
│       │   │   ├── __init__.py
│       │   │   ├── base.py           # RasterReader protocol
│       │   │   ├── geotiff.py        # GeoTIFF implementation
│       │   │   ├── cog.py            # Cloud-optimized GeoTIFF
│       │   │   └── virtual.py        # VRT/multi-source reader
│       │   ├── writers/
│       │   │   ├── __init__.py
│       │   │   ├── base.py           # RasterWriter protocol
│       │   │   ├── geotiff.py        # GeoTIFF writer
│       │   │   └── tiles.py          # Tile writer for training
│       │   ├── stac/
│       │   │   ├── __init__.py
│       │   │   ├── client.py         # STAC API client
│       │   │   ├── queries.py        # Query builders
│       │   │   └── models.py         # STAC item models
│       │   └── downloads/
│       │       ├── __init__.py
│       │       ├── naip.py           # NAIP download service
│       │       ├── dem.py            # 3DEP DEM download service
│       │       └── base.py           # Download protocol & retry logic
│       │
│       ├── processing/               # Data processing - pure transforms
│       │   ├── __init__.py
│       │   ├── normalize.py          # Single normalization implementation
│       │   ├── composite.py          # Seasonal compositing
│       │   ├── mosaic.py             # Raster mosaicking
│       │   ├── mask.py               # Cloud/quality masking
│       │   ├── topography.py         # DEM derivatives (slope, TPI, etc.)
│       │   ├── stack.py              # Multi-source stacking
│       │   └── window.py             # Windowed operations
│       │
│       ├── ml/                       # Machine learning components
│       │   ├── __init__.py
│       │   ├── datasets/
│       │   │   ├── __init__.py
│       │   │   ├── base.py           # Dataset protocol
│       │   │   ├── segmentation.py   # Segmentation dataset (no /255 hack!)
│       │   │   └── transforms.py     # Augmentation transforms
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   ├── factory.py        # Model factory
│       │   │   ├── unet.py           # UNet implementation/wrapper
│       │   │   └── checkpoints.py    # Checkpoint management
│       │   ├── training/
│       │   │   ├── __init__.py
│       │   │   ├── trainer.py        # Training loop
│       │   │   ├── metrics.py        # Evaluation metrics
│       │   │   └── callbacks.py      # Training callbacks
│       │   └── inference/
│       │       ├── __init__.py
│       │       ├── predictor.py      # Prediction orchestration
│       │       ├── sliding_window.py # Sliding window strategy
│       │       └── postprocess.py    # Prediction post-processing
│       │
│       ├── pipelines/                # High-level orchestration
│       │   ├── __init__.py
│       │   ├── base.py               # Pipeline protocol
│       │   ├── sentinel2.py          # Sentinel-2 compositing pipeline
│       │   ├── topography.py         # Topography generation pipeline
│       │   ├── training.py           # Training pipeline
│       │   ├── inference.py          # Inference pipeline
│       │   └── validation.py         # Validation pipeline
│       │
│       ├── services/                 # Application services (DI-ready)
│       │   ├── __init__.py
│       │   ├── manifest_service.py   # Manifest creation & validation
│       │   ├── tile_service.py       # Tile generation service
│       │   ├── prediction_service.py # End-to-end prediction
│       │   └── container.py          # Dependency injection container
│       │
│       └── cli/                      # Command-line interface
│           ├── __init__.py
│           ├── main.py               # Entry point, command groups
│           ├── commands/
│           │   ├── __init__.py
│           │   ├── composite.py      # wetlands composite ...
│           │   ├── train.py          # wetlands train ...
│           │   ├── predict.py        # wetlands predict ...
│           │   └── validate.py       # wetlands validate ...
│           └── utils.py              # CLI utilities
│
├── tests/
│   ├── conftest.py                   # Shared fixtures
│   ├── fixtures/
│   │   ├── configs/                  # Test configurations
│   │   ├── manifests/                # Sample manifests
│   │   ├── rasters/                  # Small test rasters
│   │   └── mocks/                    # Mock responses (STAC, etc.)
│   ├── unit/
│   │   ├── core/
│   │   ├── domain/
│   │   ├── io/
│   │   ├── processing/
│   │   └── ml/
│   ├── integration/
│   │   ├── test_sentinel2_pipeline.py
│   │   ├── test_training_pipeline.py
│   │   └── test_inference_pipeline.py
│   └── e2e/
│       └── test_full_workflow.py
│
└── docs/
    ├── architecture.md
    ├── configuration.md
    ├── api/                          # Generated API docs
    └── tutorials/
```

---

## Core Design Patterns

### 1. Protocol-Based Abstractions

```python
# src/wetlands_ml/core/protocols.py
"""Define contracts that components must fulfill."""

from typing import Protocol, Iterator
import numpy as np
from numpy.typing import NDArray

from .types import Window, BoundingBox, Transform


class RasterReader(Protocol):
    """Contract for reading raster data."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (bands, height, width)."""
        ...

    @property
    def crs(self) -> str | None:
        """Return coordinate reference system."""
        ...

    @property
    def transform(self) -> Transform:
        """Return affine transform."""
        ...

    @property
    def nodata(self) -> float | None:
        """Return nodata value."""
        ...

    def read(self, window: Window | None = None) -> NDArray[np.float32]:
        """Read raster data, optionally windowed."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class RasterWriter(Protocol):
    """Contract for writing raster data."""

    def write(
        self,
        data: NDArray[np.float32],
        transform: Transform,
        crs: str | None = None,
        nodata: float | None = None,
    ) -> None:
        """Write raster data."""
        ...


class Normalizer(Protocol):
    """Contract for data normalization strategies."""

    def normalize(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize data to model-ready format."""
        ...

    def denormalize(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Reverse normalization (if applicable)."""
        ...


class DataSource(Protocol):
    """Contract for multi-source data access."""

    @property
    def band_labels(self) -> tuple[str, ...]:
        """Return band names."""
        ...

    def read_window(self, window: Window) -> NDArray[np.float32]:
        """Read and pre-process a window of data."""
        ...


class Pipeline(Protocol):
    """Contract for data processing pipelines."""

    def run(self, config: "PipelineConfig") -> "PipelineResult":
        """Execute the pipeline."""
        ...

    def validate(self, config: "PipelineConfig") -> list[str]:
        """Validate configuration, return list of errors."""
        ...
```

### 2. Immutable Domain Models (Pydantic)

```python
# src/wetlands_ml/domain/manifest.py
"""Stack manifest domain models with validation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BandScaling(BaseModel):
    """Per-band normalization configuration."""

    min_value: float
    max_value: float

    @model_validator(mode="after")
    def validate_range(self) -> "BandScaling":
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value must be less than max_value")
        return self


class StackSourceConfig(BaseModel):
    """Configuration for a single data source in a stack."""

    type: Literal["naip", "sentinel2", "topography", "custom"]
    path: Path
    band_labels: tuple[str, ...]
    scale_max: float | None = None
    scale_min: float | None = None
    band_scaling: dict[str, BandScaling] | None = None
    nodata: float | None = None
    resample: Literal["nearest", "bilinear", "cubic"] = "bilinear"
    description: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        resolved = v.expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"Source path does not exist: {resolved}")
        return resolved

    @model_validator(mode="after")
    def validate_scaling(self) -> "StackSourceConfig":
        """Ensure scaling configuration is consistent."""
        if self.band_scaling:
            missing = set(self.band_labels) - set(self.band_scaling.keys())
            if missing:
                raise ValueError(f"Missing band_scaling for bands: {missing}")
        return self


class GridConfig(BaseModel):
    """Spatial grid configuration."""

    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    nodata: float = -9999.0


class StackManifest(BaseModel):
    """Complete stack manifest with validation."""

    version: str = "1.0"
    grid: GridConfig
    sources: tuple[StackSourceConfig, ...]
    created_at: str | None = None
    description: str | None = None

    @property
    def total_bands(self) -> int:
        return sum(len(s.band_labels) for s in self.sources)

    @property
    def all_band_labels(self) -> tuple[str, ...]:
        labels = []
        for source in self.sources:
            labels.extend(source.band_labels)
        return tuple(labels)

    def source_by_type(self, source_type: str) -> StackSourceConfig | None:
        for source in self.sources:
            if source.type == source_type:
                return source
        return None

    @classmethod
    def from_file(cls, path: Path) -> "StackManifest":
        """Load and validate manifest from JSON file."""
        import json
        data = json.loads(path.read_text())
        return cls.model_validate(data)

    def to_file(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.write_text(self.model_dump_json(indent=2))
```

### 3. Single Normalization Implementation

```python
# src/wetlands_ml/processing/normalize.py
"""Single source of truth for all normalization."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class NormalizationStrategy(Enum):
    """Available normalization strategies."""

    MINMAX = auto()      # Scale to [0, 1] using min/max
    ZSCORE = auto()      # Standardize to mean=0, std=1
    PERCENTILE = auto()  # Scale using percentiles (robust to outliers)
    IDENTITY = auto()    # No normalization


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for data normalization."""

    strategy: NormalizationStrategy = NormalizationStrategy.MINMAX
    nodata_value: float | None = -9999.0
    clip_range: tuple[float, float] | None = (0.0, 1.0)
    fill_nodata: float = 0.0
    fill_nan: float = 0.0
    fill_inf: float = 1.0


class Normalizer:
    """
    Unified normalizer for training and inference.

    This is THE ONLY place normalization logic lives.
    Both training tile generation and inference use this class.
    """

    def __init__(self, config: NormalizationConfig | None = None):
        self.config = config or NormalizationConfig()

    def normalize(
        self,
        data: NDArray[np.float32],
        *,
        warn_on_clip: bool = True,
    ) -> NDArray[np.float32]:
        """
        Normalize data to model-ready format.

        Args:
            data: Input array (C, H, W) with raw values
            warn_on_clip: Log warning if values are clipped

        Returns:
            Normalized array ready for model input
        """
        result = data.astype(np.float32, copy=True)

        # Step 1: Handle nodata
        if self.config.nodata_value is not None:
            nodata_mask = result == self.config.nodata_value
            result[nodata_mask] = self.config.fill_nodata

        # Step 2: Handle NaN and infinity
        nan_mask = np.isnan(result)
        inf_mask = np.isinf(result)
        result[nan_mask] = self.config.fill_nan
        result[inf_mask] = np.sign(result[inf_mask]) * self.config.fill_inf

        # Step 3: Apply strategy-specific normalization
        # (already done by source-level scaling in our case)

        # Step 4: Clip to valid range
        if self.config.clip_range is not None:
            lo, hi = self.config.clip_range

            if warn_on_clip:
                valid = ~(nodata_mask | nan_mask | inf_mask)
                if valid.any():
                    max_val = float(result[valid].max())
                    min_val = float(result[valid].min())
                    if max_val > hi:
                        import logging
                        logging.warning(
                            "Values %.2f > %.2f will be clipped. "
                            "Check source scaling configuration.",
                            max_val, hi
                        )
                    if min_val < lo:
                        import logging
                        logging.warning(
                            "Values %.2f < %.2f will be clipped. "
                            "Check source scaling configuration.",
                            min_val, lo
                        )

            np.clip(result, lo, hi, out=result)

        return result

    def __call__(
        self,
        data: NDArray[np.float32],
        **kwargs,
    ) -> NDArray[np.float32]:
        """Allow using normalizer as a callable."""
        return self.normalize(data, **kwargs)


# Module-level default for convenience
default_normalizer = Normalizer()


def normalize(
    data: NDArray[np.float32],
    config: NormalizationConfig | None = None,
) -> NDArray[np.float32]:
    """Convenience function for normalization."""
    normalizer = Normalizer(config) if config else default_normalizer
    return normalizer.normalize(data)
```

### 4. Dependency Injection Container

```python
# src/wetlands_ml/services/container.py
"""Dependency injection for clean architecture."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.config import Config
from ..processing.normalize import Normalizer, NormalizationConfig
from ..io.readers.geotiff import GeoTiffReader
from ..io.writers.geotiff import GeoTiffWriter
from ..ml.models.factory import ModelFactory


@dataclass
class Container:
    """
    Dependency injection container.

    Provides configured instances of all services.
    Makes testing trivial - just swap implementations.
    """

    config: Config
    _instances: dict[str, Any] = field(default_factory=dict)

    def get_normalizer(self) -> Normalizer:
        """Get configured normalizer instance."""
        if "normalizer" not in self._instances:
            norm_config = NormalizationConfig(
                nodata_value=self.config.get("nodata_value", -9999.0),
                clip_range=tuple(self.config.get("clip_range", [0.0, 1.0])),
            )
            self._instances["normalizer"] = Normalizer(norm_config)
        return self._instances["normalizer"]

    def get_reader(self, path: Path) -> GeoTiffReader:
        """Get raster reader for path."""
        return GeoTiffReader(path)

    def get_writer(self, path: Path) -> GeoTiffWriter:
        """Get raster writer for path."""
        return GeoTiffWriter(path)

    def get_model_factory(self) -> ModelFactory:
        """Get model factory."""
        if "model_factory" not in self._instances:
            self._instances["model_factory"] = ModelFactory(
                default_architecture=self.config.get("model.architecture", "unet"),
                default_encoder=self.config.get("model.encoder", "resnet34"),
            )
        return self._instances["model_factory"]

    @classmethod
    def from_config_file(cls, path: Path) -> "Container":
        """Create container from config file."""
        config = Config.from_yaml(path)
        return cls(config=config)

    @classmethod
    def for_testing(cls, **overrides) -> "Container":
        """Create container with test defaults."""
        config = Config(data={"testing": True, **overrides})
        return cls(config=config)
```

### 5. Clean Dataset Without Hacks

```python
# src/wetlands_ml/ml/datasets/segmentation.py
"""Segmentation dataset - clean implementation without /255 hack."""

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from ...processing.normalize import Normalizer, NormalizationConfig


class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation.

    Key design decisions:
    1. Tiles are ALREADY normalized to [0, 1] when written
    2. No /255 division needed - data is float32 [0, 1]
    3. Same normalizer used for tile creation and inference
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        normalizer: Normalizer | None = None,
        transform: Callable | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.normalizer = normalizer or Normalizer()
        self.transform = transform

        # Find all image tiles
        self.image_paths = sorted(self.images_dir.glob("*.tif"))
        if not self.image_paths:
            raise ValueError(f"No .tif files found in {images_dir}")

        # Verify labels exist
        for img_path in self.image_paths:
            label_path = self.labels_dir / img_path.name
            if not label_path.exists():
                raise ValueError(f"Missing label for {img_path.name}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / img_path.name

        # Read image - already normalized to [0, 1]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            # Note: NO /255 here! Data is already [0, 1]

        # Read label
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)

        # Apply augmentation transforms if provided
        if self.transform is not None:
            # Transform expects (H, W, C) for albumentations
            image_hwc = image.transpose(1, 2, 0)
            transformed = self.transform(image=image_hwc, mask=label)
            image = transformed["image"].transpose(2, 0, 1)
            label = transformed["mask"]

        return torch.from_numpy(image), torch.from_numpy(label)


class TileWriter:
    """
    Write training tiles with proper normalization.

    Ensures tiles are written with same normalization as inference.
    """

    def __init__(
        self,
        output_dir: Path,
        normalizer: Normalizer,
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.normalizer = normalizer

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def write_tile(
        self,
        image: np.ndarray,
        label: np.ndarray,
        name: str,
        transform,
        crs,
    ) -> None:
        """
        Write a training tile pair.

        Image is normalized using the same normalizer that inference uses.
        This ensures training and inference see identical data.
        """
        # Normalize image
        normalized = self.normalizer.normalize(image, warn_on_clip=True)

        # Write image tile
        profile = {
            "driver": "GTiff",
            "height": normalized.shape[1],
            "width": normalized.shape[2],
            "count": normalized.shape[0],
            "dtype": "float32",
            "transform": transform,
            "crs": crs,
            "compress": "deflate",
            "tiled": True,
        }

        img_path = self.images_dir / f"{name}.tif"
        with rasterio.open(img_path, "w", **profile) as dst:
            dst.write(normalized)

        # Write label tile
        label_profile = {
            **profile,
            "count": 1,
            "dtype": "uint8",
        }
        label_path = self.labels_dir / f"{name}.tif"
        with rasterio.open(label_path, "w", **label_profile) as dst:
            dst.write(label.astype(np.uint8), 1)
```

### 6. Unified Inference

```python
# src/wetlands_ml/ml/inference/predictor.py
"""Unified prediction with guaranteed training/inference alignment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from numpy.typing import NDArray

from ...core.protocols import RasterReader, DataSource
from ...domain.manifest import StackManifest
from ...processing.normalize import Normalizer
from ..models.factory import ModelFactory
from .sliding_window import SlidingWindowIterator


@dataclass
class PredictionConfig:
    """Configuration for prediction."""

    window_size: int = 512
    overlap: int = 128
    batch_size: int = 1
    num_classes: int = 2
    probability_threshold: float | None = None
    device: str = "auto"


class Predictor:
    """
    Unified predictor for all input types.

    Guarantees same preprocessing as training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        normalizer: Normalizer,
        config: PredictionConfig,
    ):
        self.model = model
        self.normalizer = normalizer
        self.config = config

        # Determine device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)
        self.model.eval()

    def predict_from_manifest(
        self,
        manifest: StackManifest,
    ) -> NDArray[np.uint8]:
        """
        Run prediction on a stack manifest.

        Uses the same data pipeline as training tile generation.
        """
        from ...io.readers.stack import StackReader

        reader = StackReader(manifest)
        return self._predict(
            reader=reader,
            height=manifest.grid.height,
            width=manifest.grid.width,
        )

    def predict_from_raster(
        self,
        path: Path,
        scale_factor: float | None = None,
    ) -> NDArray[np.uint8]:
        """
        Run prediction on a single raster.

        Args:
            path: Path to raster file
            scale_factor: Scale factor to apply (e.g., 255.0 for uint8)
                         If None, auto-detect from dtype
        """
        from ...io.readers.geotiff import GeoTiffReader

        reader = GeoTiffReader(path, scale_factor=scale_factor)
        return self._predict(
            reader=reader,
            height=reader.height,
            width=reader.width,
        )

    def _predict(
        self,
        reader: RasterReader,
        height: int,
        width: int,
    ) -> NDArray[np.uint8]:
        """Core prediction logic."""
        # Initialize accumulators
        prob_sum = np.zeros(
            (self.config.num_classes, height, width),
            dtype=np.float32,
        )
        count = np.zeros((height, width), dtype=np.float32)

        # Sliding window iteration
        windows = SlidingWindowIterator(
            height=height,
            width=width,
            window_size=self.config.window_size,
            overlap=self.config.overlap,
        )

        with torch.no_grad():
            for window in windows:
                # Read window data
                data = reader.read(window)

                # Apply SAME normalization as training
                normalized = self.normalizer.normalize(data)

                # Convert to tensor
                tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

                # Get predictions
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                # Accumulate
                r0, r1 = window.row_off, window.row_off + window.height
                c0, c1 = window.col_off, window.col_off + window.width
                prob_sum[:, r0:r1, c0:c1] += probs
                count[r0:r1, c0:c1] += 1

        # Average predictions
        count = np.maximum(count, 1e-6)
        avg_probs = prob_sum / count[None, :, :]

        # Threshold or argmax
        if self.config.probability_threshold is not None:
            mask = (avg_probs[1] >= self.config.probability_threshold).astype(np.uint8)
        else:
            mask = np.argmax(avg_probs, axis=0).astype(np.uint8)

        return mask

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        normalizer: Normalizer,
        config: PredictionConfig,
        architecture: str = "unet",
        encoder: str = "resnet34",
        in_channels: int = 25,
    ) -> "Predictor":
        """Create predictor from saved checkpoint."""
        factory = ModelFactory()
        model = factory.create(
            architecture=architecture,
            encoder=encoder,
            in_channels=in_channels,
            num_classes=config.num_classes,
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return cls(model=model, normalizer=normalizer, config=config)
```

### 7. Configuration System

```yaml
# configs/default.yaml
# Base configuration - all other configs inherit from this

project:
  name: "wetlands_ml"
  version: "2.0.0"

data:
  nodata_value: -9999.0
  default_crs: "EPSG:32618"

normalization:
  strategy: "minmax"
  clip_range: [0.0, 1.0]
  fill_nodata: 0.0

sentinel2:
  collection: "sentinel-2-l2a"
  bands:
    - B03  # Green
    - B04  # Red
    - B05  # Vegetation Red Edge 1
    - B06  # Vegetation Red Edge 2
    - B08  # NIR
    - B11  # SWIR 1
    - B12  # SWIR 2
  scale_factor: 0.0001
  cloud_mask:
    scl_values:
      cloud_shadow: 3
      cloud_medium: 8
      cloud_high: 9
      cirrus: 10
      snow_ice: 11
  seasons:
    spring:
      start: [3, 1]   # March 1
      end: [5, 31]    # May 31
    summer:
      start: [6, 1]
      end: [8, 31]
    fall:
      start: [9, 1]
      end: [11, 30]

topography:
  derivatives:
    - slope
    - tpi_small
    - tpi_large
    - depression_depth
  tpi_radii:
    small: 15.0
    large: 75.0
  scaling:
    slope: [0.0, 90.0]
    tpi_small: [-50.0, 50.0]
    tpi_large: [-100.0, 100.0]
    depression_depth: [0.0, 50.0]

naip:
  bands: ["R", "G", "B", "NIR"]
  scale_max: 255.0

model:
  architecture: "unet"
  encoder: "resnet34"
  encoder_weights: "imagenet"

training:
  tile_size: 512
  stride: 256
  batch_size: 4
  epochs: 25
  learning_rate: 0.001
  weight_decay: 0.0001
  validation_split: 0.2
  seed: 42

inference:
  window_size: 512
  overlap: 128
  probability_threshold: 0.5
  num_classes: 2
```

```python
# src/wetlands_ml/core/config.py
"""Configuration management."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Config:
    """
    Hierarchical configuration with validation.

    Supports:
    - YAML file loading
    - Environment variable overrides
    - Dot-notation access (config.get("model.architecture"))
    - Type validation via Pydantic
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        parts = key.split(".")
        value = self._data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def section(self, name: str) -> "Config":
        """Get a config subsection."""
        data = self.get(name, {})
        return Config(data if isinstance(data, dict) else {})

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data or {})

    @classmethod
    def from_files(cls, *paths: Path) -> "Config":
        """Load and merge multiple config files."""
        merged = {}
        for path in paths:
            if path.exists():
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                cls._deep_merge(merged, data)
        return cls(merged)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value
```

### 8. Clean CLI with Click

```python
# src/wetlands_ml/cli/main.py
"""Command-line interface."""

import click
from pathlib import Path

from ..core.config import Config
from ..services.container import Container


@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file",
)
@click.pass_context
def cli(ctx, config: Path | None):
    """Wetlands ML Pipeline - Semantic segmentation for wetland detection."""
    ctx.ensure_object(dict)

    # Load configuration
    config_paths = [Path("configs/default.yaml")]
    if config:
        config_paths.append(config)
    ctx.obj["config"] = Config.from_files(*config_paths)
    ctx.obj["container"] = Container(ctx.obj["config"])


@cli.command()
@click.option("--aoi", required=True, help="Area of interest (WKT, GeoJSON, or file path)")
@click.option("--output-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--years", "-y", multiple=True, type=int, required=True)
@click.option("--seasons", "-s", multiple=True, default=["spring", "summer", "fall"])
@click.pass_context
def composite(ctx, aoi: str, output_dir: Path, years: tuple[int], seasons: tuple[str]):
    """Generate Sentinel-2 seasonal composites."""
    from ..pipelines.sentinel2 import Sentinel2Pipeline

    container = ctx.obj["container"]
    pipeline = Sentinel2Pipeline(container)

    result = pipeline.run(
        aoi=aoi,
        output_dir=output_dir,
        years=list(years),
        seasons=list(seasons),
    )

    click.echo(f"Created manifest: {result.manifest_path}")


@cli.command()
@click.option("--manifest", "-m", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--labels", "-l", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.pass_context
def train(ctx, manifest: Path, labels: Path, output_dir: Path):
    """Train a segmentation model."""
    from ..pipelines.training import TrainingPipeline

    container = ctx.obj["container"]
    pipeline = TrainingPipeline(container)

    result = pipeline.run(
        manifest_path=manifest,
        labels_path=labels,
        output_dir=output_dir,
    )

    click.echo(f"Model saved: {result.model_path}")
    click.echo(f"Best validation IoU: {result.best_iou:.4f}")


@cli.command()
@click.option("--manifest", "-m", type=click.Path(exists=True, path_type=Path))
@click.option("--raster", "-r", type=click.Path(exists=True, path_type=Path))
@click.option("--model", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--threshold", "-t", type=float, default=0.5)
@click.pass_context
def predict(ctx, manifest: Path | None, raster: Path | None, model: Path, output: Path, threshold: float):
    """Run inference on raster data."""
    if not manifest and not raster:
        raise click.UsageError("Must provide either --manifest or --raster")

    from ..pipelines.inference import InferencePipeline

    container = ctx.obj["container"]
    pipeline = InferencePipeline(container)

    result = pipeline.run(
        manifest_path=manifest,
        raster_path=raster,
        model_path=model,
        output_path=output,
        probability_threshold=threshold,
    )

    click.echo(f"Prediction saved: {result.output_path}")


if __name__ == "__main__":
    cli()
```

---

## Key Differences from Current Implementation

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Normalization** | Split across 3 files, /255 hack | Single `Normalizer` class used everywhere |
| **Configuration** | Hardcoded constants | YAML-based, hierarchical |
| **Validation** | Runtime errors | Pydantic models, fail-fast |
| **Testing** | 6% coverage | Designed for 90%+ coverage |
| **File sizes** | 995 LOC max | 300 LOC max |
| **Dependencies** | Implicit | Injected via Container |
| **Error handling** | Bare exceptions | Custom exception hierarchy |
| **Type safety** | Partial | Full type hints + mypy strict |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONFIGURATION                                   │
│  configs/*.yaml → Config → Container → Inject dependencies                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA ACQUISITION                                   │
│  io/stac/ → Sentinel-2 items                                                │
│  io/downloads/ → NAIP tiles, DEM products                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PROCESSING                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ composite   │  │ topography  │  │ mosaic      │                         │
│  │ (seasonal)  │  │ (slope,TPI) │  │ (NAIP)      │                         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                         │
│         │                │                │                                  │
│         └────────────────┴────────────────┘                                  │
│                          │                                                   │
│                          ▼                                                   │
│                   ┌─────────────┐                                            │
│                   │   stack     │  → StackManifest                          │
│                   └─────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│         TRAINING            │   │         INFERENCE           │
│                             │   │                             │
│  StackManifest              │   │  StackManifest              │
│       │                     │   │       │                     │
│       ▼                     │   │       ▼                     │
│  ┌─────────────┐            │   │  ┌─────────────┐            │
│  │ TileWriter  │            │   │  │ StackReader │            │
│  │ + Normalizer│ ◄──────────┼───┼──│ + Normalizer│            │
│  └──────┬──────┘   SAME     │   │  └──────┬──────┘            │
│         │       NORMALIZER  │   │         │                   │
│         ▼                   │   │         ▼                   │
│  ┌─────────────┐            │   │  ┌─────────────┐            │
│  │ Dataset     │            │   │  │ Predictor   │            │
│  │ (no /255!)  │            │   │  │             │            │
│  └──────┬──────┘            │   │  └──────┬──────┘            │
│         │                   │   │         │                   │
│         ▼                   │   │         ▼                   │
│  ┌─────────────┐            │   │  ┌─────────────┐            │
│  │  Trainer    │            │   │  │  Prediction │            │
│  └──────┬──────┘            │   │  └─────────────┘            │
│         │                   │   │                             │
│         ▼                   │   │                             │
│  Model Checkpoint           │   │                             │
└─────────────────────────────┘   └─────────────────────────────┘
```

---

## Migration Path

If migrating from current to proposed:

### Phase 1: Core Foundation
1. Create `core/` module with protocols and config
2. Create `domain/` module with Pydantic models
3. Add comprehensive tests

### Phase 2: Processing Layer
1. Port normalization to single implementation
2. Port processing functions with tests
3. Deprecate old `stacking.py` functions

### Phase 3: ML Layer
1. Create clean Dataset class (no /255)
2. Port training with new normalizer
3. Port inference with new normalizer
4. **Retrain models** with new normalization

### Phase 4: CLI & Integration
1. Build new CLI with Click
2. Create pipelines as orchestrators
3. Full integration testing

---

## Summary

This architecture:

1. **Eliminates the /255 hack** by using a single normalizer everywhere
2. **Provides compile-time safety** via protocols and type hints
3. **Makes testing trivial** via dependency injection
4. **Keeps files small** (<300 LOC each)
5. **Externalizes configuration** via YAML files
6. **Validates early** via Pydantic models
7. **Separates concerns** clearly between layers

The key insight is that **training and inference must use identical preprocessing**, which is guaranteed by injecting the same `Normalizer` instance into both `TileWriter` (training) and `Predictor` (inference).
