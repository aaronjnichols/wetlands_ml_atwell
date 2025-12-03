# Wetlands ML GeoAI - Comprehensive Refactoring Plan

**Date:** December 2, 2025  
**Version:** 2.0  
**Status:** Analysis Complete, Awaiting Implementation

---

## Executive Summary

The Wetlands ML GeoAI codebase provides an end-to-end pipeline for wetland detection using multi-source satellite imagery (NAIP, Sentinel-2, topography) and UNet semantic segmentation. While functionally complete, the codebase exhibits several architectural weaknesses that impact maintainability, testability, and extensibility:

1. **Monolithic modules** – The Sentinel-2 compositing module spans ~1000 lines, coupling AOI parsing, STAC queries, data downloads, and manifest generation
2. **Duplicated configuration logic** – Training and inference CLIs each define 30+ arguments with redundant environment variable fallbacks
3. **Inconsistent data flow** – Multiple normalization stages scattered across modules led to past inference bugs (now fixed via `/255` workaround)
4. **Sparse test coverage** – Only CLI argument parsing and one normalization helper are tested
5. **Legacy tool scripts** – Standalone utilities bypass shared services

This plan proposes a phased refactoring approach that extracts cohesive services, introduces explicit configuration objects, unifies data acquisition, and raises test coverage—all while maintaining backward compatibility with existing workflows.

### Key Changes Since Previous Analysis (Nov 14, 2025)

| Issue | Status | Notes |
|-------|--------|-------|
| Topography band scaling | ✅ Fixed | `band_scaling` dict added to manifest, `RasterStack.read_window` handles per-band normalization |
| Inference `/255` mismatch | ✅ Fixed | `_prepare_window` now applies `/255` to match training pipeline |
| Test suite bug | ✅ Fixed | Test updated to include `/255` step |
| Monolithic compositing | 🔴 Not Started | Still 986 lines |
| Duplicated CLI config | 🔴 Not Started | Training/inference have separate arg parsers |
| Test coverage | 🟡 Partial | Only 3 test files, no integration tests |

---

## Current State Analysis

### Architecture Overview

```
wetlands_ml_codex/
├── src/wetlands_ml_geoai/
│   ├── __init__.py                    # Minimal exports
│   ├── data_acquisition.py            # NAIP/wetlands download (253 LOC)
│   ├── stacking.py                    # RasterStack + normalization (383 LOC)
│   ├── tiling.py                      # Channel helpers (61 LOC)
│   ├── train_unet.py                  # Training CLI entry (427 LOC)
│   ├── test_unet.py                   # Inference CLI entry (231 LOC)
│   ├── validate_seasonal_pixels.py    # Unused?
│   │
│   ├── sentinel2/
│   │   ├── cli.py                     # Thin CLI wrapper (30 LOC)
│   │   ├── compositing.py             # MONSTER FILE (995 LOC) ⚠️
│   │   ├── manifests.py               # Manifest I/O (141 LOC)
│   │   └── progress.py                # Progress bars
│   │
│   ├── topography/
│   │   ├── config.py                  # TopographyStackConfig (36 LOC)
│   │   ├── download.py                # 3DEP API client
│   │   ├── pipeline.py                # Orchestration (116 LOC)
│   │   └── processing.py              # DEM derivatives (204 LOC)
│   │
│   ├── training/
│   │   ├── extraction.py              # Tile extraction (276 LOC)
│   │   ├── sampling.py                # NWI-filtered sampling (297 LOC)
│   │   └── unet.py                    # Training orchestration (345 LOC)
│   │
│   └── inference/
│       ├── common.py                  # Path resolution (40 LOC)
│       └── unet_stream.py             # Streaming inference (313 LOC)
│
├── train_unet.py                      # Root CLI shim
├── test_unet.py                       # Root CLI shim
├── sentinel2_processing.py            # Root CLI shim
└── tests/
    ├── test_training_cli.py           # Manifest resolution tests
    ├── test_inference_cli.py          # Normalization tests
    └── test_topography_pipeline.py    # Topography smoke test
```

### Module Dependency Graph

```
                    ┌─────────────────────┐
                    │    CLI Entry Points │
                    │ train_unet.py       │
                    │ test_unet.py        │
                    │ sentinel2_proc.py   │
                    └────────┬────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ sentinel2/      │ │ training/       │ │ inference/      │
│ compositing.py  │ │ unet.py         │ │ unet_stream.py  │
│ (995 LOC) ⚠️    │ │ (345 LOC)       │ │ (313 LOC)       │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │     ┌─────────────┼───────────────────┘
         │     │             │
         ▼     ▼             ▼
    ┌────────────────────────────┐
    │        stacking.py         │
    │  RasterStack, normalize    │
    │  load_manifest (383 LOC)   │
    └────────────────────────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ data_acquisition│    │ topography/     │
│ (253 LOC)       │    │ pipeline.py     │
└─────────────────┘    └─────────────────┘
```

### Key Code Patterns

#### 1. Sentinel-2 Compositing Monolith

The `run_pipeline` function in `compositing.py` handles 10+ distinct responsibilities:

```python
# Lines 622-824 (condensed)
def run_pipeline(
    aoi: str,
    years: Sequence[int],
    output_dir: Path,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    # ... 20+ parameters
) -> None:
    # 1. Directory setup
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. NAIP collection and validation
    naip_sources = collect_naip_sources(naip_candidates)
    
    # 3. NAIP auto-download
    if auto_download_naip:
        downloaded_naip = _download_naip_tiles(naip_request)
    
    # 4. Wetlands auto-download
    if auto_download_wetlands:
        wetlands_path = _download_wetlands_delineations(wetlands_request)
    
    # 5. AOI parsing and polygon extraction
    geom = parse_aoi(aoi)
    polygons = extract_aoi_polygons(geom)
    
    # 6. NAIP footprint filtering
    naip_footprints = _collect_naip_footprints(...)
    
    # 7. Per-AOI Sentinel-2 processing loop
    for output in _iter_aoi_processing(...):
        # 8. Seasonal compositing
        combined21, labels21 = concatenate_seasons(...)
        
        # 9. NAIP reference preparation
        naip_reference_path, _, naip_band_labels = prepare_naip_reference(...)
        
        # 10. Topography auto-download
        if auto_download_topography:
            topography_path = prepare_topography_stack(topo_config)
        
        # 11. Manifest generation
        manifest_path = write_stack_manifest(...)
```

**Issues:**
- Single Responsibility Principle violation
- 200+ line function with nested loops
- Difficult to unit test individual components
- No dependency injection for external services

#### 2. Duplicated CLI Configuration

`train_unet.py` and `test_unet.py` each define similar argument parsing:

```python
# train_unet.py (lines 149-342)
def parse_args() -> argparse.Namespace:
    parser.add_argument("--tile-size", type=int, 
                        default=int(os.getenv("TILE_SIZE", DEFAULT_TILE_SIZE)))
    parser.add_argument("--batch-size", type=int,
                        default=int(os.getenv("UNET_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    # ... 30+ more arguments with env var fallbacks

# test_unet.py (lines 24-136) - similar pattern
def parse_args() -> argparse.Namespace:
    parser.add_argument("--window-size", type=int,
                        default=int(os.getenv("WINDOW_SIZE", DEFAULT_WINDOW_SIZE)))
    parser.add_argument("--batch-size", type=int,
                        default=int(os.getenv("INFER_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
```

**Issues:**
- Environment variable fallback logic repeated everywhere
- No shared configuration schema
- Hard to validate configuration combinations
- Changes require updates in multiple files

#### 3. Normalization Data Flow (Post-Fix)

The current normalization pipeline after recent fixes:

```
TRAINING PATH:
─────────────
Raw Sources → RasterStack.read_window()
              ├─ NAIP:      scale_max=255 → /255 → [0,1]
              ├─ Sentinel:  (already [0,1])
              └─ Topography: band_scaling → per-band norm → [0,1]
           ↓
normalize_stack_array() → clip to [0,1]
           ↓
Write tiles as float32 [0,1]
           ↓
geoai.train_segmentation_model() → /255 internally
           ↓
Model trains on [0, 0.004] range

INFERENCE PATH:
───────────────
Raw Sources → RasterStack.read_window() → [0,1]
           ↓
_prepare_window():
  normalize_stack_array() → [0,1]
  /255.0 → [0, 0.004]  ← Fix applied
           ↓
Model predicts on [0, 0.004] ✓
```

**Residual Issues:**
- The `/255` workaround wastes 99.6% of floating-point precision
- Normalization logic scattered across 3 modules
- No explicit "normalization contract" documentation

---

## Identified Issues and Opportunities

### Critical (High Impact, Low Effort)

| ID | Category | Issue | Location | Impact |
|----|----------|-------|----------|--------|
| C1 | Structural | Monolithic `run_pipeline` function | `compositing.py:622-824` | Hard to test, maintain, extend |
| C2 | Quality | No integration tests for critical paths | `tests/` | Regressions go undetected |
| C3 | Consistency | Normalization workaround uses 0.4% precision | `unet_stream.py:84` | Potential numerical issues |

### Major (Medium Impact, Medium Effort)

| ID | Category | Issue | Location | Impact |
|----|----------|-------|----------|--------|
| M1 | Structural | Duplicated CLI argument parsing | `train_unet.py`, `test_unet.py` | Maintenance burden |
| M2 | Structural | No shared configuration objects | Multiple | Hard to validate configs |
| M3 | Quality | Legacy tool scripts bypass services | `tools/naip_download.py` | Inconsistent behavior |
| M4 | Structural | Tight coupling to `geoai` package | `training/unet.py` | Hard to mock/test |

### Minor (Low Impact, Low Effort)

| ID | Category | Issue | Location | Impact |
|----|----------|-------|----------|--------|
| N1 | Naming | Inconsistent module naming | Various | Cognitive load |
| N2 | Style | Repeated sys.path manipulation | Root CLI shims | Boilerplate |
| N3 | Documentation | `__init__.py` exports incomplete | `src/wetlands_ml_geoai/` | Import confusion |

---

## Proposed Refactoring Plan

### Phase 0: Foundation & Safety Nets (1 week)
**Goal:** Establish testing infrastructure and observability before any structural changes.

#### Tasks

1. **Add pytest configuration and coverage tracking**
   ```ini
   # pyproject.toml additions
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   markers = ["slow: marks tests as slow"]
   
   [tool.coverage.run]
   source = ["src/wetlands_ml_geoai"]
   branch = true
   ```

2. **Create test fixtures for raster data**
   ```python
   # tests/conftest.py
   @pytest.fixture
   def minimal_naip_raster(tmp_path):
       """Create a 64x64 4-band NAIP-like raster."""
       ...
   
   @pytest.fixture
   def mock_stack_manifest(tmp_path, minimal_naip_raster):
       """Create a valid stack manifest pointing to test rasters."""
       ...
   ```

3. **Add integration smoke tests**
   ```python
   # tests/integration/test_pipeline_smoke.py
   def test_sentinel2_cli_accepts_minimal_inputs(tmp_path, mock_stac_responses):
       """Verify CLI wiring without hitting external services."""
       ...
   
   def test_training_cli_generates_tiles(tmp_path, mock_stack_manifest):
       """Verify tile export produces expected files."""
       ...
   ```

4. **Instrument key functions with timing logs**
   ```python
   # Add optional structured logging
   if os.getenv("WETLANDS_DIAGNOSTICS"):
       LOGGER.info("pipeline_event", extra={
           "event": "sentinel_fetch_complete",
           "aoi_index": index,
           "scene_count": len(items),
           "duration_s": elapsed
       })
   ```

**Acceptance Criteria:**
- [ ] `pytest` runs all tests with coverage report
- [ ] At least 3 integration smoke tests pass
- [ ] CI pipeline configured (GitHub Actions or similar)

**Rollback:** N/A (additive changes only)

---

### Phase 1: Extract Shared Configuration Layer (2 weeks)
**Goal:** Create unified configuration objects to replace scattered argument parsing.

#### Tasks

1. **Define configuration dataclasses**

   ```python
   # src/wetlands_ml_geoai/config/models.py
   from dataclasses import dataclass, field
   from pathlib import Path
   from typing import Optional, Sequence
   
   @dataclass(frozen=True)
   class TilingConfig:
       """Shared configuration for tile generation."""
       tile_size: int = 512
       stride: int = 256
       buffer_radius: int = 0
       
       def validate(self) -> None:
           if self.tile_size <= 0:
               raise ValueError("tile_size must be positive")
           if self.stride > self.tile_size:
               raise ValueError("stride cannot exceed tile_size")
   
   @dataclass(frozen=True)
   class ModelConfig:
       """Neural network configuration."""
       architecture: str = "unet"
       encoder_name: str = "resnet34"
       encoder_weights: Optional[str] = "imagenet"
       num_classes: int = 2
       num_channels: Optional[int] = None
   
   @dataclass(frozen=True)
   class TrainingConfig:
       """Full training run configuration."""
       labels_path: Path
       stack_manifest_paths: Sequence[Path] = field(default_factory=list)
       train_raster: Optional[Path] = None
       tiles_dir: Optional[Path] = None
       models_dir: Optional[Path] = None
       
       tiling: TilingConfig = field(default_factory=TilingConfig)
       model: ModelConfig = field(default_factory=ModelConfig)
       
       batch_size: int = 4
       epochs: int = 25
       learning_rate: float = 0.001
       weight_decay: float = 1e-4
       val_split: float = 0.2
       seed: int = 42
       
       def validate(self) -> None:
           if not self.labels_path.exists():
               raise FileNotFoundError(f"Labels not found: {self.labels_path}")
           if not self.train_raster and not self.stack_manifest_paths:
               raise ValueError("Provide train_raster or stack_manifest_paths")
           self.tiling.validate()
   ```

2. **Create configuration builders from CLI/env**

   ```python
   # src/wetlands_ml_geoai/config/cli.py
   def build_training_config(args: argparse.Namespace) -> TrainingConfig:
       """Build TrainingConfig from parsed CLI arguments."""
       return TrainingConfig(
           labels_path=Path(args.labels),
           stack_manifest_paths=_resolve_manifest_paths(args.stack_manifest),
           tiling=TilingConfig(
               tile_size=args.tile_size,
               stride=args.stride,
               buffer_radius=args.buffer,
           ),
           model=ModelConfig(
               architecture=args.architecture,
               encoder_name=args.encoder_name,
               # ...
           ),
           # ...
       )
   
   def add_common_training_args(parser: argparse.ArgumentParser) -> None:
       """Add training-related arguments to parser."""
       group = parser.add_argument_group("Training")
       group.add_argument("--batch-size", type=int, 
                          default=_env_int("UNET_BATCH_SIZE", 4))
       # ...
   
   def _env_int(name: str, default: int) -> int:
       """Get integer from environment with default."""
       val = os.getenv(name)
       return int(val) if val else default
   ```

3. **Add YAML configuration support**

   ```python
   # src/wetlands_ml_geoai/config/yaml_loader.py
   import yaml
   
   def load_training_config(path: Path) -> TrainingConfig:
       """Load TrainingConfig from YAML file."""
       data = yaml.safe_load(path.read_text())
       return TrainingConfig(
           labels_path=Path(data["labels_path"]),
           # ... map YAML structure to config
       )
   ```

4. **Update CLIs to use shared config layer**

   ```python
   # src/wetlands_ml_geoai/train_unet.py (refactored)
   from .config import TrainingConfig, build_training_config, add_common_training_args
   
   def parse_args() -> TrainingConfig:
       parser = argparse.ArgumentParser(...)
       add_common_training_args(parser)
       # Add training-specific args
       args = parser.parse_args()
       config = build_training_config(args)
       config.validate()
       return config
   
   def main() -> None:
       config = parse_args()
       run_training(config)
   ```

**Acceptance Criteria:**
- [ ] `TrainingConfig`, `InferenceConfig`, `SentinelPipelineConfig` dataclasses defined
- [ ] CLI argument parsing delegates to shared builders
- [ ] YAML config loading supported
- [ ] Existing CLI behavior unchanged (backward compatible)

**Rollback:** Keep legacy `parse_args` as fallback behind `WETLANDS_LEGACY_CLI=1` flag

---

### Phase 2: Decompose Sentinel-2 Compositing Module (3 weeks)
**Goal:** Split the 995-line `compositing.py` into focused, testable modules.

#### Proposed Module Structure

```
sentinel2/
├── __init__.py
├── cli.py                  # CLI entry point (unchanged)
├── aoi.py                  # AOI parsing and geometry (NEW, ~100 LOC)
├── stac_client.py          # STAC queries and band stacking (NEW, ~200 LOC)
├── cloud_masking.py        # SCL mask building (NEW, ~80 LOC)
├── seasonal.py             # Seasonal median computation (NEW, ~150 LOC)
├── naip_integration.py     # NAIP mosaicking and clipping (NEW, ~150 LOC)
├── pipeline.py             # Orchestration (REFACTORED, ~200 LOC)
├── manifests.py            # Manifest I/O (unchanged)
└── progress.py             # Progress tracking (unchanged)
```

#### Tasks

1. **Extract AOI parsing module**

   ```python
   # src/wetlands_ml_geoai/sentinel2/aoi.py
   """AOI parsing and geometry utilities."""
   
   from shapely.geometry.base import BaseGeometry
   from shapely.geometry import Polygon, MultiPolygon, box
   
   def parse_aoi(aoi: str) -> BaseGeometry:
       """Parse AOI from various input formats.
       
       Supported formats:
       - File path to GeoPackage/Shapefile
       - GeoJSON string or file
       - WKT string
       - Bounding box as comma-separated string or list
       """
       ...
   
   def buffer_in_meters(geom: BaseGeometry, buffer_m: float) -> BaseGeometry:
       """Buffer geometry in meters using UTM projection."""
       ...
   
   def extract_polygons(aoi: BaseGeometry, buffer_m: float = 0) -> list[Polygon]:
       """Extract individual polygons from AOI geometry."""
       ...
   ```

2. **Extract STAC client module**

   ```python
   # src/wetlands_ml_geoai/sentinel2/stac_client.py
   """Sentinel-2 STAC API interactions."""
   
   from pystac_client import Client
   from pystac import Item
   import xarray as xr
   
   @dataclass
   class StacConfig:
       url: str = "https://earth-search.aws.element84.com/v1"
       collection: str = "sentinel-2-l2a"
       bands: tuple[str, ...] = ("B03", "B04", "B05", "B06", "B08", "B11", "B12")
   
   class Sentinel2Client:
       """Encapsulates STAC API interactions for Sentinel-2."""
       
       def __init__(self, config: StacConfig = StacConfig()):
           self._config = config
           self._client: Client | None = None
       
       def connect(self) -> None:
           self._client = Client.open(self._config.url)
       
       def fetch_items(
           self,
           geometry: BaseGeometry,
           season: str,
           years: Sequence[int],
           cloud_cover: float = 60.0,
       ) -> list[Item]:
           """Fetch Sentinel-2 items matching criteria."""
           ...
       
       def stack_bands(
           self,
           items: Sequence[Item],
           bounds: tuple[float, float, float, float],
       ) -> xr.DataArray:
           """Create band stack from items."""
           ...
       
       def stack_scl(
           self,
           items: Sequence[Item],
           bounds: tuple[float, float, float, float],
       ) -> xr.DataArray:
           """Create SCL (scene classification) stack."""
           ...
   ```

3. **Extract cloud masking module**

   ```python
   # src/wetlands_ml_geoai/sentinel2/cloud_masking.py
   """Cloud and shadow masking utilities."""
   
   import xarray as xr
   from skimage.morphology import binary_dilation
   
   SCL_MASK_VALUES = {3, 8, 9, 10, 11}  # Clouds, shadows, etc.
   
   def build_scl_mask(
       scl: xr.DataArray,
       dilation_pixels: int = 0,
   ) -> xr.DataArray:
       """Build binary mask from SCL band.
       
       Args:
           scl: Scene Classification Layer stack
           dilation_pixels: Pixels to expand cloud mask
       
       Returns:
           Boolean mask (True = clear pixel)
       """
       ...
   ```

4. **Extract seasonal compositing module**

   ```python
   # src/wetlands_ml_geoai/sentinel2/seasonal.py
   """Seasonal composite generation."""
   
   import xarray as xr
   
   SEASON_WINDOWS = {
       "SPR": (3, 1, 5, 31),
       "SUM": (6, 1, 8, 31),
       "FAL": (9, 1, 11, 30),
   }
   
   def compute_seasonal_median(
       items: Sequence[Item],
       season: str,
       bounds: tuple[float, float, float, float],
       min_clear_obs: int,
       mask_dilation: int,
       stac_client: Sentinel2Client,
   ) -> tuple[xr.DataArray, xr.DataArray]:
       """Compute cloud-free seasonal median composite.
       
       Returns:
           Tuple of (median composite, valid observation counts)
       """
       ...
   
   def concatenate_seasons(
       seasonal_data: dict[str, xr.DataArray],
       season_order: Sequence[str],
   ) -> tuple[xr.DataArray, list[str]]:
       """Concatenate seasonal composites into multi-season stack."""
       ...
   ```

5. **Create focused orchestrator**

   ```python
   # src/wetlands_ml_geoai/sentinel2/pipeline.py
   """Sentinel-2 pipeline orchestration."""
   
   from .aoi import parse_aoi, extract_polygons
   from .stac_client import Sentinel2Client, StacConfig
   from .seasonal import compute_seasonal_median, concatenate_seasons
   from .naip_integration import prepare_naip_reference, clip_to_polygon
   from .manifests import write_stack_manifest
   
   @dataclass
   class PipelineConfig:
       """Configuration for full Sentinel-2 pipeline run."""
       aoi: str
       years: Sequence[int]
       output_dir: Path
       seasons: Sequence[str] = ("SPR", "SUM", "FAL")
       cloud_cover: float = 60.0
       min_clear_obs: int = 3
       stac: StacConfig = field(default_factory=StacConfig)
       # ... other config
   
   class Sentinel2Pipeline:
       """Orchestrates Sentinel-2 seasonal compositing."""
       
       def __init__(self, config: PipelineConfig):
           self._config = config
           self._stac_client = Sentinel2Client(config.stac)
       
       def run(self) -> list[Path]:
           """Execute pipeline and return manifest paths."""
           self._stac_client.connect()
           
           geom = parse_aoi(self._config.aoi)
           polygons = extract_polygons(geom)
           
           manifest_paths = []
           for index, polygon in enumerate(polygons, start=1):
               manifest = self._process_aoi(index, polygon)
               if manifest:
                   manifest_paths.append(manifest)
           
           return manifest_paths
       
       def _process_aoi(self, index: int, polygon: Polygon) -> Path | None:
           """Process single AOI polygon."""
           ...
   ```

**Acceptance Criteria:**
- [ ] `compositing.py` reduced to <300 lines (orchestration only)
- [ ] Each extracted module has >80% test coverage
- [ ] CLI produces identical outputs on regression dataset
- [ ] No circular imports between new modules

**Rollback:** Keep `compositing.py` functional; new modules can coexist during transition

---

### Phase 3: Unify Data Acquisition Services (2 weeks)
**Goal:** Centralize NAIP, wetlands, and topography download logic with consistent interfaces.

#### Tasks

1. **Create abstract data service interface**

   ```python
   # src/wetlands_ml_geoai/services/base.py
   from abc import ABC, abstractmethod
   
   class DataService(ABC):
       """Base class for data acquisition services."""
       
       @abstractmethod
       def download(self, request: Any) -> Path:
           """Download data and return path to result."""
           ...
       
       @abstractmethod
       def is_cached(self, request: Any) -> bool:
           """Check if data is already downloaded."""
           ...
   ```

2. **Implement NAIP service**

   ```python
   # src/wetlands_ml_geoai/services/naip.py
   from .base import DataService
   
   @dataclass
   class NaipDownloadRequest:
       aoi: BaseGeometry
       output_dir: Path
       year: int | None = None
       max_items: int | None = None
       overwrite: bool = False
       target_resolution: float | None = None
   
   class NaipService(DataService):
       """Service for downloading and preparing NAIP imagery."""
       
       def __init__(self, retry_config: RetryConfig = RetryConfig()):
           self._retry = retry_config
       
       def download(self, request: NaipDownloadRequest) -> list[Path]:
           """Download NAIP tiles for AOI."""
           ...
       
       def mosaic(self, paths: Sequence[Path], output: Path) -> Path:
           """Create mosaic from multiple tiles."""
           ...
       
       def collect_footprints(self, paths: Sequence[Path]) -> list[tuple[Path, Polygon]]:
           """Get footprints for tile filtering."""
           ...
   ```

3. **Implement wetlands service**

   ```python
   # src/wetlands_ml_geoai/services/wetlands.py
   class WetlandsService(DataService):
       """Service for downloading NWI wetlands data."""
       
       NWI_API_URL = "https://fwspublicservices.wim.usgs.gov/..."
       
       def download(self, request: WetlandsDownloadRequest) -> Path:
           """Download and clip wetlands to AOI."""
           ...
   ```

4. **Implement topography service**

   ```python
   # src/wetlands_ml_geoai/services/topography.py
   class TopographyService(DataService):
       """Service for downloading and processing DEMs."""
       
       def download(self, request: TopographyDownloadRequest) -> list[Path]:
           """Download 3DEP DEM tiles."""
           ...
       
       def compute_derivatives(
           self,
           dem_paths: Sequence[Path],
           config: TopographyStackConfig,
       ) -> Path:
           """Compute slope, TPI, depression depth."""
           ...
   ```

5. **Remove legacy tool scripts or make them thin wrappers**

   ```python
   # tools/naip_download.py (refactored)
   """CLI for NAIP downloads - delegates to NaipService."""
   
   from wetlands_ml_geoai.services.naip import NaipService, NaipDownloadRequest
   
   def main():
       args = parse_args()
       service = NaipService()
       request = NaipDownloadRequest(
           aoi=load_aoi(args.aoi),
           output_dir=Path(args.output_dir),
           year=args.year,
       )
       paths = service.download(request)
       print(f"Downloaded {len(paths)} tiles")
   ```

**Acceptance Criteria:**
- [ ] All data acquisition uses `*Service` classes
- [ ] Retry/backoff policies configurable
- [ ] `tools/naip_download.py` delegates to `NaipService`
- [ ] Unit tests mock external APIs

**Rollback:** Keep `data_acquisition.py` functions as internal implementations

---

### Phase 4: Improve Training/Inference Architecture (2 weeks)
**Goal:** Reduce coupling between CLI, configuration, and execution layers.

#### Tasks

1. **Create training service layer**

   ```python
   # src/wetlands_ml_geoai/training/service.py
   from ..config import TrainingConfig
   
   class TrainingService:
       """Orchestrates model training workflow."""
       
       def __init__(
           self,
           config: TrainingConfig,
           geoai_adapter: GeoaiAdapter = DefaultGeoaiAdapter(),
       ):
           self._config = config
           self._geoai = geoai_adapter
       
       def run(self) -> TrainingResult:
           """Execute full training workflow."""
           tiles = self._prepare_tiles()
           model_path = self._train_model(tiles)
           return TrainingResult(model_path=model_path, metrics=...)
       
       def _prepare_tiles(self) -> TilesDirectory:
           """Export and rewrite tiles from manifests."""
           ...
       
       def _train_model(self, tiles: TilesDirectory) -> Path:
           """Invoke geoai training."""
           ...
   ```

2. **Create geoai adapter for testability**

   ```python
   # src/wetlands_ml_geoai/adapters/geoai.py
   from abc import ABC, abstractmethod
   
   class GeoaiAdapter(ABC):
       """Abstract interface for geoai operations."""
       
       @abstractmethod
       def export_tiles(self, raster: Path, output: Path, labels: Path, **kwargs) -> None:
           ...
       
       @abstractmethod
       def train_model(self, images: Path, labels: Path, output: Path, **kwargs) -> None:
           ...
       
       @abstractmethod
       def run_inference(self, raster: Path, model: Path, output: Path, **kwargs) -> None:
           ...
   
   class DefaultGeoaiAdapter(GeoaiAdapter):
       """Production implementation using geoai."""
       
       def export_tiles(self, **kwargs) -> None:
           import geoai
           geoai.export_geotiff_tiles(**kwargs)
       
       # ...
   
   class MockGeoaiAdapter(GeoaiAdapter):
       """Test implementation that records calls."""
       
       def __init__(self):
           self.calls = []
       
       def export_tiles(self, **kwargs) -> None:
           self.calls.append(("export_tiles", kwargs))
   ```

3. **Unify tile export logic**

   ```python
   # src/wetlands_ml_geoai/training/tile_exporter.py
   class TileExporter:
       """Handles tile extraction and manifest rewriting."""
       
       def __init__(
           self,
           config: TilingConfig,
           geoai: GeoaiAdapter,
       ):
           self._config = config
           self._geoai = geoai
       
       def export_from_manifests(
           self,
           manifests: Sequence[StackManifest],
           labels: Path,
           output_dir: Path,
       ) -> TilesDirectory:
           """Export tiles from stack manifests with proper rewriting."""
           with StagingDirectory(output_dir) as staging:
               for idx, manifest in enumerate(manifests):
                   self._export_single_manifest(manifest, labels, staging, idx)
               return staging.finalize()
   
   @contextmanager
   def StagingDirectory(base: Path):
       """Context manager for atomic tile export."""
       staging = base / f"_staging_{time.time_ns()}"
       staging.mkdir(parents=True)
       try:
           yield StagingHelper(staging)
       except Exception:
           shutil.rmtree(staging, ignore_errors=True)
           raise
   ```

4. **Create inference service layer**

   ```python
   # src/wetlands_ml_geoai/inference/service.py
   class InferenceService:
       """Orchestrates model inference workflow."""
       
       def __init__(
           self,
           config: InferenceConfig,
           model_loader: ModelLoader = DefaultModelLoader(),
       ):
           self._config = config
           self._model_loader = model_loader
       
       def run(self) -> InferenceResult:
           """Execute inference on configured input."""
           model = self._model_loader.load(self._config.model_path)
           
           if self._config.stack_manifest:
               return self._infer_from_manifest(model)
           else:
               return self._infer_from_raster(model)
   ```

**Acceptance Criteria:**
- [ ] Training/inference use service classes
- [ ] `GeoaiAdapter` allows mock injection in tests
- [ ] `TileExporter` handles all tile preparation
- [ ] Existing CLI behavior preserved

**Rollback:** Services can wrap existing functions initially

---

### Phase 5: Comprehensive Testing & Documentation (2 weeks)
**Goal:** Achieve >60% test coverage and document the refactored architecture.

#### Tasks

1. **Add unit tests for all new modules**
   - `aoi.py`: Geometry parsing edge cases
   - `stac_client.py`: Query building (mock HTTP)
   - `cloud_masking.py`: Mask dilation logic
   - `seasonal.py`: Median computation
   - `config/`: Validation rules
   - `services/`: Download/retry logic

2. **Add integration tests**
   ```python
   # tests/integration/test_end_to_end.py
   def test_sentinel_to_training_pipeline(
       tmp_path,
       mock_stac_server,
       minimal_naip_fixture,
   ):
       """Full pipeline from Sentinel download to tile export."""
       ...
   ```

3. **Add performance benchmarks**
   ```python
   # tests/benchmarks/test_raster_stack.py
   def test_read_window_performance(large_manifest, benchmark):
       """Ensure read_window stays under 100ms for 512x512."""
       stack = RasterStack(large_manifest)
       result = benchmark(stack.read_window, Window(0, 0, 512, 512))
       assert result.shape == (25, 512, 512)
   ```

4. **Update documentation**
   - `docs/architecture_overview.md`: New module structure
   - `docs/configuration_guide.md`: Config objects and YAML
   - `docs/developer_guide.md`: How to extend services
   - `docs/api_reference.md`: Public interfaces

5. **Create architecture decision records (ADRs)**
   - ADR-001: Configuration Object Pattern
   - ADR-002: Service Layer Architecture
   - ADR-003: Normalization Pipeline

**Acceptance Criteria:**
- [ ] Test coverage >60% for `sentinel2`, `training`, `inference`, `services`
- [ ] All public APIs documented with docstrings
- [ ] Architecture diagrams in `docs/`
- [ ] CI runs tests on every PR

---

## Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| External API changes (STAC, 3DEP, NWI) | Medium | High | Adapters with version pinning; contract tests |
| geoai package breaking changes | Medium | High | `GeoaiAdapter` abstraction; pin version |
| Performance regression from refactoring | Low | Medium | Benchmark tests; profile before/after |
| Team bandwidth constraints | High | Medium | Phases are independent; can pause/resume |
| Breaking existing workflows | Medium | High | Feature flags; backward-compatible defaults |

### Rollback Strategy

Each phase maintains a rollback mechanism:

1. **Phase 0-1:** New code coexists with legacy; toggle via env vars
2. **Phase 2:** Old `compositing.py` remains functional during transition
3. **Phase 3:** Legacy `data_acquisition.py` functions still importable
4. **Phase 4:** Services wrap existing implementations initially

---

## Testing Strategy

### Unit Tests (Per Module)

| Module | Test Focus | Mock Dependencies |
|--------|------------|-------------------|
| `sentinel2/aoi.py` | Geometry parsing, buffering | None |
| `sentinel2/stac_client.py` | Query building, response parsing | HTTP responses |
| `sentinel2/cloud_masking.py` | Mask logic, dilation | None |
| `sentinel2/seasonal.py` | Median computation, concatenation | xarray fixtures |
| `config/*.py` | Validation rules | None |
| `services/naip.py` | Download, mosaic, footprints | geoai.download, rasterio |
| `training/tile_exporter.py` | Export, rewrite, staging | GeoaiAdapter |
| `inference/service.py` | End-to-end flow | ModelLoader |

### Integration Tests

```python
# tests/integration/conftest.py
@pytest.fixture
def mock_stac_server(httpserver):
    """Fake STAC server with canned responses."""
    httpserver.expect_request("/search").respond_with_json(CANNED_STAC_RESPONSE)
    yield httpserver.url_for("")

@pytest.fixture
def e2e_test_data(tmp_path):
    """Create minimal dataset for end-to-end tests."""
    # Create small NAIP, Sentinel, labels rasters
    ...
```

### Performance Tests

```python
# tests/benchmarks/test_critical_paths.py
@pytest.mark.benchmark
def test_raster_stack_read_512(benchmark, large_manifest):
    """read_window for 512x512 should be <100ms."""
    ...

@pytest.mark.benchmark  
def test_normalize_stack_array(benchmark, large_array):
    """normalize_stack_array for 25x1000x1000 should be <50ms."""
    ...
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Max function length | 200+ lines | <50 lines | `radon` analysis |
| Cyclomatic complexity (max) | >20 | <10 | `radon cc` |
| Test coverage | ~15% | >60% | `pytest-cov` |
| CLI argument duplication | 30+ duplicated | 0 | Code review |
| Module coupling | High | Low | Import analysis |
| Integration test count | 0 | >10 | `pytest` markers |
| Documentation completeness | Partial | Full | Checklist review |

---

## Implementation Timeline

```
Week 1-2:   Phase 0 - Foundation (testing infrastructure)
Week 3-4:   Phase 1 - Configuration layer
Week 5-7:   Phase 2 - Sentinel-2 decomposition
Week 8-9:   Phase 3 - Data services
Week 10-11: Phase 4 - Training/Inference architecture
Week 12-13: Phase 5 - Testing & documentation
Week 14:    Final review and cleanup
```

**Total Estimated Duration:** 14 weeks (3.5 months)

---

## Appendix A: File Changes Summary

### New Files to Create

```
src/wetlands_ml_geoai/
├── config/
│   ├── __init__.py
│   ├── models.py           # Config dataclasses
│   ├── cli.py              # Argument builders
│   └── yaml_loader.py      # YAML support
├── services/
│   ├── __init__.py
│   ├── base.py             # Abstract interface
│   ├── naip.py             # NAIP service
│   ├── wetlands.py         # Wetlands service
│   └── topography.py       # Topography service
├── adapters/
│   ├── __init__.py
│   └── geoai.py            # geoai abstraction
├── sentinel2/
│   ├── aoi.py              # AOI utilities
│   ├── stac_client.py      # STAC interactions
│   ├── cloud_masking.py    # Mask building
│   ├── seasonal.py         # Compositing
│   ├── naip_integration.py # NAIP handling
│   └── pipeline.py         # Orchestration
├── training/
│   ├── service.py          # Training orchestration
│   └── tile_exporter.py    # Tile preparation
└── inference/
    └── service.py          # Inference orchestration

tests/
├── conftest.py             # Shared fixtures
├── unit/                   # Per-module tests
├── integration/            # End-to-end tests
└── benchmarks/             # Performance tests
```

### Files to Modify

| File | Changes |
|------|---------|
| `train_unet.py` | Use `TrainingConfig`, delegate to service |
| `test_unet.py` | Use `InferenceConfig`, delegate to service |
| `sentinel2/compositing.py` | Reduced to thin wrapper |
| `sentinel2/cli.py` | Use `PipelineConfig` |
| `data_acquisition.py` | Delegate to services |
| `training/unet.py` | Use `TrainingService` |
| `inference/unet_stream.py` | Use `InferenceService` |

### Files to Delete/Deprecate

| File | Action |
|------|--------|
| `tools/naip_download.py` | Replace with thin CLI wrapper |
| Root CLI shims | Consider setuptools entry points |

---

## Appendix B: Code Quality Metrics (Current State)

### Cyclomatic Complexity (Top 10)

```
radon cc src/ -a -s -n C

sentinel2/compositing.py:run_pipeline - C (24)
sentinel2/compositing.py:_iter_aoi_processing - C (18)
train_unet.py:parse_args - B (12)
training/unet.py:train_unet - C (15)
stacking.py:read_window - B (11)
inference/unet_stream.py:_stream_inference - B (10)
```

### Lines of Code by Module

```
cloc src/wetlands_ml_geoai/

sentinel2/compositing.py    995
train_unet.py               427
stacking.py                 383
training/unet.py            345
inference/unet_stream.py    313
training/sampling.py        297
training/extraction.py      276
data_acquisition.py         253
```

### Import Dependency Analysis

```
pydeps src/wetlands_ml_geoai --cluster

Clusters:
  - sentinel2 (self-contained)
  - training (depends on stacking)
  - inference (depends on stacking)
  - stacking (core, no deps on other modules)
  - data_acquisition (depends on geoai)
  - topography (self-contained)
```

---

*This document should be reviewed quarterly and updated as refactoring progresses.*

