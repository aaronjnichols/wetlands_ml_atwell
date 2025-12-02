# Wetlands ML Codex - Comprehensive Refactoring Plan

**Date:** 2025-12-02
**Project:** wetlands_ml_codex
**Analysis Type:** Full architectural review and refactoring roadmap

---

## Executive Summary

This document presents a comprehensive refactoring plan for the Wetlands ML Codex project. The analysis identifies **22 distinct issues** across 5 severity categories, with **3 critical issues** requiring immediate attention, **6 major architectural improvements**, and **13 minor enhancements** for maintainability.

### Key Findings

| Category | Count | Impact |
|----------|-------|--------|
| Critical Bugs | 3 | Data corruption, broken inference paths |
| Architectural Issues | 6 | Maintainability, testability |
| Code Smells | 8 | Technical debt accumulation |
| Testing Gaps | 5 | Confidence in production deployments |
| Documentation Gaps | 2 | Onboarding, knowledge transfer |

### Recommended Priority

1. **Phase 1 (Immediate):** Fix critical normalization/scaling issues
2. **Phase 2 (Short-term):** Split monolithic compositing module
3. **Phase 3 (Medium-term):** Consolidate utilities and add comprehensive tests
4. **Phase 4 (Long-term):** Architectural cleanup and configuration system

---

## Current State Analysis

### Project Structure

```
wetlands_ml_codex/
├── src/wetlands_ml_geoai/          # Main package (4,269 LOC)
│   ├── sentinel2/                   # Sentinel-2 processing (1,258 LOC)
│   │   ├── compositing.py          # 995 LOC - CRITICAL: Too large
│   │   ├── manifests.py            # 141 LOC
│   │   ├── progress.py             # 81 LOC
│   │   └── cli.py                  # 29 LOC
│   ├── topography/                  # DEM processing (625 LOC)
│   │   ├── pipeline.py             # 105 LOC
│   │   ├── processing.py           # 203 LOC
│   │   ├── download.py             # 204 LOC
│   │   ├── config.py               # 35 LOC
│   │   └── cli.py                  # 70 LOC
│   ├── inference/                   # Model inference (352 LOC)
│   │   ├── unet_stream.py          # 313 LOC
│   │   └── common.py               # 39 LOC
│   ├── training/                    # Training orchestration (344 LOC)
│   │   └── unet.py                 # 344 LOC
│   ├── stacking.py                 # 383 LOC - Core data structures
│   ├── train_unet.py               # 426 LOC - CLI entry point
│   ├── test_unet.py                # 230 LOC - CLI entry point
│   ├── data_acquisition.py         # 252 LOC
│   ├── validate_seasonal_pixels.py # 331 LOC
│   ├── tiling.py                   # 61 LOC
│   └── utils/                      # EMPTY - Opportunity
├── tests/                          # Test suite (264 LOC, ~6% coverage)
│   ├── test_inference_cli.py       # 56 LOC, 3 tests
│   ├── test_topography_pipeline.py # 119 LOC, 3 tests
│   └── test_training_cli.py        # 89 LOC, 2+ tests
├── configs/                        # Configuration (minimal)
│   └── train_unet_example.yaml     # Placeholder
└── docs/                           # Documentation
```

### Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CORE LAYER                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  stacking.py                                                   │  │
│  │  - FLOAT_NODATA, StackGrid, StackSource, StackManifest        │  │
│  │  - RasterStack, normalize_stack_array, load_manifest          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ sentinel2/      │     │ topography/         │     │ inference/      │
│ compositing.py  │◄────│ pipeline.py         │     │ unet_stream.py  │
│ (995 LOC!)      │     │ processing.py       │     │ (313 LOC)       │
│                 │     │ download.py         │     └────────┬────────┘
└────────┬────────┘     └─────────────────────┘              │
         │                                                    │
         ▼                                                    ▼
┌─────────────────────┐                           ┌─────────────────┐
│ training/unet.py    │                           │ test_unet.py    │
│ (344 LOC)           │                           │ (CLI entry)     │
└─────────────────────┘                           └─────────────────┘
```

---

## Identified Issues and Opportunities

### CRITICAL (C1-C3): Immediate Action Required

#### C1: Normalization Precision Loss [PARTIALLY FIXED]

**Location:** `stacking.py:290-344`, `inference/unet_stream.py:65-85`

**Current State:**
The `/255.0` division has been added to align inference with training, but this creates a precision loss issue:
- Training tiles are normalized to `[0, 1]` then divided by 255 by geoai → `[0, 0.00392]`
- Inference now matches this behavior
- **Problem:** Model uses only 0.4% of available float32 precision

**Evidence:**
```python
# inference/unet_stream.py:83-84
array = normalize_stack_array(array, nodata_value)
array = array / 255.0  # Workaround for geoai's internal normalization
```

**Impact:** Suboptimal model training and inference precision.

**Recommended Fix:**
Option A (Short-term): Document the precision loss as acceptable trade-off.
Option B (Long-term): Create custom `NormalizedSegmentationDataset` class that skips the `/255` step for pre-normalized data.

---

#### C2: Direct Raster Inference Misalignment

**Location:** `inference/unet_stream.py:262-309`

**Current State:**
When running `infer_raster()` (not manifest-based), raw uint8 values are read and processed:

```python
# Line 292
read_window=lambda win: src.read(window=win),  # Raw uint8 values!
```

For uint8 NAIP data (0-255):
1. Values pass to `_prepare_window()`
2. `normalize_stack_array()` clips to [0,1] → all values >1 become 1.0
3. `/255` → everything becomes ~0.00392
4. Model sees near-uniform input

**Impact:** Direct raster inference produces meaningless predictions.

**Recommended Fix:**
Add dtype-aware scaling in `infer_raster()`:
```python
def infer_raster(...):
    with rasterio.open(raster_path) as src:
        scale_max = 255.0 if src.dtypes[0] == 'uint8' else None

        def read_and_scale(win):
            data = src.read(window=win).astype('float32')
            if scale_max:
                data = data / scale_max
            return data

        elapsed = _stream_inference(
            read_window=read_and_scale,
            ...
        )
```

---

#### C3: Test Suite Inconsistency [FIXED]

**Location:** `tests/test_inference_cli.py:30-56`

**Current State:** The test now correctly includes the `/255.0` step (line 52):
```python
expected = normalize_stack_array(expected, FLOAT_NODATA, warn_on_clip=False)
expected = expected / 255.0  # This step matches geoai's training normalization
```

**Status:** ✅ Fixed in recent commits.

---

### MAJOR (M1-M6): Architectural Improvements

#### M1: Monolithic Compositing Module (995 LOC)

**Location:** `sentinel2/compositing.py`

**Problem:** Single file handles 10+ distinct responsibilities:
- AOI parsing (`parse_aoi`, `extract_aoi_polygons`, `_buffer_in_meters`)
- NAIP handling (`collect_naip_sources`, `prepare_naip_reference`, `_clip_raster_to_polygon`)
- STAC queries (`fetch_items`, client management)
- Band stacking (`stack_bands`, `stack_scl`)
- Cloud masking (`build_mask`, `SCL_MASK_VALUES`)
- Seasonal compositing (`seasonal_median`, `concatenate_seasons`)
- Manifest generation
- CLI handling

**Metrics:**
- 995 lines of code
- 19 public/private functions
- Cyclomatic complexity: High (nested generators, deep nesting)

**Recommended Refactoring:**

```
sentinel2/
├── __init__.py
├── aoi.py              # 150 LOC - AOI parsing, geometry handling
│   └── parse_aoi(), extract_aoi_polygons(), _buffer_in_meters()
├── naip.py             # 200 LOC - NAIP mosaicking and clipping
│   └── collect_naip_sources(), prepare_naip_reference()
├── stac.py             # 100 LOC - STAC client and queries
│   └── fetch_items(), get_stac_client()
├── bands.py            # 120 LOC - Band stacking logic
│   └── stack_bands(), stack_scl(), SENTINEL_BANDS
├── masking.py          # 80 LOC - Cloud/SCL masking
│   └── build_mask(), SCL_MASK_VALUES
├── seasonal.py         # 150 LOC - Seasonal compositing
│   └── seasonal_median(), concatenate_seasons(), SeasonConfig
├── pipeline.py         # 200 LOC - Orchestration ONLY
│   └── run_pipeline(), _iter_aoi_processing()
├── cli.py              # 100 LOC - CLI argument parsing
│   └── configure_parser(), run_from_args()
├── manifests.py        # 141 LOC - (existing)
└── progress.py         # 81 LOC - (existing)
```

**Effort:** 4-6 hours
**Risk:** Medium - Requires careful import management

---

#### M2: Empty Utils Module

**Location:** `src/wetlands_ml_geoai/utils/__init__.py` (0 LOC)

**Opportunity:** Consolidate repeated patterns:

| Pattern | Occurrences | Proposed Utility |
|---------|-------------|------------------|
| `Path.expanduser().resolve()` | 15+ | `utils.resolve_path()` |
| `path.mkdir(parents=True, exist_ok=True)` | 21+ | `utils.ensure_dir()` |
| GeoTIFF profile construction | 5 | `utils.create_gtiff_profile()` |
| Logging setup | 4 | `utils.setup_logging()` |

**Proposed Structure:**
```
utils/
├── __init__.py
├── path.py          # Path resolution, directory creation
├── rasterio.py      # GeoTIFF profile, raster utilities
└── logging.py       # Centralized logging configuration
```

**Effort:** 2-3 hours
**Risk:** Low

---

#### M3: Scattered Normalization Logic

**Locations:**
- `stacking.py:290-344` - `normalize_stack_array()`
- `inference/unet_stream.py:65-85` - `_prepare_window()`
- `training/unet.py` - relies on geoai internals

**Problem:** Three separate implementations with subtle differences.

**Recommended Consolidation:**

```python
# stacking.py - Single source of truth

class NormalizationConfig:
    """Configuration for data normalization."""
    nodata_value: float = FLOAT_NODATA
    clip_range: Tuple[float, float] = (0.0, 1.0)
    geoai_compatibility: bool = True  # Apply /255 for geoai
    warn_on_clip: bool = True

def normalize_for_inference(
    data: np.ndarray,
    config: NormalizationConfig = NormalizationConfig(),
) -> np.ndarray:
    """Normalize data for model inference, matching training normalization."""
    result = normalize_stack_array(data, config.nodata_value, config.warn_on_clip)
    if config.geoai_compatibility:
        result = result / 255.0
    return result
```

**Effort:** 2-3 hours
**Risk:** Low-Medium (requires testing)

---

#### M4: No Configuration System

**Current State:**
- `configs/train_unet_example.yaml` is a placeholder
- Parameters scattered across CLI arguments
- Magic constants embedded in code

**Magic Constants Found:**
```python
# sentinel2/compositing.py
SCL_MASK_VALUES = {3, 8, 9, 10, 11}  # No explanation
SENTINEL_SCALE_FACTOR = 1 / 10000    # Why 10000?
SEASON_WINDOWS = {                    # Hardcoded seasons
    "SPR": (3, 1, 5, 31),
    "SUM": (6, 1, 8, 31),
    "FAL": (9, 1, 11, 30),
}

# stacking.py
FLOAT_NODATA = -9999.0  # Why this value?
```

**Recommended Configuration System:**

```yaml
# configs/sentinel2.yaml
sentinel2:
  collection: "sentinel-2-l2a"
  bands: ["B03", "B04", "B05", "B06", "B08", "B11", "B12"]
  scale_factor: 0.0001  # 1/10000
  scl_mask_values:
    cloud_shadow: 3
    cloud_medium: 8
    cloud_high: 9
    cirrus: 10
    snow_ice: 11
  seasons:
    spring:
      start_month: 3
      start_day: 1
      end_month: 5
      end_day: 31
    summer:
      start_month: 6
      start_day: 1
      end_month: 8
      end_day: 31
    fall:
      start_month: 9
      start_day: 1
      end_month: 11
      end_day: 30

# configs/training.yaml
training:
  architecture: "unet"
  encoder: "resnet34"
  encoder_weights: "imagenet"
  tile_size: 512
  stride: 256
  batch_size: 4
  epochs: 25
  learning_rate: 0.001
  weight_decay: 0.0001
  validation_split: 0.2

# configs/inference.yaml
inference:
  window_size: 512
  overlap: 128
  probability_threshold: 0.5
  num_classes: 2
```

**Effort:** 3-4 hours
**Risk:** Low

---

#### M5: Manifest Schema Validation

**Problem:** Manifests are JSON files with no schema validation. Invalid manifests fail at runtime with cryptic errors.

**Current Manifest Structure:**
```json
{
  "grid": {
    "crs": "EPSG:32618",
    "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 10.0],
    "width": 1000,
    "height": 1000,
    "nodata": -9999.0
  },
  "sources": [
    {
      "type": "naip",
      "path": "/path/to/naip.tif",
      "band_labels": ["R", "G", "B", "NIR"],
      "scale_max": 255
    },
    {
      "type": "topography",
      "path": "/path/to/topo.tif",
      "band_labels": ["Slope", "TPI_small", "TPI_large", "DepressionDepth"],
      "band_scaling": {
        "Slope": [0, 90],
        "TPI_small": [-50, 50],
        "TPI_large": [-100, 100],
        "DepressionDepth": [0, 50]
      }
    }
  ]
}
```

**Recommended Solution:**

```python
# stacking.py - Add Pydantic or dataclass validation

from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Tuple

class ManifestGridSchema(BaseModel):
    crs: Optional[str]
    transform: Tuple[float, float, float, float, float, float]
    width: int
    height: int
    nodata: float = FLOAT_NODATA

    @validator('width', 'height')
    def positive_dimension(cls, v):
        if v <= 0:
            raise ValueError('dimensions must be positive')
        return v

class ManifestSourceSchema(BaseModel):
    type: str
    path: str
    band_labels: List[str]
    scale_max: Optional[float] = None
    scale_min: Optional[float] = None
    band_scaling: Optional[Dict[str, Tuple[float, float]]] = None
    nodata: Optional[float] = None
    resample: str = "bilinear"

    @validator('band_scaling')
    def validate_band_scaling(cls, v, values):
        if v:
            for label, (min_val, max_val) in v.items():
                if min_val >= max_val:
                    raise ValueError(f"Invalid scaling for {label}: min >= max")
        return v

class ManifestSchema(BaseModel):
    grid: ManifestGridSchema
    sources: List[ManifestSourceSchema]
```

**Effort:** 2-3 hours
**Risk:** Low

---

#### M6: Training/Inference Path Divergence

**Problem:** Training and inference use different code paths with subtle normalization differences.

**Training Path:**
```
NAIP/Sentinel → RasterStack.read_window() → normalize_stack_array()
    → Write tiles as float32 [0,1] → geoai loads → /255 → Model [0, 0.00392]
```

**Inference Path (Manifest):**
```
NAIP/Sentinel → RasterStack.read_window() → _prepare_window()
    → normalize_stack_array() → /255 → Model [0, 0.00392]
```

**Inference Path (Raster):**
```
Raw raster → src.read() → _prepare_window()
    → normalize_stack_array() [CLIPS!] → /255 → Model [garbage]
```

**Recommended Unified Pipeline:**

```python
# New file: src/wetlands_ml_geoai/preprocessing/pipeline.py

class DataPipeline:
    """Unified preprocessing for training and inference."""

    def __init__(
        self,
        manifest: Optional[StackManifest] = None,
        raster_path: Optional[Path] = None,
        normalization_config: NormalizationConfig = None,
    ):
        if manifest is None and raster_path is None:
            raise ValueError("Must provide manifest or raster_path")
        self.manifest = manifest
        self.raster_path = raster_path
        self.config = normalization_config or NormalizationConfig()

    def read_window(self, window: Window) -> np.ndarray:
        """Read and normalize a window, consistent for training/inference."""
        if self.manifest:
            with RasterStack(self.manifest) as stack:
                data = stack.read_window(window)
        else:
            with rasterio.open(self.raster_path) as src:
                data = src.read(window=window).astype('float32')
                # Apply dtype-aware scaling
                if src.dtypes[0] == 'uint8':
                    data = data / 255.0

        return normalize_for_inference(data, self.config)
```

**Effort:** 4-6 hours
**Risk:** Medium (requires thorough testing)

---

### CODE SMELLS (S1-S8): Technical Debt

#### S1: Long Functions

| Function | Location | Lines | Issue |
|----------|----------|-------|-------|
| `run_pipeline()` | compositing.py:622-826 | 204 | Too many responsibilities |
| `train_unet()` | training/unet.py:101-338 | 237 | Complex manifest handling |
| `_iter_aoi_processing()` | compositing.py:549-622 | 73 | Nested complexity |
| `prepare_naip_reference()` | compositing.py:101-183 | 82 | Multiple exit points |

**Recommendation:** Extract helper functions, apply single-responsibility principle.

---

#### S2: Magic Numbers

| Value | Location | Meaning |
|-------|----------|---------|
| `{3, 8, 9, 10, 11}` | compositing.py:47 | SCL cloud/shadow classes |
| `1/10000` | compositing.py:54 | Sentinel-2 scale factor |
| `-9999.0` | stacking.py:19 | NoData sentinel value |
| `255.0` | inference/unet_stream.py:84 | geoai normalization constant |
| `0.001` | training/unet.py:118 | Default learning rate |

**Recommendation:** Extract to named constants with documentation.

---

#### S3: Bare Exception Handling

```python
# compositing.py:158
try:
    geom = gpd.read_file(path).geometry.values[0]
except Exception:  # Too broad
    pass

# topography/processing.py:45-46
try:
    crs = CRS.from_epsg(4326)
except Exception:
    crs = src.crs
```

**Recommendation:** Catch specific exceptions, log context.

---

#### S4: Duplicate Profile Construction

Found in 5 locations:
- `stacking.py:188-203`
- `inference/unet_stream.py:125-138`
- `topography/processing.py:175-190`
- `sentinel2/compositing.py` (implicit)
- `training/unet.py` (via geoai)

**Recommendation:** Extract to `utils.rasterio.create_gtiff_profile()`.

---

#### S5: Inconsistent Parameter Naming

| Pattern | Locations | Issue |
|---------|-----------|-------|
| `path` vs `manifest_path` | Multiple | Ambiguous naming |
| `output_dir` vs `out_folder` | training, compositing | Inconsistent |
| `labels_path` vs `in_class_data` | training | Mixed conventions |

**Recommendation:** Establish naming convention in CLAUDE.md.

---

#### S6: Unused Imports

```python
# sentinel2/compositing.py
from dataclasses import dataclass  # Only used once (SeasonConfig)

# training/unet.py
from typing import Set  # Only used in type hint
```

**Recommendation:** Clean up with autoflake/isort.

---

#### S7: Missing Type Hints

| File | Coverage | Issue |
|------|----------|-------|
| sentinel2/compositing.py | 60% | Missing return types |
| training/unet.py | 70% | Some Any types |
| data_acquisition.py | 50% | Limited annotations |

**Recommendation:** Add comprehensive type hints, enable mypy strict mode.

---

#### S8: No Docstrings for Public APIs

| Module | Public Functions | With Docstrings |
|--------|-----------------|-----------------|
| stacking.py | 8 | 4 (50%) |
| compositing.py | 15 | 8 (53%) |
| unet_stream.py | 2 | 1 (50%) |
| training/unet.py | 1 | 1 (100%) |

**Recommendation:** Add docstrings following NumPy style.

---

### TESTING GAPS (T1-T5)

#### T1: Zero Tests for Compositing Module

**Location:** `sentinel2/compositing.py` (995 LOC)
**Coverage:** 0%
**Risk:** Blind spots in data pipeline correctness

**Recommended Tests:**
```python
# tests/test_sentinel2_compositing.py

def test_parse_aoi_wkt():
    """Test WKT string parsing."""

def test_parse_aoi_geojson():
    """Test GeoJSON file parsing."""

def test_season_date_range():
    """Test seasonal date range computation."""

def test_stack_bands_shape():
    """Test band stacking output dimensions."""

def test_build_mask_scl_values():
    """Test SCL-based cloud masking."""

def test_seasonal_median_computation():
    """Test median compositing logic."""
```

---

#### T2: Limited Stacking Module Tests

**Location:** `stacking.py` (383 LOC)
**Current Coverage:** Indirect via `test_prepare_window_matches_training_normalization()`

**Missing Tests:**
- `RasterStack` class methods
- `load_manifest()` error handling
- `rewrite_tile_images()` functionality
- Per-band scaling logic

---

#### T3: No Training Module Unit Tests

**Location:** `training/unet.py` (344 LOC)
**Coverage:** 0% (relies on integration)

**Critical Missing Tests:**
- Manifest resolution logic
- Tile generation coordination
- Label reprojection

---

#### T4: No Integration Tests

**Issue:** No end-to-end pipeline tests exist.

**Recommended Integration Tests:**
```python
# tests/integration/test_full_pipeline.py

def test_sentinel2_to_manifest():
    """Test full Sentinel-2 compositing pipeline."""

def test_training_from_manifest():
    """Test training pipeline with manifest input."""

def test_inference_matches_training():
    """Verify inference output is consistent with training."""
```

---

#### T5: No Fixtures or Test Data

**Issue:** Tests create synthetic data inline.

**Recommendation:** Create `tests/fixtures/` with:
- Sample manifest files
- Small test rasters
- Mock STAC responses

---

## Proposed Refactoring Plan

### Phase 1: Critical Bug Fixes (Immediate)

**Duration:** 1-2 days
**Risk:** Low-Medium
**Dependencies:** None

| Task | File | Effort | Priority |
|------|------|--------|----------|
| 1.1 Add dtype-aware scaling to `infer_raster()` | inference/unet_stream.py | 2h | P0 |
| 1.2 Document normalization precision trade-off | DEEP_ANALYSIS_REPORT.md | 1h | P0 |
| 1.3 Add validation for band_scaling in manifests | stacking.py | 2h | P1 |

**Acceptance Criteria:**
- [ ] `infer_raster()` correctly handles uint8 inputs
- [ ] Normalization flow is documented
- [ ] Invalid manifests raise clear errors

---

### Phase 2: Module Decomposition (Short-term)

**Duration:** 1 week
**Risk:** Medium
**Dependencies:** Phase 1

| Task | Current | Target | Effort |
|------|---------|--------|--------|
| 2.1 Extract AOI parsing | compositing.py | sentinel2/aoi.py | 2h |
| 2.2 Extract NAIP handling | compositing.py | sentinel2/naip.py | 3h |
| 2.3 Extract STAC queries | compositing.py | sentinel2/stac.py | 2h |
| 2.4 Extract band stacking | compositing.py | sentinel2/bands.py | 2h |
| 2.5 Extract cloud masking | compositing.py | sentinel2/masking.py | 1h |
| 2.6 Extract seasonal logic | compositing.py | sentinel2/seasonal.py | 2h |
| 2.7 Consolidate utilities | scattered | utils/*.py | 3h |

**Acceptance Criteria:**
- [ ] No file exceeds 300 LOC
- [ ] All existing tests pass
- [ ] Import structure is clean (no circular imports)

---

### Phase 3: Testing & Quality (Medium-term)

**Duration:** 1-2 weeks
**Risk:** Low
**Dependencies:** Phase 2

| Task | Target Coverage | Effort |
|------|-----------------|--------|
| 3.1 Add compositing module tests | 60% | 4h |
| 3.2 Add stacking module tests | 80% | 3h |
| 3.3 Add training module tests | 60% | 3h |
| 3.4 Add integration tests | E2E | 4h |
| 3.5 Create test fixtures | N/A | 2h |
| 3.6 Add type hints (mypy) | 100% | 4h |
| 3.7 Add docstrings | 100% | 3h |

**Acceptance Criteria:**
- [ ] Overall test coverage > 60%
- [ ] mypy passes in strict mode
- [ ] All public APIs have docstrings

---

### Phase 4: Architecture Cleanup (Long-term)

**Duration:** 2-3 weeks
**Risk:** Medium-High
**Dependencies:** Phase 3

| Task | Description | Effort |
|------|-------------|--------|
| 4.1 Implement configuration system | YAML-based config | 4h |
| 4.2 Add manifest schema validation | Pydantic models | 3h |
| 4.3 Unify training/inference preprocessing | DataPipeline class | 6h |
| 4.4 Remove /255 workaround | Custom dataset class | 8h |
| 4.5 Implement proper error handling | Specific exceptions | 4h |

**Acceptance Criteria:**
- [ ] Configuration is externalized
- [ ] Manifests are validated at load time
- [ ] Training and inference use identical preprocessing
- [ ] No precision loss in normalization

---

## Risk Assessment and Mitigation

### High-Risk Changes

| Change | Risk | Mitigation |
|--------|------|------------|
| Removing /255 workaround | Model incompatibility | Requires full retrain, version models |
| Splitting compositing.py | Import breaks | Incremental extraction, alias imports |
| Unified preprocessing | Subtle behavior changes | Comprehensive test coverage first |

### Rollback Strategy

1. **Git Tags:** Create tag before each phase
2. **Feature Branches:** Develop in isolation
3. **Manifest Versioning:** Add version field to manifests
4. **Model Compatibility:** Track which model versions expect which normalization

---

## Testing Strategy

### Unit Testing

```python
# Recommended test structure
tests/
├── conftest.py              # Shared fixtures
├── fixtures/
│   ├── manifests/           # Sample manifest JSON files
│   ├── rasters/             # Small test GeoTIFFs
│   └── vectors/             # Test shapefiles/geopackages
├── unit/
│   ├── test_stacking.py
│   ├── test_normalization.py
│   ├── sentinel2/
│   │   ├── test_aoi.py
│   │   ├── test_naip.py
│   │   ├── test_stac.py
│   │   └── test_seasonal.py
│   ├── topography/
│   │   └── test_processing.py
│   └── inference/
│       └── test_unet_stream.py
└── integration/
    ├── test_sentinel2_pipeline.py
    ├── test_training_pipeline.py
    └── test_inference_pipeline.py
```

### Test Coverage Targets

| Module | Current | Phase 3 Target | Phase 4 Target |
|--------|---------|----------------|----------------|
| stacking.py | 20% | 80% | 90% |
| sentinel2/ | 0% | 60% | 80% |
| topography/ | 40% | 70% | 85% |
| inference/ | 15% | 60% | 80% |
| training/ | 0% | 50% | 70% |
| **Overall** | **~6%** | **60%** | **80%** |

---

## Success Metrics

### Code Quality

| Metric | Current | Target |
|--------|---------|--------|
| Max file LOC | 995 | 300 |
| Cyclomatic complexity | High | Medium |
| Type hint coverage | 60% | 100% |
| Docstring coverage | 50% | 100% |
| Test coverage | 6% | 80% |

### Maintainability

| Metric | Current | Target |
|--------|---------|--------|
| Time to onboard new dev | ~1 week | 2-3 days |
| Time to add new data source | ~1 day | 2-4 hours |
| Time to debug normalization issue | ~4 hours | 30 min |

### Reliability

| Metric | Current | Target |
|--------|---------|--------|
| Inference consistency | Variable | 100% match training |
| Manifest validation | None | Full schema check |
| Error messages | Cryptic | Actionable |

---

## Appendix A: File-by-File Analysis

### High-Priority Files

| File | LOC | Issues | Action |
|------|-----|--------|--------|
| sentinel2/compositing.py | 995 | Monolithic | **SPLIT** |
| stacking.py | 383 | Core, needs tests | Test |
| inference/unet_stream.py | 313 | Normalization | Fix C2 |
| training/unet.py | 344 | No tests | Test |

### Medium-Priority Files

| File | LOC | Issues | Action |
|------|-----|--------|--------|
| data_acquisition.py | 252 | No tests | Test |
| topography/processing.py | 203 | Magic constants | Document |
| topography/download.py | 204 | External API | Mock tests |
| test_unet.py | 230 | CLI coupling | Refactor |

### Low-Priority Files

| File | LOC | Issues | Action |
|------|-----|--------|--------|
| sentinel2/manifests.py | 141 | Clean | None |
| topography/pipeline.py | 105 | Clean | None |
| sentinel2/progress.py | 81 | Clean | None |
| tiling.py | 61 | Small | None |

---

## Appendix B: Dependency Update Recommendations

Current dependencies show these opportunities:

| Package | Purpose | Recommendation |
|---------|---------|----------------|
| pydantic | Schema validation | Add for manifest validation |
| pytest-cov | Coverage reporting | Add for test metrics |
| mypy | Type checking | Enable strict mode |
| black | Code formatting | Already configured |
| isort | Import sorting | Add to pre-commit |

---

## Appendix C: CLAUDE.md Update Recommendations

Add the following sections to project CLAUDE.md:

```markdown
## Normalization Standards

All image data should be normalized to [0, 1] range before model input.
The `/255.0` division is a compatibility workaround for geoai's internal
normalization. When using custom datasets, this step can be skipped.

## Manifest Format

See `documentation/manifest-schema.md` for the complete manifest specification.
All manifests should include:
- `grid`: CRS, transform, dimensions
- `sources`: Array of data sources with scaling configuration

## File Size Limits

Keep individual Python files under 300 lines of code.
Extract reusable components to `utils/` module.

## Testing Requirements

All new features must include:
- Unit tests with >60% coverage
- Integration test if affecting pipeline
- Type hints for all public functions
```

---

*Document generated by refactoring analysis on 2025-12-02*
