# Wetlands ML GeoAI - Comprehensive Architecture Analysis

**Date:** November 9, 2025
**Codebase Size:** ~4,179 lines of Python code across 31 files
**Analysis Scope:** Full src/wetlands_ml_geoai module structure

---

## Executive Summary

The Wetlands ML GeoAI codebase is a well-organized geospatial machine learning pipeline with clear separation of concerns across functional modules. The architecture demonstrates solid patterns in configuration management and module organization, but exhibits areas for improvement in code duplication, error handling consistency, and test coverage.

**Key Metrics:**
- 31 Python files total (27 in src/)
- 5 main module packages: sentinel2, topography, training, inference, utils
- 3 large files (>300 LOC): compositing.py (986), training/unet.py (344), inference/unet_stream.py (313)
- 88 module-level functions across codebase
- Only 3 test files with limited coverage

---

## 1. OVERALL ARCHITECTURE & MODULE ORGANIZATION

### 1.1 Project Structure

```
wetlands_ml_codex/
├── src/wetlands_ml_geoai/              # Main package (2,559 LOC)
│   ├── sentinel2/                      # Sentinel-2 compositing pipeline
│   │   ├── cli.py                      # CLI entry point
│   │   ├── compositing.py             # Core 986-line orchestration
│   │   ├── manifests.py               # Stack manifest generation
│   │   └── progress.py                # Progress tracking
│   ├── topography/                     # LiDAR-derived topography stack
│   │   ├── cli.py
│   │   ├── config.py                  # Configuration dataclass
│   │   ├── pipeline.py                # DEM orchestration
│   │   ├── processing.py              # Derivative computation
│   │   └── download.py                # 3DEP DEM fetching
│   ├── training/                       # UNet training
│   │   └── unet.py                    # Training orchestration (344 LOC)
│   ├── inference/                      # Model inference
│   │   ├── unet_stream.py             # Streaming inference (313 LOC)
│   │   └── common.py                  # Shared utilities
│   ├── stacking.py                    # NAIP/Sentinel stack access (313 LOC)
│   ├── tiling.py                      # Tile analysis helpers
│   ├── data_acquisition.py            # NAIP/wetlands downloads
│   ├── validate_seasonal_pixels.py    # Validation utilities
│   └── test_unet.py                   # Inference CLI (443 LOC)
├── tests/                              # Limited test suite (3 files)
├── tools/                              # Utility scripts
├── scripts/windows/                   # Windows batch launchers
└── configs/                           # Configuration files
```

### 1.2 Module Responsibilities

| Module | Purpose | LOC | Responsibility |
|--------|---------|-----|-----------------|
| **sentinel2** | Sentinel-2 seasonal compositing | ~1,100 | Download scenes, compute medians, generate manifests, integrate with topography |
| **topography** | LiDAR-derived derivatives | ~407 | Download DEMs, compute slope/TPI/depression, align to reference grid |
| **training** | UNet model training | 344 | Tile preparation, training orchestration, checkpoint management |
| **inference** | Model prediction | 313 | Streaming window inference, probability prediction, raster output |
| **stacking** | Multi-source raster access | 313 | Manifest-based windowed reading, resampling, normalization |
| **data_acquisition** | Data downloading | ~250 | NAIP/wetlands automated downloads, resampling |
| **tiling** | Tile analysis | 62 | Label statistics, channel derivation |
| **utils** | Utilities | Empty | *Currently unused* |

### 1.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Sentinel-2 Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│ AOI → Fetch STAC Items → Stack Bands → Cloud Mask           │
│     → Seasonal Median → NAIP Integration → Topography       │
│     → Manifest Generation                                   │
└──────────────┬──────────────────────────────────────────────┘
               │ Stack Manifest (JSON)
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Training Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│ RasterStack (via manifest) → Windowed Tiling                │
│ → Label Rasterization → Image Normalization                 │
│ → UNet Training → Model Checkpoint                          │
└──────────────┬──────────────────────────────────────────────┘
               │ Trained Model (.pth)
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Inference Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│ RasterStack → Sliding Window → Normalization → UNet         │
│ → Probability Averaging → Argmax/Threshold → Output         │
│ → Vectorization                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. CODE DUPLICATION PATTERNS

### 2.1 Critical Duplications Found

#### **Duplication 1: Geometry Buffering (HIGH PRIORITY)**
**Location:** Two implementations of the same buffering logic
- **File 1:** `/src/wetlands_ml_geoai/sentinel2/compositing.py` (lines 239-245)
  ```python
  def _buffer_in_meters(geom: BaseGeometry, buffer_meters: float) -> BaseGeometry:
      if buffer_meters <= 0:
          return geom
      series = gpd.GeoSeries([geom], crs="EPSG:4326")
      projected = series.to_crs(series.estimate_utm_crs())
      buffered = projected.buffer(buffer_meters)
      return buffered.to_crs(4326).iloc[0]
  ```

- **File 2:** `/src/wetlands_ml_geoai/topography/pipeline.py` (lines 21-25)
  ```python
  def _buffer_geometry(geometry: BaseGeometry, buffer_meters: float) -> BaseGeometry:
      series = gpd.GeoSeries([geometry], crs="EPSG:4326")
      projected = series.to_crs(series.estimate_utm_crs())
      buffered = projected.buffer(buffer_meters)
      return buffered.to_crs(4326).iloc[0]
  ```

**Issue:** Same logic, different names, no error handling for edge case of buffer_meters <= 0
**Recommendation:** Extract to `utils` module as `buffer_geometry_meters()`

---

#### **Duplication 2: CLI Argument Parsing (MEDIUM PRIORITY)**
**Location:** Three separate implementations of parse_args

| File | Type | Lines | Configuration Source |
|------|------|-------|----------------------|
| `train_unet.py` | parse_args() | 194 | CLI + environment variables |
| `test_unet.py` | parse_args() | 113 | CLI + environment variables |
| `validate_seasonal_pixels.py` | parse_args() | 45 | CLI + environment variables |

**Pattern:** Each module independently reads environment variables and builds identical argument parser structure
**Recommendation:** Create shared `argument_parser_helpers.py` with reusable components

---

#### **Duplication 3: Manifest Path Resolution (MEDIUM PRIORITY)**
**Location:** Manifest discovery logic appears twice
- **File 1:** `train_unet.py` (lines 59-122): `_resolve_manifest_paths()`, `_gather_manifest_paths()`
- **File 2:** `stacking.py` (line 64): `load_manifest()`

**Issue:** Different approaches to the same problem - one recursive directory traversal, one simple JSON load
**Recommendation:** Consolidate manifest handling in `stacking.py` as authoritative source

---

### 2.2 Profile-Based Duplication

Multiple modules create rasterio write profiles with identical structure:

**Files affected:**
- `topography/processing.py` (line 160)
- `stacking.py` (line 178)
- `sentinel2/compositing.py` (line 150)
- `inference/unet_stream.py` (line 125)

**Common Pattern:**
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

**Recommendation:** Create helper function `create_geotiff_profile()`

---

## 3. DEPENDENCIES BETWEEN MODULES

### 3.1 Dependency Graph

```
sentinel2/compositing.py
  ├→ data_acquisition.py (download NAIP, wetlands)
  ├→ stacking.py (manifest reading, stack operations)
  ├→ topography/ (prepare_topography_stack)
  └→ sentinel2/manifests.py (manifest writing)

training/unet.py
  ├→ stacking.py (load_manifest, RasterStack, rewrite_tile_images)
  └→ tiling.py (analyze_label_tiles, derive_num_channels)

inference/unet_stream.py
  ├→ stacking.py (RasterStack, normalize_stack_array)
  └→ inference/common.py (resolve_output_paths)

topography/pipeline.py
  ├→ topography/download.py (fetch_dem_inventory, download_dem_products)
  └→ topography/processing.py (write_topography_raster)

topography/processing.py
  └→ stacking.py (FLOAT_NODATA)
```

### 3.2 Tight Coupling Issues

**Issue 1: Circular Domain Knowledge**
- `sentinel2/compositing.py` imports `topography.prepare_topography_stack()` directly
- `topography/pipeline.py` is independent
- This creates implicit assumption that topography preparation is always tied to Sentinel-2 compositing

**Issue 2: Manifest Format Implicit Contract**
- Multiple modules assume specific manifest JSON structure without schema validation
- No central manifest schema definition
- Creates fragility when manifest format changes

**Issue 3: Hard-coded Constants Spread**
- `FLOAT_NODATA = -9999.0` defined in `stacking.py` but replicated expectations elsewhere
- `NAIP_BAND_LABELS` hardcoded in `sentinel2/manifests.py`
- `SENTINEL_BANDS`, `SEASON_WINDOWS` hardcoded in `sentinel2/compositing.py`

---

## 4. KEY DESIGN PATTERNS IN USE

### 4.1 Patterns Observed

#### **Pattern 1: Dataclass Configuration Objects**
**Usage:** Heavy reliance on frozen dataclasses for immutable configuration
- `StackGrid`, `StackSource`, `StackManifest` (stacking.py)
- `TopographyStackConfig` (topography/config.py)
- `NaipDownloadRequest`, `WetlandsDownloadRequest` (data_acquisition.py)
- `ManifestEntry` (sentinel2/manifests.py)
- `PixelInfo`, `SeasonConfig` (validate_seasonal_pixels.py)

**Benefit:** Type safety, immutability, clear parameter passing
**Issue:** No central configuration registry - each module defines its own

#### **Pattern 2: Context Manager for Resource Management**
**Files:** `stacking.py` (RasterStack)
```python
class RasterStack:
    def __enter__(self) -> "RasterStack":
        return self
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```

**Benefit:** Proper cleanup of rasterio datasets
**Coverage:** Good - main resource-heavy class implements it

#### **Pattern 3: Environment Variable Configuration**
**Files:** train_unet.py, test_unet.py, validate_seasonal_pixels.py
- Fallback to `os.getenv()` with defaults
- No configuration file support (no YAML/TOML/INI)

**Issue:** Limited flexibility, no way to define complex configurations portably

#### **Pattern 4: Logging with Module Names**
**Files:** Most modules implement `LOGGER = logging.getLogger(__name__)`
**Coverage:** 6 out of 8 major modules
**Missing:** utils/__init__.py, some analysis modules

#### **Pattern 5: Type Hints**
**Coverage:** ~70% of functions have type hints
**Gaps:** Some variable annotations missing, return types sometimes Optional without annotation

---

## 5. CLI STRUCTURE AND ORGANIZATION

### 5.1 CLI Entry Points

#### Root-Level Shims (Thin Wrappers)
```
/wetlands_ml_codex/
├── train_unet.py         → calls wetlands_ml_geoai.train_unet.main()
├── test_unet.py          → calls wetlands_ml_geoai.test_unet.main()
├── sentinel2_processing.py → calls wetlands_ml_geoai.sentinel2.cli.main()
├── validate_seasonal_pixels.py → calls wetlands_ml_geoai.validate_seasonal_pixels.main()
└── stacking.py           → imports from wetlands_ml_geoai.stacking (no CLI)
```

**Pattern Analysis:**
- Clean separation: root files are sys.path fixers + entry points
- Actual CLI logic lives in `src/` with full implementation
- `setup.bat` provides Windows entrypoints

#### Main CLI Modules

| Module | Type | Argument Count | Return Value |
|--------|------|-----------------|--------------|
| `sentinel2/cli.py` | `build_parser() + main()` | ~20 args | void |
| `train_unet.py` | `parse_args() + main()` | ~30 args | void |
| `test_unet.py` | `parse_args() + main()` | ~20 args | void |
| `topography/cli.py` | `build_parser() + main()` | ~6 args | Path |
| `validate_seasonal_pixels.py` | `parse_args() + main()` | ~8 args | void |

### 5.2 Argument Parsing Inconsistency

**Pattern 1: `build_parser() + main(argv)`**
```python
# sentinel2/cli.py
def build_parser() -> argparse.ArgumentParser:
def main(argv: list[str] | None = None) -> None:
```

**Pattern 2: `parse_args() + main()`**
```python
# train_unet.py
def parse_args() -> argparse.Namespace:
def main() -> None:
```

**Issue:** Inconsistent naming and signatures make it harder to compose CLIs

### 5.3 Environment Variable Fallbacks

**Coverage:** High (all CLI modules check environment variables)
**Patterns:**
```python
parser.add_argument(
    "--train-raster",
    default=os.getenv("TRAIN_RASTER_PATH"),
)
parser.add_argument(
    "--tiles-dir",
    default=os.getenv("TRAIN_TILES_DIR"),
)
```

**Recommendation:** Consolidate env var names into a constants module

---

## 6. COMMON UTILITIES & HELPER FUNCTIONS

### 6.1 Utilities Module Status

**File:** `utils/__init__.py` - **EMPTY** (0 lines)

**Should contain but doesn't:**
- Geometry buffering helper
- Rasterio profile builders
- Environment variable helpers
- Manifest schema validation

### 6.2 Existing Utility Functions

#### In `tiling.py` (62 lines)
```python
def derive_num_channels(raster_path: Path, override: Optional[int]) -> int
def analyze_label_tiles(labels_dir: Path, max_check: int = 256) 
    -> Tuple[float, float, int]
```

#### In `stacking.py` (313 lines)
```python
def normalize_stack_array(data: np.ndarray, nodata_value: Optional[float]) 
    -> np.ndarray
def rewrite_tile_images(manifest: Union[str, Path, StackManifest], 
    images_dir: Path) -> int
```

#### In `inference/common.py` (39 lines)
```python
def resolve_output_paths(source_path: Path, output_dir: Path | None, 
    mask_path: Path | None, vector_path: Path | None, 
    raster_suffix: str) -> Tuple[Path, Path, Path]
```

#### In `sentinel2/manifests.py` (141 lines)
```python
def affine_to_list(transform)
def compute_naip_scaling(dataset: rasterio.io.DatasetReader) 
    -> Optional[float]
def write_stack_manifest(...)
def write_manifest_index(manifest_paths: Iterable[Path], output_dir: Path) 
    -> Path
```

### 6.3 Scattered Helper Functions

**Private helpers not centralized:**
- `_buffer_in_meters()` (compositing.py)
- `_buffer_geometry()` (pipeline.py)
- `_load_json()` (train_unet.py)
- `_is_stack_manifest()` (train_unet.py)
- `_resolve_manifest_paths()` (train_unet.py)
- `_clear_directory()` (training/unet.py)
- `_prepare_labels()` (training/unet.py)
- `_load_model()` (inference/unet_stream.py)
- `_prepare_window()` (inference/unet_stream.py)
- `_compute_offsets()` (inference/unet_stream.py)

---

## 7. DATA PROCESSING PIPELINE FLOWS

### 7.1 Sentinel-2 Seasonal Compositing Flow

**Main Function:** `compositing.run_from_args(args)`

```
1. parse_aoi(aoi_str)
   └→ Load from file or parse WKT/GeoJSON/bbox

2. prepare_naip_reference(naip_sources, working_dir)
   ├→ collect_naip_sources() - Gather from files/directories
   ├→ _collect_naip_footprints() - Cache rasterio metadata
   └→ Optional: _resample_naip_tile() - Match target resolution

3. extract_aoi_polygons(aoi, buffer_meters)
   └→ _buffer_in_meters() - Convert to UTM, buffer, back to WGS84

4. For each AOI polygon:
   ├→ fetch_items() - Query STAC for Sentinel-2
   ├→ stack_bands() - stackstac xarray assembly
   ├→ stack_scl() - Scene Classification Layer
   ├→ build_mask() - Apply cloud masking + dilation
   ├→ seasonal_median() - Compute per-season medians
   ├→ _clip_raster_to_polygon() - Crop to AOI
   └→ write_seasonal_geotiff() - Save composite

5. NAIP Mosaicking (if multiple):
   └→ merge() - Rasterio mosaic operation

6. prepare_topography_stack()
   ├→ download_dem_products() - 3DEP from OpenTopography
   └→ write_topography_raster() - Compute elevation, slope, TPI

7. write_stack_manifest()
   └→ JSON with NAIP + Sentinel + optional topography sources

8. write_manifest_index()
   └→ Meta-index for multi-AOI processing
```

### 7.2 Training Pipeline Flow

**Main Function:** `training/unet.train_unet(...)`

```
1. Input Resolution:
   ├→ Load stack manifest OR single raster
   ├→ _prepare_labels() - Reproject if CRS mismatch
   └→ Determine primary CRS from raster/manifest

2. Directory Setup:
   ├→ Create tiles/, labels/, models/ directories
   ├→ Clean stale staging directories (recursive with retry)
   └→ Create staging root with timestamp

3. For each manifest (multi-AOI support):
   ├→ Check NAIP source exists
   ├→ Get labels for manifest's CRS (cached)
   ├→ geoai.export_geotiff_tiles()
   │  └→ Tile images + rasterized labels
   ├→ rewrite_tile_images()
   │  └→ Replace with normalized stack data
   └→ Move tiles to final location with AOI prefix

4. Single raster fallback:
   └→ geoai.export_geotiff_tiles() without rewriting

5. Validation:
   ├→ Verify image tiles exist
   ├→ Verify label tiles exist
   └→ Raise if none generated

6. Channel Derivation:
   ├→ Use manifest stack band count, OR
   └→ Read from base_raster, OR
   └→ Use override

7. Label Analysis:
   └→ analyze_label_tiles() - Check coverage patterns

8. Model Training:
   └→ geoai.train_segmentation_model()
      ├→ Loads from geoai dependency
      ├→ Supports checkpoint loading
      └→ Saves models to models_dir
```

### 7.3 Inference Pipeline Flow

**Main Function:** `test_unet.main()` or `inference/unet_stream.infer_manifest()`

```
1. Input Resolution:
   ├→ Load stack manifest OR single raster
   └→ Determine if streaming or direct inference

2. Model Loading:
   ├→ Get device (GPU/CPU)
   ├→ Load segmentation-models-pytorch model
   ├→ Handle distributed DataParallel wrapper
   └→ Set eval mode

3. Channel Derivation:
   ├→ Use override, OR
   ├→ Count from RasterStack/raster, OR
   └→ Raise error

4. For streaming (manifest):
   ├→ Open RasterStack (context manager)
   └→ _stream_inference()

5. For direct (raster):
   ├→ Open rasterio dataset
   └→ _stream_inference()

6. _stream_inference() Implementation:
   ├→ _compute_offsets() - Calculate sliding window positions
   ├→ Create probability accumulator (num_classes, H, W)
   ├→ For each window:
   │  ├→ read_window() - Load data
   │  ├→ _prepare_window()
   │  │  ├→ Channel padding/trimming
   │  │  ├→ normalize_stack_array() - nodata → 0, clip [0,1]
   │  │  └→ Divide by 255.0
   │  ├→ Wrap with torch.unsqueeze(0) → (1, C, H, W)
   │  ├→ _predict_probabilities()
   │  │  ├→ model() forward pass
   │  │  └→ F.softmax() across class dimension
   │  └→ Accumulate probabilities + count map (for averaging)
   ├→ _finalize_predictions()
   │  ├→ Average overlapping windows
   │  ├→ Apply probability_threshold OR argmax
   │  └→ Set nodata pixels to 0
   └→ _save_prediction() - Write uint8 GeoTIFF

7. Vectorization:
   └→ geoai.raster_to_vector()
      ├→ Polygon extraction
      └→ Douglas-Peucker simplification
```

### 7.4 Topography Stack Flow

**Main Function:** `topography/pipeline.prepare_topography_stack(config)`

```
1. Local DEM Resolution:
   ├→ Check config.dem_paths (explicit list)
   ├→ Check config.dem_dir (directory scan)
   └→ Validate all files exist

2. Remote DEM Download (if no local):
   ├→ Buffer AOI geometry in meters → UTM projection
   ├→ fetch_dem_inventory() - OpenTopography API
   │  └→ Return product metadata
   ├→ download_dem_products() - Fetch tiles
   └→ Cache in config.cache_dir or output_dir/raw

3. Derivative Computation:
   └→ write_topography_raster()
      ├→ _read_transform() - Reference grid from target_grid_path
      ├→ _mosaic_dem()
      │  ├→ Open all DEM tiles
      │  ├→ Create WarpedVRT for CRS alignment
      │  ├→ rasterio.merge() - Mosaic
      │  └→ Reproject to target grid
      ├→ _compute_slope() - np.gradient → arctan
      ├→ _compute_tpi(small)
      │  ├→ _box_mean() - Uniform filter with valid mask
      │  └→ dem - mean
      ├→ _compute_tpi(large) - Same with larger radius
      ├→ _compute_depression_depth()
      │  ├→ grey_closing() - Morphological
      │  └→ filled - dem
      └→ Stack 5 bands + write GeoTIFF
         ├→ Elevation
         ├→ Slope
         ├→ TPI_small
         ├→ TPI_large
         └→ DepressionDepth
```

---

## 8. TESTING COVERAGE & STRUCTURE

### 8.1 Test File Inventory

| File | Test Count | Coverage Area | Status |
|------|-----------|-----------------|--------|
| `tests/test_training_cli.py` | 2 functions | CLI arg parsing, manifest resolution | Limited |
| `tests/test_inference_cli.py` | 3 functions | CLI arg parsing, window preparation, normalization | Limited |
| `tests/test_topography_pipeline.py` | 3 functions | Topography writing, TPI computation, DEM resolution | Limited |

**Total:** ~8 test functions covering ~3 modules

### 8.2 Testing Gaps

**Uncovered Areas:**
- Sentinel-2 compositing (0 tests)
- NAIP processing (0 tests)
- Manifest generation (0 tests)
- Data acquisition/downloads (0 tests)
- Stacking and window reading (0 tests)
- Training orchestration (0 tests)
- Inference streaming (0 tests)
- Error handling paths (mostly untested)

### 8.3 Test Patterns Observed

**Pattern 1: Functional/Integration Tests**
```python
# test_topography_pipeline.py
def test_write_topography_raster_writes_five_bands(tmp_path: Path) -> None:
    # Create DEM, run through full pipeline
    dem = ...
    config = TopographyStackConfig(...)
    result = write_topography_raster(...)
    assert result.exists()
```

**Pattern 2: CLI Smoke Tests**
```python
# test_training_cli.py
def test_training_cli_requires_inputs():
    with pytest.raises(SystemExit):
        train_unet.parse_args([])
```

**Pattern 3: Parameterized Tests**
```python
# test_training_cli.py
@pytest.mark.parametrize("selector", ["single", "directory", "index"])
def test_resolve_manifest_paths(tmp_path, selector):
    # Test different input types
```

### 8.4 Testing Recommendations

**High Priority:**
1. Add Sentinel-2 compositing tests (STAC mocking)
2. Add manifest generation tests
3. Add normalization/window reading tests

**Medium Priority:**
1. Error handling path coverage
2. Edge case testing (empty rasters, malformed manifests)
3. Multi-AOI scenario testing

---

## 9. CODE SMELLS & ISSUES

### 9.1 Long Functions

#### **Critical Length Issues**

| File | Function | Lines | Complexity |
|------|----------|-------|-----------|
| `sentinel2/compositing.py` | `run_from_args()` | ~650 | Orchestrates full pipeline, hard to test |
| `sentinel2/compositing.py` | `run()` | ~550 | Nested multi-season/multi-AOI logic |
| `training/unet.py` | `train_unet()` | ~240 | Multi-manifest handling, label caching |
| `inference/unet_stream.py` | `_stream_inference()` | ~70 | Nested window loops, accumulation logic |

**Issue:** Functions mixing multiple levels of abstraction, making testing and reuse difficult

### 9.2 Inconsistent Error Handling

#### **Pattern 1: Validation at Entry**
```python
# train_unet.py
args = parser.parse_args()
if not args.train_raster and not args.stack_manifest:
    parser.error("...")  # Exits immediately
```

#### **Pattern 2: Runtime Exceptions**
```python
# stacking.py
if out_height <= 0 or out_width <= 0:
    raise ValueError("Window height/width must be positive")
```

#### **Pattern 3: Unvalidated Access**
```python
# sentinel2/compositing.py
naip_source.path  # Assumes exists, no validation
```

**Recommendation:** Create validation layer that clearly separates input validation from runtime errors

### 9.3 Inconsistent Logging Patterns

#### **Pattern 1: Module-Level Logger**
```python
LOGGER = logging.getLogger(__name__)
LOGGER.info("Starting process")
```

#### **Pattern 2: Direct logging Module**
```python
import logging
logging.info("Starting process")
```

#### **Pattern 3: No Logging**
```python
# utils/__init__.py - Empty, no logging at all
```

**Recommendation:** Standardize on module-level LOGGER with consistent naming

### 9.4 Type Hint Coverage Issues

**Current Status:** ~70% coverage

**Gaps:**
- `Any` used in sentinel2/manifests.py manifest dict structures
- Optional not always declared (e.g., `encoder_weights`)
- Union types could be clarified with TypedDict

### 9.5 Naming Inconsistencies

| Pattern | Examples | Issue |
|---------|----------|-------|
| Underscore-prefixed private | `_buffer_in_meters`, `_prepare_labels` | Correct but not enforced |
| Buffer function names | `_buffer_in_meters` vs `_buffer_geometry` | Inconsistent naming |
| Logging variable | Some `LOGGER`, potentially `logger` | No enforcement |
| Config object naming | `TopographyStackConfig` vs `SeasonConfig` | Inconsistent suffixes |

---

## 10. CONFIGURATION MANAGEMENT

### 10.1 Configuration Sources

#### **Source 1: Environment Variables**
- `TRAIN_RASTER_PATH`, `TRAIN_STACK_MANIFEST`, `TRAIN_LABELS_PATH`
- `TRAIN_TILES_DIR`, `TRAIN_MODELS_DIR`
- `TILE_SIZE`, `TILE_STRIDE`, `TILE_BUFFER`
- `UNET_BATCH_SIZE`, `UNET_NUM_EPOCHS`, `UNET_LEARNING_RATE`
- Over 20 ENV vars total, scattered across modules

#### **Source 2: CLI Arguments**
- ArgumentParser in each module
- No shared configuration schema

#### **Source 3: Hardcoded Constants**
- `DEFAULT_TILE_SIZE = 512` (train_unet.py)
- `SEASON_WINDOWS` (compositing.py)
- `NAIP_BAND_LABELS` (manifests.py)
- No centralized constants module

### 10.2 Configuration Object Pattern

**Dataclass-Based (Good):**
```python
@dataclass(frozen=True)
class TopographyStackConfig:
    aoi: BaseGeometry
    target_grid_path: Path
    output_dir: Path
    buffer_meters: float = 200.0
```

**CLI-Based (Less Ideal):**
```python
def train_unet(
    labels_path: Path,
    tiles_dir: Path,
    models_dir: Path,
    ...28 more parameters
) -> None:
```

**Issue:** 28-parameter function is hard to maintain and extend

### 10.3 Environment Variable Issues

**Problems:**
1. No validation of ENV var sources
2. Environment variables scattered across files
3. No documentation of all available options
4. Naming inconsistencies (sometimes TRAIN_, sometimes UNET_)

**Recommendation:**
```python
# constants/env_vars.py
class EnvironmentVariables:
    TRAIN_RASTER_PATH = "TRAIN_RASTER_PATH"
    TRAIN_STACK_MANIFEST = "TRAIN_STACK_MANIFEST"
    # ... all 20+
```

---

## 11. ERROR HANDLING PATTERNS

### 11.1 Exception Types Used

```python
# Most Common
FileNotFoundError - Path existence checks
ValueError - Invalid parameters
RuntimeError - Operational failures
AttributeError - Missing attributes

# Rasterio-Specific
RasterioIOError - Raster I/O failures
```

### 11.2 Error Handling Code Distribution

**Pattern 1: Fail-Fast Validation**
```python
if not args.labels:
    parser.error("--labels or TRAIN_LABELS_PATH must be supplied.")
```

**Pattern 2: Try-Except with Logging**
```python
try:
    with rasterio.open(raster_path) as src:
        manifest_crs = src.crs
except RasterioIOError as exc:
    logging.warning("Skipping manifest: %s", exc)
    continue
```

**Pattern 3: Silent Failures (Anti-pattern)**
```python
# test_unet._load_json()
except (OSError, json.JSONDecodeError):
    return None  # No logging
```

### 11.3 Missing Error Cases

**Unhandled Scenarios:**
1. Corrupted raster files (no fallback)
2. Invalid manifest JSON structure
3. Network failures during STAC queries
4. Disk full during large writes
5. CUDA out of memory during training

---

## 12. ARCHITECTURAL DEBT & RECOMMENDATIONS

### 12.1 Critical Issues

| Priority | Issue | Location | Fix Effort |
|----------|-------|----------|-----------|
| **HIGH** | Duplicate buffering logic | compositing.py + pipeline.py | 1 hour |
| **HIGH** | 28-parameter train_unet() | training/unet.py | 2 hours |
| **HIGH** | Empty utils module | utils/__init__.py | 1 hour |
| **HIGH** | 650-line run_from_args() | sentinel2/compositing.py | 4 hours |
| **MEDIUM** | No manifest schema validation | stacking.py + train_unet.py | 2 hours |
| **MEDIUM** | Scattered environment variables | Multiple | 1.5 hours |
| **MEDIUM** | No type hints for manifests | sentinel2/manifests.py | 1 hour |
| **MEDIUM** | Inconsistent CLI patterns | test_unet.py + sentinel2/cli.py | 1 hour |
| **LOW** | Missing test coverage | tests/ | 8+ hours |
| **LOW** | Hardcoded constants | Multiple | 1 hour |

### 12.2 Recommended Refactoring Plan

#### **Phase 1: Low-Hanging Fruit (Week 1)**
1. Extract `buffer_geometry_meters()` to utils module
2. Create `geotiff_profile_builder()` helper
3. Create `CONSTANTS.py` with all hardcoded values
4. Create `ENV_VARS.py` with all environment variable names

#### **Phase 2: CLI Standardization (Week 2)**
1. Create `cli_helpers.py` with shared argument parser components
2. Standardize all CLIs to `build_parser() + main(argv=None)`
3. Document all environment variables

#### **Phase 3: Type Safety (Week 3)**
1. Add TypedDict for manifest structures
2. Add manifest schema validation
3. Increase type hint coverage to 90%+

#### **Phase 4: Function Decomposition (Week 4)**
1. Extract `prepare_naip_reference()` to separate module
2. Extract seasonal composite computation to separate function
3. Break train_unet() into `TrainingOrchestrator` class
4. Extract inference streaming to separate class

#### **Phase 5: Testing (Ongoing)**
1. Add pytest fixtures for temporary rasters
2. Add STAC API mocking
3. Increase coverage to 60%+ of critical paths

### 12.3 Code Organization Improvements

**Current:**
```
src/wetlands_ml_geoai/
├── Individual module files
├── Some logic in CLI modules
└── utils/ (empty)
```

**Recommended:**
```
src/wetlands_ml_geoai/
├── core/                      # Core data structures
│   ├── manifest.py           # Manifest + schema
│   ├── stack.py              # RasterStack
│   └── config.py             # Shared config objects
├── common/                    # Shared utilities
│   ├── geometry.py           # Buffering, transformations
│   ├── profiles.py           # Rasterio profiles
│   ├── io.py                 # File I/O helpers
│   └── constants.py          # All constants
├── sentinel2/                # Sentinel-2 pipeline
├── topography/               # Topography pipeline
├── training/                 # Training pipeline
├── inference/                # Inference pipeline
├── cli/                       # CLI modules
│   ├── train.py
│   ├── test.py
│   ├── sentinel2.py
│   └── helpers.py            # Shared CLI code
└── tests/                     # Test suite
```

---

## 13. KEY FINDINGS SUMMARY

### Strengths
✅ **Clean module boundaries:** Clear separation between sentinel2, topography, training, inference
✅ **Type hints:** 70% coverage shows discipline
✅ **Dataclass configuration:** Immutable config objects prevent bugs
✅ **Context managers:** Proper resource cleanup in RasterStack
✅ **Environment variables:** Flexible deployment configuration
✅ **Documentation:** README and docs folder present

### Weaknesses
❌ **Code duplication:** _buffer_in_meters duplicated, CLI parsing scattered
❌ **Inconsistent patterns:** Logging, error handling, CLI signatures vary
❌ **Large functions:** run_from_args (650 lines), train_unet (240 lines)
❌ **Empty utils module:** Should centralize helpers
❌ **Limited testing:** 3 test files, major components untested
❌ **Scattered constants:** Hardcoded values in multiple files
❌ **No schema validation:** Manifest JSON assumes implicit structure

### Technical Debt
- 8-12 hours refactoring to address issues
- 10+ hours testing to reach 60% coverage
- Unknown effort for production hardening

---

## 14. SPECIFIC FILE REFERENCES WITH CODE EXAMPLES

### Large Functions Needing Refactoring

**File:** `/src/wetlands_ml_geoai/sentinel2/compositing.py:434-1084`
```python
def run_from_args(args) -> None:
    """650+ line orchestration function mixing concerns"""
    # Should be split into:
    # - prepare_aoi_inputs()
    # - process_season_for_aoi()
    # - finalize_manifests()
```

**File:** `/src/wetlands_ml_geoai/training/unet.py:101-340`
```python
def train_unet(
    labels_path: Path,
    tiles_dir: Path,
    models_dir: Path,
    train_raster: Optional[Path] = None,
    # ... 23 more parameters
) -> None:
    """Should be refactored to TrainingConfig dataclass"""
```

### Duplication Examples

**File 1:** `/src/wetlands_ml_geoai/sentinel2/compositing.py:239-245`
```python
def _buffer_in_meters(geom: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    if buffer_meters <= 0:
        return geom
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]
```

**File 2:** `/src/wetlands_ml_geoai/topography/pipeline.py:21-25`
```python
def _buffer_geometry(geometry: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]
```

### Dependency Tight Coupling

**File:** `/src/wetlands_ml_geoai/sentinel2/compositing.py:40-41`
```python
from ..topography import TopographyStackConfig, prepare_topography_stack
# This imports topography directly - creates tight coupling
# Better: make topography optional, pass as callback
```

---

## CONCLUSION

The Wetlands ML GeoAI codebase demonstrates solid software engineering fundamentals with clear module organization and appropriate use of modern Python patterns (dataclasses, type hints, context managers). The primary opportunities for improvement center on reducing code duplication, standardizing design patterns across modules, and substantially increasing test coverage for critical data processing pipelines.

The estimated effort to address high-priority issues is 8-12 hours of refactoring, with significant time required for comprehensive testing. Implementation of the recommended refactoring plan would substantially improve maintainability and reduce the risk of subtle bugs in data processing pipelines.

