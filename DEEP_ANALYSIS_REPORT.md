# Deep Analysis: Wetlands ML Codex - Normalization & Complexity Issues

**Date:** 2024
**Analysis Type:** Deep dive into training/inference pipeline issues

---

## Executive Summary

This codebase has **critical normalization bugs** that will cause models to predict incorrectly, plus significant complexity issues that make maintenance difficult. The recent `/255` fix addressed one symptom but **multiple fundamental issues remain**.

### Critical Issues (Will Break Predictions)
1. **Topography data is completely destroyed** - values clipped to [0,1] before the model sees them
2. **Direct raster inference is broken** - uint8 data gets clipped, not scaled
3. **Test suite has a bug** - doesn't verify the /255 fix

### Design Issues (Make Code Hard to Maintain)
4. Single 986-line file doing too much
5. Normalization scattered across multiple stages with no single source of truth
6. Inconsistent handling between manifest-based and raster-based paths

---

## Issue #1: Topography Data is Completely Destroyed (CRITICAL)

### The Problem

The topography stack contains 5 bands with these value ranges:
- **Elevation**: 0 to 3000+ meters
- **Slope**: 0 to 90 degrees
- **TPI_small**: -50 to +50 meters (relative values)
- **TPI_large**: -100 to +100 meters (relative values)
- **DepressionDepth**: 0 to 50+ meters

When the manifest is created (`compositing.py:782-797`), topography has **no `scale_max`**:

```python
extra_sources = [{
    "type": "topography",
    "path": str(topography_path.resolve()),
    "band_labels": [...],
    "resample": "bilinear",
    "nodata": FLOAT_NODATA,
    # NO scale_max!  <-- THIS IS THE BUG
}]
```

### What Happens

In `RasterStack.read_window()` (`stacking.py:237-241`):
```python
if source.scale_max:  # This is None for topography!
    data[mask] = data[mask] / float(source.scale_max)
```

Without `scale_max`, raw values pass through unchanged. Then in `normalize_stack_array()`:
```python
np.clip(cleaned, 0.0, 1.0, out=cleaned)  # DESTROYS ALL DATA > 1
```

**Result:**
- Elevation 500m → clipped to 1.0
- Elevation 1000m → clipped to 1.0
- Elevation 2000m → clipped to 1.0
- **All elevations above 1 meter become identical!**

- TPI -5m → clipped to 0.0
- TPI +5m → clipped to 1.0
- **All negative values become 0, all positive values above 1 become 1!**

**The model cannot distinguish terrain features because all topography information is destroyed.**

### The Fix

Option A: Add `scale_max` to topography manifest entry:
```python
extra_sources = [{
    "type": "topography",
    "scale_max": {
        "Elevation": 3000.0,
        "Slope": 90.0,
        "TPI_small": 50.0,  # Would need special handling for negative values
        "TPI_large": 100.0,
        "DepressionDepth": 50.0
    },
    ...
}]
```

Option B (Better): Per-band normalization with min/max or z-score normalization.

---

## Issue #2: Direct Raster Inference is Broken (CRITICAL)

### The Problem

When running inference with `--test-raster` (not manifest), the code path is:

```python
# test_unet.py:199
geoai.semantic_segmentation(str(test_raster_path), ...)
```

geoai internally divides input by 255, assuming uint8 (0-255) input.

BUT if you try to use the streaming inference (`infer_raster()` in `unet_stream.py`):

```python
read_window=lambda win: src.read(window=win),  # Raw values!
```

This reads raw values. For uint8 NAIP (0-255):
1. `normalize_stack_array` clips to [0, 1] → **all values > 1 become 1.0**
2. `/255` → everything becomes ~0.00392
3. **Model sees uniform input → meaningless predictions**

### The Fix

The streaming inference should NOT be used with raw rasters unless proper scaling is applied. Either:
- Only use manifest-based streaming (which has `scale_max`)
- Add explicit scaling for raster mode based on dtype

---

## Issue #3: Test Suite Bug

### The Problem

`tests/test_inference_cli.py:30-46`:
```python
def test_prepare_window_matches_training_normalization():
    ...
    expected = normalize_stack_array(expected, FLOAT_NODATA)  # NO /255
    actual = _prepare_window(raw, desired_channels=4, nodata_value=FLOAT_NODATA)  # HAS /255
    np.testing.assert_allclose(actual, expected)  # SHOULD FAIL!
```

The test compares:
- `expected`: values in [0, 1] (no /255)
- `actual`: values in [0, 0.00392] (has /255)

**This test should fail but either isn't being run or there's something else going on.**

### The Fix

Update the test to include the /255 step:
```python
expected = normalize_stack_array(expected, FLOAT_NODATA) / 255.0
```

---

## Issue #4: Complexity - 986-Line Compositing File

### The Problem

`sentinel2/compositing.py` is 986 lines doing:
- AOI parsing (parse_aoi, extract_aoi_polygons, _buffer_in_meters)
- NAIP handling (collect_naip_sources, prepare_naip_reference, _clip_raster_to_polygon)
- Sentinel-2 STAC queries (fetch_items)
- Band stacking (stack_bands, stack_scl)
- Cloud masking (build_mask, SCL_MASK_VALUES)
- Seasonal compositing (seasonal_median, concatenate_seasons)
- Topography orchestration (calls prepare_topography_stack)
- Manifest writing
- CLI argument parsing (configure_parser)
- Progress reporting

This violates single-responsibility principle and makes debugging difficult.

### The Fix

Split into focused modules:
- `aoi.py`: AOI parsing and geometry handling
- `naip.py`: NAIP mosaicking and clipping
- `sentinel2_stac.py`: STAC queries and band stacking
- `cloud_masking.py`: SCL-based masking
- `seasonal.py`: Seasonal compositing logic
- `pipeline.py`: Orchestration

---

## Issue #5: The /255 Division is a Workaround

### The Problem

The current data flow is:

**Training:**
```
Raw data → RasterStack (applies scale_max) → normalize_stack_array (clips to [0,1])
→ Write tiles as float32 [0,1]
→ geoai loads tiles → /255 → [0, 0.00392]
→ Model trains on [0, 0.00392]
```

**Inference (after fix):**
```
Raw data → RasterStack (applies scale_max) → normalize_stack_array (clips to [0,1])
→ /255 → [0, 0.00392]
→ Model predicts on [0, 0.00392]
```

This is **wasteful**:
- We normalize to [0,1], write to disk, then geoai normalizes again
- Final model input range is [0, 0.00392] instead of full [0, 1]
- Uses only 0.4% of the available precision!

### The Fix

**Option A: Don't pre-normalize training tiles**
- Let geoai handle normalization
- Write tiles with original dtype (uint8 for NAIP)
- Remove /255 from inference
- Problem: Multi-source stacks have different dtypes

**Option B: Custom dataset that skips /255**
- Create subclass of SemanticSegmentationDataset
- Skip the /255 step when tiles are already float32 [0,1]
- Inference uses normalize_stack_array only (no /255)

**Option C (Recommended): Standardized normalization layer**
- Define a single normalization function used everywhere
- All channels normalized to [0, 1] with proper per-source scaling
- No geoai internal normalization

---

## Issue #6: Inconsistent Value Ranges by Source

### Current State

| Source | Raw Range | scale_max | After read_window | After normalize | After /255 |
|--------|-----------|-----------|-------------------|-----------------|------------|
| NAIP | 0-255 | 255 | 0-1 | 0-1 | 0-0.00392 |
| Sentinel-2 | 0-1 (already scaled) | None | 0-1 | 0-1 | 0-0.00392 |
| Topography | 0-3000+ | None | 0-3000+ | 0-1 (CLIPPED!) | 0-0.00392 |

**Topography loses all information because it has no scale_max.**

### The Fix

Each source needs proper scaling:
```json
{
  "type": "topography",
  "band_scaling": [
    {"band": "Elevation", "min": 0, "max": 3000},
    {"band": "Slope", "min": 0, "max": 90},
    {"band": "TPI_small", "min": -50, "max": 50},
    {"band": "TPI_large", "min": -100, "max": 100},
    {"band": "DepressionDepth", "min": 0, "max": 50}
  ]
}
```

---

## Complete Data Flow Diagram

```
                    TRAINING PATH
                    =============

NAIP (uint8)         Sentinel-2             Topography
  0-255               0-1 (scaled)           0-3000+ (raw!)
    │                     │                      │
    ▼                     ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│              RasterStack.read_window()                   │
│   NAIP: /255 (scale_max=255) → [0-1]                    │
│   Sentinel: no change → [0-1]                            │
│   Topography: no change → [0-3000+] ← BUG!              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              normalize_stack_array()                     │
│   - Replace nodata with 0                                │
│   - Replace NaN/inf                                      │
│   - CLIP to [0, 1]  ← DESTROYS TOPOGRAPHY!              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              Tiles saved as float32 [0-1]
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         geoai.train_segmentation_model()                 │
│   SemanticSegmentationDataset: /255                      │
│   Input range: [0, 0.00392]                              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
               MODEL TRAINS ON [0, 0.00392]


                    INFERENCE PATH (Manifest)
                    =========================

                    [Same as training]
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              _prepare_window()                           │
│   normalize_stack_array() → [0-1]                        │
│   /255.0 → [0, 0.00392]  ← FIX ADDED                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
               MODEL PREDICTS ON [0, 0.00392] ✓

               (But topography is still destroyed!)
```

---

## Recommended Refactoring Plan

### Phase 1: Critical Bug Fixes (Immediate)

1. **Fix topography scaling** - Add proper scale_max or min/max scaling
2. **Fix test suite** - Update test to verify /255
3. **Add validation** - Warn if any input values are outside [0,1] before model

### Phase 2: Simplification (Short-term)

1. **Single normalization function** - One place that handles all normalization
2. **Remove /255 workaround** - Either don't pre-normalize or use custom dataset
3. **Split compositing.py** - Break into focused modules

### Phase 3: Architecture Cleanup (Medium-term)

1. **Standardize manifest format** - Per-band min/max scaling
2. **Unify training/inference paths** - Same preprocessing code
3. **Add comprehensive tests** - Cover all data paths

---

## Quick Diagnostic Commands

Check if topography values are being clipped:
```python
import numpy as np
from wetlands_ml_geoai.stacking import RasterStack, load_manifest

manifest = load_manifest("path/to/manifest.json")
with RasterStack(manifest) as stack:
    window = Window(0, 0, 100, 100)
    data = stack.read_window(window)

    # Check each band
    for i, label in enumerate(stack.band_labels):
        band = data[i]
        valid = band[band != -9999]
        print(f"{label}: min={valid.min():.2f}, max={valid.max():.2f}")

        # Flag if likely clipped
        if 'Elevation' in label or 'TPI' in label or 'Slope' in label:
            if valid.max() <= 1.0:
                print(f"  WARNING: {label} appears to be clipped!")
```

---

## Conclusion

The codebase has a **fundamental design flaw**: normalization is applied at multiple stages without a unified strategy, and different data sources have incompatible handling. The recent `/255` fix addressed the most visible symptom but:

1. **Topography data is still destroyed** - immediate action needed
2. **The /255 workaround is fragile** - proper fix needed
3. **Code complexity makes bugs hard to find** - refactoring needed

The model may appear to work on NAIP+Sentinel data alone, but **any topography-based features are completely useless** until the scaling issue is fixed.
