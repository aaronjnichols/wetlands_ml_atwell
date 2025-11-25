# Recommended Fixes - Implementation Plan

This document provides concrete, actionable fixes for the issues identified in `DEEP_ANALYSIS_REPORT.md`.

---

## Priority 1: Fix Topography Scaling (CRITICAL)

### Problem
Topography values (elevation 0-3000m, slope 0-90°, TPI ±100m) are clipped to [0,1] destroying all information.

### Solution
Add per-band scaling parameters to the topography manifest entry.

### Implementation

**Step 1: Update `stacking.py` to support per-band scaling**

```python
# In stacking.py, modify StackSource dataclass:

@dataclass(frozen=True)
class StackSource:
    type: str
    path: Path
    band_labels: Sequence[str]
    scale_max: Optional[float] = None
    scale_min: Optional[float] = None  # ADD THIS
    band_scaling: Optional[Dict[str, Tuple[float, float]]] = None  # ADD THIS (per-band min/max)
    nodata: Optional[float] = None
    resample: str = "bilinear"
    dtype: Optional[str] = None
    description: Optional[str] = None
```

**Step 2: Update `read_window()` to apply per-band scaling**

```python
# In RasterStack.read_window(), replace the scale_max block:

if source.band_scaling:
    # Per-band min-max scaling
    for band_idx, label in enumerate(source.band_labels):
        if label in source.band_scaling:
            vmin, vmax = source.band_scaling[label]
            mask = data[band_idx] != FLOAT_NODATA
            if mask.any():
                data[band_idx] = data[band_idx].copy()
                data[band_idx][mask] = (data[band_idx][mask] - vmin) / (vmax - vmin)
elif source.scale_max:
    # Single scale_max for all bands (existing behavior)
    mask = data != FLOAT_NODATA
    if mask.any():
        data = data.copy()
        if source.scale_min is not None:
            data[mask] = (data[mask] - source.scale_min) / (source.scale_max - source.scale_min)
        else:
            data[mask] = data[mask] / float(source.scale_max)
```

**Step 3: Update `compositing.py` to include topography scaling** ✅ IMPLEMENTED

```python
# In compositing.py, update the extra_sources creation:
# NOTE: Raw elevation is excluded - only relative features that generalize across regions

extra_sources = [{
    "type": "topography",
    "path": str(topography_path.resolve()),
    "band_labels": [
        "Slope",
        "TPI_small",
        "TPI_large",
        "DepressionDepth",
    ],
    "band_scaling": {
        "Slope": [0.0, 90.0],            # degrees (0=flat, 90=vertical)
        "TPI_small": [-50.0, 50.0],      # meters (negative=depression)
        "TPI_large": [-100.0, 100.0],    # meters (negative=depression)
        "DepressionDepth": [0.0, 50.0],  # meters (depth of local sinks)
    },
    "resample": "bilinear",
    "nodata": FLOAT_NODATA,
    "description": topo_config.description,
}]
```

**Step 4: Update manifest loading in `load_manifest()`**

```python
# In load_manifest(), add band_scaling parsing:

sources.append(
    StackSource(
        type=source_data["type"],
        path=Path(source_data["path"]).expanduser().resolve(),
        band_labels=tuple(source_data.get("band_labels", [])),
        scale_max=source_data.get("scale_max"),
        scale_min=source_data.get("scale_min"),
        band_scaling=source_data.get("band_scaling"),  # ADD THIS
        nodata=source_data.get("nodata"),
        resample=source_data.get("resample", "bilinear"),
        dtype=source_data.get("dtype"),
        description=source_data.get("description"),
    )
)
```

---

## Priority 2: Fix the Test Suite

### Problem
`test_prepare_window_matches_training_normalization` doesn't include the /255 step.

### Implementation

```python
# In tests/test_inference_cli.py, update the test:

def test_prepare_window_matches_training_normalization():
    raw = np.array(
        [
            [[0.2, 0.5], [np.nan, FLOAT_NODATA]],
            [[1.2, -5.0], [0.4, np.inf]],
            [[FLOAT_NODATA, 2.0], [3.0, -np.inf]],
        ],
        dtype=np.float32,
    )

    expected = np.zeros((4, raw.shape[1], raw.shape[2]), dtype=np.float32)
    expected[: raw.shape[0]] = raw
    expected = normalize_stack_array(expected, FLOAT_NODATA)
    expected = expected / 255.0  # ADD THIS LINE

    actual = _prepare_window(raw, desired_channels=4, nodata_value=FLOAT_NODATA)

    np.testing.assert_allclose(actual, expected)
```

---

## Priority 3: Add Validation Warnings

### Problem
Silent data corruption when values are outside expected range.

### Implementation

Add to `normalize_stack_array()`:

```python
def normalize_stack_array(
    data: np.ndarray,
    nodata_value: Optional[float] = FLOAT_NODATA,
    warn_on_clip: bool = True,  # ADD THIS
) -> np.ndarray:
    """Return a copy of data with nodata cleaned, NaNs filled, and values clipped to [0, 1]."""

    cleaned = data.astype(np.float32, copy=True)
    if nodata_value is not None:
        nodata_mask = cleaned == float(nodata_value)
        if nodata_mask.any():
            cleaned[nodata_mask] = 0.0

    np.nan_to_num(cleaned, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    # ADD THIS WARNING BLOCK:
    if warn_on_clip:
        valid_mask = cleaned != 0.0  # Exclude nodata that was set to 0
        if valid_mask.any():
            max_val = cleaned[valid_mask].max()
            min_val = cleaned[valid_mask].min()
            if max_val > 1.0:
                import logging
                logging.warning(
                    "Values above 1.0 detected (max=%.2f) - these will be clipped. "
                    "Check that all input sources have proper scaling configured.",
                    max_val
                )
            if min_val < 0.0:
                import logging
                logging.warning(
                    "Negative values detected (min=%.2f) - these will be clipped to 0. "
                    "Check band_scaling for sources with negative values (e.g., TPI).",
                    min_val
                )

    np.clip(cleaned, 0.0, 1.0, out=cleaned)
    return cleaned
```

---

## Priority 4: Simplify Normalization (Medium-term)

### Option A: Remove /255 workaround entirely

**Training side:** Don't pre-normalize tiles, let geoai handle uint8 → float conversion.

This requires:
1. Remove `rewrite_tile_images()` call from training
2. Export tiles with original dtype
3. Remove `/255` from inference
4. Only works if all sources can be converted to uint8

**NOT RECOMMENDED** - too complex for multi-source stacks.

### Option B (Recommended): Custom dataset class

Create a dataset that doesn't apply /255 when tiles are already float32:

```python
# New file: src/wetlands_ml_geoai/training/dataset.py

import torch
from torch.utils.data import Dataset
import rasterio
from pathlib import Path

class NormalizedSegmentationDataset(Dataset):
    """Dataset for tiles that are already normalized to [0,1]."""

    def __init__(self, images_dir: Path, labels_dir: Path, transform=None):
        self.image_paths = sorted(images_dir.glob("*.tif"))
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.labels_dir / img_path.name

        with rasterio.open(img_path) as src:
            image = src.read().astype('float32')  # Already [0,1], no /255!

        with rasterio.open(lbl_path) as src:
            label = src.read(1).astype('int64')

        if self.transform:
            transformed = self.transform(image=image.transpose(1,2,0), mask=label)
            image = transformed['image'].transpose(2,0,1)
            label = transformed['mask']

        return torch.from_numpy(image), torch.from_numpy(label)
```

Then update inference to NOT divide by 255:

```python
# In _prepare_window():
array = normalize_stack_array(array, nodata_value)
# REMOVE: array = array / 255.0
return array
```

**This requires retraining the model** but results in cleaner code and better precision.

---

## Priority 5: Split compositing.py (Medium-term)

### Proposed Structure

```
src/wetlands_ml_geoai/sentinel2/
├── __init__.py
├── aoi.py              # parse_aoi, extract_aoi_polygons, _buffer_in_meters
├── naip.py             # collect_naip_sources, prepare_naip_reference, _clip_raster_to_polygon
├── stac.py             # fetch_items, STAC client handling
├── bands.py            # stack_bands, stack_scl, SENTINEL_BANDS
├── masking.py          # build_mask, SCL_MASK_VALUES
├── seasonal.py         # seasonal_median, concatenate_seasons, write_dataarray
├── pipeline.py         # run_pipeline (orchestration only)
├── cli.py              # configure_parser, run_from_args
├── manifests.py        # (existing)
└── progress.py         # (existing)
```

Each file would be ~100-200 lines with a single responsibility.

---

## Quick Fix Checklist

For immediate relief without major refactoring:

- [x] Add `band_scaling` to topography manifest entry ✅ IMPLEMENTED
- [x] Update the failing test ✅ IMPLEMENTED
- [x] Add warning logs for clipped values ✅ IMPLEMENTED
- [x] Document the expected input ranges for each source ✅ IMPLEMENTED
- [x] Remove raw elevation (doesn't help, hurts generalization) ✅ IMPLEMENTED

### Manual Manifest Fix

If you have an **existing manifest** created before these fixes, update it to:
1. Remove "Elevation" from band_labels
2. Add band_scaling for the remaining 4 bands

```json
{
  "type": "topography",
  "path": "/path/to/topography.tif",
  "band_labels": ["Slope", "TPI_small", "TPI_large", "DepressionDepth"],
  "band_scaling": {
    "Slope": [0, 90],
    "TPI_small": [-50, 50],
    "TPI_large": [-100, 100],
    "DepressionDepth": [0, 50]
  },
  "resample": "bilinear",
  "nodata": -9999.0
}
```

**Note:** You will also need to regenerate your topography raster to have only 4 bands.

---

## Summary

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | Topography scaling | Medium | Critical - data is destroyed |
| 2 | Fix test suite | Low | Important - CI/CD integrity |
| 3 | Add validation warnings | Low | Helpful - catch future bugs |
| 4 | Remove /255 workaround | High | Cleaner code, requires retrain |
| 5 | Split compositing.py | Medium | Maintainability |

**Recommended approach:** Implement Priority 1-3 immediately, then consider Priority 4-5 when planning a larger refactor or model retrain.
