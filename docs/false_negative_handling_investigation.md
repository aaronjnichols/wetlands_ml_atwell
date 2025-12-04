# False Negative Handling Investigation

**Status:** In Progress - No final decisions made
**Date:** 2025-12-03
**Context:** Discussion about handling discontinuous wetland delineations in training data

## Problem Statement

Atwell's wetland delineations only cover project boundaries. When using an AOI larger than the project boundaries, there's risk of **false negatives** - areas that are actually wetlands but weren't delineated because they were outside the original project scope.

If the model trains on "negative" samples (pixels labeled as "not wetland") from these un-delineated areas, it could learn to classify real wetlands as non-wetlands.

## Current Implementation: NWI-Filtered Negative Sampling

The codebase implements a tile-level sampling strategy in `src/wetlands_ml_atwell/training/sampling.py`.

### How It Works

1. **Acquisition AOI ("Big Box")**: Bounding box around Atwell labels + 1000m buffer, used for imagery download

2. **Safe Zone Computation** (`_compute_safe_zone()`):
   ```
   Safe Zone = AOI - (Atwell polygons + 100m buffer) - (NWI polygons)
   ```
   - Excludes verified wetlands (Atwell) with a 100m buffer
   - Excludes potential wetlands from National Wetlands Inventory (NWI)
   - What remains is "provably safe" for negative sampling

3. **Tile Selection**:
   - **Positive tiles**: Any tile intersecting an Atwell polygon
   - **Negative tiles**: Tiles whose **centroid** falls within the Safe Zone

### Key Code Locations

| File | Function | Purpose |
|------|----------|---------|
| `training/sampling.py` | `create_acquisition_aoi()` | Creates "Big Box" AOI |
| `training/sampling.py` | `_compute_safe_zone()` | Computes safe negative region |
| `training/sampling.py` | `generate_training_manifest()` | Selects positive/negative tiles |
| `tests/experimental/test_negative_sampling.py` | - | Test with real Michigan data |

## Identified Limitation: Centroid Test

The current approach uses a **centroid test** for negative tile selection (lines 233-240 in `sampling.py`):

```python
# Find tiles whose CENTROIDS are in the Safe Zone
# This is a fast proxy for "mostly safe"
# For strict safety, check 'within', but that discards edge tiles.
candidates_centroids = candidate_negatives.copy()
candidates_centroids['geometry'] = candidates_centroids.centroid
```

### The Problem

- Tiles are 5120m × 5120m (512px at 10m resolution)
- A tile's centroid could be in the safe zone, but the tile extends **2560m in each direction**
- Up to half the tile could overlap NWI or buffered Atwell areas
- Any NWI wetlands in the overlapping portion would be labeled as "non-wetland" → **false negatives leak into training**

```
     ← 2560m →← 2560m →
    ┌─────────┬─────────┐
    │         │         │
    │   Safe  │  NWI    │  ← Corner overlaps NWI
    │   Zone  │  area   │    even though centroid is "safe"
    │         ●         │
    │         │         │
    └─────────┴─────────┘
              ↑
        Centroid (in safe zone)
```

### Quick Fix Option

Increase buffer to account for tile geometry:

```python
safe_buffer = buffer_dist + (tile_size_m / 2)  # ~2660m total
```

This guarantees that if a centroid is in the safe zone, the entire tile is safe. However, this significantly shrinks the safe zone and may reduce available negative samples.

## Proposed Better Approach: Pixel-Level Validity Masking

Instead of tile-level decisions, create a **per-pixel validity mask** that indicates which pixels to train on.

### Concept

Separate two questions:
1. **What is the label?** → Label raster (0 = not wetland, 1 = wetland)
2. **Do we trust this label?** → Validity mask (0 = ignore, 1 = train)

### Validity Mask Construction

```python
def create_validity_mask(aoi, atwell, nwi, transform, shape, buffer_dist=100.0):
    """
    Valid (1): Atwell polygons + safe zone
    Invalid (0): NWI polygons + buffer around Atwell edges
    """
    validity = np.ones(shape, dtype=np.uint8)

    # NWI areas → invalid (potential false negatives)
    nwi_mask = rasterize(nwi.geometry, out_shape=shape, transform=transform)
    validity[nwi_mask == 1] = 0

    # Buffer zone around Atwell (edge uncertainty)
    atwell_buffer = atwell.buffer(buffer_dist).difference(atwell.unary_union)
    buffer_mask = rasterize(atwell_buffer, out_shape=shape, transform=transform)
    validity[buffer_mask == 1] = 0

    return validity
```

### Training Integration

```python
# Standard loss
loss = CrossEntropy(prediction, label, reduction='none')  # [H, W]

# Apply validity mask
loss = loss * validity_mask  # uncertain pixels contribute 0
loss = loss.sum() / validity_mask.sum()  # normalize by valid pixels only
```

### Pixel Behavior

| Pixel Location | Label | Validity | Training Effect |
|----------------|-------|----------|-----------------|
| Inside Atwell polygon | 1 | 1 | Trains "this is wetland" |
| Safe zone | 0 | 1 | Trains "this is not wetland" |
| Inside NWI (not Atwell) | 0 | **0** | **Ignored** |
| Buffer around Atwell | 0 | **0** | **Ignored** |

### Advantages Over Tile-Level Approach

| Aspect | Tile-Level | Pixel-Level |
|--------|------------|-------------|
| Granularity | Coarse (5120m tiles) | Fine (10m pixels) |
| Data efficiency | Discards entire tiles | Uses all safe pixels |
| Edge handling | Problematic (centroid hack) | Natural |
| Buffer size | Must be huge (~2560m) | Can be ecologically appropriate (~100m) |
| Complex geometries | Struggles | Works naturally |

## Implementation Considerations

### What Would Need to Change

1. **New module**: `validity.py` - functions to create validity masks
2. **Tile extraction**: Export validity mask as additional channel or separate file
3. **Training loop**: Custom loss function that applies validity weighting
4. **geoai integration**: Need to investigate how to integrate with geoai's training

### Open Question: geoai Integration

The main complexity is integrating with geoai's training loop. Options to investigate:

1. Does geoai support sample/pixel weighting natively?
2. Can we create a custom `Dataset` that yields `(image, label, validity)` tuples?
3. Can we use a custom loss function that reads validity from an extra channel?
4. Do we need to modify geoai or wrap its training loop?

## Next Steps

1. **Investigate geoai** - Understand how geoai handles training to find the cleanest integration path for validity masking
2. **Prototype validity mask generation** - Create the rasterization logic
3. **Evaluate quick fix** - Consider implementing the buffer increase as an interim solution
4. **Decide on approach** - Choose between:
   - Quick fix (buffer increase) - simple but wasteful of data
   - Full pixel-level masking - correct but requires more work

## Related Files

- `src/wetlands_ml_atwell/training/sampling.py` - Current implementation
- `tests/experimental/test_negative_sampling.py` - Test infrastructure
- `src/wetlands_ml_atwell/training/unet.py` - Training orchestration
- `src/wetlands_ml_atwell/training/extraction.py` - Tile extraction

## References

- NWI (National Wetlands Inventory): Federal wetland data used as exclusion mask
- geoai-py: Training framework wrapping segmentation_models_pytorch
