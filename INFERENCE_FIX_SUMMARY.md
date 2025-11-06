# Inference Normalization Fix - Summary

## Issue Identified

**Problem:** UNet model trained with val IoU ≈0.79 was predicting only background during inference (max class-1 probability ≈1.7e-4, zero pixels >0.5).

**Root Cause:** Normalization mismatch between training and inference caused by `geoai`'s preprocessing.

## Technical Details

### Training Pipeline
1. Training tiles are written with `normalize_stack_array()` which:
   - Replaces nodata with 0
   - Clips values to [0, 1] range
   - Saves tiles with values in [0, 1]

2. `geoai.train_segmentation_model()` loads tiles via `SemanticSegmentationDataset`:
   ```python
   # geoai/train.py line 1690
   image = image / 255.0
   ```
   - Divides by 255 AGAIN
   - **Actual training data range: [0, ~0.004]** (not [0, 1])

### Inference Pipeline (BEFORE FIX)
1. Custom `infer_manifest()` used `normalize_stack_array()`:
   - Only clipped to [0, 1]
   - **No division by 255**
   - **Inference data range: [0, 1]**

### The Mismatch
- **Training:** Model learned on values in `[0, 0.004]` range
- **Inference:** Model received values in `[0, 1]` range (~250x larger)
- **Result:** Model completely confused → predicts only background

## Solution Implemented

### File Modified
`src/wetlands_ml_geoai/inference/unet_stream.py`

### Change Made
Modified `_prepare_window()` function to match training normalization:

```python
def _prepare_window(
    data: np.ndarray,
    desired_channels: int,
    nodata_value: Optional[float],
) -> np.ndarray:
    # ... channel handling code ...
    
    # Normalize using the same approach as training:
    # 1. Clean nodata, clip to [0,1]
    # 2. Divide by 255 (matching geoai's SemanticSegmentationDataset)
    array = normalize_stack_array(array, nodata_value)
    array = array / 255.0  # ← FIX: Added this line
    return array
```

## Validation Results

### Before Fix
```
Input range: [0.000000, 1.000000]
Class 1 probs: min=0.000000, max=0.000000
Pixels with class 1 > 0.5: 0
```

### After Fix
```
Input range: [0.000000, 0.003922]
Class 1 probs: min=0.000005, max=0.976188
Pixels with class 1 > 0.5: 6033
```

## Impact

### What's Fixed
✅ **Manifest-based streaming inference** (`infer_manifest()`) now produces correct predictions

### What Wasn't Affected
- **Direct raster inference** using `geoai.semantic_segmentation()` was already working correctly
  - `geoai` applies `/255` normalization internally during inference (line 2718 in `train.py`)
  - This path was always consistent

## Future Recommendations

### 1. Consider Standardizing Normalization
To avoid similar issues:
- **Option A:** Don't pre-normalize training tiles, let `geoai` handle all normalization
- **Option B:** Create custom dataset class that doesn't apply `/255` when tiles are pre-normalized

### 2. Add Normalization Tests
Add unit tests that verify:
```python
def test_normalization_consistency():
    # Ensure training and inference normalization match
    training_data = prepare_training_tile(raw_data)
    inference_data = prepare_inference_tile(raw_data)
    assert np.allclose(training_data, inference_data)
```

### 3. Document Preprocessing Pipeline
Maintain clear documentation of:
- Input data range expectations
- Normalization steps at each stage
- Expected model input range

## Notes

- The model checkpoint (`best_model.pth`) does NOT need to be retrained
- The model was trained correctly and learned the task well (val IoU 0.79)
- Only inference preprocessing needed correction
- All future inferences will now work correctly with the existing model

