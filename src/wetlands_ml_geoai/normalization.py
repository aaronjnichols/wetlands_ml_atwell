"""Normalization utilities for geoai-compatible tile formats.

=============================================================================
DATA FLOW AND NORMALIZATION EXPLAINED
=============================================================================

This module documents and handles the normalization pipeline required for
compatibility with geoai's training system.

THE PROBLEM
-----------
geoai's SemanticSegmentationDataset (geoai/train.py, line ~1690) ALWAYS
divides input images by 255, assuming uint8 [0-255] input:

    image = src.read().astype(np.float32)
    image = image / 255.0  # <-- hardcoded, no option to skip

This means:
- If we provide uint8 [0-255] tiles → geoai produces [0-1] ✓ correct
- If we provide float32 [0-1] tiles → geoai produces [0-0.004] ✗ wrong scale

THE SOLUTION
------------
We convert all normalized data to uint8 [0-255] format before writing tiles,
so that geoai's /255 division produces the correct [0-1] range.

DATA FLOW
---------
1. Raw data comes in various formats:
   - NAIP: uint8 [0-255]
   - Sentinel-2: int16 [0-10000], scaled by /10000
   - Topography: float32 with varying ranges, scaled by band_scaling
   
2. RasterStack.read_window() normalizes ALL sources to float32 [0-1]:
   - NAIP: divided by scale_max (255)
   - Sentinel-2: already scaled to [0-1] in compositing
   - Topography: min-max scaled via band_scaling
   
3. normalize_stack_array() cleans nodata and clips to [0-1]

4. to_geoai_format() converts [0-1] float32 → [0-255] uint8

5. Tiles written as uint8 [0-255]

6. geoai training reads tiles and divides by 255 → [0-1] model input ✓

7. Inference uses the same normalization (handled by geoai for direct raster,
   or by our code for streaming manifest inference)

BACKWARD COMPATIBILITY
----------------------
Existing float32 [0-1] tiles are still supported:
- Inference code detects dtype
- If float32: applies /255 to match the [0-0.004] range the model learned on
- If uint8: no workaround needed, geoai's /255 produces correct [0-1]

See INFERENCE_FIX_SUMMARY.md for the full history of this issue.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


def to_geoai_format(
    data: np.ndarray,
    validate: bool = True,
) -> np.ndarray:
    """Convert normalized [0-1] float data to uint8 [0-255] for geoai compatibility.
    
    geoai's SemanticSegmentationDataset always divides by 255, assuming uint8 input.
    This function prepares data in the format geoai expects.
    
    Args:
        data: Input array with values in [0, 1] range (float32).
        validate: If True, warn when values are outside expected range.
        
    Returns:
        uint8 array with values in [0, 255] range.
        
    Example:
        >>> normalized = normalize_stack_array(raw_data)  # [0, 1] float32
        >>> tile_data = to_geoai_format(normalized)       # [0, 255] uint8
        >>> # Write tile_data to GeoTIFF with dtype=uint8
        >>> # geoai will read and divide by 255, getting [0, 1]
    
    Why uint8?
        geoai hardcodes `image = image / 255.0` in its dataset class.
        By providing uint8 [0-255], geoai's division produces the correct
        [0-1] range that neural networks expect.
        
        If we provided float32 [0-1], geoai would produce [0-0.004], which
        is technically fine (the model learns different weight scales) but
        is confusing and non-standard.
    """
    if validate:
        valid_mask = np.isfinite(data)
        if valid_mask.any():
            min_val = float(data[valid_mask].min())
            max_val = float(data[valid_mask].max())
            
            if min_val < 0.0 or max_val > 1.0:
                LOGGER.warning(
                    "Input data outside [0, 1] range (min=%.4f, max=%.4f). "
                    "Values will be clipped before conversion to uint8. "
                    "This may indicate missing normalization upstream.",
                    min_val,
                    max_val,
                )
    
    # Clip to [0, 1] to ensure valid uint8 output
    clipped = np.clip(data, 0.0, 1.0)
    
    # Scale to [0, 255] and convert to uint8
    scaled = (clipped * 255.0).astype(np.uint8)
    
    return scaled


def prepare_for_model(
    data: np.ndarray,
    source_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Prepare tile data for model inference, handling both uint8 and float32 tiles.
    
    This function provides backward compatibility with existing float32 tiles
    while also supporting the new uint8 format.
    
    Args:
        data: Input array from a tile (either uint8 or float32).
        source_dtype: Original dtype of the tile. If None, inferred from data.
        
    Returns:
        float32 array normalized for model input.
        
    Normalization logic:
        - uint8 tiles: Assumed to be [0-255]. Model will receive [0-1] after
          geoai's /255 (for training) or our own /255 (for inference).
        - float32 tiles [0-1]: Legacy format. We apply /255 to produce
          [0-0.004], matching what the model learned during training when
          geoai applied /255 to these tiles.
          
    Note:
        For streaming inference with manifests, we apply the normalization
        ourselves since we're not going through geoai's dataset loader.
    """
    dtype = source_dtype if source_dtype is not None else data.dtype
    
    if np.issubdtype(dtype, np.integer):
        # uint8 tile: divide by 255 to get [0, 1]
        # This matches what geoai does during training
        LOGGER.debug("Processing uint8 tile - dividing by 255 for [0-1] range")
        return data.astype(np.float32) / 255.0
    else:
        # float32 tile (legacy format): divide by 255 to match training
        # During training, geoai read these [0-1] tiles and divided by 255,
        # so the model learned on [0-0.004] range. We must match that.
        LOGGER.debug(
            "Processing float32 tile (legacy format) - dividing by 255 "
            "to match training normalization"
        )
        return data.astype(np.float32) / 255.0


# Constants documenting the expected ranges at each pipeline stage
GEOAI_EXPECTED_INPUT_DTYPE = np.uint8
GEOAI_EXPECTED_INPUT_RANGE = (0, 255)
MODEL_INPUT_RANGE = (0.0, 1.0)
LEGACY_TILE_RANGE = (0.0, 1.0)  # float32 tiles from before this fix


__all__ = [
    "to_geoai_format",
    "prepare_for_model",
    "GEOAI_EXPECTED_INPUT_DTYPE",
    "GEOAI_EXPECTED_INPUT_RANGE",
    "MODEL_INPUT_RANGE",
    "LEGACY_TILE_RANGE",
]

