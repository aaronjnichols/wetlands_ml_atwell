"""Cloud and shadow masking utilities for Sentinel-2 imagery.

This module provides functions for building cloud/shadow masks from Sentinel-2
Scene Classification Layer (SCL) data.

SCL Values Reference
--------------------
The Scene Classification Layer contains the following class values:

| Value | Class Description              | Masked? |
|-------|--------------------------------|---------|
| 0     | No data                        | No      |
| 1     | Saturated or defective         | No      |
| 2     | Dark area pixels               | No      |
| 3     | Cloud shadows                  | Yes     |
| 4     | Vegetation                     | No      |
| 5     | Not vegetated                  | No      |
| 6     | Water                          | No      |
| 7     | Unclassified                   | No      |
| 8     | Cloud medium probability       | Yes     |
| 9     | Cloud high probability         | Yes     |
| 10    | Thin cirrus                    | Yes     |
| 11    | Snow                           | Yes     |

By default, values 3, 8, 9, 10, and 11 are masked (clouds, shadows, cirrus, snow).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Set, Tuple

import numpy as np
import stackstac
import xarray as xr
from pystac import Item
from skimage.morphology import binary_dilation

LOGGER = logging.getLogger(__name__)

# SCL asset identifier in STAC items
SCL_ASSET_ID = "scl"

# Default SCL values to mask: cloud shadows (3), cloud medium (8), 
# cloud high (9), thin cirrus (10), snow (11)
SCL_MASK_VALUES: Set[int] = {3, 8, 9, 10, 11}


def stack_scl(
    items: Sequence[Item],
    bounds: Tuple[float, float, float, float],
    chunks: Optional[int] = 2048,
) -> xr.DataArray:
    """Create Scene Classification Layer (SCL) stack from STAC items.
    
    The SCL band provides per-pixel classification of cloud, shadow,
    vegetation, water, and other surface types.
    
    Args:
        items: Sentinel-2 STAC items to stack.
        bounds: Bounding box in lat/lon (minx, miny, maxx, maxy).
        chunks: Dask chunk size for x/y dimensions. None disables chunking.
        
    Returns:
        3D DataArray with dimensions (time, y, x) containing SCL values.
        
    Raises:
        ValueError: If items are missing proj:epsg or SCL asset.
    """
    epsg = items[0].properties.get("proj:epsg")
    if epsg is None:
        raise ValueError("Sentinel-2 item missing proj:epsg metadata.")
    if SCL_ASSET_ID not in items[0].assets:
        raise ValueError(f"Sentinel-2 item missing {SCL_ASSET_ID} asset.")
    
    scl = stackstac.stack(
        items,
        assets=[SCL_ASSET_ID],
        resolution=10,
        epsg=int(epsg),
        bounds_latlon=bounds,
        chunksize={"x": chunks, "y": chunks} if chunks else None,
        dtype="float64",
        rescale=False,
        properties=False,
    ).squeeze("band")
    
    scl = scl.reset_coords(drop=True)
    scl.rio.write_crs(int(epsg), inplace=True)
    return scl


def build_mask(
    scl: xr.DataArray,
    dilation: int = 0,
    mask_values: Optional[Set[int]] = None,
) -> xr.DataArray:
    """Build binary clear-sky mask from Scene Classification Layer.
    
    Creates a boolean mask where True indicates clear (usable) pixels
    and False indicates cloudy/shadowy/problematic pixels.
    
    Args:
        scl: Scene Classification Layer DataArray with dimensions (time, y, x).
        dilation: Number of pixels to dilate the cloud mask. Helps eliminate
            cloud edges which may be partially cloudy. Default 0 (no dilation).
        mask_values: Set of SCL values to mask. Defaults to SCL_MASK_VALUES
            (cloud shadows, clouds, cirrus, snow).
            
    Returns:
        Boolean DataArray where True = clear pixel, False = masked pixel.
        Same dimensions as input SCL.
        
    Example:
        >>> scl = stack_scl(items, bounds)
        >>> mask = build_mask(scl, dilation=2)
        >>> clear_composite = bands.where(mask).median(dim="time")
    """
    if mask_values is None:
        mask_values = SCL_MASK_VALUES
    
    # Start with all pixels as valid (True)
    mask = xr.ones_like(scl, dtype=bool)
    
    # Mark masked values as invalid (False)
    for value in mask_values:
        mask = mask & (scl != value)

    if dilation > 0:
        # Identify cloudy pixels (inverse of mask)
        cloudy = ~mask

        def _dilate(arr: np.ndarray) -> np.ndarray:
            return binary_dilation(arr, iterations=dilation)

        # Apply dilation to expand cloud mask
        dilated = xr.apply_ufunc(
            _dilate,
            cloudy,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[bool],
        )
        # Update mask to exclude dilated cloud areas
        mask = mask & ~dilated
        
    return mask


__all__ = [
    "SCL_ASSET_ID",
    "SCL_MASK_VALUES",
    "stack_scl",
    "build_mask",
]

