"""Seasonal compositing utilities for Sentinel-2 imagery.

This module provides functions for computing cloud-free seasonal median
composites from Sentinel-2 time series, and for writing and concatenating
the resulting data.

Seasonal Compositing Pipeline
-----------------------------
1. Stack all Sentinel-2 bands for the time period
2. Build cloud/shadow mask from SCL layer
3. Apply mask to exclude cloudy observations
4. Compute pixel-wise median across time
5. Validate output ranges (reflectance should be 0-1)

The seasonal median approach produces stable, cloud-free composites that
capture the typical spectral response for each season.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
import xarray as xr
from dask import compute as dask_compute
from pystac import Item

from ..stacking import FLOAT_NODATA
from .cloud_masking import build_mask, stack_scl
from .progress import LoggingProgressBar, format_duration
from .stac_client import SENTINEL_BANDS, stack_bands

LOGGER = logging.getLogger(__name__)


def seasonal_median(
    items: Sequence[Item],
    season: str,
    min_clear_obs: int,
    bounds: Tuple[float, float, float, float],
    mask_dilation: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute cloud-free seasonal median composite from Sentinel-2 scenes.
    
    This function:
    1. Creates a band stack from all input scenes
    2. Builds a cloud/shadow mask from the SCL layer
    3. Applies the mask to exclude cloudy pixels
    4. Computes the median reflectance for each pixel across time
    5. Validates that output values are in the expected [0, 1] range
    
    Args:
        items: Sequence of Sentinel-2 STAC items for the season.
        season: Season identifier (e.g., "SPR", "SUM", "FAL") for logging.
        min_clear_obs: Minimum number of clear observations required per pixel.
            Pixels with fewer clear observations are set to nodata.
        bounds: Bounding box in lat/lon (minx, miny, maxx, maxy).
        mask_dilation: Number of pixels to dilate the cloud mask.
        
    Returns:
        Tuple of:
        - median: 3D DataArray (band, y, x) with median reflectance values
        - valid_counts: 2D DataArray (y, x) with count of clear observations
        
    Raises:
        ValueError: If reflectance values are outside expected [0, 1] range.
        
    Note:
        This function triggers dask computation and may be memory-intensive
        for large areas or many input scenes.
    """
    logging.info("Season %s: Building band stack from %d scenes...", season, len(items))
    t0 = time.perf_counter()
    stack = stack_bands(items, bounds)
    logging.info(
        "Season %s: Band stack prepared (lazy) in %s",
        season,
        format_duration(time.perf_counter() - t0),
    )

    logging.info("Season %s: Building SCL stack...", season)
    scl = stack_scl(items, bounds)

    mask = build_mask(scl, dilation=mask_dilation)
    mask3d = mask.expand_dims(band=stack.coords["band"]).transpose("time", "band", "y", "x")
    masked = stack.where(mask3d, other=np.nan)
    valid_counts = mask.astype("int16").sum(dim="time")

    median = masked.median(dim="time", skipna=True)
    clear_enough = (
        valid_counts >= min_clear_obs
    ).expand_dims(band=median.coords["band"]).transpose("band", "y", "x")
    median = median.where(clear_enough)

    median.rio.write_crs(stack.rio.crs, inplace=True)
    median.rio.write_transform(stack.rio.transform(), inplace=True)

    median = median.astype("float32")
    valid_counts = valid_counts.astype("int16")

    compute_label = f"Sentinel-2 {season} median"
    logging.info("Season %s: Computing median composite...", season)
    with LoggingProgressBar(compute_label, step=5):
        median, valid_counts = dask_compute(median, valid_counts)

    data = median.values
    if np.isfinite(data).any():
        max_val = float(np.nanmax(data))
        min_val = float(np.nanmin(data))
        if max_val > 1.0 + 1e-3:
            raise ValueError(f"Season {season}: reflectance exceeds 1.0 (max={max_val})")
        if min_val < -1e-3:
            raise ValueError(f"Season {season}: reflectance below 0.0 (min={min_val})")
    else:
        logging.warning("Season %s produced no clear pixels after masking.", season)

    return median, valid_counts


def write_dataarray(
    array: xr.DataArray,
    path: Path,
    band_labels: Sequence[str],
    nodata: float = FLOAT_NODATA,
) -> None:
    """Write an xarray DataArray to a GeoTIFF file.
    
    Writes a multi-band raster with compression and tiling, and sets
    band descriptions from the provided labels.
    
    Args:
        array: 3D DataArray (band, y, x) to write.
        path: Output file path. Parent directories are created if needed.
        band_labels: List of band description strings, one per band.
        nodata: Nodata value to set in the output file.
        
    Raises:
        ValueError: If number of band labels doesn't match array bands.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.sizes["band"] != len(band_labels):
        raise ValueError("Band label count does not match array bands")

    logging.info("Writing raster to %s", path.name)
    array = array.assign_coords({"band": np.arange(1, array.sizes["band"] + 1)})
    array.rio.write_nodata(nodata, inplace=True)
    array.rio.to_raster(
        path,
        dtype="float32",
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(path, "r+") as dst:
        for idx, label in enumerate(band_labels, start=1):
            dst.set_band_description(idx, label)


def concatenate_seasons(
    seasonal: Dict[str, xr.DataArray],
    order: Sequence[str],
) -> Tuple[xr.DataArray, List[str]]:
    """Concatenate seasonal composites into a multi-season stack.
    
    Combines individual seasonal composites into a single multi-band
    array, with band names prefixed by season (e.g., "S2_SPR_B04").
    
    Args:
        seasonal: Dictionary mapping season names to their composite arrays.
        order: Sequence specifying the order of seasons in output
            (e.g., ["SPR", "SUM", "FAL"]).
            
    Returns:
        Tuple of:
        - combined: Concatenated 3D DataArray (band, y, x)
        - labels: List of band label strings (e.g., ["S2_SPR_B03", ...])
        
    Example:
        >>> composites = {"SPR": spring_arr, "SUM": summer_arr}
        >>> combined, labels = concatenate_seasons(composites, ["SPR", "SUM"])
        >>> print(labels[:2])
        ['S2_SPR_B03', 'S2_SPR_B04']
    """
    arrays = []
    labels: List[str] = []
    for season in order:
        arrays.append(seasonal[season])
        labels.extend([f"S2_{season}_{band}" for band in SENTINEL_BANDS])
    combined = xr.concat(arrays, dim="band")
    combined.rio.write_crs(arrays[0].rio.crs, inplace=True)
    combined.rio.write_transform(arrays[0].rio.transform(), inplace=True)
    return combined.astype("float32"), labels


__all__ = [
    "seasonal_median",
    "write_dataarray",
    "concatenate_seasons",
]

