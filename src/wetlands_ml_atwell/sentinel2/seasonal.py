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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import xarray as xr
import dask
from dask import compute as dask_compute
from pystac import Item

from ..stacking import FLOAT_NODATA
from .cloud_masking import build_mask, stack_scl
from .progress import LoggingProgressBar, format_duration
from .stac_client import SENTINEL_BANDS, stack_bands

LOGGER = logging.getLogger(__name__)

# Configure dask for multi-threaded execution
# This is critical for performance on multi-core systems
dask.config.set(scheduler="threads", num_workers=None)  # None = use all available cores


def seasonal_median(
    items: Sequence[Item],
    season: str,
    min_clear_obs: int,
    bounds: Tuple[float, float, float, float],
    mask_dilation: int,
    chunks: Optional[int] = None,
    target_crs: Optional[str] = None,
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
        chunks: Dask chunk size for x/y dimensions. If None, uses default (2048).
            Use compute_chunk_size() to auto-calculate based on AOI size.
        target_crs: Target CRS for output (e.g., "EPSG:5070"). If None,
            uses the first item's native CRS.

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
    stack = stack_bands(items, bounds, chunks=chunks, target_crs=target_crs)
    logging.info(
        "Season %s: Band stack prepared (lazy) in %s",
        season,
        format_duration(time.perf_counter() - t0),
    )

    logging.info("Season %s: Building SCL stack...", season)
    scl = stack_scl(items, bounds, chunks=chunks, target_crs=target_crs)

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
        median, valid_counts = dask_compute(median, valid_counts, scheduler="threads")

    data = median.values
    if np.isfinite(data).any():
        max_val = float(np.nanmax(data))
        min_val = float(np.nanmin(data))
        if max_val > 1.0 + 1e-3 or min_val < -1e-3:
            logging.warning(
                "Season %s: reflectance outside [0, 1] range (min=%.4f, max=%.4f). "
                "Clipping to valid range.",
                season, min_val, max_val
            )
            median = median.clip(min=0.0, max=1.0)
    else:
        logging.warning("Season %s produced no clear pixels after masking.", season)

    return median, valid_counts


def seasonal_median_parallel(
    items: Sequence[Item],
    season: str,
    min_clear_obs: int,
    bounds: Tuple[float, float, float, float],
    mask_dilation: int,
    chunks: Optional[int] = 512,
    max_workers: int = 24,
    target_crs: Optional[str] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute cloud-free seasonal median using parallel pre-download.

    This function bypasses stackstac's lazy sequential reads by downloading
    all scene data for each spatial chunk in parallel before computing
    the median. This can provide 5-10x speedup for areas with many scenes.

    Args:
        items: Sequence of Sentinel-2 STAC items for the season.
        season: Season identifier for logging.
        min_clear_obs: Minimum clear observations per pixel.
        bounds: Bounding box in lat/lon (minx, miny, maxx, maxy).
        mask_dilation: Pixels to dilate cloud mask.
        chunks: Spatial chunk size for processing. Smaller = less memory.
        max_workers: Number of parallel download workers.
        target_crs: Target CRS for output (e.g., "EPSG:5070"). If None,
            uses the first item's native CRS.

    Returns:
        Tuple of (median DataArray, valid_counts DataArray).
    """
    from pystac.extensions.projection import ProjectionExtension
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.warp import transform_bounds

    from .parallel_fetch import (
        ChunkSpec,
        fetch_scenes_parallel,
        compute_chunk_median,
    )

    if not items:
        raise ValueError("No items provided for median computation")

    # Determine output CRS: use target_crs if provided, else first item's CRS
    if target_crs:
        # Parse EPSG from string like "EPSG:5070" or just "5070"
        if ":" in target_crs:
            epsg = int(target_crs.split(":")[1])
        else:
            epsg = int(target_crs)
        crs = target_crs if ":" in target_crs else f"EPSG:{target_crs}"
    else:
        # Fall back to first item's EPSG (legacy behavior)
        try:
            proj_ext = ProjectionExtension.ext(items[0])
            epsg = proj_ext.epsg
        except Exception:
            epsg = items[0].properties.get("proj:epsg")

        if epsg is None:
            raise ValueError("Missing proj:epsg on items and no target_crs specified")

        crs = f"EPSG:{epsg}"

    resolution = 10.0  # Sentinel-2 target resolution

    # Convert bounds to target CRS
    dst_bounds = transform_bounds("EPSG:4326", crs, *bounds)

    # Compute output dimensions
    minx, miny, maxx, maxy = dst_bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)

    LOGGER.info(
        "Season %s: Output grid %d x %d pixels, %d scenes, parallel fetch enabled",
        season,
        width,
        height,
        len(items),
    )

    # Initialize output arrays
    n_bands = len(SENTINEL_BANDS)
    median_data = np.full((n_bands, height, width), FLOAT_NODATA, dtype=np.float32)
    counts_data = np.zeros((height, width), dtype=np.int16)

    # Calculate chunk boundaries
    chunk_size = chunks or 512
    n_chunks_x = (width + chunk_size - 1) // chunk_size
    n_chunks_y = (height + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_x * n_chunks_y

    LOGGER.info(
        "Season %s: Processing %d chunks (%dx%d grid) with %d workers",
        season,
        total_chunks,
        n_chunks_x,
        n_chunks_y,
        max_workers,
    )

    t0 = time.perf_counter()

    chunk_idx = 0
    for row in range(n_chunks_y):
        for col in range(n_chunks_x):
            chunk_idx += 1

            # Calculate chunk window
            x_off = col * chunk_size
            y_off = row * chunk_size
            chunk_width = min(chunk_size, width - x_off)
            chunk_height = min(chunk_size, height - y_off)

            # Calculate bounds for this chunk in target CRS
            chunk_minx = minx + x_off * resolution
            chunk_maxy = maxy - y_off * resolution
            chunk_maxx = chunk_minx + chunk_width * resolution
            chunk_miny = chunk_maxy - chunk_height * resolution

            chunk_transform = transform_from_bounds(
                chunk_minx, chunk_miny, chunk_maxx, chunk_maxy, chunk_width, chunk_height
            )

            chunk_spec = ChunkSpec(
                bounds=(chunk_minx, chunk_miny, chunk_maxx, chunk_maxy),
                transform=chunk_transform,
                width=chunk_width,
                height=chunk_height,
                crs=crs,
                epsg=epsg,
            )

            LOGGER.info(
                "Season %s: Fetching chunk %d/%d (%d x %d pixels)",
                season,
                chunk_idx,
                total_chunks,
                chunk_width,
                chunk_height,
            )

            # Parallel download all scenes for this chunk
            results = fetch_scenes_parallel(items, chunk_spec, max_workers=max_workers)

            if not any(r.success for r in results):
                LOGGER.warning("Season %s: Chunk %d has no valid data", season, chunk_idx)
                continue

            # Compute median for this chunk
            try:
                chunk_median, chunk_counts = compute_chunk_median(
                    results, min_clear_obs, mask_dilation, fill_value=FLOAT_NODATA
                )

                # Insert into output arrays
                median_data[
                    :, y_off : y_off + chunk_height, x_off : x_off + chunk_width
                ] = chunk_median

                counts_data[
                    y_off : y_off + chunk_height, x_off : x_off + chunk_width
                ] = chunk_counts

            except ValueError as e:
                LOGGER.warning("Season %s: Chunk %d median failed: %s", season, chunk_idx, e)
                continue

            elapsed = time.perf_counter() - t0
            rate = chunk_idx / elapsed if elapsed > 0 else 0
            remaining = (total_chunks - chunk_idx) / rate if rate > 0 else 0

            LOGGER.info(
                "Season %s: Chunk %d/%d complete (%.1f chunks/min, ~%.1fm remaining)",
                season,
                chunk_idx,
                total_chunks,
                rate * 60,
                remaining / 60,
            )

    elapsed = time.perf_counter() - t0
    LOGGER.info("Season %s: Completed in %s", season, format_duration(elapsed))

    # Validate and clip reflectance values
    valid_mask = median_data != FLOAT_NODATA
    if valid_mask.any():
        valid_data = median_data[valid_mask]
        max_val = float(np.max(valid_data))
        min_val = float(np.min(valid_data))
        if max_val > 1.0 + 1e-3 or min_val < -1e-3:
            LOGGER.warning(
                "Season %s: reflectance outside [0, 1] range (min=%.4f, max=%.4f). "
                "Clipping to valid range.",
                season,
                min_val,
                max_val,
            )
            median_data = np.clip(median_data, 0.0, 1.0)
            # Restore nodata values
            median_data = np.where(valid_mask, median_data, FLOAT_NODATA)
    else:
        LOGGER.warning("Season %s produced no clear pixels after masking.", season)

    # Create xarray DataArrays with proper metadata
    y_coords = np.arange(height) * -resolution + maxy - resolution / 2
    x_coords = np.arange(width) * resolution + minx + resolution / 2

    median_da = xr.DataArray(
        median_data,
        dims=["band", "y", "x"],
        coords={
            "band": SENTINEL_BANDS,
            "y": y_coords,
            "x": x_coords,
        },
    )
    median_da.rio.write_crs(crs, inplace=True)
    median_da.rio.write_transform(transform, inplace=True)

    counts_da = xr.DataArray(
        counts_data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )

    return median_da, counts_da


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
    "seasonal_median_parallel",
    "write_dataarray",
    "concatenate_seasons",
]

