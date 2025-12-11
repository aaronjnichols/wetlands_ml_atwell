"""Parallel fetching utilities for Sentinel-2 COG data.

This module provides functions for downloading Sentinel-2 scene data in parallel
using ThreadPoolExecutor, bypassing stackstac's sequential reads.

The key optimization is that while stackstac reads scenes sequentially within
each spatial chunk (to_dask.py:179-189), this module downloads all scene data
for a chunk in parallel before computing the median.

Performance Impact
------------------
- stackstac: Sequential reads within each chunk (~600 HTTP reads in series)
- parallel_fetch: Concurrent reads using ThreadPoolExecutor (24+ parallel)
- Expected speedup: 5-10x for network-bound operations
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from affine import Affine
from pystac import Item

from .stac_client import BAND_TO_ASSET, SENTINEL_BANDS, SENTINEL_SCALE_FACTOR
from .cloud_masking import SCL_MASK_VALUES

LOGGER = logging.getLogger(__name__)

# Default number of parallel workers
DEFAULT_WORKERS = 24

# GDAL environment variables for efficient COG access
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",  # 5MB cache per file
}


@dataclass
class FetchResult:
    """Result of fetching a single scene's data for a spatial window."""

    item_id: str
    time_index: int
    data: np.ndarray  # Shape: (bands, height, width)
    scl_data: np.ndarray  # Shape: (height, width)
    success: bool
    error: Optional[str] = None


@dataclass
class ChunkSpec:
    """Specification for a spatial chunk to fetch."""

    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy in target CRS
    transform: Affine
    width: int
    height: int
    crs: str
    epsg: int


def _get_asset_url(item: Item, asset_key: str) -> Optional[str]:
    """Get the URL for an asset from a STAC item."""
    asset = item.assets.get(asset_key)
    if asset is None:
        return None
    return asset.href


def _fetch_scene_window(
    item: Item,
    time_index: int,
    chunk_spec: ChunkSpec,
    bands: Sequence[str] = SENTINEL_BANDS,
) -> FetchResult:
    """Fetch data for a single scene within a spatial window.

    Args:
        item: STAC Item for the scene.
        time_index: Index of this scene in the time dimension.
        chunk_spec: Specification for the spatial chunk.
        bands: List of band names to fetch.

    Returns:
        FetchResult with data arrays or error information.
    """
    try:
        band_data = []
        dst_crs = RioCRS.from_user_input(chunk_spec.crs)

        with rasterio.Env(**GDAL_ENV):
            # Fetch each band
            for band in bands:
                asset_id = BAND_TO_ASSET[band]
                url = _get_asset_url(item, asset_id)
                if url is None:
                    raise ValueError(f"Missing asset {asset_id} on item {item.id}")

                with rasterio.open(url) as src:
                    # Check if reprojection is needed
                    needs_reproject = src.crs and src.crs != dst_crs

                    if needs_reproject:
                        # Reproject chunk bounds to source CRS to determine read window
                        src_bounds = transform_bounds(
                            chunk_spec.crs, src.crs, *chunk_spec.bounds
                        )

                        # Check if bounds intersect with raster extent
                        raster_bounds = src.bounds
                        intersect_minx = max(src_bounds[0], raster_bounds.left)
                        intersect_miny = max(src_bounds[1], raster_bounds.bottom)
                        intersect_maxx = min(src_bounds[2], raster_bounds.right)
                        intersect_maxy = min(src_bounds[3], raster_bounds.top)

                        if intersect_minx >= intersect_maxx or intersect_miny >= intersect_maxy:
                            # No intersection - scene doesn't cover this chunk at all
                            raise ValueError(f"Scene does not cover chunk bounds")

                        # Use FULL requested bounds for the window
                        # boundless=True will fill outside areas with 0 (S2 nodata)
                        src_window = from_bounds(*src_bounds, transform=src.transform)

                        # Read source data (0 = nodata in Sentinel-2)
                        src_data = src.read(
                            1, window=src_window, boundless=True, fill_value=0
                        ).astype(np.float32)

                        # Calculate source transform for the window
                        src_window_transform = src.window_transform(src_window)

                        # Allocate destination array with 0 (nodata)
                        data = np.zeros(
                            (chunk_spec.height, chunk_spec.width), dtype=np.float32
                        )

                        # Reproject from source CRS to target CRS
                        # Use 0 as nodata so it doesn't contaminate valid pixels during interpolation
                        reproject(
                            source=src_data,
                            destination=data,
                            src_transform=src_window_transform,
                            src_crs=src.crs,
                            dst_transform=chunk_spec.transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=0,
                            dst_nodata=0,
                        )

                        # Scale to reflectance [0, 1] (0 stays 0, which SCL will mask out)
                        data = data * SENTINEL_SCALE_FACTOR
                    else:
                        # Same CRS - just read with window and resample
                        src_window = from_bounds(
                            *chunk_spec.bounds, transform=src.transform
                        )
                        data = src.read(
                            1,
                            window=src_window,
                            out_shape=(chunk_spec.height, chunk_spec.width),
                            resampling=Resampling.bilinear,
                            boundless=True,
                            fill_value=0,
                        ).astype(np.float32)

                        # Scale to reflectance [0, 1]
                        data = data * SENTINEL_SCALE_FACTOR

                    band_data.append(data)

            # Fetch SCL band
            scl_url = _get_asset_url(item, "scl")
            if scl_url is None:
                raise ValueError(f"Missing SCL asset on item {item.id}")

            with rasterio.open(scl_url) as src:
                needs_reproject = src.crs and src.crs != dst_crs

                if needs_reproject:
                    # Reproject chunk bounds to source CRS
                    src_bounds = transform_bounds(
                        chunk_spec.crs, src.crs, *chunk_spec.bounds
                    )

                    # Check intersection with raster extent
                    raster_bounds = src.bounds
                    intersect_minx = max(src_bounds[0], raster_bounds.left)
                    intersect_miny = max(src_bounds[1], raster_bounds.bottom)
                    intersect_maxx = min(src_bounds[2], raster_bounds.right)
                    intersect_maxy = min(src_bounds[3], raster_bounds.top)

                    if intersect_minx >= intersect_maxx or intersect_miny >= intersect_maxy:
                        raise ValueError(f"Scene does not cover chunk bounds for SCL")

                    # Use full bounds, let boundless handle edges
                    src_window = from_bounds(*src_bounds, transform=src.transform)

                    # Read source SCL in native CRS (0 = nodata)
                    src_scl = src.read(
                        1, window=src_window, boundless=True, fill_value=0
                    ).astype(np.float32)
                    src_window_transform = src.window_transform(src_window)

                    # Allocate destination with 0 (nodata for SCL)
                    scl_float = np.zeros(
                        (chunk_spec.height, chunk_spec.width), dtype=np.float32
                    )

                    # Reproject SCL (use nearest for classification data)
                    reproject(
                        source=src_scl,
                        destination=scl_float,
                        src_transform=src_window_transform,
                        src_crs=src.crs,
                        dst_transform=chunk_spec.transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )
                    scl_data = scl_float.astype(np.uint8)
                else:
                    src_window = from_bounds(
                        *chunk_spec.bounds, transform=src.transform
                    )
                    scl_data = src.read(
                        1,
                        window=src_window,
                        out_shape=(chunk_spec.height, chunk_spec.width),
                        resampling=Resampling.nearest,
                        boundless=True,
                        fill_value=0,
                    ).astype(np.uint8)

        return FetchResult(
            item_id=item.id,
            time_index=time_index,
            data=np.stack(band_data, axis=0),
            scl_data=scl_data,
            success=True,
        )

    except Exception as e:
        LOGGER.warning("Failed to fetch scene %s: %s", item.id, str(e))
        return FetchResult(
            item_id=item.id,
            time_index=time_index,
            data=np.array([]),
            scl_data=np.array([]),
            success=False,
            error=str(e),
        )


def fetch_scenes_parallel(
    items: Sequence[Item],
    chunk_spec: ChunkSpec,
    max_workers: int = DEFAULT_WORKERS,
    bands: Sequence[str] = SENTINEL_BANDS,
) -> List[FetchResult]:
    """Fetch data for multiple scenes in parallel.

    Args:
        items: STAC Items to fetch.
        chunk_spec: Spatial chunk specification.
        max_workers: Maximum number of parallel workers.
        bands: List of band names to fetch.

    Returns:
        List of FetchResult objects, one per item, sorted by time index.
    """
    results: List[FetchResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_scene_window, item, idx, chunk_spec, bands): idx
            for idx, item in enumerate(items)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Sort by time index
    results.sort(key=lambda r: r.time_index)

    successful = sum(1 for r in results if r.success)
    LOGGER.debug(
        "Fetched %d/%d scenes successfully for chunk (%d x %d)",
        successful,
        len(items),
        chunk_spec.width,
        chunk_spec.height,
    )

    return results


def compute_chunk_median(
    results: List[FetchResult],
    min_clear_obs: int,
    mask_dilation: int = 0,
    fill_value: float = -9999.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute median composite from fetched scene data.

    Args:
        results: List of FetchResult objects from fetch_scenes_parallel().
        min_clear_obs: Minimum clear observations required per pixel.
        mask_dilation: Pixels to dilate cloud mask.
        fill_value: Value for pixels with insufficient observations.

    Returns:
        Tuple of (median_data, valid_counts) arrays.
        median_data shape: (bands, height, width)
        valid_counts shape: (height, width)
    """
    from skimage.morphology import binary_dilation

    # Filter to successful results
    valid_results = [r for r in results if r.success]
    if not valid_results:
        raise ValueError("No valid scene data to compute median")

    # Get dimensions from first result
    n_bands, height, width = valid_results[0].data.shape
    n_times = len(valid_results)

    # Stack all data: (time, bands, height, width)
    all_data = np.stack([r.data for r in valid_results], axis=0)
    all_scl = np.stack([r.scl_data for r in valid_results], axis=0)

    # Build cloud mask (True = clear)
    mask = np.ones((n_times, height, width), dtype=bool)
    for value in SCL_MASK_VALUES:
        mask = mask & (all_scl != value)

    # Also mask out SCL=0 (nodata in SCL layer - areas outside scene coverage)
    mask = mask & (all_scl != 0)

    # Also mask out pixels where band data is 0 or NaN (nodata from boundless reads)
    # Check first band - if it's 0/NaN, all bands should be 0/NaN
    data_valid = (all_data[:, 0, :, :] != 0) & ~np.isnan(all_data[:, 0, :, :])
    mask = mask & data_valid

    # Apply dilation if requested
    if mask_dilation > 0:
        for t in range(n_times):
            cloudy = ~mask[t]
            dilated = binary_dilation(cloudy, footprint=np.ones((3, 3)), out=None)
            for _ in range(mask_dilation - 1):
                dilated = binary_dilation(dilated, footprint=np.ones((3, 3)), out=None)
            mask[t] = mask[t] & ~dilated

    # Count valid observations per pixel
    valid_counts = mask.sum(axis=0).astype(np.int16)

    # Apply mask to data
    mask_3d = np.broadcast_to(mask[:, np.newaxis, :, :], all_data.shape)
    masked_data = np.where(mask_3d, all_data, np.nan)

    # Compute median
    with np.errstate(all="ignore"):
        median = np.nanmedian(masked_data, axis=0)

    # Apply minimum observation threshold
    clear_enough = valid_counts >= min_clear_obs
    clear_enough_3d = np.broadcast_to(clear_enough[np.newaxis, :, :], median.shape)
    median = np.where(clear_enough_3d, median, fill_value)

    # Replace any remaining NaNs with fill value
    median = np.where(np.isnan(median), fill_value, median)

    return median.astype(np.float32), valid_counts


__all__ = [
    "FetchResult",
    "ChunkSpec",
    "fetch_scenes_parallel",
    "compute_chunk_median",
    "DEFAULT_WORKERS",
]
