"""DEM mosaicking and derivative computation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio import warp, features
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from scipy import ndimage
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from ..stacking import FLOAT_NODATA
from .config import TopographyStackConfig


LOGGER = logging.getLogger(__name__)


def _create_aoi_mask(
    aoi: BaseGeometry,
    transform: Affine,
    width: int,
    height: int,
    crs: str,
) -> np.ndarray:
    """Create a boolean mask where True = inside AOI, False = outside.

    Args:
        aoi: AOI polygon in WGS84 (EPSG:4326).
        transform: Affine transform of the target raster.
        width: Width of the target raster in pixels.
        height: Height of the target raster in pixels.
        crs: CRS string of the target raster.

    Returns:
        Boolean numpy array of shape (height, width).
    """
    aoi_geom = mapping(aoi)

    # Transform AOI geometry from WGS84 to raster CRS
    # If transformation fails (e.g., test fixtures with non-geographic coords),
    # assume geometry is already in target CRS
    try:
        aoi_transformed = transform_geom("EPSG:4326", crs, aoi_geom)
    except Exception as e:
        LOGGER.warning(
            "AOI transformation from EPSG:4326 to %s failed (%s); "
            "assuming geometry is already in target CRS",
            crs,
            e,
        )
        aoi_transformed = aoi_geom

    # Rasterize the geometry to create a mask
    mask = features.rasterize(
        [(aoi_transformed, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _read_transform(reference_path: Path) -> tuple[Affine, int, int, float, str]:
    with rasterio.open(reference_path) as ref:
        res_x, res_y = ref.res
        pixel_size = float((abs(res_x) + abs(res_y)) / 2)
        return ref.transform, ref.width, ref.height, pixel_size, ref.crs.to_string() if ref.crs else None


def _mosaic_dem(paths: Iterable[Path], target_transform: Affine, target_width: int, target_height: int, target_crs: str) -> np.ndarray:
    datasets = [rasterio.open(path) for path in paths]
    if not datasets:
        raise ValueError("No DEM tiles provided")
    src_crs = datasets[0].crs

    dst_crs = None
    if target_crs:
        try:
            dst_crs = CRS.from_user_input(target_crs)
        except ValueError:
            LOGGER.warning("Failed to parse target CRS '%s'; falling back to source CRS", target_crs)
    if dst_crs is None:
        dst_crs = src_crs
    if dst_crs is None:
        raise RuntimeError("Target grid does not define a CRS and DEM sources are missing CRS metadata")
    vrt_datasets: list[WarpedVRT] = []
    try:
        merge_sources = datasets
        if any(ds.crs != dst_crs for ds in datasets):
            vrt_opts = {
                "crs": dst_crs,
                "resampling": Resampling.bilinear,
            }
            if target_transform is not None and target_width and target_height:
                vrt_opts.update(
                    transform=target_transform,
                    width=target_width,
                    height=target_height,
                )
            vrt_datasets = [WarpedVRT(ds, **vrt_opts) for ds in datasets]
            merge_sources = vrt_datasets
        mosaicked, transform = merge(merge_sources, nodata=np.nan)
    finally:
        for vrt in vrt_datasets:
            vrt.close()
        for dataset in datasets:
            dataset.close()
    result = np.full((target_height, target_width), np.nan, dtype="float32")
    warp.reproject(
        source=mosaicked[0],
        destination=result,
        src_transform=transform,
        src_crs=dst_crs,
        dst_transform=target_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return result


def _compute_slope(dem: np.ndarray, pixel_size: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(dem, pixel_size)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    return slope.astype("float32")


def _box_mean(dem: np.ndarray, radius_pixels: int) -> np.ndarray:
    if radius_pixels <= 0:
        return dem.astype("float32", copy=False)

    window_size = radius_pixels * 2 + 1
    valid_mask = np.isfinite(dem).astype("float32")
    values = np.nan_to_num(dem, nan=0.0).astype("float32") * valid_mask
    window_area = float(window_size * window_size)

    sum_avg = ndimage.uniform_filter(
        values,
        size=window_size,
        mode="nearest",
    ) * window_area
    count_avg = ndimage.uniform_filter(
        valid_mask,
        size=window_size,
        mode="nearest",
    ) * window_area

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_avg / count_avg
    mean[count_avg == 0] = np.nan
    return mean.astype("float32")


def _compute_tpi(dem: np.ndarray, radius: float, pixel_size: float) -> np.ndarray:
    radius_pixels = int(max(radius / pixel_size, 1))
    local_mean = _box_mean(dem, radius_pixels)
    tpi = dem - local_mean
    return tpi.astype("float32")


def _compute_depression_depth(dem: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 2)
    filled = ndimage.grey_closing(dem, footprint=structure, mode="nearest")
    depth = filled - dem
    depth[depth < 0] = 0
    return depth.astype("float32")


def write_topography_raster(config: TopographyStackConfig, dem_paths: Iterable[Path], output_path: Path) -> Path:
    """Write topography-derived features to a GeoTIFF.

    Generates 4 bands of relative topographic features useful for wetland detection:
    - Slope: terrain steepness in degrees (flat areas retain water)
    - TPI_small: local terrain position at small scale (depressions = negative)
    - TPI_large: local terrain position at large scale (broader landscape context)
    - DepressionDepth: depth of local sinks (where water accumulates)

    Note: Raw elevation is intentionally excluded because:
    - Wetlands exist at all elevations (coastal to alpine)
    - Absolute elevation creates geographic bias that hurts model generalization
    - Relative features (TPI, depression depth) capture what matters for wetland detection
    """
    dem_paths_list = list(dem_paths)
    transform, width, height, pixel_size, crs = _read_transform(config.target_grid_path)
    LOGGER.info(
        "Mosaicking %s DEM tile(s) into target grid %s",
        len(dem_paths_list),
        config.target_grid_path,
    )
    dem = _mosaic_dem(dem_paths_list, transform, width, height, crs)
    dem_mask = np.isnan(dem)

    LOGGER.info("Computing slope raster")
    slope = _compute_slope(dem, pixel_size)
    slope[dem_mask] = FLOAT_NODATA

    LOGGER.info("Computing TPI (radius=%sm)", config.tpi_small_radius)
    tpi_small = _compute_tpi(dem, config.tpi_small_radius, pixel_size)
    tpi_small[dem_mask] = FLOAT_NODATA

    LOGGER.info("Computing TPI (radius=%sm)", config.tpi_large_radius)
    tpi_large = _compute_tpi(dem, config.tpi_large_radius, pixel_size)
    tpi_large[dem_mask] = FLOAT_NODATA

    LOGGER.info("Computing depression depth")
    depression = _compute_depression_depth(dem)
    depression[dem_mask] = FLOAT_NODATA

    # Apply AOI mask to clip output to the AOI polygon boundary
    if config.aoi is not None:
        LOGGER.info("Clipping topography to AOI boundary")
        aoi_mask = _create_aoi_mask(config.aoi, transform, width, height, crs)
        outside_aoi = ~aoi_mask
        slope[outside_aoi] = FLOAT_NODATA
        tpi_small[outside_aoi] = FLOAT_NODATA
        tpi_large[outside_aoi] = FLOAT_NODATA
        depression[outside_aoi] = FLOAT_NODATA

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,  # Slope, TPI_small, TPI_large, DepressionDepth (no raw elevation)
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": FLOAT_NODATA,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    # Stack only relative topographic features (no raw elevation)
    bands = np.stack([slope, tpi_small, tpi_large, depression])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing derivatives -> %s", output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(bands)
        dst.set_band_description(1, "Slope")
        dst.set_band_description(2, "TPI_small")
        dst.set_band_description(3, "TPI_large")
        dst.set_band_description(4, "DepressionDepth")

    LOGGER.info("Wrote topography raster -> %s", output_path)
    return output_path


__all__ = ["write_topography_raster"]


