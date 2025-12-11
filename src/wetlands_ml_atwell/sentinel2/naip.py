"""NAIP (National Agriculture Imagery Program) raster utilities.

This module provides functions for working with NAIP aerial imagery,
including file discovery, footprint extraction, mosaicking, resampling,
and clipping to areas of interest.

NAIP Processing Workflow
------------------------
1. Discover NAIP rasters from input paths using `collect_naip_sources()`
2. Get footprints for spatial filtering via `_collect_naip_footprints()`
3. Create a unified reference grid with `prepare_naip_reference()`
4. Clip to specific AOI polygons using `_clip_raster_to_polygon()`

NAIP imagery is typically 4-band (RGBIR) at 0.6-1m resolution with
USDA-provided ground control. It serves as the high-resolution reference
grid for aligning Sentinel-2 and topography data.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from shapely.geometry import Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .manifests import NAIP_BAND_LABELS

LOGGER = logging.getLogger(__name__)


def collect_naip_sources(candidates: Sequence[Path]) -> List[Path]:
    """Discover NAIP GeoTIFF files from input paths.
    
    Accepts a mix of file paths and directory paths. Directories are
    searched for .tif and .tiff files. Duplicates are removed based
    on resolved absolute paths.
    
    Args:
        candidates: Sequence of file or directory paths to search.
        
    Returns:
        List of unique resolved paths to NAIP GeoTIFF files.
        
    Example:
        >>> sources = collect_naip_sources([Path("naip_tiles/")])
        >>> print(f"Found {len(sources)} NAIP tiles")
    """
    paths: List[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            for pattern in ("*.tif", "*.tiff"):
                for child in sorted(path.glob(pattern)):
                    if child.is_file():
                        paths.append(child)
        else:
            paths.append(path)
    
    # Deduplicate by resolved path
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


@lru_cache(maxsize=None)
def _load_naip_footprint(path: str) -> Polygon:
    """Load the WGS84 footprint polygon for a NAIP raster.
    
    Results are cached to avoid repeated file reads when filtering
    multiple AOIs against the same tile set.
    
    Args:
        path: String path to the NAIP GeoTIFF file.
        
    Returns:
        Shapely Polygon representing the tile footprint in WGS84.
        
    Raises:
        ValueError: If the raster is missing CRS information.
    """
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"NAIP raster '{path}' is missing CRS information.")
        bounds = dataset.bounds
        footprint = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        geojson = mapping(footprint)
        footprint_wgs84 = transform_geom(dataset.crs, "EPSG:4326", geojson)
    return shape(footprint_wgs84)


def _collect_naip_footprints(sources: Sequence[Path]) -> List[Tuple[Path, Polygon]]:
    """Collect WGS84 footprints for a sequence of NAIP rasters.
    
    Args:
        sources: Sequence of paths to NAIP GeoTIFF files.
        
    Returns:
        List of (path, polygon) tuples for each input file.
    """
    footprints: List[Tuple[Path, Polygon]] = []
    for source in sources:
        polygon = _load_naip_footprint(str(Path(source).resolve()))
        footprints.append((Path(source), polygon))
    return footprints


def _resample_naip_tile(
    source_path: Path,
    target_resolution: float,
    destination_dir: Path,
) -> Path:
    """Resample a NAIP tile to the specified resolution in meters."""

    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise ValueError(f"NAIP raster {source_path} lacks a CRS; unable to resample.")
        if src.crs.is_geographic:
            raise ValueError(
                f"NAIP raster {source_path} uses a geographic CRS ({src.crs}); "
                "provide projected tiles before resampling."
            )
        transform, width, height = calculate_default_transform(
            src.crs,
            src.crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_resolution, target_resolution),
        )
        if width == src.width and height == src.height:
            # Already at desired resolution
            return source_path

        destination_dir.mkdir(parents=True, exist_ok=True)
        resampled_path = destination_dir / f"{source_path.stem}_resampled{source_path.suffix}"
        profile = src.profile.copy()
        profile.update(
            transform=transform,
            width=width,
            height=height,
        )
        with rasterio.open(resampled_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear,
                )
    LOGGER.info("Resampled %s to %s m pixels -> %s", source_path, target_resolution, resampled_path)
    return resampled_path


def _reproject_naip_to_crs(
    source_path: Path,
    target_crs: str,
    destination_dir: Path,
    target_resolution: Optional[float] = None,
) -> Path:
    """Reproject a NAIP raster to the target CRS.

    Args:
        source_path: Path to the input NAIP GeoTIFF.
        target_crs: Target CRS string (e.g., "EPSG:5070").
        destination_dir: Directory for the reprojected output.
        target_resolution: Optional target resolution. If None, uses source resolution.

    Returns:
        Path to the reprojected raster, or source_path if already in target CRS.
    """
    from rasterio.crs import CRS as RioCRS

    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise ValueError(f"NAIP raster {source_path} lacks CRS; cannot reproject.")

        # Parse target CRS
        dst_crs = RioCRS.from_user_input(target_crs)

        # Check if already in target CRS
        if src.crs == dst_crs:
            LOGGER.info("NAIP already in target CRS %s, skipping reprojection", target_crs)
            return source_path

        # Determine output resolution
        if target_resolution is not None:
            res = (target_resolution, target_resolution)
        else:
            res = src.res

        # Calculate transform for new CRS
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=res,
        )

        destination_dir.mkdir(parents=True, exist_ok=True)
        reprojected_path = destination_dir / f"{source_path.stem}_reprojected{source_path.suffix}"

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
        )

        with rasterio.open(reprojected_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
            # Copy band descriptions if present
            for idx, desc in enumerate(src.descriptions, start=1):
                if desc:
                    dst.set_band_description(idx, desc)

    LOGGER.info("Reprojected NAIP to %s -> %s", target_crs, reprojected_path)
    return reprojected_path


def _transform_to_epsg4326(geometry: BaseGeometry, source_crs) -> BaseGeometry:
    if source_crs is None:
        return geometry
    try:
        transformed = transform_geom(source_crs, "EPSG:4326", geometry.__geo_interface__)
    except Exception:  # pragma: no cover - fallback if transform fails
        return geometry
    return shape(transformed)


def compute_naip_union_extent(raster_paths: Iterable[Path]) -> Optional[BaseGeometry]:
    """Return the union of raster bounds as a shapely geometry."""

    geoms: List[BaseGeometry] = []
    for path in raster_paths:
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            tile_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            tile_geom = _transform_to_epsg4326(tile_geom, dataset.crs)
            if not tile_geom.is_valid:
                tile_geom = tile_geom.buffer(0)
            geoms.append(tile_geom)

    if not geoms:
        return None

    union_geom = unary_union(geoms)
    if hasattr(union_geom, "bounds"):
        LOGGER.info(
            "Computed NAIP union extent -> minx=%.6f, miny=%.6f, maxx=%.6f, maxy=%.6f",
            union_geom.bounds[0],
            union_geom.bounds[1],
            union_geom.bounds[2],
            union_geom.bounds[3],
        )
    return union_geom


def prepare_naip_reference(
    sources: Sequence[Path],
    working_dir: Path,
    target_resolution: Optional[float] = None,
    target_crs: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any], Sequence[str]]:
    """Create a unified NAIP reference grid from one or more tiles.

    If multiple tiles are provided, they are mosaicked together.
    Optionally resamples to a target resolution and/or reprojects to target CRS.

    Args:
        sources: Sequence of paths to NAIP GeoTIFF files.
        working_dir: Directory for output files (mosaic, resampled, reprojected).
        target_resolution: Optional target pixel size in meters.
            If None, original resolution is preserved.
        target_crs: Optional target CRS (e.g., "EPSG:5070"). If provided,
            the output will be reprojected to this CRS.

    Returns:
        Tuple of:
        - reference_path: Path to the final reference raster
        - profile: Dict with crs, transform, width, height
        - labels: Band labels (e.g., ["Red", "Green", "Blue", "NIR"])

    Raises:
        ValueError: If sources is empty, or if tiles have mismatched
            CRS, resolution, or band count.

    Note:
        All input tiles must share the same CRS, pixel size, and band count.
        The mosaic inherits compression and tiling settings from the first tile.
    """
    if not sources:
        raise ValueError("No NAIP rasters were provided.")

    # If target_crs specified and multiple tiles with different CRS, reproject first
    working_sources = list(sources)
    if target_crs and len(sources) > 1:
        # Check if tiles have different CRS - if so, reproject each to target_crs first
        crs_set = set()
        for src_path in sources:
            with rasterio.open(src_path) as dataset:
                crs_set.add(dataset.crs.to_string() if dataset.crs else None)

        if len(crs_set) > 1:
            LOGGER.info(
                "NAIP tiles span %d different CRS: %s. Reprojecting to %s before mosaicking.",
                len(crs_set), crs_set, target_crs
            )
            reprojected_dir = working_dir / "naip_reprojected_tiles"
            reprojected_dir.mkdir(parents=True, exist_ok=True)
            working_sources = []
            for i, src_path in enumerate(sources):
                reprojected_path = _reproject_naip_to_crs(
                    Path(src_path),
                    target_crs,
                    reprojected_dir,
                    target_resolution,
                )
                # Rename to avoid conflicts (multiple tiles might have same stem)
                if reprojected_path.stem.endswith("_reprojected"):
                    final_path = reprojected_dir / f"tile_{i:03d}_reprojected.tif"
                    if reprojected_path != final_path:
                        reprojected_path.rename(final_path)
                        reprojected_path = final_path
                working_sources.append(reprojected_path)
            # After reprojection, skip the final reproject step since already done
            target_crs = None

    # Validate consistency across tiles (after any pre-reprojection)
    crs = None
    resolution = None
    band_count = None
    for src_path in working_sources:
        with rasterio.open(src_path) as dataset:
            if crs is None:
                crs = dataset.crs
                resolution = dataset.res
                band_count = dataset.count
            else:
                if dataset.crs != crs:
                    raise ValueError(
                        f"All NAIP rasters must share the same CRS. "
                        f"Found {dataset.crs} and {crs}. "
                        f"Specify --target-crs to auto-reproject."
                    )
                if not np.allclose(dataset.res, resolution, atol=1e-6):
                    raise ValueError("All NAIP rasters must share the same pixel size.")
                if dataset.count != band_count:
                    raise ValueError("All NAIP rasters must share the same band count.")

    # Single tile: optionally resample and/or reproject, return directly
    if len(working_sources) == 1:
        source_path = Path(working_sources[0])
        reference_path = source_path
        if target_resolution is not None:
            reference_path = _resample_naip_tile(
                source_path,
                target_resolution,
                working_dir / "naip_resampled",
            )
        # Reproject to target CRS if specified
        if target_crs:
            reference_path = _reproject_naip_to_crs(
                reference_path,
                target_crs,
                working_dir / "naip_reprojected",
                target_resolution,
            )
        with rasterio.open(reference_path) as reference:
            profile = {
                "crs": reference.crs.to_string() if reference.crs else None,
                "transform": reference.transform,
                "width": reference.width,
                "height": reference.height,
            }
            labels = NAIP_BAND_LABELS[: reference.count]
        return reference_path, profile, labels

    # Multiple tiles: mosaic then optionally resample
    mosaic_path = working_dir / "naip_mosaic.tif"
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [rasterio.open(path) for path in working_sources]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile
        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
            }
        )
        profile.setdefault("tiled", True)
        profile.setdefault("compress", "deflate")
        profile.setdefault("BIGTIFF", "IF_SAFER")
        with rasterio.open(mosaic_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()

    reference_path = mosaic_path
    if target_resolution is not None:
        reference_path = _resample_naip_tile(
            mosaic_path,
            target_resolution,
            working_dir / "naip_resampled",
        )

    # Reproject to target CRS if specified and different from source
    if target_crs:
        reference_path = _reproject_naip_to_crs(
            reference_path,
            target_crs,
            working_dir / "naip_reprojected",
            target_resolution,
        )

    with rasterio.open(reference_path) as reference:
        profile = {
            "crs": reference.crs.to_string() if reference.crs else None,
            "transform": reference.transform,
            "width": reference.width,
            "height": reference.height,
        }
        labels = NAIP_BAND_LABELS[: reference.count]
    LOGGER.info("Generated NAIP reference at %s", reference_path)
    return reference_path, profile, labels


def _clip_raster_to_polygon(
    raster_path: Path,
    polygon: BaseGeometry,
    destination: Path,
) -> Dict[str, Any]:
    """Clip a raster to a polygon boundary and save to disk.
    
    The polygon is expected to be in WGS84 (EPSG:4326) and is
    automatically reprojected to match the raster CRS.
    
    Args:
        raster_path: Path to input GeoTIFF file.
        polygon: Shapely geometry defining the clip boundary (WGS84).
        destination: Output path for the clipped raster.
        
    Returns:
        Profile dict with crs, transform, width, height of the output.
        
    Raises:
        ValueError: If the input raster lacks CRS metadata.
        
    Note:
        Band descriptions from the source are preserved in the output.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {raster_path} lacks CRS metadata for clipping.")
        geom_src = transform_geom("EPSG:4326", src.crs, mapping(polygon))
        data, transform = mask(src, [geom_src], crop=True)
        meta = src.meta.copy()
        meta.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform,
            }
        )
        meta.setdefault("compress", "deflate")
        meta.setdefault("tiled", True)
        meta.setdefault("BIGTIFF", "IF_SAFER")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination, "w", **meta) as dst:
            dst.write(data)
            for idx, desc in enumerate(src.descriptions, start=1):
                if desc:
                    dst.set_band_description(idx, desc)

        profile = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": transform,
            "width": data.shape[2],
            "height": data.shape[1],
        }
    return profile


__all__ = [
    "collect_naip_sources",
    "compute_naip_union_extent",
    "prepare_naip_reference",
    "_load_naip_footprint",
    "_collect_naip_footprints",
    "_clip_raster_to_polygon",
]
