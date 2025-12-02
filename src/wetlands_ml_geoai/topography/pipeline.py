"""Orchestrates DEM download and derivative computation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Set

import geopandas as gpd
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from .config import TopographyStackConfig
from .download import download_dem_products, fetch_dem_inventory
from .processing import write_topography_raster


LOGGER = logging.getLogger(__name__)


def _buffer_geometry(geometry: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]


def _resolve_local_dem_paths(config: TopographyStackConfig) -> List[Path]:
    candidates: List[Path] = []

    if config.dem_paths:
        for raw_path in config.dem_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Configured DEM path not found: {path}")
            if not path.is_file():
                raise ValueError(f"Configured DEM path is not a file: {path}")
            candidates.append(path)

    if config.dem_dir:
        directory = Path(config.dem_dir).expanduser().resolve()
        LOGGER.info(f"Searching for DEMs in: {directory}")
        if not directory.exists():
            raise FileNotFoundError(f"Configured DEM directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Configured DEM directory is not a directory: {directory}")
        
        found_any = False
        for path in sorted(directory.glob("*")):
            LOGGER.debug(f"Checking file: {path}")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
                candidates.append(path.resolve())
                found_any = True
        
        if not found_any:
            LOGGER.warning(f"No .tif/.tiff files found in {directory}")
            # Debug listing
            LOGGER.warning(f"Directory contents: {[p.name for p in directory.glob('*')]}")

    unique: List[Path] = []
    seen: Set[Path] = set()
    for path in candidates:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)

    return unique


def prepare_topography_stack(config: TopographyStackConfig) -> Path:
    """Compute derivative raster for ``config`` AOI using cached or downloaded DEMs."""

    local_requested = config.dem_paths is not None or config.dem_dir is not None
    dem_paths: Sequence[Path] = _resolve_local_dem_paths(config)
    if dem_paths:
        LOGGER.info(
            "Using %s pre-existing DEM tile(s); skipping 3DEP download.",
            len(dem_paths),
        )
    else:
        if local_requested:
            raise FileNotFoundError(
                "No DEM GeoTIFFs discovered for configured dem_paths/dem_dir; cannot proceed."
            )

        buffered_geom = _buffer_geometry(config.aoi, config.buffer_meters)
        geojson = mapping(buffered_geom)
        bbox = buffered_geom.bounds

        LOGGER.info("Fetching DEM inventory for buffered AOI (buffer=%sm)", config.buffer_meters)
        products = fetch_dem_inventory(
            geojson,
            bbox=bbox,
            max_results=config.max_products,
        )
        if not products:
            raise RuntimeError("No 3DEP DEM products found for buffered AOI; adjust buffer or verify coverage.")

        LOGGER.info("DEM inventory returned %s product(s)", len(products))

        cache_dir = config.cache_dir or config.output_dir / "raw"
        LOGGER.info("Downloading DEM products to %s", cache_dir)
        dem_paths = download_dem_products(products, cache_dir)
        LOGGER.info("DEM download complete; %s file(s) ready for mosaicking", len(dem_paths))

    topography_path = config.output_dir / "topography_stack.tif"
    LOGGER.info("Computing topographic derivatives -> %s", topography_path)
    return write_topography_raster(config, dem_paths, topography_path)


__all__ = ["prepare_topography_stack"]


