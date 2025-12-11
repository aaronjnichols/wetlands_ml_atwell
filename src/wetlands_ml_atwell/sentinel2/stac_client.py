"""STAC client utilities for Sentinel-2 data access.

This module provides functions for querying Sentinel-2 imagery from
STAC (SpatioTemporal Asset Catalog) APIs and creating xarray data
stacks from the results.

STAC Query Workflow
-------------------
1. Define search parameters (geometry, date range, cloud cover)
2. Query STAC catalog using `fetch_items()`
3. Stack selected bands into xarray DataArray using `stack_bands()`

The module handles:
- Seasonal date range calculation for multi-year queries
- Band name mapping between Sentinel-2 standard names and STAC asset IDs
- Scaling reflectance values from raw DN to [0, 1] range
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import stackstac
import xarray as xr
from pystac import Item
from pystac.extensions.projection import ProjectionExtension
from pystac_client import Client
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

LOGGER = logging.getLogger(__name__)

# =============================================================================
# STAC and Sentinel-2 Constants
# =============================================================================

# STAC collection identifier for Sentinel-2 Level-2A data
SENTINEL_COLLECTION = "sentinel-2-l2a"

# Sentinel-2 bands used for compositing (10m and 20m resolution bands)
# These are resampled to 10m during stacking
SENTINEL_BANDS = ["B03", "B04", "B05", "B06", "B08", "B11", "B12"]

# Mapping from Sentinel-2 band names to STAC asset IDs
# Different STAC providers may use different asset naming conventions
BAND_TO_ASSET = {
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}

# Scale factor to convert Sentinel-2 DN values to reflectance [0, 1]
# Sentinel-2 L2A products store reflectance * 10000
SENTINEL_SCALE_FACTOR = 1 / 10000

# Default seasons and their date windows (month_start, day_start, month_end, day_end)
DEFAULT_SEASONS: Tuple[str, ...] = ("SPR", "SUM", "FAL")
SEASON_WINDOWS: Dict[str, Tuple[int, int, int, int]] = {
    "SPR": (3, 1, 5, 31),   # Spring: March 1 - May 31
    "SUM": (6, 1, 8, 31),   # Summer: June 1 - August 31
    "FAL": (9, 1, 11, 30),  # Fall: September 1 - November 30
}


# =============================================================================
# Chunk Size Utilities
# =============================================================================


def compute_chunk_size(
    bounds: Tuple[float, float, float, float],
    resolution: float = 10.0,
    min_chunk: int = 256,
    max_chunk: int = 1024,
) -> int:
    """Compute optimal dask chunk size based on AOI dimensions.

    Uses a heuristic that divides the AOI into approximately 4-16 chunks
    per dimension to enable effective dask parallelization on multi-core
    systems.

    Args:
        bounds: Bounding box in lat/lon (minx, miny, maxx, maxy).
        resolution: Target resolution in meters (default 10m for Sentinel-2).
        min_chunk: Minimum chunk size in pixels (default 256).
        max_chunk: Maximum chunk size in pixels (default 1024).

    Returns:
        Chunk size in pixels, rounded to nearest power of 2 and clamped
        between min_chunk and max_chunk.

    Example:
        >>> compute_chunk_size((-85.5, 41.5, -85.3, 41.7), resolution=10)
        512  # For ~20km AOI at 10m resolution
    """
    import math

    minx, miny, maxx, maxy = bounds

    # Approximate conversion from degrees to meters
    # 1 degree latitude ~= 111.32 km
    # 1 degree longitude varies with latitude
    lat_center = (miny + maxy) / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_center))

    width_m = abs(maxx - minx) * meters_per_deg_lon
    height_m = abs(maxy - miny) * meters_per_deg_lat

    # Convert to pixels at target resolution
    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)

    # Use the larger dimension for chunk calculation
    max_dim = max(width_px, height_px)

    if max_dim <= 0:
        return min_chunk

    # Target approximately 4 chunks per dimension for good parallelization
    # This yields 16 total chunks, suitable for multi-core processing
    target_chunk = max_dim // 4

    # Round to nearest power of 2 for memory alignment efficiency
    if target_chunk > 0:
        power = round(math.log2(target_chunk))
        target_chunk = 2 ** power
    else:
        target_chunk = min_chunk

    # Clamp to valid range
    return max(min_chunk, min(max_chunk, target_chunk))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class SeasonConfig:
    """Configuration for a seasonal date range.
    
    Attributes:
        name: Season identifier (e.g., "SPR", "SUM", "FAL")
        start: Start date of the season
        end: End date of the season
    """
    name: str
    start: date
    end: date


# =============================================================================
# Functions
# =============================================================================

def season_date_range(year: int, season: str) -> SeasonConfig:
    """Compute the date range for a season in a given year.
    
    Args:
        year: The year for the season.
        season: Season identifier ("SPR", "SUM", or "FAL").
        
    Returns:
        SeasonConfig with start and end dates.
        
    Raises:
        ValueError: If season is not a supported season identifier.
        
    Example:
        >>> cfg = season_date_range(2023, "SUM")
        >>> print(cfg.start, cfg.end)
        2023-06-01 2023-08-31
    """
    if season not in SEASON_WINDOWS:
        raise ValueError(f"Unsupported season '{season}'. Supported: {sorted(SEASON_WINDOWS)}")
    sm, sd, em, ed = SEASON_WINDOWS[season]
    return SeasonConfig(season, date(year, sm, sd), date(year, em, ed))


def fetch_items(
    client: Client,
    geometry: BaseGeometry,
    season: str,
    years: Sequence[int],
    cloud_cover: float,
) -> List[Item]:
    """Query STAC catalog for Sentinel-2 scenes matching criteria.

    Searches the STAC catalog for Sentinel-2 L2A scenes that intersect
    the given geometry within the specified seasonal date ranges and
    cloud cover threshold.

    Args:
        client: PySTAC client connected to a STAC API.
        geometry: Area of interest as a Shapely geometry.
        season: Season identifier ("SPR", "SUM", or "FAL").
        years: Sequence of years to search across.
        cloud_cover: Maximum cloud cover percentage (0-100).

    Returns:
        List of STAC Items matching the search criteria.
        Duplicates (same item ID) are automatically deduplicated.

    Example:
        >>> client = Client.open("https://earth-search.aws.element84.com/v1")
        >>> items = fetch_items(client, aoi_polygon, "SUM", [2022, 2023], 20.0)
        >>> print(f"Found {len(items)} scenes")
    """
    items: Dict[str, Item] = {}
    geojson = mapping(geometry)
    for year in years:
        cfg = season_date_range(year, season)
        search = client.search(
            collections=[SENTINEL_COLLECTION],
            intersects=geojson,
            datetime=f"{cfg.start.isoformat()}/{cfg.end.isoformat()}",
            query={"eo:cloud_cover": {"lt": cloud_cover}},
        )
        for item in search.get_items():
            items[item.id] = item
    return list(items.values())


def filter_best_scenes(
    items: List[Item],
    max_scenes: Optional[int] = None,
) -> List[Item]:
    """Filter items to keep only the N scenes with lowest cloud cover.

    This function is TILE-AWARE: it distributes the selection across all
    Sentinel-2 tiles to ensure full spatial coverage. For AOIs that span
    multiple tiles, simply selecting the N clearest scenes globally could
    result in all scenes coming from one tile, leaving other areas with
    no data.

    Args:
        items: List of STAC Items from fetch_items().
        max_scenes: Maximum number of scenes to keep. If None or greater
            than len(items), returns all items unchanged.

    Returns:
        List of Items sorted by cloud cover (ascending), limited to max_scenes.
        Scenes are selected to ensure coverage from all tiles.

    Example:
        >>> items = fetch_items(client, polygon, "SUM", [2022, 2023], 60.0)
        >>> best = filter_best_scenes(items, max_scenes=20)
        >>> print(f"Using {len(best)} clearest scenes out of {len(items)}")
    """
    if not items:
        return items

    if max_scenes is None or max_scenes >= len(items):
        return items

    # Group scenes by tile ID to ensure spatial coverage
    tiles: Dict[str, List[Item]] = {}
    for item in items:
        # Extract tile ID from scene ID (e.g., "S2A_16TFN_20231113_1_L2A" -> "16TFN")
        parts = item.id.split("_")
        tile_id = parts[1] if len(parts) >= 2 else "unknown"
        if tile_id not in tiles:
            tiles[tile_id] = []
        tiles[tile_id].append(item)

    # Sort scenes within each tile by cloud cover
    for tile_id in tiles:
        tiles[tile_id].sort(key=lambda item: item.properties.get("eo:cloud_cover", 100.0))

    # Distribute scenes across tiles proportionally, minimum 1 per tile
    n_tiles = len(tiles)
    if n_tiles == 1:
        # Single tile - just take the clearest scenes
        selected = tiles[list(tiles.keys())[0]][:max_scenes]
    else:
        # Multiple tiles - ensure each tile gets at least min_per_tile scenes
        min_per_tile = max(1, max_scenes // (n_tiles * 2))  # At least 1, up to half fair share
        remaining = max_scenes

        selected = []
        # First pass: give each tile minimum scenes
        for tile_id, tile_items in tiles.items():
            n_take = min(min_per_tile, len(tile_items), remaining)
            selected.extend(tile_items[:n_take])
            remaining -= n_take

        # Second pass: fill remaining slots with clearest scenes globally
        if remaining > 0:
            # Get scenes not yet selected, sorted by cloud cover
            selected_ids = {item.id for item in selected}
            remaining_items = [item for item in items if item.id not in selected_ids]
            remaining_items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100.0))
            selected.extend(remaining_items[:remaining])

    # Sort final selection by cloud cover for consistent output
    selected.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100.0))

    # Log tile distribution
    tile_counts = {}
    for item in selected:
        parts = item.id.split("_")
        tile_id = parts[1] if len(parts) >= 2 else "unknown"
        tile_counts[tile_id] = tile_counts.get(tile_id, 0) + 1

    LOGGER.info(
        "Filtered %d scenes to %d clearest across %d tiles: %s (cloud cover: %.1f%% - %.1f%%)",
        len(items),
        len(selected),
        len(tile_counts),
        ", ".join(f"{k}={v}" for k, v in sorted(tile_counts.items())),
        selected[0].properties.get("eo:cloud_cover", 0) if selected else 0,
        selected[-1].properties.get("eo:cloud_cover", 0) if selected else 0,
    )

    return selected


def stack_bands(
    items: Sequence[Item],
    bounds: Tuple[float, float, float, float],
    chunks: Optional[int] = 2048,
    target_crs: Optional[str] = None,
) -> xr.DataArray:
    """Create an xarray DataArray stack from Sentinel-2 STAC items.

    Stacks the specified bands from all input items into a 4D DataArray
    with dimensions (time, band, y, x). All bands are resampled to 10m
    resolution.

    The output reflectance values are scaled to [0, 1] range by applying
    the SENTINEL_SCALE_FACTOR (1/10000).

    Args:
        items: Sequence of Sentinel-2 STAC items.
        bounds: Bounding box in lat/lon (minx, miny, maxx, maxy).
        chunks: Dask chunk size for x and y dimensions. Set to None
            for in-memory processing.
        target_crs: Target CRS for output (e.g., "EPSG:5070"). If None,
            uses the first item's native CRS.

    Returns:
        4D xarray DataArray with dimensions (time, band, y, x).
        Band coordinates are set to SENTINEL_BANDS names.

    Raises:
        ValueError: If items is empty, missing proj:epsg metadata,
            or missing required band assets.

    Note:
        This function returns a lazy dask-backed array. Call .compute()
        to materialize the data.
    """
    if not items:
        raise ValueError("No Sentinel-2 items available for stacking.")

    # Determine output EPSG: use target_crs if provided, else first item's CRS
    if target_crs:
        # Parse EPSG from string like "EPSG:5070" or just "5070"
        if ":" in target_crs:
            output_epsg = int(target_crs.split(":")[1])
        else:
            output_epsg = int(target_crs)
    else:
        # Fall back to first item's EPSG (legacy behavior)
        try:
            proj_ext = ProjectionExtension.ext(items[0])
            output_epsg = proj_ext.epsg
        except Exception:
            # Fallback to legacy property access
            output_epsg = items[0].properties.get("proj:epsg")

        if output_epsg is None:
            raise ValueError("Sentinel-2 item missing projection metadata (proj:epsg).")

    asset_ids = []
    for band in SENTINEL_BANDS:
        asset_id = BAND_TO_ASSET[band]
        if asset_id not in items[0].assets:
            raise ValueError(f"Missing Sentinel-2 asset '{asset_id}' on first item.")
        asset_ids.append(asset_id)

    data = stackstac.stack(
        items,
        assets=asset_ids,
        resolution=10,
        epsg=int(output_epsg),
        bounds_latlon=bounds,
        chunksize={"x": chunks, "y": chunks} if chunks else None,
        dtype="float32",
        fill_value=np.float32(0),
        rescale=False,
        properties=False,
    )
    data = data.reset_coords(drop=True)
    data.rio.write_crs(int(output_epsg), inplace=True)
    data = data.assign_coords({"band": SENTINEL_BANDS})
    return data * SENTINEL_SCALE_FACTOR


__all__ = [
    # Constants
    "SENTINEL_COLLECTION",
    "SENTINEL_BANDS",
    "BAND_TO_ASSET",
    "SENTINEL_SCALE_FACTOR",
    "DEFAULT_SEASONS",
    "SEASON_WINDOWS",
    # Data classes
    "SeasonConfig",
    # Functions
    "season_date_range",
    "fetch_items",
    "filter_best_scenes",
    "stack_bands",
    "compute_chunk_size",
]

