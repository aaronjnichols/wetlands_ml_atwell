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


def stack_bands(
    items: Sequence[Item],
    bounds: Tuple[float, float, float, float],
    chunks: Optional[int] = 2048,
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

    # Get EPSG via projection extension (handles both legacy and new STAC formats)
    try:
        proj_ext = ProjectionExtension.ext(items[0])
        epsg = proj_ext.epsg
    except Exception:
        # Fallback to legacy property access
        epsg = items[0].properties.get("proj:epsg")

    if epsg is None:
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
        epsg=int(epsg),
        bounds_latlon=bounds,
        chunksize={"x": chunks, "y": chunks} if chunks else None,
        dtype="float64",
        rescale=False,
        properties=False,
    )
    data = data.reset_coords(drop=True)
    data.rio.write_crs(int(epsg), inplace=True)
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
    "stack_bands",
]

