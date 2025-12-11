"""Configuration objects for topographic stack preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class TopographyStackConfig:
    """Parameters controlling LiDAR-derived topographic processing."""

    aoi: BaseGeometry
    target_grid_path: Path
    output_dir: Path
    buffer_meters: float = 200.0
    tpi_small_radius: float = 30.0
    """Radius of the small-scale TPI window in meters (converted to pixels internally)."""
    tpi_large_radius: float = 150.0
    """Radius of the large-scale TPI window in meters (converted to pixels internally)."""
    cache_dir: Optional[Path] = None
    max_products: int = 500
    description: str = "USGS 3DEP derived topography"
    dem_dir: Optional[Path] = None
    """Optional directory containing pre-downloaded DEM tiles."""
    dem_paths: Optional[Sequence[Path]] = None
    """Optional explicit list of DEM GeoTIFF paths to use."""
    dem_resolution: Optional[str] = None
    """DEM resolution preference ('1m', '10m', '30m'). Default None uses 1m."""


__all__ = ["TopographyStackConfig"]


