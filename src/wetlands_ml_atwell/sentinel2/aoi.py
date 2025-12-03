"""AOI (Area of Interest) parsing and geometry utilities.

This module provides functions to parse AOI specifications from various formats
and extract individual polygons for processing.

Supported AOI Formats
---------------------
1. **Bounding box string**: Comma-separated "minx,miny,maxx,maxy"
   Example: "-85.5,41.5,-85.0,42.0"

2. **WKT (Well-Known Text)**: Standard geometry representation
   Example: "POLYGON ((-85.5 41.5, -85.0 41.5, -85.0 42.0, -85.5 42.0, -85.5 41.5))"

3. **GeoJSON**: JSON object with geometry
   Example: '{"type": "Polygon", "coordinates": [[[...]]]]}'

4. **File path**: Path to GeoPackage (.gpkg) or Shapefile (.shp)
   - Automatically reprojects to EPSG:4326 if needed
   - Unions all features into single geometry

5. **JSON array**: Bounding box as JSON array [minx, miny, maxx, maxy]
   Example: "[-85.5, 41.5, -85.0, 42.0]"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry

LOGGER = logging.getLogger(__name__)


def parse_aoi(aoi: str) -> BaseGeometry:
    """Parse AOI from various input formats.
    
    Args:
        aoi: AOI specification as string. Can be:
            - File path to GeoPackage/Shapefile
            - GeoJSON string or file
            - WKT string
            - Bounding box as comma-separated string
            - Bounding box as JSON array
            
    Returns:
        Parsed geometry in EPSG:4326 coordinates.
        
    Raises:
        ValueError: If AOI file is empty or geometry is invalid.
        FileNotFoundError: If AOI file path doesn't exist.
    """
    candidate = aoi.strip()
    path = Path(candidate)
    geom: Optional[BaseGeometry] = None

    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".gpkg", ".shp"}:
            gdf = gpd.read_file(path)
            if gdf.empty:
                raise ValueError(f"AOI file '{path}' contains no features.")
            if gdf.crs is not None:
                gdf = gdf.to_crs(4326)
            else:
                LOGGER.warning("AOI file %s has no CRS; assuming EPSG:4326 coordinates.", path)
            geom_series = gdf.geometry.dropna()
            if geom_series.empty:
                raise ValueError(f"AOI file '{path}' contains no valid geometries.")
            geom = geom_series.union_all()
        else:
            candidate = path.read_text(encoding="utf-8").strip()

    if geom is None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            geom = shape(payload.get("geometry", payload))
        elif isinstance(payload, list) and len(payload) == 4:
            geom = box(*payload)
        else:
            if "," in candidate and candidate.count(",") == 3:
                parts = [float(x) for x in candidate.split(",")]
                geom = box(*parts)
            else:
                geom = wkt.loads(candidate)

    if geom.is_empty:
        raise ValueError("AOI geometry is empty.")
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def buffer_in_meters(geom: BaseGeometry, buffer_meters: float) -> BaseGeometry:
    """Buffer a geometry by a distance in meters.
    
    The geometry is temporarily projected to an appropriate UTM zone
    for accurate distance-based buffering, then reprojected back to WGS84.
    
    Args:
        geom: Input geometry in EPSG:4326.
        buffer_meters: Buffer distance in meters. If <= 0, returns input unchanged.
        
    Returns:
        Buffered geometry in EPSG:4326.
    """
    if buffer_meters <= 0:
        return geom
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    projected = series.to_crs(series.estimate_utm_crs())
    buffered = projected.buffer(buffer_meters)
    return buffered.to_crs(4326).iloc[0]


def extract_aoi_polygons(aoi: BaseGeometry, buffer_meters: float = 0.0) -> List[Polygon]:
    """Extract individual polygons from an AOI geometry.
    
    Handles Polygon, MultiPolygon, and other geometry types by converting
    to polygons. Optionally buffers each polygon by a distance in meters.
    
    Args:
        aoi: Input geometry (Polygon, MultiPolygon, or other).
        buffer_meters: Optional buffer distance in meters to apply to each polygon.
        
    Returns:
        List of individual Polygon geometries.
        
    Raises:
        ValueError: If no valid polygons can be extracted.
    """
    if isinstance(aoi, Polygon):
        parts: List[BaseGeometry] = [aoi]
    elif isinstance(aoi, MultiPolygon):
        parts = list(aoi.geoms)
    else:
        parts = [shape(mapping(aoi))]

    result: List[Polygon] = []
    for part in parts:
        candidate = buffer_in_meters(part, buffer_meters)
        if candidate.is_empty:
            continue
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if isinstance(candidate, Polygon):
            result.append(candidate)
        elif isinstance(candidate, MultiPolygon):
            for poly in candidate.geoms:
                if not poly.is_empty:
                    result.append(poly)
        else:
            candidate_poly = candidate.convex_hull
            if isinstance(candidate_poly, Polygon) and not candidate_poly.is_empty:
                result.append(candidate_poly)

    if not result:
        raise ValueError("No valid AOI polygons extracted.")
    return result


__all__ = [
    "parse_aoi",
    "buffer_in_meters",
    "extract_aoi_polygons",
]

