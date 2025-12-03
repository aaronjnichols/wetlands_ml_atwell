"""CLI for preparing LiDAR-derived topographic bands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from shapely.geometry import shape
import geopandas as gpd

from .config import TopographyStackConfig
from .pipeline import prepare_topography_stack


LOGGER = logging.getLogger(__name__)


def _load_aoi(path: Path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI file {path} contains no geometries")
    geom = gdf.unary_union
    return geom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LiDAR-derived topography stack.")
    parser.add_argument("--aoi", required=True, help="Path to AOI vector file (GeoJSON/GPKG).")
    parser.add_argument("--target-grid", required=True, help="Reference raster for grid alignment (e.g., NAIP).")
    parser.add_argument("--output-dir", required=True, help="Directory to store processed rasters.")
    parser.add_argument("--buffer", type=float, default=200.0, help="Buffer distance in meters around AOI.")
    parser.add_argument("--cache-dir", help="Optional directory to cache raw DEM downloads.")
    parser.add_argument("--tpi-small", type=float, default=30.0, help="Radius in meters for small-scale TPI.")
    parser.add_argument("--tpi-large", type=float, default=150.0, help="Radius in meters for large-scale TPI.")
    return parser


def main(args=None) -> Path:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)

    aoi_path = Path(parsed.aoi)
    target_grid_path = Path(parsed.target_grid)
    output_dir = Path(parsed.output_dir)
    cache_dir = Path(parsed.cache_dir) if parsed.cache_dir else None

    aoi_geom = _load_aoi(aoi_path)

    config = TopographyStackConfig(
        aoi=aoi_geom,
        target_grid_path=target_grid_path,
        output_dir=output_dir,
        buffer_meters=parsed.buffer,
        tpi_small_radius=parsed.tpi_small,
        tpi_large_radius=parsed.tpi_large,
        cache_dir=cache_dir,
    )

    raster_path = prepare_topography_stack(config)
    LOGGER.info("Topography stack ready -> %s", raster_path)
    return raster_path


if __name__ == "__main__":
    main()


