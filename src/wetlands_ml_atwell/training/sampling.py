"""
Sampling strategies for creating balanced, bias-free training datasets.

This module handles the logic of selecting *where* to train the model.
It implements the NWI-Filtered Negative Sampling strategy to avoid
false negatives in unsurveyed areas.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

LOGGER = logging.getLogger(__name__)


def create_acquisition_aoi(
    labels_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    buffer_meters: float = 1000.0,
    target_crs: str = "EPSG:5070"
) -> Path:
    """
    Generates a simple bounding box AOI covering all provided labels.
    
    This "Big Box" is used for downloading imagery (Data Acquisition),
    as opposed to the "Swiss Cheese" mask used for sampling.

    Args:
        labels_path: Path to the positive labels (e.g., Atwell wetlands).
        output_path: Where to save the resulting AOI GPKG. If None, defaults
                     to the same directory as labels_path with '_aoi' suffix.
        buffer_meters: Distance to expand the bounding box (context).
        target_crs: Projected CRS to use for buffering (default: NAD83 Conus Albers).

    Returns:
        Path to the saved AOI file.
    """
    labels_path = Path(labels_path)
    if output_path is None:
        output_path = labels_path.parent / f"{labels_path.stem}_aoi.gpkg"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating Acquisition AOI from %s...", labels_path)
    
    # Load and reproject
    gdf = gpd.read_file(labels_path)
    if gdf.crs is None:
        LOGGER.warning("Labels lack CRS. Assuming EPSG:4326.")
        gdf.set_crs("EPSG:4326", inplace=True)
        
    # Reproject to metric CRS for accurate buffering
    gdf_proj = gdf.to_crs(target_crs)
    
    # Calculate total bounds (minx, miny, maxx, maxy)
    bounds = gdf_proj.total_bounds
    
    # Create a single box geometry
    bbox_geom = box(*bounds)
    
    # Buffer the box
    buffered_geom = bbox_geom.buffer(buffer_meters)
    
    # Create output GeoDataFrame
    aoi_gdf = gpd.GeoDataFrame(
        {"geometry": [buffered_geom]},
        crs=target_crs
    )
    
    # Reproject back to EPSG:4326 for standard interoperability with downloaders
    # (Most APIs like Planetary Computer/EE expect WGS84)
    aoi_final = aoi_gdf.to_crs("EPSG:4326")
    
    aoi_final.to_file(output_path, driver="GPKG")
    LOGGER.info("Saved Acquisition AOI to %s", output_path)
    
    return output_path


def _compute_safe_zone(
    aoi_gdf: gpd.GeoDataFrame,
    atwell_gdf: gpd.GeoDataFrame,
    nwi_gdf: gpd.GeoDataFrame,
    buffer_dist: float = 100.0
) -> gpd.GeoDataFrame:
    """
    Internal helper to subtract Unsafe areas (Atwell+Buffer + NWI) from AOI.
    """
    LOGGER.info("Computing 'Safe Negative Zone' (excluding NWI and buffers)...")
    
    # Ensure common projected CRS
    target_crs = aoi_gdf.crs
    
    # 1. Create Exclusion Geometries
    # Buffer Atwell (Positives)
    atwell_clean = atwell_gdf.geometry.buffer(0)
    atwell_buffer = atwell_clean.buffer(buffer_dist)
    
    # NWI (Potential Positives)
    nwi_clean = nwi_gdf.geometry.buffer(0)
    
    # 2. Union Exclusions
    # Use union_all() if available (Geopandas 0.14+), else unary_union
    exclusion_parts = list(atwell_buffer) + list(nwi_clean)
    
    # Quick check for empty NWI
    if not exclusion_parts:
        LOGGER.warning("No exclusions found! Safe zone will be entire AOI.")
        return aoi_gdf

    exclusion_series = gpd.GeoSeries(exclusion_parts, crs=target_crs)
    
    if hasattr(exclusion_series, "union_all"):
        exclusion_union = exclusion_series.union_all()
    else:
        exclusion_union = exclusion_series.unary_union
        
    # 3. Subtract from AOI
    if hasattr(aoi_gdf.geometry, "union_all"):
        aoi_geom = aoi_gdf.geometry.union_all()
    else:
        aoi_geom = aoi_gdf.geometry.unary_union
        
    safe_geom = aoi_geom.difference(exclusion_union)
    
    # 4. Explode to single polygons
    safe_gdf = gpd.GeoDataFrame(
        {"geometry": [safe_geom]},
        crs=target_crs
    ).explode(index_parts=True).reset_index(drop=True)
    
    # Filter tiny slivers (< 900m2, approx 3x3 pixels at 10m)
    safe_gdf = safe_gdf[safe_gdf.area > 900]
    
    return safe_gdf


def generate_training_manifest(
    aoi_path: Union[str, Path],
    atwell_path: Union[str, Path],
    nwi_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    tile_size_m: float = 5120.0,  # 512px * 10m
    buffer_dist: float = 100.0,
    negative_ratio: float = 1.0,
    random_seed: int = 42
) -> Path:
    """
    Generates a Sampling Manifest: a list of specific tiles to extract for training.
    
    Implements Balanced Sampling:
    1. Identifies all tiles touching Atwell polygons (Positives).
    2. Identifies candidate tiles strictly inside the Safe Zone (Negatives).
    3. Samples negatives to match the count of positives (multiplied by negative_ratio).

    Args:
        aoi_path: Path to the Acquisition AOI (Big Box).
        atwell_path: Path to Verified Positive labels.
        nwi_path: Path to Unverified/Potential labels (to be excluded).
        output_path: Where to save the manifest GPKG.
        tile_size_m: Size of training tiles in meters (e.g. 5120 for 512px @ 10m).
        buffer_dist: Safety buffer around positive polygons in meters.
        negative_ratio: Ratio of negatives to positives (1.0 = balanced).
        random_seed: Seed for reproducibility.

    Returns:
        Path to the saved Manifest GPKG.
    """
    if output_path is None:
        output_path = Path(atwell_path).parent / "training_manifest.gpkg"
    else:
        output_path = Path(output_path)
        
    LOGGER.info("Generating Training Manifest...")
    
    # 1. Load Data
    # Using a projected CRS (EPSG:5070) for all spatial logic
    target_crs = "EPSG:5070"
    
    aoi_gdf = gpd.read_file(aoi_path).to_crs(target_crs)
    
    # Load Atwell (clipped to AOI bounds for speed)
    aoi_bounds = tuple(aoi_gdf.total_bounds)
    atwell_gdf = gpd.read_file(atwell_path, bbox=aoi_bounds).to_crs(target_crs)
    
    # Load NWI (clipped to AOI bounds)
    nwi_gdf = gpd.read_file(nwi_path, bbox=aoi_bounds).to_crs(target_crs)
    
    # Strict clip to AOI polygon
    atwell_gdf = gpd.clip(atwell_gdf, aoi_gdf)
    nwi_gdf = gpd.clip(nwi_gdf, aoi_gdf)
    
    # 2. Generate Grid
    LOGGER.info(f"Generating tile grid ({tile_size_m}m)...")
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    cols = np.arange(minx, maxx, tile_size_m)
    rows = np.arange(miny, maxy, tile_size_m)
    
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(box(x, y, x + tile_size_m, y + tile_size_m))
            
    grid_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=target_crs)
    
    # 3. Identify Positives
    # Any tile touching an Atwell polygon
    pos_tiles = gpd.sjoin(grid_gdf, atwell_gdf, how="inner", predicate="intersects")
    pos_tiles = pos_tiles[~pos_tiles.index.duplicated()].copy()
    pos_tiles["class_label"] = 1  # Positive
    
    num_pos = len(pos_tiles)
    LOGGER.info(f"Found {num_pos} Positive tiles.")
    
    if num_pos == 0:
        raise ValueError("No positive tiles found! Check your labels and AOI overlap.")
    
    # 4. Identify Candidate Negatives
    # First, filter grid to those NOT in positive set
    candidate_negatives = grid_gdf.drop(pos_tiles.index)
    
    # Compute Safe Zone
    safe_gdf = _compute_safe_zone(aoi_gdf, atwell_gdf, nwi_gdf, buffer_dist)
    
    # Find tiles whose CENTROIDS are in the Safe Zone
    # This is a fast proxy for "mostly safe"
    # For strict safety, check 'within', but that discards edge tiles.
    candidates_centroids = candidate_negatives.copy()
    candidates_centroids['geometry'] = candidates_centroids.centroid
    
    center_safe = gpd.sjoin(candidates_centroids, safe_gdf, how="inner", predicate="intersects")
    valid_negatives = candidate_negatives.loc[center_safe.index]
    valid_negatives = valid_negatives[~valid_negatives.index.duplicated()].copy()
    valid_negatives["class_label"] = 0  # Negative
    
    num_neg_candidates = len(valid_negatives)
    LOGGER.info(f"Found {num_neg_candidates} valid Negative candidate tiles.")
    
    # 5. Sample Negatives
    target_neg = int(num_pos * negative_ratio)
    
    if num_neg_candidates > target_neg:
        LOGGER.info(f"Sampling {target_neg} negatives from {num_neg_candidates} candidates...")
        selected_negatives = valid_negatives.sample(n=target_neg, random_state=random_seed)
    else:
        LOGGER.info(f"Using all {num_neg_candidates} available negative tiles.")
        selected_negatives = valid_negatives
        
    # 6. Merge and Save
    manifest = pd.concat([pos_tiles, selected_negatives], ignore_index=True)
    manifest = manifest[["geometry", "class_label"]] # Keep clean
    
    # Transform back to 4326 for compatibility with extraction tools
    manifest_out = manifest.to_crs("EPSG:4326")
    
    manifest_out.to_file(output_path, driver="GPKG")
    LOGGER.info(f"Saved Training Manifest ({len(manifest)} tiles) to {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Generate training tile manifest using NWI-Filtered Negative Sampling.")
    parser.add_argument("--aoi-path", required=True, help="Path to Acquisition AOI")
    parser.add_argument("--atwell-path", required=True, help="Path to Atwell (positive) labels")
    parser.add_argument("--nwi-path", required=True, help="Path to NWI (exclusion) labels")
    parser.add_argument("--output-path", help="Path to save manifest (default: beside atwell labels)")
    parser.add_argument("--tile-size-m", type=float, default=5120.0, help="Tile size in meters")
    parser.add_argument("--buffer-dist", type=float, default=100.0, help="Buffer distance for safe zone")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Ratio of negatives to positives")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generate_training_manifest(
        aoi_path=args.aoi_path,
        atwell_path=args.atwell_path,
        nwi_path=args.nwi_path,
        output_path=args.output_path,
        tile_size_m=args.tile_size_m,
        buffer_dist=args.buffer_dist,
        negative_ratio=args.negative_ratio,
        random_seed=args.seed
    )
