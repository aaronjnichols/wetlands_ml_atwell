
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point, box
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def load_real_data(
    aoi_path: Path,
    atwell_path: Path,
    nwi_path: Path
):
    """Loads real datasets for processing."""
    LOGGER.info("Loading real data...")
    
    # Load AOI
    LOGGER.info(f"Loading AOI from {aoi_path}")
    aoi_gdf = gpd.read_file(aoi_path)
    
    # Determine common CRS (Projected)
    # If AOI is geographic (Lat/Lon), we must project it.
    # Using EPSG:5070 (NAD83 / Conus Albers) is standard for US-wide area calculations,
    # or we can use the local UTM zone. For MI/OK mix, 5070 is safer.
    target_crs = "EPSG:5070"
    
    aoi_proj = aoi_gdf.to_crs(target_crs)
    
    # Performance optimization: Read bbox of AOI first
    aoi_bounds = tuple(aoi_gdf.total_bounds)
    
    # Load Atwell (Positives)
    LOGGER.info(f"Loading Atwell Wetlands from {atwell_path}")
    # Only read valid geometries within AOI bbox
    atwell_gdf = gpd.read_file(atwell_path, bbox=aoi_bounds, engine="pyogrio")
    atwell_proj = atwell_gdf.to_crs(target_crs)
    
    # Load NWI (Potential/Unverified)
    LOGGER.info(f"Loading NWI Wetlands from {nwi_path} (clipped to AOI)...")
    nwi_gdf = gpd.read_file(nwi_path, bbox=aoi_bounds, engine="pyogrio")
    
    if nwi_gdf.empty:
        LOGGER.warning("No NWI wetlands found within the AOI!")
    else:
        LOGGER.info(f"Loaded {len(nwi_gdf)} NWI polygons within AOI.")
        
    nwi_proj = nwi_gdf.to_crs(target_crs)
    
    # Strict Clipping to AOI geometry
    # This removes features that might be inside the bbox but outside the polygon
    LOGGER.info("Clipping geometries to exact AOI polygon...")
    atwell_proj = gpd.clip(atwell_proj, aoi_proj)
    nwi_proj = gpd.clip(nwi_proj, aoi_proj)
    
    return aoi_proj, atwell_proj, nwi_proj

def generate_safe_negatives(
    aoi_gdf: gpd.GeoDataFrame,
    atwell_gdf: gpd.GeoDataFrame,
    nwi_gdf: gpd.GeoDataFrame,
    buffer_dist: float = 100.0
) -> gpd.GeoDataFrame:
    """
    Generates safe negative sampling areas.
    """
    LOGGER.info(f"Computing safe negatives with {buffer_dist}m buffer...")
    
    target_crs = aoi_gdf.crs
    
    # 1. Create Exclusion Zones
    
    # Buffer Atwell polygons
    # Use buffer(0) first to fix any invalid topologies
    atwell_clean = atwell_gdf.geometry.buffer(0)
    atwell_buffer = atwell_clean.buffer(buffer_dist)
    
    # NWI Wetlands
    # Also clean NWI
    nwi_clean = nwi_gdf.geometry.buffer(0)
    
    # Combine exclusions
    # Using union_all() for speed (Geopandas 0.14+) or unary_union
    LOGGER.info("Unioning exclusion zones (this may take a moment)...")
    start_time = time.time()
    
    # Convert to list to avoid index alignment issues
    exclusion_geoms = list(atwell_buffer) + list(nwi_clean)
    exclusion_series = gpd.GeoSeries(exclusion_geoms, crs=target_crs)
    
    # Use union_all if available, else unary_union
    if hasattr(exclusion_series, "union_all"):
        exclusion_union = exclusion_series.union_all()
    else:
        exclusion_union = exclusion_series.unary_union
        
    LOGGER.info(f"Union complete in {time.time() - start_time:.2f}s")
    
    # 2. Subtract Exclusion from AOI
    LOGGER.info("Subtracting exclusions from AOI...")
    if hasattr(aoi_gdf.geometry, "union_all"):
        aoi_geom = aoi_gdf.geometry.union_all()
    else:
        aoi_geom = aoi_gdf.geometry.unary_union
        
    safe_geom = aoi_geom.difference(exclusion_union)
    
    # 3. Create Result GDF
    safe_gdf = gpd.GeoDataFrame({"geometry": [safe_geom]}, crs=target_crs).explode(index_parts=True).reset_index(drop=True)
    
    # Filter tiny areas (e.g. < 900 sqm = 30x30m pixel)
    original_count = len(safe_gdf)
    safe_gdf = safe_gdf[safe_gdf.area > 900]
    LOGGER.info(f"Filtered {original_count - len(safe_gdf)} small slivers (<900m²). Remaining patches: {len(safe_gdf)}")
    
    return safe_gdf, exclusion_union

def simulate_balanced_sampling(
    safe_gdf: gpd.GeoDataFrame,
    atwell_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    tile_size_m: float = 512.0  # Assuming 1m pixels for simplicity, typically 10m -> 5120m
):
    """
    Simulates the tile selection process to ensure balanced classes.
    """
    LOGGER.info("Simulating balanced tile sampling...")
    
    # 1. Generate Grid over AOI
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    cols = np.arange(minx, maxx, tile_size_m)
    rows = np.arange(miny, maxy, tile_size_m)
    
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(box(x, y, x + tile_size_m, y + tile_size_m))
            
    grid_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=aoi_gdf.crs)
    LOGGER.info(f"Generated {len(grid_gdf)} potential tiles.")
    
    # 2. Identify Positive Tiles
    # Any tile touching an Atwell polygon
    # sjoin is fast for this
    positive_tiles = gpd.sjoin(grid_gdf, atwell_gdf, how="inner", predicate="intersects")
    # Drop duplicates (if a tile touches multiple polygons)
    positive_tiles = positive_tiles[~positive_tiles.index.duplicated()]
    
    num_positives = len(positive_tiles)
    LOGGER.info(f"Found {num_positives} Positive Tiles (containing verified wetland).")
    
    # 3. Identify Candidate Negative Tiles
    # Tiles strictly inside the "Safe Zone"
    # We use 'within' or a slight buffer to ensure it's fully safe
    # Let's use centroid check for speed, or strictly 'within'
    
    # Filter grid to those NOT in positive set
    candidate_negatives = grid_gdf.drop(positive_tiles.index)
    
    # Check intersection with safe zone
    # A tile is a valid negative if it is COMPLETELY within the safe zone
    # Optimization: check if centroid is in safe zone first, then geometry
    
    # First, join candidates with safe_gdf
    # We want tiles that are covered by safe_gdf. 
    # 'within' requires the tile to be fully inside a single safe polygon.
    # safe_gdf might be fragmented.
    
    # Robust method: Intersection area
    # Calculate intersection of candidates with safe_union
    # If intersection area ~= tile area, it's safe.
    
    LOGGER.info("Identifying valid negative tiles (this is the slow part)...")
    
    # Pre-filter using centroids for speed
    candidates_centroids = candidate_negatives.copy()
    candidates_centroids['geometry'] = candidates_centroids.centroid
    
    # Find tiles whose centers are in safe zone
    center_safe = gpd.sjoin(candidates_centroids, safe_gdf, how="inner", predicate="intersects")
    potential_negatives = candidate_negatives.loc[center_safe.index]
    potential_negatives = potential_negatives[~potential_negatives.index.duplicated()]
    
    LOGGER.info(f"Found {len(potential_negatives)} candidate negative tiles (center in safe zone).")
    
    # 4. Sample Negatives (Balance 1:1 or user ratio)
    # If we have more negatives than positives, sample.
    if len(potential_negatives) > num_positives:
        LOGGER.info(f"Downsampling negatives to match positives ({num_positives})...")
        selected_negatives = potential_negatives.sample(n=num_positives, random_state=42)
    else:
        LOGGER.info("Using all available negative tiles (positives outnumber safe negatives).")
        selected_negatives = potential_negatives
        
    return positive_tiles, selected_negatives

def plot_results(
    aoi: gpd.GeoDataFrame,
    atwell: gpd.GeoDataFrame,
    nwi: gpd.GeoDataFrame,
    safe: gpd.GeoDataFrame,
    pos_tiles: gpd.GeoDataFrame,
    neg_tiles: gpd.GeoDataFrame,
    output_path: Path
):
    """Visualizes the sampling strategy."""
    LOGGER.info(f"Plotting results to {output_path}...")
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # 1. AOI Background
    aoi.plot(ax=ax, color='white', edgecolor='black', linestyle='--', alpha=0.5)
    
    # 2. Safe Zone (Light Green)
    safe.plot(ax=ax, color='#E0FFE0', alpha=0.4, label='Safe Zone')
    
    # 3. NWI (Blue Hatch)
    if not nwi.empty:
        nwi.plot(ax=ax, color='none', edgecolor='blue', hatch='//', linewidth=0.3, alpha=0.3)
        
    # 4. Selected TILES (The actual training data)
    # Negatives = Blue Squares
    neg_tiles.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1.5, label='Negative Tile')
    
    # Positives = Red Squares
    pos_tiles.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, label='Positive Tile')
    
    # 5. Raw Atwell Data (Solid Red)
    atwell.plot(ax=ax, color='red', alpha=0.5)
    
    ax.set_title(f"Balanced Sampling Strategy\nPositives: {len(pos_tiles)} | Negatives: {len(neg_tiles)}", fontsize=15)
    ax.axis('off')
    
    # Custom Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color='red', alpha=0.5, label='Atwell Wetland'),
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=1.5, label='Positive Tile'),
        mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=1.5, label='Negative Tile (Sampled)'),
        mpatches.Patch(color='#E0FFE0', alpha=0.4, label='Safe Zone'),
        mpatches.Patch(facecolor='none', edgecolor='blue', hatch='//', label='NWI (Excluded)'),
    ]
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def simulate_balanced_sampling_within(
    safe_gdf: gpd.GeoDataFrame,
    atwell_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    tile_size_m: float = 512.0
):
    """
    Simulates the tile selection using WITHIN test (Option 2).
    Returns positive tiles, selected negative tiles, AND rejected tiles for comparison.
    """
    LOGGER.info("Simulating balanced tile sampling with WITHIN test...")

    # 1. Generate Grid over AOI
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    cols = np.arange(minx, maxx, tile_size_m)
    rows = np.arange(miny, maxy, tile_size_m)

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(box(x, y, x + tile_size_m, y + tile_size_m))

    grid_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=aoi_gdf.crs)
    LOGGER.info(f"Generated {len(grid_gdf)} potential tiles.")

    # 2. Identify Positive Tiles (any tile touching Atwell)
    positive_tiles = gpd.sjoin(grid_gdf, atwell_gdf, how="inner", predicate="intersects")
    positive_tiles = positive_tiles[~positive_tiles.index.duplicated()].copy()

    num_positives = len(positive_tiles)
    LOGGER.info(f"Found {num_positives} Positive Tiles (containing verified wetland).")

    # 3. Identify Candidate Negative Tiles (not positive)
    candidate_negatives = grid_gdf.drop(positive_tiles.index)

    # 4. WITHIN TEST: Tiles must be ENTIRELY inside safe zone
    LOGGER.info("Identifying valid negative tiles using WITHIN test...")
    valid_negatives_joined = gpd.sjoin(
        candidate_negatives, safe_gdf, how="inner", predicate="within"
    )
    valid_negatives = candidate_negatives.loc[valid_negatives_joined.index]
    valid_negatives = valid_negatives[~valid_negatives.index.duplicated()].copy()

    LOGGER.info(f"Found {len(valid_negatives)} valid negative tiles (entirely within safe zone).")

    # 5. Identify REJECTED tiles (for visualization comparison)
    # These are tiles that would have passed centroid test but fail within test
    rejected_indices = candidate_negatives.index.difference(valid_negatives.index)
    rejected_tiles = candidate_negatives.loc[rejected_indices].copy()

    # Further filter rejected to those whose centroid IS in safe zone (edge tiles)
    rejected_centroids = rejected_tiles.copy()
    rejected_centroids['geometry'] = rejected_tiles.centroid
    centroid_in_safe = gpd.sjoin(rejected_centroids, safe_gdf, how="inner", predicate="intersects")
    edge_rejected = rejected_tiles.loc[
        rejected_tiles.index.intersection(centroid_in_safe.index)
    ].copy()

    LOGGER.info(f"Found {len(edge_rejected)} edge tiles (centroid safe, but tile extends outside).")

    # 6. Sample Negatives (Balance 1:1)
    if len(valid_negatives) > num_positives:
        LOGGER.info(f"Downsampling negatives to match positives ({num_positives})...")
        selected_negatives = valid_negatives.sample(n=num_positives, random_state=42)
    else:
        LOGGER.info("Using all available negative tiles.")
        selected_negatives = valid_negatives

    return positive_tiles, selected_negatives, valid_negatives, edge_rejected


def main():
    # Setup directories
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Real Data Paths - Michigan small dataset
    atwell_path = Path(r"C:\_code\python\wetlands-ml-atwell\data\MI_Atwell\small\atwell_wetlands.gpkg")
    nwi_path = Path(r"C:\_code\python\wetlands-ml-atwell\data\MI_Atwell\small\nwi_mi_wetlands.gpkg")

    # Generate AOI from Atwell data (bounding box + buffer)
    LOGGER.info("Generating AOI from Atwell delineations...")
    target_crs = "EPSG:5070"

    atwell_raw = gpd.read_file(atwell_path)
    if atwell_raw.crs is None:
        atwell_raw.set_crs("EPSG:4326", inplace=True)
    atwell_proj = atwell_raw.to_crs(target_crs)

    # Create AOI as bounding box + 1000m buffer
    aoi_buffer_m = 1000.0
    bounds = atwell_proj.total_bounds
    aoi_geom = box(*bounds).buffer(aoi_buffer_m)
    aoi = gpd.GeoDataFrame({"geometry": [aoi_geom]}, crs=target_crs)

    LOGGER.info(f"AOI created: {aoi_geom.bounds}")

    # Load NWI clipped to AOI
    # Note: bbox filter uses the CRS of the file being read, which is EPSG:5070
    aoi_bounds_proj = tuple(aoi.total_bounds)

    nwi_raw = gpd.read_file(nwi_path, bbox=aoi_bounds_proj)
    if nwi_raw.empty:
        LOGGER.warning("No NWI wetlands found within AOI!")
        nwi_proj = gpd.GeoDataFrame({"geometry": []}, crs=target_crs)
    else:
        nwi_proj = nwi_raw.to_crs(target_crs)
        nwi_proj = gpd.clip(nwi_proj, aoi)
        LOGGER.info(f"Loaded {len(nwi_proj)} NWI polygons within AOI.")

    # Clip Atwell to AOI
    atwell = gpd.clip(atwell_proj, aoi)

    # 2. Run Safe Zone Logic
    buffer_dist = 100.0
    safe_negatives, exclusion_union = generate_safe_negatives(aoi, atwell, nwi_proj, buffer_dist=buffer_dist)

    # 3. Run Balanced Sampling with WITHIN test
    # Using 512m tile size (512px at 1m resolution)
    tile_size_m = 512.0
    pos_tiles, neg_tiles_sampled, neg_tiles_all, edge_rejected = simulate_balanced_sampling_within(
        safe_negatives, atwell, aoi, tile_size_m=tile_size_m
    )

    # 4. Save ALL layers for QGIS inspection
    LOGGER.info(f"Saving vector outputs to {output_dir}...")

    # Core layers
    aoi.to_file(output_dir / "aoi.gpkg", driver="GPKG")
    atwell.to_file(output_dir / "atwell_wetlands.gpkg", driver="GPKG")
    nwi_proj.to_file(output_dir / "nwi_wetlands.gpkg", driver="GPKG")
    safe_negatives.to_file(output_dir / "safe_zone.gpkg", driver="GPKG")

    # Tile layers
    pos_tiles.to_file(output_dir / "tiles_positive.gpkg", driver="GPKG")
    neg_tiles_sampled.to_file(output_dir / "tiles_negative_sampled.gpkg", driver="GPKG")
    neg_tiles_all.to_file(output_dir / "tiles_negative_all_valid.gpkg", driver="GPKG")

    if not edge_rejected.empty:
        edge_rejected.to_file(output_dir / "tiles_edge_rejected.gpkg", driver="GPKG")
        LOGGER.info(f"  - tiles_edge_rejected.gpkg: {len(edge_rejected)} tiles")

    # Create Atwell buffer zone for visualization
    atwell_buffer = atwell.copy()
    atwell_buffer['geometry'] = atwell.buffer(buffer_dist)
    atwell_buffer.to_file(output_dir / "atwell_buffer_zone.gpkg", driver="GPKG")

    LOGGER.info("Saved layers:")
    LOGGER.info(f"  - aoi.gpkg: AOI bounding box")
    LOGGER.info(f"  - atwell_wetlands.gpkg: {len(atwell)} Atwell delineations")
    LOGGER.info(f"  - atwell_buffer_zone.gpkg: {buffer_dist}m buffer around Atwell")
    LOGGER.info(f"  - nwi_wetlands.gpkg: {len(nwi_proj)} NWI polygons")
    LOGGER.info(f"  - safe_zone.gpkg: {len(safe_negatives)} safe zone polygons")
    LOGGER.info(f"  - tiles_positive.gpkg: {len(pos_tiles)} positive tiles")
    LOGGER.info(f"  - tiles_negative_all_valid.gpkg: {len(neg_tiles_all)} valid negative tiles")
    LOGGER.info(f"  - tiles_negative_sampled.gpkg: {len(neg_tiles_sampled)} sampled negatives")

    # Summary stats
    LOGGER.info("\n=== SUMMARY ===")
    LOGGER.info(f"Tile size: {tile_size_m}m x {tile_size_m}m")
    LOGGER.info(f"Positive tiles: {len(pos_tiles)}")
    LOGGER.info(f"Valid negative tiles: {len(neg_tiles_all)}")
    LOGGER.info(f"Edge-rejected tiles: {len(edge_rejected)} (would leak NWI pixels)")
    LOGGER.info(f"Sampled negatives: {len(neg_tiles_sampled)}")

    # 5. Visualize
    plot_results(aoi, atwell, nwi_proj, safe_negatives, pos_tiles, neg_tiles_sampled, output_dir / "balanced_sampling_viz.png")
    LOGGER.info(f"Plot saved to {output_dir / 'balanced_sampling_viz.png'}")

if __name__ == "__main__":
    main()
