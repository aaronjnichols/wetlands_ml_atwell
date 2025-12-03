
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

def main():
    # Setup directories
    base_dir = Path("tests/experimental")
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Real Data Paths
    # Using raw strings for Windows paths
    aoi_path = base_dir / "aoi.gpkg"
    atwell_path = Path(r"C:\_code\python\wetlands_ml_codex\data\Atwell_Wetlands_MI.gpkg")
    nwi_path = Path(r"C:\_code\python\wetlands_ml_codex\data\NWI\MI_Wetlands_Geopackage.gpkg")
    
    # 1. Load Real Data
    try:
        aoi, atwell, nwi = load_real_data(aoi_path, atwell_path, nwi_path)
    except Exception as e:
        LOGGER.error(f"Failed to load data: {e}")
        return
    
    # 2. Run Safe Zone Logic
    safe_negatives, _ = generate_safe_negatives(aoi, atwell, nwi, buffer_dist=100.0)
    
    # 3. Run Balanced Sampling Simulation
    # Using 256m tile size for visualization purposes (approx 25 pixels at 10m res, or bigger if Sentinel)
    # Standard Sentinel training chip is often 256px * 10m = 2560m. 
    # Let's use a smaller tile size (e.g. 500m) just to show the density for this visual check.
    pos_tiles, neg_tiles = simulate_balanced_sampling(safe_negatives, atwell, aoi, tile_size_m=500.0)
    
    # 4. Save Results
    output_gpkg = output_dir / "real_data_safe_negatives.gpkg"
    safe_negatives.to_file(output_gpkg, driver="GPKG")
    
    # Save tiles for inspection
    pos_tiles.to_file(output_dir / "tiles_positive.gpkg", driver="GPKG")
    neg_tiles.to_file(output_dir / "tiles_negative.gpkg", driver="GPKG")
    
    LOGGER.info(f"Saved vector outputs to {output_dir}")
    
    # 5. Visualize
    plot_results(aoi, atwell, nwi, safe_negatives, pos_tiles, neg_tiles, output_dir / "balanced_sampling_viz.png")

if __name__ == "__main__":
    main()
