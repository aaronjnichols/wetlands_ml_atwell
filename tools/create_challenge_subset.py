"""
Script to find and generate a 'Challenge' subset of training data.

A 'Challenge' subset is defined as an area where:
1. We have Atwell data (Verified Positives).
2. We have NWI data (Unverified).
3. The NWI data is MORE abundant than the Atwell data.

This simulates the "False Negative" risk scenario: areas that look like wetlands
(according to NWI) but are not labeled by Atwell. We want to ensure our
sampling strategy correctly excludes these from the 'Negative' set.
"""

import logging
from pathlib import Path
import random
import geopandas as gpd
from shapely.geometry import box

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def find_challenge_area(
    atwell_path: Path,
    nwi_path: Path,
    output_dir: Path,
    window_size_meters: float = 5000.0,
    max_attempts: int = 50
):
    LOGGER.info(f"Loading Atwell data from {atwell_path}...")
    atwell_full = gpd.read_file(atwell_path)
    
    # Ensure we are in a metric CRS for the window calculation
    # If geographic, we'll project temporarily
    original_crs = atwell_full.crs
    working_crs = "EPSG:5070" # NAD83 / Conus Albers
    
    atwell_metric = atwell_full.to_crs(working_crs)
    
    LOGGER.info(f"Scanning for challenge areas (Window: {window_size_meters/1000}km x {window_size_meters/1000}km)...")
    
    best_candidate = None
    best_score = -1
    
    # Try random locations centered on Atwell polygons
    # We want to find a spot with a lot of "Extra" NWI data
    for i in range(max_attempts):
        # Pick a random Atwell polygon to center our search
        center_poly = atwell_metric.sample(1).iloc[0].geometry
        cx, cy = center_poly.centroid.x, center_poly.centroid.y
        
        # Create the search window
        minx = cx - window_size_meters / 2
        miny = cy - window_size_meters / 2
        maxx = cx + window_size_meters / 2
        maxy = cy + window_size_meters / 2
        
        search_box = box(minx, miny, maxx, maxy)
        
        # Convert search box back to NWI's CRS (likely 4326 or 5070) for query
        # Assuming NWI file allows bbox filtering in its native CRS
        # Let's assume NWI is same CRS or we reproject the box to 4326 for the reader
        search_box_geo = gpd.GeoSeries([search_box], crs=working_crs).to_crs(original_crs).iloc[0]
        
        # 1. Count Atwell in this box
        # (We already have full atwell in memory)
        # Reproject box to atwell's native CRS for intersection
        atwell_in_box = atwell_full[atwell_full.intersects(search_box_geo)]
        atwell_count = len(atwell_in_box)
        
        if atwell_count < 2:
            continue # Too sparse
            
        # 2. Count NWI in this box (Read from disk with filter)
        try:
            # Pass the bbox tuple
            nwi_subset = gpd.read_file(nwi_path, bbox=search_box_geo.bounds)
        except Exception as e:
            LOGGER.warning(f"Error reading NWI subset: {e}")
            continue
            
        nwi_count = len(nwi_subset)
        
        # 3. Evaluate "Challenge Score"
        # High score = Lots of NWI relative to Atwell (High potential for false negatives)
        # But we need at least SOME Atwell to train on.
        
        if nwi_count > atwell_count:
            ratio = nwi_count / atwell_count
            score = nwi_count  # Prioritize density
            
            LOGGER.info(f"Attempt {i+1}: Found {atwell_count} Atwell vs {nwi_count} NWI (Ratio {ratio:.1f})")
            
            if ratio > 1.5: # At least 50% more NWI than Atwell
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "atwell": atwell_in_box,
                        "nwi": nwi_subset,
                        "bbox": search_box_geo
                    }
                    
                    # If we find a really good one, break early
                    if nwi_count > 50 and ratio > 2.0:
                        LOGGER.info("Found excellent candidate! Stopping search.")
                        break
    
    if best_candidate:
        LOGGER.info(f"Selected area with {len(best_candidate['atwell'])} Atwell and {len(best_candidate['nwi'])} NWI polygons.")
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        
        atwell_out = output_dir / "sample_atwell.gpkg"
        nwi_out = output_dir / "sample_nwi.gpkg"
        
        best_candidate["atwell"].to_file(atwell_out, driver="GPKG")
        best_candidate["nwi"].to_file(nwi_out, driver="GPKG")
        
        LOGGER.info(f"Saved subsets to:")
        LOGGER.info(f"  - {atwell_out}")
        LOGGER.info(f"  - {nwi_out}")
        
        return atwell_out
    else:
        LOGGER.error("Could not find a suitable challenge area after multiple attempts.")
        return None

if __name__ == "__main__":
    # Define Paths
    base_dir = Path(__file__).parents[1] # Up one level from tools/
    data_dir = base_dir / "data"
    
    atwell_path = data_dir / "Atwell_Wetlands_MI.gpkg"
    nwi_path = data_dir / "NWI/MI_Wetlands_Geopackage.gpkg"
    output_dir = data_dir / "samples"
    
    if not atwell_path.exists() or not nwi_path.exists():
        print(f"Error: Data files not found at expected paths:\n{atwell_path}\n{nwi_path}")
    else:
        find_challenge_area(atwell_path, nwi_path, output_dir)


