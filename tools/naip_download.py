import sys
from pathlib import Path
import geopandas as gpd

# Ensure src is in path if running from tools/
project_root = Path(__file__).parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from wetlands_ml_atwell.services.download import NaipService, NaipDownloadRequest


def main():
    # user inputs
    output_dir = Path("data/naip_dte")
    aoi_gpkg = Path("data/del.gpkg")
    max_items = 2
    year = 2022
    
    if not aoi_gpkg.exists():
        print(f"Error: AOI file not found: {aoi_gpkg}")
        print("Please update the 'aoi_gpkg' variable in the script.")
        return

    # get aoi polygons in WGS84
    aoi_gdf = gpd.read_file(aoi_gpkg)
    aoi_wgs84 = aoi_gdf.to_crs(4326)

    service = NaipService()
    all_naip_paths = []

    # Loop through each feature/polygon in the AOI
    print(f"Processing {len(aoi_wgs84)} AOI polygons...")
    for idx, row in aoi_wgs84.iterrows():
        geom = row.geometry
        
        request = NaipDownloadRequest(
            aoi=geom,
            output_dir=output_dir,
            year=year,
            max_items=max_items,
            overwrite=False,
            preview=False
        )
        
        try:
            naip_paths = service.download(request)
            all_naip_paths.append(naip_paths)
            print(f"AOI {idx}: Downloaded {len(naip_paths)} tiles")
        except Exception as e:
            print(f"AOI {idx}: Failed - {e}")

    print("\nAll download paths:")
    print(all_naip_paths)


if __name__ == "__main__":
    main()
