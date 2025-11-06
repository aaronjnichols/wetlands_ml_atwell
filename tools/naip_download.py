import geopandas as gpd
from geoai.download import download_naip


# user inputs
output_dir = r"data\naip_dte"
aoi_gpkg = r"data\del.gpkg"
max_items = 2
year = 2022

# get aoi polygons in WGS84
aoi_gdf = gpd.read_file(aoi_gpkg)
aoi_wgs84 = aoi_gdf.to_crs(4326)

# Loop through each feature/polygon in the AOI
all_naip_paths = []
for idx, row in aoi_wgs84.iterrows():
    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds
    aoi_bbox = (minx, miny, maxx, maxy)

    # download naip
    naip_paths = download_naip(
        aoi_bbox,
        output_dir=output_dir,
        year=year,
        max_items=max_items,
        overwrite=False,
        preview=False,
    )
    all_naip_paths.append(naip_paths)

print(all_naip_paths)