"""NAIP Download Tool with optional resolution reduction.

Downloads NAIP imagery for an AOI, optionally at reduced resolution for faster
downloads when full 0.6m resolution isn't needed.

Usage:
    python tools/naip_download.py

Configuration:
    Edit the USER INPUTS section below before running.
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Ensure src is in path if running from tools/
project_root = Path(__file__).parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from pystac_client import Client

# Rate limiting settings
DELAY_BETWEEN_TILES = 0.5  # seconds between tile downloads
DELAY_BETWEEN_AOIS = 1.0   # seconds between AOI queries
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # seconds to wait before retry


def download_naip_at_resolution(
    aoi_bounds: tuple,
    output_dir: Path,
    year: int,
    target_resolution: float = 10.0,
    max_items: int = 100,
    overwrite: bool = False,
) -> list[Path]:
    """Download NAIP imagery at a specified resolution using COG overviews.

    This approach streams data at the target resolution directly from the COG
    overviews, avoiding the need to download full-resolution tiles.

    Args:
        aoi_bounds: (minx, miny, maxx, maxy) in WGS84
        output_dir: Directory to save downloaded tiles
        year: Target year for NAIP imagery
        target_resolution: Desired resolution in meters (default 10m)
        max_items: Maximum number of STAC items to retrieve
        overwrite: Whether to overwrite existing files

    Returns:
        List of paths to downloaded GeoTIFF files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use Planetary Computer STAC for NAIP (has good COG support)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        collections=["naip"],
        bbox=aoi_bounds,
        datetime=f"{year}-01-01/{year}-12-31",
        max_items=max_items,
    )

    items = list(search.items())
    if not items:
        print(f"No NAIP items found for year {year}")
        return []

    print(f"Found {len(items)} NAIP items for year {year}")

    downloaded_paths = []

    for item in items:
        # Create output filename from item ID
        item_id = item.id.replace("/", "_").replace(":", "_")
        output_path = output_dir / f"{item_id}_{int(target_resolution)}m.tif"

        if output_path.exists() and not overwrite:
            print(f"  Skipping {item_id} (already exists)")
            downloaded_paths.append(output_path)
            continue

        print(f"  Downloading {item_id} at {target_resolution}m resolution...")

        # Get the image asset URL
        image_asset = item.assets.get("image")
        if not image_asset:
            print(f"    No 'image' asset found, skipping")
            continue

        image_url = image_asset.href

        # Retry logic for server errors
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Read at reduced resolution using rasterio overviews
                with rasterio.open(image_url) as src:
                    # Calculate the overview level closest to target resolution
                    native_res = src.res[0]
                    scale_factor = target_resolution / native_res

                    # Find appropriate overview level
                    if src.overviews(1):
                        overviews = src.overviews(1)
                        # Find overview closest to our target scale
                        best_overview = min(overviews, key=lambda x: abs(x - scale_factor))
                        out_shape = (
                            src.count,
                            int(src.height / best_overview),
                            int(src.width / best_overview),
                        )
                    else:
                        # No overviews, just downsample
                        out_shape = (
                            src.count,
                            int(src.height / scale_factor),
                            int(src.width / scale_factor),
                        )

                    # Read at reduced resolution
                    data = src.read(
                        out_shape=out_shape,
                        resampling=Resampling.bilinear,
                    )

                    # Calculate new transform
                    transform = src.transform * src.transform.scale(
                        src.width / out_shape[2],
                        src.height / out_shape[1],
                    )

                    src_crs = src.crs

                # Write to GeoTIFF
                profile = {
                    "driver": "GTiff",
                    "dtype": data.dtype,
                    "width": out_shape[2],
                    "height": out_shape[1],
                    "count": data.shape[0],
                    "crs": src_crs,
                    "transform": transform,
                    "compress": "deflate",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                }

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(data)

                    # Set band descriptions
                    band_names = ["Red", "Green", "Blue", "NIR"]
                    for i, name in enumerate(band_names[:data.shape[0]], 1):
                        dst.set_band_description(i, name)

                print(f"    Saved: {output_path}")
                downloaded_paths.append(output_path)

                # Delay between tiles to avoid rate limiting
                time.sleep(DELAY_BETWEEN_TILES)
                break  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e)
                if "500" in error_msg or "Internal Server Error" in error_msg or "503" in error_msg:
                    if attempt < MAX_RETRIES:
                        print(f"    Server error (attempt {attempt}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"    Failed after {MAX_RETRIES} attempts: {e}")
                else:
                    print(f"    Failed: {e}")
                break  # Non-retryable error or max retries reached

    return downloaded_paths


def main():
    # =========================================================================
    # USER INPUTS - Edit these values before running
    # =========================================================================
    output_dir = Path(r"C:\_code\python\wetlands-ml-atwell\data\MI_Atwell\base\naip")
    aoi_gpkg = Path(r"C:\_code\python\wetlands-ml-atwell\data\MI_Atwell\base\aois_base_train.gpkg")
    max_items = 100
    year = 2022

    # Target resolution in meters (set to None for full resolution ~0.6m)
    # Using 10m is ~250x less data than 0.6m (10^2 / 0.6^2)
    target_resolution = 10.0

    # =========================================================================

    if not aoi_gpkg.exists():
        print(f"Error: AOI file not found: {aoi_gpkg}")
        print("Please update the 'aoi_gpkg' variable in the script.")
        return

    # Get AOI bounds in WGS84
    aoi_gdf = gpd.read_file(aoi_gpkg)
    aoi_wgs84 = aoi_gdf.to_crs(4326)

    all_naip_paths = []

    if target_resolution is not None:
        # Fast download at reduced resolution using COG overviews
        print(f"Downloading NAIP at {target_resolution}m resolution (fast mode)...")
        print(f"Processing {len(aoi_wgs84)} AOI polygons...")

        for idx, row in aoi_wgs84.iterrows():
            geom = row.geometry
            bounds = geom.bounds  # (minx, miny, maxx, maxy)

            try:
                naip_paths = download_naip_at_resolution(
                    aoi_bounds=bounds,
                    output_dir=output_dir,
                    year=year,
                    target_resolution=target_resolution,
                    max_items=max_items,
                    overwrite=False,
                )
                all_naip_paths.extend(naip_paths)
                print(f"AOI {idx}: Downloaded {len(naip_paths)} tiles")

                # Delay between AOIs to avoid rate limiting
                time.sleep(DELAY_BETWEEN_AOIS)
            except Exception as e:
                print(f"AOI {idx}: Failed - {e}")
                # Still delay after failures to avoid hammering the server
                time.sleep(RETRY_DELAY)
    else:
        # Original full-resolution download via geoai
        from wetlands_ml_atwell.services.download import NaipService, NaipDownloadRequest

        print("Downloading NAIP at full resolution (slow mode)...")
        service = NaipService()

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
                all_naip_paths.extend(naip_paths)
                print(f"AOI {idx}: Downloaded {len(naip_paths)} tiles")
            except Exception as e:
                print(f"AOI {idx}: Failed - {e}")

    print("\nAll download paths:")
    for p in all_naip_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
