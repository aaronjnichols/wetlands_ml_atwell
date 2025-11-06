# Topography Module Design

## Package Layout

- `src/wetlands_ml_geoai/topography/__init__.py`
  - Export public helpers: `prepare_topography_stack`, `TopographyStackConfig`.
- `src/wetlands_ml_geoai/topography/config.py`
  - Dataclasses for AOI, buffer distance (meters), DEM resolution, local cache paths, optional pre-downloaded DEM sources, and manifest interaction.
- `src/wetlands_ml_geoai/topography/download.py`
  - `_fetch_3dep_inventory` to query TNM Access API.
  - `_download_products` to stream GeoTIFFs with caching; returns metadata records.
  - `_prepare_dem_tiles` to reproject/resample downloaded tiles to stack CRS.
- `src/wetlands_ml_geoai/topography/processing.py`
  - `_mosaic_buffered_dem` to stitch buffered DEM array aligned to NAIP grid.
  - `_compute_slope`, `_compute_tpi`, `_compute_depression_depth` using rasterio/numpy/scipy.
  - `_write_derivative_raster` to crop to the training extent and set band names.
- `src/wetlands_ml_geoai/topography/cli.py`
  - Argument parsing mirroring `sentinel2.cli` options (AOI path, buffer meters, output dir, manifest path).
  - Calls `prepare_topography_stack` and prints manifest guidance.

## Data Flow

1. Parse CLI args → build `TopographyStackConfig` with AOI geometry, target grid (from NAIP or existing manifest), buffer size.
2. Query TNM Access with buffered AOI; cache raw DEM GeoTIFFs under `data/topography/raw/`. When `dem_dir`/`dem_paths` are provided, reuse those GeoTIFFs and skip the download step.
3. Mosaic and reproject DEM into target grid + buffer using rasterio.merge; maintain nodata mask and retain the resampled elevation surface.
4. Run derivatives on buffered DEM; drop buffer and write `float32` GeoTIFF at stack resolution with band labels `['Elevation', 'Slope', 'TPI_small', 'TPI_large', 'DepressionDepth']`.
5. Return metadata object referencing the derivative raster for manifest integration.

## Batch Script Integration

- New Windows batch runner `scripts/windows/run_topography_processing.bat` to activate venv and invoke `python -m wetlands_ml_geoai.topography.cli`.
- Align CLI env vars (`AOI`, `BUFFER_METERS`, `OUTPUT_DIR`, `STACK_MANIFEST`) with existing sentinel scripts.


