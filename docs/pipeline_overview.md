# Wetlands ML GeoAI Pipeline Overview

This document summarizes the major data and control flows in the current codebase. It serves as a baseline reference before the refactor described in `ref.plan.md`.

## Training Pipelines

- Entrypoints: `train_unet.py` located both at project root (CLI shim) and in `src/wetlands_ml_geoai/` (full implementation).
- Shared flow:
  1. Parse CLI arguments/env vars to resolve raster, labels, manifest (single file or directory of manifests), tiling, and optimization parameters.
  2. Expand paths, validate existence, and derive a tile export directory.
  3. Export image/label tiles via `geoai.export_geotiff_tiles`, using NAIP rasters or manifest-backed stacks.
  4. For manifest scenarios, call `rewrite_tile_images` from `stacking.py` for each manifest in the manifest directory, enabling per-AOI stacks to rewrite tiles in place.
  5. Derive input channel count from raster metadata or the first manifest band count.
  6. Launch `geoai` UNet training helpers (`train_segmentation_model`).
  7. Persist checkpoints into `<tiles>/models_unet` folders.

- Unique pieces:
  - UNet training computes optional target resizing and label tile statistics (`_analyze_label_tiles`).

## Inference Pipelines

- Entrypoints: `test_unet.py` in both root and `src/`.
- Flow for manifest-backed streaming inference:
  1. Parse CLI options to resolve raster/manifest, checkpoints, window sizes, thresholds, and output directories.
  2. Load manifests via `stacking.load_manifest`; instantiate `RasterStack` for windowed reads.
  3. Build Torch UNet models (`get_smp_model`) with channel counts from stack metadata.
  4. Slide windows using `_compute_offsets`, gather predictions per window, and aggregate into raster arrays.
  5. Write GeoTIFF predictions and vectorize via `geoai.raster_to_vector`.

- When a raw raster path is supplied, the CLI defers to `geoai.semantic_segmentation`, keeping only output-path orchestration locally.

## Sentinel-2 Seasonal Processing

- Entrypoint: `sentinel2_processing.py` (root and `src/`).
- Combined responsibilities inside `src/wetlands_ml_geoai/sentinel2_processing.py`:
  - Argument parsing and CLI dispatch.
  - AOI parsing (files, inline JSON, bbox lists, or WKT strings).
  - NAIP reference preparation, including mosaicking, resampling, and manifest scaffolding. Multiple disjoint AOI polygons are processed independently, with Sentinel-2 composites and NAIP/topography rasters clipped per AOI.
  - STAC queries for Sentinel-2 imagery, filtering by season, cloud masking via SCL, generating composites with Stackstac & Dask.
  - Progress reporting utilities (custom progress bar classes).
  - Optional downloads for NAIP, wetlands, and LiDAR topography that feed into the training stack. Topography processing can also reuse pre-downloaded DEM tiles supplied via `--topography-dem-dir`.
- Manifest writing (`write_stack_manifest`) now supports per-AOI outputs and produces a directory of manifests plus an index file (`manifest_index.json`), allowing UNet training to ingest multi-AOI stacks without manual concatenation.

## LiDAR Topography Processing

- Entrypoint: `topography/cli.py` (Python module and `scripts/windows/run_topography_processing.bat`).
- Responsibilities:
  - Query USGS 3DEP (TNM Access) for buffered AOI coverage, download and cache 1 m DEM tiles, or reuse pre-downloaded DEM GeoTIFFs when supplied through configuration.
  - Mosaic DEM to the NAIP/Sentinel grid, respecting a configurable buffer to avoid edge artifacts.
  - Compute derived bands: slope, TPI (small/large radii), and depression depth; output float32 raster with nodata propagation. Note: Raw elevation is intentionally excluded as it creates geographic bias and doesn't improve wetland detection (wetlands exist at all elevations).
  - Register topography raster in the stack manifest so training/inference consume the additional terrain channels transparently.
  - Sentinel-2 seasonal pipeline can auto-generate these rasters when the corresponding flags are provided; the standalone CLI remains available for manual runs.
