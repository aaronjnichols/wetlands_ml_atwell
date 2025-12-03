"""Shared test fixtures for wetlands_ml_geoai tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS


@pytest.fixture
def minimal_raster_path(tmp_path: Path) -> Path:
    """Create a minimal 64x64 4-band GeoTIFF for testing.
    
    The raster simulates NAIP imagery with:
    - 4 bands (R, G, B, NIR)
    - uint8 dtype (0-255 range)
    - UTM Zone 16N CRS (common for US midwest)
    - 1m pixel resolution
    
    Returns:
        Path to the created GeoTIFF file.
    """
    raster_path = tmp_path / "test_naip.tif"
    
    # Create synthetic image data
    height, width = 64, 64
    bands = 4
    
    # Generate gradient patterns for each band (easy to verify visually)
    data = np.zeros((bands, height, width), dtype=np.uint8)
    for band in range(bands):
        # Different pattern per band: horizontal gradient, vertical, diagonal, etc.
        if band == 0:  # Red: horizontal gradient
            data[band] = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        elif band == 1:  # Green: vertical gradient
            data[band] = np.tile(np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1), (1, width))
        elif band == 2:  # Blue: checkerboard
            for i in range(height):
                for j in range(width):
                    data[band, i, j] = 255 if (i // 8 + j // 8) % 2 == 0 else 0
        else:  # NIR: uniform with some variation
            data[band] = 128 + (np.random.RandomState(42).rand(height, width) * 20).astype(np.uint8)
    
    # Define geospatial metadata
    transform = Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 4500000.0)  # 1m pixels
    crs = CRS.from_epsg(32616)  # UTM Zone 16N
    
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": bands,
        "crs": crs,
        "transform": transform,
        "nodata": None,
    }
    
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        dst.set_band_description(4, "NIR")
    
    return raster_path


@pytest.fixture
def minimal_float_raster_path(tmp_path: Path) -> Path:
    """Create a minimal 64x64 float32 raster for testing (e.g., Sentinel-2 style).
    
    Returns:
        Path to the created GeoTIFF file.
    """
    raster_path = tmp_path / "test_sentinel.tif"
    
    height, width = 64, 64
    bands = 7  # Typical Sentinel-2 band count per season
    
    # Generate data in 0-1 range (typical for Sentinel-2 reflectance)
    data = np.random.RandomState(42).rand(bands, height, width).astype(np.float32)
    
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)  # 10m pixels
    crs = CRS.from_epsg(32616)
    
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": bands,
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
    }
    
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data)
        for i, band_name in enumerate(["B03", "B04", "B05", "B06", "B08", "B11", "B12"], start=1):
            dst.set_band_description(i, band_name)
    
    return raster_path


@pytest.fixture
def mock_stack_manifest(tmp_path: Path, minimal_raster_path: Path) -> Path:
    """Create a minimal stack manifest JSON pointing to the mock raster.
    
    The manifest follows the format expected by RasterStack and training/inference.
    
    Args:
        tmp_path: pytest temporary directory fixture
        minimal_raster_path: Path to the mock NAIP raster
    
    Returns:
        Path to the created manifest JSON file.
    """
    manifest_path = tmp_path / "stack_manifest.json"
    
    # Read raster metadata to populate manifest
    with rasterio.open(minimal_raster_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs.to_string() if src.crs else None
    
    manifest = {
        "version": 1,
        "created_utc": "2025-01-01T00:00:00Z",
        "grid": {
            "crs": crs,
            "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            "width": width,
            "height": height,
            "nodata": -9999.0,
        },
        "sources": [
            {
                "type": "naip",
                "path": str(minimal_raster_path.resolve()),
                "band_labels": ["NAIP_R", "NAIP_G", "NAIP_B", "NAIP_NIR"],
                "scale_max": 255.0,
                "nodata": None,
            }
        ],
    }
    
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


@pytest.fixture
def mock_labels_gpkg(tmp_path: Path, minimal_raster_path: Path) -> Path:
    """Create a minimal GeoPackage with label polygons for training tests.
    
    Creates a single polygon covering a portion of the raster extent.
    
    Returns:
        Path to the created GeoPackage file.
    """
    import geopandas as gpd
    from shapely.geometry import box
    
    labels_path = tmp_path / "labels.gpkg"
    
    # Read raster bounds
    with rasterio.open(minimal_raster_path) as src:
        bounds = src.bounds
        crs = src.crs
    
    # Create a polygon covering the center of the raster
    center_x = (bounds.left + bounds.right) / 2
    center_y = (bounds.bottom + bounds.top) / 2
    size = 20  # 20m box
    
    polygon = box(
        center_x - size / 2,
        center_y - size / 2,
        center_x + size / 2,
        center_y + size / 2,
    )
    
    gdf = gpd.GeoDataFrame({"class": [1], "geometry": [polygon]}, crs=crs)
    gdf.to_file(labels_path, driver="GPKG")
    
    return labels_path


@pytest.fixture
def mock_manifest_index(tmp_path: Path, mock_stack_manifest: Path) -> Path:
    """Create a manifest index JSON listing one manifest.
    
    Returns:
        Path to the created manifest index JSON file.
    """
    index_path = tmp_path / "manifest_index.json"
    
    index = {
        "version": 1,
        "created_utc": "2025-01-01T00:00:00Z",
        "count": 1,
        "manifests": [str(mock_stack_manifest.resolve())],
    }
    
    index_path.write_text(json.dumps(index, indent=2))
    return index_path

