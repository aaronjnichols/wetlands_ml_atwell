"""Tests to verify shared fixtures work correctly."""

import json

import numpy as np
import pytest
import rasterio


class TestMinimalRasterFixture:
    """Verify the minimal_raster_path fixture creates valid GeoTIFFs."""

    def test_raster_exists(self, minimal_raster_path):
        """The fixture should create a file."""
        assert minimal_raster_path.exists()
        assert minimal_raster_path.suffix == ".tif"

    def test_raster_is_readable(self, minimal_raster_path):
        """The raster should be readable with rasterio."""
        with rasterio.open(minimal_raster_path) as src:
            assert src.count == 4  # 4 bands (R, G, B, NIR)
            assert src.width == 64
            assert src.height == 64
            assert src.dtypes[0] == "uint8"
            assert src.crs is not None

    def test_raster_has_valid_data(self, minimal_raster_path):
        """The raster should contain valid pixel values."""
        with rasterio.open(minimal_raster_path) as src:
            data = src.read()
            assert data.shape == (4, 64, 64)
            assert data.dtype == np.uint8
            # Check data is not all zeros or all same value
            assert data.min() != data.max()


class TestMockStackManifest:
    """Verify the mock_stack_manifest fixture creates valid JSON."""

    def test_manifest_exists(self, mock_stack_manifest):
        """The fixture should create a JSON file."""
        assert mock_stack_manifest.exists()
        assert mock_stack_manifest.suffix == ".json"

    def test_manifest_is_valid_json(self, mock_stack_manifest):
        """The manifest should be parseable JSON."""
        content = mock_stack_manifest.read_text()
        data = json.loads(content)
        assert "grid" in data
        assert "sources" in data

    def test_manifest_points_to_valid_raster(self, mock_stack_manifest, minimal_raster_path):
        """The manifest should reference the mock raster path."""
        data = json.loads(mock_stack_manifest.read_text())
        naip_source = data["sources"][0]
        assert naip_source["type"] == "naip"
        # The path should be resolvable
        from pathlib import Path
        raster_path = Path(naip_source["path"])
        assert raster_path.exists()

    def test_manifest_can_be_loaded(self, mock_stack_manifest):
        """The manifest should be loadable by the stacking module."""
        from wetlands_ml_geoai.stacking import load_manifest
        
        manifest = load_manifest(mock_stack_manifest)
        assert manifest.grid is not None
        assert len(manifest.sources) >= 1
        assert manifest.naip is not None


class TestMockLabelsFixture:
    """Verify the mock_labels_gpkg fixture creates valid vector data."""

    def test_labels_exists(self, mock_labels_gpkg):
        """The fixture should create a GeoPackage file."""
        assert mock_labels_gpkg.exists()
        assert mock_labels_gpkg.suffix == ".gpkg"

    def test_labels_is_readable(self, mock_labels_gpkg):
        """The labels should be readable with geopandas."""
        import geopandas as gpd
        
        gdf = gpd.read_file(mock_labels_gpkg)
        assert len(gdf) >= 1
        assert gdf.crs is not None
        assert "geometry" in gdf.columns

