"""Unit tests for seasonal compositing module."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from wetlands_ml_atwell.sentinel2.seasonal import (
    concatenate_seasons,
    write_dataarray,
    SENTINEL_BANDS,
)


class TestSentinelBands:
    """Tests for SENTINEL_BANDS constant."""
    
    def test_sentinel_bands_count(self):
        """Should have 7 bands."""
        assert len(SENTINEL_BANDS) == 7
    
    def test_sentinel_bands_values(self):
        """Should contain expected Sentinel-2 band names."""
        assert "B03" in SENTINEL_BANDS  # Green
        assert "B04" in SENTINEL_BANDS  # Red
        assert "B08" in SENTINEL_BANDS  # NIR


class TestWriteDataarray:
    """Tests for write_dataarray() function."""

    @pytest.fixture
    def mock_dataarray(self) -> xr.DataArray:
        """Create a mock 3D DataArray for testing."""
        data = np.random.rand(4, 10, 10).astype(np.float32)
        arr = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={
                "band": [1, 2, 3, 4],
                "y": range(10),
                "x": range(10),
            }
        )
        # Add CRS and transform (required by rioxarray)
        arr.rio.write_crs("EPSG:4326", inplace=True)
        arr.rio.write_transform(
            rasterio.transform.from_bounds(-85.5, 41.5, -85.0, 42.0, 10, 10),
            inplace=True
        )
        return arr

    def test_write_creates_file(self, tmp_path: Path, mock_dataarray: xr.DataArray):
        """Should create a GeoTIFF file."""
        output_path = tmp_path / "test.tif"
        labels = ["band1", "band2", "band3", "band4"]
        
        write_dataarray(mock_dataarray, output_path, labels)
        
        assert output_path.exists()

    def test_write_creates_parent_directories(self, tmp_path: Path, mock_dataarray: xr.DataArray):
        """Should create parent directories if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "test.tif"
        labels = ["band1", "band2", "band3", "band4"]
        
        write_dataarray(mock_dataarray, output_path, labels)
        
        assert output_path.exists()

    def test_write_correct_band_count(self, tmp_path: Path, mock_dataarray: xr.DataArray):
        """Output file should have correct number of bands."""
        output_path = tmp_path / "test.tif"
        labels = ["band1", "band2", "band3", "band4"]
        
        write_dataarray(mock_dataarray, output_path, labels)
        
        with rasterio.open(output_path) as src:
            assert src.count == 4

    def test_write_sets_band_descriptions(self, tmp_path: Path, mock_dataarray: xr.DataArray):
        """Should set band descriptions from labels."""
        output_path = tmp_path / "test.tif"
        labels = ["Red", "Green", "Blue", "NIR"]
        
        write_dataarray(mock_dataarray, output_path, labels)
        
        with rasterio.open(output_path) as src:
            assert src.descriptions[0] == "Red"
            assert src.descriptions[1] == "Green"
            assert src.descriptions[2] == "Blue"
            assert src.descriptions[3] == "NIR"

    def test_write_uses_deflate_compression(self, tmp_path: Path, mock_dataarray: xr.DataArray):
        """Should use deflate compression."""
        output_path = tmp_path / "test.tif"
        labels = ["band1", "band2", "band3", "band4"]
        
        write_dataarray(mock_dataarray, output_path, labels)
        
        with rasterio.open(output_path) as src:
            assert src.compression.name == "deflate"

    def test_write_raises_on_label_count_mismatch(self, tmp_path: Path, mock_dataarray: xr.DataArray):
        """Should raise ValueError if label count doesn't match bands."""
        output_path = tmp_path / "test.tif"
        labels = ["band1", "band2"]  # Only 2 labels for 4 bands
        
        with pytest.raises(ValueError, match="Band label count"):
            write_dataarray(mock_dataarray, output_path, labels)


class TestConcatenateSeasons:
    """Tests for concatenate_seasons() function."""

    @pytest.fixture
    def mock_seasonal_composites(self) -> dict:
        """Create mock seasonal composites for testing."""
        # Each season has 7 bands (matching SENTINEL_BANDS)
        def create_composite(seed):
            np.random.seed(seed)
            data = np.random.rand(7, 10, 10).astype(np.float32)
            arr = xr.DataArray(
                data,
                dims=["band", "y", "x"],
                coords={
                    "band": [1, 2, 3, 4, 5, 6, 7],
                    "y": range(10),
                    "x": range(10),
                }
            )
            arr.rio.write_crs("EPSG:4326", inplace=True)
            arr.rio.write_transform(
                rasterio.transform.from_bounds(-85.5, 41.5, -85.0, 42.0, 10, 10),
                inplace=True
            )
            return arr
        
        return {
            "SPR": create_composite(1),
            "SUM": create_composite(2),
            "FAL": create_composite(3),
        }

    def test_concatenate_returns_correct_shape(self, mock_seasonal_composites):
        """Combined array should have bands from all seasons."""
        combined, labels = concatenate_seasons(
            mock_seasonal_composites,
            order=["SPR", "SUM", "FAL"]
        )
        
        # 7 bands per season * 3 seasons = 21 bands
        assert combined.sizes["band"] == 21
        assert combined.sizes["y"] == 10
        assert combined.sizes["x"] == 10

    def test_concatenate_returns_correct_labels(self, mock_seasonal_composites):
        """Labels should include season prefix."""
        combined, labels = concatenate_seasons(
            mock_seasonal_composites,
            order=["SPR", "SUM", "FAL"]
        )
        
        # Check first season labels
        assert labels[0] == "S2_SPR_B03"
        assert labels[1] == "S2_SPR_B04"
        
        # Check second season starts at index 7
        assert labels[7] == "S2_SUM_B03"
        
        # Total labels
        assert len(labels) == 21

    def test_concatenate_respects_order(self, mock_seasonal_composites):
        """Order parameter should control season ordering."""
        combined1, labels1 = concatenate_seasons(
            mock_seasonal_composites,
            order=["SPR", "SUM", "FAL"]
        )
        combined2, labels2 = concatenate_seasons(
            mock_seasonal_composites,
            order=["FAL", "SUM", "SPR"]
        )
        
        # First band of combined1 should be SPR
        assert labels1[0].startswith("S2_SPR")
        
        # First band of combined2 should be FAL
        assert labels2[0].startswith("S2_FAL")

    def test_concatenate_preserves_crs(self, mock_seasonal_composites):
        """Combined array should preserve CRS from input."""
        combined, labels = concatenate_seasons(
            mock_seasonal_composites,
            order=["SPR", "SUM"]
        )
        
        assert combined.rio.crs is not None

    def test_concatenate_output_is_float32(self, mock_seasonal_composites):
        """Output should be float32 dtype."""
        combined, labels = concatenate_seasons(
            mock_seasonal_composites,
            order=["SPR", "SUM"]
        )
        
        assert combined.dtype == np.float32

    def test_concatenate_subset_of_seasons(self, mock_seasonal_composites):
        """Should work with subset of seasons."""
        combined, labels = concatenate_seasons(
            mock_seasonal_composites,
            order=["SPR"]  # Just one season
        )
        
        # Only 7 bands from one season
        assert combined.sizes["band"] == 7
        assert len(labels) == 7
        assert all("SPR" in label for label in labels)

