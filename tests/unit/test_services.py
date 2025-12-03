"""Unit tests for data acquisition services."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import box

from wetlands_ml_geoai.services.download import (
    NaipDownloadRequest,
    NaipService,
    WetlandsDownloadRequest,
    WetlandsService,
    TopographyService,
    DemProduct,
)

# Mock geoai if not installed (for CI/local envs where it might be missing)
if "geoai" not in sys.modules:
    sys.modules["geoai"] = MagicMock()
    sys.modules["geoai.download"] = MagicMock()


class TestNaipService:
    @pytest.fixture
    def service(self):
        return NaipService()

    @pytest.fixture
    def aoi(self):
        return box(0, 0, 1, 1)

    def test_download_delegates_to_geoai(self, service, aoi, tmp_path):
        request = NaipDownloadRequest(
            aoi=aoi,
            output_dir=tmp_path,
            year=2022,
            max_items=5,
            overwrite=True,
        )

        with patch("wetlands_ml_geoai.services.download.geo_download") as mock_geo:
            mock_geo.download_naip.return_value = ["tile1.tif", "tile2.tif"]
            
            results = service.download(request)
            
            mock_geo.download_naip.assert_called_once()
            call_args = mock_geo.download_naip.call_args
            assert call_args[0][0] == aoi.bounds
            assert call_args[1]["year"] == 2022
            assert call_args[1]["max_items"] == 5
            assert len(results) == 2
            assert isinstance(results[0], Path)

    def test_download_handles_missing_geoai(self, service, aoi, tmp_path):
        request = NaipDownloadRequest(aoi=aoi, output_dir=tmp_path, year=2022)

        # Simulate geoai module missing attribute
        with patch("wetlands_ml_geoai.services.download.geo_download", spec=[]) as mock_geo:
            with pytest.raises(AttributeError, match="geoai.download.download_naip is not available"):
                service.download(request)


class TestWetlandsService:
    @pytest.fixture
    def service(self):
        return WetlandsService()

    def test_download_skips_existing(self, service, tmp_path):
        output_path = tmp_path / "wetlands.gpkg"
        output_path.touch()
        
        request = WetlandsDownloadRequest(
            aoi=box(0, 0, 1, 1),
            output_path=output_path,
            overwrite=False
        )
        
        with patch("wetlands_ml_geoai.services.download.requests") as mock_requests:
            result = service.download(request)
            
            assert result == output_path
            mock_requests.get.assert_not_called()

    def test_download_fetches_and_clips(self, service, tmp_path):
        output_path = tmp_path / "wetlands.gpkg"
        request = WetlandsDownloadRequest(
            aoi=box(0, 0, 1, 1),
            output_path=output_path,
            overwrite=True
        )
        
        fake_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[0,0], [0,1], [1,1], [1,0], [0,0]]]},
                    "properties": {"attribute": "wetland"}
                }
            ]
        }

        with patch("wetlands_ml_geoai.services.download.requests") as mock_req:
            mock_req.get.return_value.json.return_value = fake_geojson
            mock_req.get.return_value.status_code = 200
            
            # Mock geopandas to avoid writing actual file and complex clipping logic in unit test
            with patch("wetlands_ml_geoai.services.download.gpd") as mock_gpd:
                mock_df = MagicMock()
                mock_gpd.GeoDataFrame.from_features.return_value = mock_df
                mock_gpd.clip.return_value = mock_df
                mock_df.__len__.return_value = 1
                
                result = service.download(request)
                
                assert result == output_path
                mock_req.get.assert_called_once()
                mock_df.to_file.assert_called_with(output_path, driver="GPKG")


class TestTopographyService:
    @pytest.fixture
    def service(self):
        return TopographyService()

    def test_download_orchestration(self, service, tmp_path):
        aoi = box(0, 0, 1, 1)
        
        # Mock the helper functions within the module
        with patch("wetlands_ml_geoai.services.download.fetch_dem_inventory") as mock_fetch, \
             patch("wetlands_ml_geoai.services.download.download_dem_products") as mock_down:
            
            mock_fetch.return_value = [
                DemProduct("id1", "url1", 100, [0,0,1,1], "2022")
            ]
            mock_down.return_value = [tmp_path / "dem.tif"]
            
            results = service.download(aoi, tmp_path)
            
            assert len(results) == 1
            mock_fetch.assert_called_once()
            mock_down.assert_called_once()

    def test_download_returns_empty_if_no_products(self, service, tmp_path):
        aoi = box(0, 0, 1, 1)
        
        with patch("wetlands_ml_geoai.services.download.fetch_dem_inventory") as mock_fetch:
            mock_fetch.return_value = []
            results = service.download(aoi, tmp_path)
            assert results == []

