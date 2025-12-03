"""Smoke tests for Sentinel-2 compositing pipeline.

These tests verify the CLI can be invoked and basic argument parsing works.
External API calls (STAC, NAIP downloads) are mocked to avoid network dependencies.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSentinel2CliParsing:
    """Test that CLI argument parsing works correctly."""

    def test_cli_module_imports(self):
        """The sentinel2.cli module should be importable."""
        from wetlands_ml_geoai.sentinel2 import cli
        assert hasattr(cli, "main")
        assert hasattr(cli, "build_parser")

    def test_parser_requires_aoi(self):
        """The parser should require --aoi argument."""
        from wetlands_ml_geoai.sentinel2.cli import build_parser
        
        parser = build_parser()
        with pytest.raises(SystemExit):
            # Missing required --aoi
            parser.parse_args(["--years", "2023", "--output-dir", "/tmp/out"])

    def test_parser_requires_years(self):
        """The parser should require --years argument."""
        from wetlands_ml_geoai.sentinel2.cli import build_parser
        
        parser = build_parser()
        with pytest.raises(SystemExit):
            # Missing required --years
            parser.parse_args(["--aoi", "test.gpkg", "--output-dir", "/tmp/out"])

    def test_parser_requires_output_dir(self):
        """The parser should require --output-dir argument."""
        from wetlands_ml_geoai.sentinel2.cli import build_parser
        
        parser = build_parser()
        with pytest.raises(SystemExit):
            # Missing required --output-dir
            parser.parse_args(["--aoi", "test.gpkg", "--years", "2023"])

    def test_parser_accepts_valid_args(self):
        """The parser should accept all required arguments."""
        from wetlands_ml_geoai.sentinel2.cli import build_parser
        
        parser = build_parser()
        args = parser.parse_args([
            "--aoi", "test.gpkg",
            "--years", "2022", "2023",
            "--output-dir", "/tmp/output",
        ])
        
        assert args.aoi == "test.gpkg"
        assert args.years == [2022, 2023]
        assert args.output_dir == Path("/tmp/output")

    def test_parser_accepts_optional_args(self):
        """The parser should accept optional arguments."""
        from wetlands_ml_geoai.sentinel2.cli import build_parser
        
        parser = build_parser()
        args = parser.parse_args([
            "--aoi", "test.gpkg",
            "--years", "2023",
            "--output-dir", "/tmp/output",
            "--cloud-cover", "30",
            "--seasons", "SPR", "SUM",
            "--auto-download-naip",
        ])
        
        assert args.cloud_cover == 30.0
        assert args.seasons == ["SPR", "SUM"]
        assert args.auto_download_naip is True


class TestSentinel2Compositing:
    """Test compositing module functionality."""

    def test_compositing_module_imports(self):
        """The compositing module should be importable."""
        from wetlands_ml_geoai.sentinel2 import compositing
        assert hasattr(compositing, "run_pipeline")
        assert hasattr(compositing, "parse_aoi")

    def test_parse_aoi_with_bbox_string(self):
        """parse_aoi should accept bbox as comma-separated string."""
        from wetlands_ml_geoai.sentinel2.compositing import parse_aoi
        
        # Simple bounding box
        geom = parse_aoi("-85.5,41.5,-85.0,42.0")
        
        assert geom is not None
        assert not geom.is_empty
        # Check bounds approximately match input
        bounds = geom.bounds
        assert bounds[0] == pytest.approx(-85.5, rel=1e-3)
        assert bounds[1] == pytest.approx(41.5, rel=1e-3)

    def test_parse_aoi_with_wkt(self):
        """parse_aoi should accept WKT geometry."""
        from wetlands_ml_geoai.sentinel2.compositing import parse_aoi
        
        wkt = "POLYGON ((-85.5 41.5, -85.0 41.5, -85.0 42.0, -85.5 42.0, -85.5 41.5))"
        geom = parse_aoi(wkt)
        
        assert geom is not None
        assert not geom.is_empty
        assert geom.geom_type == "Polygon"

    def test_parse_aoi_with_geojson(self):
        """parse_aoi should accept GeoJSON string."""
        from wetlands_ml_geoai.sentinel2.compositing import parse_aoi
        
        geojson = json.dumps({
            "type": "Polygon",
            "coordinates": [[
                [-85.5, 41.5],
                [-85.0, 41.5],
                [-85.0, 42.0],
                [-85.5, 42.0],
                [-85.5, 41.5],
            ]]
        })
        geom = parse_aoi(geojson)
        
        assert geom is not None
        assert not geom.is_empty

    def test_parse_aoi_with_gpkg_file(self, mock_labels_gpkg):
        """parse_aoi should accept GeoPackage file path."""
        from wetlands_ml_geoai.sentinel2.compositing import parse_aoi
        
        geom = parse_aoi(str(mock_labels_gpkg))
        
        assert geom is not None
        assert not geom.is_empty


@pytest.mark.integration
class TestSentinel2PipelineWithMocks:
    """Integration tests with mocked external services."""

    def test_pipeline_validates_naip_paths(self, tmp_path):
        """Pipeline should raise error for non-existent NAIP paths."""
        from rasterio.errors import RasterioIOError
        from wetlands_ml_geoai.sentinel2.compositing import run_pipeline
        
        # The pipeline raises RasterioIOError when it tries to read footprints
        # from a non-existent file
        with pytest.raises(RasterioIOError):
            run_pipeline(
                aoi="-85.5,41.5,-85.0,42.0",
                years=[2023],
                output_dir=tmp_path / "output",
                naip_paths=[tmp_path / "nonexistent.tif"],
            )

    @patch("wetlands_ml_geoai.sentinel2.compositing.Client")
    def test_pipeline_creates_output_directory(
        self,
        mock_stac_client,
        tmp_path,
        minimal_raster_path,
    ):
        """Pipeline should create output directory if it doesn't exist."""
        from wetlands_ml_geoai.sentinel2.compositing import run_pipeline
        
        output_dir = tmp_path / "new_output"
        assert not output_dir.exists()
        
        # Mock STAC client to return empty results (no scenes found)
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value.get_items.return_value = []
        mock_stac_client.open.return_value = mock_client_instance
        
        # Run pipeline with valid NAIP path
        run_pipeline(
            aoi="-85.5,41.5,-85.0,42.0",
            years=[2023],
            output_dir=output_dir,
            naip_paths=[minimal_raster_path],
        )
        
        # Output directory should now exist
        assert output_dir.exists()

