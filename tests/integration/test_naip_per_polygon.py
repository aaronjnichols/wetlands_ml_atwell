"""Integration tests for per-polygon NAIP download behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import box

from wetlands_ml_atwell.services.download import NaipDownloadRequest


@pytest.fixture
def separated_aois_gpkg(tmp_path: Path) -> Path:
    """Create GeoPackage with two spatially separated polygons (~2.5 degrees apart)."""
    # Two polygons far enough apart that their union bbox would be much larger
    polygon1 = box(-85.5, 41.5, -85.0, 42.0)  # Michigan area (~0.5 x 0.5 degrees)
    polygon2 = box(-83.0, 39.5, -82.5, 40.0)  # Ohio area (~0.5 x 0.5 degrees)
    # Union bbox would be: -85.5 to -82.5 (3 degrees) x 39.5 to 42.0 (2.5 degrees)

    gdf = gpd.GeoDataFrame(
        {"id": [1, 2], "geometry": [polygon1, polygon2]},
        crs="EPSG:4326"
    )

    gpkg_path = tmp_path / "separated_aois.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    return gpkg_path


@pytest.fixture
def adjacent_aois_gpkg(tmp_path: Path) -> Path:
    """Create GeoPackage with two adjacent polygons (share a border)."""
    polygon1 = box(-85.5, 41.5, -85.0, 42.0)
    polygon2 = box(-85.0, 41.5, -84.5, 42.0)  # Adjacent to polygon1

    gdf = gpd.GeoDataFrame(
        {"id": [1, 2], "geometry": [polygon1, polygon2]},
        crs="EPSG:4326"
    )

    gpkg_path = tmp_path / "adjacent_aois.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    return gpkg_path


@pytest.mark.integration
def test_naip_download_per_polygon_not_union(separated_aois_gpkg: Path, tmp_path: Path) -> None:
    """NAIP download should be called per polygon, not for union bbox.

    When AOIs are spatially separated, downloading for the union bbox would
    include unnecessary tiles in the gap between them. The per-polygon approach
    should call NaipService.download once per polygon.
    """
    from wetlands_ml_atwell.sentinel2.compositing import run_pipeline

    output_dir = tmp_path / "output"
    captured_requests: list = []

    def mock_download(request: NaipDownloadRequest) -> list:
        captured_requests.append(request)
        return []  # Return empty list (no actual downloads)

    with patch("wetlands_ml_atwell.sentinel2.compositing.NaipService") as MockService:
        mock_instance = MagicMock()
        mock_instance.download.side_effect = mock_download
        MockService.return_value = mock_instance

        # Also mock the STAC client to avoid network calls
        with patch("wetlands_ml_atwell.sentinel2.compositing.Client"):
            # Mock the processing iterator to avoid Sentinel-2 processing
            with patch("wetlands_ml_atwell.sentinel2.compositing._iter_aoi_processing") as mock_iter:
                mock_iter.return_value = iter([])  # Empty iterator

                run_pipeline(
                    aoi=str(separated_aois_gpkg),
                    years=[2022],
                    output_dir=output_dir,
                    auto_download_naip=True,
                    auto_download_naip_year=2022,
                )

    # Should be called twice (once per polygon), not once for the union
    assert len(captured_requests) == 2, (
        f"Expected 2 NAIP download calls (per polygon), got {len(captured_requests)}"
    )

    # Each bbox should be approximately 0.5 degrees, not the union's 2.5-3 degrees
    for i, req in enumerate(captured_requests):
        bounds = req.aoi.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        assert width <= 1.0, (
            f"Request {i+1} bbox too wide: {width:.2f} degrees (expected ~0.5, "
            f"would be ~3.0 if using union)"
        )
        assert height <= 1.0, (
            f"Request {i+1} bbox too tall: {height:.2f} degrees (expected ~0.5, "
            f"would be ~2.5 if using union)"
        )


@pytest.mark.integration
def test_naip_download_uses_shared_output_dir(separated_aois_gpkg: Path, tmp_path: Path) -> None:
    """All NAIP downloads should use the same output directory for deduplication."""
    from wetlands_ml_atwell.sentinel2.compositing import run_pipeline

    output_dir = tmp_path / "output"
    captured_output_dirs: list = []

    def mock_download(request: NaipDownloadRequest) -> list:
        captured_output_dirs.append(request.output_dir)
        return []

    with patch("wetlands_ml_atwell.sentinel2.compositing.NaipService") as MockService:
        mock_instance = MagicMock()
        mock_instance.download.side_effect = mock_download
        MockService.return_value = mock_instance

        with patch("wetlands_ml_atwell.sentinel2.compositing.Client"):
            with patch("wetlands_ml_atwell.sentinel2.compositing._iter_aoi_processing") as mock_iter:
                mock_iter.return_value = iter([])

                run_pipeline(
                    aoi=str(separated_aois_gpkg),
                    years=[2022],
                    output_dir=output_dir,
                    auto_download_naip=True,
                    auto_download_naip_year=2022,
                )

    # All downloads should target the same directory (for filename-based deduplication)
    assert len(captured_output_dirs) == 2
    assert captured_output_dirs[0] == captured_output_dirs[1], (
        "NAIP downloads should use shared output directory for deduplication"
    )
    assert captured_output_dirs[0] == output_dir / "naip_auto"


@pytest.mark.integration
def test_naip_download_deduplicates_results(tmp_path: Path) -> None:
    """Downloaded paths should be deduplicated across polygons.

    This test verifies that when multiple polygons return overlapping tiles,
    the deduplication logic removes duplicates from the final naip_sources list.
    """
    from wetlands_ml_atwell.sentinel2.compositing import _deduplicate_paths

    # Create fake tile paths (we test the actual deduplication function directly)
    fake_dir = tmp_path / "naip_auto"
    fake_dir.mkdir(parents=True)

    tile_a = fake_dir / "tile_a.tif"
    tile_b = fake_dir / "tile_b.tif"
    tile_a.touch()
    tile_b.touch()

    # Simulate what happens when two polygons download overlapping tiles
    polygon1_downloads = [tile_a, tile_b]
    polygon2_downloads = [tile_a]  # tile_a is shared

    all_downloads = polygon1_downloads + polygon2_downloads  # [a, b, a]

    # Verify deduplication works
    deduplicated = _deduplicate_paths(all_downloads)

    assert len(deduplicated) == 2, (
        f"Expected 2 unique tiles, got {len(deduplicated)}"
    )
    assert tile_a in deduplicated
    assert tile_b in deduplicated


@pytest.mark.integration
def test_single_polygon_aoi_works(tmp_path: Path) -> None:
    """Single polygon AOI should work without issues."""
    from wetlands_ml_atwell.sentinel2.compositing import run_pipeline

    polygon = box(-85.5, 41.5, -85.0, 42.0)
    gdf = gpd.GeoDataFrame({"id": [1], "geometry": [polygon]}, crs="EPSG:4326")
    gpkg_path = tmp_path / "single_aoi.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    output_dir = tmp_path / "output"
    captured_requests: list = []

    def mock_download(request: NaipDownloadRequest) -> list:
        captured_requests.append(request)
        return []

    with patch("wetlands_ml_atwell.sentinel2.compositing.NaipService") as MockService:
        mock_instance = MagicMock()
        mock_instance.download.side_effect = mock_download
        MockService.return_value = mock_instance

        with patch("wetlands_ml_atwell.sentinel2.compositing.Client"):
            with patch("wetlands_ml_atwell.sentinel2.compositing._iter_aoi_processing") as mock_iter:
                mock_iter.return_value = iter([])

                run_pipeline(
                    aoi=str(gpkg_path),
                    years=[2022],
                    output_dir=output_dir,
                    auto_download_naip=True,
                    auto_download_naip_year=2022,
                )

    # Should be called once for the single polygon
    assert len(captured_requests) == 1
