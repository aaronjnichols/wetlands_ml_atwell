"""Unit tests for wetlands_ml_atwell.sentinel2.naip module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, box

from wetlands_ml_atwell.sentinel2.naip import (
    collect_naip_sources,
    prepare_naip_reference,
    _load_naip_footprint,
    _collect_naip_footprints,
    _clip_raster_to_polygon,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_naip_raster(tmp_path: Path) -> Path:
    """Create a sample NAIP-like GeoTIFF for testing."""
    raster_path = tmp_path / "naip_sample.tif"
    width, height = 100, 100
    # Use UTM zone 10N (EPSG:32610) like typical NAIP tiles
    crs = CRS.from_epsg(32610)
    # Bounds in UTM coords: small area around x=500000, y=4500000
    transform = from_bounds(
        500000, 4500000, 500100, 4500100, width, height
    )
    data = np.random.randint(0, 255, (4, height, width), dtype=np.uint8)
    
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        dst.set_band_description(4, "NIR")
    
    return raster_path


@pytest.fixture
def naip_tile_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple NAIP tiles."""
    tile_dir = tmp_path / "naip_tiles"
    tile_dir.mkdir()
    
    # Create multiple tiles
    for i in range(3):
        tile_path = tile_dir / f"tile_{i}.tif"
        width, height = 50, 50
        crs = CRS.from_epsg(32610)
        transform = from_bounds(
            500000 + (i * 50), 4500000, 500050 + (i * 50), 4500050, width, height
        )
        data = np.random.randint(0, 255, (4, height, width), dtype=np.uint8)
        
        with rasterio.open(
            tile_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=4,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)
    
    return tile_dir


# =============================================================================
# Tests for collect_naip_sources
# =============================================================================


class TestCollectNaipSources:
    """Tests for the collect_naip_sources function."""
    
    def test_collect_from_directory(self, naip_tile_directory: Path):
        """Should find all .tif files in a directory."""
        sources = collect_naip_sources([naip_tile_directory])
        assert len(sources) == 3
        for source in sources:
            assert source.suffix == ".tif"
    
    def test_collect_single_file(self, sample_naip_raster: Path):
        """Should return the file when given a direct path."""
        sources = collect_naip_sources([sample_naip_raster])
        assert len(sources) == 1
        assert sources[0] == sample_naip_raster.resolve()
    
    def test_deduplicates_paths(self, sample_naip_raster: Path):
        """Should remove duplicate paths."""
        # Pass the same file twice
        sources = collect_naip_sources([sample_naip_raster, sample_naip_raster])
        assert len(sources) == 1
    
    def test_mixed_files_and_directories(
        self, sample_naip_raster: Path, naip_tile_directory: Path
    ):
        """Should handle a mix of file and directory inputs."""
        sources = collect_naip_sources([sample_naip_raster, naip_tile_directory])
        # 1 direct file + 3 from directory
        assert len(sources) == 4
    
    def test_empty_input(self):
        """Should return empty list for empty input."""
        sources = collect_naip_sources([])
        assert sources == []
    
    def test_nonexistent_directory(self, tmp_path: Path):
        """Should handle nonexistent directories gracefully."""
        # This will treat it as a file path (non-directory)
        fake_path = tmp_path / "nonexistent"
        sources = collect_naip_sources([fake_path])
        # Will include the path even if it doesn't exist
        assert len(sources) == 1


# =============================================================================
# Tests for _load_naip_footprint
# =============================================================================


class TestLoadNaipFootprint:
    """Tests for the _load_naip_footprint function."""
    
    def test_returns_wgs84_polygon(self, sample_naip_raster: Path):
        """Should return a polygon in WGS84 coordinates."""
        # Clear cache for clean test
        _load_naip_footprint.cache_clear()
        
        footprint = _load_naip_footprint(str(sample_naip_raster))
        assert isinstance(footprint, Polygon)
        
        # Footprint should be in reasonable WGS84 range
        bounds = footprint.bounds
        # WGS84 longitude/latitude bounds
        assert -180 <= bounds[0] <= 180  # min_x (lon)
        assert -90 <= bounds[1] <= 90   # min_y (lat)
        assert -180 <= bounds[2] <= 180  # max_x (lon)
        assert -90 <= bounds[3] <= 90   # max_y (lat)
    
    def test_caches_result(self, sample_naip_raster: Path):
        """Should cache footprint results."""
        _load_naip_footprint.cache_clear()
        
        # First call
        footprint1 = _load_naip_footprint(str(sample_naip_raster))
        # Second call should return cached result
        footprint2 = _load_naip_footprint(str(sample_naip_raster))
        
        # Should be the same object (cached)
        assert footprint1 is footprint2
    
    def test_raises_for_missing_crs(self, tmp_path: Path):
        """Should raise ValueError if raster has no CRS."""
        # Create a raster without CRS
        no_crs_path = tmp_path / "no_crs.tif"
        data = np.zeros((1, 10, 10), dtype=np.uint8)
        with rasterio.open(
            no_crs_path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=np.uint8,
            # No crs specified
        ) as dst:
            dst.write(data)
        
        _load_naip_footprint.cache_clear()
        
        with pytest.raises(ValueError, match="missing CRS"):
            _load_naip_footprint(str(no_crs_path))


# =============================================================================
# Tests for _collect_naip_footprints
# =============================================================================


class TestCollectNaipFootprints:
    """Tests for the _collect_naip_footprints function."""
    
    def test_returns_path_polygon_tuples(self, naip_tile_directory: Path):
        """Should return list of (path, polygon) tuples."""
        _load_naip_footprint.cache_clear()
        
        sources = collect_naip_sources([naip_tile_directory])
        footprints = _collect_naip_footprints(sources)
        
        assert len(footprints) == 3
        for path, polygon in footprints:
            assert isinstance(path, Path)
            assert isinstance(polygon, Polygon)


# =============================================================================
# Tests for prepare_naip_reference
# =============================================================================


class TestPrepareNaipReference:
    """Tests for the prepare_naip_reference function."""
    
    def test_single_tile_no_resample(self, sample_naip_raster: Path, tmp_path: Path):
        """Should return the original tile when single input and no resampling."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        
        ref_path, profile, labels = prepare_naip_reference(
            [sample_naip_raster], working_dir
        )
        
        # Should return original path
        assert ref_path == sample_naip_raster
        assert "crs" in profile
        assert "transform" in profile
        # Labels use NAIP_BAND_LABELS from manifests module
        assert labels == ["NAIP_R", "NAIP_G", "NAIP_B", "NAIP_NIR"]
    
    def test_raises_for_empty_sources(self, tmp_path: Path):
        """Should raise ValueError for empty source list."""
        with pytest.raises(ValueError, match="No NAIP rasters"):
            prepare_naip_reference([], tmp_path)
    
    def test_raises_for_mismatched_crs(self, tmp_path: Path):
        """Should raise ValueError if tiles have different CRS."""
        # Create two tiles with different CRS
        tile1 = tmp_path / "tile1.tif"
        tile2 = tmp_path / "tile2.tif"
        
        data = np.zeros((4, 10, 10), dtype=np.uint8)
        
        # Tile 1: UTM Zone 10N
        with rasterio.open(
            tile1, "w", driver="GTiff",
            height=10, width=10, count=4, dtype=np.uint8,
            crs=CRS.from_epsg(32610),
            transform=from_bounds(0, 0, 10, 10, 10, 10),
        ) as dst:
            dst.write(data)
        
        # Tile 2: UTM Zone 11N (different!)
        with rasterio.open(
            tile2, "w", driver="GTiff",
            height=10, width=10, count=4, dtype=np.uint8,
            crs=CRS.from_epsg(32611),
            transform=from_bounds(0, 0, 10, 10, 10, 10),
        ) as dst:
            dst.write(data)
        
        with pytest.raises(ValueError, match="same CRS"):
            prepare_naip_reference([tile1, tile2], tmp_path)
    
    def test_raises_for_mismatched_band_count(self, tmp_path: Path):
        """Should raise ValueError if tiles have different band counts."""
        tile1 = tmp_path / "tile1.tif"
        tile2 = tmp_path / "tile2.tif"
        
        crs = CRS.from_epsg(32610)
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        
        # Tile 1: 4 bands
        with rasterio.open(
            tile1, "w", driver="GTiff",
            height=10, width=10, count=4, dtype=np.uint8,
            crs=crs, transform=transform,
        ) as dst:
            dst.write(np.zeros((4, 10, 10), dtype=np.uint8))
        
        # Tile 2: 3 bands (different!)
        with rasterio.open(
            tile2, "w", driver="GTiff",
            height=10, width=10, count=3, dtype=np.uint8,
            crs=crs, transform=transform,
        ) as dst:
            dst.write(np.zeros((3, 10, 10), dtype=np.uint8))
        
        with pytest.raises(ValueError, match="same band count"):
            prepare_naip_reference([tile1, tile2], tmp_path)
    
    def test_mosaic_multiple_tiles(self, naip_tile_directory: Path, tmp_path: Path):
        """Should mosaic multiple tiles into a single reference."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        
        sources = collect_naip_sources([naip_tile_directory])
        ref_path, profile, labels = prepare_naip_reference(sources, working_dir)
        
        # Should create a mosaic
        assert ref_path.name == "naip_mosaic.tif"
        assert ref_path.exists()
        
        # Profile should have merged dimensions
        assert profile["width"] is not None
        assert profile["height"] is not None


# =============================================================================
# Tests for _clip_raster_to_polygon
# =============================================================================


class TestClipRasterToPolygon:
    """Tests for the _clip_raster_to_polygon function."""
    
    def test_clips_to_polygon(self, sample_naip_raster: Path, tmp_path: Path):
        """Should clip raster to polygon boundary."""
        destination = tmp_path / "clipped.tif"
        
        # Get footprint of the raster to create a valid clip polygon
        footprint = _load_naip_footprint(str(sample_naip_raster))
        # Use a smaller polygon inside the footprint
        clip_polygon = footprint.buffer(-0.0001)  # Slightly smaller
        
        profile = _clip_raster_to_polygon(
            sample_naip_raster, clip_polygon, destination
        )
        
        assert destination.exists()
        assert "crs" in profile
        assert "transform" in profile
        assert "width" in profile
        assert "height" in profile
    
    def test_preserves_band_descriptions(self, sample_naip_raster: Path, tmp_path: Path):
        """Should preserve band descriptions in clipped output."""
        destination = tmp_path / "clipped.tif"
        
        footprint = _load_naip_footprint(str(sample_naip_raster))
        clip_polygon = footprint.buffer(-0.0001)
        
        _clip_raster_to_polygon(sample_naip_raster, clip_polygon, destination)
        
        with rasterio.open(destination) as dst:
            descriptions = dst.descriptions
            # At least some descriptions should be preserved
            assert any(desc is not None for desc in descriptions)
    
    def test_raises_for_missing_crs(self, tmp_path: Path):
        """Should raise ValueError if raster lacks CRS."""
        # Create raster without CRS
        no_crs_path = tmp_path / "no_crs.tif"
        data = np.zeros((1, 10, 10), dtype=np.uint8)
        with rasterio.open(
            no_crs_path, "w", driver="GTiff",
            height=10, width=10, count=1, dtype=np.uint8,
        ) as dst:
            dst.write(data)
        
        destination = tmp_path / "clipped.tif"
        polygon = box(-122, 37, -121, 38)  # WGS84 coords
        
        with pytest.raises(ValueError, match="lacks CRS"):
            _clip_raster_to_polygon(no_crs_path, polygon, destination)
    
    def test_creates_parent_directories(self, sample_naip_raster: Path, tmp_path: Path):
        """Should create parent directories if they don't exist."""
        destination = tmp_path / "nested" / "path" / "clipped.tif"
        
        footprint = _load_naip_footprint(str(sample_naip_raster))
        clip_polygon = footprint.buffer(-0.0001)
        
        _clip_raster_to_polygon(sample_naip_raster, clip_polygon, destination)
        
        assert destination.exists()
        assert destination.parent.exists()


# =============================================================================
# Module-level import tests
# =============================================================================


class TestNaipModuleImports:
    """Tests for verifying module exports work correctly."""
    
    def test_imports_from_sentinel2_package(self):
        """Should be importable from sentinel2 package."""
        from wetlands_ml_atwell.sentinel2 import (
            collect_naip_sources,
            prepare_naip_reference,
        )
        assert callable(collect_naip_sources)
        assert callable(prepare_naip_reference)
    
    def test_all_exports_defined(self):
        """Should have __all__ defined with expected exports."""
        from wetlands_ml_atwell.sentinel2 import naip
        
        expected = [
            "collect_naip_sources",
            "prepare_naip_reference",
            "_load_naip_footprint",
            "_collect_naip_footprints",
            "_clip_raster_to_polygon",
        ]
        for name in expected:
            assert name in naip.__all__

