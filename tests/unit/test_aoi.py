"""Unit tests for AOI parsing module."""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from wetlands_ml_geoai.sentinel2.aoi import (
    parse_aoi,
    buffer_in_meters,
    extract_aoi_polygons,
)


class TestParseAoi:
    """Tests for parse_aoi() function."""

    def test_parse_bbox_string(self):
        """Should parse comma-separated bounding box string."""
        aoi = "-85.5,41.5,-85.0,42.0"
        geom = parse_aoi(aoi)
        
        assert geom.is_valid
        assert not geom.is_empty
        bounds = geom.bounds
        assert bounds[0] == pytest.approx(-85.5)
        assert bounds[1] == pytest.approx(41.5)
        assert bounds[2] == pytest.approx(-85.0)
        assert bounds[3] == pytest.approx(42.0)

    def test_parse_json_array_bbox(self):
        """Should parse JSON array bounding box."""
        aoi = "[-85.5, 41.5, -85.0, 42.0]"
        geom = parse_aoi(aoi)
        
        assert geom.is_valid
        bounds = geom.bounds
        assert bounds[0] == pytest.approx(-85.5)
        assert bounds[1] == pytest.approx(41.5)

    def test_parse_wkt_polygon(self):
        """Should parse WKT polygon string."""
        aoi = "POLYGON ((-85.5 41.5, -85.0 41.5, -85.0 42.0, -85.5 42.0, -85.5 41.5))"
        geom = parse_aoi(aoi)
        
        assert isinstance(geom, Polygon)
        assert geom.is_valid

    def test_parse_geojson_geometry(self):
        """Should parse GeoJSON geometry object."""
        aoi = json.dumps({
            "type": "Polygon",
            "coordinates": [[
                [-85.5, 41.5],
                [-85.0, 41.5],
                [-85.0, 42.0],
                [-85.5, 42.0],
                [-85.5, 41.5]
            ]]
        })
        geom = parse_aoi(aoi)
        
        assert isinstance(geom, Polygon)
        assert geom.is_valid

    def test_parse_geojson_feature(self):
        """Should parse GeoJSON feature with geometry property."""
        aoi = json.dumps({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-85.5, 41.5],
                    [-85.0, 41.5],
                    [-85.0, 42.0],
                    [-85.5, 42.0],
                    [-85.5, 41.5]
                ]]
            },
            "properties": {}
        })
        geom = parse_aoi(aoi)
        
        assert isinstance(geom, Polygon)
        assert geom.is_valid

    def test_parse_geopackage_file(self, tmp_path: Path):
        """Should parse GeoPackage file."""
        gpkg_path = tmp_path / "aoi.gpkg"
        
        # Create a simple polygon
        gdf = gpd.GeoDataFrame(
            {"name": ["test"]},
            geometry=[box(-85.5, 41.5, -85.0, 42.0)],
            crs="EPSG:4326"
        )
        gdf.to_file(gpkg_path, driver="GPKG")
        
        geom = parse_aoi(str(gpkg_path))
        
        assert geom.is_valid
        assert not geom.is_empty

    def test_parse_geopackage_with_reprojection(self, tmp_path: Path):
        """Should reproject GeoPackage from different CRS."""
        gpkg_path = tmp_path / "aoi_utm.gpkg"
        
        # Create polygon in UTM zone 16N
        gdf = gpd.GeoDataFrame(
            {"name": ["test"]},
            geometry=[box(500000, 4600000, 550000, 4650000)],
            crs="EPSG:32616"  # UTM 16N
        )
        gdf.to_file(gpkg_path, driver="GPKG")
        
        geom = parse_aoi(str(gpkg_path))
        
        # Result should be in WGS84 (bounds roughly in lon/lat range)
        bounds = geom.bounds
        assert -180 <= bounds[0] <= 180
        assert -90 <= bounds[1] <= 90

    def test_parse_empty_file_raises_error(self, tmp_path: Path):
        """Should raise ValueError for empty GeoPackage."""
        gpkg_path = tmp_path / "empty.gpkg"
        
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        gdf.to_file(gpkg_path, driver="GPKG")
        
        with pytest.raises(ValueError, match="no features"):
            parse_aoi(str(gpkg_path))

    def test_parse_empty_geometry_raises_error(self):
        """Should raise ValueError for empty geometry result."""
        # This WKT represents an empty polygon
        aoi = "POLYGON EMPTY"
        
        with pytest.raises(ValueError, match="empty"):
            parse_aoi(aoi)

    def test_parse_fixes_invalid_geometry(self):
        """Should fix invalid geometries with buffer(0)."""
        # Self-intersecting polygon (bowtie shape)
        aoi = "POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))"
        geom = parse_aoi(aoi)
        
        # After buffer(0), should be valid
        assert geom.is_valid


class TestBufferInMeters:
    """Tests for buffer_in_meters() function."""

    def test_no_buffer_returns_unchanged(self):
        """Should return unchanged geometry when buffer is 0."""
        geom = box(-85.5, 41.5, -85.0, 42.0)
        result = buffer_in_meters(geom, 0)
        
        assert result.equals(geom)

    def test_negative_buffer_returns_unchanged(self):
        """Should return unchanged geometry when buffer is negative."""
        geom = box(-85.5, 41.5, -85.0, 42.0)
        result = buffer_in_meters(geom, -100)
        
        assert result.equals(geom)

    def test_positive_buffer_expands_geometry(self):
        """Should expand geometry by approximately the buffer distance."""
        geom = box(-85.5, 41.5, -85.0, 42.0)
        buffer_m = 1000  # 1 km
        
        result = buffer_in_meters(geom, buffer_m)
        
        # Buffered geometry should be larger
        assert result.area > geom.area
        
        # Bounds should be expanded
        orig_bounds = geom.bounds
        new_bounds = result.bounds
        assert new_bounds[0] < orig_bounds[0]  # minx smaller
        assert new_bounds[1] < orig_bounds[1]  # miny smaller
        assert new_bounds[2] > orig_bounds[2]  # maxx larger
        assert new_bounds[3] > orig_bounds[3]  # maxy larger


class TestExtractAoiPolygons:
    """Tests for extract_aoi_polygons() function."""

    def test_extract_single_polygon(self):
        """Should return list with single polygon."""
        polygon = box(-85.5, 41.5, -85.0, 42.0)
        result = extract_aoi_polygons(polygon)
        
        assert len(result) == 1
        assert isinstance(result[0], Polygon)

    def test_extract_multipolygon(self):
        """Should return list of individual polygons from MultiPolygon."""
        poly1 = box(-86, 41, -85.5, 41.5)
        poly2 = box(-85, 42, -84.5, 42.5)
        multi = MultiPolygon([poly1, poly2])
        
        result = extract_aoi_polygons(multi)
        
        assert len(result) == 2
        assert all(isinstance(p, Polygon) for p in result)

    def test_extract_with_buffer(self):
        """Should apply buffer to extracted polygons."""
        polygon = box(-85.5, 41.5, -85.0, 42.0)
        
        unbuffered = extract_aoi_polygons(polygon, buffer_meters=0)
        buffered = extract_aoi_polygons(polygon, buffer_meters=1000)
        
        # Buffered polygon should be larger
        assert buffered[0].area > unbuffered[0].area

    def test_extract_empty_raises_error(self):
        """Should raise ValueError when no valid polygons extracted."""
        # Create a polygon then try to extract with a huge negative buffer
        # that would make it empty (but our function doesn't support negative buffers)
        # Instead, test with an empty multipolygon
        from shapely.geometry import MultiPolygon
        empty_multi = MultiPolygon([])
        
        with pytest.raises(ValueError, match="No valid AOI polygons"):
            extract_aoi_polygons(empty_multi)

    def test_extract_fixes_invalid_geometry(self):
        """Should fix invalid geometries during extraction."""
        # Self-intersecting polygon
        from shapely.geometry import Polygon as ShapelyPolygon
        invalid = ShapelyPolygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        
        result = extract_aoi_polygons(invalid)
        
        # All results should be valid
        assert all(p.is_valid for p in result)

