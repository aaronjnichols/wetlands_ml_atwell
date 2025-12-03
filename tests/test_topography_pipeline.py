from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from shapely.geometry import box

import pytest

from wetlands_ml_geoai.topography.config import TopographyStackConfig
from wetlands_ml_geoai.topography.pipeline import prepare_topography_stack
from wetlands_ml_geoai.topography.processing import _compute_tpi, write_topography_raster


def test_write_topography_raster_writes_four_bands(tmp_path: Path) -> None:
    """Test that topography raster contains only relative features (no raw elevation).

    Raw elevation is excluded because:
    - Wetlands exist at all elevations (coastal to alpine)
    - Absolute elevation creates geographic bias
    - Relative features (TPI, depression depth) capture what matters
    """
    dem_path = tmp_path / "dem.tif"
    transform = rasterio.transform.from_origin(0, 10, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": "EPSG:32618",
    }
    data = np.linspace(0, 100, 64, dtype="float32").reshape(8, 8)
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(data, 1)

    config = TopographyStackConfig(
        aoi=box(0, 0, 8, 8),
        target_grid_path=dem_path,
        output_dir=tmp_path,
        buffer_meters=0.0,
    )

    output_path = tmp_path / "topo.tif"
    result = write_topography_raster(config, [dem_path], output_path)

    assert result.exists()
    with rasterio.open(result) as src:
        assert src.count == 4  # Only relative features, no raw elevation
        names = list(src.descriptions)
        assert names == ["Slope", "TPI_small", "TPI_large", "DepressionDepth"]


def _legacy_tpi(dem: np.ndarray, radius: float, pixel_size: float) -> np.ndarray:
    kernel_size = int(max(radius / pixel_size, 1))
    footprint = np.ones((kernel_size * 2 + 1, kernel_size * 2 + 1))
    mean_filtered = ndimage.generic_filter(dem, np.nanmean, footprint=footprint, mode="nearest")
    return (dem - mean_filtered).astype("float32")


def test_compute_tpi_matches_previous_implementation() -> None:
    rng = np.random.default_rng(1234)
    dem = rng.uniform(0, 50, size=(256, 256)).astype("float32")
    dem[10:15, 40:45] = np.nan
    dem[120:130, 200:220] = np.nan
    pixel_size = 1.0

    for radius in (15.0, 75.0):
        legacy = _legacy_tpi(dem, radius, pixel_size)
        current = _compute_tpi(dem, radius, pixel_size)

        mask = np.isfinite(legacy)
        assert np.allclose(current[mask], legacy[mask], atol=1e-3)


def test_prepare_topography_stack_uses_local_dems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dem_path = tmp_path / "dem_local.tif"
    transform = rasterio.transform.from_origin(0, 10, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": "EPSG:32618",
    }
    data = np.linspace(0, 100, 64, dtype="float32").reshape(8, 8)
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(data, 1)

    def _should_not_run(*args, **kwargs):  # pragma: no cover - safety
        raise AssertionError("Remote DEM fetching should be skipped when dem_paths are provided")

    monkeypatch.setattr(
        "wetlands_ml_geoai.services.download.TopographyService.download",
        _should_not_run,
    )

    output_dir = tmp_path / "topography"
    config = TopographyStackConfig(
        aoi=box(0, 0, 8, 8),
        target_grid_path=dem_path,
        output_dir=output_dir,
        buffer_meters=0.0,
        dem_paths=(dem_path,),
    )

    result = prepare_topography_stack(config)
    assert result.exists()
    assert result.parent == output_dir

