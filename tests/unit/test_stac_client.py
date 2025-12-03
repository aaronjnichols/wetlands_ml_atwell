"""Unit tests for STAC client module."""

from datetime import date

import pytest

from wetlands_ml_atwell.sentinel2.stac_client import (
    SENTINEL_BANDS,
    SENTINEL_COLLECTION,
    SENTINEL_SCALE_FACTOR,
    DEFAULT_SEASONS,
    SEASON_WINDOWS,
    BAND_TO_ASSET,
    SeasonConfig,
    season_date_range,
)


class TestConstants:
    """Tests for STAC client constants."""

    def test_sentinel_collection_value(self):
        """Should be the Sentinel-2 L2A collection ID."""
        assert SENTINEL_COLLECTION == "sentinel-2-l2a"

    def test_sentinel_bands_count(self):
        """Should have 7 bands."""
        assert len(SENTINEL_BANDS) == 7

    def test_sentinel_bands_includes_key_bands(self):
        """Should include key spectral bands."""
        assert "B03" in SENTINEL_BANDS  # Green
        assert "B04" in SENTINEL_BANDS  # Red
        assert "B08" in SENTINEL_BANDS  # NIR
        assert "B11" in SENTINEL_BANDS  # SWIR

    def test_sentinel_scale_factor(self):
        """Scale factor should convert DN to reflectance."""
        # DN of 10000 should give reflectance of 1.0
        assert SENTINEL_SCALE_FACTOR == 1 / 10000
        assert 10000 * SENTINEL_SCALE_FACTOR == 1.0

    def test_default_seasons(self):
        """Should include spring, summer, fall."""
        assert "SPR" in DEFAULT_SEASONS
        assert "SUM" in DEFAULT_SEASONS
        assert "FAL" in DEFAULT_SEASONS
        assert len(DEFAULT_SEASONS) == 3

    def test_season_windows_keys(self):
        """Season windows should match default seasons."""
        assert set(SEASON_WINDOWS.keys()) == set(DEFAULT_SEASONS)

    def test_season_windows_spring(self):
        """Spring should be March through May."""
        sm, sd, em, ed = SEASON_WINDOWS["SPR"]
        assert sm == 3  # March
        assert em == 5  # May

    def test_season_windows_summer(self):
        """Summer should be June through August."""
        sm, sd, em, ed = SEASON_WINDOWS["SUM"]
        assert sm == 6  # June
        assert em == 8  # August

    def test_season_windows_fall(self):
        """Fall should be September through November."""
        sm, sd, em, ed = SEASON_WINDOWS["FAL"]
        assert sm == 9   # September
        assert em == 11  # November

    def test_band_to_asset_mapping(self):
        """Should map all SENTINEL_BANDS to asset IDs."""
        for band in SENTINEL_BANDS:
            assert band in BAND_TO_ASSET
            assert BAND_TO_ASSET[band]  # non-empty string

    def test_band_to_asset_b08_is_nir(self):
        """B08 should map to NIR asset."""
        assert BAND_TO_ASSET["B08"] == "nir"

    def test_band_to_asset_b04_is_red(self):
        """B04 should map to red asset."""
        assert BAND_TO_ASSET["B04"] == "red"


class TestSeasonConfig:
    """Tests for SeasonConfig dataclass."""

    def test_season_config_creation(self):
        """Should create a SeasonConfig with all fields."""
        config = SeasonConfig(
            name="SUM",
            start=date(2023, 6, 1),
            end=date(2023, 8, 31)
        )
        assert config.name == "SUM"
        assert config.start == date(2023, 6, 1)
        assert config.end == date(2023, 8, 31)

    def test_season_config_is_frozen(self):
        """SeasonConfig should be immutable."""
        config = SeasonConfig("SUM", date(2023, 6, 1), date(2023, 8, 31))
        with pytest.raises(AttributeError):
            config.name = "FAL"  # type: ignore


class TestSeasonDateRange:
    """Tests for season_date_range function."""

    def test_spring_date_range(self):
        """Spring should return March 1 - May 31."""
        config = season_date_range(2023, "SPR")
        assert config.name == "SPR"
        assert config.start == date(2023, 3, 1)
        assert config.end == date(2023, 5, 31)

    def test_summer_date_range(self):
        """Summer should return June 1 - August 31."""
        config = season_date_range(2023, "SUM")
        assert config.name == "SUM"
        assert config.start == date(2023, 6, 1)
        assert config.end == date(2023, 8, 31)

    def test_fall_date_range(self):
        """Fall should return September 1 - November 30."""
        config = season_date_range(2023, "FAL")
        assert config.name == "FAL"
        assert config.start == date(2023, 9, 1)
        assert config.end == date(2023, 11, 30)

    def test_different_year(self):
        """Should work with different years."""
        config = season_date_range(2020, "SUM")
        assert config.start.year == 2020
        assert config.end.year == 2020

    def test_invalid_season_raises_error(self):
        """Should raise ValueError for invalid season."""
        with pytest.raises(ValueError, match="Unsupported season"):
            season_date_range(2023, "WIN")

    def test_invalid_season_shows_supported(self):
        """Error message should list supported seasons."""
        with pytest.raises(ValueError) as exc_info:
            season_date_range(2023, "INVALID")
        assert "FAL" in str(exc_info.value)
        assert "SPR" in str(exc_info.value)
        assert "SUM" in str(exc_info.value)


class TestStackBandsValidation:
    """Tests for stack_bands validation logic.
    
    Note: Full integration tests for stack_bands require mocking STAC items
    and stackstac. These tests focus on validation and error handling.
    """

    def test_stack_bands_empty_items_raises(self):
        """Should raise ValueError when items list is empty."""
        from wetlands_ml_atwell.sentinel2.stac_client import stack_bands
        
        with pytest.raises(ValueError, match="No Sentinel-2 items"):
            stack_bands([], (-85.5, 41.5, -85.0, 42.0))

