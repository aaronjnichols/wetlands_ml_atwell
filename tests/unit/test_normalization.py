"""Unit tests for normalization module."""

import numpy as np
import pytest

from wetlands_ml_geoai.normalization import (
    to_geoai_format,
    prepare_for_model,
    GEOAI_EXPECTED_INPUT_DTYPE,
    GEOAI_EXPECTED_INPUT_RANGE,
    MODEL_INPUT_RANGE,
)


class TestToGeoaiFormat:
    """Tests for to_geoai_format() function."""
    
    def test_converts_to_uint8(self):
        """Output should be uint8 dtype."""
        data = np.array([[[0.5]]], dtype=np.float32)
        result = to_geoai_format(data)
        assert result.dtype == np.uint8
    
    def test_scales_to_255_range(self):
        """Values should be scaled from [0-1] to [0-255]."""
        data = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result = to_geoai_format(data)
        np.testing.assert_array_equal(result, [[[0, 127, 255]]])
    
    def test_clips_values_above_one(self):
        """Values > 1.0 should be clipped to 255."""
        data = np.array([[[1.5]]], dtype=np.float32)
        result = to_geoai_format(data, validate=False)
        assert result[0, 0, 0] == 255
    
    def test_clips_negative_values(self):
        """Negative values should be clipped to 0."""
        data = np.array([[[-0.5]]], dtype=np.float32)
        result = to_geoai_format(data, validate=False)
        assert result[0, 0, 0] == 0
    
    def test_warns_on_out_of_range_values(self, caplog):
        """Should log warning when values are outside [0-1]."""
        import logging
        caplog.set_level(logging.WARNING)
        data = np.array([[[1.5]]], dtype=np.float32)
        to_geoai_format(data, validate=True)
        assert "outside [0, 1] range" in caplog.text
    
    def test_preserves_shape(self):
        """Output shape should match input shape."""
        data = np.random.rand(4, 64, 64).astype(np.float32)
        result = to_geoai_format(data)
        assert result.shape == data.shape
    
    def test_matches_geoai_expected_dtype(self):
        """Output dtype should match GEOAI_EXPECTED_INPUT_DTYPE."""
        data = np.array([[[0.5]]], dtype=np.float32)
        result = to_geoai_format(data)
        assert result.dtype == GEOAI_EXPECTED_INPUT_DTYPE


class TestPrepareForModel:
    """Tests for prepare_for_model() function."""
    
    def test_uint8_input_divided_by_255(self):
        """uint8 input should be divided by 255."""
        data = np.array([[[255, 127, 0]]], dtype=np.uint8)
        result = prepare_for_model(data)
        expected = np.array([[[1.0, 127/255, 0.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_float32_input_divided_by_255(self):
        """float32 input (legacy) should also be divided by 255."""
        data = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)
        result = prepare_for_model(data)
        expected = np.array([[[1.0/255, 0.5/255, 0.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_output_is_float32(self):
        """Output should always be float32."""
        uint8_data = np.array([[[128]]], dtype=np.uint8)
        float32_data = np.array([[[0.5]]], dtype=np.float32)
        
        assert prepare_for_model(uint8_data).dtype == np.float32
        assert prepare_for_model(float32_data).dtype == np.float32
    
    def test_explicit_dtype_override(self):
        """source_dtype parameter should override inferred dtype."""
        # Data is uint8 but we say it's float32
        data = np.array([[[128]]], dtype=np.uint8)
        result = prepare_for_model(data, source_dtype=np.float32)
        # Should still divide by 255 (float32 legacy behavior)
        assert result[0, 0, 0] == pytest.approx(128 / 255)


class TestRoundTrip:
    """Tests for round-trip normalization consistency."""
    
    def test_float_to_uint8_to_model_gives_correct_range(self):
        """Full pipeline: [0-1] → uint8 → /255 should give [0-1]."""
        # Start with normalized float32 [0-1] data
        original = np.array([[[0.0, 0.25, 0.5, 0.75, 1.0]]], dtype=np.float32)
        
        # Convert to geoai format (uint8)
        uint8_tiles = to_geoai_format(original)
        
        # Simulate what geoai does during training (/255)
        model_input = uint8_tiles.astype(np.float32) / 255.0
        
        # Should be back in [0-1] range, close to original
        # Note: some precision loss due to uint8 quantization
        np.testing.assert_array_almost_equal(model_input, original, decimal=2)
    
    def test_model_input_range_is_zero_to_one(self):
        """Model input should be in [0, 1] range."""
        original = np.random.rand(4, 32, 32).astype(np.float32)
        uint8_tiles = to_geoai_format(original)
        model_input = uint8_tiles.astype(np.float32) / 255.0
        
        assert model_input.min() >= MODEL_INPUT_RANGE[0]
        assert model_input.max() <= MODEL_INPUT_RANGE[1]


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy float32 tiles."""
    
    def test_legacy_normalization_produces_small_values(self):
        """Legacy float32 [0-1] tiles divided by 255 give [0-0.004]."""
        # Simulate old float32 tiles
        legacy_tile = np.array([[[1.0]]], dtype=np.float32)
        
        # Apply legacy normalization (/255)
        result = legacy_tile / 255.0
        
        # Max value should be ~0.004 (1/255)
        assert result.max() == pytest.approx(1/255, rel=1e-5)
    
    def test_new_uint8_tiles_give_correct_range(self):
        """New uint8 [0-255] tiles divided by 255 give [0-1]."""
        # New uint8 tiles
        new_tile = np.array([[[255]]], dtype=np.uint8)
        
        # geoai's /255
        result = new_tile.astype(np.float32) / 255.0
        
        # Should be 1.0
        assert result.max() == pytest.approx(1.0)
    
    def test_both_formats_work_with_prepare_for_model(self):
        """prepare_for_model handles both legacy and new tile formats."""
        legacy_tile = np.array([[[0.5]]], dtype=np.float32)
        new_tile = np.array([[[128]]], dtype=np.uint8)
        
        # Both should produce valid output
        legacy_result = prepare_for_model(legacy_tile)
        new_result = prepare_for_model(new_tile)
        
        assert legacy_result.dtype == np.float32
        assert new_result.dtype == np.float32
        
        # Both in valid range (though different values)
        assert 0.0 <= legacy_result.min() <= legacy_result.max() <= 1.0
        assert 0.0 <= new_result.min() <= new_result.max() <= 1.0


class TestConstants:
    """Tests for normalization constants."""
    
    def test_geoai_expected_dtype(self):
        """GEOAI_EXPECTED_INPUT_DTYPE should be uint8."""
        assert GEOAI_EXPECTED_INPUT_DTYPE == np.uint8
    
    def test_geoai_expected_range(self):
        """GEOAI_EXPECTED_INPUT_RANGE should be (0, 255)."""
        assert GEOAI_EXPECTED_INPUT_RANGE == (0, 255)
    
    def test_model_input_range(self):
        """MODEL_INPUT_RANGE should be (0.0, 1.0)."""
        assert MODEL_INPUT_RANGE == (0.0, 1.0)

