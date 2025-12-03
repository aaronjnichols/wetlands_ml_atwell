"""Integration tests for pipeline wiring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wetlands_ml_geoai.config import (
    TrainingConfig,
    InferenceConfig,
    TilingConfig,
    ModelConfig,
    TrainingHyperparameters,
)
from wetlands_ml_geoai.training.unet import train_unet_from_config
from wetlands_ml_geoai.inference.unet_stream import infer_from_config


@pytest.fixture
def valid_training_config(tmp_path):
    """Create a minimal valid training config."""
    raster = tmp_path / "train.tif"
    raster.touch()
    labels = tmp_path / "labels.gpkg"
    labels.touch()
    
    return TrainingConfig(
        labels_path=labels,
        train_raster=raster,
        tiling=TilingConfig(tile_size=256),
        model=ModelConfig(architecture="unet"),
        hyperparameters=TrainingHyperparameters(epochs=1),
        tiles_dir=tmp_path / "tiles",
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def valid_inference_config(tmp_path):
    """Create a minimal valid inference config."""
    raster = tmp_path / "test.tif"
    raster.touch()
    model = tmp_path / "model.pth"
    model.touch()
    
    return InferenceConfig(
        test_raster=raster,
        model_path=model,
        output_dir=tmp_path / "output",
        model=ModelConfig(architecture="unet", num_classes=2),
    )


def test_train_wiring_calls_core_function(valid_training_config):
    """Test that train_unet_from_config calls the core train_unet function."""
    
    # Patch the train_unet function inside the module where it is defined
    with patch("wetlands_ml_geoai.training.unet.train_unet") as mock_train:
        mock_train.return_value = Path("model.pth")
        
        train_unet_from_config(valid_training_config)
        
        mock_train.assert_called_once()
        
        # Verify some arguments were unpacked correctly
        kwargs = mock_train.call_args[1]
        assert kwargs["labels_path"] == valid_training_config.labels_path
        assert kwargs["tile_size"] == 256
        assert kwargs["epochs"] == 1


def test_infer_wiring_calls_geoai(valid_inference_config):
    """Test that infer_from_config calls geoai.semantic_segmentation for rasters."""
    
    with patch("wetlands_ml_geoai.inference.unet_stream.geoai") as mock_geoai:
        # Mock rasterio.open to return channel count
        with patch("rasterio.open") as mock_open:
            mock_src = MagicMock()
            mock_src.count = 4
            mock_open.return_value.__enter__.return_value = mock_src
            
            infer_from_config(valid_inference_config)
            
            mock_geoai.semantic_segmentation.assert_called_once()
            mock_geoai.raster_to_vector.assert_called_once()

