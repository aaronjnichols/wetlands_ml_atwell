"""Unit tests for configuration dataclasses and YAML loading."""

from pathlib import Path
import pytest

from wetlands_ml_atwell.config import (
    TilingConfig,
    ModelConfig,
    TrainingHyperparameters,
    TrainingConfig,
    InferenceConfig,
    ConfigurationError,
    load_training_config,
    load_inference_config,
)


class TestTilingConfig:
    """Tests for TilingConfig dataclass."""
    
    def test_default_values(self):
        config = TilingConfig()
        assert config.tile_size == 512
        assert config.stride == 256
        assert config.buffer_radius == 0
    
    def test_custom_values(self):
        config = TilingConfig(tile_size=256, stride=128, buffer_radius=16)
        assert config.tile_size == 256
        assert config.stride == 128
        assert config.buffer_radius == 16
    
    def test_validate_valid_config(self):
        config = TilingConfig(tile_size=512, stride=256, buffer_radius=0)
        config.validate()  # Should not raise
    
    def test_validate_tile_size_must_be_positive(self):
        config = TilingConfig(tile_size=0)
        with pytest.raises(ValueError, match="tile_size must be positive"):
            config.validate()
    
    def test_validate_stride_must_be_positive(self):
        config = TilingConfig(stride=0)
        with pytest.raises(ValueError, match="stride must be positive"):
            config.validate()
    
    def test_validate_stride_cannot_exceed_tile_size(self):
        config = TilingConfig(tile_size=256, stride=512)
        with pytest.raises(ValueError, match="stride .* cannot exceed tile_size"):
            config.validate()
    
    def test_validate_buffer_must_be_non_negative(self):
        config = TilingConfig(buffer_radius=-1)
        with pytest.raises(ValueError, match="buffer_radius must be non-negative"):
            config.validate()


class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_default_values(self):
        config = ModelConfig()
        assert config.architecture == "unet"
        assert config.encoder_name == "resnet34"
        assert config.encoder_weights == "imagenet"
        assert config.num_classes == 2
        assert config.num_channels is None
    
    def test_validate_valid_config(self):
        config = ModelConfig()
        config.validate()  # Should not raise
    
    def test_validate_architecture_required(self):
        config = ModelConfig(architecture="")
        with pytest.raises(ValueError, match="architecture must be specified"):
            config.validate()
    
    def test_validate_encoder_name_required(self):
        config = ModelConfig(encoder_name="")
        with pytest.raises(ValueError, match="encoder_name must be specified"):
            config.validate()
    
    def test_validate_num_classes_minimum(self):
        config = ModelConfig(num_classes=1)
        with pytest.raises(ValueError, match="num_classes must be at least 2"):
            config.validate()
    
    def test_validate_num_channels_positive(self):
        config = ModelConfig(num_channels=0)
        with pytest.raises(ValueError, match="num_channels must be positive"):
            config.validate()


class TestTrainingHyperparameters:
    """Tests for TrainingHyperparameters dataclass."""
    
    def test_default_values(self):
        config = TrainingHyperparameters()
        assert config.batch_size == 4
        assert config.epochs == 25
        assert config.learning_rate == 0.001
        assert config.weight_decay == 1e-4
        assert config.val_split == 0.2
        assert config.seed == 42
    
    def test_validate_valid_config(self):
        config = TrainingHyperparameters()
        config.validate()  # Should not raise
    
    def test_validate_batch_size_positive(self):
        config = TrainingHyperparameters(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
    
    def test_validate_epochs_positive(self):
        config = TrainingHyperparameters(epochs=0)
        with pytest.raises(ValueError, match="epochs must be positive"):
            config.validate()
    
    def test_validate_learning_rate_positive(self):
        config = TrainingHyperparameters(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()
    
    def test_validate_weight_decay_non_negative(self):
        config = TrainingHyperparameters(weight_decay=-1)
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            config.validate()
    
    def test_validate_val_split_range(self):
        config = TrainingHyperparameters(val_split=1.0)
        with pytest.raises(ValueError, match="val_split must be in"):
            config.validate()


class TestYamlTrainingConfig:
    """Tests for YAML TrainingConfig loading."""
    
    def test_load_valid_training_config(self, tmp_path: Path, minimal_raster_path: Path, mock_labels_gpkg: Path):
        """Test loading a valid training config YAML."""
        config_yaml = tmp_path / "train_config.yaml"
        config_yaml.write_text(f"""
labels_path: {mock_labels_gpkg}
train_raster: {minimal_raster_path}

tiling:
  tile_size: 256
  stride: 128

model:
  architecture: unet
  encoder_name: efficientnet-b0
  num_classes: 3

hyperparameters:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0005
""")
        
        config = load_training_config(config_yaml)
        
        assert config.labels_path == mock_labels_gpkg
        assert config.train_raster == minimal_raster_path
        assert config.tiling.tile_size == 256
        assert config.tiling.stride == 128
        assert config.model.architecture == "unet"
        assert config.model.encoder_name == "efficientnet-b0"
        assert config.model.num_classes == 3
        assert config.hyperparameters.batch_size == 8
        assert config.hyperparameters.epochs == 50
        assert config.hyperparameters.learning_rate == 0.0005
    
    def test_load_training_config_missing_labels_path(self, tmp_path: Path):
        """Test that missing labels_path raises ConfigurationError."""
        config_yaml = tmp_path / "train_config.yaml"
        config_yaml.write_text("""
train_raster: some_raster.tif
""")
        
        with pytest.raises(ConfigurationError, match="Missing required field: labels_path"):
            load_training_config(config_yaml)
    
    def test_load_training_config_file_not_found(self, tmp_path: Path):
        """Test that non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_training_config(tmp_path / "nonexistent.yaml")
    
    def test_load_training_config_invalid_yaml(self, tmp_path: Path):
        """Test that invalid YAML raises ConfigurationError."""
        config_yaml = tmp_path / "train_config.yaml"
        config_yaml.write_text("""
labels_path: test.gpkg
  bad_indentation: true
""")
        
        with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
            load_training_config(config_yaml)
    
    def test_load_training_config_relative_paths(self, tmp_path: Path, minimal_raster_path: Path, mock_labels_gpkg: Path):
        """Test that relative paths are resolved relative to the config file."""
        # Create config in a subdirectory
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        # Copy files to make them siblings of the config
        raster_copy = config_dir / "raster.tif"
        labels_copy = config_dir / "labels.gpkg"
        
        # Just copy file contents (this is simpler than using shutil)
        import shutil
        shutil.copy(minimal_raster_path, raster_copy)
        shutil.copy(mock_labels_gpkg, labels_copy)
        
        config_yaml = config_dir / "train_config.yaml"
        config_yaml.write_text("""
labels_path: labels.gpkg
train_raster: raster.tif
""")
        
        config = load_training_config(config_yaml)
        
        assert config.labels_path == labels_copy
        assert config.train_raster == raster_copy


class TestYamlInferenceConfig:
    """Tests for YAML InferenceConfig loading."""
    
    def test_load_valid_inference_config(self, tmp_path: Path, minimal_raster_path: Path):
        """Test loading a valid inference config YAML."""
        # Create a mock model file
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"mock model data")
        
        config_yaml = tmp_path / "infer_config.yaml"
        config_yaml.write_text(f"""
model_path: {model_path}
test_raster: {minimal_raster_path}

model:
  architecture: deeplabv3plus
  encoder_name: resnet50
  num_classes: 2

window_size: 256
overlap: 64
batch_size: 16
min_area: 500.0
simplify_tolerance: 0.5
probability_threshold: 0.6
""")
        
        config = load_inference_config(config_yaml)
        
        assert config.model_path == model_path
        assert config.test_raster == minimal_raster_path
        assert config.model.architecture == "deeplabv3plus"
        assert config.model.encoder_name == "resnet50"
        assert config.window_size == 256
        assert config.overlap == 64
        assert config.batch_size == 16
        assert config.min_area == 500.0
        assert config.simplify_tolerance == 0.5
        assert config.probability_threshold == 0.6
    
    def test_load_inference_config_missing_model_path(self, tmp_path: Path):
        """Test that missing model_path raises ConfigurationError."""
        config_yaml = tmp_path / "infer_config.yaml"
        config_yaml.write_text("""
test_raster: some_raster.tif
""")
        
        with pytest.raises(ConfigurationError, match="Missing required field: model_path"):
            load_inference_config(config_yaml)


class TestConfigurationDefaults:
    """Tests for configuration default values."""
    
    def test_training_config_uses_nested_defaults(self, tmp_path: Path, minimal_raster_path: Path, mock_labels_gpkg: Path):
        """Test that TrainingConfig uses defaults from nested configs when not specified."""
        config_yaml = tmp_path / "train_config.yaml"
        config_yaml.write_text(f"""
labels_path: {mock_labels_gpkg}
train_raster: {minimal_raster_path}
""")
        
        config = load_training_config(config_yaml)
        
        # Should have defaults from TilingConfig
        assert config.tiling.tile_size == 512
        assert config.tiling.stride == 256
        
        # Should have defaults from ModelConfig
        assert config.model.architecture == "unet"
        assert config.model.encoder_name == "resnet34"
        
        # Should have defaults from TrainingHyperparameters
        assert config.hyperparameters.batch_size == 4
        assert config.hyperparameters.epochs == 25

