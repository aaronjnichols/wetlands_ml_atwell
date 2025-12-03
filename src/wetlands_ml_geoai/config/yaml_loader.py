"""YAML configuration file loading for wetlands_ml_geoai.

This module provides functions to load TrainingConfig and InferenceConfig
from YAML files, with validation and sensible error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .models import (
    InferenceConfig,
    ModelConfig,
    TilingConfig,
    TrainingConfig,
    TrainingHyperparameters,
)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def _resolve_path(base_dir: Path, path_str: Optional[str]) -> Optional[Path]:
    """Resolve a path string relative to the config file's directory.
    
    If the path is absolute, it's returned as-is.
    If the path is relative, it's resolved relative to base_dir.
    
    Args:
        base_dir: Directory containing the config file.
        path_str: Path string from config, or None.
        
    Returns:
        Resolved Path, or None if path_str is None.
    """
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def _parse_tiling_config(data: Dict[str, Any]) -> TilingConfig:
    """Parse tiling configuration from a dict."""
    tiling_data = data.get("tiling", {})
    return TilingConfig(
        tile_size=tiling_data.get("tile_size", TilingConfig.tile_size),
        stride=tiling_data.get("stride", TilingConfig.stride),
        buffer_radius=tiling_data.get("buffer_radius", TilingConfig.buffer_radius),
    )


def _parse_model_config(data: Dict[str, Any]) -> ModelConfig:
    """Parse model configuration from a dict."""
    model_data = data.get("model", {})
    return ModelConfig(
        architecture=model_data.get("architecture", ModelConfig.architecture),
        encoder_name=model_data.get("encoder_name", ModelConfig.encoder_name),
        encoder_weights=model_data.get("encoder_weights", ModelConfig.encoder_weights),
        num_classes=model_data.get("num_classes", ModelConfig.num_classes),
        num_channels=model_data.get("num_channels", ModelConfig.num_channels),
    )


def _parse_hyperparameters(data: Dict[str, Any]) -> TrainingHyperparameters:
    """Parse training hyperparameters from a dict."""
    hyper_data = data.get("hyperparameters", {})
    return TrainingHyperparameters(
        batch_size=hyper_data.get("batch_size", TrainingHyperparameters.batch_size),
        epochs=hyper_data.get("epochs", TrainingHyperparameters.epochs),
        learning_rate=hyper_data.get("learning_rate", TrainingHyperparameters.learning_rate),
        weight_decay=hyper_data.get("weight_decay", TrainingHyperparameters.weight_decay),
        val_split=hyper_data.get("val_split", TrainingHyperparameters.val_split),
        seed=hyper_data.get("seed", TrainingHyperparameters.seed),
    )


def load_training_config(
    config_path: Union[str, Path],
    validate: bool = True,
) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        validate: Whether to validate the configuration (default: True).
        
    Returns:
        A TrainingConfig instance.
        
    Raises:
        ConfigurationError: If the file cannot be read or parsed.
        FileNotFoundError: If config_path doesn't exist.
        ValueError: If validate=True and the configuration is invalid.
        
    Example YAML structure:
        ```yaml
        # Required paths
        labels_path: ./labels.gpkg
        train_raster: ./composite.tif  # or stack_manifest: ./stack.json
        
        # Optional paths
        tiles_dir: ./tiles
        models_dir: ./models
        
        # Tiling configuration
        tiling:
          tile_size: 512
          stride: 256
          buffer_radius: 0
        
        # Model configuration
        model:
          architecture: unet
          encoder_name: resnet34
          encoder_weights: imagenet
          num_classes: 2
        
        # Hyperparameters
        hyperparameters:
          batch_size: 4
          epochs: 25
          learning_rate: 0.001
          val_split: 0.2
        ```
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    base_dir = config_path.parent
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}")
    
    if data is None:
        raise ConfigurationError("Configuration file is empty")
    
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must be a YAML mapping (dict)")
    
    # Required field
    if "labels_path" not in data:
        raise ConfigurationError("Missing required field: labels_path")
    
    # Parse target_size tuple if present
    target_size = None
    if "target_size" in data:
        ts = data["target_size"]
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            target_size = (int(ts[0]), int(ts[1]))
        else:
            raise ConfigurationError(
                f"target_size must be a list of [height, width], got {ts}"
            )
    
    config = TrainingConfig(
        labels_path=_resolve_path(base_dir, data["labels_path"]),
        train_raster=_resolve_path(base_dir, data.get("train_raster")),
        stack_manifest=_resolve_path(base_dir, data.get("stack_manifest")),
        tiles_dir=_resolve_path(base_dir, data.get("tiles_dir")),
        models_dir=_resolve_path(base_dir, data.get("models_dir")),
        tiling=_parse_tiling_config(data),
        model=_parse_model_config(data),
        hyperparameters=_parse_hyperparameters(data),
        target_size=target_size,
        resize_mode=data.get("resize_mode", "resize"),
        num_workers=data.get("num_workers"),
        save_best_only=data.get("save_best_only", True),
        plot_curves=data.get("plot_curves", False),
        checkpoint_path=_resolve_path(base_dir, data.get("checkpoint_path")),
        resume_training=data.get("resume_training", False),
    )
    
    if validate:
        config.validate()
    
    return config


def load_inference_config(
    config_path: Union[str, Path],
    validate: bool = True,
) -> InferenceConfig:
    """Load an InferenceConfig from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        validate: Whether to validate the configuration (default: True).
        
    Returns:
        An InferenceConfig instance.
        
    Raises:
        ConfigurationError: If the file cannot be read or parsed.
        FileNotFoundError: If config_path doesn't exist.
        ValueError: If validate=True and the configuration is invalid.
        
    Example YAML structure:
        ```yaml
        # Required path
        model_path: ./models/best_model.pth
        
        # Input (one required)
        test_raster: ./test_area.tif  # or stack_manifest: ./stack.json
        
        # Optional output paths
        output_dir: ./predictions
        masks_path: ./predictions/mask.tif
        vectors_path: ./predictions/vectors.gpkg
        
        # Model configuration (must match training)
        model:
          architecture: unet
          encoder_name: resnet34
          num_classes: 2
        
        # Inference parameters
        window_size: 512
        overlap: 256
        batch_size: 4
        min_area: 1000.0
        simplify_tolerance: 1.0
        probability_threshold: 0.5
        ```
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    base_dir = config_path.parent
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}")
    
    if data is None:
        raise ConfigurationError("Configuration file is empty")
    
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must be a YAML mapping (dict)")
    
    # Required field
    if "model_path" not in data:
        raise ConfigurationError("Missing required field: model_path")
    
    config = InferenceConfig(
        model_path=_resolve_path(base_dir, data["model_path"]),
        test_raster=_resolve_path(base_dir, data.get("test_raster")),
        stack_manifest=_resolve_path(base_dir, data.get("stack_manifest")),
        output_dir=_resolve_path(base_dir, data.get("output_dir")),
        masks_path=_resolve_path(base_dir, data.get("masks_path")),
        vectors_path=_resolve_path(base_dir, data.get("vectors_path")),
        model=_parse_model_config(data),
        window_size=data.get("window_size", InferenceConfig.window_size),
        overlap=data.get("overlap", InferenceConfig.overlap),
        batch_size=data.get("batch_size", InferenceConfig.batch_size),
        min_area=data.get("min_area", InferenceConfig.min_area),
        simplify_tolerance=data.get("simplify_tolerance", InferenceConfig.simplify_tolerance),
        probability_threshold=data.get("probability_threshold"),
    )
    
    if validate:
        config.validate()
    
    return config


__all__ = [
    "ConfigurationError",
    "load_training_config",
    "load_inference_config",
]

