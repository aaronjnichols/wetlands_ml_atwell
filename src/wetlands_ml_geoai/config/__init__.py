"""Configuration management for wetlands_ml_geoai.

This module provides dataclass-based configuration objects for training and inference
workflows, with support for validation and YAML-based configuration files.
"""

from .models import (
    InferenceConfig,
    ModelConfig,
    TilingConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from .yaml_loader import (
    ConfigurationError,
    load_inference_config,
    load_training_config,
)

__all__ = [
    # Dataclasses
    "TilingConfig",
    "ModelConfig",
    "TrainingHyperparameters",
    "TrainingConfig",
    "InferenceConfig",
    # YAML loading
    "ConfigurationError",
    "load_training_config",
    "load_inference_config",
]

