"""Configuration dataclasses for wetlands_ml_atwell workflows.

These dataclasses provide validated, type-safe configuration for training
and inference pipelines. They can be instantiated from CLI arguments,
environment variables, or YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


# =============================================================================
# Default Values (matching existing CLI defaults)
# =============================================================================

DEFAULT_TILE_SIZE = 512
DEFAULT_STRIDE = 256
DEFAULT_BUFFER = 0

DEFAULT_ARCHITECTURE = "unet"
DEFAULT_ENCODER_NAME = "resnet34"
DEFAULT_ENCODER_WEIGHTS = "imagenet"
DEFAULT_NUM_CLASSES = 2

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 25
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42

DEFAULT_WINDOW_SIZE = 512
DEFAULT_OVERLAP = 256
DEFAULT_MIN_AREA = 1000.0
DEFAULT_SIMPLIFY = 1.0
DEFAULT_RESIZE_MODE = "resize"


# =============================================================================
# Component Configurations
# =============================================================================

@dataclass
class TilingConfig:
    """Configuration for tile generation during training.
    
    Attributes:
        tile_size: Size of each tile in pixels (default: 512).
        stride: Step size between tiles in pixels (default: 256).
        buffer_radius: Buffer around each tile in pixels (default: 0).
    """
    tile_size: int = DEFAULT_TILE_SIZE
    stride: int = DEFAULT_STRIDE
    buffer_radius: int = DEFAULT_BUFFER
    
    def validate(self) -> None:
        """Validate tiling configuration.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.stride > self.tile_size:
            raise ValueError(
                f"stride ({self.stride}) cannot exceed tile_size ({self.tile_size})"
            )
        if self.buffer_radius < 0:
            raise ValueError(f"buffer_radius must be non-negative, got {self.buffer_radius}")


@dataclass
class ModelConfig:
    """Configuration for neural network architecture.
    
    Attributes:
        architecture: Model architecture (e.g., 'unet', 'fpn', 'deeplabv3plus').
        encoder_name: Backbone encoder (e.g., 'resnet34', 'efficientnet-b0').
        encoder_weights: Pretrained weights (e.g., 'imagenet') or None.
        num_classes: Number of output classes including background.
        num_channels: Number of input channels (None = auto-detect from raster).
    """
    architecture: str = DEFAULT_ARCHITECTURE
    encoder_name: str = DEFAULT_ENCODER_NAME
    encoder_weights: Optional[str] = DEFAULT_ENCODER_WEIGHTS
    num_classes: int = DEFAULT_NUM_CLASSES
    num_channels: Optional[int] = None
    
    def validate(self) -> None:
        """Validate model configuration.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.architecture:
            raise ValueError("architecture must be specified")
        if not self.encoder_name:
            raise ValueError("encoder_name must be specified")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if self.num_channels is not None and self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {self.num_channels}")


@dataclass
class TrainingHyperparameters:
    """Hyperparameters for model training.
    
    Attributes:
        batch_size: Number of samples per batch.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization weight.
        val_split: Fraction of data for validation (0-1).
        seed: Random seed for reproducibility.
    """
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    val_split: float = DEFAULT_VAL_SPLIT
    seed: int = DEFAULT_SEED
    
    def validate(self) -> None:
        """Validate hyperparameters.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not 0 <= self.val_split < 1:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")


# =============================================================================
# Workflow Configurations
# =============================================================================

@dataclass
class TrainingConfig:
    """Complete configuration for a training run.
    
    Attributes:
        labels_path: Path to training labels (GeoPackage or shapefile).
        train_raster: Path to training raster (optional if stack_manifest provided).
        stack_manifest: Path to stack manifest JSON (optional if train_raster provided).
        tiles_dir: Directory for exported tiles (default: derived from raster path).
        models_dir: Directory for model checkpoints (default: tiles_dir/models_unet).
        tiling: Tile generation configuration.
        model: Neural network configuration.
        hyperparameters: Training hyperparameters.
        target_size: Optional resize dimensions (height, width).
        resize_mode: How to resize ('resize' or 'pad').
        num_workers: DataLoader workers (None = auto).
        save_best_only: Only save best checkpoint.
        plot_curves: Generate training curves plot.
        checkpoint_path: Path to checkpoint to resume from.
        resume_training: Resume optimizer state from checkpoint.
    """
    labels_path: Path
    train_raster: Optional[Path] = None
    stack_manifest: Optional[Path] = None
    tiles_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    
    tiling: TilingConfig = field(default_factory=TilingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    
    target_size: Optional[Tuple[int, int]] = None
    resize_mode: str = "resize"
    num_workers: Optional[int] = None
    save_best_only: bool = True
    plot_curves: bool = False
    checkpoint_path: Optional[Path] = None
    resume_training: bool = False
    
    def validate(self) -> None:
        """Validate training configuration.
        
        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files don't exist.
        """
        # Validate labels path
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        # Require either train_raster or stack_manifest
        if self.train_raster is None and self.stack_manifest is None:
            raise ValueError(
                "Either train_raster or stack_manifest must be provided"
            )
        
        # Validate paths if provided
        if self.train_raster is not None and not self.train_raster.exists():
            raise FileNotFoundError(f"Training raster not found: {self.train_raster}")
        if self.stack_manifest is not None and not self.stack_manifest.exists():
            raise FileNotFoundError(f"Stack manifest not found: {self.stack_manifest}")
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Validate resize mode
        if self.resize_mode not in ("resize", "pad"):
            raise ValueError(f"resize_mode must be 'resize' or 'pad', got '{self.resize_mode}'")
        
        # Validate target_size
        if self.target_size is not None:
            if len(self.target_size) != 2:
                raise ValueError(f"target_size must be (height, width), got {self.target_size}")
            if self.target_size[0] <= 0 or self.target_size[1] <= 0:
                raise ValueError(f"target_size dimensions must be positive, got {self.target_size}")
        
        # Validate num_workers
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        
        # Validate nested configs
        self.tiling.validate()
        self.model.validate()
        self.hyperparameters.validate()


@dataclass
class InferenceConfig:
    """Complete configuration for an inference run.
    
    Attributes:
        model_path: Path to trained model checkpoint.
        test_raster: Path to raster for inference (optional if stack_manifest provided).
        stack_manifest: Path to stack manifest JSON (optional if test_raster provided).
        output_dir: Directory for prediction outputs (default: derived from input).
        masks_path: Explicit path for mask output GeoTIFF.
        vectors_path: Explicit path for vectorized output.
        model: Neural network configuration (must match training).
        window_size: Sliding window size in pixels.
        overlap: Overlap between windows in pixels.
        batch_size: Inference batch size.
        min_area: Minimum polygon area in square meters.
        simplify_tolerance: Douglas-Peucker simplification tolerance.
        probability_threshold: Threshold for binary predictions (None = argmax).
    """
    model_path: Path
    test_raster: Optional[Path] = None
    stack_manifest: Optional[Path] = None
    output_dir: Optional[Path] = None
    masks_path: Optional[Path] = None
    vectors_path: Optional[Path] = None
    
    model: ModelConfig = field(default_factory=ModelConfig)
    
    window_size: int = DEFAULT_WINDOW_SIZE
    overlap: int = DEFAULT_OVERLAP
    batch_size: int = DEFAULT_BATCH_SIZE
    min_area: float = DEFAULT_MIN_AREA
    simplify_tolerance: float = DEFAULT_SIMPLIFY
    probability_threshold: Optional[float] = None
    
    def validate(self) -> None:
        """Validate inference configuration.
        
        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files don't exist.
        """
        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Require either test_raster or stack_manifest
        if self.test_raster is None and self.stack_manifest is None:
            raise ValueError(
                "Either test_raster or stack_manifest must be provided"
            )
        
        # Validate paths if provided
        if self.test_raster is not None and not self.test_raster.exists():
            raise FileNotFoundError(f"Test raster not found: {self.test_raster}")
        if self.stack_manifest is not None and not self.stack_manifest.exists():
            raise FileNotFoundError(f"Stack manifest not found: {self.stack_manifest}")
        
        # Validate window parameters
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")
        if self.overlap >= self.window_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than window_size ({self.window_size})"
            )
        
        # Validate other parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.min_area < 0:
            raise ValueError(f"min_area must be non-negative, got {self.min_area}")
        if self.simplify_tolerance < 0:
            raise ValueError(f"simplify_tolerance must be non-negative, got {self.simplify_tolerance}")
        if self.probability_threshold is not None:
            if not 0 <= self.probability_threshold <= 1:
                raise ValueError(
                    f"probability_threshold must be in [0, 1], got {self.probability_threshold}"
                )
        
        # Validate nested config
        self.model.validate()


__all__ = [
    "TilingConfig",
    "ModelConfig",
    "TrainingHyperparameters",
    "TrainingConfig",
    "InferenceConfig",
    # Default values for reference
    "DEFAULT_TILE_SIZE",
    "DEFAULT_STRIDE",
    "DEFAULT_BUFFER",
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_ENCODER_NAME",
    "DEFAULT_ENCODER_WEIGHTS",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_VAL_SPLIT",
    "DEFAULT_SEED",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_OVERLAP",
    "DEFAULT_MIN_AREA",
    "DEFAULT_SIMPLIFY",
    "DEFAULT_RESIZE_MODE",
]

