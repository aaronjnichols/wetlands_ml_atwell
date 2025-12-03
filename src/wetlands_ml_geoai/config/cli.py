"""CLI argument parsing and configuration building for wetlands_ml_geoai."""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

from .models import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER,
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_WEIGHTS,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_AREA,
    DEFAULT_NUM_CLASSES,
    DEFAULT_OVERLAP,
    DEFAULT_RESIZE_MODE,
    DEFAULT_SEED,
    DEFAULT_SIMPLIFY,
    DEFAULT_STRIDE,
    DEFAULT_TILE_SIZE,
    DEFAULT_VAL_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WINDOW_SIZE,
    InferenceConfig,
    ModelConfig,
    TilingConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from .yaml_loader import load_inference_config, load_training_config, ConfigurationError


def strtobool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_target_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    normalized = (
        text.replace("by", " ")
        .replace("x", " ")
        .replace("*", " ")
        .replace(",", " ")
    )
    parts = [p for p in normalized.split() if p]
    if not parts:
        return None
    if len(parts) == 1:
        size = int(float(parts[0]))
        return size, size
    height = int(float(parts[0]))
    width = int(float(parts[1]))
    return height, width


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Add standard training arguments to an ArgumentParser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file. CLI arguments override YAML values.",
    )
    parser.add_argument(
        "--train-raster",
        default=os.getenv("TRAIN_RASTER_PATH"),
        help="Path to the training raster (expects 4- or 25-band stack).",
    )
    parser.add_argument(
        "--stack-manifest",
        default=os.getenv("TRAIN_STACK_MANIFEST"),
        help="Path to a stack manifest JSON or directory containing per-AOI manifests.",
    )
    parser.add_argument(
        "--labels",
        default=os.getenv("TRAIN_LABELS_PATH"),
        help="Path to the vector training labels (GeoPackage or shapefile).",
    )
    parser.add_argument(
        "--tiles-dir",
        default=os.getenv("TRAIN_TILES_DIR"),
        help="Directory for exported image/label tiles. Defaults to <raster_parent>/tiles.",
    )
    parser.add_argument(
        "--models-dir",
        default=os.getenv("TRAIN_MODELS_DIR"),
        help="Directory to store trained model checkpoints. Defaults to <tiles-dir>/models.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=int(os.getenv("TILE_SIZE", DEFAULT_TILE_SIZE)),
        help="Tile size in pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=int(os.getenv("TILE_STRIDE", DEFAULT_STRIDE)),
        help="Stride in pixels between tiles.",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=int(os.getenv("TILE_BUFFER", DEFAULT_BUFFER)),
        help="Buffer radius for tile extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("UNET_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("UNET_NUM_EPOCHS", DEFAULT_EPOCHS)),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("UNET_LEARNING_RATE", DEFAULT_LEARNING_RATE)),
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(os.getenv("UNET_WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY)),
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=float(os.getenv("VAL_SPLIT", DEFAULT_VAL_SPLIT)),
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Override the input channel count; derived from raster if omitted.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=int(os.getenv("UNET_NUM_CLASSES", DEFAULT_NUM_CLASSES)),
        help="Number of segmentation classes (including background).",
    )
    parser.add_argument(
        "--architecture",
        default=os.getenv("UNET_ARCHITECTURE", DEFAULT_ARCHITECTURE),
        help="segmentation-models-pytorch architecture to use (e.g., unet, fpn, deeplabv3plus).",
    )
    parser.add_argument(
        "--encoder-name",
        default=os.getenv("UNET_ENCODER_NAME", DEFAULT_ENCODER_NAME),
        help="Backbone encoder (e.g., resnet34, efficientnet-b3, mit_b0).",
    )
    default_encoder_weights = os.getenv("UNET_ENCODER_WEIGHTS", DEFAULT_ENCODER_WEIGHTS)
    parser.add_argument(
        "--encoder-weights",
        default=default_encoder_weights,
        help="Encoder weights preset (e.g., imagenet) or 'none' to disable.",
    )
    parser.add_argument(
        "--no-encoder-weights",
        dest="encoder_weights",
        action="store_const",
        const=None,
        help="Disable pretrained encoder weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("UNET_SEED", DEFAULT_SEED)),
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--target-size",
        default=os.getenv("UNET_TARGET_SIZE"),
        help="Optional target HxW for resizing tiles (e.g., '512x512').",
    )
    parser.add_argument(
        "--resize-mode",
        default=os.getenv("UNET_RESIZE_MODE", DEFAULT_RESIZE_MODE),
        choices=["resize", "pad"],
        help="Resize strategy when target-size is provided.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=(
            int(os.getenv("UNET_NUM_WORKERS"))
            if os.getenv("UNET_NUM_WORKERS") is not None
            else None
        ),
        help="Number of DataLoader workers (defaults to geoai's platform-aware choice).",
    )
    default_save_best = strtobool(os.getenv("UNET_SAVE_BEST_ONLY", "true"))
    parser.set_defaults(save_best_only=default_save_best)
    parser.add_argument(
        "--save-best-only",
        dest="save_best_only",
        action="store_true",
        help="Only persist the best checkpoint (default).",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        dest="save_best_only",
        action="store_false",
        help="Persist periodic checkpoints in addition to the best model.",
    )
    parser.add_argument(
        "--plot-curves",
        dest="plot_curves",
        action="store_true",
        help="Generate loss/metric plots after training.",
    )
    parser.add_argument(
        "--no-plot-curves",
        dest="plot_curves",
        action="store_false",
        help="Skip plotting curves (default).",
    )
    parser.set_defaults(plot_curves=strtobool(os.getenv("UNET_PLOT_CURVES", "false")))
    parser.add_argument(
        "--checkpoint-path",
        default=os.getenv("UNET_CHECKPOINT_PATH"),
        help="Optional checkpoint to load before training.",
    )
    parser.add_argument(
        "--resume-training",
        dest="resume_training",
        action="store_true",
        help="Resume optimizer/scheduler state from checkpoint (requires --checkpoint-path).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Build a TrainingConfig from parsed CLI arguments and optional YAML config."""
    # Start with YAML config if provided
    if args.config is not None:
        try:
            config = load_training_config(args.config, validate=False)
        except (ConfigurationError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load config from {args.config}: {e}")
        
        # Override with CLI arguments if they differ from defaults
        if args.labels:
            config.labels_path = Path(args.labels).expanduser().resolve()
        if args.train_raster:
            config.train_raster = Path(args.train_raster).expanduser().resolve()
        if args.stack_manifest:
            config.stack_manifest = Path(args.stack_manifest).expanduser().resolve()
        if args.tiles_dir:
            config.tiles_dir = Path(args.tiles_dir).expanduser().resolve()
        if args.models_dir:
            config.models_dir = Path(args.models_dir).expanduser().resolve()
        
        # Tiling config
        config.tiling.tile_size = args.tile_size
        config.tiling.stride = args.stride
        config.tiling.buffer_radius = args.buffer
        
        # Model config
        config.model.architecture = args.architecture
        config.model.encoder_name = args.encoder_name
        if args.encoder_weights is not None:
            cleaned = str(args.encoder_weights).strip().strip('"').strip("'")
            config.model.encoder_weights = None if cleaned.lower() in {"", "none", "null"} else cleaned
        if args.num_channels is not None:
            config.model.num_channels = args.num_channels
        config.model.num_classes = args.num_classes
        
        # Hyperparameters
        config.hyperparameters.batch_size = args.batch_size
        config.hyperparameters.epochs = args.epochs
        config.hyperparameters.learning_rate = args.learning_rate
        config.hyperparameters.weight_decay = args.weight_decay
        config.hyperparameters.val_split = args.val_split
        config.hyperparameters.seed = args.seed
        
        # Other settings
        if args.target_size:
            config.target_size = parse_target_size(args.target_size)
        config.resize_mode = args.resize_mode
        config.num_workers = args.num_workers
        config.save_best_only = args.save_best_only
        config.plot_curves = args.plot_curves
        if args.checkpoint_path:
            config.checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        config.resume_training = args.resume_training
        
        # Validate the final config
        config.validate()
        return config
    
    # Build config from CLI arguments only
    labels_path = Path(args.labels).expanduser().resolve()
    
    encoder_weights = args.encoder_weights
    if isinstance(encoder_weights, str):
        cleaned = encoder_weights.strip().strip('"').strip("'")
        if cleaned.lower() in {"", "none", "null"}:
            encoder_weights = None
        else:
            encoder_weights = cleaned
    
    config = TrainingConfig(
        labels_path=labels_path,
        train_raster=Path(args.train_raster).expanduser().resolve() if args.train_raster else None,
        stack_manifest=Path(args.stack_manifest).expanduser().resolve() if args.stack_manifest else None,
        tiles_dir=Path(args.tiles_dir).expanduser().resolve() if args.tiles_dir else None,
        models_dir=Path(args.models_dir).expanduser().resolve() if args.models_dir else None,
        tiling=TilingConfig(
            tile_size=args.tile_size,
            stride=args.stride,
            buffer_radius=args.buffer,
        ),
        model=ModelConfig(
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
        ),
        hyperparameters=TrainingHyperparameters(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_split=args.val_split,
            seed=args.seed,
        ),
        target_size=parse_target_size(args.target_size),
        resize_mode=args.resize_mode,
        num_workers=args.num_workers,
        save_best_only=args.save_best_only,
        plot_curves=args.plot_curves,
        checkpoint_path=Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None,
        resume_training=args.resume_training,
    )
    
    return config


def add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add standard inference arguments to an ArgumentParser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file. CLI arguments override YAML values.",
    )
    parser.add_argument(
        "--test-raster",
        default=os.getenv("TEST_RASTER_PATH"),
        help="Path to the raster to evaluate (expects 4- or 25-band stack).",
    )
    parser.add_argument(
        "--stack-manifest",
        default=os.getenv("TEST_STACK_MANIFEST"),
        help="Path to a stack manifest JSON for streaming inference.",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH"),
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("PREDICTIONS_DIR"),
        help="Directory where prediction rasters/vectors will be written. Defaults to <raster_parent>/predictions.",
    )
    parser.add_argument(
        "--masks",
        default=os.getenv("PREDICTION_MASK_PATH"),
        help="Optional explicit path for the predicted mask GeoTIFF.",
    )
    parser.add_argument(
        "--vectors",
        default=os.getenv("PREDICTION_VECTOR_PATH"),
        help="Optional explicit path for the vectorized predictions.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.getenv("WINDOW_SIZE", DEFAULT_WINDOW_SIZE)),
        help="Sliding window size in pixels.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=int(os.getenv("WINDOW_OVERLAP", DEFAULT_OVERLAP)),
        help="Overlap in pixels between windows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("INFER_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Inference batch size (used for direct GeoTIFF inference).",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Override the input channel count; derived from raster if omitted.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=int(os.getenv("UNET_NUM_CLASSES", DEFAULT_NUM_CLASSES)),
        help="Number of segmentation classes (including background).",
    )
    parser.add_argument(
        "--architecture",
        default=os.getenv("UNET_ARCHITECTURE", DEFAULT_ARCHITECTURE),
        help="segmentation-models-pytorch architecture used during training (e.g., unet, deeplabv3plus).",
    )
    parser.add_argument(
        "--encoder-name",
        default=os.getenv("UNET_ENCODER_NAME", DEFAULT_ENCODER_NAME),
        help="Backbone encoder used during training (e.g., resnet34, efficientnet-b3).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=float(os.getenv("MIN_VECTOR_AREA", DEFAULT_MIN_AREA)),
        help="Minimum polygon area (square meters) to keep during vectorization.",
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=float(os.getenv("SIMPLIFY_TOLERANCE", DEFAULT_SIMPLIFY)),
        help="Douglas-Peucker tolerance for geometry simplification.",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=(
            float(os.getenv("UNET_PROBABILITY_THRESHOLD"))
            if os.getenv("UNET_PROBABILITY_THRESHOLD") is not None
            else None
        ),
        help="Optional positive-class probability threshold for binary predictions (default uses argmax).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )


def build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    """Build an InferenceConfig from parsed CLI arguments and optional YAML config."""
    # Start with YAML config if provided
    if args.config is not None:
        try:
            config = load_inference_config(args.config, validate=False)
        except (ConfigurationError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load config from {args.config}: {e}")
        
        # Override with CLI arguments if provided
        if args.model_path:
            config.model_path = Path(args.model_path).expanduser().resolve()
        if args.test_raster:
            config.test_raster = Path(args.test_raster).expanduser().resolve()
        if args.stack_manifest:
            config.stack_manifest = Path(args.stack_manifest).expanduser().resolve()
        if args.output_dir:
            config.output_dir = Path(args.output_dir).expanduser().resolve()
        if args.masks:
            config.masks_path = Path(args.masks).expanduser().resolve()
        if args.vectors:
            config.vectors_path = Path(args.vectors).expanduser().resolve()
        
        # Model config
        config.model.architecture = args.architecture
        config.model.encoder_name = args.encoder_name
        config.model.num_classes = args.num_classes
        if args.num_channels is not None:
            config.model.num_channels = args.num_channels
        
        # Inference parameters
        config.window_size = args.window_size
        config.overlap = args.overlap
        config.batch_size = args.batch_size
        config.min_area = args.min_area
        config.simplify_tolerance = args.simplify_tolerance
        if args.probability_threshold is not None:
            config.probability_threshold = args.probability_threshold
        
        # Validate the final config
        config.validate()
        return config
    
    # Build config from CLI arguments only
    config = InferenceConfig(
        model_path=Path(args.model_path).expanduser().resolve(),
        test_raster=Path(args.test_raster).expanduser().resolve() if args.test_raster else None,
        stack_manifest=Path(args.stack_manifest).expanduser().resolve() if args.stack_manifest else None,
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        masks_path=Path(args.masks).expanduser().resolve() if args.masks else None,
        vectors_path=Path(args.vectors).expanduser().resolve() if args.vectors else None,
        model=ModelConfig(
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_classes=args.num_classes,
            num_channels=args.num_channels,
        ),
        window_size=args.window_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        min_area=args.min_area,
        simplify_tolerance=args.simplify_tolerance,
        probability_threshold=args.probability_threshold,
    )
    
    return config

