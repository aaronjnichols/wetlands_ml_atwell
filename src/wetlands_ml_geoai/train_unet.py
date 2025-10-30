"""UNet-based semantic segmentation training entry point for wetlands_ml_geoai."""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Any, Dict, List, Set

import numpy as np
import rasterio

from .stacking import load_manifest


def strtobool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

from .training.unet import train_unet as run_training

DEFAULT_TILE_SIZE = 512
DEFAULT_STRIDE = 256
DEFAULT_BUFFER = 0
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 25
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_NUM_CLASSES = 2
DEFAULT_ARCHITECTURE = "unet"
DEFAULT_ENCODER_NAME = "resnet34"
DEFAULT_ENCODER_WEIGHTS = "imagenet"
DEFAULT_SEED = 42
DEFAULT_RESIZE_MODE = "resize"


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


def _resolve_manifest_paths(stack_manifest: Optional[str]) -> Sequence[Path]:
    if stack_manifest is None:
        return []

    manifest_path = Path(stack_manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stack manifest path not found: {manifest_path}")

    resolved: List[Path] = []
    _gather_manifest_paths(manifest_path, resolved, set())

    unique: List[Path] = []
    seen: Set[Path] = set()
    for candidate in resolved:
        path = candidate.resolve()
        if path not in seen:
            if not path.exists():
                raise FileNotFoundError(f"Stack manifest referenced but not found: {path}")
            seen.add(path)
            unique.append(path)

    if not unique:
        raise FileNotFoundError(
            f"No stack manifest JSON files discovered for path: {manifest_path}"
        )

    return unique


def _gather_manifest_paths(target: Path, results: List[Path], visited: Set[Path]) -> None:
    normalized = target.resolve()
    if normalized in visited:
        return
    visited.add(normalized)

    if normalized.is_dir():
        entries = sorted(normalized.iterdir(), key=lambda path: path.name.lower())
        for child in entries:
            if child.is_dir() or child.suffix.lower() == ".json":
                _gather_manifest_paths(child, results, visited)
        return

    if normalized.suffix.lower() != ".json":
        return

    data = _load_json(normalized)
    if data is None:
        return

    if _is_manifest_index(data):
        manifests = data.get("manifests", [])
        parent = normalized.parent
        for entry in manifests:
            entry_path = Path(entry)
            if not entry_path.is_absolute():
                entry_path = (parent / entry_path).resolve()
            else:
                entry_path = entry_path.resolve()
            _gather_manifest_paths(entry_path, results, visited)
        return

    if _is_stack_manifest(data):
        results.append(normalized)
        return


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_manifest_index(data: Dict[str, Any]) -> bool:
    manifests = data.get("manifests")
    if not isinstance(manifests, list):
        return False
    return all(isinstance(item, str) and item for item in manifests)


def _is_stack_manifest(data: Dict[str, Any]) -> bool:
    grid = data.get("grid")
    sources = data.get("sources")
    if not isinstance(grid, dict) or "transform" not in grid:
        return False
    if not isinstance(sources, list) or not sources:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tiles and train a wetlands UNet semantic segmentation model."
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

    args = parser.parse_args()

    if not args.train_raster and not args.stack_manifest:
        parser.error(
            "Provide --train-raster or --stack-manifest (or set TRAIN_RASTER_PATH / TRAIN_STACK_MANIFEST)."
        )
    if not args.labels:
        parser.error("--labels or TRAIN_LABELS_PATH must be supplied.")

    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    labels_path = Path(args.labels).expanduser().resolve()
    if not labels_path.exists():
        raise FileNotFoundError(f"Training labels not found: {labels_path}")

    manifest_paths = _resolve_manifest_paths(args.stack_manifest)

    if manifest_paths:
        first_manifest = load_manifest(manifest_paths[0])
        naip_source = first_manifest.naip
        if naip_source is None:
            raise ValueError("Stack manifest does not include a NAIP source.")
        base_raster = Path(naip_source.path)
        default_parent = Path(manifest_paths[0]).parent
    else:
        if args.train_raster is None:
            raise ValueError(
                "--train-raster must be provided when no stack manifest is supplied."
            )
        base_raster = Path(args.train_raster).expanduser().resolve()
        if not base_raster.exists():
            raise FileNotFoundError(f"Training raster not found: {base_raster}")
        default_parent = base_raster.parent

    tiles_dir = (
        Path(args.tiles_dir).expanduser().resolve()
        if args.tiles_dir
        else default_parent / "tiles"
    )
    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir
        else tiles_dir / "models_unet"
    )

    encoder_weights = args.encoder_weights
    if isinstance(encoder_weights, str):
        cleaned = encoder_weights.strip().strip('"').strip("'")
        if cleaned.lower() in {"", "none", "null"}:
            encoder_weights = None
        else:
            encoder_weights = cleaned

    target_size = parse_target_size(args.target_size)

    run_training(
        labels_path=labels_path,
        train_raster=base_raster,
        tiles_dir=tiles_dir,
        models_dir=models_dir,
        stack_manifest_path=manifest_paths,
        tile_size=args.tile_size,
        stride=args.stride,
        buffer_radius=args.buffer,
        num_channels_override=args.num_channels,
        num_classes=args.num_classes,
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_split=args.val_split,
        save_best_only=args.save_best_only,
        plot_curves=args.plot_curves,
        target_size=target_size,
        resize_mode=args.resize_mode,
        num_workers=args.num_workers,
        checkpoint_path=Path(args.checkpoint_path).expanduser().resolve()
        if args.checkpoint_path
        else None,
        resume_training=args.resume_training,
    )


if __name__ == "__main__":
    main()
