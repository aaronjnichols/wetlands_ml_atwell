"""UNet-based semantic segmentation inference entry point for wetlands_ml_geoai."""
import argparse
import logging
import os
from pathlib import Path

import geoai
import rasterio

from .inference.common import resolve_output_paths
from .inference.unet_stream import infer_manifest
from .stacking import load_manifest

DEFAULT_WINDOW_SIZE = 512
DEFAULT_OVERLAP = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_MIN_AREA = 1000.0
DEFAULT_SIMPLIFY = 1.0
DEFAULT_ARCHITECTURE = "unet"
DEFAULT_ENCODER_NAME = "resnet34"
DEFAULT_NUM_CLASSES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation inference with a trained UNet model."
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

    args = parser.parse_args()

    if not args.test_raster and not args.stack_manifest:
        parser.error(
            "Provide --test-raster or --stack-manifest (or set TEST_RASTER_PATH / TEST_STACK_MANIFEST)."
        )
    if not args.model_path:
        parser.error("--model-path or MODEL_PATH must be supplied.")

    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    manifest_path = None
    if args.stack_manifest:
        manifest_raw = args.stack_manifest.strip()
        if manifest_raw.startswith('"') and manifest_raw.endswith('"'):
            manifest_raw = manifest_raw[1:-1]
        manifest_path = Path(manifest_raw).expanduser().resolve()
    stack_manifest = load_manifest(manifest_path) if manifest_path else None

    if stack_manifest is not None:
        source_path = manifest_path
        logging.info("Using stack manifest at %s for streaming inference", manifest_path)
    else:
        if args.test_raster is None:
            raise ValueError("--test-raster must be supplied when no stack manifest is provided.")
        test_raster = Path(args.test_raster).expanduser().resolve()
        if not test_raster.exists():
            raise FileNotFoundError(f"Test raster not found: {test_raster}")
        source_path = test_raster

    output_dir, masks_path, vectors_path = resolve_output_paths(
        source_path=source_path,
        output_dir=(
            Path(args.output_dir).expanduser().resolve() if args.output_dir else None
        ),
        mask_path=Path(args.masks).expanduser().resolve() if args.masks else None,
        vector_path=Path(args.vectors).expanduser().resolve() if args.vectors else None,
        raster_suffix="unet_predictions",
    )

    if stack_manifest is not None:
        infer_manifest(
            manifest=stack_manifest,
            model_path=model_path,
            output_path=masks_path,
            window_size=args.window_size,
            overlap=args.overlap,
            num_channels=args.num_channels,
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_classes=args.num_classes,
            probability_threshold=args.probability_threshold,
        )
    else:
        assert args.test_raster is not None  # for type-checkers
        test_raster_path = Path(args.test_raster).expanduser().resolve()
        if args.num_channels is None:
            with rasterio.open(test_raster_path) as src:
                num_channels = src.count
        else:
            num_channels = args.num_channels

        logging.info("Running semantic inference with %s input channels", num_channels)
        geoai.semantic_segmentation(
            str(test_raster_path),
            str(masks_path),
            str(model_path),
            architecture=args.architecture,
            encoder_name=args.encoder_name,
            num_channels=num_channels,
            num_classes=args.num_classes,
            window_size=args.window_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            probability_threshold=args.probability_threshold,
        )

    logging.info("Raster predictions written to %s", masks_path)

    geoai.raster_to_vector(
        str(masks_path),
        str(vectors_path),
        min_area=args.min_area,
        simplify_tolerance=args.simplify_tolerance,
    )

    logging.info("Vector predictions written to %s", vectors_path)


if __name__ == "__main__":
    main()




