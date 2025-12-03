"""UNet-based semantic segmentation training entry point for wetlands_ml_geoai."""
import argparse
import logging

from .config import (
    TrainingConfig,
    load_training_config,
    ConfigurationError,
)
from .config.cli import (
    add_common_training_args, 
    build_training_config
)

# Note: We now use the wrapper function that takes the config object
from .training.unet import train_unet_from_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for training.
    
    Args:
        argv: Optional argument list. If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Prepare tiles and train a wetlands UNet semantic segmentation model."
    )
    add_common_training_args(parser)
    
    args = parser.parse_args(argv)

    # If YAML config is provided, defer validation to config loading
    if args.config is None:
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

    try:
        config = build_training_config(args)
        train_unet_from_config(config)
    except (ConfigurationError, FileNotFoundError, ValueError) as e:
        logging.error("Configuration error: %s", e)
        exit(1)
    except Exception as e:
        logging.exception("Unexpected error during training: %s", e)
        exit(1)


if __name__ == "__main__":
    main()
