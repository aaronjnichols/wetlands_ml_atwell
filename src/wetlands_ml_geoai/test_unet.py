"""UNet-based semantic segmentation inference entry point for wetlands_ml_geoai."""
import argparse
import logging

from .config import (
    InferenceConfig,
    load_inference_config,
    ConfigurationError,
)
from .config.cli import (
    add_common_inference_args,
    build_inference_config
)
# Note: We now use the wrapper function that takes the config object
from .inference.unet_stream import infer_from_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for inference.
    
    Args:
        argv: Optional argument list. If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation inference with a trained UNet model."
    )
    add_common_inference_args(parser)

    args = parser.parse_args(argv)

    # If YAML config is provided, defer validation to config loading
    if args.config is None:
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

    try:
        config = build_inference_config(args)
        infer_from_config(config)
    except (ConfigurationError, FileNotFoundError, ValueError) as e:
        logging.error("Configuration error: %s", e)
        exit(1)
    except Exception as e:
        logging.exception("Unexpected error during inference: %s", e)
        exit(1)


if __name__ == "__main__":
    main()
