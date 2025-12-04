"""Inference modules for wetlands ML segmentation."""

from wetlands_ml_atwell.inference.unet_stream import (
    infer_raster,
    infer_manifest,
    infer_from_config,
)
from wetlands_ml_atwell.inference.common import resolve_output_paths

__all__ = [
    "infer_raster",
    "infer_manifest",
    "infer_from_config",
    "resolve_output_paths",
]
