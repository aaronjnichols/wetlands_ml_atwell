"""UNet sliding-window inference for wetlands_ml_geoai."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import geoai
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from geoai.train import get_smp_model
from geoai.utils import get_device

from ..stacking import (
    FLOAT_NODATA,
    RasterStack,
    StackManifest,
    load_manifest,
    normalize_stack_array,
)
from ..config import InferenceConfig
from .common import resolve_output_paths


def _compute_offsets(size: int, window: int, overlap: int) -> List[int]:
    if size <= window:
        return [0]
    step = max(window - overlap, 1)
    offsets = list(range(0, size - window + 1, step))
    last = size - window
    if offsets[-1] != last:
        offsets.append(last)
    return sorted(set(offsets))


def _load_model(
    model_path: Path,
    architecture: str,
    encoder_name: str,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = get_smp_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _prepare_window(
    data: np.ndarray,
    desired_channels: int,
    nodata_value: Optional[float],
    legacy_normalization: bool = True,
) -> np.ndarray:
    """Prepare a data window for model inference.
    
    This function normalizes input data to match what the model learned during training.
    
    Normalization Background
    ------------------------
    geoai's training always applies /255 to tile data, assuming uint8 [0-255] input.
    
    With OLD float32 [0-1] tiles:
        - geoai /255 → model trained on [0-0.004] range
        - Inference must apply /255 to match (legacy_normalization=True)
    
    With NEW uint8 [0-255] tiles:
        - geoai /255 → model trained on [0-1] range (correct!)
        - Inference should NOT apply /255 (legacy_normalization=False)
        
    See normalization.py for the full data flow documentation.
    
    Args:
        data: Input array from RasterStack (float32, scaled to [0-1]).
        desired_channels: Number of channels expected by the model.
        nodata_value: Value representing nodata pixels.
        legacy_normalization: If True, apply /255 for models trained on 
            old float32 tiles. If False, skip /255 for models trained on
            new uint8 tiles.
            
    Returns:
        float32 array normalized for model input.
    """
    array = data.astype(np.float32, copy=False)

    channel_count = array.shape[0]
    if channel_count > desired_channels:
        array = array[:desired_channels]
    elif channel_count < desired_channels:
        padded = np.zeros((desired_channels, array.shape[1], array.shape[2]), dtype=array.dtype)
        padded[:channel_count] = array
        array = padded

    # Clean nodata and clip to [0, 1]
    array = normalize_stack_array(array, nodata_value)
    
    if legacy_normalization:
        # BACKWARD COMPATIBILITY with models trained on float32 [0-1] tiles:
        # geoai applied /255 to those tiles during training, so models learned
        # on [0-0.004] range. We must apply /255 here to match.
        #
        # For models trained on NEW uint8 [0-255] tiles, set legacy_normalization=False
        # since those models expect [0-1] input (geoai's /255 on uint8 gives [0-1]).
        array = array / 255.0
        
    return array


def _predict_probabilities(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    output = model(tensor)
    if isinstance(output, (list, tuple)):
        output = output[0]
    if isinstance(output, torch.Tensor) and output.dim() == 4:
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.dim() != 3:
        raise TypeError("Unexpected model output type for semantic segmentation")

    return F.softmax(output, dim=0).cpu().numpy()


def _finalize_predictions(
    prob_accumulator: np.ndarray,
    count_accumulator: np.ndarray,
    probability_threshold: Optional[float],
    num_classes: int,
) -> np.ndarray:
    denom = np.maximum(count_accumulator, 1e-6)
    averaged = prob_accumulator / denom[None, :, :]

    if probability_threshold is not None and num_classes >= 2:
        mask = (averaged[1] >= probability_threshold).astype(np.uint8)
    else:
        mask = np.argmax(averaged, axis=0).astype(np.uint8)

    mask[count_accumulator == 0] = 0
    return mask


def _save_prediction(
    output_path: Path,
    prediction: np.ndarray,
    transform: rasterio.Affine,
    crs: Optional[rasterio.crs.CRS],
) -> None:
    height, width = prediction.shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if crs is not None:
        profile["crs"] = crs

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction, 1)


def _stream_inference(
    *,
    read_window: Callable[[Window], np.ndarray],
    height: int,
    width: int,
    transform,
    crs,
    nodata_value: Optional[float],
    channel_count: int,
    window_size: int,
    overlap: int,
    model: torch.nn.Module,
    num_classes: int,
    probability_threshold: Optional[float],
    device: torch.device,
    output_path: Path,
    context_label: str,
    legacy_normalization: bool = True,
) -> float:
    window_h = min(window_size, height)
    window_w = min(window_size, width)

    row_offsets = _compute_offsets(height, window_h, overlap)
    col_offsets = _compute_offsets(width, window_w, overlap)

    total_windows = len(row_offsets) * len(col_offsets)
    logging.info(
        "%s inference across %s windows (%sx%s, overlap=%s)",
        context_label,
        total_windows,
        window_h,
        window_w,
        overlap,
    )

    prob_accumulator = np.zeros((num_classes, height, width), dtype=np.float32)
    count_accumulator = np.zeros((height, width), dtype=np.float32)

    start = time.perf_counter()

    with torch.no_grad():
        for row_off in row_offsets:
            for col_off in col_offsets:
                win = Window(col_off, row_off, window_w, window_h)
                data = read_window(win)
                prepared = _prepare_window(data, channel_count, nodata_value, legacy_normalization)
                tensor = torch.from_numpy(prepared).to(device).unsqueeze(0)

                probabilities = _predict_probabilities(model, tensor)

                row_end = min(row_off + window_h, height)
                col_end = min(col_off + window_w, width)
                h = row_end - row_off
                w = col_end - col_off

                prob_accumulator[:, row_off:row_end, col_off:col_end] += probabilities[:, :h, :w]
                count_accumulator[row_off:row_end, col_off:col_end] += 1.0

    predicted = _finalize_predictions(
        prob_accumulator,
        count_accumulator,
        probability_threshold,
        num_classes,
    )
    _save_prediction(output_path, predicted, transform, crs)

    return time.perf_counter() - start


def infer_manifest(
    manifest: StackManifest | Path,
    model_path: Path,
    output_path: Path,
    window_size: int,
    overlap: int,
    num_channels: Optional[int],
    architecture: str,
    encoder_name: str,
    num_classes: int,
    probability_threshold: Optional[float],
    legacy_normalization: bool = True,
) -> None:
    """Run sliding-window inference on a manifest-defined raster stack.
    
    Args:
        manifest: Stack manifest defining the input data sources.
        model_path: Path to trained model checkpoint.
        output_path: Path to write prediction GeoTIFF.
        window_size: Size of sliding window in pixels.
        overlap: Overlap between windows in pixels.
        num_channels: Number of input channels (None to auto-detect).
        architecture: Model architecture (e.g., 'unet').
        encoder_name: Encoder backbone (e.g., 'resnet34').
        num_classes: Number of output classes.
        probability_threshold: Threshold for binary prediction (None for argmax).
        legacy_normalization: If True (default), apply /255 normalization for
            models trained on old float32 [0-1] tiles. Set to False for models
            trained on new uint8 [0-255] tiles. See normalization.py for details.
    """
    manifest_obj = manifest if isinstance(manifest, StackManifest) else load_manifest(manifest)
    device = get_device()

    channel_count = num_channels
    if channel_count is None:
        with RasterStack(manifest_obj) as stack:
            channel_count = stack.band_count

    model = _load_model(
        model_path=model_path,
        architecture=architecture,
        encoder_name=encoder_name,
        in_channels=channel_count,
        num_classes=num_classes,
        device=device,
    )

    with RasterStack(manifest_obj) as stack:
        elapsed = _stream_inference(
            read_window=stack.read_window,
            height=stack.height,
            width=stack.width,
            transform=stack.transform,
            crs=stack.crs,
            nodata_value=FLOAT_NODATA,
            channel_count=channel_count,
            window_size=window_size,
            overlap=overlap,
            model=model,
            num_classes=num_classes,
            probability_threshold=probability_threshold,
            device=device,
            output_path=output_path,
            context_label="Streaming UNet",
            legacy_normalization=legacy_normalization,
        )

    logging.info("Streaming semantic inference finished in %.2f seconds", elapsed)


def infer_raster(
    raster_path: Path,
    model_path: Path,
    output_path: Path,
    window_size: int,
    overlap: int,
    num_channels: Optional[int],
    architecture: str,
    encoder_name: str,
    num_classes: int,
    probability_threshold: Optional[float],
) -> None:
    device = get_device()

    with rasterio.open(raster_path) as src:
        channel_count = num_channels if num_channels is not None else src.count
        logging.info("Running semantic inference with %s input channels", channel_count)
        if probability_threshold is not None:
            logging.info("Applying probability threshold %.3f", probability_threshold)

        model = _load_model(
            model_path=model_path,
            architecture=architecture,
            encoder_name=encoder_name,
            in_channels=channel_count,
            num_classes=num_classes,
            device=device,
        )

        elapsed = _stream_inference(
            read_window=lambda win: src.read(window=win),
            height=src.height,
            width=src.width,
            transform=src.transform,
            crs=src.crs,
            nodata_value=src.nodata,
            channel_count=channel_count,
            window_size=window_size,
            overlap=overlap,
            model=model,
            num_classes=num_classes,
            probability_threshold=probability_threshold,
            device=device,
            output_path=output_path,
            context_label="Raster UNet",
        )

    logging.info("Raster semantic inference finished in %.2f seconds", elapsed)


def infer_from_config(config: InferenceConfig) -> Tuple[Path, Path]:
    """Run inference workflow from an InferenceConfig object.
    
    This function handles manifest/raster detection, output path resolution,
    inference execution, and vectorization.
    
    Args:
        config: Validated InferenceConfig instance.
        
    Returns:
        Tuple of (masks_path, vectors_path) where predictions were written.
        
    Raises:
        FileNotFoundError: If model or input files are not found.
        ValueError: If configuration is invalid.
    """
    # Validate config inputs up front to surface clear errors.
    config.validate()

    # Validate model path
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {config.model_path}")
    
    # Determine source path and load manifest if applicable
    stack_manifest: Optional[StackManifest] = None
    
    if config.stack_manifest is not None:
        if not config.stack_manifest.exists():
            raise FileNotFoundError(f"Stack manifest not found: {config.stack_manifest}")
        stack_manifest = load_manifest(config.stack_manifest)
        source_path = config.stack_manifest
        logging.info("Using stack manifest at %s for streaming inference", config.stack_manifest)
    else:
        if config.test_raster is None:
            raise ValueError("Either test_raster or stack_manifest must be provided")
        if not config.test_raster.exists():
            raise FileNotFoundError(f"Test raster not found: {config.test_raster}")
        source_path = config.test_raster
    
    # Resolve output paths
    _, masks_path, vectors_path = resolve_output_paths(
        source_path=source_path,
        output_dir=config.output_dir,
        mask_path=config.masks_path,
        vector_path=config.vectors_path,
        raster_suffix="unet_predictions",
    )
    
    # Run inference
    if stack_manifest is not None:
        infer_manifest(
            manifest=stack_manifest,
            model_path=config.model_path,
            output_path=masks_path,
            window_size=config.window_size,
            overlap=config.overlap,
            num_channels=config.model.num_channels,
            architecture=config.model.architecture,
            encoder_name=config.model.encoder_name,
            num_classes=config.model.num_classes,
            probability_threshold=config.probability_threshold,
            # Training tiles are now uint8 [0-255], which geoai divides by 255
            # to get [0-1]. RasterStack.read_window() also returns [0-1] data.
            # So we do NOT divide by 255 again (legacy_normalization=False).
            legacy_normalization=False,
        )
    else:
        # Direct raster inference using geoai
        assert config.test_raster is not None  # for type-checkers
        
        # Determine channel count
        num_channels = config.model.num_channels
        if num_channels is None:
            with rasterio.open(config.test_raster) as src:
                num_channels = src.count
        
        logging.info("Running semantic inference with %s input channels", num_channels)
        geoai.semantic_segmentation(
            str(config.test_raster),
            str(masks_path),
            str(config.model_path),
            architecture=config.model.architecture,
            encoder_name=config.model.encoder_name,
            num_channels=num_channels,
            num_classes=config.model.num_classes,
            window_size=config.window_size,
            overlap=config.overlap,
            batch_size=config.batch_size,
            probability_threshold=config.probability_threshold,
        )

    logging.info("Raster predictions written to %s", masks_path)

    geoai.raster_to_vector(
        str(masks_path),
        str(vectors_path),
        min_area=config.min_area,
        simplify_tolerance=config.simplify_tolerance,
    )

    logging.info("Vector predictions written to %s", vectors_path)
    return masks_path, vectors_path


__all__ = ["infer_manifest", "infer_raster", "infer_from_config"]
