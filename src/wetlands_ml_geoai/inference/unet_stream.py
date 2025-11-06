"""UNet sliding-window inference for wetlands_ml_geoai."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

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
) -> np.ndarray:
    array = data.astype(np.float32, copy=False)

    channel_count = array.shape[0]
    if channel_count > desired_channels:
        array = array[:desired_channels]
    elif channel_count < desired_channels:
        padded = np.zeros((desired_channels, array.shape[1], array.shape[2]), dtype=array.dtype)
        padded[:channel_count] = array
        array = padded

    # Normalize using the same approach as training:
    # 1. Clean nodata, clip to [0,1]
    # 2. Divide by 255 (matching geoai's SemanticSegmentationDataset)
    array = normalize_stack_array(array, nodata_value)
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
                prepared = _prepare_window(data, channel_count, nodata_value)
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
) -> None:
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


__all__ = ["infer_manifest", "infer_raster"]

