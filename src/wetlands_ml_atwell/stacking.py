"""Utilities for streaming NAIP + Sentinel-2 stacks defined by manifest files."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

FLOAT_NODATA = -9999.0
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StackGrid:
    crs: Optional[CRS]
    transform: Affine
    width: int
    height: int
    nodata: float = FLOAT_NODATA


@dataclass(frozen=True)
class StackSource:
    type: str
    path: Path
    band_labels: Sequence[str]
    scale_max: Optional[float] = None
    scale_min: Optional[float] = None
    band_scaling: Optional[Dict[str, Sequence[float]]] = None  # Per-band [min, max] scaling
    nodata: Optional[float] = None
    resample: str = "bilinear"
    dtype: Optional[str] = None
    description: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.band_labels)


@dataclass(frozen=True)
class StackManifest:
    path: Path
    grid: StackGrid
    sources: Sequence[StackSource]

    def source_by_type(self, source_type: str) -> Optional[StackSource]:
        for source in self.sources:
            if source.type == source_type:
                return source
        return None

    @property
    def naip(self) -> Optional[StackSource]:
        return self.source_by_type("naip")


def load_manifest(path: Union[str, Path]) -> StackManifest:
    manifest_path = Path(path).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    grid_data = data.get("grid", {})
    transform = Affine(*grid_data["transform"])
    crs_value = grid_data.get("crs")
    crs = CRS.from_string(crs_value) if crs_value else None
    grid = StackGrid(
        crs=crs,
        transform=transform,
        width=int(grid_data["width"]),
        height=int(grid_data["height"]),
        nodata=float(grid_data.get("nodata", FLOAT_NODATA)),
    )

    sources: List[StackSource] = []
    for source_data in data.get("sources", []):
        # Parse band_scaling if present (convert lists to tuples for hashability)
        raw_band_scaling = source_data.get("band_scaling")
        band_scaling = None
        if raw_band_scaling and isinstance(raw_band_scaling, dict):
            band_scaling = {k: tuple(v) for k, v in raw_band_scaling.items()}

        sources.append(
            StackSource(
                type=source_data["type"],
                path=Path(source_data["path"]).expanduser().resolve(),
                band_labels=tuple(source_data.get("band_labels", [])),
                scale_max=source_data.get("scale_max"),
                scale_min=source_data.get("scale_min"),
                band_scaling=band_scaling,
                nodata=source_data.get("nodata"),
                resample=source_data.get("resample", "bilinear"),
                dtype=source_data.get("dtype"),
                description=source_data.get("description"),
            )
        )

    return StackManifest(path=manifest_path, grid=grid, sources=tuple(sources))


class RasterStack:
    """Provide windowed access to a NAIP + Sentinel stack using on-demand resampling."""

    def __init__(self, manifest: Union[str, Path, StackManifest]) -> None:
        if isinstance(manifest, (str, Path)):
            self.manifest = load_manifest(manifest)
        else:
            self.manifest = manifest
        self._base_datasets: List[rasterio.io.DatasetReader] = []
        self._readers: List[rasterio.io.DatasetReader] = []
        self._band_labels: List[str] = []
        self._open_sources()

    def _open_sources(self) -> None:
        target_crs = self.manifest.grid.crs
        target_transform = self.manifest.grid.transform
        target_width = self.manifest.grid.width
        target_height = self.manifest.grid.height

        for source in self.manifest.sources:
            base = rasterio.open(source.path, "r", sharing=True)
            reader = base
            resampling = Resampling.nearest
            if source.type != "naip":
                resampling = Resampling.bilinear if source.resample == "bilinear" else Resampling.nearest
                reader = WarpedVRT(
                    base,
                    crs=target_crs,
                    transform=target_transform,
                    width=target_width,
                    height=target_height,
                    resampling=resampling,
                    src_nodata=source.nodata,
                    nodata=FLOAT_NODATA,
                )
            self._base_datasets.append(base)
            self._readers.append(reader)
            self._band_labels.extend(source.band_labels)
        self._cached_sources = list(self.manifest.sources)

    def close(self) -> None:
        for reader, base in zip(self._readers, self._base_datasets):
            if reader is not base:
                reader.close()
            base.close()
        self._readers.clear()
        self._base_datasets.clear()

    def __enter__(self) -> "RasterStack":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # Exposed metadata -------------------------------------------------
    @property
    def transform(self) -> Affine:
        return self.manifest.grid.transform

    @property
    def crs(self) -> Optional[CRS]:
        return self.manifest.grid.crs

    @property
    def width(self) -> int:
        return self.manifest.grid.width

    @property
    def height(self) -> int:
        return self.manifest.grid.height

    @property
    def band_labels(self) -> Sequence[str]:
        return tuple(self._band_labels)

    @property
    def band_count(self) -> int:
        return len(self._band_labels)

    @property
    def profile(self) -> Dict[str, Any]:
        profile = {
            "driver": "GTiff",
            "width": self.width,
            "height": self.height,
            "count": self.band_count,
            "dtype": "float32",
            "transform": self.transform,
            "nodata": FLOAT_NODATA,
            "compress": "deflate",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
        if self.crs:
            profile["crs"] = self.crs
        return profile

    # Reading ----------------------------------------------------------


    def read_window(self, window: Window) -> np.ndarray:
        out_height = int(math.ceil(window.height))
        out_width = int(math.ceil(window.width))
        if out_height <= 0 or out_width <= 0:
            raise ValueError("Window height/width must be positive")

        row_off = window.row_off
        col_off = window.col_off
        row_end = row_off + window.height
        col_end = col_off + window.width

        clip_row_off = max(int(math.floor(row_off)), 0)
        clip_col_off = max(int(math.floor(col_off)), 0)
        clip_row_end = min(int(math.ceil(row_end)), self.height)
        clip_col_end = min(int(math.ceil(col_end)), self.width)

        arrays: List[np.ndarray] = []

        for source, reader in zip(self._cached_sources, self._readers):
            resampling = Resampling.bilinear if source.type != "naip" and source.resample == "bilinear" else Resampling.nearest
            result = np.full((source.count, out_height, out_width), FLOAT_NODATA, dtype="float32")

            if clip_row_end > clip_row_off and clip_col_end > clip_col_off:
                clip_height = clip_row_end - clip_row_off
                clip_width = clip_col_end - clip_col_off
                clip_window = Window(clip_col_off, clip_row_off, clip_width, clip_height)

                data = reader.read(
                    window=clip_window,
                    out_shape=(source.count, clip_height, clip_width),
                    resampling=resampling,
                ).astype("float32", copy=False)

                if source.nodata is not None:
                    nodata_mask = data == float(source.nodata)
                    if nodata_mask.any():
                        data = data.copy()
                        data[nodata_mask] = FLOAT_NODATA

                # Apply scaling to normalize values to [0, 1] range
                if source.band_scaling:
                    # Per-band min-max scaling (e.g., for topography with different ranges)
                    data = data.copy()
                    for band_idx, label in enumerate(source.band_labels):
                        if label in source.band_scaling:
                            vmin, vmax = source.band_scaling[label]
                            band_mask = data[band_idx] != FLOAT_NODATA
                            if band_mask.any():
                                # Normalize: (value - min) / (max - min)
                                scale_range = float(vmax) - float(vmin)
                                if scale_range > 0:
                                    data[band_idx][band_mask] = (
                                        data[band_idx][band_mask] - float(vmin)
                                    ) / scale_range
                elif source.scale_max is not None:
                    # Single scale_max for all bands (e.g., NAIP with 0-255)
                    mask = data != FLOAT_NODATA
                    if mask.any():
                        data = data.copy()
                        if source.scale_min is not None:
                            scale_range = float(source.scale_max) - float(source.scale_min)
                            if scale_range > 0:
                                data[mask] = (data[mask] - float(source.scale_min)) / scale_range
                        else:
                            data[mask] = data[mask] / float(source.scale_max)

                row_insert = clip_row_off - row_off
                col_insert = clip_col_off - col_off
                row_insert_int = int(round(row_insert))
                col_insert_int = int(round(col_insert))
                result[:, row_insert_int:row_insert_int + clip_height, col_insert_int:col_insert_int + clip_width] = data

            arrays.append(result)

        return np.concatenate(arrays, axis=0)

    def window_from_transform(self, transform: Affine, width: int, height: int) -> Window:
        x_ul = transform.c
        y_ul = transform.f
        row, col = rowcol(self.transform, x_ul, y_ul)
        return Window(col, row, width, height)

def normalize_stack_array(
    data: np.ndarray,
    nodata_value: Optional[float] = FLOAT_NODATA,
    warn_on_clip: bool = True,
) -> np.ndarray:
    """Return a copy of ``data`` with nodata cleaned, NaNs filled, and values clipped to [0, 1].

    Args:
        data: Input array to normalize.
        nodata_value: Value representing nodata pixels (will be set to 0).
        warn_on_clip: If True, log warnings when values are clipped.

    Returns:
        Normalized array with values in [0, 1] range.
    """
    import logging

    cleaned = data.astype(np.float32, copy=True)
    if nodata_value is not None:
        nodata_mask = cleaned == float(nodata_value)
        if nodata_mask.any():
            cleaned[nodata_mask] = 0.0

    # Replace NaNs and infinities with finite bounds prior to clipping.
    np.nan_to_num(cleaned, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    # Warn if values will be clipped - indicates missing scaling configuration
    if warn_on_clip:
        # Check valid (non-zero after nodata replacement) values
        valid_mask = np.ones_like(cleaned, dtype=bool)
        if nodata_value is not None:
            valid_mask = data != float(nodata_value)

        if valid_mask.any():
            valid_values = cleaned[valid_mask]
            max_val = float(valid_values.max())
            min_val = float(valid_values.min())

            if max_val > 1.0:
                logging.warning(
                    "Values above 1.0 detected (max=%.2f) - these will be clipped to 1.0. "
                    "This likely means a data source is missing 'scale_max' or 'band_scaling' "
                    "in its manifest configuration. Check topography or other raw-value sources.",
                    max_val
                )
            if min_val < 0.0:
                logging.warning(
                    "Negative values detected (min=%.2f) - these will be clipped to 0.0. "
                    "For sources with negative values (e.g., TPI), use 'band_scaling' with "
                    "appropriate min/max range in the manifest.",
                    min_val
                )

    np.clip(cleaned, 0.0, 1.0, out=cleaned)
    return cleaned


def rewrite_tile_images(manifest: Union[str, Path, StackManifest], images_dir: Path) -> int:
    """Rewrite GeoTIFF tiles in-place using the manifest-defined stack.
    
    This function reads tile data from the manifest sources, normalizes all
    values to [0-1] range, then converts to uint8 [0-255] format for
    compatibility with geoai's training system.
    
    Why uint8?
        geoai's SemanticSegmentationDataset always divides by 255, assuming
        uint8 input. By writing uint8 [0-255] tiles, geoai's /255 division
        produces the correct [0-1] range that neural networks expect.
        
        See normalization.py for the full data flow documentation.
    
    Args:
        manifest: Path to manifest JSON or StackManifest object.
        images_dir: Directory containing tile GeoTIFFs to rewrite.
        
    Returns:
        Number of tiles rewritten.
    """
    # Import here to avoid circular dependency
    from .normalization import to_geoai_format
    
    manifest_obj = load_manifest(manifest) if not isinstance(manifest, StackManifest) else manifest
    image_paths = sorted(p for p in images_dir.glob("*.tif") if p.is_file())
    if not image_paths:
        return 0

    with RasterStack(manifest_obj) as stack:
        base_profile = stack.profile
        for tile_path in image_paths:
            with rasterio.open(tile_path) as tile_src:
                window = stack.window_from_transform(tile_src.transform, tile_src.width, tile_src.height)
                data = stack.read_window(window)
                # Normalize to [0-1] range (handles nodata, clips values)
                data = normalize_stack_array(data, FLOAT_NODATA)
                # Convert to uint8 [0-255] for geoai compatibility
                data = to_geoai_format(data)
                tile_transform = tile_src.transform
                tile_height = tile_src.height
                tile_width = tile_src.width
            
            # Update profile for uint8 output
            profile = base_profile.copy()
            profile.update(
                width=tile_width,
                height=tile_height,
                transform=tile_transform,
                dtype="uint8",  # geoai expects uint8 tiles
                nodata=0,  # uint8 nodata (was FLOAT_NODATA for float32)
            )
            
            with rasterio.open(tile_path, "w", **profile) as dst:
                dst.write(data)
                for idx, label in enumerate(stack.band_labels, start=1):
                    dst.set_band_description(idx, label)
                    
    LOGGER.info(
        "Rewrote %d tiles as uint8 [0-255] for geoai compatibility",
        len(image_paths)
    )
    return len(image_paths)


__all__ = [
    "FLOAT_NODATA",
    "StackGrid",
    "StackSource",
    "StackManifest",
    "load_manifest",
    "RasterStack",
    "normalize_stack_array",
    "rewrite_tile_images",
]

