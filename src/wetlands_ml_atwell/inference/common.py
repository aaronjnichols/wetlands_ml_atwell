"""Shared inference utilities for wetlands_ml_atwell."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def resolve_output_paths(
    source_path: Path,
    output_dir: Path | None,
    mask_path: Path | None,
    vector_path: Path | None,
    raster_suffix: str,
) -> Tuple[Path, Path, Path]:
    """Return normalized output directory, mask path, and vector path."""

    base_output = output_dir if output_dir is not None else source_path.parent / "predictions"
    base_output = base_output.expanduser().resolve()
    base_output.mkdir(parents=True, exist_ok=True)

    resolved_mask = (
        mask_path.expanduser().resolve()
        if mask_path is not None
        else base_output / f"{source_path.stem}_{raster_suffix}.tif"
    )
    resolved_vector = (
        vector_path.expanduser().resolve()
        if vector_path is not None
        else base_output / f"{source_path.stem}_{raster_suffix}.gpkg"
    )

    resolved_mask.parent.mkdir(parents=True, exist_ok=True)
    resolved_vector.parent.mkdir(parents=True, exist_ok=True)
    return base_output, resolved_mask, resolved_vector


__all__ = ["resolve_output_paths"]

