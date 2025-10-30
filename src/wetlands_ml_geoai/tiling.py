"""Shared tiling helpers for wetlands_ml_geoai training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import rasterio


def derive_num_channels(raster_path: Path, override: Optional[int]) -> int:
    """Return channel count from ``raster_path`` unless ``override`` is provided."""

    if override is not None:
        return override
    with rasterio.open(raster_path) as src:
        return src.count


def analyze_label_tiles(labels_dir: Path, max_check: int = 256) -> Tuple[float, float, int]:
    """Inspect up to ``max_check`` label tiles and return (all_one_frac, mean_cover, count).

    ``all_one_frac`` denotes the fraction of tiles dominated by foreground (>= 99.9%).
    ``mean_cover`` is the average foreground ratio across inspected tiles.
    ``count`` is the number of readable tiles sampled.
    """

    paths = sorted(p for p in labels_dir.glob("*.tif"))
    if not paths:
        return 0.0, 0.0, 0

    total = 0
    all_one = 0
    cover_fracs = []
    for path in paths[:max_check]:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1)
        except Exception:  # pragma: no cover - ignore unreadable tiles
            continue

        total += 1
        size = arr.size
        if size == 0:
            continue

        positives = int((arr > 0).sum())
        frac = positives / float(size)
        cover_fracs.append(frac)
        if frac >= 0.999:
            all_one += 1

    if total == 0:
        return 0.0, 0.0, 0

    mean_cover = sum(cover_fracs) / len(cover_fracs) if cover_fracs else 0.0
    return all_one / total, mean_cover, total


__all__ = ["derive_num_channels", "analyze_label_tiles"]

