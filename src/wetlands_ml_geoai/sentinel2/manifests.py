"""Manifest helpers for Sentinel-2 and NAIP stacks."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from dataclasses import dataclass

import numpy as np
import rasterio

from ..stacking import FLOAT_NODATA
from .progress import format_duration


@dataclass(frozen=True)
class ManifestEntry:
    """Captures per-AOI stack component paths."""

    naip_path: Path
    sentinel_path: Path
    reference_profile: Dict[str, any]
    extra_sources: Optional[Sequence[Dict[str, any]]] = None


NAIP_BAND_LABELS = ["NAIP_R", "NAIP_G", "NAIP_B", "NAIP_NIR"]


def affine_to_list(transform):
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def compute_naip_scaling(dataset: rasterio.io.DatasetReader) -> Optional[float]:
    if dataset.nodata is None and not np.issubdtype(np.dtype(dataset.dtypes[0]), np.integer):
        return None
    dtype = np.dtype(dataset.dtypes[0])
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if info.max <= 0:
            return None
        return float(info.max)
    return None


def write_stack_manifest(
    output_dir: Path,
    naip_path: Path,
    naip_labels: Sequence[str],
    sentinel_path: Path,
    sentinel_labels: Sequence[str],
    reference_profile: Dict[str, any],
    extra_sources: Optional[Sequence[Dict[str, any]]] = None,
) -> Path:
    logging.info("Preparing stack manifest")
    manifest = {
        "version": 1,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "grid": {
            "crs": reference_profile.get("crs"),
            "transform": affine_to_list(reference_profile["transform"]),
            "width": reference_profile["width"],
            "height": reference_profile["height"],
            "pixel_size": [
                reference_profile["transform"].a,
                -reference_profile["transform"].e,
            ],
            "dtype": "float32",
            "nodata": FLOAT_NODATA,
        },
        "sources": [],
    }

    naip_entry = {
        "type": "naip",
        "path": str(Path(naip_path).resolve()),
        "band_labels": list(naip_labels),
        "scale_max": None,
        "nodata": None,
    }

    with rasterio.open(naip_path) as src:
        naip_entry["nodata"] = src.nodata
        naip_entry["dtype"] = src.dtypes[0]
        scale = compute_naip_scaling(src)
        if scale:
            naip_entry["scale_max"] = scale
    manifest["sources"].append(naip_entry)

    sentinel_entry = {
        "type": "sentinel",
        "path": str(Path(sentinel_path).resolve()),
        "band_labels": list(sentinel_labels),
        "resample": "bilinear",
        "nodata": FLOAT_NODATA,
    }
    manifest["sources"].append(sentinel_entry)

    if extra_sources:
        logging.info("Appending %s extra source(s) to manifest", len(extra_sources))
        for source in extra_sources:
            manifest["sources"].append(source)

    manifest_path = output_dir / "stack_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logging.info("Stack manifest written -> %s", manifest_path)
    return manifest_path


def write_manifest_index(manifest_paths: Iterable[Path], output_dir: Path) -> Path:
    """Persist an index JSON listing per-AOI manifest paths."""

    manifest_list = [str(Path(path).resolve()) for path in manifest_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    index_payload = {
        "version": 1,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "count": len(manifest_list),
        "manifests": manifest_list,
    }
    index_path = output_dir / "manifest_index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    logging.info("Manifest index written -> %s", index_path)
    return index_path


__all__ = [
    "ManifestEntry",
    "affine_to_list",
    "compute_naip_scaling",
    "write_stack_manifest",
    "write_manifest_index",
]

