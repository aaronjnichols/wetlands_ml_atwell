"""UNet training orchestration for wetlands_ml_geoai."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Optional, Sequence, Tuple

import geoai
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError

from ..stacking import RasterStack, StackManifest, load_manifest, rewrite_tile_images
from ..tiling import analyze_label_tiles, derive_num_channels


def _prepare_labels(
    labels_path: Path,
    tiles_dir: Path,
    target_crs: CRS | None,
) -> Path:
    """Return a vector dataset whose CRS matches ``target_crs`` for tiling.

    The original label file is reprojected into ``tiles_dir`` when needed. This ensures
    that rasterized training masks line up with the imagery grid and prevents the kind
    of consistent vertical striping observed when label geometries stayed in WGS84.
    """

    if not labels_path.exists():
        raise FileNotFoundError(f"Label dataset not found: {labels_path}")

    gdf = gpd.read_file(labels_path)
    if gdf.empty:
        raise ValueError(f"Label dataset contains no features: {labels_path}")
    if gdf.crs is None:
        raise ValueError(f"Label dataset is missing a CRS definition: {labels_path}")

    if target_crs is None:
        return labels_path

    # If CRS already matches, we can keep the original dataset.
    label_crs = CRS.from_user_input(gdf.crs)
    if label_crs == target_crs:
        return labels_path

    tiles_dir.mkdir(parents=True, exist_ok=True)
    target_epsg = target_crs.to_epsg()
    suffix = f"_reprojected_{target_epsg}" if target_epsg else "_reprojected"
    reproj_path = tiles_dir / f"{labels_path.stem}{suffix}.gpkg"

    logging.info(
        "Reprojecting labels from %s to %s -> %s",
        label_crs,
        target_crs,
        reproj_path,
    )
    gdf = gdf.to_crs(target_crs)
    gdf.to_file(reproj_path, driver="GPKG")
    return reproj_path


def _clear_directory(path: Path) -> None:
    """Remove ``path`` and all contents, handling read-only files."""

    if not path.exists():
        return

    def _onerror(func, p, excinfo):
        exc = excinfo[1]
        if isinstance(exc, PermissionError):
            os.chmod(p, stat.S_IWRITE)
            func(p)
        else:
            raise exc

    for attempt in range(5):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5)


def _load_manifests(paths: Optional[Sequence[Path | str]]) -> Sequence[StackManifest]:
    if not paths:
        return []
    manifests: list[StackManifest] = []
    for path in paths:
        manifests.append(load_manifest(path))
    return manifests


def train_unet(
    labels_path: Path,
    tiles_dir: Path,
    models_dir: Path,
    train_raster: Optional[Path] = None,
    stack_manifest_path: Optional[Sequence[Path | str]] = None,
    tile_size: int = 512,
    stride: int = 256,
    buffer_radius: int = 0,
    num_channels_override: Optional[int] = None,
    num_classes: int = 2,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    batch_size: int = 4,
    epochs: int = 25,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_split: float = 0.2,
    save_best_only: bool = True,
    plot_curves: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    resize_mode: str = "resize",
    num_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume_training: bool = False,
) -> None:
    manifests = _load_manifests(stack_manifest_path)

    primary_manifest: Optional[StackManifest] = None
    base_raster: Optional[Path] = None

    if manifests:
        for manifest in manifests:
            naip_source = manifest.naip
            if naip_source is None:
                continue
            raster_path = naip_source.path
            if raster_path.exists():
                primary_manifest = manifest
                base_raster = raster_path
                break
        if primary_manifest is None:
            logging.warning(
                "No usable NAIP source found in supplied manifests; falling back to TRAIN_RASTER."
            )

    if base_raster is None:
        if train_raster is None:
            raise ValueError(
                "train_raster must be provided when no valid stack manifest is supplied."
            )
        base_raster = train_raster

    tiles_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    images_dir = tiles_dir / "images"
    labels_dir = tiles_dir / "labels"

    if images_dir.exists():
        logging.info("Clearing existing image tiles at %s", images_dir)
        _clear_directory(images_dir)
    if labels_dir.exists():
        logging.info("Clearing existing label tiles at %s", labels_dir)
        _clear_directory(labels_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    staging_candidates = sorted(tiles_dir.glob("_staging_tiles*"))
    for stale in staging_candidates:
        try:
            _clear_directory(stale)
        except PermissionError:
            logging.warning("Unable to remove stale staging directory %s", stale)

    staging_root = tiles_dir / f"_staging_tiles_{int(time.time())}"
    staging_root.mkdir(parents=True, exist_ok=True)

    label_cache: dict[str, Path] = {}

    def _labels_for_crs(target_crs: CRS | None) -> Path:
        key = target_crs.to_string() if target_crs else "None"
        if key not in label_cache:
            label_cache[key] = _prepare_labels(labels_path, tiles_dir, target_crs)
        return label_cache[key]

    total_tiles = 0

    if manifests:
        for idx, manifest in enumerate(manifests):
            naip_source = manifest.naip
            if naip_source is None:
                logging.warning(
                    "Skipping manifest %s because it lacks a NAIP source.",
                    manifest.path,
                )
                continue

            raster_path = naip_source.path
            try:
                with rasterio.open(raster_path) as src:
                    manifest_crs = src.crs
            except RasterioIOError as exc:
                logging.warning(
                    "Skipping manifest %s because the NAIP raster is unavailable: %s",
                    manifest.path,
                    exc,
                )
                continue

            labels_source = _labels_for_crs(manifest_crs)

            staging_dir = staging_root / f"manifest_{idx:03d}"
            if staging_dir.exists():
                _clear_directory(staging_dir)
            logging.info(
                "Exporting tiles for manifest %s to %s",
                manifest.path,
                staging_dir,
            )
            geoai.export_geotiff_tiles(
                in_raster=str(raster_path),
                out_folder=str(staging_dir),
                in_class_data=str(labels_source),
                tile_size=tile_size,
                stride=stride,
                buffer_radius=buffer_radius,
            )

            staging_images = staging_dir / "images"
            staging_labels = staging_dir / "labels"
            image_paths = sorted(staging_images.glob("*.tif")) if staging_images.exists() else []

            if not image_paths:
                logging.warning("No tiles were generated for manifest %s", manifest.path)
                _clear_directory(staging_dir)
                continue

            rewritten = rewrite_tile_images(manifest, staging_images)
            logging.info(
                "Rewrote %s image tiles using manifest %s",
                rewritten,
                manifest.path,
            )

            for image_path in image_paths:
                dest_name = f"aoi{idx:02d}_{image_path.name}"
                shutil.move(str(image_path), images_dir / dest_name)
                label_path = staging_labels / image_path.name
                if not label_path.exists():
                    raise FileNotFoundError(
                        f"Missing label tile matching {image_path.name} in {staging_labels}"
                    )
                shutil.move(str(label_path), labels_dir / dest_name)

            total_tiles += len(image_paths)
            _clear_directory(staging_dir)
    else:
        with rasterio.open(base_raster) as src:
            raster_crs = src.crs
        labels_source = _labels_for_crs(raster_crs)
        logging.info("Exporting tiles to %s", tiles_dir)
        geoai.export_geotiff_tiles(
            in_raster=str(base_raster),
            out_folder=str(tiles_dir),
            in_class_data=str(labels_source),
            tile_size=tile_size,
            stride=stride,
            buffer_radius=buffer_radius,
        )
        total_tiles = len(list(images_dir.glob("*.tif")))

    try:
        _clear_directory(staging_root)
    except PermissionError:
        logging.warning("Unable to remove staging directory %s; continuing.", staging_root)

    if not images_dir.exists() or not any(images_dir.glob("*.tif")):
        raise RuntimeError(
            "No image tiles were generated. Check stack manifests and tiling parameters."
        )
    if not labels_dir.exists() or not any(labels_dir.glob("*.tif")):
        raise RuntimeError(
            "No label tiles were generated. Verify the label dataset overlaps the raster extent."
        )

    logging.info(
        "Prepared %s tiles for training in %s (labels in %s)",
        total_tiles,
        images_dir,
        labels_dir,
    )

    if primary_manifest is not None and num_channels_override is None:
        with RasterStack(primary_manifest) as stack:
            num_channels = stack.band_count
    else:
        num_channels = derive_num_channels(base_raster, num_channels_override)

    if num_channels_override is not None:
        num_channels = num_channels_override

    logging.info("Training UNet model with %s input channels", num_channels)

    all_one_frac, avg_cover, checked = analyze_label_tiles(labels_dir)
    if checked:
        logging.info(
            "Analyzed %s label tiles – %.1f%% all-one, mean foreground cover %.3f",
            checked,
            all_one_frac * 100,
            avg_cover,
        )

    geoai.train_segmentation_model(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        output_dir=str(models_dir),
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_channels=num_channels,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        val_split=val_split,
        save_best_only=save_best_only,
        plot_curves=plot_curves,
        target_size=target_size,
        resize_mode=resize_mode,
        num_workers=num_workers,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        resume_training=resume_training,
    )

    logging.info("Training complete. Models saved to %s", models_dir)


__all__ = ["train_unet"]

