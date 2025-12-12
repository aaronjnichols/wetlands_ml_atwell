"""UNet training orchestration for wetlands_ml_atwell."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import geoai
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
import json

from ..stacking import RasterStack, StackManifest, load_manifest, rewrite_tile_images
from ..tiling import analyze_label_tiles, derive_num_channels
from ..config import TrainingConfig, SamplingConfig


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


def _get_tile_extents(images_dir: Path) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of tile bounding boxes from image files.

    Args:
        images_dir: Directory containing tile GeoTIFF files.

    Returns:
        GeoDataFrame with tile geometries and filenames.
    """
    from shapely.geometry import box

    tile_files = sorted(images_dir.glob("*.tif"))
    records = []

    for tile_path in tile_files:
        with rasterio.open(tile_path) as src:
            bounds = src.bounds
            crs = src.crs
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            records.append({
                "filename": tile_path.name,
                "geometry": geom,
            })

    if not records:
        return gpd.GeoDataFrame(columns=["filename", "geometry"])

    # Use CRS from last tile (all should be same)
    return gpd.GeoDataFrame(records, crs=crs)


def _apply_balanced_sampling(
    images_dir: Path,
    labels_dir: Path,
    labels_path: Path,
    nwi_path: Path,
    positive_negative_ratio: float = 1.0,
    safe_zone_buffer: float = 100.0,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """Apply NWI-filtered balanced sampling to generated tiles.

    After geoai generates all tiles, this function filters them to keep:
    - All positive tiles (intersecting with training labels)
    - Sampled negative tiles from the "safe zone" (away from labels and NWI)

    Args:
        images_dir: Directory containing generated image tiles.
        labels_dir: Directory containing generated label tiles.
        labels_path: Path to training labels (positive polygons).
        nwi_path: Path to NWI wetlands for exclusion.
        positive_negative_ratio: Ratio of negative to positive tiles.
        safe_zone_buffer: Buffer around positive labels in meters.
        seed: Random seed for sampling.

    Returns:
        Tuple of (num_positive, num_negative, num_removed).
    """
    from shapely.ops import unary_union
    import numpy as np

    logging.info("Applying balanced sampling with NWI exclusion...")

    # Get tile extents
    tiles_gdf = _get_tile_extents(images_dir)
    if tiles_gdf.empty:
        logging.warning("No tiles found for balanced sampling")
        return 0, 0, 0

    original_crs = tiles_gdf.crs
    total_tiles = len(tiles_gdf)

    # Use projected CRS for spatial operations
    target_crs = "EPSG:5070"  # NAD83 Conus Albers
    tiles_proj = tiles_gdf.to_crs(target_crs)

    # Load labels (training positives)
    labels_gdf = gpd.read_file(labels_path)
    if labels_gdf.crs is None:
        labels_gdf.set_crs("EPSG:4326", inplace=True)
    labels_proj = labels_gdf.to_crs(target_crs)

    # Load NWI
    nwi_gdf = gpd.read_file(nwi_path)
    if nwi_gdf.crs is None:
        nwi_gdf.set_crs("EPSG:4326", inplace=True)
    nwi_proj = nwi_gdf.to_crs(target_crs)

    # Identify positive tiles (intersecting with labels)
    positive_join = gpd.sjoin(tiles_proj, labels_proj, how="inner", predicate="intersects")
    positive_indices = set(positive_join.index.unique())

    num_positive = len(positive_indices)
    logging.info(f"Found {num_positive} positive tiles (intersecting labels)")

    if num_positive == 0:
        logging.warning("No positive tiles found! Keeping all tiles.")
        return 0, total_tiles, 0

    # Identify candidate negative tiles
    candidate_negative_indices = set(tiles_proj.index) - positive_indices
    candidate_negatives = tiles_proj.loc[list(candidate_negative_indices)]

    # Compute safe zone: exclude areas near labels and NWI
    # Buffer the labels
    labels_buffered = labels_proj.geometry.buffer(safe_zone_buffer)
    if hasattr(labels_buffered, "union_all"):
        labels_exclusion = labels_buffered.union_all()
    else:
        labels_exclusion = unary_union(labels_buffered)

    # Union NWI geometries
    nwi_clean = nwi_proj.geometry.buffer(0)  # Fix invalid geometries
    if hasattr(nwi_clean, "union_all"):
        nwi_exclusion = nwi_clean.union_all()
    else:
        nwi_exclusion = unary_union(nwi_clean)

    # Combined exclusion zone
    exclusion_zone = labels_exclusion.union(nwi_exclusion)

    # Create safe zone GeoDataFrame for spatial join
    safe_zone_geom = tiles_proj.unary_union.difference(exclusion_zone)
    safe_zone_gdf = gpd.GeoDataFrame(
        {"geometry": [safe_zone_geom]}, crs=target_crs
    ).explode(index_parts=True).reset_index(drop=True)

    # Filter to tiles entirely within safe zone
    if not candidate_negatives.empty and not safe_zone_gdf.empty:
        safe_join = gpd.sjoin(
            candidate_negatives, safe_zone_gdf, how="inner", predicate="within"
        )
        safe_negative_indices = set(safe_join.index.unique())
    else:
        safe_negative_indices = set()

    num_safe_negatives = len(safe_negative_indices)
    logging.info(f"Found {num_safe_negatives} safe negative tiles (within safe zone)")

    # Sample negatives based on ratio
    target_negatives = int(num_positive * positive_negative_ratio)

    if num_safe_negatives > target_negatives:
        np.random.seed(seed)
        sampled_indices = np.random.choice(
            list(safe_negative_indices), size=target_negatives, replace=False
        )
        selected_negative_indices = set(sampled_indices)
        logging.info(f"Sampled {target_negatives} negatives from {num_safe_negatives} safe candidates")
    else:
        selected_negative_indices = safe_negative_indices
        logging.info(f"Using all {num_safe_negatives} safe negative tiles (fewer than target {target_negatives})")

    # Determine tiles to keep
    keep_indices = positive_indices | selected_negative_indices
    remove_indices = set(tiles_proj.index) - keep_indices

    # Delete non-selected tiles
    num_removed = 0
    for idx in remove_indices:
        filename = tiles_gdf.loc[idx, "filename"]
        image_path = images_dir / filename
        label_path = labels_dir / filename

        if image_path.exists():
            image_path.unlink()
            num_removed += 1
        if label_path.exists():
            label_path.unlink()

    num_negative = len(selected_negative_indices)
    logging.info(
        f"Balanced sampling complete: {num_positive} positive, {num_negative} negative, "
        f"{num_removed} removed"
    )

    return num_positive, num_negative, num_removed


# =============================================================================
# Manifest Resolution (moved from train_unet.py)
# =============================================================================


def _is_manifest_index(data: Dict[str, Any]) -> bool:
    manifests = data.get("manifests")
    if not isinstance(manifests, list):
        return False
    return all(isinstance(item, str) and item for item in manifests)


def _is_stack_manifest(data: Dict[str, Any]) -> bool:
    grid = data.get("grid")
    sources = data.get("sources")
    if not isinstance(grid, dict) or "transform" not in grid:
        return False
    if not isinstance(sources, list) or not sources:
        return False
    return True


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _gather_manifest_paths(target: Path, results: List[Path], visited: Set[Path]) -> None:
    normalized = target.resolve()
    if normalized in visited:
        return
    visited.add(normalized)

    if normalized.is_dir():
        entries = sorted(normalized.iterdir(), key=lambda path: path.name.lower())
        for child in entries:
            if child.is_dir() or child.suffix.lower() == ".json":
                _gather_manifest_paths(child, results, visited)
        return

    if normalized.suffix.lower() != ".json":
        return

    data = _load_json(normalized)
    if data is None:
        return

    if _is_manifest_index(data):
        manifests = data.get("manifests", [])
        parent = normalized.parent
        for entry in manifests:
            entry_path = Path(entry)
            if not entry_path.is_absolute():
                entry_path = (parent / entry_path).resolve()
            else:
                entry_path = entry_path.resolve()
            _gather_manifest_paths(entry_path, results, visited)
        return

    if _is_stack_manifest(data):
        results.append(normalized)
        return


def _resolve_manifest_paths(stack_manifest: Optional[Union[str, Path]]) -> Sequence[Path]:
    if stack_manifest is None:
        return []

    manifest_path = Path(stack_manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stack manifest path not found: {manifest_path}")

    resolved: List[Path] = []
    _gather_manifest_paths(manifest_path, resolved, set())

    unique: List[Path] = []
    seen: Set[Path] = set()
    for candidate in resolved:
        path = candidate.resolve()
        if path not in seen:
            if not path.exists():
                raise FileNotFoundError(f"Stack manifest referenced but not found: {path}")
            seen.add(path)
            unique.append(path)

    if not unique:
        raise FileNotFoundError(
            f"No stack manifest JSON files discovered for path: {manifest_path}"
        )

    return unique


# =============================================================================
# Training Orchestration
# =============================================================================


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
    skip_tiling: bool = False,
    # Balanced sampling parameters
    balanced_sampling: bool = False,
    nwi_path: Optional[Path] = None,
    positive_negative_ratio: float = 1.0,
    safe_zone_buffer: float = 100.0,
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

    # Check if we can skip tiling
    if skip_tiling:
        existing_images = list(images_dir.glob("*.tif")) if images_dir.exists() else []
        existing_labels = list(labels_dir.glob("*.tif")) if labels_dir.exists() else []
        if existing_images and existing_labels:
            logging.info(
                "Skipping tile generation (--skip-tiling): found %d image tiles, %d label tiles",
                len(existing_images),
                len(existing_labels),
            )
            total_tiles = len(existing_images)
        else:
            raise RuntimeError(
                f"--skip-tiling specified but tiles not found. "
                f"images_dir={images_dir} has {len(existing_images)} tiles, "
                f"labels_dir={labels_dir} has {len(existing_labels)} tiles. "
                f"Run without --skip-tiling first to generate tiles."
            )
    else:
        # Normal tiling flow: clear and regenerate
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

    # Apply balanced sampling if enabled (only when generating new tiles)
    if balanced_sampling and not skip_tiling:
        if nwi_path is None:
            raise ValueError("nwi_path is required when balanced_sampling is enabled")
        if not nwi_path.exists():
            raise FileNotFoundError(f"NWI file not found: {nwi_path}")

        num_positive, num_negative, num_removed = _apply_balanced_sampling(
            images_dir=images_dir,
            labels_dir=labels_dir,
            labels_path=labels_path,
            nwi_path=nwi_path,
            positive_negative_ratio=positive_negative_ratio,
            safe_zone_buffer=safe_zone_buffer,
            seed=seed,
        )
        total_tiles = num_positive + num_negative
        logging.info(
            "After balanced sampling: %s total tiles (%s positive, %s negative)",
            total_tiles,
            num_positive,
            num_negative,
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


def train_unet_from_config(config: TrainingConfig) -> None:
    """Run the training workflow from a TrainingConfig object.

    This function handles manifest resolution and default directory computation,
    then delegates to train_unet() for the actual training workflow.

    Args:
        config: Validated TrainingConfig instance.

    Raises:
        FileNotFoundError: If labels or manifests are not found.
        ValueError: If configuration is invalid.
    """
    # Resolve manifest paths if stack_manifest is provided
    manifest_paths = _resolve_manifest_paths(config.stack_manifest)
    
    # Determine base raster and default parent directory for outputs
    if manifest_paths:
        first_manifest = load_manifest(manifest_paths[0])
        naip_source = first_manifest.naip
        if naip_source is None:
            raise ValueError("Stack manifest does not include a NAIP source.")
        base_raster = Path(naip_source.path)
        default_parent = Path(manifest_paths[0]).parent
    else:
        if config.train_raster is None:
            raise ValueError(
                "train_raster must be provided when no stack manifest is supplied."
            )
        base_raster = config.train_raster
        if not base_raster.exists():
            raise FileNotFoundError(f"Training raster not found: {base_raster}")
        default_parent = base_raster.parent
    
    # Compute tiles and models directories with defaults if not specified
    tiles_dir = (
        config.tiles_dir 
        if config.tiles_dir 
        else default_parent / "tiles"
    )
    models_dir = (
        config.models_dir 
        if config.models_dir 
        else tiles_dir / "models_unet"
    )

    train_unet(
        labels_path=config.labels_path,
        tiles_dir=tiles_dir,
        models_dir=models_dir,
        train_raster=base_raster,
        stack_manifest_path=manifest_paths if manifest_paths else None,
        tile_size=config.tiling.tile_size,
        stride=config.tiling.stride,
        buffer_radius=config.tiling.buffer_radius,
        num_channels_override=config.model.num_channels,
        num_classes=config.model.num_classes,
        architecture=config.model.architecture,
        encoder_name=config.model.encoder_name,
        encoder_weights=config.model.encoder_weights,
        batch_size=config.hyperparameters.batch_size,
        epochs=config.hyperparameters.epochs,
        learning_rate=config.hyperparameters.learning_rate,
        weight_decay=config.hyperparameters.weight_decay,
        seed=config.hyperparameters.seed,
        val_split=config.hyperparameters.val_split,
        save_best_only=config.save_best_only,
        plot_curves=config.plot_curves,
        target_size=config.target_size,
        resize_mode=config.resize_mode,
        num_workers=config.num_workers,
        checkpoint_path=config.checkpoint_path,
        resume_training=config.resume_training,
        skip_tiling=config.skip_tiling,
        # Balanced sampling parameters
        balanced_sampling=config.sampling.enabled,
        nwi_path=config.sampling.nwi_path,
        positive_negative_ratio=config.sampling.positive_negative_ratio,
        safe_zone_buffer=config.sampling.safe_zone_buffer,
    )


__all__ = ["train_unet", "train_unet_from_config"]
