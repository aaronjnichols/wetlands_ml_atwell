"""
Tile extraction utilities for the NWI-Filtered pipeline.

This module reads the Training Manifest (created by sampling.py)
and physically extracts the image chips and label masks from source rasters
or directly from stack manifests produced by the Sentinel-2 pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds, transform as window_transform
from shapely.geometry import box, mapping

from ..stacking import RasterStack, StackManifest, load_manifest

LOGGER = logging.getLogger(__name__)


def _load_stack_manifests(index_path: Path) -> Sequence[StackManifest]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    manifest_paths = data.get("manifests", [])
    if not manifest_paths:
        raise ValueError(f"Manifest index at {index_path} contains no manifests.")
    return [load_manifest(Path(path)) for path in manifest_paths]


def _manifest_polygon(manifest: StackManifest) -> box:
    bounds = array_bounds(
        manifest.grid.height,
        manifest.grid.width,
        manifest.grid.transform,
    )
    return box(*bounds)


def extract_tiles_from_manifest(
    manifest_path: Union[str, Path],
    source_raster_path: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    labels_path: Optional[Union[str, Path]] = None,
    tile_size_px: int = 512,
    overwrite: bool = False,
    stack_manifest_index: Optional[Union[str, Path]] = None,
) -> None:
    """
    Extracts image and label tiles defined in the manifest.

    The tiles can be read either from a monolithic raster (source_raster_path)
    or streamed directly from stack manifests (stack_manifest_index).
    """

    if source_raster_path is None and stack_manifest_index is None:
        raise ValueError(
            "Either source_raster_path or stack_manifest_index must be provided."
        )

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading sampling manifest from %s...", manifest_path)
    manifest_gdf = gpd.read_file(manifest_path)

    label_gdf = None
    if labels_path:
        LOGGER.info("Loading labels from %s...", labels_path)
        label_gdf = gpd.read_file(labels_path)

    stack_manifests: Optional[Sequence[StackManifest]] = None
    stack_polygons: Optional[List[box]] = None
    stack_cache: Dict[Path, RasterStack] = {}

    if stack_manifest_index:
        index_path = Path(stack_manifest_index)
        LOGGER.info("Streaming tiles from stack manifests listed in %s", index_path)
        stack_manifests = _load_stack_manifests(index_path)
        stack_polygons = [_manifest_polygon(m) for m in stack_manifests]

    if stack_manifests is None:
        source_raster = Path(source_raster_path)
        with rasterio.open(source_raster) as src:
            raster_crs = src.crs
            manifest_proj = manifest_gdf.to_crs(raster_crs)
            if label_gdf is not None:
                label_gdf = label_gdf.to_crs(raster_crs)

            LOGGER.info("Extracting %s tiles from %s...", len(manifest_proj), source_raster.name)
            for idx, row in manifest_proj.iterrows():
                _extract_from_raster(
                    idx,
                    row.geometry,
                    src,
                    images_dir,
                    labels_dir,
                    label_gdf,
                    tile_size_px,
                    overwrite,
                )
    else:
        raster_crs = stack_manifests[0].grid.crs
        manifest_proj = manifest_gdf.to_crs(raster_crs)
        if label_gdf is not None:
            label_gdf = label_gdf.to_crs(raster_crs)

        LOGGER.info(
            "Extracting %s tiles using %s stack manifest(s)...",
            len(manifest_proj),
            len(stack_manifests),
        )

        for idx, row in manifest_proj.iterrows():
            tile_geom = row.geometry
            tile_id = f"tile_{idx:05d}"
            img_out_path = images_dir / f"{tile_id}.tif"
            lbl_out_path = labels_dir / f"{tile_id}.tif"
            if img_out_path.exists() and not overwrite:
                continue

            manifest_match = None
            for manifest_obj, poly in zip(stack_manifests, stack_polygons):
                if poly.intersects(tile_geom):
                    manifest_match = manifest_obj
                    break
            if manifest_match is None:
                LOGGER.error("Tile %s does not intersect any stack manifest.", tile_id)
                continue

            stack = stack_cache.get(manifest_match.path)
            if stack is None:
                stack = RasterStack(manifest_match)
                stack_cache[manifest_match.path] = stack

            try:
                left, bottom, right, top = tile_geom.bounds
                window = from_bounds(
                    left,
                    bottom,
                    right,
                    top,
                    transform=stack.transform,
                    height=stack.height,
                    width=stack.width,
                )
                out_image = stack.read_window(window)
                out_transform = window_transform(window, stack.transform)

                out_meta = stack.profile.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "compress": "deflate",
                    }
                )
                with rasterio.open(img_out_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                if label_gdf is not None:
                    mask_arr = _rasterize_labels(label_gdf, tile_geom, out_transform, out_image.shape[1:])
                    lbl_meta = out_meta.copy()
                    lbl_meta.update({"count": 1, "dtype": "uint8", "nodata": None})
                    with rasterio.open(lbl_out_path, "w", **lbl_meta) as dest:
                        dest.write(mask_arr, 1)
            except Exception as exc:
                LOGGER.error("Failed to process tile %s: %s", tile_id, exc)
                continue

        for stack in stack_cache.values():
            stack.close()

    LOGGER.info("Extraction complete.")


def _rasterize_labels(label_gdf, tile_geom, transform, shape_hw):
    possible_index = list(label_gdf.sindex.intersection(tile_geom.bounds))
    possible = label_gdf.iloc[possible_index]
    precise = possible[possible.intersects(tile_geom)]
    if precise.empty:
        return np.zeros(shape_hw, dtype=np.uint8)
    mask_arr = rasterize(
        shapes=precise.geometry,
        out_shape=shape_hw,
        transform=transform,
        default_value=1,
        fill=0,
        dtype=np.uint8,
    )
    return mask_arr


def _extract_from_raster(
    idx,
    tile_geom,
    src,
    images_dir,
    labels_dir,
    label_gdf,
    tile_size_px,
    overwrite,
):
    from rasterio.mask import mask

    tile_id = f"tile_{idx:05d}"
    img_out_path = images_dir / f"{tile_id}.tif"
    lbl_out_path = labels_dir / f"{tile_id}.tif"
    if img_out_path.exists() and not overwrite:
        return

    out_image, out_transform = mask(src, [mapping(tile_geom)], crop=True)
    out_meta = src.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "deflate",
        }
    )
    with rasterio.open(img_out_path, "w", **out_meta) as dest:
        dest.write(out_image)

    if label_gdf is None:
        return

    mask_arr = _rasterize_labels(label_gdf, tile_geom, out_transform, out_image.shape[1:])
    lbl_meta = out_meta.copy()
    lbl_meta.update({"count": 1, "dtype": "uint8", "nodata": None})
    with rasterio.open(lbl_out_path, "w", **lbl_meta) as dest:
        dest.write(mask_arr, 1)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract training tiles from manifest.")
    parser.add_argument("--manifest-path", required=True, help="Path to Sampling Manifest GPKG")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted tiles")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-raster-path", help="Path to single source raster")
    group.add_argument("--stack-manifest-index", help="Path to stack manifest index JSON")

    parser.add_argument("--labels-path", help="Path to labels (for creating masks)")
    parser.add_argument("--tile-size-px", type=int, default=512, help="Tile size in pixels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tiles")

    args = parser.parse_args()

    extract_tiles_from_manifest(
        manifest_path=args.manifest_path,
        source_raster_path=args.source_raster_path,
        output_dir=args.output_dir,
        labels_path=args.labels_path,
        tile_size_px=args.tile_size_px,
        overwrite=args.overwrite,
        stack_manifest_index=args.stack_manifest_index,
    )
