"""Sentinel-2 seasonal compositing orchestration."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import xarray as xr
from pystac_client import Client
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping, shape

from ..services.download import NaipService, WetlandsService, NaipDownloadRequest, WetlandsDownloadRequest
from ..training.sampling import create_acquisition_aoi
from ..topography import TopographyStackConfig, prepare_topography_stack
from .aoi import parse_aoi, extract_aoi_polygons
from .manifests import write_manifest_index, write_stack_manifest
from .naip import (
    collect_naip_sources,
    prepare_naip_reference,
    _collect_naip_footprints,
    _clip_raster_to_polygon,
    compute_naip_union_extent,
)
from ..stacking import FLOAT_NODATA
from .progress import RasterProgress
from .seasonal import seasonal_median, concatenate_seasons, write_dataarray
from .stac_client import DEFAULT_SEASONS, SENTINEL_BANDS, fetch_items

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AoiProcessingOutput:
    index: int
    directory: Path
    polygon: Polygon
    seasonal_data: Dict[str, xr.DataArray]
    seasonal_labels: Dict[str, List[str]]
    sentinel_stack_path: Optional[Path]
    manifest_path: Optional[Path]
    has_manifest: bool


def _iter_aoi_processing(
    client: Client,
    polygons: Sequence[Polygon],
    years: Sequence[int],
    seasons: Sequence[str],
    cloud_cover: float,
    min_clear_obs: int,
    mask_dilation: int,
    tiles_per_season: Path,
    chunk_size: Optional[int] = None,
    max_scenes_per_season: Optional[int] = None,
    parallel_fetch: bool = False,
    fetch_workers: int = 24,
    target_crs: Optional[str] = None,
) -> Iterable[AoiProcessingOutput]:
    for index, polygon in enumerate(polygons, start=1):
        aoi_dir = tiles_per_season / f"aoi_{index:02d}"
        aoi_dir.mkdir(parents=True, exist_ok=True)
        seasonal_data: Dict[str, xr.DataArray] = {}
        seasonal_labels: Dict[str, List[str]] = {}
        bounds = polygon.bounds

        shapely_mask = polygon
        mask_cache: Dict[str, Polygon] = {}

        for season in seasons:
            raster_label = f"AOI {index:02d} :: Sentinel-2 {season}"
            LOGGER.info("=" * 60)
            LOGGER.info("%s -- fetching Sentinel-2 scenes", raster_label)
            LOGGER.info("=" * 60)
            items = fetch_items(client, polygon, season, years, cloud_cover)
            total_found = len(items)
            LOGGER.info("%s -- found %d scenes", raster_label, total_found)

            # Log unique tile IDs to help diagnose coverage issues
            if items:
                tile_ids = set()
                for item in items:
                    # Extract tile ID from scene ID (e.g., "S2A_16TFN_20231113_1_L2A" -> "16TFN")
                    parts = item.id.split("_")
                    if len(parts) >= 2:
                        tile_ids.add(parts[1])
                LOGGER.info("%s -- tiles: %s", raster_label, ", ".join(sorted(tile_ids)))

            if not items:
                LOGGER.warning("%s -- no Sentinel-2 scenes", raster_label)
                continue

            # Filter to best N scenes by cloud cover if requested
            if max_scenes_per_season and len(items) > max_scenes_per_season:
                from .stac_client import filter_best_scenes
                items = filter_best_scenes(items, max_scenes_per_season)
                LOGGER.info(
                    "%s -- filtered to %d/%d clearest scenes",
                    raster_label, len(items), total_found
                )

            # Use parallel fetch strategy if enabled, otherwise use standard stackstac
            if parallel_fetch:
                from .seasonal import seasonal_median_parallel
                median, counts = seasonal_median_parallel(
                    items, season, min_clear_obs, bounds, mask_dilation,
                    chunks=chunk_size, max_workers=fetch_workers, target_crs=target_crs
                )
            else:
                median, counts = seasonal_median(
                    items, season, min_clear_obs, bounds, mask_dilation,
                    chunks=chunk_size, target_crs=target_crs
                )

            median_crs = median.rio.crs
            clip_geom = shapely_mask
            if median_crs is not None:
                crs_key = median_crs.to_string() if hasattr(median_crs, "to_string") else str(median_crs)
                clip_geom = mask_cache.get(crs_key)
                if clip_geom is None:
                    transformed = transform_geom("EPSG:4326", median_crs, mapping(shapely_mask))
                    clip_geom = shape(transformed)
                    mask_cache[crs_key] = clip_geom

            clip_mask = median.rio.clip([mapping(clip_geom)], all_touched=False, drop=False)
            seasonal_data[season] = median.where(clip_mask.notnull(), FLOAT_NODATA)
            seasonal_labels[season] = [f"S2_{season}_{band}" for band in SENTINEL_BANDS]

            LOGGER.info(
                "AOI %s Season %s clear obs stats -> min=%s mean=%.2f max=%s",
                index,
                season,
                int(counts.min().item()),
                float(counts.mean().item()),
                int(counts.max().item()),
            )

            season_path = aoi_dir / f"s2_{season.lower()}_median_7band.tif"
            write_dataarray(seasonal_data[season], season_path, seasonal_labels[season])
            LOGGER.info("%s -- wrote %s", raster_label, season_path)

        yield AoiProcessingOutput(
            index=index,
            directory=aoi_dir,
            polygon=polygon,
            seasonal_data=seasonal_data,
            seasonal_labels=seasonal_labels,
            sentinel_stack_path=None,
            manifest_path=None,
            has_manifest=False,
        )


def _deduplicate_paths(paths: Sequence[Path]) -> List[Path]:
    """Remove duplicate paths by resolved absolute path.

    Preserves order (first occurrence kept).

    Args:
        paths: Sequence of paths that may contain duplicates.

    Returns:
        List of unique paths.
    """
    seen: set = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def run_pipeline(
    aoi: str,
    years: Sequence[int],
    output_dir: Path,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    cloud_cover: float = 60.0,
    min_clear_obs: int = 3,
    stac_url: str = "https://earth-search.aws.element84.com/v1",
    naip_paths: Optional[Sequence[Path]] = None,
    auto_download_naip: bool = False,
    auto_download_naip_year: Optional[int] = None,
    auto_download_naip_max_items: Optional[int] = None,
    auto_download_naip_overwrite: bool = False,
    auto_download_naip_preview: bool = False,
    auto_download_wetlands: bool = False,
    wetlands_output_path: Optional[Path] = None,
    wetlands_overwrite: bool = False,
    naip_target_resolution: Optional[float] = None,
    target_crs: str = "EPSG:5070",
    mask_dilation: int = 0,
    auto_download_topography: bool = False,
    topography_cache_dir: Optional[Path] = None,
    topography_dem_dir: Optional[Path] = None,
    topography_buffer_meters: float = 200.0,
    topography_tpi_small: float = 30.0,
    topography_tpi_large: float = 150.0,
    dem_resolution: Optional[str] = None,
    chunk_size: Optional[str] = "auto",
    max_scenes_per_season: Optional[int] = None,
    parallel_fetch: bool = False,
    fetch_workers: int = 24,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    naip_candidates: List[Path] = list(naip_paths) if naip_paths else []
    naip_sources = collect_naip_sources(naip_candidates) if naip_candidates else []
    if naip_candidates and not naip_sources:
        inputs = ", ".join(str(path) for path in naip_candidates)
        raise FileNotFoundError(f"No NAIP rasters found for --naip-path inputs: {inputs}")
    geom = parse_aoi(aoi)
    client = Client.open(stac_url)

    # Extract polygons BEFORE NAIP download to enable per-polygon downloading
    # This avoids downloading unnecessary tiles in gaps between spatially separated AOIs
    polygons = extract_aoi_polygons(geom)
    if not polygons:
        raise ValueError("No valid AOI polygons extracted.")

    if auto_download_naip:
        if naip_sources:
            logging.info("NAIP sources already provided; skipping auto download.")
        else:
            # Download NAIP per-polygon for efficiency (avoids downloading tiles in gaps)
            naip_output_dir = output_dir / "naip_auto"
            all_downloaded: List[Path] = []

            for idx, polygon in enumerate(polygons, start=1):
                LOGGER.info("Downloading NAIP for polygon %d/%d", idx, len(polygons))
                naip_request = NaipDownloadRequest(
                    aoi=polygon,  # Per-polygon, not union
                    output_dir=naip_output_dir,  # Shared dir for deduplication
                    year=auto_download_naip_year,
                    max_items=auto_download_naip_max_items,
                    overwrite=auto_download_naip_overwrite,
                    preview=auto_download_naip_preview,
                    target_resolution=naip_target_resolution,
                )
                downloaded = NaipService().download(naip_request)
                all_downloaded.extend(downloaded)

            # Deduplicate paths (geoai deduplicates by filename, but paths may repeat)
            naip_sources = _deduplicate_paths(all_downloaded)
            LOGGER.info(
                "Downloaded %d unique NAIP tiles for %d polygons",
                len(naip_sources), len(polygons)
            )

    has_naip = bool(naip_sources)

    if auto_download_wetlands:
        if not has_naip:
            raise ValueError(
                "Auto downloading wetlands requires NAIP tiles (either existing or downloaded)."
            )
        union_extent = compute_naip_union_extent(naip_sources)
        if union_extent is None:
            raise RuntimeError("Unable to determine union extent from NAIP tiles.")
        target_path = wetlands_output_path or (output_dir / "wetlands_auto" / "wetlands.gpkg")
        
        # Use WetlandsService
        wetlands_request = WetlandsDownloadRequest(
            aoi=union_extent,
            output_path=target_path,
            overwrite=wetlands_overwrite,
        )
        wetlands_path = WetlandsService().download(wetlands_request)
        logging.info("Wetlands delineations saved to %s", wetlands_path)

    # polygons already extracted before NAIP download (line 221)

    # Resolve chunk size for dask processing
    from .stac_client import compute_chunk_size

    resolved_chunk_size: Optional[int] = None
    if chunk_size == "auto":
        # Compute based on first polygon's bounds (AOIs should be similar size)
        if polygons:
            resolved_chunk_size = compute_chunk_size(polygons[0].bounds)
            LOGGER.info("Auto-computed chunk size: %d pixels", resolved_chunk_size)
    elif chunk_size is not None and chunk_size.lower() != "none":
        try:
            resolved_chunk_size = int(chunk_size)
            LOGGER.info("Using specified chunk size: %d pixels", resolved_chunk_size)
        except (ValueError, TypeError):
            LOGGER.warning("Invalid chunk_size '%s', using default (2048)", chunk_size)

    manifest_paths: List[Path] = []
    progress = RasterProgress(total=len(polygons) * (len(seasons) + int(has_naip)))

    naip_footprints: Optional[List[Tuple[Path, Polygon]]] = None
    if has_naip:
        try:
            naip_footprints = _collect_naip_footprints(tuple(sorted(naip_sources)))
        except ValueError as exc:
            raise RuntimeError(f"Failed to read NAIP footprints: {exc}") from exc

    for output in _iter_aoi_processing(
        client=client,
        polygons=polygons,
        years=years,
        seasons=seasons,
        cloud_cover=cloud_cover,
        min_clear_obs=min_clear_obs,
        mask_dilation=mask_dilation,
        tiles_per_season=output_dir,
        chunk_size=resolved_chunk_size,
        max_scenes_per_season=max_scenes_per_season,
        parallel_fetch=parallel_fetch,
        fetch_workers=fetch_workers,
        target_crs=target_crs,
    ):
        index = output.index
        aoi_dir = output.directory
        seasonal_data = output.seasonal_data
        seasonal_labels = output.seasonal_labels

        if len(seasonal_data) != len(seasons):
            LOGGER.warning("AOI %s: missing one or more seasons; skipping composite.", index)
            continue

        composite_label = f"AOI {index:02d} :: Seasonal stack"
        progress.start(composite_label)
        combined21, labels21 = concatenate_seasons(seasonal_data, seasons)
        combined21 = combined21.where(combined21 != FLOAT_NODATA, FLOAT_NODATA)
        combined21_path = aoi_dir / "s2_sprsumfal_median_21band.tif"
        write_dataarray(combined21, combined21_path, labels21)
        progress.finish(composite_label)

        if not has_naip:
            continue

        stack_label = f"AOI {index:02d} :: Stack manifest"
        progress.start(stack_label)
        filtered_naip_sources = naip_sources
        if naip_footprints is not None:
            filtered_naip_sources = [
                path
                for path, footprint in naip_footprints
                if footprint.intersects(output.polygon)
            ]
            if not filtered_naip_sources:
                LOGGER.warning(
                    "AOI %s: no NAIP tiles intersect polygon; skipping composite and manifest.",
                    index,
                )
                progress.finish(stack_label)
                continue
            LOGGER.info(
                "AOI %s: using %d NAIP tile(s) after footprint filtering.",
                index,
                len(filtered_naip_sources),
            )
        naip_reference_path, _, naip_band_labels = prepare_naip_reference(
            filtered_naip_sources,
            aoi_dir,
            target_resolution=naip_target_resolution,
            target_crs=target_crs,
        )

        clipped_naip_path = aoi_dir / "naip_clipped.tif"
        reference_profile = _clip_raster_to_polygon(naip_reference_path, output.polygon, clipped_naip_path)

        extra_sources = None
        if auto_download_topography:
            topo_output_dir = aoi_dir / "topography"
            topo_config = TopographyStackConfig(
                aoi=output.polygon,
                target_grid_path=clipped_naip_path,
                output_dir=topo_output_dir,
                buffer_meters=topography_buffer_meters,
                tpi_small_radius=topography_tpi_small,
                tpi_large_radius=topography_tpi_large,
                cache_dir=topography_cache_dir,
                dem_dir=topography_dem_dir,
                dem_resolution=dem_resolution,
            )
            topography_path = prepare_topography_stack(topo_config)
            extra_sources = [
                {
                    "type": "topography",
                    "path": str(topography_path.resolve()),
                    # Only relative topographic features - no raw elevation
                    # Raw elevation is excluded because wetlands exist at all elevations
                    # and absolute elevation creates geographic bias that hurts generalization
                    "band_labels": [
                        "Slope",
                        "TPI_small",
                        "TPI_large",
                        "DepressionDepth",
                    ],
                    # Per-band scaling to normalize topography values to [0, 1]
                    "band_scaling": {
                        "Slope": [0.0, 90.0],           # degrees (0=flat, 90=vertical)
                        "TPI_small": [-50.0, 50.0],     # meters (negative=depression)
                        "TPI_large": [-100.0, 100.0],   # meters (negative=depression)
                        "DepressionDepth": [0.0, 50.0], # meters (depth of local sinks)
                    },
                    "resample": "bilinear",
                    "nodata": FLOAT_NODATA,
                    "description": topo_config.description,
                }
            ]

        manifest_path = write_stack_manifest(
            output_dir=aoi_dir,
            naip_path=clipped_naip_path,
            naip_labels=naip_band_labels,
            sentinel_path=combined21_path,
            sentinel_labels=labels21,
            reference_profile=reference_profile,
            extra_sources=extra_sources,
        )
        manifest_paths.append(manifest_path)
        progress.finish(stack_label)
        LOGGER.info("AOI %s: manifest written -> %s", index, manifest_path)

    if manifest_paths:
        index_path = write_manifest_index(manifest_paths, output_dir)
        LOGGER.info("Manifest index created -> %s", index_path)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--aoi",
        help="AOI path or geometry (GeoJSON, WKT, or bbox). Required unless --labels-path is provided.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        help=(
            "Path to wetland labels (e.g., Atwell delineations GeoPackage). "
            "If provided, an AOI will be auto-generated from the labels bounding box + buffer. "
            "Use this instead of --aoi to derive the processing extent from your training data."
        ),
    )
    parser.add_argument(
        "--labels-buffer",
        type=float,
        default=1000.0,
        help="Buffer distance in meters around labels when auto-generating AOI (default: 1000m).",
    )
    parser.add_argument("--years", required=True, nargs="+", type=int, help="Years to include (e.g., 2022 2023)")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated GeoTIFFs",
    )
    parser.add_argument(
        "--cloud-cover",
        type=float,
        default=60.0,
        help="Maximum eo:cloud_cover percentage for Sentinel-2 items",
    )
    parser.add_argument(
        "--min-clear-obs",
        type=int,
        default=3,
        help="Minimum number of clear observations per pixel per season",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="Season codes to process (subset of SPR SUM FAL)",
    )
    parser.add_argument(
        "--stac-url",
        default="https://earth-search.aws.element84.com/v1",
        help="STAC API endpoint for Sentinel-2 L2A",
    )
    parser.add_argument(
        "--naip-path",
        type=Path,
        nargs="+",
        help=(
            "Optional NAIP GeoTIFF(s) or directories to define the target grid. "
            "Multiple rasters will be mosaicked automatically."
        ),
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=0,
        help="Number of pixels to dilate the SCL cloud mask",
    )
    parser.add_argument(
        "--chunk-size",
        default="auto",
        help=(
            "Dask chunk size for Sentinel-2 processing. Options: "
            "'auto' (compute based on AOI size, recommended), integer value "
            "(e.g., 512), or 'none' to disable chunking. Default: 'auto'. "
            "Smaller chunks improve parallelization but increase overhead."
        ),
    )
    parser.add_argument(
        "--max-scenes-per-season",
        type=int,
        default=None,
        help=(
            "Maximum number of scenes to use per season, selecting the N clearest "
            "by cloud cover. Reduces processing time significantly. "
            "Default: None (use all scenes). Recommended: 20 for faster processing."
        ),
    )
    parser.add_argument(
        "--parallel-fetch",
        action="store_true",
        help=(
            "Use parallel pre-download strategy instead of stackstac lazy loading. "
            "Downloads all scene data in parallel before computing median, providing "
            "5-10x speedup. Recommended for most use cases."
        ),
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=24,
        help=(
            "Number of parallel workers for scene downloads when --parallel-fetch "
            "is enabled. Default: 24. Adjust based on network bandwidth and CPU cores."
        ),
    )
    parser.add_argument(
        "--auto-download-naip",
        action="store_true",
        help="Automatically download NAIP tiles for the AOI if --naip-path is not provided.",
    )
    parser.add_argument(
        "--auto-download-naip-year",
        type=int,
        help="NAIP imagery year to request when auto downloading.",
    )
    parser.add_argument(
        "--auto-download-naip-max-items",
        type=int,
        help="Maximum number of NAIP tiles to fetch when auto downloading.",
    )
    parser.add_argument(
        "--auto-download-naip-overwrite",
        action="store_true",
        help="Allow overwriting existing auto-downloaded NAIP tiles.",
    )
    parser.add_argument(
        "--auto-download-naip-preview",
        action="store_true",
        help="Use preview mode when downloading NAIP (quicklook tiles).",
    )
    parser.add_argument(
        "--auto-download-wetlands",
        action="store_true",
        help="Automatically download wetlands delineations covering the NAIP union extent.",
    )
    parser.add_argument(
        "--wetlands-output-path",
        type=Path,
        help="Destination path for downloaded wetlands delineations (GeoPackage).",
    )
    parser.add_argument(
        "--wetlands-overwrite",
        action="store_true",
        help="Allow overwriting existing wetlands download outputs.",
    )
    parser.add_argument(
        "--naip-target-resolution",
        type=float,
        help="Optional NAIP resampling resolution in meters.",
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:5070",
        help=(
            "Target CRS for all output rasters. EPSG:5070 (NAD83 Conus Albers) is "
            "recommended for continental US coverage to avoid UTM zone boundary issues. "
            "Default: EPSG:5070"
        ),
    )
    parser.add_argument(
        "--auto-download-topography",
        action="store_true",
        help="Automatically derive LiDAR-based topographic features for the stack.",
    )
    parser.add_argument(
        "--topography-buffer-meters",
        type=float,
        default=200.0,
        help="Buffer distance (meters) applied to the AOI before DEM download.",
    )
    parser.add_argument(
        "--topography-tpi-small",
        type=float,
        default=30.0,
        help="Radius in meters for the small-scale TPI.",
    )
    parser.add_argument(
        "--topography-tpi-large",
        type=float,
        default=150.0,
        help="Radius in meters for the large-scale TPI.",
    )
    parser.add_argument(
        "--topography-cache-dir",
        type=Path,
        help="Optional directory for caching raw DEM downloads.",
    )
    parser.add_argument(
        "--topography-dem-dir",
        type=Path,
        help="Directory of pre-downloaded DEM GeoTIFF tiles to use instead of fetching from 3DEP.",
    )
    parser.add_argument(
        "--dem-resolution",
        type=str,
        choices=["1m", "10m", "30m"],
        default=None,
        help=(
            "DEM resolution preference for topography downloads. "
            "Options: '1m' (default, highest detail), '10m' (faster, good for 10m stacks), "
            "'30m' (fastest, coarse). If not set, defaults to 1m."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )


def run_from_args(args: argparse.Namespace) -> None:
    # Determine AOI: either from --aoi directly or auto-generated from --labels-path
    aoi = args.aoi
    if args.labels_path:
        if not args.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
        LOGGER.info(
            "Generating AOI from labels: %s (buffer: %.0fm)",
            args.labels_path,
            args.labels_buffer,
        )
        generated_aoi_path = create_acquisition_aoi(
            labels_path=args.labels_path,
            output_path=args.output_dir / "generated_aoi.gpkg",
            buffer_meters=args.labels_buffer,
        )
        aoi = str(generated_aoi_path)
        LOGGER.info("Generated AOI saved to: %s", aoi)
    elif not aoi:
        raise ValueError("Either --aoi or --labels-path must be provided.")

    run_pipeline(
        aoi=aoi,
        years=args.years,
        output_dir=args.output_dir,
        seasons=tuple(args.seasons),
        cloud_cover=args.cloud_cover,
        min_clear_obs=args.min_clear_obs,
        stac_url=args.stac_url,
        naip_paths=tuple(args.naip_path) if args.naip_path else None,
        auto_download_naip=args.auto_download_naip,
        auto_download_naip_year=args.auto_download_naip_year,
        auto_download_naip_max_items=args.auto_download_naip_max_items,
        auto_download_naip_overwrite=args.auto_download_naip_overwrite,
        auto_download_naip_preview=args.auto_download_naip_preview,
        auto_download_wetlands=args.auto_download_wetlands,
        wetlands_output_path=args.wetlands_output_path,
        wetlands_overwrite=args.wetlands_overwrite,
        naip_target_resolution=args.naip_target_resolution,
        target_crs=args.target_crs,
        mask_dilation=args.mask_dilation,
        auto_download_topography=args.auto_download_topography,
        topography_cache_dir=args.topography_cache_dir,
        topography_dem_dir=args.topography_dem_dir,
        topography_buffer_meters=args.topography_buffer_meters,
        topography_tpi_small=args.topography_tpi_small,
        topography_tpi_large=args.topography_tpi_large,
        dem_resolution=args.dem_resolution,
        chunk_size=args.chunk_size,
        max_scenes_per_season=args.max_scenes_per_season,
        parallel_fetch=args.parallel_fetch,
        fetch_workers=args.fetch_workers,
    )


__all__ = [
    "configure_parser",
    "run_from_args",
    "run_pipeline",
    "collect_naip_sources",
    "prepare_naip_reference",
]
