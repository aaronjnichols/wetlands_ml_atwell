"""Sentinel-2 processing subpackage."""

from .aoi import parse_aoi, buffer_in_meters, extract_aoi_polygons
from .cli import build_parser, main
from .cloud_masking import SCL_ASSET_ID, SCL_MASK_VALUES, stack_scl, build_mask
from .manifests import write_stack_manifest
from .naip import collect_naip_sources, prepare_naip_reference
from .seasonal import seasonal_median, concatenate_seasons, write_dataarray
from .stac_client import (
    SENTINEL_COLLECTION,
    SENTINEL_BANDS,
    SENTINEL_SCALE_FACTOR,
    DEFAULT_SEASONS,
    SEASON_WINDOWS,
    SeasonConfig,
    season_date_range,
    fetch_items,
    stack_bands,
)

__all__ = [
    # AOI parsing
    "parse_aoi",
    "buffer_in_meters",
    "extract_aoi_polygons",
    # Cloud masking
    "SCL_ASSET_ID",
    "SCL_MASK_VALUES",
    "stack_scl",
    "build_mask",
    # NAIP integration
    "collect_naip_sources",
    "prepare_naip_reference",
    # Seasonal compositing
    "seasonal_median",
    "concatenate_seasons",
    "write_dataarray",
    # STAC client
    "SENTINEL_COLLECTION",
    "SENTINEL_BANDS",
    "SENTINEL_SCALE_FACTOR",
    "DEFAULT_SEASONS",
    "SEASON_WINDOWS",
    "SeasonConfig",
    "season_date_range",
    "fetch_items",
    "stack_bands",
    # CLI
    "build_parser",
    "main",
    # Manifest
    "write_stack_manifest",
]

