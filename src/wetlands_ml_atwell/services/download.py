"""Data acquisition services for wetlands_ml_atwell.

This module unifies access to external data sources (NAIP, NWI Wetlands, USGS 3DEP).
"""

from __future__ import annotations

import logging
import os
import time
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

try:
    from geoai import download as geo_download
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "The geoai package is required for automatic data acquisition. "
        "Install geoai or disable the auto-download options."
    ) from exc

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass(frozen=True)
class NaipDownloadRequest:
    """Configuration for NAIP downloads."""

    aoi: BaseGeometry
    output_dir: Path
    year: Optional[int]
    max_items: Optional[int] = None
    overwrite: bool = False
    preview: bool = False
    target_resolution: Optional[float] = None


@dataclass(frozen=True)
class WetlandsDownloadRequest:
    """Configuration for wetlands delineation downloads."""

    aoi: BaseGeometry
    output_path: Path
    overwrite: bool = False


@dataclass(frozen=True)
class DemProduct:
    product_id: str
    download_url: str
    size: int
    bbox: List[float]
    last_updated: Optional[str]

    def filename(self) -> str:
        safe_id = (
            self.product_id.replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
        )
        return f"{safe_id}.tif"


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# NAIP Implementation
# =============================================================================

def _download_naip_tiles(request: NaipDownloadRequest) -> List[Path]:
    """Download NAIP tiles intersecting ``request.aoi`` and return file paths."""

    bbox = request.aoi.bounds
    output_dir = _ensure_directory(request.output_dir)
    LOGGER.info(
        "Downloading NAIP tiles (year=%s, max_items=%s) to %s",
        request.year,
        request.max_items,
        output_dir,
    )

    if not hasattr(geo_download, "download_naip"):
        raise AttributeError("geoai.download.download_naip is not available in this environment")

    naip_paths = geo_download.download_naip(  # type: ignore[attr-defined]
        bbox,
        output_dir=str(output_dir),
        year=request.year,
        max_items=request.max_items,
        overwrite=request.overwrite,
        preview=request.preview,
    )

    resolved = [Path(path).expanduser().resolve() for path in naip_paths]
    LOGGER.info("Fetched %s NAIP tile(s)", len(resolved))
    return resolved


# =============================================================================
# Wetlands Implementation
# =============================================================================

def _download_wetlands_delineations(request: WetlandsDownloadRequest) -> Path:
    """Download wetlands delineations for the provided ``request.aoi`` bounds."""

    output_path = request.output_path
    _ensure_directory(output_path.parent)

    # Check if file exists and handle overwrite
    if output_path.exists() and not request.overwrite:
        LOGGER.info("Wetlands file already exists: %s (skipping download)", output_path)
        return output_path

    LOGGER.info("Downloading wetlands delineations to %s", output_path)

    # Get bounds from the AOI geometry
    bounds = request.aoi.bounds  # (minx, miny, maxx, maxy)
    
    # Format geometry as comma-separated bbox for NWI API
    geometry_str = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"

    # NWI API endpoint
    nwi_url = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query"
    
    # Query parameters
    params = {
        "where": "1=1",
        "geometry": geometry_str,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson"
    }

    LOGGER.info("Querying NWI API for bounds: minx=%.6f, miny=%.6f, maxx=%.6f, maxy=%.6f",
                bounds[0], bounds[1], bounds[2], bounds[3])

    # Make the API request
    response = requests.get(nwi_url, params=params, timeout=120)
    response.raise_for_status()

    # Parse the GeoJSON response
    geojson_data = response.json()
    
    if "features" not in geojson_data or not geojson_data["features"]:
        LOGGER.warning("No wetlands features found for the specified area")
        # Create an empty GeoDataFrame with expected schema
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    else:
        # Convert GeoJSON to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")
        
        # Clip wetlands to the exact AOI extent (not just the bounding box)
        LOGGER.info("Clipping %d wetlands features to AOI extent", len(gdf))
        aoi_gdf = gpd.GeoDataFrame([{"geometry": request.aoi}], crs="EPSG:4326")
        gdf = gpd.clip(gdf, aoi_gdf)
        
        LOGGER.info("After clipping: %d wetlands features remain", len(gdf))

    # Save to file
    gdf.to_file(output_path, driver="GPKG")
    
    LOGGER.info("Saved %d wetlands features to %s", len(gdf), output_path)
    return output_path


# =============================================================================
# Topography Implementation
# =============================================================================

TNM_BASE_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEFAULT_DATASETS: Sequence[str] = (
    "Digital Elevation Model (DEM) 1 meter",
    "1-meter DEM",
    "3DEP Elevation: DEM (1 meter)",
    "Seamless 1-meter DEM (Limited Availability)",
    "DEM Source (OPR)",
    "National Elevation Dataset (NED) 1/9 arc-second",
    "National Elevation Dataset (NED) 1/3 arc-second",
    "National Elevation Dataset (NED) 1 arc-second",
)

# Dataset priorities by resolution preference
DATASETS_10M: Sequence[str] = (
    "National Elevation Dataset (NED) 1/3 arc-second",
    "National Elevation Dataset (NED) 1/9 arc-second",
    "National Elevation Dataset (NED) 1 arc-second",
)

DATASETS_30M: Sequence[str] = (
    "National Elevation Dataset (NED) 1 arc-second",
    "National Elevation Dataset (NED) 1/3 arc-second",
    "National Elevation Dataset (NED) 1/9 arc-second",
)

DEM_RESOLUTION_DATASETS = {
    "1m": DEFAULT_DATASETS,
    "10m": DATASETS_10M,
    "30m": DATASETS_30M,
}


def _build_query_params(
    bbox: Tuple[float, float, float, float],
    dataset: str,
    max_results: int,
) -> dict:
    return {
        "bbox": ",".join(str(v) for v in bbox),
        "datasets": dataset,
        "prodFormats": "GeoTIFF,IMG,TIF",
        "outputFormat": "JSON",
        "max": max_results,
    }


def _extract_primary_url(urls: Any) -> Optional[str]:
    if not urls:
        return None

    if isinstance(urls, list):
        for entry in urls:
            if isinstance(entry, dict):
                url = entry.get("url") or entry.get("URL")
                if url:
                    return str(url)
            elif isinstance(entry, str) and entry.startswith("http"):
                return entry

    if isinstance(urls, dict):
        for value in urls.values():
            if isinstance(value, dict):
                url = value.get("url") or value.get("URL")
                if url:
                    return str(url)
            elif isinstance(value, str) and value.startswith("http"):
                return value

    return None


def fetch_dem_inventory(
    aoi_geojson: dict,
    bbox: Tuple[float, float, float, float],
    datasets: Sequence[str] = DEFAULT_DATASETS,
    session: Optional[requests.Session] = None,
    max_results: int = 100,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> List[DemProduct]:
    """Return DEM products for ``bbox`` scanning preferred datasets."""

    client = session or requests.Session()
    headers = {
        "Accept": "application/json",
        "User-Agent": os.getenv(
            "USGS_USER_AGENT",
            "wetlands-ml-atwell/0.1 (https://github.com/aaronjnichols/wetlands_ml_atwell)",
        ),
    }
    api_key = os.getenv("USGS_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    for dataset in datasets:
        params = _build_query_params(bbox, dataset, max_results)
        attempt = 0
        while True:
            attempt += 1
            LOGGER.info(
                "Querying 3DEP dataset '%s' (attempt %s) for bbox=%s",
                dataset,
                attempt,
                bbox,
            )
            response = client.get(
                TNM_BASE_URL,
                params=params,
                timeout=180,
                headers=headers,
            )
            if response.status_code in {403, 429} and attempt <= retries:
                wait = backoff_seconds * attempt
                LOGGER.warning(
                    "Dataset '%s' request returned %s; retrying in %.1f s",
                    dataset,
                    response.status_code,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            break

        data = response.json()
        items = data.get("items", [])
        if not items:
            LOGGER.info("No products found for dataset '%s'", dataset)
            continue

        products: List[DemProduct] = []
        for item in items:
            primary_url = _extract_primary_url(item.get("urls"))
            if not primary_url:
                continue
            product_id = item.get("title") or item.get("entityId") or item.get("id", "dem_tile")
            products.append(
                DemProduct(
                    product_id=str(product_id),
                    download_url=primary_url,
                    size=int(item.get("sizeInBytes") or item.get("unitSize") or 0),
                    bbox=item.get("boundingBox", []),
                    last_updated=item.get("lastUpdated"),
                )
            )

        if products:
            LOGGER.info("Found %s product(s) using dataset '%s'", len(products), dataset)
            return products

    LOGGER.warning(
        "No DEM products found after scanning datasets: %s",
        ", ".join(datasets),
    )
    return []


def download_dem_products(products: Iterable[DemProduct], output_dir: Path) -> List[Path]:
    """Download DEM products to ``output_dir``; return resolved paths."""

    paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for product in products:
        target_path = output_dir / product.filename()
        if target_path.exists():
            LOGGER.info("DEM tile cached -> %s", target_path)
            paths.append(target_path.resolve())
            continue

        LOGGER.info("Downloading DEM tile %s -> %s", product.product_id, target_path)
        with session.get(product.download_url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with target_path.open("wb") as dst:
                for chunk in resp.iter_content(chunk_size=1_048_576):
                    if chunk:
                        dst.write(chunk)
        LOGGER.info("Download complete -> %s", target_path)
        paths.append(target_path.resolve())

    return paths


# =============================================================================
# Services
# =============================================================================

class NaipService:
    """Service for downloading NAIP imagery."""

    def download(self, request: NaipDownloadRequest) -> List[Path]:
        """Download NAIP tiles matching the request.
        
        Args:
            request: Configuration for the download (AOI, year, etc).
            
        Returns:
            List of paths to downloaded NAIP tiles.
        """
        return _download_naip_tiles(request)


class WetlandsService:
    """Service for downloading NWI wetlands data."""

    def download(self, request: WetlandsDownloadRequest) -> Path:
        """Download wetlands delineations.
        
        Args:
            request: Configuration for the download.
            
        Returns:
            Path to the downloaded GeoPackage.
        """
        return _download_wetlands_delineations(request)


class TopographyService:
    """Service for downloading USGS 3DEP DEM data."""

    def download(
        self,
        aoi_geometry: BaseGeometry,
        output_dir: Path,
        datasets: Optional[Sequence[str]] = None,
        resolution: Optional[str] = None,
    ) -> List[Path]:
        """Download DEM tiles covering the AOI.

        Args:
            aoi_geometry: Shapely geometry defining the area of interest.
            output_dir: Directory to save downloaded tiles.
            datasets: Optional list of dataset names to query (priority order).
            resolution: DEM resolution preference ('1m', '10m', '30m').
                If provided, overrides datasets parameter.

        Returns:
            List of paths to downloaded DEM tiles.
        """
        bbox = aoi_geometry.bounds

        LOGGER.info("TopographyService.download called with resolution=%r", resolution)

        # Resolution takes precedence over explicit datasets
        if resolution:
            if resolution not in DEM_RESOLUTION_DATASETS:
                raise ValueError(
                    f"Invalid DEM resolution '{resolution}'. "
                    f"Choose from: {list(DEM_RESOLUTION_DATASETS.keys())}"
                )
            datasets = DEM_RESOLUTION_DATASETS[resolution]
            LOGGER.info("Using %s DEM resolution (datasets: %s)", resolution, datasets[0])
        else:
            LOGGER.info("No resolution specified, using DEFAULT_DATASETS (1m priority)")

        products = fetch_dem_inventory(
            aoi_geojson=aoi_geometry.__geo_interface__ if hasattr(aoi_geometry, "__geo_interface__") else {},
            bbox=bbox,
            datasets=datasets if datasets else DEFAULT_DATASETS,
        )

        if not products:
            LOGGER.warning("No DEM products found for AOI")
            return []

        return download_dem_products(products, output_dir)


__all__ = [
    "NaipService",
    "WetlandsService",
    "TopographyService",
    "NaipDownloadRequest",
    "WetlandsDownloadRequest",
    "DemProduct",
    "DEM_RESOLUTION_DATASETS",
]
