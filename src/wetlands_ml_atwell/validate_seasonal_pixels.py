#!/usr/bin/env python3
"""Compare seasonal median pixels against raw Sentinel-2 observations."""
import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import rasterio
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box
from pystac_client import Client

from .sentinel2_processing import (
    SENTINEL_ASSET_MAP,
    SENTINEL_BANDS,
    SENTINEL_SCALE_FACTOR,
    fetch_items,
)

PIXEL_ALIASES: Dict[str, str] = {
    "ll": "lower_left",
    "lr": "lower_right",
    "ul": "upper_left",
    "ur": "upper_right",
}


@dataclass(frozen=True)
class PixelInfo:
    name: str
    row: int
    col: int
    lon: float
    lat: float
    median_values: Dict[str, float]
    band_labels: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the four corners (or custom pixels) of a Sentinel-2 seasonal median raster "
            "and collect every raw observation that fed the composite via the STAC API."
        )
    )
    parser.add_argument("--season-raster", required=True, type=Path, help="Seasonal median GeoTIFF to inspect.")
    parser.add_argument("--season-label", required=True, help="Season code used during compositing (e.g., SPR).")
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        type=int,
        help="Years that were included in the composite (e.g., 2022 2023).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Destination CSV for sampled pixel values.",
    )
    parser.add_argument(
        "--pixels",
        nargs="+",
        default=["lower_left", "lower_right", "upper_left", "upper_right"],
        help="Pixel names to sample (supports lower_left/lower_right/upper_left/upper_right or ll/lr/ul/ur).",
    )
    parser.add_argument(
        "--stac-url",
        default="https://earth-search.aws.element84.com/v1",
        help="STAC API endpoint (defaults to Element84 Earth Search).",
    )
    parser.add_argument(
        "--cloud-cover",
        type=float,
        default=60.0,
        help="Maximum eo:cloud_cover threshold used during the composite search.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def resolve_pixel(name: str, width: int, height: int) -> tuple[int, int]:
    alias = PIXEL_ALIASES.get(name.lower(), name.lower())
    mapping = {
        "lower_left": (height - 1, 0),
        "lower_right": (height - 1, width - 1),
        "upper_left": (0, 0),
        "upper_right": (0, width - 1),
    }
    if alias not in mapping:
        raise ValueError(f"Unsupported pixel name '{name}'.")
    return mapping[alias]


def gather_pixel_info(dataset: rasterio.io.DatasetReader, pixel_names: Iterable[str]) -> List[PixelInfo]:
    band_labels = [desc if desc else f"band_{idx}" for idx, desc in enumerate(dataset.descriptions, start=1)]
    base_bands = SENTINEL_BANDS[: dataset.count]
    infos: List[PixelInfo] = []
    for name in pixel_names:
        row, col = resolve_pixel(name, dataset.width, dataset.height)
        window = Window(col_off=col, row_off=row, width=1, height=1)
        data = dataset.read(window=window)
        if data.shape[1] == 0 or data.shape[2] == 0:
            continue
        median_values: Dict[str, float] = {}
        band_label_lookup: Dict[str, str] = {}
        for idx, band_code in enumerate(base_bands):
            value = float(data[idx, 0, 0])
            median_values[band_code] = value
            band_label_lookup[band_code] = band_labels[idx]
        x, y = dataset.xy(row, col, offset="center")
        lon, lat = transform(dataset.crs, "EPSG:4326", [x], [y])
        infos.append(
            PixelInfo(
                name=name,
                row=row,
                col=col,
                lon=float(lon[0]),
                lat=float(lat[0]),
                median_values=median_values,
                band_labels=band_label_lookup,
            )
        )
    return infos


def format_years(years: Sequence[int]) -> str:
    return ",".join(str(year) for year in years)


def append_median_rows(
    records: List[Dict[str, object]],
    raster_path: Path,
    season_label: str,
    years: Sequence[int],
    pixel_infos: Sequence[PixelInfo],
) -> None:
    years_str = format_years(years)
    for info in pixel_infos:
        for band_code, median_value in info.median_values.items():
            records.append(
                {
                    "season_raster": str(raster_path),
                    "season_label": season_label,
                    "years": years_str,
                    "pixel_name": info.name,
                    "band": band_code,
                    "band_label": info.band_labels.get(band_code, band_code),
                    "source_type": "median",
                    "item_id": "",
                    "observation_datetime": "",
                    "asset_id": "",
                    "raw_value": "",
                    "scaled_value": median_value,
                    "median_value": median_value,
                    "lon": info.lon,
                    "lat": info.lat,
                    "row": info.row,
                    "col": info.col,
                }
            )


def observe_items(
    records: List[Dict[str, object]],
    raster_path: Path,
    season_label: str,
    years: Sequence[int],
    pixel_infos: Sequence[PixelInfo],
    items,
) -> None:
    if not items:
        logging.warning("No Sentinel-2 items found for comparison.")
        return

    years_str = format_years(years)
    lons = [info.lon for info in pixel_infos]
    lats = [info.lat for info in pixel_infos]

    with rasterio.Env(AWS_REQUEST_PAYER="requester"):
        for item in items:
            epsg = item.properties.get("proj:epsg")
            if epsg is None:
                logging.warning("Skipping item %s without proj:epsg metadata", item.id)
                continue
            try:
                xs, ys = transform("EPSG:4326", f"EPSG:{int(epsg)}", lons, lats)
            except Exception as exc:
                logging.warning("Coordinate transform failed for item %s: %s", item.id, exc)
                continue

            timestamp = ""
            if item.datetime:
                timestamp = item.datetime.isoformat()

            for band_code in SENTINEL_BANDS:
                if band_code not in pixel_infos[0].median_values:
                    continue
                asset_id = SENTINEL_ASSET_MAP.get(band_code)
                if not asset_id or asset_id not in item.assets:
                    logging.debug("Item %s missing asset %s", item.id, asset_id)
                    continue
                asset = item.assets[asset_id]
                href = asset.get_absolute_href()
                if href is None:
                    logging.debug("Item %s asset %s has no href", item.id, asset_id)
                    continue
                try:
                    with rasterio.open(href) as src:
                        nodata = src.nodata
                        samples = list(src.sample(zip(xs, ys)))
                except Exception as exc:
                    logging.warning("Failed to sample %s %s: %s", item.id, asset_id, exc)
                    continue

                for info, sample in zip(pixel_infos, samples):
                    if sample is None or len(sample) == 0:
                        value = None
                    else:
                        value = float(sample[0])
                        if nodata is not None and np.isfinite(nodata) and np.isclose(value, float(nodata)):
                            value = None
                    scaled_value = value * SENTINEL_SCALE_FACTOR if value is not None else None
                    median_value = info.median_values.get(band_code)
                    band_label = info.band_labels.get(band_code, band_code)
                    records.append(
                        {
                            "season_raster": str(raster_path),
                            "season_label": season_label,
                            "years": years_str,
                            "pixel_name": info.name,
                            "band": band_code,
                            "band_label": band_label,
                            "source_type": "observation",
                            "item_id": item.id,
                            "observation_datetime": timestamp,
                            "asset_id": asset_id,
                            "raw_value": value if value is not None else "",
                            "scaled_value": scaled_value if scaled_value is not None else "",
                            "median_value": median_value,
                            "lon": info.lon,
                            "lat": info.lat,
                            "row": info.row,
                            "col": info.col,
                        }
                    )


def write_csv(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        raise ValueError("No records were generated; check inputs and STAC filters.")
    fieldnames = [
        "season_raster",
        "season_label",
        "years",
        "pixel_name",
        "band",
        "band_label",
        "source_type",
        "item_id",
        "observation_datetime",
        "asset_id",
        "raw_value",
        "scaled_value",
        "median_value",
        "lon",
        "lat",
        "row",
        "col",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    raster_path = args.season_raster.expanduser().resolve()
    if not raster_path.exists():
        raise FileNotFoundError(f"Seasonal median raster not found: {raster_path}")

    season_label = args.season_label.strip().upper()
    years = tuple(sorted(set(args.years)))
    pixel_names = [name.strip() for name in args.pixels]

    with rasterio.open(raster_path) as dataset:
        pixel_infos = gather_pixel_info(dataset, pixel_names)
        if not pixel_infos:
            raise ValueError("No valid pixels sampled from the raster.")
        bounds_wgs84 = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21)
    aoi_geom = box(*bounds_wgs84)

    logging.info("Sampling %d pixels across %s bands", len(pixel_infos), len(pixel_infos[0].median_values))

    client = Client.open(args.stac_url)
    items = fetch_items(client, aoi_geom, season_label, years, args.cloud_cover)
    logging.info("Fetched %d Sentinel-2 items for season %s (%s)", len(items), season_label, format_years(years))

    records: List[Dict[str, object]] = []
    append_median_rows(records, raster_path, season_label, years, pixel_infos)
    observe_items(records, raster_path, season_label, years, pixel_infos, items)

    # Sort for readability
    records.sort(key=lambda row: (
        row["pixel_name"],
        row["band"],
        row["source_type"],
        row.get("observation_datetime", ""),
        row.get("item_id", ""),
    ))

    output_path = args.output_csv.expanduser().resolve()
    write_csv(output_path, records)
    logging.info("Wrote %d rows to %s", len(records), output_path)


if __name__ == "__main__":
    main()
