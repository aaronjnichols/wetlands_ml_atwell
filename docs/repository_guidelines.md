# Repository Guidelines

## Project Structure & Module Organization
The repository is intentionally lean. `train_unet.py` routes the tiling workflow into GeoAI's segmentation-models-pytorch trainers for UNet-style nets, while `test_unet.py` provides the semantic-segmentation inference path. `sentinel2_processing.py` houses the Sentinel-2 seasonal compositing CLI. Dependencies live in `requirements.txt`, and environment bootstrap steps are scripted in `setup.bat`. Replace hard-coded NAIP/Sentinel paths with environment variables (for example, `os.environ["NAIP_RASTER_DIR"]`) before sharing scripts or opening pull requests.

## Build, Test, and Development Commands
Run `setup.bat` from a Developer Command Prompt to create the `venv`. After activation (`venv\Scripts\activate.bat`), install packages with `pip install -r requirements.txt`. Use `python sentinel2_processing.py --aoi data\aoi.gpkg --years 2022 2023 --output-dir data\s2 --naip-path data\naip` to generate seasonal composites and the NAIP-aligned 25-band stack; the `--aoi` argument accepts WKT/GeoJSON strings, GeoJSON files, GeoPackages (.gpkg), and shapefiles (.shp). Semantic segmentation training lives under `python train_unet.py --train-raster ... --labels ... --architecture unet --encoder-name resnet34`. Run inference via `python test_unet.py --test-raster ... --model-path models\unet_best.pth`; alternatively set `TEST_RASTER_PATH`, `MODEL_PATH`, `PREDICTIONS_DIR`, and related environment variables.

When preparing inputs from scratch, supply `--auto-download-naip` so the CLI pulls NAIP tiles for the AOI (configure year and tile count with `--auto-download-naip-year` and `--auto-download-naip-max-items`). Pair it with `--auto-download-wetlands` to fetch wetlands delineations clipped to the union extent of the downloaded tiles; use `--wetlands-output-path` to control the GeoPackage destination.

## Coding Style & Naming Conventions
Code targets Python 3.8+. Follow PEP 8 with four-space indentation and `snake_case` for variables, functions, and tile names. Group constants (paths, hyperparameters) near the top of each script and gate environment-specific overrides inside `if __name__ == "__main__":`. Prefer `pathlib.Path` over raw strings for filesystem logic and keep GeoAI and Sentinel helper calls compact with short comments when logic is non-obvious.

## Testing Guidelines
Automated coverage is minimal today; use `python test_unet.py` as a smoke test before pushing. When adding new utilities, place pytest-compatible tests under a future `tests/` directory and aim for at least one unit or integration check per feature. Name files `test_<feature>.py`, avoid hard-coding private paths, and mock or parameterize raster locations.

## Commit & Pull Request Guidelines
Write commits in imperative present tense (for example, `train_unet: refactor tile export flow`). Each PR should describe dataset assumptions, attached artifacts (model checkpoints, sample outputs), and verification steps (`python test_unet.py` run logs, screenshots of vector overlays). Link related issues and call out breaking changes such as new raster requirements.

## Data & Security Considerations
Never commit proprietary rasters, geopackages, or credentials. Store large artifacts in secured cloud buckets and reference them via environment variables or README notes. Scrub absolute client paths before publishing notebooks, scripts, or composite outputs.
