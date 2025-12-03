## Executive Summary
Wetlands ML GeoAI’s core flows (Sentinel-2 compositing, NAIP/topography stacking, UNet training, and inference) are functionally rich but structurally monolithic. The Sentinel-2 pipeline couples AOI parsing, STAC querying, NAIP mosaicking, wetlands/topography downloads, and manifest generation inside a single 800+ line function, while training/inference CLIs each re‑implement large argument parsers and orchestration logic. Supporting utilities such as `tools/naip_download.py` bypass shared services, and automated tests only cover a few CLI sanity checks. This plan proposes a phased refactor that extracts cohesive services, introduces explicit configuration objects, unifies data acquisition, and raises test coverage without disrupting ongoing modeling work.

## Current State Analysis

### Sentinel-2 Seasonal Pipeline
`src/wetlands_ml_atwell/sentinel2/compositing.py` hosts most of the orchestration logic, including CLI behavior, AOI parsing, STAC calls, NAIP/topography downloads, manifest creation, and progress tracking inside `run_pipeline`.

```
622:810:src/wetlands_ml_atwell/sentinel2/compositing.py
def run_pipeline(
    aoi: str,
    years: Sequence[int],
    output_dir: Path,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    ...
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
```

This single function owns dependency discovery (`geoai`, `stackstac`, `pystac_client`, `geopandas`, NAIP downloads, wetlands service calls, and DEM pipelines), making it hard to unit test or selectively reuse pieces (e.g., only computing Sentinel composites without manifests).

### Training & Inference Workflows
Argument parsing, configuration, tiling, manifest resolution, and geoai integration live in `src/wetlands_ml_atwell/train_unet.py` and `src/wetlands_ml_atwell/training/unet.py`. The CLI defines dozens of arguments and environment variable fallbacks inline.

```
149:342:src/wetlands_ml_atwell/train_unet.py
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tiles and train a wetlands UNet semantic segmentation model."
    )
    parser.add_argument("--train-raster", default=os.getenv("TRAIN_RASTER_PATH"), ...)
    parser.add_argument("--stack-manifest", default=os.getenv("TRAIN_STACK_MANIFEST"), ...)
    parser.add_argument("--labels", default=os.getenv("TRAIN_LABELS_PATH"), ...)
    # ~35 additional options with duplicated env-var fallbacks
    ...
    if not args.train_raster and not args.stack_manifest:
        parser.error(
            "Provide --train-raster or --stack-manifest (or set TRAIN_RASTER_PATH / TRAIN_STACK_MANIFEST)."
        )
```

Tile export, manifest rewriting, staging-directory cleanup, and geoai invocation then execute imperatively within `training/unet.py`.

```
191:340:src/wetlands_ml_atwell/training/unet.py
if manifests:
    for idx, manifest in enumerate(manifests):
        ...
        geoai.export_geotiff_tiles(...)
        rewritten = rewrite_tile_images(manifest, staging_images)
        for image_path in image_paths:
            dest_name = f"aoi{idx:02d}_{image_path.name}"
            shutil.move(str(image_path), images_dir / dest_name)
            ...
else:
    geoai.export_geotiff_tiles(...)
...
geoai.train_segmentation_model(
    images_dir=str(images_dir),
    labels_dir=str(labels_dir),
    output_dir=str(models_dir),
    ...
)
```

Inference follows a similar pattern (`test_unet.py` plus `inference/unet_stream.py`) with separate argument parsing and orchestration logic.

### Data Acquisition & Utilities
While `data_acquisition.py` already exposes `_download_naip_tiles`, the checked-in helper script rewrites download logic with hard-coded paths, duplicating behavior and risking drift.

```
5:32:tools/naip_download.py
# user inputs
output_dir = r"data\IL\naip_tiles"
aoi_gpkg = r"C:\_code\python\wetlands_ml_atwell\data\IL\test_aoi.gpkg"
...
naip_paths = download_naip(
    aoi_bbox,
    output_dir=output_dir,
    year=year,
    max_items=max_items,
    overwrite=False,
    preview=False,
)
```

### Testing Footprint
Automated tests exercise only CLI argument validation and a couple of helper functions. They do not cover Sentinel-2 compositing, data acquisition, stacking, or end-to-end training/inference flows.

```
11:47:tests/test_inference_cli.py
def test_inference_cli_requires_inputs():
    with pytest.raises(SystemExit):
        test_unet.parse_args([])
...
def test_prepare_window_matches_training_normalization():
    raw = np.array([...], dtype=np.float32)
    expected = normalize_stack_array(...)
    actual = _prepare_window(raw, desired_channels=4, nodata_value=FLOAT_NODATA)
    np.testing.assert_allclose(actual, expected)
```

## Identified Issues and Opportunities

### Critical
- **Structural · Monolithic Sentinel-2 orchestration** – `run_pipeline` blends CLI validation, AOI parsing, STAC client life cycle, NAIP/wetlands/topography downloads, Dask computations, and manifest persistence in one function. Any change risks regressions across the entire pipeline, and unit testing is impractical because there are no seam points for dependency injection. This violates SRP and hinders reuse.
- **Structural · Lack of shared configuration/DI for training & inference** – Both CLIs maintain duplicated argument parsing, environmental overrides, and file-system orchestration. There is no shared `Config` object or service layer, leading to tight coupling with the `geoai` package and making it hard to substitute components or run headless tests.

### Major
- **Structural · Imperative tiling/rewriting pipeline** – `training/unet.py` manually copies tiles between staging directories, rewrites them via manifests, and cleans up directories with bespoke code. Failures mid-run leave partially moved files, and the process isn’t stream-friendly or resumable.
- **Behavioral · Divergent NAIP tooling** – The standalone `tools/naip_download.py` script bypasses `_download_naip_tiles`, hard-codes local paths, and shares no validation with `sentinel2.compositing`, increasing the chance of inconsistent preprocessing.
- **Quality · Sparse automated tests** – Current tests only ensure argument parsing works and a single window-normalization helper matches expectations. Core geospatial math, manifest generation, and CLI flows have no regression coverage, so large refactors are risky.

### Minor
- **Naming/Consistency · CLI shims** – Root-level `sentinel2_processing.py`, `train_unet.py`, and `test_unet.py` replicate identical sys.path bootstrapping. A shared entrypoint strategy (setuptools console_scripts or a thin `python -m` guidance) would reduce boilerplate.
- **Performance · Repeated raster opens** – `RasterStack.read_window` and `_collect_naip_footprints` repeatedly open datasets without caching footprints or using context pools, which becomes noticeable on AOIs with dozens of tiles.

## Proposed Refactoring Plan

### Phase 0 – Baseline Observability (0.5 sprint, Medium)
- **Goal**: Capture current behavior and guard against regressions before code movement.
- **Key steps**: 
  - Instrument `sentinel2.compositing` to emit structured events (timings, per-AOI stats) behind a feature flag.
  - Add integration smoke tests that mock STAC/NAIP/topography dependencies to assert CLI wiring succeeds.
  - Introduce `nox`/`tox` targets for `pytest` and type checking to establish repeatable CI gates.
- **Intermediate state**: No functional changes; metrics emitted only when `WETLANDS_DIAGNOSTICS=1`.
- **Acceptance criteria**: New tests run in CI; baseline run logs persisted for future comparison.
- **Rollback**: Disable flag to return to current logging.

### Phase 1 – Modular Sentinel-2 Pipeline (1.5–2 sprints, High)
- **Goal**: Break `run_pipeline` into cohesive services with dependency injection.
- **Key steps**:
  - Extract AOI parsing/buffering into `sentinel2/aoi.py` (pure functions).
  - Introduce `StacCompositeBuilder` (handles `Client` interactions, `stackstac` usage) and `SeasonalMixer` (cloud masking + median).
  - Move NAIP/topography integration into a `StackAssembler` class responsible only for merging precomputed rasters and writing manifests.
  - Isolate download orchestration into strategies that accept a session or file service, enabling mocks.
  - Replace global functions with `@dataclass` configs (e.g., `SentinelPipelineConfig`) consumed by a short `run_pipeline(config, services)` entrypoint.
- **Intermediate states**: 
  - First milestone: `sentinel2.cli` builds a config and calls the new orchestrator while the old helper functions wrap the new services.
  - Second milestone: remove unused legacy helpers once tests pass.
- **Acceptance criteria**: Unit tests cover AOI parsing and STAC fetching; `sentinel2_processing.py` still produces identical manifests on a regression dataset.
- **Rollback**: Keep legacy `run_pipeline` behind a feature flag until new path deemed stable.

### Phase 2 – Shared Data Acquisition & Manifest Services (1 sprint, Medium)
- **Goal**: Centralize NAIP/wetlands/topography download logic and manifest writing.
- **Key steps**:
  - Promote `_download_naip_tiles`/`_download_wetlands_delineations` to public services with explicit interfaces and retry/backoff policies.
  - Replace `tools/naip_download.py` with a CLI thin wrapper that imports the shared service (or add `python -m wetlands_ml_atwell.tools.naip`).
  - Create a `manifest` service that encapsulates NAIP footprint filtering and manifest index writing, enabling reuse by training and inference flows.
  - Add filesystem abstraction hooks (e.g., `PathService`) for easier testing and remote storage support.
- **Intermediate state**: Keep existing internal function names but re-export from a new module to avoid breaking imports, then update all callsites (`sentinel2`, `train_unet`, tests) to use the shared service before removing leading underscores.
- **Acceptance criteria**: CLI script delegates entirely to the shared service; unit tests cover NAIP download parameter validation and manifest serialization.
- **Rollback**: Keep the legacy tool script (hidden) until new CLI validated.

### Phase 3 – Training & Inference Workflow Convergence (1.5 sprints, High)
- **Goal**: Introduce shared configuration and service layers for training/inference to reduce duplication and ease testing.
- **Key steps**:
  - Define `TrainingConfig`/`InferenceConfig` dataclasses capturing validated inputs (raster paths, manifests, tiling params, model metadata).
  - Extract argument parsing into reusable builders (e.g., `cli/build_common_training_parser()`), reducing drift between root and package CLIs.
  - Wrap tiling + manifest rewriting inside a `TileExporter` service that can stream AOIs and surfaces progress/rollback hooks; use context managers for staging directories to ensure cleanup.
  - Abstract geoai interactions behind an adapter (e.g., `GeoaiTrainer`, `GeoaiInferencer`) so we can stub them in tests and evolve dependencies gradually.
  - Mirror the pattern for inference, allowing manifest-backed and raster-backed paths to share windowing logic.
- **Intermediate state**: New services co-exist while CLIs still call them via thin compatibility wrappers; once stable, drop the older procedural blocks.
- **Acceptance criteria**: 
  - CLI usage remains unchanged for end users.
  - New unit tests cover config validation, manifest resolution, and tiling flows (with filesystem fixtures).
  - At least one end-to-end dry run (with mocked I/O) verifies training pipeline wiring.
- **Rollback**: Keep old functions under `_legacy` namespace for a release cycle.

### Phase 4 – Extended Testing & Documentation (0.5–1 sprint, Medium)
- **Goal**: Solidify confidence in the refactor and document new architecture.
- **Key steps**:
  - Add scenario tests for `StackAssembler`, `TileExporter`, and `infer_manifest` using synthetic rasters.
  - Document the new module boundaries in `docs/pipeline_overview.md` and create developer guides for plugging in new data sources.
  - Establish regression fixtures (small AOIs, mock STAC responses) for deterministic CI.
- **Acceptance criteria**: 
  - Test suite covers >60% of `sentinel2`, `stacking`, `training`, and `inference` modules.
  - Updated documentation explains how to extend download services and run CI tasks locally.
  - `docs/repository_guidelines.md` references the new structure.
- **Effort/Complexity**: Moderate; depends on availability of sample rasters.
- **Rollback**: N/A (documentation/test additions are additive).

## Risk Assessment and Mitigation
- **External service variability**: STAC and USGS APIs can throttle or change schemas. Mitigate by introducing request adapters with retries and by recording contract tests against canned JSON.
- **Large file handling**: Refactors must avoid re-writing entire rasters unnecessarily. Use temporary directories and atomic moves; add disk-space checks before mosaicking.
- **geoai dependency changes**: Wrapping geoai calls behind adapters allows version upgrades or replacements without touching CLI layers; maintain compatibility shims until new adapters are battle-tested.
- **Team bandwidth**: High-effort phases overlap; sequence work so that Phase 1 completes before Phase 3 to avoid rework.
- **Rollback strategy**: Maintain feature flags (e.g., `WETLANDS_LEGACY_PIPELINES`) so operators can switch back during the next release cycle if regressions appear.

## Testing Strategy
- **Unit tests**: Cover AOI parsing, STAC query builders, manifest serialization, tile exporter edge cases, and inference window aggregation.
- **Contract tests**: Record/replay minimal STAC and TNM responses to ensure request payloads remain valid.
- **Integration tests**: Use in-memory rasters (small GeoTIFFs) to run sentinel-to-manifest-to-tiling pipelines end to end with mocked downloads.
- **Performance smoke tests**: Add benchmarks (even informal) for `RasterStack.read_window` and the new services to ensure refactors do not regress throughput.
- **Tooling**: Enforce `pytest -m "not slow"` in CI and leave slow tests for nightly builds; add coverage thresholds for touched modules.

## Success Metrics
- Sentinel-2 module split reduces maximum function length below 120 lines and cyclomatic complexity by >40%.
- Shared configuration/services eliminate duplicated argument definitions, reducing CLI LOC by ~30%.
- Automated test coverage for `sentinel2`, `stacking`, `training`, `inference`, and `data_acquisition` exceeds 60%, and critical paths have deterministic fixtures.
- End-to-end sample pipeline completes within ±5% of current runtime, demonstrating no performance regressions.
- Documentation (`docs/pipeline_overview.md`, new refactor plan) reflects the updated architecture and is referenced in contributor onboarding.


