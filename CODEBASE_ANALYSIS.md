# Wetlands ML Codex - Comprehensive Codebase Analysis

## Executive Summary

This codebase is a well-intentioned geospatial machine learning project for wetland detection, but it suffers from:

1. **Critical bugs** that break core functionality (broken imports)
2. **Architectural violations** (files 2-3x the intended size limit)
3. **Minimal test coverage** (<5% of total lines)
4. **Organizational debt** from recent refactoring (inconsistent naming, incomplete migrations)

**Estimated cleanup effort**: 5-10 days to achieve acceptable code quality

---

## Current State Overview

### Project Structure
- **Language**: Python 3.8+
- **Total Size**: 4,384 lines of code across 31 Python files
- **Main Package**: `src/wetlands_ml_geoai/` (1,608 lines)
- **Test Coverage**: ~170 lines of test code (3.8% of total)
- **Documentation**: 212 lines across 5 markdown files

### Architecture Pattern
- **Intended**: Package-based with thin CLI shims at root
- **Reality**: Mix of package and root-level convenience files
- **Strength**: Clear separation of concerns at high level (sentinel2, training, inference, topography)
- **Weakness**: Implementation files violate single-responsibility principle

---

## Critical Issues (Fix Immediately)

### 1. Broken Import - Blocks Validation Module
**File**: `src/wetlands_ml_geoai/validate_seasonal_pixels.py`, line 17

```python
# BROKEN ❌
from .sentinel2_processing import (
    SENTINEL_ASSET_MAP,
    SENTINEL_BANDS,
    SENTINEL_SCALE_FACTOR,
    fetch_items,
)

# ERROR: ModuleNotFoundError: No module named 'sentinel2_processing'
# (Was renamed to sentinel2/compositing.py during refactoring)
```

**Impact**: The entire validation module fails to load
**Fix Time**: 5 minutes
**Test**: Add import test to test suite

### 2. Empty Configuration File
**File**: `configs/train_unet_example.yaml` (0 bytes)

**Problem**: File exists but is completely empty - no reference for users
**Impact**: Users have no template for configuration
**Fix Time**: 15 minutes
**Solution**: Create realistic YAML example with documented parameters

---

## Major Structural Issues

### 3. File Size Violations (2-3x Architecture Limit)

The `.cursorrules` document specifies ~400 line limit for cohesive modules:

| File | Lines | Limit | Violation |
|------|-------|-------|-----------|
| sentinel2/compositing.py | 978 | 400 | 2.4x |
| training/unet.py | 344 | 400 | 0.9x (but dense) |
| validate_seasonal_pixels.py | 331 | 400 | 0.8x (but dense) |

**sentinel2/compositing.py** is the main concern:
- Lines 60-99: NAIP file utilities
- Lines 101-200: NAIP mosaic/reference prep
- Lines 200-320: AOI parsing, season config
- Lines 328-400: Sentinel-2 STAC queries
- Lines 600+: Full pipeline orchestration
- Lines 750+: Argument parsing (50+ parameters)
- Lines 850+: Main entry point with downloads

**Recommended splits**:
```
sentinel2/
  ├── compositing.py       → Keep orchestration only
  ├── aoi.py              → Parse AOI geometry
  ├── sources.py          → NAIP collection and prep
  ├── stac_query.py       → Sentinel-2 data fetching
  └── download.py         → NAIP/wetlands/topography acquisition
```

### 4. Incomplete/Incorrect Package Exports

**File**: `src/wetlands_ml_geoai/sentinel2/__init__.py`

```python
__all__ = [
    "build_parser",
    "main",
    "generate_sentinel_composites",  # ❌ NOT FOUND ANYWHERE
    "write_stack_manifest",          # ✓ Exists in manifests.py
]
```

**Issues**:
- Exports non-existent function `generate_sentinel_composites`
- Missing import: `from . import compositing`
- Missing import: `from .manifests import NAIP_BAND_LABELS`

**File**: `src/wetlands_ml_geoai/utils/__init__.py`
- Completely empty (0 bytes) - should contain utility exports or deleted if unused

**File**: `src/wetlands_ml_geoai/inference/`
- Missing `__init__.py` - not a proper Python package

### 5. Minimal Test Coverage

**Current state**:
- 3 test files
- ~170 lines of test code
- 3.8% coverage by line count

**Test inventory**:
- `test_training_cli.py`: 90 lines - CLI smoke tests only, no business logic
- `test_inference_cli.py`: 21 lines - 2 basic tests
- `test_topography_pipeline.py`: 60 lines - 1 real test + legacy comparison

**Missing**:
- Unit tests for utilities (AOI parsing, NAIP processing, STAC queries)
- Integration tests (multi-AOI workflows, end-to-end pipelines)
- Edge cases (empty inputs, invalid CRS, corrupted data)
- Regression tests for recent topography module additions
- No pytest fixtures or conftest.py

---

## Major Code Quality Issues

### 6. Inconsistent Documentation

**Missing docstrings**:
- `parse_target_size()` - No explanation of input format
- `_resolve_manifest_paths()` - Recursive logic unexplained
- `_gather_manifest_paths()` - Manifest index format undocumented
- `collect_naip_sources()` - Input/output not documented

**Unclear design decisions**:
- Why specific Sentinel-2 bands (B03, B04, B05, B06, B08, B11, B12)?
- Why tile size 512x512 and stride 256px?
- Manifest index format (.json with list of paths) not explained
- Multi-AOI processing flow complex but undocumented
- RasterStack limitations and performance characteristics unknown

**Documentation gaps**:
- No API reference documentation
- No architecture diagram showing data flow
- No migration guide for refactored modules
- No performance tuning guide
- No troubleshooting section

### 7. Configuration and Environment Variable Sprawl

**Argument parsing spread across modules**:
- `train_unet.py`: 60+ command-line arguments
- `test_unet.py`: 20+ command-line arguments  
- `sentinel2/compositing.py`: 50+ command-line arguments

**Environment variables**: 20+ different env vars (TRAIN_*, UNET_*, etc.)

**Problems**:
- No centralized configuration management
- Each module re-implements similar argument patterns
- Defaults scattered across different modules
- No ConfigMap, dataclass, or pydantic validation
- Example config file is empty

### 8. Inconsistent Error Handling

**Examples**:
```python
# Explicit validation (good)
if not labels_path.exists():
    raise FileNotFoundError(f"Label dataset not found: {labels_path}")

# Implicit exception (bad - unclear)
gdf = gpd.read_file(labels_path)  # Could raise RasterioIOError or OSError

# Mixed logging patterns
logging.info("...")  # Structured
print(f"Value: {x}")  # No logging
```

### 9. Incomplete Type Hints

**Good examples**:
- `def parse_target_size(value: Optional[str]) -> Optional[Tuple[int, int]]:`
- `def _resolve_manifest_paths(stack_manifest: Optional[str]) -> Sequence[Path]:`

**Missing examples**:
- Many functions lack return type hints
- Some use bare `list` instead of `List[Type]`
- Union types sometimes explicit, sometimes implicit

### 10. Missing Development Configuration

**Missing files**:
- No `pyproject.toml` (PEP 517/518 compliance)
- No `.flake8`, `.pylintrc`, or linting config
- No `mypy.ini` for type checking
- No `.pre-commit-config.yaml`
- No `setup.py` or entry points

**Impact**: No automated code quality gates

---

## Minor Issues

### 11. Root-Level Shim Pattern Not Ideal

**Current approach**:
- Root-level files (sentinel2_processing.py, train_unet.py, etc.)
- Use `sys.path` manipulation to import from src/
- Some re-export all symbols with `from ... import *`

**Better approach**:
- Use setuptools entry points in `setup.py` or `pyproject.toml`
- Eliminates sys.path hacks
- Cleaner from user perspective

### 12. Windows Scripts Unpolished

**Location**: `scripts/windows/` (5 .bat files)
- Hardcoded assumptions
- No error handling
- Not cross-platform

**Missing**: Unix shell equivalents

### 13. Dependency Review Needed

**Currently installed**:
- PyTorch, torchvision (large dependencies)
- pystac, pystac-client, stackstac (Sentinel-2)
- rasterio, rioxarray, geopandas (geospatial)
- Dask (for lazy evaluation)

**Potential issues**:
- Dask imported but not explicitly used in main pipeline
- Some dependencies may be unused
- No version pinning beyond requirements.txt

---

## Detailed Issue Breakdown

### Issue Summary Table

| Issue | Severity | Scope | Effort | Impact |
|-------|----------|-------|--------|--------|
| Broken import in validate_seasonal_pixels.py | CRITICAL | 1 file | 5 min | Runtime failure |
| Empty config file | CRITICAL | 1 file | 15 min | User confusion |
| sentinel2/compositing.py oversized | HIGH | 1 file | 2-4 hrs | Maintainability |
| Incomplete __init__.py exports | HIGH | 2-3 files | 30 min | API confusion |
| Low test coverage | HIGH | Overall | 2-4 hrs | Regression risk |
| Missing docstrings | MEDIUM | 10+ functions | 1-2 hrs | Developer experience |
| Configuration sprawl | MEDIUM | 3 files | 2-3 hrs | Code duplication |
| Inconsistent error handling | MEDIUM | 10+ functions | 1-2 hrs | Code quality |
| No linting/type checking setup | MEDIUM | Overall | 1-2 hrs | IDE support |
| Missing architecture docs | MEDIUM | Overall | 1-2 hrs | Onboarding |
| Windows scripts hardcoding | LOW | 5 files | 1 hr | Maintainability |
| Dependency review | LOW | Overall | 30 min | Build size |

---

## Recommended Cleanup Phases

### Phase 1: Critical Fixes (1-2 days)
**Objective**: Fix breaking issues and incomplete refactoring

1. **Fix broken imports** (30 min)
   - Update `validate_seasonal_pixels.py` line 17
   - Test the fix with import tests

2. **Create example config** (30 min)
   - Generate `configs/train_unet_example.yaml`
   - Document all parameters

3. **Fix package exports** (30 min)
   - Update sentinel2/__init__.py (remove non-existent exports, add missing ones)
   - Create inference/__init__.py
   - Evaluate utils/__init__.py (keep or remove)

4. **Add basic docstrings** (1 hr)
   - Docstrings for all public functions in root modules
   - Module-level docstrings for newly split files

### Phase 2: Structural Refactoring (3-5 days)
**Objective**: Reorganize code to follow architecture guidelines

1. **Split sentinel2/compositing.py** (2-3 hrs)
   - Extract aoi.py (parsing, geometry utilities)
   - Extract sources.py (NAIP collection, preparation)
   - Extract stac_query.py (Sentinel-2 STAC interaction)
   - Extract download.py (data acquisition)
   - Keep compositing.py for orchestration

2. **Add comprehensive docstrings** (1-2 hrs)
   - All public functions
   - Explain non-obvious parameters and return values
   - Document design decisions

3. **Expand test coverage** (2-3 hrs)
   - Add unit tests for utilities (AOI parsing, NAIP processing)
   - Add integration test fixtures
   - Create conftest.py for shared test utilities
   - Target 60%+ coverage

4. **Add development configuration** (1 hr)
   - Create pyproject.toml with tool configs
   - Add mypy.ini for type checking
   - Create pre-commit hook config (optional)

### Phase 3: Documentation & Polish (2-3 days)
**Objective**: Improve developer experience and maintainability

1. **Architecture documentation** (1 hr)
   - Data flow diagrams
   - Manifest format explanation with examples
   - Performance characteristics

2. **Update existing docs** (1 hr)
   - Add API reference section
   - Create migration guide for refactored modules
   - Update repository_guidelines.md with refactoring notes

3. **Code quality improvements** (1-2 hrs)
   - Consistent type hints (run mypy)
   - Consistent error handling patterns
   - Code comments explaining non-obvious logic

### Phase 4: Optional Enhancements (1-2 days)
**Objective**: Polish and performance improvements

1. **Setup and distribution** (1 hr)
   - Create setup.py or use pyproject.toml
   - Define entry points for CLI commands
   - Remove sys.path hacks from shim files

2. **Unix script equivalents** (30 min)
   - Shell script versions of Windows .bat files
   - Cross-platform setup script

3. **Dependency audit** (30 min)
   - Identify unused dependencies
   - Consider version pinning for reproducibility
   - Document minimum Python version requirements

4. **Pre-commit hooks** (30 min)
   - Configure black, isort, flake8, mypy
   - Automatic enforcement before commits

---

## Specific File Locations for Reference

### Critical Files
- `/home/user/wetlands_ml_codex/src/wetlands_ml_geoai/validate_seasonal_pixels.py` (broken import)
- `/home/user/wetlands_ml_codex/configs/train_unet_example.yaml` (empty)
- `/home/user/wetlands_ml_codex/src/wetlands_ml_geoai/sentinel2/__init__.py` (bad exports)

### Files Needing Refactoring
- `/home/user/wetlands_ml_codex/src/wetlands_ml_geoai/sentinel2/compositing.py` (978 lines)
- `/home/user/wetlands_ml_codex/src/wetlands_ml_geoai/training/unet.py` (344 lines)
- `/home/user/wetlands_ml_codex/src/wetlands_ml_geoai/validate_seasonal_pixels.py` (331 lines)

### Files Needing Documentation
- `/home/user/wetlands_ml_codex/src/wetlands_ml_geoai/stacking.py` (301 lines)
- `/home/user/wetlands_ml_codex/src/wetlands_ml_geoai/data_acquisition.py` (252 lines)

### Test Files
- `/home/user/wetlands_ml_codex/tests/test_training_cli.py`
- `/home/user/wetlands_ml_codex/tests/test_inference_cli.py`
- `/home/user/wetlands_ml_codex/tests/test_topography_pipeline.py`

---

## Next Steps

1. **Immediate** (this sprint):
   - Fix broken import in validate_seasonal_pixels.py
   - Create example config file
   - Fix __init__.py exports
   - Add basic docstrings to public functions

2. **Soon** (next 1-2 weeks):
   - Split sentinel2/compositing.py
   - Expand test suite
   - Add linting/type checking setup
   - Update documentation

3. **Later** (polish phase):
   - Create architecture documentation
   - Add pre-commit hooks
   - Review and pin dependencies
   - Create Unix script equivalents

---

## Conclusion

The codebase has a solid foundation with clear intent and good separation of concerns at the package level. However, recent refactoring work (moving from flat structure to package structure) has left broken imports and incomplete migrations. These issues, combined with minimal test coverage and oversized implementation files, create technical debt that should be addressed soon.

The estimated effort for complete cleanup is **5-10 days** depending on test coverage targets. Prioritizing the critical and high-priority issues could improve code quality significantly within **2-3 days**.

