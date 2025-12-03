"""Smoke tests for training CLI entry points."""

from pathlib import Path

import pytest

from wetlands_ml_atwell import train_unet


def test_training_cli_requires_inputs():
    with pytest.raises(SystemExit):
        train_unet.parse_args([])


@pytest.mark.parametrize(
    "selector",
    [
        "single",
        "directory",
        "index",
    ],
)
def test_resolve_manifest_paths(tmp_path, selector):
    manifest_root = tmp_path

    manifest_single = manifest_root / "manifest.json"
    manifest_single.write_text(
        """
        {"grid": {"transform": [1, 0, 0, 0, -1, 0]}, "sources": [{"type": "naip", "path": "x.tif", "band_labels": ["r"]}]}
        """
    )

    manifest_dir = manifest_root / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "a.json").write_text(
        """
        {"grid": {"transform": [1, 0, 0, 0, -1, 0]}, "sources": [{"type": "naip", "path": "y.tif", "band_labels": ["r"]}]}
        """
    )
    (manifest_dir / "b.json").write_text(
        """
        {"grid": {"transform": [1, 0, 0, 0, -1, 0]}, "sources": [{"type": "naip", "path": "z.tif", "band_labels": ["r"]}]}
        """
    )

    aoi_01 = manifest_root / "aoi_01"
    aoi_01.mkdir()
    (aoi_01 / "stack_manifest.json").write_text(
        """
        {"grid": {"transform": [1, 0, 0, 0, -1, 0]}, "sources": [{"type": "naip", "path": "ao1.tif", "band_labels": ["r"]}]}
        """
    )

    aoi_02 = manifest_root / "aoi_02"
    aoi_02.mkdir()
    (aoi_02 / "stack_manifest.json").write_text(
        """
        {"grid": {"transform": [1, 0, 0, 0, -1, 0]}, "sources": [{"type": "naip", "path": "ao2.tif", "band_labels": ["r"]}]}
        """
    )

    manifest_index = manifest_root / "index.json"
    manifest_index.write_text(
        """
        {
            "manifests": [
                "aoi_01/stack_manifest.json",
                "aoi_02/stack_manifest.json"
            ]
        }
        """
    )

    selectors = {
        "single": (manifest_single, [manifest_single]),
        "directory": (manifest_dir, sorted([manifest_dir / "a.json", manifest_dir / "b.json"])),
        "index": (
            manifest_index,
            [
                aoi_01 / "stack_manifest.json",
                aoi_02 / "stack_manifest.json",
            ],
        ),
    }

    selected_input, expected = selectors[selector]
    result = train_unet._resolve_manifest_paths(str(selected_input))
    assert [path.resolve() for path in result] == [p.resolve() for p in expected]

