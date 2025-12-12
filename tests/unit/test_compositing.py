"""Unit tests for sentinel2.compositing module."""

from pathlib import Path

import pytest


class TestDeduplicatePaths:
    """Tests for the _deduplicate_paths helper function."""

    def test_removes_duplicate_paths(self, tmp_path: Path) -> None:
        """Should remove duplicate paths and preserve order."""
        from wetlands_ml_atwell.sentinel2.compositing import _deduplicate_paths

        file1 = tmp_path / "a.tif"
        file2 = tmp_path / "b.tif"
        file1.touch()
        file2.touch()

        # Same files repeated
        paths = [file1, file2, file1, file2, file1]
        result = _deduplicate_paths(paths)

        assert len(result) == 2
        assert result[0] == file1  # First occurrence preserved
        assert result[1] == file2

    def test_handles_empty_list(self) -> None:
        """Should handle empty input list."""
        from wetlands_ml_atwell.sentinel2.compositing import _deduplicate_paths

        result = _deduplicate_paths([])
        assert result == []

    def test_handles_single_path(self, tmp_path: Path) -> None:
        """Should handle single path without change."""
        from wetlands_ml_atwell.sentinel2.compositing import _deduplicate_paths

        file1 = tmp_path / "single.tif"
        file1.touch()

        result = _deduplicate_paths([file1])
        assert len(result) == 1
        assert result[0] == file1

    def test_deduplicates_by_resolved_path(self, tmp_path: Path) -> None:
        """Should deduplicate based on resolved (absolute) path."""
        from wetlands_ml_atwell.sentinel2.compositing import _deduplicate_paths

        # Create file
        file1 = tmp_path / "test.tif"
        file1.touch()

        # Create different Path objects pointing to same file
        path_a = tmp_path / "test.tif"
        path_b = tmp_path / "." / "test.tif"  # Relative path variant

        result = _deduplicate_paths([path_a, path_b])
        assert len(result) == 1

    def test_preserves_first_occurrence(self, tmp_path: Path) -> None:
        """Should preserve the first occurrence when deduplicating."""
        from wetlands_ml_atwell.sentinel2.compositing import _deduplicate_paths

        file1 = tmp_path / "test.tif"
        file1.touch()

        # Use different path representations
        original = tmp_path / "test.tif"
        variant = tmp_path / "." / "test.tif"

        result = _deduplicate_paths([original, variant])
        assert len(result) == 1
        # First path should be preserved (not the variant)
        assert result[0] == original
