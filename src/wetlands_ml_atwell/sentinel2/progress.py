"""Progress reporting utilities for Sentinel-2 workflows."""

from __future__ import annotations

import logging
import time

from dask.diagnostics import ProgressBar


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes}m {remainder:.2f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e9:.2f} ns"


class RasterProgress:
    """Simple logger-backed tracker for raster exports."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0

    def extend(self, count: int) -> None:
        if count > 0:
            self.total += count

    def start(self, label: str) -> None:
        total = max(self.total, self.completed + 1)
        logging.info("Starting raster %d/%d: %s", self.completed + 1, total, label)

    def finish(self, label: str) -> None:
        self.completed += 1
        total = max(self.total, self.completed)
        logging.info("Finished raster %d/%d: %s", self.completed, total, label)

    def skip(self, label: str) -> None:
        if self.total > self.completed:
            self.total -= 1
        total = max(self.total, self.completed)
        logging.info("Skipping raster %s; progress %d/%d.", label, self.completed, total)


class LoggingProgressBar(ProgressBar):
    """Dask progress bar that emits percent updates via logging."""

    def __init__(self, label: str, step: int = 10) -> None:
        super().__init__(minimum=0)
        self.label = label
        self.step = max(step, 1)
        self._last_percent = -self.step
        self._file = None

    def _draw_bar(self, frac, elapsed):
        percent = int(frac * 100)
        if percent >= self._last_percent + self.step or percent >= 100 or self._last_percent < 0:
            try:
                from dask.utils import format_time
            except ImportError:
                format_time = lambda value: f"{value:.2f}s"
            logging.info(
                "%s progress: %d%% (elapsed %s)",
                self.label,
                percent,
                format_time(elapsed),
            )
            self._last_percent = percent


__all__ = ["format_duration", "RasterProgress", "LoggingProgressBar"]

