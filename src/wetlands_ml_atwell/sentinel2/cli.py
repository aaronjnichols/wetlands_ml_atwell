"""Command-line interface for Sentinel-2 seasonal compositing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import compositing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Sentinel-2 seasonal composites and optional stack manifests.",
    )
    compositing.configure_parser(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    compositing.run_from_args(args)


__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    main()

