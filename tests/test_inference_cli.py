"""Smoke tests for inference CLI entry points."""

import pytest

from wetlands_ml_geoai import test_unet


def test_inference_cli_requires_inputs():
    with pytest.raises(SystemExit):
        test_unet.parse_args([])


def test_unet_probability_threshold_parses():
    args = test_unet.parse_args(
        [
            "--stack-manifest",
            "dummy.json",
            "--model-path",
            "model.pth",
            "--probability-threshold",
            "0.25",
        ]
    )
    assert args.probability_threshold == pytest.approx(0.25)
