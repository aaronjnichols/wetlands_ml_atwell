"""Smoke tests for inference CLI entry points."""

import numpy as np
import pytest

from wetlands_ml_geoai import test_unet
from wetlands_ml_geoai.inference.unet_stream import _prepare_window
from wetlands_ml_geoai.stacking import FLOAT_NODATA, normalize_stack_array


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


def test_prepare_window_matches_training_normalization():
    raw = np.array(
        [
            [[0.2, 0.5], [np.nan, FLOAT_NODATA]],
            [[1.2, -5.0], [0.4, np.inf]],
            [[FLOAT_NODATA, 2.0], [3.0, -np.inf]],
        ],
        dtype=np.float32,
    )

    expected = np.zeros((4, raw.shape[1], raw.shape[2]), dtype=np.float32)
    expected[: raw.shape[0]] = raw
    expected = normalize_stack_array(expected, FLOAT_NODATA)

    actual = _prepare_window(raw, desired_channels=4, nodata_value=FLOAT_NODATA)

    np.testing.assert_allclose(actual, expected)
