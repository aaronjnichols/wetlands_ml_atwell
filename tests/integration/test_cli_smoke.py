"""Smoke tests for training and inference CLI entry points.

These tests verify the CLIs can accept valid arguments and perform basic validation
without running the full training/inference pipelines (which would require GPUs, etc).
"""

import pytest


class TestTrainingCliSmoke:
    """Smoke tests for train_unet.py CLI."""

    def test_training_module_imports(self):
        """The train_unet module should be importable."""
        from wetlands_ml_geoai import train_unet
        assert hasattr(train_unet, "main")
        assert hasattr(train_unet, "parse_args")

    def test_parser_requires_labels(self):
        """The parser should require --labels argument."""
        from wetlands_ml_geoai.train_unet import parse_args
        
        with pytest.raises(SystemExit):
            # Has raster but no labels
            parse_args([
                "--train-raster", "dummy.tif",
            ])

    def test_parser_requires_input_source(self):
        """The parser should require either --train-raster or --stack-manifest."""
        from wetlands_ml_geoai.train_unet import parse_args
        
        with pytest.raises(SystemExit):
            # Has labels but no raster/manifest
            parse_args([
                "--labels", "dummy.gpkg",
            ])

    def test_parser_accepts_train_raster(self, minimal_raster_path, mock_labels_gpkg):
        """The parser should accept --train-raster with --labels."""
        from wetlands_ml_geoai.train_unet import parse_args
        
        args = parse_args([
            "--train-raster", str(minimal_raster_path),
            "--labels", str(mock_labels_gpkg),
        ])
        
        assert args.train_raster == str(minimal_raster_path)
        assert args.labels == str(mock_labels_gpkg)

    def test_parser_accepts_stack_manifest(self, mock_stack_manifest, mock_labels_gpkg):
        """The parser should accept --stack-manifest with --labels."""
        from wetlands_ml_geoai.train_unet import parse_args
        
        args = parse_args([
            "--stack-manifest", str(mock_stack_manifest),
            "--labels", str(mock_labels_gpkg),
        ])
        
        assert args.stack_manifest == str(mock_stack_manifest)
        assert args.labels == str(mock_labels_gpkg)

    def test_parser_accepts_training_hyperparameters(self, minimal_raster_path, mock_labels_gpkg):
        """The parser should accept training hyperparameters."""
        from wetlands_ml_geoai.train_unet import parse_args
        
        args = parse_args([
            "--train-raster", str(minimal_raster_path),
            "--labels", str(mock_labels_gpkg),
            "--batch-size", "8",
            "--epochs", "10",
            "--learning-rate", "0.0001",
            "--tile-size", "256",
            "--stride", "128",
        ])
        
        assert args.batch_size == 8
        assert args.epochs == 10
        assert args.learning_rate == 0.0001
        assert args.tile_size == 256
        assert args.stride == 128

    def test_parser_accepts_model_architecture(self, minimal_raster_path, mock_labels_gpkg):
        """The parser should accept model architecture options."""
        from wetlands_ml_geoai.train_unet import parse_args
        
        args = parse_args([
            "--train-raster", str(minimal_raster_path),
            "--labels", str(mock_labels_gpkg),
            "--architecture", "deeplabv3plus",
            "--encoder-name", "efficientnet-b0",
            "--no-encoder-weights",
        ])
        
        assert args.architecture == "deeplabv3plus"
        assert args.encoder_name == "efficientnet-b0"
        assert args.encoder_weights is None


class TestInferenceCliSmoke:
    """Smoke tests for test_unet.py CLI."""

    def test_inference_module_imports(self):
        """The test_unet module should be importable."""
        from wetlands_ml_geoai import test_unet
        assert hasattr(test_unet, "main")
        assert hasattr(test_unet, "parse_args")

    def test_parser_requires_model_path(self):
        """The parser should require --model-path argument."""
        from wetlands_ml_geoai.test_unet import parse_args
        
        with pytest.raises(SystemExit):
            # Has raster but no model
            parse_args([
                "--test-raster", "dummy.tif",
            ])

    def test_parser_requires_input_source(self, tmp_path):
        """The parser should require either --test-raster or --stack-manifest."""
        from wetlands_ml_geoai.test_unet import parse_args
        
        # Create a fake model file
        model_path = tmp_path / "model.pth"
        model_path.touch()
        
        with pytest.raises(SystemExit):
            # Has model but no raster/manifest
            parse_args([
                "--model-path", str(model_path),
            ])

    def test_parser_accepts_test_raster(self, minimal_raster_path, tmp_path):
        """The parser should accept --test-raster with --model-path."""
        from wetlands_ml_geoai.test_unet import parse_args
        
        # Create a fake model file
        model_path = tmp_path / "model.pth"
        model_path.touch()
        
        args = parse_args([
            "--test-raster", str(minimal_raster_path),
            "--model-path", str(model_path),
        ])
        
        assert args.test_raster == str(minimal_raster_path)
        assert args.model_path == str(model_path)

    def test_parser_accepts_stack_manifest(self, mock_stack_manifest, tmp_path):
        """The parser should accept --stack-manifest with --model-path."""
        from wetlands_ml_geoai.test_unet import parse_args
        
        # Create a fake model file
        model_path = tmp_path / "model.pth"
        model_path.touch()
        
        args = parse_args([
            "--stack-manifest", str(mock_stack_manifest),
            "--model-path", str(model_path),
        ])
        
        assert args.stack_manifest == str(mock_stack_manifest)
        assert args.model_path == str(model_path)

    def test_parser_accepts_inference_options(self, minimal_raster_path, tmp_path):
        """The parser should accept inference options."""
        from wetlands_ml_geoai.test_unet import parse_args
        
        # Create a fake model file
        model_path = tmp_path / "model.pth"
        model_path.touch()
        
        args = parse_args([
            "--test-raster", str(minimal_raster_path),
            "--model-path", str(model_path),
            "--window-size", "256",
            "--overlap", "64",
            "--batch-size", "8",
            "--probability-threshold", "0.5",
        ])
        
        assert args.window_size == 256
        assert args.overlap == 64
        assert args.batch_size == 8
        assert args.probability_threshold == pytest.approx(0.5)

    def test_parser_accepts_output_paths(self, minimal_raster_path, tmp_path):
        """The parser should accept output path options."""
        from wetlands_ml_geoai.test_unet import parse_args
        
        # Create a fake model file
        model_path = tmp_path / "model.pth"
        model_path.touch()
        
        args = parse_args([
            "--test-raster", str(minimal_raster_path),
            "--model-path", str(model_path),
            "--output-dir", str(tmp_path / "predictions"),
            "--masks", str(tmp_path / "mask.tif"),
            "--vectors", str(tmp_path / "vectors.gpkg"),
        ])
        
        assert str(args.output_dir).endswith("predictions")
        assert str(args.masks).endswith("mask.tif")
        assert str(args.vectors).endswith("vectors.gpkg")


class TestStackingIntegration:
    """Test that stacking module integrates with fixtures."""

    def test_raster_stack_loads_from_manifest(self, mock_stack_manifest):
        """RasterStack should load from our mock manifest."""
        from wetlands_ml_geoai.stacking import RasterStack
        
        with RasterStack(mock_stack_manifest) as stack:
            assert stack.band_count == 4  # 4 NAIP bands
            assert stack.width == 64
            assert stack.height == 64
            assert stack.crs is not None

    def test_raster_stack_reads_window(self, mock_stack_manifest):
        """RasterStack should read windows from the mock raster."""
        from rasterio.windows import Window
        from wetlands_ml_geoai.stacking import RasterStack
        
        with RasterStack(mock_stack_manifest) as stack:
            window = Window(0, 0, 32, 32)
            data = stack.read_window(window)
            
            assert data.shape == (4, 32, 32)  # 4 bands, 32x32 window
            # Data should be normalized to [0, 1] range (due to scale_max=255)
            assert data.min() >= 0.0
            assert data.max() <= 1.0


class TestManifestResolution:
    """Test manifest resolution with mock data."""

    def test_resolve_single_manifest(self, mock_stack_manifest):
        """_resolve_manifest_paths should find a single manifest."""
        from wetlands_ml_geoai.training.unet import _resolve_manifest_paths
        
        paths = _resolve_manifest_paths(str(mock_stack_manifest))
        
        assert len(paths) == 1
        assert paths[0].name == "stack_manifest.json"

    def test_resolve_manifest_index(self, mock_manifest_index, mock_stack_manifest):
        """_resolve_manifest_paths should resolve manifest index."""
        from wetlands_ml_geoai.training.unet import _resolve_manifest_paths
        
        paths = _resolve_manifest_paths(str(mock_manifest_index))
        
        assert len(paths) == 1
        assert paths[0].name == "stack_manifest.json"

