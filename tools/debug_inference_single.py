import torch
import numpy as np
import rasterio
from wetlands_ml_geoai.inference.unet_stream import infer_raster
from pathlib import Path

# Settings
MODEL_PATH = r"data/MI_Atwell/small/models_training/best_model.pth"
TEST_TILE = list(Path("data/MI_Atwell/small/tiles_training/images").glob("*.tif"))[0]
OUTPUT_DIR = Path("debug_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Testing on tile: {TEST_TILE}")

# 1. Run inference on a single known tile
output_mask = OUTPUT_DIR / "debug_mask.tif"
infer_raster(
    raster_path=TEST_TILE,
    model_path=MODEL_PATH,
    output_path=output_mask,
    window_size=256,
    overlap=0,
    num_channels=34, # 21 (S2) + 4 (NAIP) + 4 (Topo - Slope, TPI, TPI, Depth) + padding if needed
                     # Actually, wait. The tile likely has exactly what was in the manifest.
                     # Let's rely on infer_raster's auto-detection or just read the file first.
    architecture="unet",
    encoder_name="resnet34",
    num_classes=1,
    probability_threshold=0.001 # Super low threshold to see if ANYTHING is detected
)

# 2. Analyze output
with rasterio.open(output_mask) as src:
    data = src.read(1)
    print(f"Prediction Stats:")
    print(f"  Min: {data.min()}")
    print(f"  Max: {data.max()}")
    print(f"  Unique values: {np.unique(data)}")
    print(f"  Non-zero pixels: {np.count_nonzero(data)}")

# 3. Check the tile's own stats to confirm it matches what we expect
with rasterio.open(TEST_TILE) as src:
    print(f"\nInput Tile Stats (Channels: {src.count}):")
    img = src.read()
    # Print mean of first few channels
    print(f"  Means per channel (0-4): {img.mean(axis=(1,2))[:5]}")
    # Print mean of last few channels (Topo)
    if src.count > 25:
        print(f"  Means per channel (25+): {img.mean(axis=(1,2))[25:]}")
