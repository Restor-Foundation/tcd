import matplotlib.pyplot as plt
import numpy as np
import rasterio

from modelrunner import ModelRunner

runner = ModelRunner("default_tta.yaml")

image_path = "./data/5f058f16ce2c9900068d83ed.tif"
use_cache = False

if use_cache:
    from post_processing import PostProcessor

    process = PostProcessor(runner.config, image=rasterio.open(image_path))
    process.process_cached()
    results = process.process_tiled_result()
else:
    results = runner.predict(image_path, tiled=True)

results.serialise(
    "./data/5f058f16ce2c9900068d83ed_pred", image_path=image_path, save_coco=False
)
