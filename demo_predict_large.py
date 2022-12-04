import logging
import os

import rasterio

from modelrunner import ModelRunner

logger = logging.getLogger(__name__)

runner = ModelRunner("default_tta.yaml")

image_path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"
output_path = os.path.join(
    os.path.dirname(image_path),
    os.path.splitext(os.path.basename(image_path))[0] + "_pred",
)
use_cache = False

if use_cache:
    from post_processing import PostProcessor

    process = PostProcessor(runner.config, image=rasterio.open(image_path))
    process.process_cached()
    results = process.process_tiled_result()
else:
    results = runner.predict(image_path, tiled=True)

results.serialise(output_path, image_path=image_path)
results.save_masks(output_path, image_path=image_path)
