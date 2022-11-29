import matplotlib.pyplot as plt
import numpy as np
import rasterio

from modelrunner import ModelRunner

runner = ModelRunner("default.yaml")

image_path = "./data/5f058f16ce2c9900068d83ed.tif"
results = runner.predict(image_path, tiled=True, tile_size=512, overlap=100)

"""
from post_processing import PostProcessor

process = PostProcessor(runner.config, image=rasterio.open(image_path))
process.process_cached()
results = process.process_tiled_result()
"""

results.serialise("./data/5f058f16ce2c9900068d83ed_pred", image_path=image_path)
