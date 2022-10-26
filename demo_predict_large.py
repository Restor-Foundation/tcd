import matplotlib.pyplot as plt
import numpy as np
import rasterio

from modelrunner import ModelRunner

runner = ModelRunner("default_tta.yaml")

image_path = "./data/5f058f16ce2c9900068d83ed.tif"
results = runner.predict(image_path, tiled=True, tile_size=1024, overlap=100)

results.serialise("./data/5f058f16ce2c9900068d83ed_pred", image_path=image_path)
# results.visualise()
