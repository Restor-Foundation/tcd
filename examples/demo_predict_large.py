import logging
import os

import rasterio

from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.util import convert_to_projected

logger = logging.getLogger(__name__)

runner = ModelRunner("./default.yaml")


def run(image_name):
    image_path = f"./data/{image_name}.tif"
    image_path_new = f"./data/small/{image_name}.tif"
    convert_to_projected(image_path, image_path_new, resample=True)

    output_path = "./serializations/small"
    filename = f"{image_name}.json"
    results = runner.predict(image_path_new)
    results.serialise(output_path, image_path=image_path_new, file_name=filename)


image_names = [
    "mixed",
    "mixed2_forestfire_green",
    "mixed2",
    "mixed3",
    "open_canopy",
    "open_canopy2",
    "open_different_tree_types",
    "palm_tree_plantation",
    "palm_tree_plantation2",
    "plantation_multiple_small",
    "plantation_multiple",
    "plantation",
    "plantations_mixed",
]
for image_name in image_names:
    run(image_name)
