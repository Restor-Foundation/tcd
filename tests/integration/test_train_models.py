import os
from glob import glob

import numpy as np
import pytest
import rasterio

from tcd_pipeline.modelrunner import ModelRunner


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam/masks"),
    reason="Run locally not on CI for now",
)
def test_train_segmentation():
    runner = ModelRunner("config/test_semantic_segmentation.yaml")
    runner.train()


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam"),
    reason="Run locally not on CI for now",
)
def test_train_mask_rcnn():
    runner = ModelRunner("config/test_instance_segmentation.yaml")
    runner.train()
