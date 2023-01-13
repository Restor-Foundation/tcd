import os
from glob import glob

import numpy as np
import pytest
import rasterio

from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.models.semantic_segmentation import (
    ImageDataset,
    SemanticSegmentationTaskPlus,
)

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)

with rasterio.open(test_image_path) as fp:
    image_shape = fp.shape


@pytest.fixture()
def segmentation_runner(tmpdir):
    runner = ModelRunner("config/base_semantic_segmentation.yaml")
    return runner


def check_valid(results):

    # Masks and confidence map should be the same as the image
    assert results.canopy_mask.shape == image_shape
    assert results.confidence_map.shape == image_shape

    # Results should not be empty
    assert not np.allclose(results.canopy_mask, 0)
    assert not np.allclose(results.confidence_map, 0)


def test_segmentation(segmentation_runner):

    results = segmentation_runner.predict(test_image_path, warm_start=False)

    check_valid(results)

    # We expect only a single "tile" for 2048
    files = segmentation_runner.model.post_processor._get_cache_tile_files()
    assert len(files) == 1


def test_segmentation_warm(segmentation_runner):

    results = segmentation_runner.predict(test_image_path, warm_start=False)

    # We expect only a single "tile" for 2048
    files = segmentation_runner.model.post_processor._get_cache_tile_files()
    assert len(files) == 1

    results = segmentation_runner.predict(test_image_path, warm_start=True)

    check_valid(results)

    # We expect only a single "tile" for 2048
    files = segmentation_runner.model.post_processor._get_cache_tile_files()
    assert len(files) == 1


def test_load_segmentation_grid():
    for model in ["unet", "unet++", "deeplabv3+"]:
        for backbone in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            for loss in ["focal", "ce"]:
                _ = SemanticSegmentationTaskPlus(
                    segmentation_model=model,
                    encoder_name=backbone,
                    encoder_weights="imagenet",
                    in_channels=3,
                    num_classes=2,
                    loss=loss,
                    ignore_index=None,
                    learning_rate=1e-3,
                    learning_rate_schedule_patience=5,
                )