import os
from glob import glob

import numpy as np
import pytest
import rasterio

from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.models.smpmodule import SMPModule

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)

with rasterio.open(test_image_path) as fp:
    image_shape = fp.shape


@pytest.fixture()
def segmentation_runner(tmpdir):
    runner = ModelRunner(
        "semantic",
        overrides=[
            "model=semantic_segmentation/train_test_run",
            "model.weights=restor/tcd-segformer-mit-b0",
            "postprocess.cleanup=False",
        ],
    )

    return runner


def check_valid(results):
    # Masks and confidence map should be the same as the image
    assert results.mask.shape == image_shape
    assert results.confidence_map.shape == image_shape

    # Results should not be empty
    assert not np.allclose(results.mask, 0)
    assert not np.allclose(results.confidence_map, 0)


def test_segmentation(segmentation_runner):
    results = segmentation_runner.predict(test_image_path, warm_start=False)

    check_valid(results)

    # We expect only a single "tile" for 2048
    assert len(segmentation_runner.model.post_processor.cache) == 1


def test_segmentation_warm(segmentation_runner):
    results = segmentation_runner.predict(test_image_path, warm_start=False)

    # We expect only a single "tile" for 2048
    assert len(segmentation_runner.model.post_processor.cache) == 1

    results = segmentation_runner.predict(test_image_path, warm_start=True)

    check_valid(results)

    # We expect only a single "tile" for 2048
    assert len(segmentation_runner.model.post_processor.cache) == 1


def test_load_segmentation_grid():
    for model in ["unet", "unet++", "deeplabv3+"]:
        for backbone in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            for loss in ["focal", "ce"]:
                _ = SMPModule(
                    model=model,
                    backbone=backbone,
                    weights="imagenet",
                    in_channels=3,
                    num_classes=2,
                    loss=loss,
                    ignore_index=None,
                    lr=1e-3,
                    patience=5,
                )
