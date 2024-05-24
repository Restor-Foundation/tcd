import os
from glob import glob

import numpy as np
import pytest
import rasterio
import transformers

from tcd_pipeline.models.smpmodule import SMPModule
from tcd_pipeline.pipeline import Pipeline

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)

with rasterio.open(test_image_path) as fp:
    image_shape = fp.shape


@pytest.fixture()
def segmentation_pipeline(tmpdir):
    pipeline = Pipeline(
        "semantic",
        overrides=[
            "data.tile_size=1024",  # Large tiles will fail on GH actions
            "model=semantic_segmentation/train_test_run",
            "postprocess.cleanup=False",
        ],
    )

    return pipeline


def check_valid(results):
    # Masks and confidence map should be the same as the image
    assert results.mask.shape == image_shape
    assert results.confidence_map.shape == image_shape

    # Results should not be empty
    assert not np.allclose(results.mask, 0)
    assert not np.allclose(results.confidence_map, 0)


def test_segmentation(segmentation_pipeline):
    results = segmentation_pipeline.predict(test_image_path, warm_start=False)

    check_valid(results)

    assert len(segmentation_pipeline.model.post_processor.cache) == 9


def test_segmentation_warm(segmentation_pipeline):
    results = segmentation_pipeline.predict(test_image_path, warm_start=False)

    assert len(segmentation_pipeline.model.post_processor.cache) == 9

    results = segmentation_pipeline.predict(test_image_path, warm_start=True)

    check_valid(results)

    assert len(segmentation_pipeline.model.post_processor.cache) == 9


def test_load_segmentation_grid_smp():
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
