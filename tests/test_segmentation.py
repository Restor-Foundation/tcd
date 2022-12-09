import os
from glob import glob

import pytest
import rasterio

from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.models.semantic_segmentation import SemanticSegmentationTaskPlus

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)

with rasterio.open(test_image_path) as fp:
    image_shape = fp.shape


def _get_cache_files(runner):
    return glob(os.path.join(runner.model.post_processor.cache_folder, "*.npz"))


@pytest.fixture()
def segmentation_runner(tmpdir):
    runner = ModelRunner("config/base_semantic_segmentation.yaml")
    return runner


def test_segmentation_untiled(segmentation_runner):

    results = segmentation_runner.predict(
        test_image_path, tiled=False, warm_start=False
    )

    assert results.prediction_mask.shape == image_shape
    assert results.confidence_map.shape == image_shape

    # We expect only a single "tile"
    files = _get_cache_files(segmentation_runner)
    assert len(files) == 1


def test_segmentation_untiled_warm(segmentation_runner):

    results = segmentation_runner.predict(
        test_image_path, tiled=False, warm_start=False
    )

    assert results.prediction_mask.shape == image_shape
    assert results.confidence_map.shape == image_shape

    results = segmentation_runner.predict(test_image_path, tiled=False, warm_start=True)

    assert results.prediction_mask.shape == image_shape
    assert results.confidence_map.shape == image_shape

    # We expect only a single "tile"
    files = _get_cache_files(segmentation_runner)
    assert len(files) == 1


def test_segmentation_tiled(segmentation_runner):
    results = segmentation_runner.predict(test_image_path, tiled=True, warm_start=False)

    assert results.prediction_mask.shape == image_shape
    assert results.confidence_map.shape == image_shape


def test_segmentation_tiled_warm(segmentation_runner):

    results = segmentation_runner.predict(
        test_image_path, tiled=False, warm_start=False
    )

    assert results.prediction_mask.shape == image_shape
    assert results.confidence_map.shape == image_shape

    results = segmentation_runner.predict(test_image_path, tiled=True, warm_start=True)

    assert results.prediction_mask.shape == image_shape
    assert results.confidence_map.shape == image_shape


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


def test_train_segmentation():
    pass
