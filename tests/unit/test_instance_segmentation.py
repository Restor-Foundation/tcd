import os

import pytest
import rasterio

from tcd_pipeline.modelrunner import ModelRunner

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)

with rasterio.open(test_image_path) as fp:
    image_shape = fp.shape


@pytest.fixture()
def instance_segmentation_runner(tmpdir):
    # Disable cleanup so we can check cache files
    runner = ModelRunner(
        "instance",
        overrides=[
            "model.config=detectron2/detectron_mask_rcnn_test",
            "postprocess.cleanup=False",
            "data.tile_size=1024",
        ],
    )
    return runner


def test_rcnn(instance_segmentation_runner):
    _ = instance_segmentation_runner.predict(test_image_path, warm_start=False)

    # We expect 9 tiles for 2048
    files = instance_segmentation_runner.model.post_processor._get_cache_tile_files()
    assert len(files) == 9


def test_rcnn_warm(instance_segmentation_runner):
    _ = instance_segmentation_runner.predict(test_image_path, warm_start=False)

    # We expect 9 tiles for 1024
    files = instance_segmentation_runner.model.post_processor._get_cache_tile_files()
    assert len(files) == 9

    _ = instance_segmentation_runner.predict(test_image_path, warm_start=True)

    # We expect 9 tiles for 1024
    files = instance_segmentation_runner.model.post_processor._get_cache_tile_files()
    assert len(files) == 9
