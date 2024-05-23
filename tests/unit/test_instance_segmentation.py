import os

import pytest
import rasterio

from tcd_pipeline.pipeline import Pipeline

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)

with rasterio.open(test_image_path) as fp:
    image_shape = fp.shape


@pytest.fixture()
def instance_segmentation_pipeline(tmpdir):
    # Disable cleanup so we can check cache files
    pipeline = Pipeline(
        "instance",
        overrides=[
            "model.config=detectron2/detectron_mask_rcnn_test",
            "postprocess.cleanup=False",
            "data.tile_size=1024",
        ],
    )
    return pipeline


def test_rcnn(instance_segmentation_pipeline):
    _ = instance_segmentation_pipeline.predict(test_image_path, warm_start=False)

    # We expect 9 tiles for 2048
    files = instance_segmentation_pipeline.model.post_processor.cache
    assert len(files) == 9


def test_rcnn_warm(instance_segmentation_pipeline):
    _ = instance_segmentation_pipeline.predict(test_image_path, warm_start=False)

    # We expect 9 tiles for 1024
    files = instance_segmentation_pipeline.model.post_processor.cache
    assert len(files) == 9

    _ = instance_segmentation_pipeline.predict(test_image_path, warm_start=True)

    # We expect 9 tiles for 1024
    files = instance_segmentation_pipeline.model.post_processor.cache
    assert len(files) == 9
