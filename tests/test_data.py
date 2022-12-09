import os

import pytest
import rasterio

from tcd_pipeline.data import dataloader_from_image

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)


def test_dataloader_small_tile():

    with rasterio.open(test_image_path) as image:
        dataloader = dataloader_from_image(image, tile_size_px=1024, stride_px=256)
        assert len(dataloader) > 0


@pytest.mark.xfail()
def test_dataloader_equal_size():

    with rasterio.open(test_image_path) as image:
        dataloader = dataloader_from_image(image, tile_size_px=2048, stride_px=256)
        assert len(dataloader) > 0
