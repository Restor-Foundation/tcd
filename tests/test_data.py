import os

import pytest
import rasterio

from tcd_pipeline.data import dataloader_from_image
from tcd_pipeline.models.semantic_segmentation import ImageDataset

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


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam/masks"),
    reason="Will fail if semantic masks are not present",
)
def test_imagedataset():

    for split in ["train", "test", "val"]:

        data = ImageDataset("data/restor-tcd-oam", split, transform=None)

        for idx in range(len(data)):
            sample = data[idx]
            assert (
                sample["image"][0, :, :].shape == sample["mask"].shape
            ), f"Failed on index: {idx}"


if __name__ == "__main__":
    test_imagedataset()
