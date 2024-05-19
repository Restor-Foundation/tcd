import os

import pytest
import rasterio
import torch

from tcd_pipeline.data.dataset import dataloader_from_image

# 2048 x 2048
test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)


def test_dataloader_small_tile():
    with rasterio.open(test_image_path) as image:
        dataloader = dataloader_from_image(image, tile_size_px=1024, overlap_px=512)

        assert len(dataloader) == 9, [tile for tile in dataloader.dataset.tiles]
        # We expect 3x3 tiles
        for data in dataloader:
            assert data["image"][0].shape[:2] == torch.Size([1024, 1024])


@pytest.mark.xfail
def test_dataloader_tile_size_not_divisible_32():
    with rasterio.open(test_image_path) as image:
        dataloader = dataloader_from_image(image, tile_size_px=1032, overlap_px=512)

        assert len(dataloader) == 9

        for data in dataloader:
            assert data["image"].shape[:2] == torch.Size([1024, 1024])


def test_dataloader_equal_size():
    with rasterio.open(test_image_path) as image:
        dataloader = dataloader_from_image(image, tile_size_px=2048, overlap_px=512)
        assert len(dataloader) == 1

        for data in dataloader:
            assert data["image"][0].shape[:2] == torch.Size([2048, 2048])


def test_dataloader_large_tile():
    with rasterio.open(test_image_path) as image:
        dataloader = dataloader_from_image(
            image, tile_size_px=4096, overlap_px=512, pad_if_needed=False
        )
        assert len(dataloader) == 1

        for data in dataloader:
            assert data["image"][0].shape[:2] == torch.Size([2048, 2048])

        dataloader = dataloader_from_image(
            image, tile_size_px=4096, overlap_px=512, pad_if_needed=True
        )
        assert len(dataloader) == 1

        for data in dataloader:
            assert data["image"][0].shape[:2] == torch.Size([4096, 4096])
