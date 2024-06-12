import tempfile

import numpy as np
import pytest
from PIL import Image
from util import test_image_path

from tcd_pipeline.data.tiling import TiledGeoImage, TiledImage, Tiler

"""
Test tile generator:
"""


def test_tiler_equal_size():
    tiler = Tiler(2048, 2048, 2048, 256)

    tiles = [t for t in tiler.tiles]
    assert len(tiles) == 1
    slice_x, slice_y = tiles[0]

    assert slice_x.stop - slice_x.start == 2048
    assert slice_y.stop - slice_y.start == 2048


def test_tiler_overlap():
    tiler = Tiler(2048, 2048, 1024, 256)

    for tile in tiler.tiles:
        slice_x, slice_y = tile

        assert slice_x.stop - slice_x.start == 1024
        assert slice_y.stop - slice_y.start == 1024


def test_tiler_zero_overlap():
    tiler = Tiler(2048, 2048, 1024, 0)
    assert len([t for t in tiler.tiles]) == 4


@pytest.mark.xfail
def test_tiler_too_big_overlap():
    _ = Tiler(2048, 2048, 512, 1024)


"""
Image tiling
"""


def test_non_geo_image_tiler():
    with tempfile.NamedTemporaryFile() as temp_file:
        image_name = temp_file.name + ".jpg"

        array = np.random.randint(0, 255, size=(2048, 2048, 3), dtype=np.uint8)
        Image.fromarray(array).save(image_name)

        tiles = TiledImage(image_name, tile_size=1024, overlap=256)
        assert len(tiles) == 9


def test_geo_image_tiler_downsample(test_image_path):
    tiles = TiledGeoImage(test_image_path, tile_size=1024, target_gsd=0.2)
    assert len(tiles) == 1


def test_geo_image_tiler_same_gsd(test_image_path):
    tiles = TiledGeoImage(test_image_path, tile_size=1024, overlap=256, target_gsd=0.1)
    assert len(tiles) == 9


def test_geo_image_tiler_oversize(test_image_path):
    tiles = TiledGeoImage(test_image_path, tile_size=4096, overlap=256, target_gsd=0.1)
    assert len(tiles) == 1
    w = tiles[0]["window"]
    assert w.width == 4096
    assert w.height == 4096


def test_geo_image_tiler_visualise(test_image_path):
    tiled_image = TiledGeoImage(test_image_path, tile_size=1024, target_gsd=0.1)

    tiled_image.visualise(edges=True, boxes=True, midpoints=True)

    for i in range(len(tiled_image)):
        tiled_image.visualise(idx=i)
        tiled_image.visualise_tile(idx=i, show_valid=True)
