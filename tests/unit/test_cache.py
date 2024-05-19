import os
import random

import numpy as np
import pytest
import rasterio
import rasterio.windows
from shapely.geometry import box
from util import random_bbox, random_polygon

from tcd_pipeline.cache.cache import ResultsCache
from tcd_pipeline.cache.instance import (
    COCOInstanceCache,
    InstanceSegmentationCache,
    PickleInstanceCache,
)
from tcd_pipeline.cache.semantic import (
    NumpySemanticCache,
    PickleSemanticCache,
    SemanticSegmentationCache,
)
from tcd_pipeline.postprocess.processedinstance import ProcessedInstance


@pytest.fixture()
def test_image():
    path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"
    assert os.path.exists(path)
    return path


@pytest.fixture()
def mock_instances(test_image):
    image = rasterio.open(test_image)

    instances = []

    for i in range(10):
        bbox = random_bbox(image, 200, 200)
        polygon = random_polygon(bbox)
        score = random.random()
        class_index = random.choice([1, 2])

        instance = ProcessedInstance(
            score=score,
            bbox=bbox,
            class_index=class_index,
            global_polygon=polygon,
        )

        instances.append(instance)

    return instances


@pytest.fixture
def mock_mask(test_image):
    image = rasterio.open(test_image)
    shape = (image.height, image.width, random.randint(2, 5))
    return np.random.random(shape)


@pytest.fixture()
def coco_instance_cache(tmp_path, test_image):
    cache = COCOInstanceCache(tmp_path, image_path=test_image)
    cache.initialise()

    return cache


@pytest.fixture()
def pickle_instance_cache(tmp_path, test_image):
    cache = PickleInstanceCache(tmp_path, image_path=test_image)
    cache.initialise()

    return cache


@pytest.fixture()
def numpy_semantic_cache(tmp_path, test_image):
    cache = NumpySemanticCache(tmp_path, image_path=test_image)
    cache.initialise()

    return cache


@pytest.fixture()
def pickle_semantic_cache(tmp_path, test_image):
    cache = PickleSemanticCache(tmp_path, image_path=test_image)
    cache.initialise()

    return cache


# Instance segmentation cache tests


def check_instance_result(instances, result):
    """
    Check that results in an instance cache matches what we "put" in it.
    """
    cached_instances = result["instances"]

    for x in list(zip(instances, cached_instances)):
        mock, cached = x
        assert np.allclose(mock.score, cached.score)
        assert np.allclose(mock.bbox.bounds, cached.bbox.bounds)
        assert mock.class_index == cached.class_index

        intersection = mock.polygon.intersection(cached.polygon).area
        union = mock.polygon.union(cached.polygon).area

        assert intersection / union > 0.99


def check_cache_instance(cache: InstanceSegmentationCache, mock_instances, test_image):
    """
    Wrapper to test an instance cache type
    """
    ds = rasterio.open(test_image)
    mock_bbox = box(*ds.bounds)
    cache.save(mock_instances, bbox=mock_bbox)

    # Check that we can save a debug image
    cache.cache_image(
        ds,
        window=rasterio.windows.from_bounds(*mock_bbox.bounds, transform=ds.transform),
    )
    assert os.path.exists(os.path.join(cache.cache_folder, "1_tile.tif"))

    # Load results
    cache.load()

    # We expect a single cached result and the bounding boxes should match
    assert len(cache) == 1
    result = cache.results[0]
    assert np.allclose(result["bbox"].bounds, mock_bbox.bounds)

    check_instance_result(mock_instances, result)

    cache.clear()
    cache.load()
    assert len(cache.results) == 0


def test_save_instance_coco(coco_instance_cache, mock_instances, test_image):
    check_cache_instance(coco_instance_cache, mock_instances, test_image)


def test_save_instance_pickle(pickle_instance_cache, mock_instances, test_image):
    check_cache_instance(pickle_instance_cache, mock_instances, test_image)


# Semantic segmentation cache tests


def check_cache_semantic(cache: SemanticSegmentationCache, mock_mask, test_image):
    """
    Wrapper to test an semantic cache type
    """
    ds = rasterio.open(test_image)
    mock_bbox = box(*ds.bounds)
    cache.save(mock_mask, mock_bbox)

    # Check that we can save a debug image
    cache.cache_image(
        ds,
        window=rasterio.windows.from_bounds(*mock_bbox.bounds, transform=ds.transform),
    )
    assert os.path.exists(os.path.join(cache.cache_folder, "1_tile.tif"))

    # Load results
    cache.load()

    # We expect a single cached result and the bounding boxes should match
    assert len(cache) == 1
    result = cache.results[0]
    assert np.allclose(result["bbox"].bounds, mock_bbox.bounds)
    assert np.allclose(result["mask"], mock_mask)

    cache.clear()
    cache.load()
    assert len(cache.results) == 0


def test_save_semantic_numpy(numpy_semantic_cache, mock_mask, test_image):
    check_cache_semantic(numpy_semantic_cache, mock_mask, test_image)


def test_save_semantic_pickle(pickle_semantic_cache, mock_mask, test_image):
    check_cache_semantic(pickle_semantic_cache, mock_mask, test_image)
