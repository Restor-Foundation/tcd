import os

import pytest
import torch

from tcd_pipeline.pipeline import Pipeline


@pytest.fixture()
def test_image_path():
    if torch.cuda.is_available():
        path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"
    else:
        path = "./data/5c15321f63d9810007f8b06f_10_00000_crop.tif"
    assert os.path.exists(path)
    return path


@pytest.fixture(scope="session")
def pipeline():
    pipeline = Pipeline("instance")
    return pipeline


def test_load_detectron(pipeline):
    """Test if we can load a basic configuration"""
    assert pipeline.config is not None


def test_predict_simple(pipeline, test_image_path):
    """Test if we can perform simple prediction"""
    results = pipeline.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tiled_coco(pipeline, test_image_path):
    """Test if we can cache to COCO json"""
    pipeline.model.post_processor.cache_format = "coco"
    pipeline.model.post_processor.setup_cache()
    results = pipeline.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tiled_pickle(pipeline, test_image_path):
    """Test if we can cache to pickle"""
    pipeline.model.post_processor.cache_format = "pickle"
    pipeline.model.post_processor.setup_cache()
    results = pipeline.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tta(test_image_path):
    """Test if we can load a complex configuration (nested config)
    perform a prediction with TTA enabled
    """
    pipeline = Pipeline(
        "instance", overrides=["model.config=detectron2/detectron_mask_rcnn_tta"]
    )
    results = pipeline.predict(test_image_path)
    assert len(results.get_trees()) > 0
