import os

import pytest

from tcd_pipeline.modelrunner import ModelRunner


@pytest.fixture()
def test_image_path():
    path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"
    assert os.path.exists(path)
    return path


@pytest.fixture(scope="session")
def runner():
    runner = ModelRunner("instance")
    return runner


def test_load_detectron(runner):
    """Test if we can load a basic configuration"""
    assert runner.config is not None


def test_predict_simple(runner, test_image_path):
    """Test if we can perform simple prediction"""
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tiled_coco(runner, test_image_path):
    """Test if we can cache to COCO json"""
    runner.model.post_processor.cache_format = "coco"
    runner.model.post_processor.setup_cache()
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tiled_pickle(runner, test_image_path):
    """Test if we can cache to pickle"""
    runner.model.post_processor.cache_format = "pickle"
    runner.model.post_processor.setup_cache()
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tta(test_image_path):
    """Test if we can load a complex configuration (nested config)
    perform a prediction with TTA enabled
    """
    runner = ModelRunner(
        "instance", overrides=["model.config=detectron2/detectron_mask_rcnn_tta"]
    )
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0
