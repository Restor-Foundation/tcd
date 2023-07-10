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
    runner = ModelRunner("./config/base_detectron.yaml")
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
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tiled_numpy(runner, test_image_path):
    """Test if we can cache to numpy"""
    runner.model.post_processor.cache_format = "numpy"
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tiled_pickle(runner, test_image_path):
    """Test if we can cache to pickle"""
    runner.model.post_processor.cache_format = "pickle"
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0


def test_predict_tta(test_image_path):
    """Test if we can load a complex configuration (nested config)
    perform a prediction with TTA enabled
    """
    runner = ModelRunner("./config/detectron_tta.yaml")
    results = runner.predict(test_image_path)
    assert len(results.get_trees()) > 0
