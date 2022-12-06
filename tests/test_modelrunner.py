import os

from modelrunner import ModelRunner

test_image_path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"


def test_load():
    runner = ModelRunner("./config/base_detectron.yaml")
    assert runner.config is not None


def test_predict_simple():
    runner = ModelRunner("./config/base_detectron.yaml")
    results = runner.predict(test_image_path, tiled=False)
    assert len(results.get_trees()) > 0


def test_predict_tta():
    runner = ModelRunner("./config/detectron_tta.yaml")
    results = runner.predict(test_image_path, tiled=False)
    assert len(results.get_trees()) > 0
