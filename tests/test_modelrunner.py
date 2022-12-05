import os

from modelrunner import ModelRunner


def test_load():
    runner = ModelRunner("./config/base_detectron.yaml")
    assert runner.config is not None


def test_predict_simple():
    runner = ModelRunner("./config/base_detectron.yaml")
    image_path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"
    output_path = "./_test_output/predict_simple"
    os.makedirs("./_test_output", exist_ok=True)

    results = runner.predict(image_path, tiled=False)
    assert len(results.get_trees()) > 0
