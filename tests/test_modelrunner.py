from tcd_pipeline.modelrunner import ModelRunner

test_image_path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"


def test_load_detectron():
    """Test if we can load a basic configuration"""
    runner = ModelRunner("./config/base_detectron.yaml")
    assert runner.config is not None


def test_predict_simple():
    """Test if we can perform simple prediction"""
    runner = ModelRunner("./config/base_detectron.yaml")
    results = runner.predict(test_image_path, tiled=False)
    assert len(results.get_trees()) > 0


def test_predict_tiled_coco():
    """Test if we can perform simple prediction"""
    runner = ModelRunner("./config/base_detectron.yaml")
    runner.model.post_processor.cache_format = "coco"
    results = runner.predict(test_image_path, tiled=True)
    assert len(results.get_trees()) > 0


def test_predict_tiled_numpy():
    """Test if we can perform simple prediction"""
    runner = ModelRunner("./config/base_detectron.yaml")
    runner.model.post_processor.cache_format = "numpy"
    results = runner.predict(test_image_path, tiled=True)
    assert len(results.get_trees()) > 0


def test_predict_tiled_pickle():
    """Test if we can perform simple prediction"""
    runner = ModelRunner("./config/base_detectron.yaml")
    runner.model.post_processor.cache_format = "pickle"
    results = runner.predict(test_image_path, tiled=True)
    assert len(results.get_trees()) > 0


def test_predict_tta():
    """Test if we can load a complex configuration (nested config)
    perform a prediction with TTA enabled
    """
    runner = ModelRunner("./config/detectron_tta.yaml")
    results = runner.predict(test_image_path, tiled=False)
    assert len(results.get_trees()) > 0
