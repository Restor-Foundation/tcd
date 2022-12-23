import logging
import os

import pytest
import rasterio
from modelrunner import ModelRunner

logger = logging.getLogger(__name__)

TEST_IMAGE_PATH = "./data/5c15321f63d9810007f8b06f_10_00000.tif"


@pytest.fixture
def runner():
    return ModelRunner("default_tta.yaml")


def test_inference_stateful(runner):

    output_path = os.path.join(
        "./",
        os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0] + "_pred",
    )

    results = runner.predict(TEST_IMAGE_PATH)

    results.serialise(output_path, image_path=TEST_IMAGE_PATH)
    results.save_masks(output_path, image_path=TEST_IMAGE_PATH)

    assert os.path.exists(output_path, "canopy_mask.tif")
    assert os.path.exists(output_path, "canopy_mask.tif")
    assert os.path.exists(output_path, "results.json")
