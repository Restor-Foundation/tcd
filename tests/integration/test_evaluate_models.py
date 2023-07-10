import os

import pytest

from tcd_pipeline.modelrunner import ModelRunner


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam"),
    reason="Run locally not on CI for now",
)
def test_evaluate_mask_rcnn(tmpdir):
    runner = ModelRunner("config/test_instance_segmentation.yaml")
    runner.evaluate(
        annotation_file="tests/test_20221010_single.json",
        image_folder="data/restor-tcd-oam/images",
        output_folder=tmpdir,
    )
