import os

import pytest

from tcd_pipeline.pipeline import Pipeline


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam"),
    reason="Run locally not on CI for now",
)
def test_evaluate_mask_rcnn(tmpdir):
    runner = Pipeline("instance")
    runner.evaluate(
        annotation_file="tests/test_20221010_single.json",
        image_folder="data/restor-tcd-oam/images",
        output_folder=tmpdir,
    )
