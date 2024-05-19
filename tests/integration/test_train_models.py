import os

import pytest

from tcd_pipeline.modelrunner import ModelRunner

os.environ["WANDB_MODE"] = "disabled"


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam/masks"),
    reason="Run locally not on CI for now",
)
def test_train_segmentation():
    runner = ModelRunner(
        "semantic", overrides=["model=semantic_segmentation/train_test_run"]
    )
    runner.train()


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam"),
    reason="Run locally not on CI for now",
)
def test_train_mask_rcnn():
    runner = ModelRunner(
        "instance",
        overrides=[
            "model.config=detectron2/detectron_mask_rcnn_test",
            "model.eval_after_train=False",
            "data.root=data/folds/kfold_0",
            "data.output=tests/temp",
            "data.validation=test.json",
        ],
    )
    runner.train()
