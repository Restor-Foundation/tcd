import os

import pytest

from tcd_pipeline.pipeline import Pipeline

os.environ["WANDB_MODE"] = "disabled"


@pytest.mark.skipif(
    not os.path.exists("data/folds"),
    reason="Run locally not on CI for now",
)
def test_train_segmentation():
    runner = Pipeline(
        "semantic",
        overrides=[
            "model=semantic_segmentation/train_test_run",
            "data.root=data/folds/kfold_0",
            "data.output=tests/output",
        ],
    )
    runner.train()


# Train with local dataset
def test_train_mask_rcnn():
    runner = Pipeline(
        "instance",
        overrides=[
            "model.config=detectron2/detectron_mask_rcnn_test",
            "model.eval_after_train=False",
            "data.root=tests",
            "data.output=tests/temp",
            "data.validation=test_20221010_single.json",
            "data.train=test_20221010_single.json",
        ],
    )
    runner.train()
