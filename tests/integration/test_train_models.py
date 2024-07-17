import os

import pytest
import torch

from tcd_pipeline.pipeline import Pipeline

os.environ["WANDB_MODE"] = "disabled"
FOLD_ROOT = "data/folds"


@pytest.mark.skipif(
    not os.path.exists(FOLD_ROOT),
    reason="Run locally not on CI for now",
)
def test_train_segmentation():
    runner = Pipeline(
        "semantic",
        options=[
            "model=train_test_run_semantic",
            f"data.root={os.path.join(FOLD_ROOT, 'kfold_0')}",
            "data.output=tests/output",
        ],
    )
    runner.train()


# Train with local dataset
@pytest.mark.skipif(
    not os.path.exists(FOLD_ROOT),
    reason="Run locally not on CI for now",
)
def test_train_mask_rcnn():
    runner = Pipeline(
        "instance",
        options=[
            "model.config=detectron2/detectron_mask_rcnn_test",
            "model.eval_after_train=False",
            "model.device=cuda" if torch.cuda.is_available() else "model.device=cpu",
            "data.root=tests",
            "data.output=tests/temp",
            "data.validation=test_20221010_single.json",
            "data.train=test_20221010_single.json",
        ],
    )
    runner.train()
