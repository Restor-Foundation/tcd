import os

from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.models.semantic_segmentation import SemanticSegmentationTaskPlus

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)


def test_segmentation():
    runner = ModelRunner("config/base_semantic_segmentation.yaml")
    _ = runner.predict(test_image_path, tiled=False)


def test_segmentation_tiled():
    runner = ModelRunner("config/base_semantic_segmentation.yaml")
    _ = runner.predict(test_image_path, tiled=True)


def test_load_segmentation_grid():
    for model in ["unet", "unet++", "deeplabv3+"]:
        for backbone in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            for loss in ["focal", "ce"]:
                _ = SemanticSegmentationTaskPlus(
                    segmentation_model=model,
                    encoder_name=backbone,
                    encoder_weights="imagenet",
                    in_channels=3,
                    num_classes=2,
                    loss=loss,
                    ignore_index=None,
                    learning_rate=1e-3,
                    learning_rate_schedule_patience=5,
                )


def test_train_segmentation():
    pass
