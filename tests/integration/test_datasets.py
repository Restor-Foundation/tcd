import os
from collections import defaultdict

import pytest
from pycocotools.coco import COCO

from tcd_pipeline.data.datamodule import TCDDataModule
from tcd_pipeline.data.imagedataset import ImageDataset


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam/masks"),
    reason="Will fail if semantic masks are not present",
)
def test_tree_datamodule_subset():
    datamodule = TCDDataModule("data/restor-tcd-oam", data_frac=0.01)
    datamodule.prepare_data()
    train_dataloader = datamodule.train_dataloader()

    for sample in train_dataloader:
        assert sample is not None


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam/masks"),
    reason="Will fail if semantic masks are not present",
)
def test_imagedataset_only():
    for split in ["train", "test", "val"]:
        data = ImageDataset("data/restor-tcd-oam", split, transform=None)

        for idx in range(len(data)):
            sample = data[idx]
            assert (
                sample["image"][0, :, :].shape == sample["mask"].shape
            ), f"Failed on index: {idx}"


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam/masks"),
    reason="Will fail if semantic masks are not present",
)
def test_tree_datamodule_full():
    datamodule = TCDDataModule("data/restor-tcd-oam")
    datamodule.prepare_data()

    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    val_dataloader = datamodule.val_dataloader()


@pytest.mark.skipif(
    not os.path.exists("data/folds"),
    reason="Will fail if folds do not exist",
)
def test_images_exist_in_data_folds():
    """
    Verify that all images in each data fold/split exist
    """

    missing = defaultdict(list)
    unique = set()

    for fold in range(5):
        image_root = f"data/folds/kfold_{fold}/images"

        for split in ["train", "test"]:
            data = COCO(os.path.join(f"data/folds/kfold_{fold}/{split}.json"))

            for idx in data.imgs:
                image_path = os.path.join(image_root, data.imgs[idx]["file_name"])
                assert os.path.exists(image_path)

    return missing, unique
