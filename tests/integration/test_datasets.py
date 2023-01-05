import os

import pytest

from tcd_pipeline.models.semantic_segmentation import ImageDataset, TreeDataModule


@pytest.mark.skipif(
    not os.path.exists("data/restor-tcd-oam/masks"),
    reason="Will fail if semantic masks are not present",
)
def test_tree_datamodule_subset():

    datamodule = TreeDataModule("data/restor-tcd-oam", data_frac=0.01)
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

    datamodule = TreeDataModule("data/restor-tcd-oam")
    datamodule.prepare_data()

    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    val_dataloader = datamodule.val_dataloader()
