import os

from tcd_pipeline.data.datamodule import COCODataModule


def test_datamodule():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    datamodule = COCODataModule(
        data_root=root,
        train_path=os.path.join(root, "test_20221010_single.json"),
        test_path=os.path.join(root, "test_20221010_single.json"),
        val_path=os.path.join(root, "test_20221010_single.json"),
    )

    datamodule.prepare_data()

    dl = datamodule.train_dataloader()
    assert len(dl) == 1
    dl = datamodule.val_dataloader()
    assert len(dl) == 1
    dl = datamodule.test_dataloader()
    assert len(dl) == 1
