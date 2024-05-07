"""Semantic segmentation model framework, using smp models"""

import logging
import os
import warnings
from typing import Any, List

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .imagedataset import SemanticSegmentationDataset

logger = logging.getLogger("__name__")
warnings.filterwarnings("ignore")


# TODO: check typing
def get_dataloaders(
    *datasets: List[Dataset],
    num_workers: int = 8,
    data_frac: float = 1,
    batch_size: int = 1,
    shuffle: bool = True
):
    """Construct dataloaders from a list of datasets

    Args:
        *datasets (Dataset): List of datasets to use
        num_workers (int, optional): Number of workers to use. Defaults to 8.
        data_frac (float, optional): Fraction of the data to use. Defaults to 1.0.
        batch_size (int, optional): Batch size. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        List[DataLoader]: List of dataloaders

    """
    if data_frac != 1.0:
        datasets = [
            torch.utils.data.Subset(
                dataset,
                np.random.choice(
                    len(dataset), int(len(dataset) * data_frac), replace=False
                ),
            )
            for dataset in datasets
        ]

    return [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(num_workers),
            collate_fn=collate_fn,
        )
        for dataset in datasets
    ]


def collate_fn(batch: Any) -> Any:
    """Collate function for dataloader

    Default collation function, filtering out empty
    values in the batch.

    Args:
        batch (Any): data batch

    Returns:
        Any: Collated batch
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class TCDDataModule(pl.LightningDataModule):
    """Datamodule for TCD"""

    def __init__(
        self,
        data_root,
        train_path="train.json",
        val_path="val.json",
        test_path="test.json",
        num_workers=8,
        data_frac=1.0,
        batch_size=1,
        tile_size=1024,
        augment=True,
    ):
        """
        Initialise the datamodule

        Args:
            data_root (str): Path to the data directory
            num_workers (int, optional): Number of workers to use. Defaults to 8.
            data_frac (float, optional): Fraction of the data to use. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 1.
            tile_size (int, optional): Tile size to return. Defaults to 1024.
            augment (bool, optional): Whether to apply data augmentation. Defaults to True.

        """
        super().__init__()
        self.data_frac = data_frac
        self.augment = augment
        self.batch_size = batch_size
        self.data_root = data_root
        self.train_path = os.path.join(self.data_root, train_path)
        self.val_path = os.path.join(self.data_root, val_path)
        self.test_path = os.path.join(self.data_root, test_path)
        self.num_workers = num_workers
        self.tile_size = tile_size

        logger.info("Data root: %s", self.data_root)

    def prepare_data(self) -> None:
        """
        Construct train/val/test datasets.

        Test datasets do not use data augmentation and simply
        return a tensor. This is to avoid stochastic results
        during evaluation.

        Tensors are returned **not** normalised, as this is
        handled by the forward functions in SMP and transformers.
        """
        logger.info("Preparing datasets")
        if self.augment:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(),
                    A.RandomBrightnessContrast(),
                    A.OneOf([A.Blur(p=0.2), A.Sharpen(p=0.2)]),
                    A.HueSaturationValue(
                        hue_shift_limit=5, sat_shift_limit=4, val_shift_limit=5
                    ),
                    A.RandomCrop(width=1024, height=1024),
                    ToTensorV2(),
                ]
            )
            logger.debug("Train-time augmentation is enabled.")
        else:
            transform = None

        self.train_data = SemanticSegmentationDataset(
            self.data_root,
            self.train_path,
            transform=transform,
            tile_size=self.tile_size,
        )

        self.test_data = SemanticSegmentationDataset(
            self.data_root, self.test_path, transform=A.Compose(ToTensorV2())
        )

        if os.path.exists(self.val_path):
            self.val_data = SemanticSegmentationDataset(
                self.data_root, self.val_path, transform=None, tile_size=self.tile_size
            )
        else:
            self.val_data = self.test_data

    def train_dataloader(self) -> DataLoader:
        """Get training dataloaders:

        Returns:
            List[DataLoader]: List of training dataloaders
        """
        return get_dataloaders(
            self.train_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )[0]

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloaders:

        Returns:
            List[DataLoader]: List of validation dataloaders
        """
        return get_dataloaders(
            self.val_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )[0]

    def test_dataloader(self) -> DataLoader:
        """Get test dataloaders:

        Returns:
            List[DataLoader]: List of test dataloaders
        """
        # Don't shuffle the test loader so we can
        # more easily compare runs on wandb
        return get_dataloaders(
            self.test_data,
            data_frac=self.data_frac,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )[0]
