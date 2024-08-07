"""Semantic segmentation model framework, using smp models"""

import json
import logging
import os
import warnings
from typing import Any, Callable, List, Union

import albumentations as A
import numpy as np
import torch.multiprocessing
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class COCOSegmentationDataset(Dataset):
    """Image dataset for semantic segmentation tasks."""

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        transform: Union[Callable, Any] = None,
        tile_size: int = 2048,
        image_dirname: str = "images",
        mask_dirname: str = "masks",
        binary_labels: bool = True,
    ):
        """
        Initialise the dataset

        This dataset is designed to work with a COCO annotation file,
        and assumes that the images and masks are stored in the
        supplied image_dirname and mask_dirname folders.

        If a tile_size is provided, the dataset will return a
        random absolute crop of the desired size.

        If you provide a custom transform, ensure that it returns image
        and a mask tensors. This will also override the tile_size.

        Args:
            data_root (str): Path to the data directory
            annotation_path (str): Path to the annotation JSON file
            image_dirname (str): Path to a folder containing images in the dataset
            mask_dirname (str): Path to a folder containing image masks.
            transform (Union[Callable, Any]): Optional transforms to be applied
            tile_size (int, optional): Tile size to return, default to 2048
        """

        self.data_root = data_root
        self.image_path = os.path.join(data_root, image_dirname)
        self.mask_path = os.path.join(data_root, mask_dirname)
        self.binary_labels = binary_labels

        logger.info(f"Looking for images in {self.image_path}")
        logger.info(f"Looking for masks in {self.mask_path}")
        logger.info(f"Loading annotations from: {annotation_path}")

        # TODO: Use MS-COCO
        with open(
            annotation_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose(
                [A.RandomCrop(width=tile_size, height=tile_size), ToTensorV2()]
            )

        self.images = []
        for image in tqdm(self.metadata["images"]):
            # Check if mask exists:
            base = os.path.splitext(image["file_name"])[0]
            mask_path = os.path.join(self.mask_path, base + ".png")
            if os.path.exists(mask_path):
                self.images.append(image)
            else:
                logger.debug(f"Mask not found for {image['file_name']}")

        logger.info(
            "Found {} valid images in {}".format(len(self.images), annotation_path)
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """Returns a dataset sample

        Args:
            idx (int): Index of the sample to return

        Returns:
            dict: containing "image" and "mask" tensors
        """

        annotation = self.images[idx]

        img_name = annotation["file_name"]

        img_path = os.path.abspath(os.path.join(self.image_path, img_name))
        base = os.path.splitext(img_name)[0]

        if self.binary_labels:
            mask = np.array(
                Image.open(os.path.join(self.mask_path, base + ".png")).convert("L"),
                dtype="int",
            )
        else:
            mask = np.array(
                Image.open(os.path.join(self.mask_path, base + ".png")), dtype=int
            )

        # Albumentations handles conversion to torch tensor
        image = Image.open(img_path)

        if image.mode != "RGB" or len(image.getbands()) != 2:
            image = image.convert("RGB")

        image = np.array(image)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].float()

        # Hack for transformer models where the ground truth
        # shouldn't be empty.
        if torch.all(transformed["mask"] == 0):
            transformed["mask"][0, 0] = 1
        elif torch.all(transformed["mask"] == 1):
            transformed["mask"][0, 0] = 0

        mask = (transformed["mask"] > 0).long()

        return {
            "image": image,
            "mask": mask,
            "image_path": img_path,
            "image_name": img_name,
        }
