"""Semantic segmentation model framework, using torchgeo as the backend."""

import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Callable, List, Union

import albumentations as A
import cv2
import dotmap
import lightning.pytorch as pl
import numpy as np
import plotly.express as px
import segmentation_models_pytorch as smp
import torch
import torchvision
import ttach as tta
import yaml
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import (
    Accuracy,
    ClasswiseWrapper,
    ConfusionMatrix,
    Dice,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from torchvision.utils import draw_segmentation_masks
from tqdm.auto import tqdm

import wandb
from tcd_pipeline.models.model import TiledModel
from tcd_pipeline.post_processing import SegmentationPostProcessor

logger = logging.getLogger("__name__")
warnings.filterwarnings("ignore")

# collect data and create dataset
class ImageDataset(Dataset):
    """Image dataset for semantic segmentation tasks."""

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        transform: Union[Callable, Any],
        tile_size: int = 2048,
        image_dirname: str = "images",
        mask_dirname: str = "masks",
    ):
        """
        Initialise the dataset

        This dataset is designed to work with a COCO annotation file,
        and assumes that the images and masks are stored in the
        supplied data_dir folder. Providing a factor other than 1
        will attempt to load a down-sampled image which must have
        been pre-generated.

        If a tile_size is provided, the dataset will return a
        random crop of the desired size.

        If you provide a custom transform, ensure that it returns image
        and a mask tensors. This will also override the tile_size.

        Args:
            data_dir (str): Path to the data directory
            setname (str): Name of the dataset, either "train", "val" or "test"
            transform (Union[Callable, Any]): Optional transforms to be applied
            factor (int, optional): Factor to downsample the image by, defaults to 1
            tile_size (int, optional): Tile size to return, default to 2048
        """

        self.data_root = data_root
        self.image_path = os.path.join(data_root, image_dirname)
        self.mask_path = os.path.join(data_root, mask_dirname)

        logger.info(f"Looking for images in {self.image_path}")
        logger.info(f"Looking for masks in {self.mask_path}")

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

        img_path = os.path.join(self.image_path, img_name)
        base = os.path.splitext(img_name)[0]
        mask = np.array(
            Image.open(os.path.join(self.mask_path, base + ".png")), dtype=int
        )

        # Albumentations handles conversion to torch tensor
        image = Image.open(img_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].float()
        mask = (transformed["mask"] > 0).long()

        return {"image": image, "mask": mask}


class TreeDataModule(pl.LightningDataModule):
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
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.num_workers = num_workers
        self.tile_size = tile_size

        logger.info("Data root: %s", self.data_root)

    def prepare_data(self) -> None:
        """
        Construct train/val/test datasets.

        Test datasets do not use data augmentation and simply
        return a tensor. This is to avoid stochastic results
        during evaluation.
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

        self.train_data = ImageDataset(
            self.data_root,
            self.train_path,
            transform=transform,
            tile_size=self.tile_size,
        )

        self.test_data = ImageDataset(
            self.data_root, self.test_path, transform=A.Compose(ToTensorV2())
        )

        if os.path.exists(self.val_path):
            self.val_data = ImageDataset(
                self.data_root, self.val_path, transform=None, tile_size=self.tile_size
            )
        else:
            self.val_data = self.test_data

    def train_dataloader(self) -> List[DataLoader]:
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

    def val_dataloader(self) -> List[DataLoader]:
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

    def test_dataloader(self) -> List[DataLoader]:
        """Get test dataloaders:

        Returns:
            List[DataLoader]: List of test dataloaders
        """
        # Don't shuffle the test loader so we can
        # more easily compare runs on wandb
        return get_dataloaders(
            self.test_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )[0]


# TODO: check typing
def get_dataloaders(*datasets, num_workers=8, data_frac=1, batch_size=1, shuffle=True):
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


class SemanticSegmentationTaskPlus(SemanticSegmentationTask):
    """Semantic segmentation task with additional metric calculation"""

    def __init__(self, *args, **kwargs):
        """Initialise the task and setup metrics for training

        Training metrics are: accuracy, precision, recall, f1,
        jaccard index (iou), dice and confusion matrices.

        During testing, we also compute a PR curve.

        Args:
            *args: Arguments to pass to the SemanticSegmentationTask
            **kwargs: Keyword arguments to pass to the SemanticSegmentationTask

        """

        super().__init__(*args, **kwargs)

        class_labels = ["background", "tree"]

        self.example_input_array = torch.rand((1, 3, 1024, 1024))
        metric_task = "multiclass"

        self.train_metrics = MetricCollection(
            metrics={
                "accuracy": ClasswiseWrapper(
                    Accuracy(
                        task=metric_task,
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "precision": ClasswiseWrapper(
                    Precision(
                        task=metric_task,
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "recall": ClasswiseWrapper(
                    Recall(
                        task=metric_task,
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "f1": ClasswiseWrapper(
                    F1Score(
                        task=metric_task,
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "jaccard_index": ClasswiseWrapper(
                    JaccardIndex(
                        task=metric_task,
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "confusion_matrix": ConfusionMatrix(
                    task=metric_task,
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
            },
            prefix="train_",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val_")

        self.test_metrics = self.train_metrics.clone(prefix="test_")
        # Note, since this is computed over all images, this can be *extremely*
        # compute intensive to calculate in full. Best done once at the end of training.
        # Setting thresholds in advance uses constant memory.
        self.test_metrics.add_metrics(
            {
                "pr_curve": MulticlassPrecisionRecallCurve(
                    num_classes=self.hyperparams["num_classes"],
                    thresholds=torch.from_numpy(np.linspace(0, 1, 20)),
                )
            }
        )

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""

        if self.hyperparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hyperparams["encoder_name"],
                encoder_weights=self.hyperparams["encoder_weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hyperparams["encoder_name"],
                encoder_weights=self.hyperparams["encoder_weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["segmentation_model"] == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=self.hyperparams["encoder_name"],
                encoder_weights=self.hyperparams["encoder_weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['segmentation_model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hyperparams["loss"] == "ce":
            ignore_value = -1000 if self.ignore_index is None else self.ignore_index
            self.loss = nn.CrossEntropyLoss(ignore_index=ignore_value)
        elif self.hyperparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hyperparams["num_classes"]
            )
        elif self.hyperparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=self.ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def log_image(self, image: torch.Tensor, key: str, caption: str = "") -> None:
        """Log an image to wandb

        Args:
            image (torch.Tensor): Image to log
            key (str): Key to use for logging
            caption (str, optional): Caption to use for logging. Defaults to "".

        """
        logger.debug("Logging image (%s)", caption)
        images = wandb.Image(image, caption)

        if wandb.run:
            wandb.log({key: images})

    # pylint: disable=arguments-differ
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Compute validation loss and log example predictions.

        Only logs sample images for the first 10 batches.

        Args:
            batch (dict): output from dataloader
            batch_idx (int): batch index

        """

        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        y = batch["mask"]

        self.val_metrics(y_hat, y)

        if batch_idx < 10:

            batch["prediction"] = y_hat_hard

            p = torch.nn.Softmax2d()
            batch["probability"] = p(y_hat)

            self._log_prediction_images(batch, "val")

    # pylint: disable=arguments-differ
    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Compute test loss and log example predictions

        Args:
            batch (dict): output from dataloader
            batch_idx (int): batch index
        """

        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        y = batch["mask"]
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat, y)
        batch["prediction"] = y_hat_hard

        p = torch.nn.Softmax2d()
        batch["probability"] = p(y_hat)

        self._log_prediction_images(batch, "test")

    def _predict_batch(self, batch):
        """Predict on a batch of data.

        Returns:
            loss (torch.Tensor): Loss for the batch
            y_hat (torch.Tensor): Softmax output from the model
            y_hat_hard (torch.Tensor): Argmax output from the model
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        return loss, y_hat, y_hat_hard

    def _log_prediction_images(self, batch, split):
        """Plot images during training

        Args:
            batch (dict): output from dataloader
            split (str): dataset split (e.g. 'test', 'train', 'validation')
        """

        try:
            for key in ["image", "mask", "prediction", "probability"]:
                batch[key] = batch[key].cpu()

            # Hacky probability map
            prob = np.transpose(
                cv2.cvtColor(
                    cv2.applyColorMap(
                        (255 * batch["probability"][0][1]).numpy().astype(np.uint8),
                        colormap=cv2.COLORMAP_INFERNO,
                    ),
                    cv2.COLOR_RGB2BGR,
                ),
                (2, 0, 1),
            )

            images = {
                "image": batch["image"][0],
                "masked": draw_segmentation_masks(
                    batch["image"][0].type(torch.uint8),
                    batch["mask"][0].type(torch.bool),
                    alpha=0.5,
                    colors="red",
                ),
                "prediction": draw_segmentation_masks(
                    batch["image"][0].type(torch.uint8),
                    batch["prediction"][0].type(torch.bool),
                    alpha=0.5,
                    colors="red",
                ),
                "probability": torch.from_numpy(prob),
            }
            resize = torchvision.transforms.Resize(512)
            image_grid = torchvision.utils.make_grid(
                [resize(value.float()) for _, value in images.items()],
                value_range=(0, 255),
                normalize=True,
            )
            logger.debug("Logging %s images", split)
            self.log_image(
                image_grid,
                key=f"{split}_examples (original/ground truth/prediction/confidence)",
                caption=f"Sample {split} images",
            )
        except AttributeError as e:
            logger.error(e)

    def _log_metrics(self, computed: dict, split: str):
        """Logs metrics for a particular split

        Args:
            computed (dict): Dictionary of metrics from MetricCollection
            split (str): dataset split (e.g. 'test', 'train', 'validation')

        """
        # Pop + log confusion matrix
        conf_mat = computed.pop(f"{split}_confusion_matrix").cpu().numpy()

        # Log everything else
        logger.debug("Logging %s metrics", split)
        self.log_dict(computed)

        if not wandb.run:
            return

        if split in ["val", "test"]:
            conf_mat = (conf_mat / np.sum(conf_mat)) * 100
            cm_plot = px.imshow(conf_mat, text_auto=".2f")
            logger.debug("Logging %s confusion matrix", split)
            wandb.log({f"{split}_confusion_matrix": cm_plot})

        # Pop + log PR curve
        key = f"{split}_pr_curve"
        if key in computed:
            logger.info("Logging PR curve")

            precision, recall, _ = computed.pop(key)
            classes = ["background", "tree"]

            for pr_class in zip(precision, recall, classes):

                curr_precision, curr_recall, curr_class = pr_class

                recall_np = curr_recall.cpu().numpy()
                precision_np = curr_precision.cpu().numpy()

                data = [[x, y] for (x, y) in zip(recall_np, precision_np)]

                table = wandb.Table(data=data, columns=["Recall", "Precision"])

                wandb.log(
                    {
                        f"{split}_pr_curve_{curr_class}": wandb.plot.line(
                            table,
                            "Recall",
                            "Precision",
                            title=f"Precision Recall for {curr_class}",
                        )
                    }
                )

    def on_train_epoch_end(self) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        self._log_metrics(computed, "train")
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        computed = self.val_metrics.compute()
        self._log_metrics(computed, "val")
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        computed = self.test_metrics.compute()
        self._log_metrics(computed, "test")
        self.test_metrics.reset()


class SemanticSegmentationModel(TiledModel):
    """Tiled model subclass for torchgeo/smp semantic segmentation models."""

    def __init__(self, config):
        """Initialize the model.

        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.post_processor = SegmentationPostProcessor(config)

        with open(self.config.model.config, "r", encoding="utf-8") as fp:
            self._cfg = dotmap.DotMap(yaml.load(fp, yaml.SafeLoader), _dynamic=False)

    def load_model(self):
        """Load the model from a checkpoint"""

        logging.info("Loading checkpoint: %s", self.config.model.weights)
        self.model = SemanticSegmentationTaskPlus.load_from_checkpoint(
            self.config.model.weights, strict=True
        ).to(self.device)

        if self.config.model.tta:

            self.transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 180]),
                    tta.Scale(scales=[1, 0.5]),
                ]
            )

    def sweep(self, sweep_id: str = None):
        """Run a hyperparameter sweep using wandb

        Args:
            sweep_id (str): Sweep ID to run
        """
        sweep_configuration = {
            "method": "grid",
            "name": self._cfg["wandb"]["project_name"],
            "program": "vanilla_model.py",
            "metric": {"goal": "minimize", "name": "loss"},
            "parameters": {
                "loss": {"values": ["ce", "focal"]},  #
                "segmentation_model": {"values": ["unet", "deeplabv3+", "unet++"]},  #
                "encoder_name": {
                    "values": ["resnet18", "resnet34", "resnet50", "resnet101"]
                },  #
                "augment": {"values": ["off", "on"]},  #
            },
        }

        if sweep_id is None:
            sweep_id = wandb.sweep(
                sweep=sweep_configuration, project=self._cfg["wandb"]["project_name"]
            )

        wandb.agent(
            sweep_id, project=self._cfg["wandb"]["project_name"], function=self.train
        )  # , count=5)

        if wandb.run:
            wandb.log(sweep_configuration)

    def train(self):
        """Train the model

        Returns:
            bool: True if training was successful
        """

        # init wandb
        wandb.init(
            # config=conf["model"],
            entity="dsl-ethz-restor",
            project=self._cfg.wandb.project_name,
        )

        pl.seed_everything(42)

        if self._cfg["experiment"]["sweep"]:
            logger.info("Training with a sweep configuration")
            self._cfg["datamodule"]["augment"] = str(wandb.config.augment)
            self._cfg["model"]["loss"] = str(wandb.config.loss)
            self._cfg["model"]["segmentation_model"] = str(
                wandb.config.segmentation_model
            )
            self._cfg["model"]["encoder_name"] = str(wandb.config.encoder_name)

        self.model = SemanticSegmentationTaskPlus(
            segmentation_model=self._cfg.model.segmentation_model,
            encoder_name=self._cfg.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=int(self._cfg.model.in_channels),
            num_classes=int(self._cfg.model.num_classes),
            loss=self._cfg.model.loss,
            ignore_index=None,
            learning_rate=float(self._cfg.model.learning_rate),
            learning_rate_schedule_patience=int(
                self._cfg.model.learning_rate_schedule_patience
            ),
        )

        # load data
        data_module = TreeDataModule(
            self.config.data.data_root,
            train_path=self.config.data.train,
            val_path=self.config.data.validation,
            test_path=self.config.data.test,
            augment=self._cfg["datamodule"]["augment"] == "on",
            batch_size=int(self._cfg["datamodule"]["batch_size"]),
            num_workers=int(self._cfg["datamodule"]["num_workers"]),
            tile_size=int(self.config.data.tile_size),
        )

        log_dir = os.path.join(
            self.config.model.log_dir, time.strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(log_dir, exist_ok=True)

        # checkpoints and loggers
        checkpoint_callback = ModelCheckpoint(
            monitor="val_multiclassf1score_tree",
            mode="max",
            dirpath=os.path.join(log_dir, "checkpoints"),
            auto_insert_metric_name=True,
            save_top_k=1,
            save_last=True,
            verbose=True,
        )

        # Setting a short patience of ~O(10) might trigger
        # premature stopping due to a "lucky" batch.
        stopping_patience = int(self._cfg["trainer"]["early_stopping_patience"])
        early_stopping_callback = EarlyStopping(
            monitor="val_multiclassf1score_tree",
            min_delta=0.00,
            patience=stopping_patience,
            check_finite=True,
            mode="max",
        )

        csv_logger = CSVLogger(save_dir=log_dir, name="logs")
        wandb_logger = WandbLogger(
            project=self._cfg["wandb"]["project_name"], log_model=False
        )  # log_model='all' cache gets full quite fast

        lr_monitor = LearningRateMonitor(logging_interval="step")
        # stats_monitor = DeviceStatsMonitor(cpu_stats=True)

        wandb.run.summary["log_dir"] = log_dir

        # auto_scale_batch = self._cfg["trainer"]["auto_scale_batch_size"]
        # auto_lr = self._cfg["trainer"]["auto_lr_find"]
        # debug_run = self._cfg["trainer"]["debug_run"]

        if wandb.run:
            wandb_logger.watch(self.model, log="parameters", log_graph=False)

        # auto scaling
        """
        if not debug_run:
            if auto_scale_batch or auto_lr:

                logger.info("Tuning trainer")
                results = trainer.tune(model=self.model, train_dataloaders=data_module)

                if auto_scale_batch:
                    batch_size = results["scale_batch_size"]
        """

        batch_size = int(self._cfg["datamodule"]["batch_size"])
        if batch_size > 32:
            accumulate = 1
        else:
            accumulate = max(1, int(32 / batch_size))

        wandb.run.summary["train_batch_size"] = batch_size
        wandb.run.summary["accumulate"] = accumulate
        wandb.run.summary["effective_batch_size"] = batch_size * accumulate

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, lr_monitor],
            logger=[csv_logger, wandb_logger],
            default_root_dir=log_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=int(self._cfg["trainer"]["max_epochs"]),
            accumulate_grad_batches=accumulate,
            fast_dev_run=self._cfg["trainer"]["debug_run"],
            devices=1,
        )

        try:
            logger.info("Starting trainer")
            trainer.fit(model=self.model, datamodule=data_module)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
            logger.error(traceback.print_exc())
            wandb.finish()
            exit(1)

        try:
            logger.info("Train complete, starting test")
            trainer.test(model=self.model, datamodule=data_module)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
            logger.error(traceback.print_exc())
            wandb.finish()
            exit(1)

        wandb.finish()
        return True

    def evaluate(self):
        """
        Evaluate the model on the dataset provided in the config.

        Does not log to wandb.
        """

        self.load_model()

        log_dir = os.path.join(
            self.config.model.log_dir, time.strftime("%Y%m%d-%H%M%S_eval")
        )
        os.makedirs(log_dir, exist_ok=True)

        # Evaluate without augmentation
        data_module = TreeDataModule(
            self.config.data.data_root,
            augment=self._cfg["datamodule"]["augment"] == "off",
            batch_size=int(self._cfg["datamodule"]["batch_size"]),
            num_workers=int(self._cfg["datamodule"]["num_workers"]),
            tile_size=int(self.config.data.tile_size),
        )

        csv_logger = CSVLogger(save_dir=log_dir, name="logs")

        # Eval "trainer"
        evaluator = pl.Trainer(
            logger=[csv_logger],
            default_root_dir=log_dir,
            accelerator="gpu",
            auto_lr_find=False,
            auto_scale_batch_size=False,
            devices=1,
        )

        try:
            logger.info("Starting evaluation on test data")
            evaluator.test(model=self.model, datamodule=data_module)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Evaluation failed")
            logger.error(e)
            logger.error(traceback.print_exc())

    def _predict_tensor(self, image_tensor: torch.Tensor):
        """Run inference on an image file or Tensor

        Args:
            image (Union[str, torch.Tensor]): Path to image, or, float tensor in CHW order, un-normalised

        Returns:
            predictions: Detectron2 prediction dictionary
        """

        self.model.eval()
        self.should_reload = False
        predictions = None

        t_start_s = time.time()

        with torch.no_grad():
            # removing alpha channel
            inputs = torch.unsqueeze(image_tensor[:3, :, :], dim=0).to(self.device)

            try:
                predictions = self.model(inputs)
            except RuntimeError as e:
                logger.error("Runtime error: %s", e)
                self.should_reload = True
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Failed to run inference: %s. Attempting to reload model.", e
                )
                self.should_reload = True

        t_elapsed_s = time.time() - t_start_s
        logger.debug("Predicted tile in %1.2fs", t_elapsed_s)

        return predictions[0]
