import argparse
import configparser
import json
import logging
import os
import time
import traceback
import warnings
from typing import Optional

import albumentations as A
import cv2
import dotmap
import numpy as np
import plotly.express as px
import rasterio
import torch
import torchvision
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
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

import wandb

from .. import downsample
from ..config import load_config
from ..post_processing import SegmentationPostProcessor
from .model import TiledModel

logger = logging.getLogger("__name__")

# TODO fix warnings
warnings.filterwarnings("ignore")

# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(
        self, data_dir, setname, transform, suffix="_20221010", factor=1, tile_size=2048
    ):

        self.data_dir = data_dir
        self.setname = setname
        self.factor = factor
        assert setname in ["train", "test", "val"]

        with open(os.path.join(self.data_dir, f"{setname}{suffix}.json"), "r") as file:
            self.metadata = json.load(file)

        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose(
                [A.RandomCrop(width=tile_size, height=tile_size), ToTensorV2()]
            )

    def __len__(self):
        return len(self.metadata["images"])

    def __getitem__(self, idx):

        annotation = self.metadata["images"][idx]

        img_name = annotation["file_name"]
        coco_idx = annotation["id"]

        if self.factor == 1:
            img_path = os.path.join(self.data_dir, "images", img_name)
            mask = np.load(
                os.path.join(
                    self.data_dir, "masks", f"{self.setname}_mask_{coco_idx}.npz"
                )
            )["arr_0"].astype(int)
        else:
            img_path = os.path.join(
                self.data_dir,
                "downsampled_images",
                f"sampling_factor_{self.factor}/{img_name}",
            )
            mask = np.load(
                os.path.join(
                    self.data_dir,
                    "downsampled_masks",
                    f"sampling_factor_{self.factor}",
                    f"{self.setname}_mask_{coco_idx}.npz",
                )
            )["arr_0"].astype(int)

        # Albumentations handles conversion to torch tensor
        image = np.array(Image.open(img_path), dtype=np.float32)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()

        return {"image": image, "mask": mask}


class TreeDataModule(LightningDataModule):
    def __init__(
        self,
        data_root,
        num_workers=8,
        data_frac=1.0,
        batch_size=1,
        tile_size=1024,
        augment=True,
    ):
        super().__init__()
        self.data_frac = data_frac
        self.augment = augment
        self.batch_size = batch_size
        self.data_root = data_root
        self.num_workers = num_workers
        self.tile_size = tile_size

    def prepare_data(self) -> None:

        if self.augment:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(),
                    A.RandomBrightnessContrast(p=0.2),
                    A.RandomCrop(width=self.tile_size, height=self.tile_size),
                    ToTensorV2(),
                ]
            )
            logger.debug("Train-time augmentation is enabled.")
        else:
            transform = None

        self.train_data = ImageDataset(
            self.data_root, "train", transform=transform, tile_size=self.tile_size
        )
        self.val_data = ImageDataset(
            self.data_root, "val", transform=None, tile_size=self.tile_size
        )
        self.test_data = ImageDataset(
            self.data_root, "test", transform=None, tile_size=self.tile_size
        )

    def train_dataloader(self):
        return get_dataloaders(
            self.train_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )[0]

    def val_dataloader(self):
        return get_dataloaders(
            self.val_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )[0]

    def test_dataloader(self):
        # Don't shuffle the test loader so we can
        # more easily compare runs on wandb
        return get_dataloaders(
            self.test_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )[0]


def get_dataloaders(*datasets, num_workers=8, data_frac=1, batch_size=1, shuffle=True):

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


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class SemanticSegmentationTaskPlus(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class_labels = ["background", "tree"]

        self.train_metrics = MetricCollection(
            metrics={
                "accuracy": ClasswiseWrapper(
                    Accuracy(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "precision": ClasswiseWrapper(
                    Precision(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "recall": ClasswiseWrapper(
                    Recall(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "f1": ClasswiseWrapper(
                    F1Score(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "jaccard_index": ClasswiseWrapper(
                    JaccardIndex(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "dice": ClasswiseWrapper(
                    Dice(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "confusion_matrix": ConfusionMatrix(
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
        import segmentation_models_pytorch as smp
        from torch import nn

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

    def log_image(self, image, key, caption=""):
        logger.debug(f"Logging image ({caption})")
        images = wandb.Image(image, caption)
        wandb.log({key: images})

    def validation_step(self, *args, **kwargs) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]

        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        self.log(f"val_loss", loss, on_step=False, on_epoch=True)
        y = batch["mask"]

        self.val_metrics(y_hat, y)

        if batch_idx < 10:

            batch["prediction"] = y_hat_hard

            sm = torch.nn.Softmax2d()
            batch["probability"] = sm(y_hat)

            self._log_prediction_images(batch, "val")

    def test_step(self, *args, **kwargs) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        y = batch["mask"]
        self.log(f"test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat, y)
        batch["prediction"] = y_hat_hard

        sm = torch.nn.Softmax2d()
        batch["probability"] = sm(y_hat)

        self._log_prediction_images(batch, "test")

    def _predict_batch(self, batch):
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        return loss, y_hat, y_hat_hard

    def _log_prediction_images(self, batch, split):
        """Plot images during training

        Parameters
        ----------
        batch
            batch from dataloader
        split
            dataset split (e.g. 'test', 'train', 'validation')
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
            logger.debug(f"Logging {split} images")
            self.log_image(
                image_grid,
                key=f"{split}_examples (original/ground truth/prediction/confidence)",
                caption=f"Sample {split} images",
            )
        except AttributeError as e:
            logger.error(e)

    def _log_metrics(self, computed, split):
        """Logs metrics for a particular split

        Parameters
        ----------
        computed
            Dictionary of results from a MetricCollection
        split
            Dataset split
        """

        # Pop + log confusion matrix
        conf_mat = computed.pop(f"{split}_confusion_matrix").cpu().numpy()

        if split in ["val", "test"]:
            conf_mat = (conf_mat / np.sum(conf_mat)) * 100
            cm = px.imshow(conf_mat, text_auto=".2f")
            logger.debug(f"Logging {split} confusion matrix")
            wandb.log({f"{split}_confusion_matrix": cm})

        # Pop + log PR curve
        key = f"{split}_pr_curve"
        if key in computed:
            logger.info("Logging PR curve")

            precision, recall, _ = computed.pop(key)
            classes = ["background", "tree"]

            for i in range(len(precision)):

                recall_np = recall[i].cpu().numpy()
                precision_np = precision[i].cpu().numpy()

                data = [[x, y] for (x, y) in zip(recall_np, precision_np)]

                table = wandb.Table(data=data, columns=["Recall", "Precision"])
                wandb.log(
                    {
                        f"{split}_pr_curve_{classes[i]}": wandb.plot.line(
                            table,
                            "Recall",
                            "Precision",
                            title=f"Precision Recall for {classes[i]}",
                        )
                    }
                )

        # Log everything else
        logger.debug(f"Logging {split} metrics")
        self.log_dict(computed)

    def training_epoch_end(self, outputs):
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        self._log_metrics(computed, "train")
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs):
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        computed = self.val_metrics.compute()
        self._log_metrics(computed, "val")
        self.val_metrics.reset()

    def test_epoch_end(self, outputs):
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        computed = self.test_metrics.compute()
        self._log_metrics(computed, "test")
        self.test_metrics.reset()


class SemanticSegmentationModel(TiledModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_processor = SegmentationPostProcessor(config)

    def load_model(self):

        with open(self.config.model.config, "r") as fp:
            self._cfg = dotmap.DotMap(yaml.load(fp, yaml.SafeLoader), _dynamic=False)

        logging.info(f"Loading checkpoint: {self.config.model.weights}")
        self.model = SemanticSegmentationTaskPlus.load_from_checkpoint(
            self.config.model.weights, strict=True
        ).to(self.device)

        if self.config.model.tta:
            import ttach as tta

            self.model = tta.SegmentationTTAWrapper(
                self.model, transforms=tta.aliases.d4_transform()
            )

    def sweep(self, sweep_id=None):

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

        wandb.log(sweep_configuration)

    def train(self):

        # init wandb
        wandb.init(
            # config=conf["model"],
            entity="dsl-ethz-restor",
            project=self._cfg.wandb.project_name,
        )

        if self._cfg["experiment"]["sweep"] == True:
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
            monitor="val_f1score_tree",
            mode="max",
            dirpath=os.path.join(log_dir, "checkpoints"),
            auto_insert_metric_name=True,
            save_top_k=1,
            save_last=True,
            verbose=True,
        )

        # Setting a short patience of ~O(10) might trigger premature stopping due to a "lucky" batch.
        stopping_patience = int(self._cfg["trainer"]["early_stopping_patience"])
        early_stopping_callback = EarlyStopping(
            monitor="val_f1score_tree",
            min_delta=0.00,
            patience=stopping_patience,
            check_finite=True,
            mode="max",
        )

        csv_logger = CSVLogger(save_dir=log_dir, name="logs")
        wandb_logger = WandbLogger(
            project=self._cfg["wandb"]["project_name"], log_model=True
        )  # log_model='all' cache gets full quite fast

        lr_monitor = LearningRateMonitor(logging_interval="step")
        # stats_monitor = DeviceStatsMonitor(cpu_stats=True)

        wandb.run.summary["log_dir"] = log_dir

        # trainer

        auto_scale_batch = self._cfg["trainer"]["auto_scale_batch_size"] == True
        auto_lr = self._cfg["trainer"]["auto_lr_find"] == True

        trainer = Trainer(
            callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
            logger=[csv_logger, wandb_logger],
            default_root_dir=log_dir,
            accelerator="gpu",
            max_epochs=int(self._cfg["trainer"]["max_epochs"]),
            max_time=self._cfg["trainer"]["max_time"],
            auto_lr_find=auto_lr,
            auto_scale_batch_size="binsearch" if auto_scale_batch else False,
            fast_dev_run=self._cfg["trainer"]["debug_run"] == True,
        )

        wandb_logger.watch(self.model, log="parameters", log_graph=True)

        batch_size = 1

        if auto_scale_batch or auto_lr:
            try:
                logger.info("Tuning trainer")
                results = trainer.tune(model=self.model, train_dataloaders=data_module)
                batch_size = results["scale_batch_size"]
            except Exception as e:
                logger.error("Tuning stage failed")
                logger.error(e)
                logger.error(traceback.print_exc())

        if batch_size > 32:
            accumulate = 1
        else:
            accumulate = max(1, int(32 / batch_size))

        wandb.run.summary["train_batch_size"] = batch_size
        wandb.run.summary["accumulate"] = accumulate
        wandb.run.summary["batch_size"] = batch_size * accumulate

        trainer = Trainer(
            callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
            logger=[csv_logger, wandb_logger],
            default_root_dir=log_dir,
            accelerator="gpu",
            max_epochs=int(self._cfg["trainer"]["max_epochs"]),
            auto_lr_find=False,
            accumulate_grad_batches=accumulate,
            auto_scale_batch_size=False,
            fast_dev_run=self._cfg["trainer"]["debug_run"] == True,
        )

        try:
            logger.info("Starting trainer")
            trainer.fit(model=self.model, datamodule=data_module)
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
            logger.error(traceback.print_exc())

        try:
            logger.info("Train complete, starting test")
            trainer.test(model=self.model, datamodule=data_module)
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
            logger.error(traceback.print_exc())

    def evaluate(self, dataset, output_folder):
        pass

    def predict(self, image):
        """Run inference on an image file or Tensor

        Args:
            image (Union[str, torch.Tensor]): Path to image, or, float tensor in CHW order, un-normalised

        Returns:
            predictions: Detectron2 prediction dictionary
        """

        if isinstance(image, str):
            image = np.array(Image.open(image))
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        elif isinstance(image, torch.Tensor):
            image_tensor = image
        elif isinstance(image, rasterio.io.DatasetReader):
            image_tensor = torch.as_tensor(image.read().astype("float32"))
        else:
            logger.error(
                f"Provided image of type {type(image)} which is not supported."
            )
            raise NotImplementedError

        if self.model is None:
            self.load_model()

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
                logger.error(f"Runtime error: {e}")
                self.should_reload = True
            except Exception as e:
                logger.error(
                    f"Failed to run inference: {e}. Attempting to reload model."
                )
                self.should_reload = True

        t_elapsed_s = time.time() - t_start_s
        logger.debug(f"Predicted tile in {t_elapsed_s:1.2f}s")

        return predictions[0]
