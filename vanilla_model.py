import argparse
import configparser
import json
import logging
import os
import string
import sys
import time
import traceback
import warnings
from ctypes import cast
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
import torchgeo
import torchvision
import yaml
from albumentations.pytorch import ToTensorV2
from decouple import config
from PIL import Image
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets.utils import unbind_samples
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
from utils import downsample

logger = logging.getLogger("vanilla_model")
logging.basicConfig(level=logging.INFO)

# TODO fix warnings
warnings.filterwarnings("ignore")

DATA_DIR = config("DATA_DIR")
LOG_DIR = config("LOG_DIR")
REPO_DIR = config("REPO_DIR")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf",
    type=str,
    nargs="?",
    const=True,
    default="conf.yaml",
    help="Choose config file for setup",
)

parser.add_argument(
    "--segmentation_model", help="segmentation architecture", type=str, default=None
)
parser.add_argument("--loss", help="loss function", type=str, default=None)
parser.add_argument("--encoder_name", help="backbone structure", type=str, default=None)
parser.add_argument("--learning_rate", type=float, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--factor", type=int, default=None)
parser.add_argument("--augment", type=str, default=None)

args = parser.parse_args()
conf = configparser.ConfigParser()
conf.read(args.conf)

if args.segmentation_model is not None:
    conf["model"]["segmentation_model"] = args.segmentation_model

if args.loss is not None:
    conf["model"]["loss"] = args.loss

if args.encoder_name is not None:
    conf["model"]["encoder_name"] = args.encoder_name

if args.augment is not None:
    conf["datamodule"]["augment"] = args.augment

if args.factor:
    FACTOR = args.factor
else:
    FACTOR = int(conf["experiment"]["factor"])

if conf["experiment"]["setup"] == "True":
    from utils import clean_data

if FACTOR != 1:
    if not os.path.exists(
        f"{DATA_DIR}images/downsampled_images/sampling_factor_{FACTOR}"
    ):
        downsample.sampler(int(FACTOR))

# if the imports throw OMP error #15, try $ conda install nomkl
# or, as an unsafe quick fix like above, import os; os.environ['KMP_DUPLICATE_LIB_OK']='True';

# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, transform):

        self.data_dir = DATA_DIR
        self.setname = setname
        assert setname in ["train", "test", "val"]

        with open(self.data_dir + setname + "_20221010.json", "r") as file:
            self.metadata = json.load(file)

        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.metadata["images"])

    def __getitem__(self, idx):
        annotation = self.metadata["images"][idx]

        img_name = annotation["file_name"]
        coco_idx = annotation["id"]

        if FACTOR == 1:
            img_path = os.path.join(self.data_dir, "images", img_name)
            try:
                mask = np.load(
                    os.path.join(
                        self.data_dir, "masks", f"{self.setname}_mask_{coco_idx}.npz"
                    )
                )["arr_0"].astype(int)
            except:
                return None
        else:
            img_path = os.path.join(
                self.data_dir,
                "downsampled_images",
                f"sampling_factor_{FACTOR}/{img_name}",
            )

            try:
                mask = np.load(
                    os.path.join(
                        self.data_dir,
                        "downsampled_masks",
                        f"sampling_factor_{FACTOR}",
                        f"{self.setname}_mask_{coco_idx}.npz",
                    )
                )["arr_0"].astype(int)
            except:
                return None

        # Albumentations handles conversion to torch tensor
        image = np.array(Image.open(img_path), dtype=np.float32)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        return {"image": image, "mask": mask}


class TreeDataModule(LightningDataModule):
    def __init__(self, conf, data_frac=1.0, augment=True):
        super().__init__()
        self.conf = conf
        self.data_frac = data_frac
        self.augment = augment

    def prepare_data(self) -> None:

        if self.augment:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    ToTensorV2(),
                ]
            )
            logger.info("Train-time augmentation is enabled.")
        else:
            transform = None

        self.train_data = ImageDataset("train", transform=transform)
        self.val_data = ImageDataset("val", transform=None)
        self.test_data = ImageDataset("test", transform=None)

    def train_dataloader(self):
        return get_dataloaders(self.conf, self.train_data, data_frac=self.data_frac)[0]

    def val_dataloader(self):
        return get_dataloaders(self.conf, self.val_data, data_frac=self.data_frac)[0]

    def test_dataloader(self):
        return get_dataloaders(self.conf, self.test_data, data_frac=self.data_frac)[0]


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

        if batch_idx < 10:

            loss, y_hat, y_hat_hard = self._predict_batch(batch)
            y = batch["mask"]
            self.log(f"val_loss", loss, on_step=False, on_epoch=True)
            self.val_metrics(y_hat, y)
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


def get_dataloaders(conf, *datasets, data_frac=1):
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
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=True,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
        )
        for dataset in datasets
    ]


def train():

    # init wandb
    wandb.init(
        # config=conf["model"],
        entity="dsl-ethz-restor",
        project=conf["wandb"]["project_name"],
    )
    # load data
    data_module = TreeDataModule(conf, augment=conf["datamodule"]["augment"] == "on")

    log_dir = os.path.join(LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # checkpoints and loggers
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=log_dir + "/checkpoints",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10
    )
    csv_logger = CSVLogger(save_dir=log_dir, name="logs")
    wandb_logger = WandbLogger(
        project=conf["wandb"]["project_name"], log_model=True
    )  # log_model='all' cache gets full quite fast

    wandb.run.summary["log_dir"] = log_dir

    # task
    task = SemanticSegmentationTaskPlus(
        segmentation_model=conf["model"]["segmentation_model"],
        encoder_name=conf["model"]["encoder_name"],
        encoder_weights="imagenet" if conf["model"]["pretrained"] == "True" else "None",
        in_channels=int(conf["model"]["in_channels"]),
        num_classes=int(conf["model"]["num_classes"]),
        loss=conf["model"]["loss"],
        ignore_index=None,
        learning_rate=float(conf["model"]["learning_rate"]),
        learning_rate_schedule_patience=int(
            conf["model"]["learning_rate_schedule_patience"]
        ),
    )

    # trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[csv_logger, wandb_logger],
        default_root_dir=log_dir,
        accelerator="gpu",
        max_epochs=int(conf["trainer"]["max_epochs"]),
        max_time=conf["trainer"]["max_time"],
        auto_lr_find=conf["trainer"]["auto_lr_find"] == "True",
        auto_scale_batch_size="binsearch"
        if (conf["trainer"]["auto_scale_batch_size"] == "True")
        else False,
    )

    wandb_logger.watch(task.model, log="parameters", log_graph=True)

    try:
        logger.info("Starting trainer")
        trainer.fit(model=task, datamodule=data_module)
    except Exception as e:
        logger.error("Training failed")
        logger.error()
        logger.error(traceback.print_exc())

    try:
        logger.info("Train complete, starting test")
        trainer.test(model=task, datamodule=data_module)
    except Exception as e:
        logger.error("Training failed")
        logger.error(e)
        logger.error(traceback.print_exc())


if __name__ == "__main__":

    # sweep: hyperparameter tuning
    project_name = conf["wandb"]["project_name"]
    logger.info(f"Using project {project_name}")

    if conf["experiment"]["sweep"] == "True":

        logger.info("Sweep enabled")

        sweep_file = "conf_sweep.yaml"
        with open(sweep_file, "r") as fp:
            conf_sweep = yaml.safe_load(fp)

        sweep_configuration = {
            "method": "grid",
            "name": "vanilla-model-sweep-runs",
            "program": "vanilla_model.py",
            "metric": {"goal": "minimize", "name": "loss"},
            "parameters": {
                "loss": {
                    "values": conf_sweep["parameters"]["loss"]["values"]
                },  # ['ce','focal']
                "segmentation_model": {
                    "values": conf_sweep["parameters"]["segmentation_model"]["values"]
                },  # ['unet','deeplabv3+']
                "encoder_name": {
                    "values": conf_sweep["parameters"]["encoder_name"]["values"]
                },  # ['resnet18','resnet34','resnet50']
                "augment": {
                    "values": conf_sweep["parameters"]["augment"]["values"]
                },  # ['off','on']
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(sweep_id=sweep_id, function=train)  # , count=5)

        logger.debug("Logging sweep config")
        wandb.log(sweep_configuration)

    torch.cuda.empty_cache()
    train()

    wandb.finish()
