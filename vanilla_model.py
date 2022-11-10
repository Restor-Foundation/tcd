import argparse
import configparser
import json
import yaml
import os
import sys
import time
import warnings
from ctypes import cast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
import torchgeo
import torchvision
import yaml
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
    ConfusionMatrix,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)
from torchvision.utils import draw_segmentation_masks

import wandb
from utils import downsample

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

parser.add_argument("--segmentation_model", type=str, default=None)
parser.add_argument("--loss", type=str, default=None)
parser.add_argument("--backbone", type=str, default=None)
parser.add_argument("--learning_rate", type=float, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--factor", type=int, default=None)

args = parser.parse_args()
conf = configparser.ConfigParser()
conf.read(args.conf)

if args.factor:
    FACTOR = args.factor
else:
    FACTOR = conf["experiment"]["factor"]

if conf["experiment"]["setup"] == "True":
    from utils import clean_data

if FACTOR != "1":
    if not os.path.exists(
        f"{DATA_DIR}images/downsampled_images/sampling_factor_{FACTOR}"
    ):
        downsample.sampler(int(FACTOR))

# sweep: hyperparameter tuning
if conf["experiment"]["sweep"] == "True":
    sweep_file = "conf_sweep.yaml" 
    with open(sweep_file, "r") as fp:
      conf_sweep = yaml.safe_load(fp)
    sweep_id = wandb.sweep(sweep=conf_sweep, project="vanilla-model-sweep-runs")
    wandb.agent(sweep_id=sweep_id) #function= , count= ,

# if the imports throw OMP error #15, try $ conda install nomkl
# or, as an unsafe quick fix like above, import os; os.environ['KMP_DUPLICATE_LIB_OK']='True';

# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, transform=None, target_transform=None):

        self.data_dir = DATA_DIR
        self.setname = setname
        assert setname in ["train", "test", "val"]

        with open(self.data_dir + setname + "_20221010.json", "r") as file:
            self.metadata = json.load(file)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata["images"])

    def __getitem__(self, idx):
        annotation = self.metadata["images"][idx]

        img_name = annotation["file_name"]
        coco_idx = annotation["id"]

        if FACTOR == 1:
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
                f"sampling_factor_{FACTOR}/{img_name}",
            )

            mask = np.load(
                os.path.join(
                    self.data_dir,
                    "downsampled_masks",
                    f"sampling_factor_{FACTOR}",
                    f"{self.setname}_mask_{coco_idx}.npz",
                )
            )["arr_0"].astype(int)
        try:
            image = torch.Tensor(np.array(Image.open(img_path)))
        except:
            return None
        image = torch.permute(image, (2, 0, 1))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return {"image": image, "mask": mask}


class TreeDataModule(LightningDataModule):
    def __init__(self, conf, data_frac=1.0):
        super().__init__()
        self.conf = conf
        self.data_frac = data_frac

    def prepare_data(self) -> None:
        self.train_data, self.val_data, self.test_data = (
            ImageDataset("train"),
            ImageDataset("val"),
            ImageDataset("test"),
        )

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

        self.train_metrics = MetricCollection(
            [
                Accuracy(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                JaccardIndex(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                Precision(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                Recall(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                F1Score(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                ConfusionMatrix(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def log_image(self, image, key, caption=""):
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
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
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
                }
                resize = torchvision.transforms.Resize(512)
                image_grid = torchvision.utils.make_grid(
                    [resize(value.float()) for key, value in images.items()],
                    value_range=(0, 255),
                    normalize=True,
                )
                self.log_image(
                    image_grid,
                    key="val_examples (original/groud truth/prediction)",
                    caption="Sample validation images",
                )
                wandb.log(
                    {
                        "pr": wandb.plot.pr_curve(
                            torch.reshape(batch["mask"][0], (-1,)),
                            torch.reshape(
                                y_hat[0].cpu(), (-1, self.hyperparams["num_classes"])
                            ),
                            labels=None,
                            classes_to_plot=None,
                        )
                    }
                )
            except AttributeError:
                pass

    def test_step(self, *args, **kwargs) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

        try:
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
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
            }
            resize = torchvision.transforms.Resize(512)
            image_grid = torchvision.utils.make_grid(
                [resize(value.float()) for key, value in images.items()],
                value_range=(0, 255),
                normalize=True,
            )
            self.log_image(
                image_grid,
                key="test_examples (original/groud truth/prediction)",
                caption="Sample test images",
            )
        except AttributeError:
            pass

    def training_epoch_end(self, outputs):
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        conf_mat = computed["train_ConfusionMatrix"].cpu().numpy()
        conf_mat = (conf_mat / np.sum(conf_mat)) * 100
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["train_ConfusionMatrix"])
        }
        cm = px.imshow(conf_mat, text_auto=".2f")
        wandb.log({"train_confusion_matrix": cm})
        self.log_dict(new_metrics)
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs):
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        computed = self.val_metrics.compute()
        conf_mat = computed["val_ConfusionMatrix"].cpu().numpy()
        conf_mat = (conf_mat / np.sum(conf_mat)) * 100
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["val_ConfusionMatrix"])
        }
        cm = px.imshow(conf_mat, text_auto=".2f")
        wandb.log({"val_confusion_matrix": cm})
        self.log_dict(new_metrics)
        self.val_metrics.reset()

    def test_epoch_end(self, outputs):
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        computed = self.test_metrics.compute()
        conf_mat = computed["test_ConfusionMatrix"].cpu().numpy()
        conf_mat = (conf_mat / np.sum(conf_mat)) * 100
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["test_ConfusionMatrix"])
        }
        fig = px.imshow(conf_mat, text_auto=".2f")
        wandb.log({"test_confusion_matrix": fig})
        self.log_dict(new_metrics)
        self.test_metrics.reset()


def get_dataloaders(conf, *datasets, data_frac=1.0):
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


if __name__ == "__main__":

    wandb.init(entity="dsl-ethz-restor", project="vanilla-model-sweep-runs")

    if args.segmentation_model is not None:
        conf["model"]["segmentation_model"] = args.segmentation_model
        
    if args.loss is not None:
        conf["model"]["loss"] = args.loss

    if args.backbone is not None:
        conf["model"]["backbone"] = args.backbone

    # load data
    data_module = TreeDataModule(conf)

    log_dir = LOG_DIR + time.strftime("%Y%m%d-%H%M%S")

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
    wandb_logger = WandbLogger(project="vanilla-model-sweep-runs", log_model=True) #log_model='all' cache gets full quite fast

    # task
    task = SemanticSegmentationTaskPlus(
        segmentation_model=conf["model"]["segmentation_model"],
        encoder_name=conf["model"]["backbone"],
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
        auto_scale_batch_size= 'binsearch' if (conf["trainer"]["auto_scale_batch_size"] == "True") else False,
    )

    trainer.fit(task, datamodule=data_module)

    trainer.test(model=task, datamodule=data_module)

    wandb.finish()
