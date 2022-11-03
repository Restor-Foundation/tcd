import argparse
import configparser
import json
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

import wandb

# TODO fix warnings
warnings.filterwarnings("ignore")

DATA_DIR = config("DATA_DIR")
LOG_DIR = config("LOG_DIR")
REPO_DIR = config("REPO_DIR")


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
        return len(list(self.metadata.items())[2][1])

    def __getitem__(self, idx):
        img_name = self.metadata["images"][idx]["file_name"]
        img_path = os.path.join(self.data_dir, "images", img_name)
        try:
            image = torch.Tensor(np.array(Image.open(img_path)))
        except:
            return None
        image = torch.permute(image, (2, 0, 1))

        mask = np.load(
            self.data_dir + "masks/" + self.setname + "_mask_" + str(idx) + ".npz"
        )["arr_0"].astype(int)

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

    def plot(self, sample):
        img, mask, pred = sample["image"], sample["mask"], sample["prediction"]
        # pred_mask = pred.argmax(axis=0)
        WBMASK = wb_mask(
            img.cpu().numpy().transpose(1, 2, 0), pred.cpu().numpy(), mask.cpu().numpy()
        )
        return WBMASK


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
            print(
                "1###################################################################################################"
            )
            print("I am still alive")
            try:
                datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                new_batch = {
                    "image": batch["image"],
                    "mask": batch["mask"].unsqueeze(1),
                    "prediction": batch["prediction"].unsqueeze(1),
                }
                for key in ["image", "mask", "prediction"]:
                    print(new_batch[key].shape)
                # sample = unbind_samples(batch)[0]
                # fig = datamodule.plot(sample)
                print(
                    "2###################################################################################################"
                )
                print("I am still alive and I am about to log shit")
                resize = torchvision.transforms.Resize(512)
                image_grid = torchvision.utils.make_grid(
                    [resize(value.float()) for key, value in new_batch.items()],
                    value_range=(0, 255),
                    normalize=True,
                )
                print(
                    "69###################################################################################################"
                )
                print(
                    "I am still alive and I am about to log shit printing just to be safe"
                )
                self.log_image(
                    image_grid, key="val_examples", caption="Sample validation images"
                )
                # summary_writer = self.logger.experiment  # type: ignore[union-attr]
                # wandb.log({'images (original/original mask/prediction)': fig})
                print(
                    "#3##################################################################################################"
                )
                print("I am still alive and have looged shit")
            except AttributeError as e:
                print(e)
                print("I am about to die :(")
                exit()
                pass

    def training_epoch_end(self, outputs):
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        conf_mat = computed["train_ConfusionMatrix"].cpu().numpy()
        conf_mat = (conf_mat / np.sum(conf_mat)) * 100
        # df_cm = pd.DataFrame(
        #     conf_mat,
        #     index=range(self.hyperparams["num_classes"]),
        #     columns=range(self.hyperparams["num_classes"]),
        # )
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["train_ConfusionMatrix"])
        }
        fig = px.imshow(conf_mat, text_auto=".2f")
        # wandb.log({"train_ConfusionMatrix": wandb.Table(dataframe=df_cm)})
        wandb.log({"train_confusion_matrix": fig})
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
        # df_cm = pd.DataFrame(
        #     conf_mat,
        #     index=range(self.hyperparams["num_classes"]),
        #     columns=range(self.hyperparams["num_classes"]),
        # )
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["val_ConfusionMatrix"])
        }
        fig = px.imshow(conf_mat, text_auto=".2f")
        # wandb.log({"val_ConfusionMatrix": wandb.Table(dataframe=df_cm)})
        wandb.log({"val_confusion_matrix": fig})
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
        # df_cm = pd.DataFrame(
        #     conf_mat,
        #     index=range(self.hyperparams["num_classes"]),
        #     columns=range(self.hyperparams["num_classes"]),
        # )
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["test_ConfusionMatrix"])
        }
        fig = px.imshow(conf_mat, text_auto=".2f")
        # wandb.log({"test_ConfusionMatrix": wandb.Table(dataframe=df_cm)})
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


try:
    conf = configparser.ConfigParser()
    conf.read("conf.yaml")
except Exception as e:
    pass  # no conf

segmentation_classes = ["no tree", "tree"]


def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l


def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(
        bg_img,
        masks={
            "prediction": {"mask_data": pred_mask, "class_labels": labels()},
            "ground truth": {"mask_data": true_mask, "class_labels": labels()},
        },
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        nargs="?",
        const=True,
        default="conf.yaml",
        help="Choose config file for setup",
    )
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    conf.read(args.conf)

    if conf["experiment"]["setup"] == "True":
        from utils import clean_data

    wandb.init(entity="dsl-ethz-restor", project="vanilla-model-more-metrics")

    # # create datasets
    # setname = "train"
    # train_data = ImageDataset(setname)
    # setname = "val"
    # val_data = ImageDataset(setname)
    # setname = "test"
    # test_data = ImageDataset(setname)

    # # DataLoader
    # train_dataloader = DataLoader(train_data, batch_size=int(conf['datamodule']['batch_size']),
    #                               shuffle=True, num_workers=int(conf['datamodule']['num_workers']),collate_fn=collate_fn)
    # val_dataloader = DataLoader(val_data, batch_size=int(conf['datamodule']['batch_size']),
    #                             shuffle=False, num_workers=int(conf['datamodule']['num_workers']),collate_fn=collate_fn)
    # test_dataloader = DataLoader(test_data, batch_size=int(conf['datamodule']['batch_size']),
    #                              shuffle=False, num_workers=int(conf['datamodule']['num_workers']),collate_fn=collate_fn)

    # load data
    data_module = TreeDataModule(conf, data_frac=0.05)

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
    wandb_logger = WandbLogger(project="vanilla-model-more-metrics", log_model="all")

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
        max_epochs=2,  # int(conf["trainer"]["max_epochs"]),
        max_time="00:23:50:00",
    )

    trainer.fit(task, datamodule=data_module)

    trainer.test(model=task, datamodule=data_module)

    wandb.finish()
