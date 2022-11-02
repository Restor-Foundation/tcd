import argparse
import configparser
import json
import os
import sys
from ctypes import cast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchgeo
import yaml
from decouple import config
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import SemanticSegmentationTask

from utils import downsample

wandb_logger = WandbLogger(log_model="all", project="vanilla-model")


DATA_DIR = config("DATA_DIR")
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
args = parser.parse_args()

conf = configparser.ConfigParser()
conf.read(args.conf)

FACTOR = conf["experiment"]["factor"]

if conf["experiment"]["setup"] == "True":
    from utils import clean_data

if FACTOR != "1":
    if not os.path.exists(
        f"{DATA_DIR}images/downsampled_images/sampling_factor_{FACTOR}"
    ):
        downsample.sampler(int(FACTOR))

# if the imports throw OMP error #15, try $ conda install nomkl
# or, as an unsafe quick fix like above, import os; os.environ['KMP_DUPLICATE_LIB_OK']='True';

# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, transform=None, target_transform=None):

        # Define the  mask file and the json file for retrieving images
        # self.data_dir = os.getcwd()
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
        img_name = list(self.metadata.items())[2][1][idx]["file_name"]
        if FACTOR == "1":
            img_path = os.path.join(self.data_dir, "images", img_name)
            mask = np.load(
                self.data_dir + "masks/" + self.setname + "_mask_" + str(idx) + ".npz"
            )["arr_0"].astype(int)
        else:
            img_path = (
                f"{self.data_dir}downsampled_images/sampling_factor_{FACTOR}/{img_name}"
            )
            mask = np.load(
                self.data_dir
                + f"downsampled_masks/sampling_factor_{FACTOR}/"
                + self.setname
                + "_mask_"
                + str(idx)
                + ".npz"
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


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":

    # create datasets
    setname = "train"
    train_data = ImageDataset(setname)
    setname = "val"
    val_data = ImageDataset(setname)
    setname = "test"
    test_data = ImageDataset(setname)

    # DataLoader
    train_dataloader = DataLoader(
        train_data,
        batch_size=int(conf["datamodule"]["batch_size"]),
        shuffle=True,
        num_workers=int(conf["datamodule"]["num_workers"]),
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=int(conf["datamodule"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["datamodule"]["num_workers"]),
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=int(conf["datamodule"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["datamodule"]["num_workers"]),
        collate_fn=collate_fn,
    )

    experiment_dir = os.path.join("REPO_DIR", "results")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=experiment_dir, save_top_k=1, save_last=True
    )

    csv_logger = CSVLogger(save_dir=experiment_dir, name="logs")

    # set up task
    task = SemanticSegmentationTask(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        default_root_dir=experiment_dir,
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

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=int(conf["trainer"]["max_epochs"]),
        max_time=conf["trainer"]["max_time"],
        logger=wandb_logger,
        auto_lr_find=conf["trainer"]["auto_lr_find"] == "True",
        auto_scale_batch_size=conf["trainer"]["auto_scale_batch_size"] == "True",
    )
    trainer.fit(task, train_dataloader, val_dataloader)

    trainer.test(model=task, dataloaders=test_dataloader, verbose=True)
