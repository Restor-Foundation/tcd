import argparse
import math
import os
import random
import shutil
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.multiprocessing.set_sharing_strategy("file_system")

import logging

import albumentations as A
import torchmetrics
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.tuner.tuning import Tuner

import wandb

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, processor, tile_size=512, train=True, binary=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.processor = processor
        self.train = train
        self.tile_size = tile_size
        self.binary_labels = binary
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "masks")

        if train:
            self.annotations = COCO(os.path.join(root_dir, "train.json"))
        else:
            self.annotations = COCO(os.path.join(root_dir, "test.json"))

        # read images
        annotation_images = set(
            [img["file_name"] for img in self.annotations.imgs.values()]
        )
        self.image_names = []

        for image in annotation_images:
            mask_name = self._im_to_mask(image)

            if os.path.exists(os.path.join(self.ann_dir, mask_name)):
                self.image_names.append(image)

        # Remove any randomness from the sampler
        self.image_names = sorted(self.image_names)

        self.train_transform = A.Compose(
            [
                A.RandomCrop(width=tile_size, height=tile_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )

        logger.info(f"Found {len(self.image_names)} images with associated masks")

    def check_data(self):
        for sample in track(self):
            assert sample is not None

    @lru_cache
    def _im_to_mask(self, basename):
        return basename.replace(".tif", ".png")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]

        image = Image.open(os.path.join(self.img_dir, image_path))

        if len(image.getbands()) != 2:
            image = image.convert("RGB")

        segmentation_map = Image.open(
            os.path.join(self.ann_dir, self._im_to_mask(image_path))
        )
        image = np.array(image)

        segmentation_map = np.array(segmentation_map)

        if self.binary_labels:
            segmentation_map[segmentation_map != 0] = 1

        if self.train:
            augmented = self.train_transform(image=image, mask=segmentation_map)
            image = augmented["image"]
            segmentation_map = augmented["mask"]

        if self.binary_labels:
            if np.all(segmentation_map == 0):
                segmentation_map[0, 0] = 1
            elif np.all(segmentation_map == 1):
                segmentation_map[0, 0] = 0

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.processor(
            image, segmentation_map, reduce_size=False, return_tensors="pt"
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        encoded_inputs["raw_image"] = image

        return encoded_inputs


class SegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        processor=None,
        tile_size=512,
        dataset_root="../checkpoints/kfold_0/",
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_root = dataset_root
        self.processor = processor
        self.tile_size = tile_size

    def setup(self, stage: str):
        self.train_dataset = SemanticSegmentationDataset(
            root_dir=self.dataset_root,
            tile_size=self.tile_size,
            processor=self.processor,
            train=True,
        )
        self.test_dataset = SemanticSegmentationDataset(
            root_dir=self.dataset_root, processor=self.processor, train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, num_workers=4, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=4, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=4, batch_size=1, shuffle=False)


import json

from huggingface_hub import cached_download, hf_hub_url
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class LightningSegformer(LightningModule):
    def __init__(
        self,
        variant="nvidia/mit-b0",
        learning_rate=1e-4,
        learning_rate_schedule_patience=20,
        config=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # load id2label mapping from a JSON on the hub
        id2label = json.load(open("../checkpoints/index_to_name_binary.json", "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

        # define model
        if config is None:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                variant,
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                local_files_only=True if ckpt is not None else False,
            )
        else:
            self.model = SegformerForSemanticSegmentation.from_pretrained(config)

        self.jaccard_metric = torchmetrics.JaccardIndex(
            task="multiclass", threshold=0.5, num_classes=len(id2label)
        )

        self.f1_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=len(id2label)
        )

        self.accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=len(id2label)
        )

    def forward(self, x):
        return self.model.forward(x.pixel_values)

    def training_step(self, batch, batch_idx):
        loss = self.model(pixel_values=batch.pixel_values, labels=batch.labels).loss

        if batch_idx % 10 == 0:
            self.log("train/loss", loss, prog_bar=True)

        if self.current_epoch == 0 and batch_idx == 0:
            gt_image = batch.labels.cpu()

            self.logger.experiment.add_image(
                "train/images/rgb",
                batch.raw_image[0].cpu().numpy() / 255.0,
                global_step=self.trainer.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "train/images/gt",
                gt_image[0].numpy(),
                global_step=self.trainer.global_step,
                dataformats="HW",
            )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_batch_index = random.randint(
            0, len(self.trainer.val_dataloaders)
        )

        self.average_val_loss = []

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        self.jaccard_metric.update(predicted, labels)
        self.accuracy_metric.update(predicted, labels)
        self.f1_metric.update(predicted, labels)

        if batch_idx % 10 == 0:
            self.log("val/loss", outputs.loss, prog_bar=True)

        self.average_val_loss.append(float(outputs.loss))

        if batch_idx == self.validation_batch_index:
            input_image = batch.pixel_values.cpu().numpy()
            gt_image = batch.labels.cpu().numpy()
            predicted_image = predicted.cpu().numpy()

            self.logger.experiment.add_image(
                "val/images/rgb",
                batch.raw_image[0].cpu().numpy() / 255.0,
                global_step=self.trainer.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val/images/gt",
                gt_image[0],
                global_step=self.trainer.global_step,
                dataformats="HW",
            )
            self.logger.experiment.add_image(
                "val/images/pred",
                predicted_image[0],
                global_step=self.trainer.global_step,
                dataformats="HW",
            )

        return loss

    def on_validation_epoch_end(self):
        metric = self.jaccard_metric.compute()
        self.log("val/jaccard", metric)

        metric = self.f1_metric.compute()
        self.log("val/f1", metric)

        metric = self.accuracy_metric.compute()
        self.log("val/accuracy", metric)

        self.average_val_loss = np.array(self.average_val_loss)
        self.log("val/mean_loss", self.average_val_loss.mean())
        self.log("val/std_loss", self.average_val_loss.std())

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        # From https://pytorch.org/tutorials/intermediate
        # /torchvision_tutorial.html
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    verbose=True,
                    patience=self.hparams.learning_rate_schedule_patience,
                ),
                "monitor": "val/mean_loss",
                "frequency": self.trainer.check_val_every_n_epoch,
            },
        }

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nvidia/mit-b0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="../checkpoints/kfold_0")
    parser.add_argument("--output_dir", type=str, default="../models/segformer")
    parser.add_argument("--wandb", type=str, default=None)
    args = parser.parse_args()

    seed = 42
    seed_everything(seed)

    log_root = os.path.join(
        args.output_dir, args.model.split("/")[-1], os.path.basename(args.dataset)
    )

    # Handle automatic resume from the last saved ckpt file
    ckpt = None
    if args.resume:
        ckpt = os.path.join(log_root, "checkpoints", "last.ckpt")
        assert os.path.exists(ckpt), "Last checkpoint file doesn't exist"
    else:
        assert not os.path.exists(
            os.path.join(log_root, "checkpoints", "last.ckpt")
        ), "Existing checkpoint exists, you might want to resume."

    # Since these loggers will try to save to separate folders otherwise, we re-use the 'version' from CSV
    # If the run is resumed, we need to make a new logging folder as Lightning won't append
    logger.info(f"Logging to root folder: {log_root}")
    csv_logger = CSVLogger(save_dir=log_root, name="logs")
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_root, name="logs", version=csv_logger.version
    )
    loggers = [tensorboard_logger, csv_logger]

    if args.wandb:
        wandb_logger = WandbLogger(
            save_dir=log_root,
            log_model="all",
            project="tcd-segformer",
            entity=args.wandb,
            version=None,
        )
        os.makedirs(os.path.join(log_root, "wandb"), exist_ok=True)
        loggers.append(wandb_logger)

    matmul_precision = "medium"
    torch.set_float32_matmul_precision(matmul_precision)

    best_checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=os.path.join(log_root, "checkpoints"),
        auto_insert_metric_name=True,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_root, "checkpoints"),
        filename="epoch={epoch}-val_iou={val/jaccard:.2f}",
        auto_insert_metric_name=False,
        every_n_epochs=100,
        save_top_k=-1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    target_batch = 32
    accumulate_grad = math.ceil(target_batch / args.batch_size)

    # Passing tensorboard_logger first allows us to access it via self.logger.experiment
    # - hacky, but works well enough.
    trainer = Trainer(
        max_epochs=800,
        accelerator="auto",
        callbacks=[lr_monitor, epoch_checkpoint_callback, best_checkpoint_callback],
        logger=loggers,
        check_val_every_n_epoch=10,
        accumulate_grad_batches=accumulate_grad,
    )

    if args.wandb:
        wandb_logger.experiment.config.update(
            {
                "model": args.model,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "dataset": args.dataset,
                "matmul_precision": matmul_precision,
            }
        )

    # Load the processor and model etc after we've loaded the checkpoint
    # as some of this can be preloaded:
    if not args.resume:
        processor = SegformerImageProcessor.from_pretrained(
            args.model,
            do_resize=False,
            do_reduce_labels=False,
            local_files_only=True if ckpt is not None else False,
        )

        processor.save_pretrained(os.path.join(log_root, "checkpoints"))
    else:
        processor = SegformerImageProcessor.from_pretrained(
            os.path.join(log_root, "checkpoints")
        )

    # Depends on the processor
    dm = SegmentationDataModule(
        batch_size=args.batch_size,
        dataset_root=args.dataset,
        tile_size=args.tile_size,
        processor=processor,
        nworkers=4,
    )

    # Load the checkpoint if we have it; technically we shouldn't need to but loading
    # the config from cache sometimes fails.
    if not args.resume:
        model = LightningSegformer(args.model, learning_rate=args.learning_rate)
        model.model.save_pretrained(os.path.join(log_root, "checkpoints"))
    else:
        model = LightningSegformer(
            args.model, config=os.path.join(log_root, "checkpoints")
        )

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt)
    shutil.copy(
        os.path.join(csv_logger.log_dir, "hparams.yaml"),
        os.path.join(log_root, "checkpoints"),
    )

    wandb.finish()
