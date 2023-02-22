"""Restor TCD OAM Dataset."""

import argparse
import copy
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.multiprocessing
import torchmetrics
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer
from kornia.augmentation.container import AugmentationSequential
from pycocotools.coco import COCO
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from rich import progress
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import Compose, Normalize
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

torch.multiprocessing.set_sharing_strategy("file_system")

import sys
from dataclasses import dataclass

import detectron2
import detectron2.data.transforms as T
import pytorch_lightning
import torchvision
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.structures.boxes import BoxMode

# See https://pytorch.org/docs/stable/tensors.html
from torch import ByteTensor, FloatTensor, LongTensor
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn_v2
from torchvision.models.resnet import ResNet50_Weights

if not sys.warnoptions:
    import warnings

    warnings.filterwarnings(
        "ignore", message="Iteration over multi-part geometries is deprecated"
    )

register_coco_instances(
    name="tcd_train",
    metadata={},
    json_file="../data/restor-tcd-oam/train_20221010_noempty.json",
    image_root="../data/restor-tcd-oam/images/",
)

register_coco_instances(
    name="tcd_val",
    metadata={},
    json_file="../data/restor-tcd-oam/val_20221010_noempty.json",
    image_root="../data/restor-tcd-oam/images/",
)

register_coco_instances(
    name="tcd_test",
    metadata={},
    json_file="../data/restor-tcd-oam/test_20221010_noempty.json",
    image_root="../data/restor-tcd-oam/images/",
)


def load_biomes(path="../data/restor/oam_wwf_biome_map.csv"):
    id_biome_map = pd.read_csv("./oam_wwf_biome_map.csv")
    id_biome_map.fillna(-1, inplace=True)

    out = {}
    for (index, row) in id_biome_map.iterrows():
        biome_id = int(row.wwf_biome_id)
        out[row.id] = biome_id

    return out


id_biome_map = load_biomes()


class CrowdDatasetMapper(DatasetMapper):
    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj,
                transforms,
                image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            )
            for obj in dataset_dict.pop("annotations")
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes and len(instances) > 0:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        try:
            oam_id = dataset_dict["file_name"].split("_")[0]
            biome_id = id_biome_map[oam_id]

            if biome_id < 1 or biome_id > 14:
                biome_id = 15  # Unspecified

            dataset_dict["biome"] = biome_id
        except KeyError:
            dataset_dict["biome"] = 15

    def __call__(self, dataset_dict):

        out = super().__call__(dataset_dict)

        return out


def detectron2_to_pytorch(batch):

    images = []
    targets = []

    for sample in batch:
        new_target = {}

        images.append(sample["image"].float())

        if sample.get("instances"):
            instances = sample.get("instances")

            new_target["masks"] = instances.get("gt_masks").tensor

            # Convert from Detectron2/COCO > Pytorch XYXY
            boxes = instances.get("gt_boxes")
            new_target["boxes"] = BoxMode.convert(
                boxes.tensor, from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS
            )

            new_target["labels"] = instances.get("gt_classes")
            new_target["biome"] = torch.tensor(sample.get("biome", 15)).byte()
            targets.append(new_target)

        else:

            _, h, w = sample["image"].shape
            n_class = 2
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            new_target["masks"] = torch.zeros(size=(1, h, w)).bool()
            new_target["boxes"] = torch.tensor([[x, y, x + 1, y + 1]]).float()
            new_target["labels"] = torch.tensor([random.randint(0, n_class)]).long()
            new_target["biome"] = torch.tensor([15])

            targets.append(new_target)

    images = torch.stack(images) / 255.0

    return images, targets


def get_detectron2_dataloader(
    cfg,
    dataset_name,
    augment=False,
    batch_size=4,
    mode="train",
    classes=["canopy", "tree"],
    colors=[(254, 20, 147), (204, 255, 0)],
):

    dataset = DatasetCatalog.get(dataset_name)

    if classes and not MetadataCatalog.get(dataset_name).thing_classes:
        MetadataCatalog.get(dataset_name).thing_classes = classes

    if colors:
        MetadataCatalog.get(dataset_name).thing_colors = colors

    # D4 symmetry transform + crop to tile
    if augment:
        transforms = [
            T.RandomRotation(180),
            T.RandomFlip(horizontal=False, vertical=True),
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomCrop(crop_type="absolute", crop_size=(1024, 1024)),
        ]
    else:
        transforms = []

    if mode == "train":
        return build_detection_train_loader(
            cfg,
            dataset=dataset,
            total_batch_size=batch_size,
            collate_fn=detectron2_to_pytorch,
            num_workers=0,
            aspect_ratio_grouping=False,
            sampler=TrainingSampler(size=len(dataset), shuffle=True),
            mapper=CrowdDatasetMapper(
                is_train=True,
                use_instance_mask=True,
                instance_mask_format="bitmask",
                recompute_boxes=True,
                image_format="RGB",
                augmentations=transforms,
            ),
        )
    else:
        from torch.utils.data import RandomSampler

        """
        if mode == 'val':
            sampler = RandomSampler(data_source=dataset)
        else:
            sampler = None
        """
        sampler = None

        return build_detection_test_loader(
            dataset=dataset,
            batch_size=1,
            collate_fn=detectron2_to_pytorch,
            num_workers=0,
            sampler=sampler,
            mapper=CrowdDatasetMapper(
                is_train=True,
                use_instance_mask=True,
                instance_mask_format="bitmask",
                recompute_boxes=True,
                image_format="RGB",
                augmentations=[],
            ),
        )


class TCDDetectron2DataModule(LightningDataModule):
    def __init__(self, train="tcd_train", val="tcd_val", test="tcd_test", batch_size=2):

        self.cfg = get_cfg()  # obtain detectron2's default config
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        self.cfg.DATALOADER.NUM_WORKERS = 1

        self._train_dataset_name = train
        self._val_dataset_name = val
        self._test_dataset_name = test
        self.batch_size = batch_size

        super().__init__()

    @property
    def classes(self):
        return {1: "canopy", 2: "tree"}

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        return get_detectron2_dataloader(
            self.cfg,
            self._train_dataset_name,
            augment=True,
            mode="train",
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return get_detectron2_dataloader(
            self.cfg, self._test_dataset_name, mode="val", batch_size=1
        )

    def test_dataloader(self):
        return get_detectron2_dataloader(
            self.cfg, self._test_dataset_name, mode="test", batch_size=1
        )


class TCDMaskRCNN(LightningModule):
    def get_model_instance_segmentation(self, num_classes):

        # Background
        num_classes += 1

        if "pretrained" in self.hyperparams and self.hyperparams["pretrained"] == True:
            weights = (
                torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
            print("Using pretrained weights")
        else:
            weights = None

        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=weights,
            image_mean=(111.1589 / 255.0, 114.4200 / 255.0, 91.1464 / 255.0),
            image_std=(61.1331 / 255.0, 59.4855 / 255.0, 56.6022 / 255.0),
        )

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    def __init__(self, **kwargs: Any):
        super().__init__()

        self.biome_weight = 1.0

        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.model = self.get_model_instance_segmentation(2)

        if self.hyperparams["predict_biome"]:
            self.biome_model = torchvision.models.resnet34(
                progress=True, weights=torchvision.models.ResNet34_Weights.DEFAULT
            )
            self.biome_model.fc = nn.Linear(512, 16)

        # Compute both segmentation and bounding box mAPs
        self.valid_map_segm = MeanAveragePrecision(
            iou_type="segm", box_format="xyxy", class_metrics=True
        )

        self.valid_map_bbox = MeanAveragePrecision(
            iou_type="bbox", box_format="xyxy", class_metrics=True
        )

        self.valid_biome_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=4
        )

        self.log_n_images = 5

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, targets = batch

        if len(targets) == 0:
            return None

        images = images.to("cuda")

        # Assume images here are in [0,255]. Normalisation is done inside
        # the model itself.

        assert images.max() <= 1

        loss_dict = self.model.forward(images, targets=targets)

        if self.hyperparams["predict_biome"]:
            biome_preds = self.biome_model.forward(images)
            target_biomes = torch.ByteTensor([target["biome"] for target in targets])

            from torch.nn import CrossEntropyLoss

            loss = nn.CrossEntropyLoss()
            biome_loss = loss(biome_preds, target_biomes)
        else:
            biome_loss = 0

        train_loss = sum(loss_dict.values()) + biome_loss * self.biome_weight
        [self.log(f"train/losses/{key}", loss_dict[key]) for key in loss_dict]
        self.log("train/losses/total_loss", train_loss)

        if batch_idx == 0 or batch_idx % 100 == 0:

            with torch.no_grad():
                self.model.eval()
                preds = self.model.forward(images, targets=targets)

                for pred in preds:
                    pred["masks"] = (pred["masks"] > 0.5).squeeze(1).bool()

                self._log_image(
                    images[0], preds[0], tag=f"train/sample", step=batch_idx
                )

            self.model.train()

        return cast(Tensor, train_loss)

    def on_validation_epoch_start(self):
        self.n_val_images = len(self.trainer.datamodule.val_dataloader().dataset)
        self.validation_plot_indices = torch.randint(
            0, self.n_val_images, (self.log_n_images,)
        )

    def _log_image(self, image, label_dict, tag="", step=None, thresh=0.5):

        if step is None:
            step = self.global_step

        masks = label_dict["masks"].detach().cpu()
        boxes = label_dict["boxes"].detach().cpu()
        image = image.detach().cpu().byte()

        # Discard bad predictions
        if "scores" in label_dict["masks"]:
            score_mask = label_dict["scores"] > thresh
            masks = masks[score_mask]
            boxes = masks[boxes]

        tensorboard = self.logger.experiment

        if len(boxes) > 0:
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)

            image_plot = draw_segmentation_masks(image, masks, alpha=0.9)
            # image_plot = draw_bounding_boxes(image_plot, boxes)

            tensorboard.add_image(tag, image_plot, global_step=step)

            # masks = torch.any(masks, dim=0)
            # tensorboard.add_image(tag+"_mask", masks.byte()*255, dataformats='HW')
        else:
            tensorboard.add_image(tag, image, global_step=step)

    def validation_step(self, batch, batch_idx, thresh=0.5):

        images, targets = batch

        assert images.max() <= 1

        preds = self.model.forward(images)

        # 'Hard'en masks, 0.5 thresh
        for pred in preds:
            pred["masks"] = (pred["masks"] > thresh).squeeze(1).bool()

        self.valid_map_segm.update(target=targets, preds=preds)
        self.valid_map_bbox.update(target=targets, preds=preds)

        if self.hyperparams["predict_biome"]:
            biome_preds = self.biome_model.forward(images)
            preds_hard = biome_preds.softmax(dim=0).argmax(dim=1)
            self.valid_biome_accuracy.update(
                target=torch.ByteTensor([target["biome"] for target in targets]),
                preds=preds_hard,
            )

        if batch_idx < 5:
            self._log_image(images[0], preds[0], tag=f"val/pred_{batch_idx}")
            self._log_image(images[0], targets[0], tag=f"val/target_{batch_idx}")

    def _log_map(self, metric, prefix="val/"):
        """
        Log mAP results
        """

        map_result = {prefix + k: v for k, v in metric.compute().items()}
        mAPs_per_class = map_result.pop(f"{prefix}map_per_class")
        mARs_per_class = map_result.pop(f"{prefix}mar_100_per_class")
        self.log_dict(map_result, sync_dist=True)
        self.log_dict(
            {
                f"{prefix}map_{label}": value
                for label, value in zip(
                    self.trainer.datamodule.classes.values(), mAPs_per_class
                )
            },
            sync_dist=True,
        )
        self.log_dict(
            {
                f"{prefix}mar_100_{label}": value
                for label, value in zip(
                    self.trainer.datamodule.classes.values(), mARs_per_class
                )
            },
            sync_dist=True,
        )

    def validation_epoch_end(self, outputs):

        self._log_map(self.valid_map_bbox, "val/bbox/")
        self._log_map(self.valid_map_segm, "val/segm/")
        self.log("val/biome_accuracy", self.valid_biome_accuracy.compute())

        self.valid_map_bbox.reset()
        self.valid_map_segm.reset()
        self.valid_biome_accuracy.reset()

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        # From https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"],
            momentum=0.9,
            weight_decay=0.0005,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val/bbox/map",
            },
        }


def train(args):

    pytorch_lightning.seed_everything(42, workers=True)

    batch_size = 6
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./output/maskrcnn_pt/")
    checkpoint_dir = "./output/maskrcnn_pt/"

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_train_steps=4000,
        save_top_k=2,
        monitor="val/bbox/map",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    datamodule = TCDDetectron2DataModule(batch_size=batch_size)

    trainer = pytorch_lightning.Trainer(
        default_root_dir=checkpoint_dir,
        callbacks=[lr_monitor, checkpoint_callback],
        # accumulate_grad_batches=32//batch_size,
        val_check_interval=4000,
        log_every_n_steps=5,
        logger=tb_logger,
        max_steps=20000,
        limit_val_batches=100,
        accelerator="gpu",
    )

    model = TCDMaskRCNN(
        learning_rate_schedule_patience=10,
        learning_rate=1e-3,
        n_class=2,
        pretrained=args.pretrained,
    )

    print("Training")
    trainer.fit(model=model, datamodule=datamodule)


def validate(args):
    pytorch_lightning.seed_everything(42, workers=True)

    batch_size = 6
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./output/maskrcnn_pt/")
    checkpoint_dir = "./output/maskrcnn_pt/"

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, save_top_k=2, monitor="val/bbox/map"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    datamodule = TCDDetectron2DataModule(batch_size=batch_size)

    trainer = pytorch_lightning.Trainer(
        default_root_dir=checkpoint_dir,
        callbacks=[lr_monitor, checkpoint_callback],
        accumulate_grad_batches=32 // batch_size,
        val_check_interval=10000 // batch_size,
        log_every_n_steps=5,
        logger=tb_logger,
        accelerator="gpu",
    )

    model = TCDMaskRCNN(
        learning_rate_schedule_patience=10,
        learning_rate=1e-3,
        n_class=2,
        pretrained=args.pretrained,
    )

    print("Validation")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=args.weights)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(dest="mode", choices=["train", "val"])
    parser.add_argument(
        "--weights", type=str, help="Pre-trained weights for validation/testing"
    )
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "val":
        validate(args)
