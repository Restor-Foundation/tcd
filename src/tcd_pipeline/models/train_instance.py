"""Instance segmentation model framework, using Detectron2 as the backend."""

import gc
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Union

import detectron2.data.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torchvision
import wandb
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class RandomScale(T.Augmentation):
    """
    Outputs an image scaled by a multiplicative factor.
    """

    def __init__(self, scale_range):
        """
        Args:
            scale_range (l, h): Range of input-to-output size scaling factor. For a fixed scale, set l == h
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img_h, img_w = image.shape[:2]
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])

        return T.ScaleTransform(
            img_h,
            img_w,
            int(img_h * scale_factor),
            int(img_w * scale_factor),
            interp="bilinear",
        )


class Trainer(DefaultTrainer):
    """
    Custom trainer class for Detectron2.

    Allows control over augmentation and other
    parameters.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name) -> COCOEvaluator:
        """Build a COCO segmentation evaluation task.

        Args:
            cfg (CfgNode): Configuration object
            dataset_name (str): Name of the dataset to evaluate on

        Returns:
            dataset_evaluator (COCOEvaluator): COCO evaluation task
        """
        return COCOEvaluator(
            dataset_name,
            tasks=["segm"],
            allow_cached_coco=False,
            max_dets_per_image=256,
            output_dir=cfg.OUTPUT_DIR,
        )

    @classmethod
    def build_train_loader(cls, cfg) -> DataLoader:
        """
        Returns:
            iterable

        Train loader with extra augmentation

        """

        augs = [
            T.RandomFlip(vertical=True, horizontal=False, prob=0.5),
            T.RandomFlip(horizontal=True, vertical=False, prob=0.5),
            T.RandomContrast(0.75, 1.25),
            T.RandomBrightness(0.75, 1.25),
            T.RandomSaturation(0.75, 1.25),
        ]  # type: T.Augmentation

        if cfg.INPUT.SCALE_FACTOR != 1:
            augs.append(
                RandomScale(
                    scale_range=(cfg.INPUT.SCALE_FACTOR, cfg.INPUT.SCALE_FACTOR)
                )
            )

        augs.extend(
            [
                T.RandomApply(
                    T.RandomRotation(
                        (0, 90 if cfg.INPUT.SCALE_FACTOR == 1 else 10), expand=True
                    ),
                    prob=0.5,
                ),
                T.FixedSizeCrop(
                    crop_size=(cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
                ),
            ]
        )

        # Override the augmentations loaded in the config
        return build_detection_train_loader(
            cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs)
        )


class TrainExampleHook(HookBase):
    """Train-time hook that logs example images to wandb before training."""

    def __init__(self, config, val_name, n_examples=5):
        self._config = config
        self.val_name = val_name
        self.n_examples = n_examples
        self.conf_thresh = 0.5

    def log_image(self, image: torch.Tensor, key: str) -> None:
        """Log an image to Tensorboard.

        Args:
            image (torch.Tensor): Image to log
            key (str): Key to use for logging

        """
        self.trainer.storage.put_image(key, image)

    def after_step(self) -> None:
        if self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            return

        if (
            self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD
        ) == 1 and self.trainer.iter > self.trainer.cfg.TEST.EVAL_PERIOD:
            resize = torchvision.transforms.Resize(512)

            from pycocotools.coco import COCO

            # register_coco_instances(
            eval_metadata = MetadataCatalog.get(self.val_name)
            gt = COCO(eval_metadata.get("json_file"))
            res = gt.loadRes(
                os.path.join(self.trainer.cfg.OUTPUT_DIR, "coco_instances_results.json")
            )

            import random

            img_ids = random.choices(res.getImgIds(), k=self.n_examples)
            anns = [(res.imgs[i]["file_name"], res.imgToAnns[i]) for i in img_ids]

            images = []
            labels = []
            for image_path, ann in anns:
                from PIL import Image

                image = np.array(
                    Image.open(
                        os.path.join(eval_metadata.get("image_root"), image_path)
                    )
                )
                images.append(resize(torch.tensor(image).permute((2, 0, 1))))

                viz = Visualizer(
                    # CHW -> HWC
                    img_rgb=image,
                    metadata=eval_metadata,
                    instance_mode=ColorMode.SEGMENTATION,
                )

                label = viz.overlay_instances(
                    masks=[
                        a["segmentation"] for a in ann if a["score"] > self.conf_thresh
                    ],
                    labels=[
                        int(a["category_id"])
                        for a in ann
                        if a["score"] > self.conf_thresh
                    ],
                ).get_image()
                labels.append(resize(torch.from_numpy(label).permute((2, 0, 1))))

            grid = torch.stack((images + labels)).float()

            image_grid = torchvision.utils.make_grid(
                grid, value_range=(0, 255), normalize=True, nrow=len(images)
            )

            self.log_image(image_grid, key="val_examples")

    def before_train(self) -> None:
        """Log example images to wandb before training."""
        data = self.trainer.data_loader
        batch = next(iter(data))[: self.n_examples]
        resize = torchvision.transforms.Resize(512)
        bgr_permute = [2, 1, 0]

        # Cast to float here, otherwise torchvision complains
        images = []
        labels = []
        for s in batch:
            image = s["image"].float()[bgr_permute, :, :].to("cpu")
            instances = s["instances"]
            images.append(resize(image))

            viz = Visualizer(
                # CHW -> HWC
                img_rgb=image.permute((1, 2, 0)),
                metadata=MetadataCatalog.get(self._config.data.name),
                instance_mode=ColorMode.SEGMENTATION,
            )

            label = viz.overlay_instances(
                labels=instances.gt_classes,
                masks=instances.gt_masks,
                boxes=instances.gt_boxes,
            ).get_image()

            labels.append(resize(torch.from_numpy(label).permute((2, 0, 1))))

        grid = torch.stack((images + labels))

        image_grid = torchvision.utils.make_grid(
            grid, value_range=(0, 255), normalize=True, nrow=len(images)
        )
        self.log_image(image_grid, key="train_examples")


def train(config) -> bool:
    """Initiate model training, uses provided configuration.

    Returns:
        bool: True if training was successful, False otherwise
    """

    # Detectron starts tensorboard
    setup_logger()

    image_path = os.path.join(config.data.root, config.data.images)
    train_path = os.path.join(config.data.root, config.data.train)
    val_path = os.path.join(config.data.root, config.data.validation)

    assert os.path.exists(image_path), image_path
    assert os.path.exists(train_path), train_path
    assert os.path.exists(val_path), val_path

    register_coco_instances("train", {}, train_path, image_path)
    register_coco_instances("validate", {}, val_path, image_path)

    # Seems to prevent dataloading issues on some systems.
    torch.multiprocessing.set_sharing_strategy("file_system")

    gc.collect()

    if "cuda" in config.model.device:
        with torch.no_grad():
            torch.cuda.empty_cache()

    cfg = get_cfg()
    if config.model.resume:
        cfg.OUTPUT_DIR = config.data.output
        # TODO - is scale factor necessary, why does this need to be re-specified here?
        cfg.INPUT.SCALE_FACTOR = config.data.scale_factor
        cfg.merge_from_file(os.path.join(cfg.OUTPUT_DIR, "config.yaml"))
        cfg.freeze()
    else:
        # Load the basic config from the arch that we want
        cfg.merge_from_file(model_zoo.get_config_file(config.model.architecture))
        # Override with user/pipeline config settings
        if isinstance(config.model.config, str):
            assert os.path.exists(config.model.config)
            cfg.merge_from_file(config.model.config)
        elif isinstance(config.model.config, DictConfig):
            cfg.merge_from_other_cfg(
                CfgNode(OmegaConf.to_container(config.model.config))
            )
        else:
            raise NotImplementedError

        # If we elect to use a pre-trained model (really if
        # we elect to use COCO)
        if config.model.train_pretrained:
            logger.info("Using pre-trained weights")
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.model.architecture)
        # Note providing a weight file will only be used if pre-trained
        # is off. Maybe should rename that arg.
        else:
            cfg.MODEL.WEIGHTS = config.model.weights

        logger.info(f"Initialising model using: {cfg.MODEL.WEIGHTS}")

        # Various options that we need to infer from data
        # automatically, and "derived" settings like
        # checkpoint periods and loss step changes.
        n_classes = len(config.data.classes)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
        logger.info(f"Training a model with {n_classes} classes")

        # CPU / CUDA etc.
        cfg.MODEL.DEVICE = config.model.device

        # Datasets we set up earlier
        cfg.DATASETS.TRAIN = [
            "train",
        ]
        cfg.DATASETS.TEST = [
            "validate",
        ]

        # Save a checkpoint for each eval
        cfg.TEST.EVAL_PERIOD = max(cfg.SOLVER.MAX_ITER // 10, 1)
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD
        cfg.VIS_PERIOD = 0  # cfg.TEST.EVAL_PERIOD

        # Learning rate scaler
        if cfg.SOLVER.MAX_ITER > 10:
            cfg.SOLVER.STEPS = (
                int(cfg.SOLVER.MAX_ITER * 0.5),
                int(cfg.SOLVER.MAX_ITER * 0.8),
                int(cfg.SOLVER.MAX_ITER * 0.9),
            )
        else:
            cfg.SOLVER.STEPS = [1]

        # Scale factor for training on different size images
        cfg.INPUT.SCALE_FACTOR = config.data.scale_factor

        # Training folder is the current day, since it takes ~1.5 days to
        # train a model on a T4.
        now = datetime.now()
        cfg.OUTPUT_DIR = os.path.join(config.data.output, now.strftime("%Y%m%d_%H%M"))
        os.makedirs(cfg.OUTPUT_DIR)

        # Freeze and backup the config
        cfg.freeze()
        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as fp:
            # Dump exports valid yaml, don't pass it through yaml.dump as well
            fp.write(cfg.dump())

    # Ensure that this gets run before any serious
    # PyTorch stuff happens (but after config is fine)
    if config.model.use_wandb and wandb.run is None:
        wandb.tensorboard.patch(root_logdir=config.data.output, pytorch=True)
        wandb.init(
            project=config.model.wandb_project,
            config=config,
            settings=wandb.Settings(start_method="thread", console="off"),
        )

    trainer = Trainer(cfg)

    # Setup hooks
    example_logger = TrainExampleHook(config, "validate")
    best_checkpointer = hooks.BestCheckpointer(
        eval_period=cfg.TEST.EVAL_PERIOD,
        checkpointer=trainer.checkpointer,
        val_metric="segm/AP50",
        mode="max",
    )
    memory_stats = hooks.TorchMemoryStats()

    trainer.register_hooks([example_logger, best_checkpointer, memory_stats])

    # Remove the default eval hook
    trainer.resume_or_load(resume=config.model.resume)

    # Run summary information for debugging/tracking, contains:

    if config.model.use_wandb:
        wandb.run.summary["train_size"] = len(DatasetCatalog.get("train"))
        wandb.run.summary["val_size"] = len(DatasetCatalog.get("validate"))

    # Let's go!
    trainer.train()
    logger.info("Training is complete.")

    return True
