"""Instance segmentation model framework, using Detectron2 as the backend."""

import gc
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
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

import wandb
from tcd_pipeline.models.model import TiledModel
from tcd_pipeline.post_processing import PostProcessor

logger = logging.getLogger("__name__")


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
        return COCOEvaluator(dataset_name, tasks=["segm"], output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
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
                T.FixedSizeCrop(
                    crop_size=(cfg.INPUT.TRAIN_IMAGE_SIZE, cfg.INPUT.TRAIN_IMAGE_SIZE)
                ),
                T.RandomApply(
                    T.RandomRotation(
                        (0, 90 if cfg.INPUT.SCALE_FACTOR == 1 else 10), expand=True
                    ),
                    prob=0.5,
                ),
            ]
        )

        # Override the augmentations loaded in the config
        return build_detection_train_loader(
            cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs)
        )


class TrainExampleHook(HookBase):
    """Train-time hook that logs example images to wandb before training."""

    def log_image(self, image, key, caption="") -> None:
        """Log an image to Tensorboard.

        Args:
            image (torch.Tensor): Image to log
            key (str): Key to use for logging
            caption (str): Caption to use for logging

        """
        # images = wandb.Image(image, caption)
        # wandb.log({key: images})
        storage = get_event_storage()
        storage.put_image(key, image)

    def before_train(self) -> None:
        """Log example images to wandb before training."""
        data = self.trainer.data_loader
        batch = next(iter(data))[:5]
        resize = torchvision.transforms.Resize(512)
        bgr_permute = [2, 1, 0]

        # Cast to float here, otherwise torchvision complains
        image_grid = torchvision.utils.make_grid(
            [resize(s["image"].float()[bgr_permute, :, :]) for s in batch],
            value_range=(0, 255),
            normalize=True,
        )
        self.log_image(
            image_grid, key="train_examples", caption="Sample training images"
        )


class DetectronModel(TiledModel):
    """Tiled model subclass for Detectron2 models."""

    def __init__(self, config: dict):
        """Initialize the model.

        Args:
            config: The configuration object
        """
        super().__init__(config)
        self.post_processor = PostProcessor(config)
        self.predictor = None
        self.should_reload = False
        self._cfg = None

    def load_model(self) -> None:
        """Load a detectron2 model from the provided config."""
        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if torch.cuda.is_available():
            with torch.no_grad():
                torch.cuda.empty_cache()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model.architecture))
        cfg.merge_from_file(self.config.model.config)
        cfg.MODEL.WEIGHTS = self.config.model.weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        _cfg = cfg.clone()

        self.predictor = DefaultPredictor(_cfg)

        if _cfg.TEST.AUG.ENABLED:
            logger.info("Using Test-Time Augmentation")
            self.model = GeneralizedRCNNWithTTA(
                _cfg, self.predictor.model, batch_size=self.config.model.tta_batch_size
            )
        else:
            logger.info("Test-Time Augmentation is disabled")
            self.model = self.predictor.model

        MetadataCatalog.get(
            self.config.data.name
        ).thing_classes = self.config.data.classes
        self.num_classes = len(self.config.data.classes)
        self.max_detections = _cfg.TEST.DETECTIONS_PER_IMAGE

        self._cfg = _cfg

    def train(self) -> bool:
        """Initiate model training, uses provided configuration.

        Returns:
            bool: True if training was successful, False otherwise
        """

        os.makedirs(self.config.data.output, exist_ok=True)

        # Ensure that this gets run before any serious
        # PyTorch stuff happens (but after config is fine)
        if wandb.run is None:
            wandb.tensorboard.patch(root_logdir=self.config.data.output, pytorch=True)
            wandb.init(
                project=self.config.model.wandb_project,
                config=self.config,
                settings=wandb.Settings(start_method="thread", console="off"),
            )

        # Detectron starts tensorboard
        setup_logger()

        register_coco_instances(
            "train", {}, self.config.data.train, self.config.data.images
        )
        register_coco_instances(
            "validate", {}, self.config.data.validation, self.config.data.images
        )

        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if "cuda" in self.device:
            with torch.no_grad():
                torch.cuda.empty_cache()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model.architecture))
        cfg.merge_from_file(self.config.model.config)

        # If we elect to use a pre-trained model (really if
        # we elect to use COCO)
        if self.config.model.train_pretrained:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        # Note providing a weight file will only be used if pre-trained
        # is off. Maybe should rename that arg.
        else:
            cfg.MODEL.WEIGHTS = self.config.model.weights

        # Various options that we need to infer from data
        # automatically, and "derived" settings like
        # checkpoint periods and loss step changes.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        cfg.DATASETS.TRAIN = [
            "train",
        ]
        cfg.DATASETS.TEST = [
            "validate",
        ]

        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER // 10
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER // 5
        cfg.SOLVER.STEPS = (
            int(cfg.SOLVER.MAX_ITER * 0.5),
            int(cfg.SOLVER.MAX_ITER * 0.8),
            int(cfg.SOLVER.MAX_ITER * 0.9),
        )

        # Scale factor for training on different size images
        cfg.INPUT.SCALE_FACTOR = self.config.data.scale_factor
        cfg.INPUT.TRAIN_IMAGE_SIZE = self.config.data.tile_size

        # Training folder is the current day, since it takes ~1.5 days to
        # train a model on a T4.
        now = datetime.now()
        cfg.OUTPUT_DIR = os.path.join(self.config.data.output, now.strftime("%Y%m%d"))
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Freeze and backup the config
        cfg.freeze()
        import yaml

        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as fp:
            yaml.dump(cfg.dump(), fp)

        trainer = Trainer(cfg)
        example_logger = TrainExampleHook()
        trainer.register_hooks([example_logger])
        trainer.resume_or_load(resume=False)

        # Run summary information for debugging/tracking, contains:
        wandb.run.summary["train_size"] = len(DatasetCatalog.get("train"))
        wandb.run.summary["val_size"] = len(DatasetCatalog.get("validate"))

        # Let's go!
        try:
            trainer.train()
        except Exception as e:  # pylint: disable=broad-except
            logger.error(e)
            return False

        return True

    def evaluate(
        self,
        annotation_file=None,
        image_folder=None,
        output_location=None,
        evaluate=True,
    ) -> None:
        """Evaluate the model on the test set."""

        if self.model is not None:
            self.load_model()

        if annotation_file is None:
            annotation_file = self.config.data.test
        elif image_folder is None:
            raise ValueError(
                "Please provide an image folder if using a custom annotation file."
            )

        if image_folder is None:
            image_folder = self.config.data.images

        logger.info("Image folder: %s", image_folder)
        logger.info("Annotation file: %s", annotation_file)

        # Setup the "test" dataset with the provided annotation file
        if "eval_test" not in DatasetCatalog.list():
            register_coco_instances("eval_test", {}, annotation_file, image_folder)
        else:
            logger.warning("Skipping test dataset registration, already registered.")

        test_loader = (
            build_detection_test_loader(  # pylint: disable=too-many-function-args
                self._cfg,
                "eval_test",
                batch_size=1,
            )
        )

        if output_location is None:
            output_location = self.config.data.output
        os.makedirs(output_location, exist_ok=True)

        # Use the segm task since we're doing instance segmentation
        if evaluate:
            evaluator = COCOEvaluator(
                dataset_name="eval_test",
                tasks=["segm"],
                distributed=False,
                output_dir=output_location,
                max_dets_per_image=self._cfg.TEST.DETECTIONS_PER_IMAGE,
                allow_cached_coco=False,
            )
        else:
            evaluator = None

        inference_on_dataset(self.model, test_loader, evaluator)

    def _predict_tensor(
        self, image_tensor: Union[torch.Tensor, List[torch.Tensor]]
    ) -> List[Dict]:

        self.model.eval()
        self.should_reload = False
        predictions = None

        t_start_s = time.time()

        with torch.no_grad():

            inputs = []

            if isinstance(image_tensor, list):
                for image in image_tensor:
                    height, width = image.shape[1:]
                    inputs.append(
                        {"image": image[:3, :, :], "height": height, "width": width}
                    )
            else:
                height, width = image_tensor.shape[1:]
                inputs.append(
                    {"image": image_tensor[:3, :, :], "height": height, "width": width}
                )

            try:
                predictions = self.model(inputs)[0]["instances"]

                if len(predictions) >= self.max_detections:
                    logger.warning(
                        "Maximum detections reached (%s), possibly re-run with a higher threshold.",
                        self.max_detections,
                    )

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

        return predictions

    def visualise(self, image, results, confidence_thresh=0.5, **kwargs) -> None:
        """Visualise model results using Detectron's provided utils

        Args:
            image (array): Numpy array for image (HWC)
            results (Instances): Instances from predictions
            confidence_thresh (float, optional): Confidence threshold to plot. Defaults to 0.5.
            **kwargs: Passed to matplotlib figure
        """

        mask = results.scores > confidence_thresh
        viz = Visualizer(
            image,
            MetadataCatalog.get(self.config.data.name),
            scale=1.2,
            instance_mode=ColorMode.SEGMENTATION,
        )
        out = viz.draw_instance_predictions(results[mask].to("cpu"))

        plt.figure(**kwargs)
        plt.imshow(out.get_image())
