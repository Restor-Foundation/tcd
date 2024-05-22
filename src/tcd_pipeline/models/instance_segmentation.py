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

import wandb
from tcd_pipeline.models.model import Model
from tcd_pipeline.postprocess.instanceprocessor import InstanceSegmentationPostProcessor

torch.multiprocessing.set_sharing_strategy("file_system")

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
        return COCOEvaluator(
            dataset_name,
            tasks=["segm"],
            allow_cached_coco=False,
            max_dets_per_image=256,
            output_dir=cfg.OUTPUT_DIR,
        )

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
        self.config = config
        self.val_name = val_name
        self.n_examples = n_examples
        self.conf_thresh = 0.5

    def log_image(self, image, key) -> None:
        """Log an image to Tensorboard.

        Args:
            image (torch.Tensor): Image to log
            key (str): Key to use for logging
            caption (str): Caption to use for logging

        """
        # images = wandb.Image(image, caption)
        # wandb.log({key: images})
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
                metadata=MetadataCatalog.get(self.config.data.name),
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


class DetectronModel(Model):
    """Tiled model subclass for Detectron2 models."""

    def __init__(self, config: dict):
        """Initialize the model.

        Args:
            config: The configuration object
        """
        super().__init__(config)
        self.post_processor = InstanceSegmentationPostProcessor(config)
        self.predictor = None
        self.should_reload = False
        self._cfg = None

        self.load_config()

    def load_config(self) -> None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model.architecture))

        if isinstance(self.config.model.config, str):
            cfg.merge_from_file(self.config.model.config)
        elif isinstance(self.config.model.config, DictConfig):
            cfg.merge_from_other_cfg(
                CfgNode(OmegaConf.to_container(self.config.model.config))
            )
        else:
            raise NotImplementedError

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        self._cfg = cfg.clone()

        MetadataCatalog.get(
            self.config.data.name
        ).thing_classes = self.config.data.classes
        self.num_classes = len(self.config.data.classes)
        self.max_detections = self._cfg.TEST.DETECTIONS_PER_IMAGE

    def load_model(self) -> None:
        """Load a detectron2 model from the provided config."""
        gc.collect()

        if torch.cuda.is_available():
            with torch.no_grad():
                torch.cuda.empty_cache()

        if not os.path.exists(self.config.model.weights):
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                self._cfg.MODEL.WEIGHTS = api.hf_hub_download(
                    self.config.model.weights, filename="model.pth"
                )
            except Exception as e:
                logger.warning("Failed to download checkpoint from HF hub")
        else:
            self._cfg.MODEL.WEIGHTS = self.config.model.weights

        self.predictor = DefaultPredictor(self._cfg)

        if self._cfg.TEST.AUG.ENABLED:
            logger.info("Using Test-Time Augmentation")
            self.model = GeneralizedRCNNWithTTA(
                self._cfg,
                self.predictor.model,
                batch_size=self.config.model.tta_batch_size,
            )
        else:
            logger.info("Test-Time Augmentation is disabled")
            self.model = self.predictor.model

    def train(self) -> bool:
        """Initiate model training, uses provided configuration.

        Returns:
            bool: True if training was successful, False otherwise
        """

        # Detectron starts tensorboard
        setup_logger()

        image_path = os.path.join(self.config.data.root, self.config.data.images)
        train_path = os.path.join(self.config.data.root, self.config.data.train)
        val_path = os.path.join(self.config.data.root, self.config.data.validation)

        assert os.path.exists(image_path), image_path
        assert os.path.exists(train_path), train_path
        assert os.path.exists(val_path), val_path

        register_coco_instances("train", {}, train_path, image_path)
        register_coco_instances("validate", {}, val_path, image_path)

        # Seems to prevent dataloading issues on some systems.
        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if "cuda" in self.device:
            with torch.no_grad():
                torch.cuda.empty_cache()

        cfg = get_cfg()
        if self.config.model.resume:
            cfg.OUTPUT_DIR = self.config.data.output
            # TODO - is scale factor necessary, why does this need to be re-specified here?
            cfg.INPUT.SCALE_FACTOR = self.config.data.scale_factor
            cfg.merge_from_file(os.path.join(cfg.OUTPUT_DIR, "config.yaml"))
            cfg.freeze()
        else:
            # Load the basic config from the arch that we want
            cfg.merge_from_file(
                model_zoo.get_config_file(self.config.model.architecture)
            )
            # Override with user/pipeline config settings
            if isinstance(self.config.model.config, str):
                assert os.path.exists(self.config.model.config)
                cfg.merge_from_file(self.config.model.config)
            elif isinstance(self.config.model.config, DictConfig):
                cfg.merge_from_other_cfg(
                    CfgNode(OmegaConf.to_container(self.config.model.config))
                )
            else:
                raise NotImplementedError

            # If we elect to use a pre-trained model (really if
            # we elect to use COCO)
            if self.config.model.train_pretrained:
                logger.info("Using pre-trained weights")
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    self.config.model.architecture
                )
            # Note providing a weight file will only be used if pre-trained
            # is off. Maybe should rename that arg.
            else:
                cfg.MODEL.WEIGHTS = self.config.model.weights

            logger.info(f"Initialising model using: {cfg.MODEL.WEIGHTS}")

            # Various options that we need to infer from data
            # automatically, and "derived" settings like
            # checkpoint periods and loss step changes.
            n_classes = len(self.config.data.classes)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
            logger.info(f"Training a model with {n_classes} classes")

            # CPU / CUDA etc.
            cfg.MODEL.DEVICE = self.device

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
            cfg.INPUT.SCALE_FACTOR = self.config.data.scale_factor

            # Training folder is the current day, since it takes ~1.5 days to
            # train a model on a T4.
            now = datetime.now()
            cfg.OUTPUT_DIR = os.path.join(
                self.config.data.output, now.strftime("%Y%m%d_%H%M")
            )
            os.makedirs(cfg.OUTPUT_DIR)

            # Freeze and backup the config
            cfg.freeze()
            with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as fp:
                # Dump exports valid yaml, don't pass it through yaml.dump as well
                fp.write(cfg.dump())

        # Ensure that this gets run before any serious
        # PyTorch stuff happens (but after config is fine)
        if self.config.model.use_wandb and wandb.run is None:
            wandb.tensorboard.patch(root_logdir=self.config.data.output, pytorch=True)
            wandb.init(
                project=self.config.model.wandb_project,
                config=self.config,
                settings=wandb.Settings(start_method="thread", console="off"),
            )

        trainer = Trainer(cfg)

        # Setup hooks
        example_logger = TrainExampleHook(self.config, "validate")
        best_checkpointer = hooks.BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=trainer.checkpointer,
            val_metric="segm/AP50",
            mode="max",
        )
        memory_stats = hooks.TorchMemoryStats()

        trainer.register_hooks([example_logger, best_checkpointer, memory_stats])

        # Remove the default eval hook
        trainer.resume_or_load(resume=self.config.model.resume)

        # Run summary information for debugging/tracking, contains:

        if self.config.model.use_wandb:
            wandb.run.summary["train_size"] = len(DatasetCatalog.get("train"))
            wandb.run.summary["val_size"] = len(DatasetCatalog.get("validate"))

        # Let's go!
        trainer.train()
        logger.info("Training is complete.")

        return True

    def evaluate(
        self,
        annotation_file=None,
        image_folder=None,
        output_folder=None,
        evaluate=True,
    ) -> None:
        """Evaluate the model on the test set."""

        if self.model is None:
            self.load_model()
            assert self.model is not None

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

        assert self._cfg is not None

        test_loader = (
            build_detection_test_loader(  # pylint: disable=too-many-function-args
                self._cfg, "eval_test", batch_size=1
            )
        )

        if output_folder is None:
            output_folder = self.config.data.output
        os.makedirs(output_folder, exist_ok=True)

        # Use the segm task since we're doing instance segmentation
        if evaluate:
            evaluator = COCOEvaluator(
                dataset_name="eval_test",
                tasks=["segm"],
                distributed=False,
                output_dir=output_folder,
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
                predictions = [p["instances"] for p in self.model(inputs)]

                for prediction in predictions:
                    if len(prediction) >= self.max_detections:
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
