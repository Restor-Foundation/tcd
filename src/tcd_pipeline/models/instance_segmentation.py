"""
Instance segmentation model framework, using Detectron2 as the backend.
"""

import gc
import logging
import os
import time
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
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
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

import wandb
from tcd_pipeline.models.model import TiledModel
from tcd_pipeline.post_processing import PostProcessor

logger = logging.getLogger("__name__")


class Trainer(DefaultTrainer):
    """
    Subclass of the default training class
    so that we have control over things like
    augmentation.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, tasks=["segm"], output_dir=cfg.OUTPUT_DIR)


class TrainExampleHook(HookBase):
    """
    Train-time hook that logs example images to wandb before training.
    """

    def log_image(self, image, key, caption=""):
        """
        Log an image to wandb
        """

        images = wandb.Image(image, caption)
        wandb.log({key: images})

    def before_train(self):
        """
        Log example images to wandb before training.
        """

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
    """
    Subclass of the default tiled model class which implements an
    instance segmentation model using Detectron2.
    """

    def __init__(self, config):
        super().__init__(config)
        self.post_processor = PostProcessor(config)
        self.predictor = None
        self.should_reload = False
        self._cfg = None

    def load_model(self):
        """
        Load a detectron2 model from the provided config.
        """
        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if "cuda" in self.device:
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

    def train(self):
        """Initiate model training, uses provided configuration"""

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

        # If a checkpoint isn't provided
        if self.config.model.train_pretrained:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        else:
            cfg.MODEL.WEIGHTS = self.config.model.weights

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        cfg.DATASETS.TRAIN = "train"
        cfg.DATASETS.TEST = "validate"

        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER // 10
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER // 5
        cfg.SOLVER.STEPS = (
            int(cfg.SOLVER.MAX_ITER * 0.75),
            int(cfg.SOLVER.MAX_ITER * 0.9),
        )

        now = datetime.now()  # current date and time
        cfg.OUTPUT_DIR = os.path.join(self.config.data.output, now.strftime("%Y%m%d"))
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        trainer = Trainer(cfg)

        example_logger = TrainExampleHook()
        trainer.register_hooks([example_logger])
        trainer.resume_or_load(resume=False)

        # Run summary information for debugging/tracking, contains:
        wandb.run.summary["train_size"] = len(DatasetCatalog.get("train"))
        wandb.run.summary["val_size"] = len(DatasetCatalog.get("validate"))

        try:
            trainer.train()
        except Exception as e:  # pylint: disable=broad-except
            logger.error(e)
            return False

        return True

    def evaluate(self):
        """
        Evaluate the model on the test set.
        """

        if self.model is not None:
            self.load_model()

        annotation_file = self.config.data.test
        image_folder = self.config.data.images

        # Setup the "test" dataset with the provided annotation file
        register_coco_instances("test", {}, annotation_file, image_folder)

        test_loader = (
            build_detection_test_loader(  # pylint: disable=too-many-function-args
                self._cfg,
                "test",
                batch_size=1,
                mapper=DatasetMapper.from_config(self._cfg, is_train=False),
            )
        )

        os.makedirs(self.config.data.output, exist_ok=True)

        # Use the segm task since we're doing instance segmentation
        evaluator = COCOEvaluator(
            dataset_name="test",
            tasks=["segm"],
            distributed=False,
            output_dir=self.config.data.output,
            max_dets_per_image=self._cfg.TEST.DETECTIONS_PER_IMAGE,
            allow_cached_coco=False,
        )

        inference_on_dataset(self.model, test_loader, evaluator)

    def _predict_tensor(self, image_tensor: torch.Tensor) -> List[Dict]:

        self.model.eval()
        self.should_reload = False
        predictions = None

        t_start_s = time.time()

        with torch.no_grad():
            _, height, width = image_tensor.shape

            # removing alpha channel
            inputs = {"image": image_tensor[:3, :, :], "height": height, "width": width}

            try:
                predictions = self.model([inputs])[0]["instances"]

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

    def visualise(self, image, results, confidence_thresh=0.5, **kwargs):
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
