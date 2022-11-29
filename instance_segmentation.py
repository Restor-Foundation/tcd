import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from tqdm.auto import tqdm

import wandb
from data import dataloader_from_image
from model import TiledModel
from post_processing import PostProcessor

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
    def log_image(self, image, key, caption=""):
        images = wandb.Image(image, caption)
        wandb.log({key: images})

    def before_train(self):

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

    def after_step(self):
        pass


class DetectronModel(TiledModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_processor = PostProcessor(config)

    def load_model(self):

        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if "cuda" in self.device:
            with torch.no_grad():
                torch.cuda.empty_cache()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model.architecture))
        cfg.merge_from_other_cfg(CfgNode(self.config.evaluate.detectron))
        cfg.MODEL.WEIGHTS = self.config.model.weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        _cfg = cfg.clone()

        self.predictor = DefaultPredictor(_cfg)

        if self.config.evaluate.detectron.TEST.AUG.ENABLED:
            logger.info("Using Test-Time Augmentation")
            self.model = GeneralizedRCNNWithTTA(
                _cfg, self.predictor.model, batch_size=6
            )
        else:
            logger.info("Test-Time Augmentation is disabled")
            self.model = self.predictor.model

        MetadataCatalog.get(
            self.config.data.name
        ).thing_classes = self.config.data.classes
        self.num_classes = len(self.config.data.classes)
        self.max_detections = self.config.evaluate.detectron.TEST.DETECTIONS_PER_IMAGE

    def train(self):
        """Initiate model training, uses configuration"""

        os.makedirs(self.config.data.output, exist_ok=True)

        # Ensure that this gets run before any serious
        # PyTorch stuff happens (but after config is fine)
        if wandb.run is None:
            wandb.tensorboard.patch(root_logdir=self.config.data.output, pytorch=True)
            wandb.init(
                config=self.config,
                settings=wandb.Settings(start_method="thread", console="off"),
            )

        # Detectron starts tensorboard
        setup_logger()

        from detectron2.data.datasets import register_coco_instances

        register_coco_instances(
            "train", {}, self.config.data.train, self.config.data.images
        )
        register_coco_instances(
            "test", {}, self.config.data.test, self.config.data.images
        )
        register_coco_instances(
            "validate", {}, self.config.data.validation, self.config.data.images
        )

        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if "cuda" in self.device:
            with torch.no_grad():
                torch.cuda.empty_cache()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model.architecture))
        cfg.merge_from_other_cfg(CfgNode(self.config.evaluate.detectron))

        # Checkpoint
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        cfg.DATASETS.TRAIN = "train"
        cfg.DATASETS.TEST = "validate"

        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER // 10
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER // 5
        cfg.SOLVER.BASE_LR = 1e-3
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

        """
        Run summary information for debugging/tracking, contains:

        - dataset sizes 

        """
        wandb.run.summary["train_size"] = len(DatasetCatalog.get("train"))
        wandb.run.summary["test_size"] = len(DatasetCatalog.get("test"))
        wandb.run.summary["val_size"] = len(DatasetCatalog.get("validate"))

        try:
            trainer.train()
        except:
            return False

        return True

    def evaluate(self, dataset, output_folder):

        if self.model is not None:
            self.load_model()

        test_loader = build_detection_test_loader(_cfg, dataset_test, batch_size=1)

        os.makedirs(eval_output_dir, exist_ok=True)
        evaluator = COCOEvaluator(
            dataset_name=dataset,
            tasks=["segm"],
            distributed=False,
            output_dir=output_folder,
            max_dets_per_image=500,
            allow_cached_coco=False,
        )

        # Run the evaluation
        inference_on_dataset(model, test_loader, evaluator)

    def predict(self, image):
        """Run inference on an image file or Tensor

        Args:
            image (Union[str, torch.Tensor]): Path to image, or, float tensor in CHW order, un-normalised

        Returns:
            predictions: Detectron2 prediction dictionary
        """

        if isinstance(image, str):
            image = np.array(Image.open(image))
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        elif isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            raise NotImplementedError

        if self.model is None:
            self.load_model()

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
                        f"Maximum detections reached ({self.max_detections}), possibly re-run with a higher threshold."
                    )

            except RuntimeError as e:
                logger.error(f"Runtime error: {e}")
                self.should_reload = True
            except Exception as e:
                logger.error(
                    f"Failed to run inference: {e}. Attempting to reload model."
                )
                self.should_reload = True

        t_elapsed_s = time.time() - t_start_s
        logger.debug(f"Predicted tile in {t_elapsed_s:1.2f}s")

        return predictions

    def on_after_predict(self, results, stateful=False):
        """Append tiled results to the post processor, or cache

        Args:
            results (list): Prediction results from one tile
        """

        if stateful:
            self.post_processor.cache_tiled_result(results)
        else:
            self.post_processor.append_tiled_result(results)

    def post_process(self, stateful=False):
        """Run post-processing to merge results

        Returns:
            ProcessedResult: merged results
        """
        if stateful:
            return self.post_processor.process_cached()
        else:
            return self.post_processor.process_tiled_result()

    def visualise(self, image, results, confidence_thresh=0.5, **kwargs):
        """Visualise model results using Detectron's provided utils

        Args:
            image (array): Numpy array for image (HWC)
            results (Instances): Instances from predictions
            confidence_thresh (float, optional): Confidence threshold to plot. Defaults to 0.5.
        """

        mask = results.scores > confidence_thresh
        v = Visualizer(
            image,
            MetadataCatalog.get(self.config.data.name),
            scale=1.2,
            instance_mode=ColorMode.SEGMENTATION,
        )
        out = v.draw_instance_predictions(results[mask].to("cpu"))

        plt.figure(**kwargs)
        plt.imshow(out.get_image())
