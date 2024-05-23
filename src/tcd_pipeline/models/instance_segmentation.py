"""Instance segmentation model framework, using Detectron2 as the backend."""

import gc
import logging
import os
import time
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils.visualizer import ColorMode, Visualizer
from omegaconf import DictConfig, OmegaConf

from tcd_pipeline.models.model import Model
from tcd_pipeline.postprocess.instanceprocessor import InstanceSegmentationPostProcessor

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)


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

    def setup(self):
        pass

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

    def predict_batch(
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
