import logging
import os
from typing import Any, Optional, Union

import rasterio
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from .models.instance_segmentation import DetectronModel
from .result.processedresult import ProcessedResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(config_name: str, overrides: Union[str, list] = []) -> DictConfig:
    if not GlobalHydra().is_initialized():
        initialize(config_path="config", version_base=None)

    if isinstance(overrides, str):
        overrides = [overrides]

    cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


class Pipeline:
    """Class for wrapping model instances"""

    config: DictConfig = None

    def __init__(
        self,
        config: Union[dict, str, DictConfig] = "config.yaml",
        overrides=None,
    ) -> None:
        """Initialise model runner

        Args:
            config (Union[str, dict]): Config file or path to config file
        """

        if isinstance(config, str):
            logger.debug(
                f"Attempting to load config: {config} with overrides: {overrides}"
            )
            self.config = load_config(config, overrides)
        elif isinstance(config, DictConfig):
            self.config = config
            if overrides is not None:
                self.config.merge_with(overrides)

        self.model = None
        self._setup()

    def _setup(self) -> None:
        """Setups the model runner. The primary aim of this function is to assess whether the
        model weights are:

        (1) A local checkpoint
        (2) A reference to an online checkpoint, hosted on HuggingFace (e.g. restor/tcd-segformer-mit-b1)

        First, the function will attempt to locate the weights file either using an asbolute path, or a path
        relative to the package root folder (for example if you have a checkpoint folder stored within
        the repo root). If one of these paths is found, the function will update the config key with the
        absolute path.
        """

        # Attempt to locate weights:
        # 1 Does the absolute path exist?
        if os.path.exists(os.path.abspath(self.config.model.weights)):
            self.config.model.weights = os.path.abspath(self.config.model.weights)
            logger.info(
                f"Found weights file at absolute path: {self.config.model.weights}"
            )
        # 2 Relative to package folder
        elif os.path.exists(
            os.path.join(
                os.path.dirname(__file__), "..", "..", self.config.model.weights
            )
        ):
            self.config.model.weights = os.path.join(
                os.path.dirname(__file__), "..", "..", self.config.model.weights
            )
            logger.info(
                f"Found weights file relative to package install: {self.config.model.weights}"
            )

        task = self.config.model.task
        if task == "instance_segmentation":
            self.config.model.config = os.path.join(
                os.path.dirname(__file__),
                "config/model/instance_segmentation",
                os.path.splitext(self.config.model.config)[0] + ".yaml",
            )
            self.model = DetectronModel(self.config)
        elif task == "semantic_segmentation":
            if (
                self.config.model.name == "segformer"
                or "segformer" in self.config.model.weights
            ):
                from .models.segformer import Segformer

                self.model = Segformer(self.config)
            else:
                from .models.smp import SMPModel

                self.model = SMPModel(self.config)
        else:
            logger.error(f"Task: {task} is not yet implemented")

    def predict(
        self,
        image: Union[str, rasterio.DatasetReader],
        **kwargs: Any,
    ) -> ProcessedResult:
        """Run prediction on an image

        If you want to predict over individual arrays/tensors, use the
        `model.predict` method directly.

        Args:
            image (Union[str, DatasetReader]): Path to image, or rasterio image

        Returns:
            ProcessedResult: processed results from the model (e.g. merged tiles)
        """

        if isinstance(image, str):
            image = rasterio.open(image)

        return self.model.predict_tiled(image, **kwargs)

    def train(self) -> Any:
        """Train the model using settings defined in the configuration file

        Returns:
            bool: Whether training was successful or not
        """
        from .models import train_instance, train_semantic

        if self.config.model.task == "instance_segmentation":
            train_instance.train(self.config)
        elif self.config.model.task == "semantic_segmentation":
            train_semantic.train(self.config)

    def evaluate(self, **kwargs) -> Any:
        """Evaluate the model

        Uses settings in the configuration file.

        """
        return self.model.evaluate(**kwargs)


def default_instance_predictor():
    return Pipeline("instance.yaml")


def default_semantic_predictor():
    return Pipeline("segmentation.yaml")
