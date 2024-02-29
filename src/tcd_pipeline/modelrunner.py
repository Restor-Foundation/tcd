import logging
import os
from typing import Any, Optional, Union

import dotmap
import hydra
import rasterio
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .models.instance_segmentation import DetectronModel
from .models.semantic_segmentation import SemanticSegmentationModel
from .post_processing import ProcessedResult

logger = logging.getLogger("__name__")
logging.basicConfig(level=logging.INFO)


def load_config(config_name: str, overrides: list = []) -> DictConfig:
    if not GlobalHydra().is_initialized():
        initialize(config_path="config", version_base=None)

    cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


class ModelRunner:
    """Class for wrapping model instances"""

    config: None

    def __init__(
        self, config: Union[dict, str, DictConfig] = "config.yaml", overrides=None
    ) -> None:
        """Initialise model runner

        Args:
            config (Union[str, dict]): Config file or path to config file
        """

        if isinstance(config, str):
            self.config = load_config(config, overrides)
        elif isinstance(config, dict):
            self.config = load_config("config.yaml", overrides)
            self.config.merge_with(config)
        elif isinstance(config, DictConfig):
            self.config = load_config("config.yaml", overrides)
            self.config.merge_with(config)

        logger.debug(self.config)

        # if isinstance(config, str):
        #    config_dict["config_root"] = os.path.abspath(os.path.dirname(config))

        self.model = None
        self._setup()

    def _setup(self) -> None:
        """Setups the model runner, internal method

        Args:
            config (dict): Configuration dictionary

        Raises:
            NotImplementedError: If the prediction task is not implemented.
        """

        task = self.config.model.task

        if task == "instance_segmentation":
            self.config.model.config = os.path.join(
                os.path.dirname(__file__),
                "config/model/instance_segmentation",
                os.path.splitext(self.config.model.config)[0] + ".yaml",
            )
            self.model = DetectronModel(self.config)
        elif task == "semantic_segmentation":
            self.model = SemanticSegmentationModel(self.config)
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
        return self.model.train()

    def evaluate(self, **kwargs) -> Any:
        """Evaluate the model

        Uses settings in the configuration file.

        """
        return self.model.evaluate(**kwargs)


def default_instance_predictor():
    return ModelRunner("instance.yaml")


def default_semantic_predictor():
    return ModelRunner("segmentation.yaml")
