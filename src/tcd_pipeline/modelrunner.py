import logging
import os
from typing import Any, Optional, Union

import dotmap
import rasterio

from .config import load_config
from .models.instance_segmentation import DetectronModel
from .models.semantic_segmentation import SemanticSegmentationModel
from .post_processing import ProcessedResult

logger = logging.getLogger("__name__")
logging.basicConfig(level=logging.INFO)


class ModelRunner:
    """Class for wrapping model instances"""

    config: None

    def __init__(self, config: Union[dict, str]) -> None:
        """Initialise model runner

        Args:
            config (Union[str, dict]): Config file or path to config file
        """

        config_dict = load_config(config)

        if isinstance(config, str):
            config_dict["config_root"] = os.path.abspath(os.path.dirname(config))

        self.model = None

        self._setup(config_dict)

    def predict(
        self,
        image: Union[str, rasterio.DatasetReader],
        tiled: Optional[bool] = True,
        **kwargs: Any,
    ) -> ProcessedResult:
        """Run prediction on an image

        Args:
            image (Union[str, DatasetReader]): Path to image, or rasterio image
            tiled (bool, optional): Whether to run the model in tiled mode. Defaults to True.

        Returns:
            ProcessedResult: processed results from the model (e.g. merged tiles)
        """

        if isinstance(image, str):
            image = rasterio.open(image)

        if tiled:
            return self.model.predict_tiled(image, **kwargs)
        else:
            return self.model.predict_untiled(image)

    def train(self) -> Any:
        """Train the model using settings defined in the configuration file

        Returns:
            bool: Whether training was successful or not
        """
        return self.model.train()

    def evaluate(self) -> Any:
        """Evaluate the model

        Uses settings in the configuration file.

        """
        return self.model.evaluate()

    def _setup(self, config: dict) -> None:
        """Setups the model runner, internal method

        Args:
            config (dict): Configuration dictionary

        Raises:
            NotImplementedError: If the prediction task is not implemented.
        """
        self.config = dotmap.DotMap(config, _dynamic=False)
        task = self.config.model.task

        # Locate the model config file, relative to the config file
        self.config.model.config = os.path.join(
            self.config.config_root, self.config.model.config
        )

        # Locate the model weights file, relative to the config file directory
        self.config.model.weights = os.path.abspath(
            os.path.join(self.config.config_root, self.config.model.weights)
        )

        if task == "instance_segmentation":
            self.model = DetectronModel(self.config)
        elif task == "semantic_segmentation":
            self.model = SemanticSegmentationModel(self.config)
        else:
            logger.error(f"Task: {task} is not yet implemented")
