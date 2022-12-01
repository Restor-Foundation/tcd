import logging
from typing import Any, Union

import dotmap
import yaml
from rasterio import DatasetReader
from typing import Optional, Any
from post_processing import ProcessedResult
from instance_segmentation import DetectronModel

logger = logging.getLogger("__name__")
logging.basicConfig(level=logging.INFO)


class ModelRunner:
    """Class for wrapping model instances"""

    def __init__(self, config: Union[dict, str]) -> None:
        """Initialise model runner

        Args:
            config (Union[str, dict]): Config file or path to config file
        """

        if isinstance(config, str):
            with open(config, "r") as fp:
                config = yaml.safe_load(fp)
        elif isinstance(config, dict):
            pass
        else:
            raise NotImplementedError(
                "Please provide a dictionary or a path to a config file"
            )

        self.model = None

        self._setup(config)

    def predict(
        self,
        image: Union[str, DatasetReader],
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

        if tiled:
            return self.model.predict_tiled(image, **kwargs)
        else:
            return self.model.predict(image)

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
        self.config = dotmap.DotMap(config)

        task = self.config.model.task

        if task == "instance_segmentation":
            self.model = DetectronModel(self.config)
        elif task == "semantic_segmentation":
            # TODO: Hold for satellite branch merge
            raise NotImplementedError
        else:
            logger.error(f"Task: {task} is not yet implemented")
