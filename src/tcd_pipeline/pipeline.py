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


known_models = {
    "restor/tcd-unet-r34": ("unet_resnet34"),
    "restor/tcd-unet-r50": ("unet_resnet50"),
    "restor/tcd-unet-r101": ("unet_resnet101"),
    "restor/tcd-segformer-mit-b0": ("segformer-mit-b0"),
    "restor/tcd-segformer-mit-b1": ("segformer-mit-b1"),
    "restor/tcd-segformer-mit-b2": ("segformer-mit-b2"),
    "restor/tcd-segformer-mit-b3": ("segformer-mit-b3"),
    "restor/tcd-segformer-mit-b4": ("segformer-mit-b4"),
    "restor/tcd-segformer-mit-b5": ("segformer-mit-b5"),
    "restor/tcd-mask-rcnn-r50": ("mask-rcnn-r50"),
}


class Pipeline:
    """Class for wrapping model instances"""

    config: DictConfig = None

    def __init__(
        self,
        model_or_config: Optional[str] = Union[dict, str, DictConfig],
        options: Optional[list[str]] = None,
    ) -> None:
        """Initialise model pipeline. The simplest way to use this class is to
           specify a model e.g. "restor/tcd-segformer-mit-b0".

           You can also pass a generic configuration "instance" or "semantic" to
           either the model or config parameters.

        Args:
            model_or_config Union(str, DictConfig): Model name (repository ID) or config name
            options: List of options passed to Hydra
        """

        # If we get a dict config
        if isinstance(model_or_config, DictConfig):
            self.config = model_or_config
            if options is not None:
                self.config.merge_with(options)

        elif isinstance(model_or_config, str):
            # Check if the input is a standard config (e.g. semantic/instance):
            if model_or_config in ["semantic", "instance"]:
                config = model_or_config
            # Or a known model from the zoo:
            elif model_or_config in known_models:
                model = known_models[model_or_config]

                if not options:
                    options = []

                if "unet" in model or "segformer" in model:
                    config = "semantic"
                    options.append(f"model={model}")
                elif "rcnn" in model:
                    config = "instance"
                    options.append(f"model={model}")
                else:
                    raise ValueError("Unknown model type")

                options.append(f"model.weights={model_or_config}")
            # Otherwise just try and load it as a config name
            else:
                config = model_or_config

            logger.debug(
                f"Attempting to load config: {config} with overrides: {options}"
            )

            self.config = load_config(config, options)

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
                "config/model",
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
        output: str = None,
        **kwargs: Any,
    ) -> ProcessedResult:
        """Run prediction on an image

        If you want to predict over individual arrays/tensors, use the
        `model.predict` method directly.

        If you don't provide an output folder, one will be created in temporary
        system storage (tempfile.mkdtemp).

        Args:
            image (Union[str, DatasetReader]): Path to image, or rasterio image
            output Optional[str]: Path to output folder

        Returns:
            ProcessedResult: processed results from the model (e.g. merged tiles)
        """

        if not self.config.data.output:
            if output:
                self.config.data.output = output
            else:
                import tempfile

                # If the file is open in w+ mode, we get a writer not a reader (but we can still read)
                if isinstance(image, rasterio.io.DatasetReader) or isinstance(
                    image, rasterio.io.DatasetWriter
                ):
                    image_name = image.name
                else:
                    image_name = image

                self.config.data.output = tempfile.mkdtemp(
                    prefix=f"tcd_{os.path.splitext(os.path.basename(image_name))[0]}_"
                )

        logger.info(f"Saving results to {self.config.data.output}")

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
