"""
This abstract class provides support for
tiled prediction, and is the base class for all models used
in this library.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import psutil
import rasterio
import torch
from PIL import Image
from rasterio.io import DatasetReader
from tqdm.auto import tqdm

from tcd_pipeline.data import dataloader_from_image
from tcd_pipeline.result import ProcessedResult

logger = logging.getLogger("__name__")


class TiledModel(ABC):
    """Abstract class for tiled inference models"""

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary

        """

        self.config = config

        if config.model.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Failed to use CUDA, falling back to CPU", config.model.device
            )
            self.device = "cpu"

        self.model = None
        self.should_reload = False
        self.post_processor = None
        self.failed_images = []
        self.should_exit = False
        self.load_model()

        logger.info("Running inference using: %s", self.device)

    @abstractmethod
    def load_model(self):
        """Load the model, defined by subclass"""

    def predict(self, image: Union[str, torch.Tensor, DatasetReader]) -> Any:
        """Run inference on an image file, rasterio dataset or Tensor.

        Args:
            image (Union[str, Tensor, DatasetReader]): Path to image, or, float tensor
                                              in CHW order, un-normalised

        Returns:
            Any: Prediction results

        Raises:
            NotImplementedError: If the image type is not supported


        """
        if isinstance(image, str):
            image = np.array(Image.open(image))
            image_tensor = torch.from_numpy(image.astype("float32").transpose(2, 0, 1))
        elif isinstance(image, torch.Tensor):
            image_tensor = image
        elif isinstance(image, DatasetReader):
            image_tensor = torch.from_numpy(image.read().astype("float32"))
        else:
            logger.error(
                "Provided image of type %s which is not supported.", type(image)
            )
            raise NotImplementedError

        if self.model is None:
            self.load_model()

        return self._predict_tensor(image_tensor)

    @abstractmethod
    def _predict_tensor(self, image_tensor: torch.Tensor) -> Any:
        """Run inference on a tensor"""

    @abstractmethod
    def train(self) -> bool:
        """Train the model

        Returns:
            bool: Whether training was successful
        """

    @abstractmethod
    def evaluate(self):
        """Evaluate the model"""

    def on_after_predict(self, results, stateful: Optional[bool] = False) -> None:
        """Append tiled results to the post processor, or cache

        Args:
            results (list): Prediction results from one tile
            stateful (bool): Whether to cache results or not

        """

        if stateful:
            self.post_processor.cache_tiled_result(results)
        else:
            self.post_processor.append_tiled_result(results)

    def post_process(self, stateful: Optional[bool] = False) -> ProcessedResult:
        """Run post-processing to merge results

        Args:
            stateful (bool): Whether to cache results or not

        Returns:
            ProcessedResult: merged results
        """

        if stateful:
            self.post_processor.process_cached()

        return self.post_processor.process_tiled_result()

    def attempt_reload(self) -> None:
        """Attempts to reload the model."""
        if "cuda" not in self.device:
            return

        del self.model

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.load_model()

    def predict_tiled(
        self,
        image: rasterio.DatasetReader,
        skip_empty: Optional[bool] = True,
        warm_start: Optional[bool] = True,
    ) -> ProcessedResult:
        """Run inference on an image using tiling. Outputs a ProcessedResult
        Args:
            image (rasterio.DatasetReader): Image
            skip_empty (bool, optional): Skip empty/all-black images. Defaults to True.
            warm_start (bool, option): Whether or not to continue from where one left off
                                       Defaults to True.

        Returns:
            ProcessedResult: A list of predictions and the bounding boxes for those detections.

        Raises:
            ValueError: If the dataloader is empty
        """

        dataloader = dataloader_from_image(
            image,
            tile_size_px=self.config.data.tile_size,
            stride_px=self.config.data.tile_size - self.config.data.tile_overlap,
            gsd_m=self.config.data.gsd,
        )

        if len(dataloader) == 0:
            raise ValueError("No tiles to process")

        if self.post_processor is not None:
            logger.debug("Initialising post processor")
            self.post_processor.initialise(image, warm_start=warm_start)

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        self.failed_images = []
        self.should_exit = False
        # Predict on each tile
        for index, batch in progress_bar:
            if index < self.post_processor.tile_count and warm_start:  # already done
                continue

            if self.should_exit:
                break

            if self.should_reload:
                self.attempt_reload()

            if torch.cuda.is_available():
                _, used_memory_b = torch.cuda.mem_get_info()

            image = batch["image"][0].float()

            if image.mean() < 1 and skip_empty:
                progress_bar.set_postfix_str("Empty frame")
                continue

            t_start = time.time()
            predictions = self.predict(image).to("cpu")
            t_predict = time.time() - t_start

            # Typically if this happens we hit an OOM...
            if predictions is None:
                progress_bar.set_postfix_str("Error")
                logger.error("Failed to run inference on image.")
                self.failed_images.append(image)
            else:

                t_start = time.time()
                self.on_after_predict(
                    (predictions, batch["bbox"][0]), self.config.postprocess.stateful
                )
                t_postprocess = time.time() - t_start

                process = psutil.Process(os.getpid())
                cpu_mem_usage = process.memory_info().rss / 1073741824

                pbar_string = f"#objs: {len(predictions)}"

                if "cuda" in self.device:
                    gpu_mem_usage = used_memory_b / 1073741824
                    pbar_string += f", GPU: {gpu_mem_usage:1.2f}G"

                pbar_string += f", CPU: {cpu_mem_usage:1.2f}G"
                pbar_string += f", t_pred: {t_predict:1.2f}s"
                pbar_string += f", t_post: {t_postprocess:1.2f}s"

                progress_bar.set_postfix_str(pbar_string)

        return self.post_process(self.config.postprocess.stateful)
