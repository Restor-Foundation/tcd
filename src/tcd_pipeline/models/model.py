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

from tcd_pipeline.data.dataset import dataloader_from_image
from tcd_pipeline.postprocess.postprocessor import PostProcessor
from tcd_pipeline.result import ProcessedResult
from tcd_pipeline.util import image_to_tensor

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
            logger.warning("Failed to use CUDA, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = config.model.device

        self.model = None
        self.should_reload = False
        self.post_processor: PostProcessor = None
        self.failed_images = []
        self.should_exit = False

        logger.info("Device: %s", self.device)

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

        t_start = time.time()

        image_tensor = image_to_tensor(image)

        if self.model is None:
            self.load_model()

        res = self._predict_tensor(image_tensor)

        self.t_predict = time.time() - t_start

        return res

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

    def on_after_predict(self, results: dict) -> None:
        """Append tiled results to the post processor, or cache

        Args:
            results (list): Prediction results from one tile

        """

        t_start = time.time()

        # Invert dict-of-lists to list-of-dicts
        results = [dict(zip(results, t)) for t in zip(*results.values())]
        self.post_processor.add(results)

        self.t_postprocess = time.time() - t_start

    def post_process(self) -> ProcessedResult:
        """Run post-processing to merge results

        Returns:
            ProcessedResult: merged results
        """

        res = self.post_processor.process()

        if self.config.postprocess.cleanup:
            logger.info("Cleaning up post processor")
            self.post_processor.cache.clear()

        return res

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
        warm_start: Optional[bool] = False,
    ) -> ProcessedResult:
        """Run inference on an image using tiling. Outputs a ProcessedResult
        Args:
            image (rasterio.DatasetReader): Image
            skip_empty (bool, optional): Skip empty/all-black images. Defaults to True.
            warm_start (bool, option): Whether or not to continue from where one left off
                                       Defaults to False.

        Returns:
            ProcessedResult: A list of predictions and the bounding boxes for those detections.

        Raises:
            ValueError: If the dataloader is empty
        """

        dataloader = dataloader_from_image(
            image,
            tile_size_px=self.config.data.tile_size,
            overlap_px=self.config.data.tile_overlap,
            gsd_m=self.config.data.gsd,
            clip_tiles=False,
        )

        if len(dataloader) == 0:
            raise ValueError("No tiles to process")

        if self.post_processor is not None:
            logger.debug("Initialising post processor")
            self.post_processor.initialise(image, warm_start=warm_start)

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        self.failed_images = []
        self.should_exit = False

        if not warm_start:
            self.post_processor.setup_cache()

        # Predict on each tile
        for index, batch in progress_bar:
            if index < self.post_processor.tile_count and warm_start:  # already done
                continue

            if self.should_exit:
                break

            if self.should_reload:
                self.attempt_reload()

            image = batch["image"][0].float()

            # Skip images that are all black or all white
            if image.mean() < 1 and skip_empty:
                progress_bar.set_postfix_str("Empty frame")
                continue

            if image.mean() > 254 and skip_empty:
                progress_bar.set_postfix_str("Empty frame")
                continue

            predictions = self.predict(image).to("cpu")

            # Typically if this happens we hit an OOM...
            if predictions is None:
                progress_bar.set_postfix_str("Error")
                logger.error("Failed to run inference on image.")
                self.failed_images.append(image)
            else:
                # Run the post-processor.
                batch["predictions"] = [predictions]
                self.on_after_predict(batch)

                # Logging
                process = psutil.Process(os.getpid())
                cpu_mem_usage_gb = process.memory_info().rss / 1073741824

                pbar_string = f"#objs: {len(predictions)}"

                if "cuda" in self.device and torch.cuda.is_available():
                    _, used_memory_b = torch.cuda.mem_get_info()
                    gpu_mem_usage_gb = used_memory_b / 1073741824
                    pbar_string += f", GPU: {gpu_mem_usage_gb:1.2f}G"

                pbar_string += f", CPU: {cpu_mem_usage_gb:1.2f}G"
                pbar_string += f", t_pred: {self.t_predict:1.2f}s"
                pbar_string += f", t_post: {self.t_postprocess:1.2f}s"

                progress_bar.set_postfix_str(pbar_string)

        return self.post_process()
