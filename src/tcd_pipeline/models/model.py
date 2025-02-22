"""
This abstract class provides support for
tiled prediction, and is the base class for all models used
in this library.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, List, Optional, Union

import numpy as np
import psutil
import rasterio
import torch
from omegaconf import DictConfig
from PIL import Image
from rasterio.io import DatasetReader
from tqdm.auto import tqdm

from tcd_pipeline.data.imagedataset import dataloader_from_image
from tcd_pipeline.postprocess.postprocessor import PostProcessor
from tcd_pipeline.result.processedresult import ProcessedResult
from tcd_pipeline.util import image_to_tensor

logger = logging.getLogger(__name__)


class Model(ABC):
    """Abstract class for tiled inference models"""

    def __init__(self, config: DictConfig):
        """
        Args:
            config (DictConfig): Configuration dictionary

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
        self.failed_images = set()
        self.should_exit = False

        logger.info("Device: %s", self.device)
        self.setup()

    @abstractmethod
    def setup(self):
        """Perform any setup actions as needed"""

    @abstractmethod
    def load_model(self):
        """Load the model, defined by subclass"""

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
            batch_size=self.config.model.batch_size,
        )

        if len(dataloader) == 0:
            raise ValueError("No tiles to process")

        if self.post_processor is not None:
            logger.debug("Initialising post processor")
            self.post_processor.initialise(image, warm_start=warm_start)

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        self.should_exit = False

        if not warm_start:
            self.post_processor.setup_cache()

        # Predict on each tile
        processed_tiles = 0
        for _, batch in progress_bar:
            if self.should_exit:
                break

            if self.should_reload:
                self.attempt_reload()

            # Skip images that are all black or all white - do this first?
            if skip_empty:
                filtered = defaultdict(list)
                for idx, im in enumerate(batch["image"]):
                    img_mean = im.mean()

                    if img_mean <= 1 or img_mean >= 254:
                        continue

                    for key in batch:
                        filtered[key].append(batch[key][idx])

                batch = filtered
                if len(batch["image"]) == 0:
                    progress_bar.set_postfix_str(f"Empty batch, skipping.")
                    continue

            processed_tiles += len(batch["image"])

            if (
                processed_tiles <= self.post_processor.tile_count and warm_start
            ):  # already done
                progress_bar.set_postfix_str(
                    f"Processed batch, skipping - valid tile count {processed_tiles}"
                )
            else:
                predictions = [p.to("cpu") for p in self.predict(batch["image"])]

                # Typically if this happens we hit an OOM...
                if predictions is None:
                    progress_bar.set_postfix_str("Error")
                    logger.error("Failed to run inference on image.")
                    self.failed_images.add(image)
                else:
                    # Run the post-processor.
                    batch["predictions"] = predictions
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

    def predict(self, image: List[Union[str, torch.Tensor, DatasetReader]]) -> Any:
        """Run inference on an image file, rasterio dataset or Tensor.

        Args:
            image (Union[str, Tensor, DatasetReader]): List of (Path to image, or, float tensor
                                              in CHW order, un-normalised)

        Returns:
            Any: Raw prediction results

        Raises:
            NotImplementedError: If the image type is not supported


        """

        t_start = time.time()

        if not isinstance(image, list):
            image = [image]

        image_tensor = [image_to_tensor(i) for i in image]

        if self.model is None:
            self.load_model()

        res = self.predict_batch(image_tensor)

        self.t_predict = time.time() - t_start

        return res

    @abstractmethod
    def predict_batch(self, image_tensor: List[torch.Tensor]) -> Any:
        """Run inference on a batch of tensors"""
