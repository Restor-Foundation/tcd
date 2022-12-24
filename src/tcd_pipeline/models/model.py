import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import psutil
import rasterio
import torch
from tqdm.auto import tqdm

from ..data import dataloader_from_image
from ..post_processing import Bbox

logger = logging.getLogger("__name__")


class TiledModel(ABC):
    def __init__(self, config: dict):

        self.config = config

        try:
            torch.tensor([1]).to(config.model.device)
            self.device = config.model.device
        except:
            logger.warning(
                f"Failed to use device: {config.model.device}, falling back to CPU"
            )
            self.device = "cpu"

        self.model = None
        self.should_reload = False
        self.post_processor = None
        self.load_model()

        logger.info(f"Running inference using: {self.device}")

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def on_after_predict(self, results, stateful: Optional[bool] = False):
        """Append tiled results to the post processor, or cache

        Args:
            results (list): Prediction results from one tile
        """

        if stateful:
            self.post_processor.cache_tiled_result(results)
        else:
            self.post_processor.append_tiled_result(results)

    def post_process(self, stateful: Optional[bool] = False):
        """Run post-processing to merge results

        Returns:
            ProcessedResult: merged results
        """

        if stateful:
            self.post_processor.process_cached()

        return self.post_processor.process_tiled_result()

    def attempt_reload(self):
        """Attempts to reload the model."""
        if "cuda" not in self.device:
            return

        del self.model
        torch.cuda.synchronize()
        self.load_model()

    def predict_tiled(
        self,
        image: rasterio.DatasetReader,
        skip_empty: Optional[bool] = True,
        warm_start: Optional[bool] = True,
    ):
        """Run inference on an image using tiling. Outputs a ProcessedResult
        Args:
            image (rasterio.DatasetReader): Image
            skip_empty (bool, optional): Skip empty/all-black images. Defaults to True.
            warm_start (bool, option): Whether or not to continue from where one left off. Defaults to True.

        Returns:
            ProcessedResult: A list of predictions and the bounding boxes for those detections.
        """

        gsd_m = self.config.data.gsd
        dataloader = dataloader_from_image(
            image,
            tile_size_px=self.config.data.tile_size,
            stride_px=self.config.data.tile_overlap,
            gsd_m=gsd_m,
        )

        if len(dataloader) == 0:
            raise ValueError("No tiles to process")

        if self.post_processor is not None:
            logger.debug("Initialising post processor")
            self.post_processor.initialise(image, warm_start=warm_start)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        self.failed_images = []
        self.should_exit = False
        # Predict on each tile
        for index, batch in pbar:
            if index < self.post_processor.tile_count and warm_start:  # already done
                continue

            if self.should_exit:
                break

            if self.should_reload:
                self.attempt_reload()

            if "cuda" in self.device:
                _, used_memory_b = torch.cuda.mem_get_info()

            image = batch["image"][0].float()

            if image.mean() < 1 and skip_empty:
                pbar.set_postfix_str(f"Empty frame")
                continue

            tstart = time.time()
            predictions = self.predict(image).to("cpu")
            tpredict = time.time() - tstart

            # Typically if this happens we hit an OOM...
            if predictions is None:
                pbar.set_postfix_str("Error")
                logger.error("Failed to run inference on image.")
                self.failed_images.append(image)
            else:

                tstart = time.time()
                self.on_after_predict(
                    (predictions, batch["bbox"][0]), self.config.postprocess.stateful
                )
                tpostprocess = time.time() - tstart

                process = psutil.Process(os.getpid())
                cpu_mem_usage = process.memory_info().rss / 1073741824

                pbar_string = f"#objs: {len(predictions)}"

                if "cuda" in self.device:
                    gpu_mem_usage = used_memory_b / 1073741824
                    pbar_string += f", GPU: {gpu_mem_usage:1.2f}G"

                pbar_string += f", CPU: {cpu_mem_usage:1.2f}G"
                pbar_string += f", t_pred: {tpredict:1.2f}s"
                pbar_string += f", t_post: {tpostprocess:1.2f}s"

                pbar.set_postfix_str(pbar_string)

        return self.post_process(self.config.postprocess.stateful)
