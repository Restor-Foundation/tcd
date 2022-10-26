import gc
import logging
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import psutil
import rasterio
import torch
import torchvision
from PIL import Image
from tqdm.auto import tqdm

from data import dataloader_from_image

logger = logging.getLogger("__name__")


class TiledModel(ABC):
    def __init__(self, config):

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def on_after_predict(self, results):
        pass

    def post_process(self):
        pass

    def attempt_reload(self, timeout_s=60):

        if "cuda" not in self.device:
            return

        del self.model
        torch.cuda.synchronize()
        self.load_model()

    def predict_tiled(
        self,
        image_path,
        tile_size=1024,
        overlap=512,
        confidence_thresh=0.5,
        stateful=True,
        output_folder=None,
        skip_empty=True,
    ):
        """Run inference on an image using tiling

        The output from this function is a list of predictions per-tile. Each output is a standard detectron2 result
        dictionary with the associated geo bounding box. This can be used to geo-locate the predictions, or to map
        to the original image.

        Args:
            image_path (str): Path to image file
            tile_size (int, optional): Tile size. Defaults to 1024.
            overlap (int, optional): Amount to pad around image when tiling (e.g. stride). Defaults to 512.
            confidence_thresh (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            skip_empty (bool, optional): Skip empty/all-black images. Defaults to True.

        Returns:
            list(tuple(prediction, bounding_box)): A list of predictions and the bounding boxes for those detections.
        """

        dataloader = dataloader_from_image(
            image_path, tile_size_px=tile_size, stride_px=tile_size - overlap
        )

        input_image = rasterio.open(image_path)

        # TODO: Handle this better
        if self.post_processor is not None:
            self.post_processor.initialise(input_image)

        image_dir = os.path.dirname(image_path)
        image_basename, image_ext = os.path.splitext(os.path.basename(image_path))

        if stateful:
            if output_folder is None:
                output_folder = os.path.join(image_dir, image_basename + "_pred")
                os.makedirs(output_folder, exist_ok=True)
            else:
                assert os.path.exists(output_folder)

        pbar = tqdm(dataloader, total=len(dataloader))
        self.failed_images = []
        self.should_exit = False
        self.confidence_thresh = confidence_thresh

        # Predict on each tile
        for batch in pbar:

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

            predictions = self.predict(image).to("cpu")

            # Typically if this happens we hit an OOM...
            if predictions is None:
                pbar.set_postfix_str("Error")
                logger.error("Failed to run inference on image.")
                self.failed_images.append(image)
            else:

                process = psutil.Process(os.getpid())
                cpu_mem_usage = process.memory_info().rss / 1073741824

                pbar_string = f"Instances: {len(predictions)}"

                if "cuda" in self.device:
                    gpu_mem_usage = used_memory_b / 1073741824
                    pbar_string += f", GPU: {gpu_mem_usage:1.2f}G"

                pbar_string += f", CPU: {cpu_mem_usage:1.2f}G"

                pbar.set_postfix_str(pbar_string)

            self.on_after_predict((predictions, batch["bbox"][0]))

        return self.post_process()
