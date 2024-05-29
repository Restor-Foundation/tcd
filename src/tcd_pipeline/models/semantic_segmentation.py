"""Semantic segmentation model framework, using smp models"""

import logging
import time
import traceback
import warnings
from abc import abstractmethod
from typing import List

import torch
import torch.multiprocessing

from tcd_pipeline.models.model import Model
from tcd_pipeline.postprocess.semanticprocessor import SemanticSegmentationPostProcessor

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class SemanticSegmentationModel(Model):
    """Tiled model subclass for smp semantic segmentation models."""

    def __init__(self, config):
        """Initialize the model. Does not load weights - this is
        performed automatically at prediction or train time.

        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.post_processor = SemanticSegmentationPostProcessor(config)
        self._cfg = config

    @abstractmethod
    def forward(self):
        """Model forward pass (i.e. predict), sub-classed by specific architectures"""

    def predict_batch(self, image_tensor: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run inference on a batch of images

        Args:
            image_tensor (List[torch.Tensor]): List of images to predict

        Returns:
            predictions (List[torch.Tensor]): List of output prediction tensors
        """

        self.model.eval()
        self.should_reload = False
        predictions = None

        t_start_s = time.time()

        with torch.no_grad():
            # removing alpha channel
            inputs = [im[:3, :, :].to(self.device) for im in image_tensor]

            try:
                predictions = self.forward(inputs)
            except RuntimeError as e:
                logger.error("Runtime error: %s", e)
                self.should_reload = True
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Failed to run inference: %s. Attempting to reload model.", e
                )
                self.should_reload = True

        t_elapsed_s = time.time() - t_start_s
        logger.debug("Predicted tile in %1.2fs", t_elapsed_s)

        assert len(predictions.shape) == 4

        return [p for p in predictions]

    def evaluate(self):
        """
        Evaluate the model on the dataset provided in the config.

        Does not log to wandb.
        """

        self.load_model()

        log_dir = os.path.join(
            self.config.model.log_dir, time.strftime("%Y%m%d-%H%M%S_eval")
        )
        os.makedirs(log_dir, exist_ok=True)

        # Evaluate without augmentation
        data_module = COCODataModule(
            self.config.data.data_root,
            augment=_cfg["datamodule"]["augment"] == "off",
            batch_size=int(_cfg["datamodule"]["batch_size"]),
            num_workers=int(_cfg["datamodule"]["num_workers"]),
            tile_size=int(config.data.tile_size),
        )

        csv_logger = CSVLogger(save_dir=log_dir, name="logs")

        # Eval "trainer"
        evaluator = pl.Trainer(
            logger=[csv_logger],
            default_root_dir=log_dir,
            accelerator="gpu",
            auto_lr_find=False,
            auto_scale_batch_size=False,
            devices=1,
        )

        try:
            logger.info("Starting evaluation on test data")
            evaluator.test(model=self.model, datamodule=data_module)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Evaluation failed")
            logger.error(e)
            logger.error(traceback.print_exc())
