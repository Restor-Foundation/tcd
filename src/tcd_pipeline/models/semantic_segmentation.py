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

    def evaluate(self, augment=False):
        """
        Evaluate the model on the dataset provided in the config.

        Does not log to wandb.
        """

        import os

        import lightning.pytorch as pl
        from lightning.pytorch.loggers import CSVLogger

        from tcd_pipeline.data.datamodule import COCODataModule

        self.load_model()

        # Rely on Lightning for directory setup within this folder
        log_dir = self.config.data.output

        os.makedirs(log_dir, exist_ok=True)

        # Evaluate without augmentation
        data_module = COCODataModule(
            self.config.data.root,
            train_path=self.config.data.train,
            val_path=self.config.data.validation,
            test_path=self.config.data.validation,
            augment=augment,
            batch_size=int(self.config.model.batch_size),
            num_workers=int(self.config.model.num_workers),
            tile_size=int(self.config.data.tile_size),
        )

        csv_logger = CSVLogger(save_dir=log_dir, name="logs")

        # Eval "trainer"
        evaluator = pl.Trainer(
            logger=[csv_logger],
            default_root_dir=log_dir,
            accelerator=self.device,
            devices=1,
        )

        from tcd_pipeline.models.segformermodule import SegformerModule
        from tcd_pipeline.models.smpmodule import SMPModule

        if self.config.model.name == "segformer":
            module = SegformerModule(
                model=self.config.model.name,
                backbone=self.config.model.backbone,
                ignore_index=None,
                id2label=os.path.join(
                    os.path.dirname(__file__), "index_to_name_binary.json"
                ),
            )
        elif self.config.model.name == "unet":
            module = SMPModule(
                model=self.config.model.name,
                backbone=self.config.model.backbone,
                weights=self.config.model.pretrained,
                in_channels=int(self.config.model.in_channels),
                num_classes=int(self.config.model.num_classes),
                loss=self.config.model.loss,
            )
        else:
            raise NotImplementedError

        # Drop in model we've just loaded
        logger.info("Initialising empty Lightning module")
        module.configure_models(init_pretrained=False)
        module.configure_losses()
        module.configure_metrics()
        logger.info(f"Mapping model weights from {self.config.model.weights}")
        module.model = self.model

        try:
            logger.info("Starting evaluation on test data")
            evaluator.test(model=module, datamodule=data_module)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Evaluation failed")
            logger.error(e)
            logger.error(traceback.print_exc())
