"""Semantic segmentation model framework, using smp models"""

import logging
import os
import time
import traceback
import warnings
from typing import List

import lightning.pytorch as pl
import torch
import torch.multiprocessing
import ttach as tta
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch import nn
from tqdm.auto import tqdm

import wandb
from tcd_pipeline.data.datamodule import TCDDataModule
from tcd_pipeline.models.model import TiledModel
from tcd_pipeline.postprocess.semanticprocessor import SemanticSegmentationPostProcessor

from .segformermodule import SegformerModule
from .smpmodule import SMPModule

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class SemanticSegmentationModel(TiledModel):
    """Tiled model subclass for smp semantic segmentation models."""

    def __init__(self, config):
        """Initialize the model.

        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.post_processor = SemanticSegmentationPostProcessor(config)
        self._cfg = config

    def load_model(self, strict=True):
        """Load the model from a checkpoint"""

        logging.info(
            "Loading checkpoint: %s", os.path.abspath(self.config.model.weights)
        )

        if not os.path.exists(self.config.model.weights):
            logging.warn("Checkpoint does not exist - perhaps you need to train first?")
            return

        if self.config.model.config.model == "segformer":
            logger.info("Loading segformer-type model")
            module = SegformerModule
        else:
            logger.info("Loading SMP-type model")
            module = SMPModule

        self.model = module.load_from_checkpoint(
            self.config.model.weights, strict=strict
        ).to(self.device)

        logger.info("Loaded model from checkpoint.")

        assert self.model is not None

        if self.config.model.tta:
            logger.info("Using test-time augmentation")
            self.transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 180]),
                    tta.Scale(scales=[1, 0.5]),
                ]
            )

    def train(self):
        """Train the model

        Returns:
            bool: True if training was successful
        """

        pl.seed_everything(42)

        ckpt = None
        """
        if self.config.model.ckpt:
            assert os.path.exists(self.config.model.ckpt)
            ckpt = self.config.model.ckpt
            log_dir = ckpt
        else:
        """

        log_dir = os.path.join(self.config.data.output)
        os.makedirs(log_dir, exist_ok=True)

        csv_logger = CSVLogger(save_dir=log_dir, name="logs")
        tb_logger = TensorBoardLogger(
            save_dir=log_dir, name="logs", version=csv_logger.version
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        stats_monitor = DeviceStatsMonitor(cpu_stats=True)

        logger.info(f"Logging to: {csv_logger.log_dir}")
        os.makedirs(csv_logger.log_dir, exist_ok=True)

        from omegaconf import OmegaConf

        OmegaConf.save(
            self.config, os.path.join(csv_logger.log_dir, "pipeline_config.yaml")
        )

        # For convenience
        model_config = self._cfg.model.config

        if model_config.model == "segformer":
            self.model = SegformerModule(
                model=model_config.model,
                backbone=model_config.backbone,
                ignore_index=None,
                id2label=os.path.join(
                    os.path.dirname(__file__), "index_to_name_binary.json"
                ),
                learning_rate=float(model_config.learning_rate),
                learning_rate_schedule_patience=int(
                    model_config.learning_rate_schedule_patience
                ),
                checkpoint=ckpt,
            )

            # Dump initial config/model so we can load checkpoints later.
            self.model.processor.save_pretrained(csv_logger.log_dir)
            self.model.model.save_pretrained(csv_logger.log_dir)

        else:
            self.model = SMPModule(
                model=model_config.model,
                backbone=model_config.backbone,
                weights="imagenet",
                in_channels=int(model_config.in_channels),
                num_classes=int(model_config.num_classes),
                loss=model_config.loss,
                ignore_index=None,
                learning_rate=float(model_config.learning_rate),
                learning_rate_schedule_patience=int(
                    model_config.learning_rate_schedule_patience
                ),
            )

        # Common setup, don't need to do this if only evaluating
        self.model.save_hyperparameters()
        self.model.configure_models()
        self.model.configure_losses()
        self.model.configure_metrics()

        # load data
        datamodule_config = self._cfg.model.datamodule
        data_module = TCDDataModule(
            self.config.data.root,
            train_path=self.config.data.train,
            val_path=self.config.data.validation,
            test_path=self.config.data.validation,
            augment=datamodule_config.augment == "on",
            batch_size=int(model_config.batch_size),
            num_workers=int(datamodule_config.num_workers),
            tile_size=int(self.config.data.tile_size),
        )

        # checkpoints and loggers
        checkpoint_callback = ModelCheckpoint(
            monitor="val/multiclassf1score_tree",
            mode="max",
            auto_insert_metric_name=False,
            save_top_k=1,
            filename="{epoch}-f1tree:{val/multiclassf1score_tree:.2f}-loss:{val/loss:.2f}",
            save_last=True,
            verbose=True,
        )

        batch_size = int(model_config.batch_size)
        if batch_size > 32:
            accumulate = 1
        else:
            accumulate = max(1, int(32 / batch_size))

        loggers = [tb_logger, csv_logger]

        matmul_precision = "medium"
        torch.set_float32_matmul_precision(matmul_precision)

        trainer_config = self.config.model.trainer
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, lr_monitor, stats_monitor],
            logger=loggers,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=int(trainer_config.max_epochs),
            accumulate_grad_batches=accumulate,
            fast_dev_run=trainer_config.debug_run,
            devices=1,
        )

        try:
            logger.info("Starting trainer")
            trainer.fit(model=self.model, datamodule=data_module, ckpt_path=ckpt)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
            logger.error(traceback.print_exc())
            wandb.finish()
            exit(1)

        try:
            logger.info("Train complete, starting test")
            trainer.test(model=self.model, datamodule=data_module)
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
            logger.error(traceback.print_exc())
            wandb.finish()
            exit(1)

        wandb.finish()
        return True

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
        data_module = TCDDataModule(
            self.config.data.data_root,
            augment=self._cfg["datamodule"]["augment"] == "off",
            batch_size=int(self._cfg["datamodule"]["batch_size"]),
            num_workers=int(self._cfg["datamodule"]["num_workers"]),
            tile_size=int(self.config.data.tile_size),
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

    def _predict_tensor(self, image_tensor: List[torch.Tensor]):
        """Run inference on an image tensor

        Args:
            image (List[torch.Tensor]): Path to image, or, float tensor in CHW order, un-normalised

        Returns:
            predictions: Detectron2 prediction dictionary
        """

        self.model.eval()
        self.should_reload = False
        predictions = None

        t_start_s = time.time()

        with torch.no_grad():
            # removing alpha channel
            inputs = [im[:3, :, :].to(self.device) for im in image_tensor]

            try:
                predictions = self.model(inputs)
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

        return predictions
