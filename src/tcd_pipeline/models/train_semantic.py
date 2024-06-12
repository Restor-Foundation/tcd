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
import wandb
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch import nn
from tqdm.auto import tqdm

from tcd_pipeline.data.datamodule import COCODataModule

from .segformermodule import SegformerModule
from .smpmodule import SMPModule

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)
import hydra


def train(config) -> bool:
    """Train the model

    Returns:
        success (bool): True if training was successful
    """

    pl.seed_everything(42)

    log_dir = os.path.join(config.data.output)
    os.makedirs(log_dir, exist_ok=True)

    csv_logger = CSVLogger(save_dir=log_dir, name="logs")
    tb_logger = TensorBoardLogger(
        save_dir=log_dir, name="logs", version=csv_logger.version
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Removing stats monitor because it clutters logs and
    # doesn't seem to be particularly useful.
    # stats_monitor = DeviceStatsMonitor(cpu_stats=True)

    logger.info(f"Logging to: {csv_logger.log_dir}")
    os.makedirs(csv_logger.log_dir, exist_ok=True)

    from omegaconf import OmegaConf

    OmegaConf.save(config, os.path.join(csv_logger.log_dir, "pipeline_config.yaml"))

    # For convenience
    ckpt = config.model.checkpoint

    if ckpt == "last":
        logger.info("Attempting to find most recent checkpoint")
        from glob import glob

        checkpoints = glob(os.path.join(log_dir, "*", "*", "checkpoints", "last.ckpt"))

        checkpoints = sorted(checkpoints, key=lambda x: os.stat(x).st_ctime)

        if len(checkpoints) > 0:
            ckpt = checkpoints[-1]
        else:
            raise FileNotFoundError("No checkpoints were found in the output directory")

    if ckpt is not None:
        logger.info(f"Attempting to resume from {ckpt}")

    if config.model.name == "segformer":
        model = SegformerModule(
            model=config.model.name,
            backbone=config.model.backbone,
            ignore_index=None,
            id2label=os.path.join(
                os.path.dirname(__file__), "index_to_name_binary.json"
            ),
            learning_rate=float(config.model.learning_rate),
            learning_rate_schedule_patience=int(
                config.model.learning_rate_schedule_patience
            ),
        )

    else:
        model = SMPModule(
            model=config.model.name,
            backbone=config.model.backbone,
            weights=config.model.pretrained,
            in_channels=int(config.model.in_channels),
            num_classes=int(config.model.num_classes),
            loss=config.model.loss,
            ignore_index=None,
            learning_rate=float(config.model.learning_rate),
            learning_rate_schedule_patience=int(
                config.model.learning_rate_schedule_patience
            ),
        )

    # Common setup, don't need to do this if only evaluating
    model.configure_models(init_pretrained=True if not ckpt else False)
    model.configure_losses()
    model.configure_metrics()

    # load data
    datamodule_config = config.model.datamodule
    data_module = COCODataModule(
        config.data.root,
        train_path=config.data.train,
        val_path=config.data.validation,
        test_path=config.data.validation,
        augment=config.model.augment == "on",
        batch_size=int(config.model.batch_size),
        num_workers=int(config.model.num_workers),
        tile_size=int(config.data.tile_size),
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

    batch_size = int(config.model.batch_size)
    if batch_size > 32:
        accumulate = 1
    else:
        accumulate = max(1, int(32 / batch_size))

    loggers = [tb_logger, csv_logger]

    matmul_precision = "medium"
    torch.set_float32_matmul_precision(matmul_precision)

    trainer_config = config.model.trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        logger=loggers,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=int(trainer_config.max_epochs),
        accumulate_grad_batches=accumulate,
        fast_dev_run=trainer_config.debug_run,
        devices=1,
    )

    try:
        logger.info("Starting trainer")
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt)
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Training failed")
        logger.error(e)
        logger.error(traceback.print_exc())
        wandb.finish()
        exit(1)

    try:
        logger.info("Train complete, starting test")
        trainer.test(model=model, datamodule=data_module, verbose=True)
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Training failed at test time")
        logger.error(e)
        logger.error(traceback.print_exc())
        wandb.finish()
        exit(1)

    if config.model.name == "segformer":
        # Dump initial config/model so we can load checkpoints later.

        if os.path.exists(checkpoint_callback.best_model_path):
            logger.info("Saving model state dictionary")
            model = SegformerModule.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )

            torch.save(
                model.model.state_dict(),
                os.path.join(
                    os.path.dirname(checkpoint_callback.best_model_path), "best.pt"
                ),
            )

    wandb.finish()
    return True
