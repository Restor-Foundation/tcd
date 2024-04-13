"""Semantic segmentation model framework, using smp models"""
import logging
import os
import time
import traceback
import warnings
from typing import Any

import cv2
import lightning.pytorch as pl
import numpy as np
import plotly.express as px
import segmentation_models_pytorch as smp
import torch
import torch.multiprocessing
import torchvision
import ttach as tta
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch import nn
from torchmetrics import (
    Accuracy,
    ClasswiseWrapper,
    ConfusionMatrix,
    Dice,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from torchvision.utils import draw_segmentation_masks
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

import wandb
from tcd_pipeline.data.datamodule import TCDDataModule
from tcd_pipeline.models.model import TiledModel
from tcd_pipeline.postprocess.semanticprocessor import SemanticSegmentationPostProcessor

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger("__name__")
warnings.filterwarnings("ignore")


class SMPTrainer(pl.LightningModule):
    """Semantic segmentation trainer with additional metric calculation

    This trainer is loosely based on torchgeo's but with some extra
    bits for more informative logging and to remove an additional
    dependency on the library.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the task and setup metrics for training

        Training metrics are: accuracy, precision, recall, f1,
        jaccard index (iou), dice and confusion matrices.

        During testing, we also compute a PR curve.

        Args:
            *args: Arguments to pass to the SemanticSegmentationTask
            **kwargs: Keyword arguments to pass to the SemanticSegmentationTask

        """
        super().__init__()

        self.ignore_index = None
        self.save_hyperparameters()
        self.configure_models()
        self.configure_losses()
        self.configure_metrics()

        self.example_input_array = torch.rand((1, 3, 1024, 1024))

    def configure_metrics(self) -> None:
        metric_task = "multiclass"
        class_labels = ["background", "tree"]
        self.num_classes = len(class_labels)

        self.train_metrics = MetricCollection(
            metrics={
                "accuracy": ClasswiseWrapper(
                    Accuracy(
                        task=metric_task,
                        num_classes=self.num_classes,
                        ignore_index=self.ignore_index,
                        multidim_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "precision": ClasswiseWrapper(
                    Precision(
                        task=metric_task,
                        num_classes=self.num_classes,
                        ignore_index=self.ignore_index,
                        multidim_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "recall": ClasswiseWrapper(
                    Recall(
                        task=metric_task,
                        num_classes=self.num_classes,
                        ignore_index=self.ignore_index,
                        multidim_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "f1": ClasswiseWrapper(
                    F1Score(
                        task=metric_task,
                        num_classes=self.num_classes,
                        ignore_index=self.ignore_index,
                        multidim_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "jaccard_index": JaccardIndex(
                    task=metric_task,
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                ),
            },
            prefix="train/",
        )

        logger.info("Setup training metrics")

        self.val_metrics = self.train_metrics.clone(prefix="val/")
        logger.info("Setup val metrics")

        self.test_metrics = self.train_metrics.clone(prefix="test/")
        logger.info("Setup test metrics")
        # Note, since this is computed over all images, this can be *extremely*
        # compute intensive to calculate in full. Best done once at the end of training.
        # Setting thresholds in advance uses constant memory.
        self.test_metrics.add_metrics(
            {
                "pr_curve": MulticlassPrecisionRecallCurve(
                    num_classes=self.num_classes,
                    thresholds=torch.from_numpy(np.linspace(0, 1, 20)),
                ),
                "confusion_matrix": ConfusionMatrix(
                    task=metric_task,
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                ),
            }
        )

    def configure_models(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""

        if self.hparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["backbone"],
                encoder_weights=self.hparams["weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["backbone"],
                encoder_weights=self.hparams["weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["model"] == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=self.hparams["backbone"],
                encoder_weights=self.hparams["weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]

        weight = self.hparams.get("class_weights")

        if loss == "ce":
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=weight
            )
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def log_image(
        self, image: torch.Tensor, key: str, caption: str = "", prefix=""
    ) -> None:
        """Log an image to wandb

        Args:
            image (torch.Tensor): Image to log
            key (str): Key to use for logging
            caption (str, optional): Caption to use for logging. Defaults to "".

        """
        logger.debug("Logging image (%s)", caption)

        self.logger.experiment.add_image(
            f"{prefix}/images/rgb",
            image,
            global_step=self.trainer.global_step,
            dataformats="CHW",
        )

    # pylint: disable=arguments-differ
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Compute validation loss and log example predictions.

        Only logs sample images for the first 10 batches.

        Args:
            batch (dict): output from dataloader
            batch_idx (int): batch index

        """

        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        self.log(
            "val/loss",
            loss,
            batch_size=len(batch["mask"]),
            on_step=False,
            on_epoch=True,
        )
        y = batch["mask"]

        self.val_metrics(y_hat, y)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard

            p = torch.nn.Softmax2d()
            batch["probability"] = p(y_hat)

            self._log_prediction_images(batch, "val")

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        y = batch["mask"]
        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        self.log("train/loss", loss)
        self.train_metrics(y_hat_hard, y)
        self.log_dict(self.train_metrics)  # type: ignore[arg-type]

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard

            p = torch.nn.Softmax2d()
            batch["probability"] = p(y_hat)

            self._log_prediction_images(batch, "train")

        return loss

    # pylint: disable=arguments-differ
    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Compute test loss and log example predictions

        Args:
            batch (dict): output from dataloader
            batch_idx (int): batch index
        """

        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        y = batch["mask"]
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat, y)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard

            p = torch.nn.Softmax2d()
            batch["probability"] = p(y_hat)

            self._log_prediction_images(batch, "val")

    def _predict_batch(self, batch):
        """Predict on a batch of data, used in train/val/test steps

        Returns:
            loss (torch.Tensor): Loss for the batch
            y_hat (torch.Tensor): Softmax output from the model
            y_hat_hard (torch.Tensor): Argmax output from the model
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.criterion(y_hat, y)

        return loss, y_hat, y_hat_hard

    def _log_prediction_images(self, batch, split):
        """Plot images during training

        Args:
            batch (dict): output from dataloader
            split (str): dataset split (e.g. 'test', 'train', 'validation')
        """

        try:
            for key in ["image", "mask", "prediction", "probability"]:
                batch[key] = batch[key].detach().cpu()

            # Hacky probability map
            prob = np.transpose(
                cv2.cvtColor(
                    cv2.applyColorMap(
                        (255 * batch["probability"][0][1]).numpy().astype(np.uint8),
                        colormap=cv2.COLORMAP_INFERNO,
                    ),
                    cv2.COLOR_RGB2BGR,
                ),
                (2, 0, 1),
            )

            images = {
                "image": batch["image"][0],
                "masked": draw_segmentation_masks(
                    batch["image"][0].type(torch.uint8),
                    batch["mask"][0].type(torch.bool),
                    alpha=0.5,
                    colors="red",
                ),
                "prediction": draw_segmentation_masks(
                    batch["image"][0].type(torch.uint8),
                    batch["prediction"][0].type(torch.bool),
                    alpha=0.5,
                    colors="red",
                ),
                "probability": torch.from_numpy(prob),
            }
            resize = torchvision.transforms.Resize(512)
            image_grid = torchvision.utils.make_grid(
                [resize(value.float()) for _, value in images.items()],
                value_range=(0, 255),
                normalize=True,
            )
            logger.debug("Logging %s images", split)
            self.log_image(
                image_grid,
                prefix=split,
                key=f"{split}_examples (original/ground truth/prediction/confidence)",
                caption=f"Sample {split} images",
            )
        except AttributeError as e:
            logger.error(e)

    def _log_metrics(self, computed: dict, split: str):
        """Logs metrics for a particular split

        Args:
            computed (dict): Dictionary of metrics from MetricCollection
            split (str): dataset split (e.g. 'test', 'train', 'validation')

        """
        # Pop + log confusion matrix

        logger.info("Logging metrics")

        if f"{split}_confusion_matrix" in computed:
            conf_mat = computed.pop(f"{split}_confusion_matrix").cpu().numpy()

        # Log everything else
        logger.debug("Logging %s metrics", split)
        self.log_dict(computed)

        if not wandb.run:
            return

        if split in ["val", "test"] and f"{split}_confusion_matrix" in computed:
            conf_mat = (conf_mat / np.sum(conf_mat)) * 100
            cm_plot = px.imshow(conf_mat, text_auto=".2f")
            logger.debug("Logging %s confusion matrix", split)
            wandb.log({f"{split}_confusion_matrix": cm_plot})

        # Pop + log PR curve
        key = f"{split}_pr_curve"
        if key in computed:
            logger.info("Logging PR curve")

            precision, recall, _ = computed.pop(key)
            classes = ["background", "tree"]

            for pr_class in zip(precision, recall, classes):
                curr_precision, curr_recall, curr_class = pr_class

                recall_np = curr_recall.cpu().numpy()
                precision_np = curr_precision.cpu().numpy()

                data = [[x, y] for (x, y) in zip(recall_np, precision_np)]

                table = wandb.Table(data=data, columns=["Recall", "Precision"])

                wandb.log(
                    {
                        f"{split}_pr_curve_{curr_class}": wandb.plot.line(
                            table,
                            "Recall",
                            "Precision",
                            title=f"Precision Recall for {curr_class}",
                        )
                    }
                )

    def on_train_epoch_end(self) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        self._log_metrics(computed, "train")
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        computed = self.val_metrics.compute()
        self._log_metrics(computed, "val")
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        computed = self.test_metrics.compute()
        self._log_metrics(computed, "test")
        self.test_metrics.reset()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        y_hat: torch.Tensor = self(x).softmax(dim=1)
        return y_hat

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        # From https://pytorch.org/tutorials/intermediate
        # /torchvision_tutorial.html
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    verbose=True,
                    patience=self.hparams.learning_rate_schedule_patience,
                ),
                "monitor": "val/loss",
                "frequency": self.trainer.check_val_every_n_epoch,
            },
        }


class SegformerTrainer(SMPTrainer):
    model: SegformerForSemanticSegmentation

    def configure_losses(self):
        # Loss is built into transformers models
        pass

    def configure_models(self):
        import json

        id2label = json.load(open(self.hparams["id2label"], "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        self.num_classes = len(id2label)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.hparams["backbone"],
            num_labels=self.num_classes,
            id2label=id2label,
            label2id=label2id,
        )

        self.processor = SegformerImageProcessor.from_pretrained(
            self.hparams["backbone"], do_resize=False, do_reduce_labels=False
        )

    def _predict_batch(self, batch):
        """Predict on a batch of data. This function is subclassed to handle
        specific details of the transformers library since we need to

        (a) Pre-process data into the correct format (this could also be done
            at the data loader stage)

        (b) Post-process data so that the predicted masks are the correct shape
            with respect to the input. This could also be done in the dataloader
            by passing a (h, w) tuple so we know how to resize the image. However
            we should really to compute loss with respect to the original mask
            and not a downscaled one.

        Returns:
            loss (torch.Tensor): Loss for the batch
            y_hat (torch.Tensor): Softmax'd logits from the model
            y_hat_hard (torch.Tensor): Argmax output from the model (i.e. predictions)
        """

        encoded_inputs = self.processor(
            batch["image"], batch["mask"], return_tensors="pt"
        )

        # TODO Move device checking and data pre-processing to the dataloader/datamodule
        # For some reason, the processor doesn't respect device and moves everything back
        # to CPU.
        outputs = self.model(
            pixel_values=encoded_inputs.pixel_values.to(self.device),
            labels=encoded_inputs.labels.to(self.device),
        )

        # We need to reshape according to the input mask, not the encoded version
        # as the sizes are likely different. We want to keep hold of the probabilities
        # and not just the segmentation so we don't use the built-in converter:
        # y_hat_hard = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[m.shape[-2] for m in batch['mask']]))
        y_hat = nn.functional.interpolate(
            outputs.logits,
            size=batch["mask"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        y_hat_hard = y_hat.argmax(dim=1)

        return outputs.loss, y_hat, y_hat_hard

    def predict_step(self, batch):
        encoded_inputs = self.processor(
            batch["image"], reduce_size=False, return_tensors="pt"
        )

        logits = self.model(pixel_values=encoded_inputs.pixel_values).logits

        pred = nn.functional.interpolate(
            logits,
            size=encoded_inputs.pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return pred


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

    def load_model(self, strict=False):
        """Load the model from a checkpoint"""

        logging.info(
            "Loading checkpoint: %s", os.path.abspath(self.config.model.weights)
        )

        if not os.path.exists(self.config.model.weights):
            logging.warn("Checkpoint does not exist - perhaps you need to train first?")
            return

        if self.config.model.config.model == "segformer":
            trainer = SegformerTrainer
        else:
            trainer = SMPTrainer

        self.model = trainer.load_from_checkpoint(
            self.config.model.weights, strict=strict
        ).to(self.device)

        assert self.model is not None

        if self.config.model.tta:
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

        log_dir = os.path.join(self.config.data.output)
        os.makedirs(log_dir, exist_ok=True)

        # Setting a short patience of ~O(10) might trigger
        # premature stopping due to a "lucky" batch.
        # Currently disabled as it seemed unreliable to false triggers
        """
        stopping_patience = int(trainer_config.early_stopping_patience)
        
        early_stopping_callback = EarlyStopping(
            monitor="val/multiclassf1score_tree",
            min_delta=0.00,
            patience=stopping_patience,
            check_finite=True,
            mode="max",
        )       
        """

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
            self.model = SegformerTrainer(
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
            )

            # Dump initial config/model so we can load checkpoints later.
            self.model.processor.save_pretrained(csv_logger.log_dir)
            self.model.model.save_pretrained(csv_logger.log_dir)

        else:
            self.model = SMPTrainer(
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

        # load data
        datamodule_config = self._cfg.model.datamodule
        data_module = TCDDataModule(
            self.config.data.root,
            train_path=self.config.data.train,
            val_path=self.config.data.validation,
            test_path=self.config.data.validation,
            augment=datamodule_config.augment == "on",
            batch_size=int(datamodule_config.batch_size),
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

        # if wandb.run:
        #    wandb_logger.watch(self.model, log="parameters", log_graph=False)
        #    wandb.run.summary["log_dir"] = log_dir

        # auto scaling
        """
        if not debug_run:
            if auto_scale_batch or auto_lr:

                logger.info("Tuning trainer")
                results = trainer.tune(model=self.model, train_dataloaders=data_module)

                if auto_scale_batch:
                    batch_size = results["scale_batch_size"]
        """

        batch_size = int(datamodule_config.batch_size)
        if batch_size > 32:
            accumulate = 1
        else:
            accumulate = max(1, int(32 / batch_size))

        if wandb.run:
            wandb.run.summary["train_batch_size"] = batch_size
            wandb.run.summary["accumulate"] = accumulate
            wandb.run.summary["effective_batch_size"] = batch_size * accumulate

        loggers = [tb_logger, csv_logger]
        # if wandb.run:
        #    loggers.append(wandb_logger)

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
            trainer.fit(model=self.model, datamodule=data_module)
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

    def _predict_tensor(self, image_tensor: torch.Tensor):
        """Run inference on an image file or Tensor

        Args:
            image (Union[str, torch.Tensor]): Path to image, or, float tensor in CHW order, un-normalised

        Returns:
            predictions: Detectron2 prediction dictionary
        """

        self.model.eval()
        self.should_reload = False
        predictions = None

        t_start_s = time.time()

        with torch.no_grad():
            # removing alpha channel
            inputs = torch.unsqueeze(image_tensor[:3, :, :], dim=0).to(self.device)

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

        return predictions[0]
