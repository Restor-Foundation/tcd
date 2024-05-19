import logging
import warnings
from typing import Any

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torchvision
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

import wandb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class SegmentationModule(pl.LightningModule):
    """Semantic segmentation trainer with additional metric calculation

    This trainer is loosely based on torchgeo's but with some extra
    bits for more informative logging and to remove an additional
    dependency on the library.
    """

    def __init__(self, **kwargs):
        """Initialise the task and setup metrics for training

        Training metrics are: accuracy, precision, recall, f1,
        jaccard index (iou), dice and confusion matrices.

        During testing, we also compute a PR curve.

        """
        super().__init__()

        self.ignore_index = None
        self.example_input_array = torch.rand((1, 3, 1024, 1024))
        self.save_hyperparameters()
        self.configure_metrics()

    def configure_models(self):
        pass

    def configure_losses(self):
        pass

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
            y_hat (torch.Tensor): Logit output from the model
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

        if f"{split}/confusion_matrix" in computed:
            conf_mat = computed.pop(f"{split}/confusion_matrix").cpu().numpy()

        # Log everything else
        logger.debug("Logging %s metrics", split)
        self.log_dict(computed)

        if not wandb.run:
            return

        # TODO Fix confusion matrix logging
        """
        if split in ["val", "test"] and f"{split}_confusion_matrix" in computed:
            conf_mat = (conf_mat / np.sum(conf_mat)) * 100
            cm_plot = px.imshow(conf_mat, text_auto=".2f")
            logger.debug("Logging %s confusion matrix", split)
            wandb.log({f"{split}_confusion_matrix": cm_plot})
        """

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
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        raise NotImplementedError

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
