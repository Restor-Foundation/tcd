import argparse
import configparser
import json
import logging
import os
import time
import traceback
import warnings
from typing import Optional

import albumentations as A
import cv2
import dotmap
import numpy as np
import plotly.express as px
import rasterio
import torch
import torchvision
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import SemanticSegmentationTask
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

from .. import downsample
from ..config import load_config
from ..post_processing import SegmentationPostProcessor
from .model import TiledModel

logger = logging.getLogger("__name__")

# TODO fix warnings
warnings.filterwarnings("ignore")

# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, data_dir, setname, transform, FACTOR=1):

        self.data_dir = data_dir
        self.setname = setname
        self.FACTOR = FACTOR
        assert setname in ["train", "test", "val"]

        with open(self.data_dir + setname + "_20221010.json", "r") as file:
            self.metadata = json.load(file)

        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.metadata["images"])

    def __getitem__(self, idx):
        annotation = self.metadata["images"][idx]

        img_name = annotation["file_name"]
        coco_idx = annotation["id"]

        if self.FACTOR == 1:
            img_path = os.path.join(self.data_dir, "images", img_name)
            try:
                mask = np.load(
                    os.path.join(
                        self.data_dir, "masks", f"{self.setname}_mask_{coco_idx}.npz"
                    )
                )["arr_0"].astype(int)
            except:
                return None
        else:
            img_path = os.path.join(
                self.data_dir,
                "downsampled_images",
                f"sampling_factor_{self.FACTOR}/{img_name}",
            )

            try:
                mask = np.load(
                    os.path.join(
                        self.data_dir,
                        "downsampled_masks",
                        f"sampling_factor_{self.FACTOR}",
                        f"{self.setname}_mask_{coco_idx}.npz",
                    )
                )["arr_0"].astype(int)
            except:
                return None

        # Albumentations handles conversion to torch tensor
        image = np.array(Image.open(img_path), dtype=np.float32)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        return {"image": image, "mask": mask}


class TreeDataModule(LightningDataModule):
    def __init__(self, conf, data_frac=1.0, batch_size=1, augment=True):
        super().__init__()
        self.conf = conf
        self.data_frac = data_frac
        self.augment = augment
        self.batch_size = batch_size

        wandb.run.summary["batch_size"] = self.batch_size

    def prepare_data(self) -> None:

        if self.augment:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(),
                    A.RandomBrightnessContrast(p=0.2),
                    ToTensorV2(),
                ]
            )
            logger.debug("Train-time augmentation is enabled.")
        else:
            transform = None

        self.train_data = ImageDataset(self.conf.data_dir, "train", transform=transform)
        self.val_data = ImageDataset(self.conf.data_dir, "val", transform=None)
        self.test_data = ImageDataset(self.conf.data_dir, "test", transform=None)

    def train_dataloader(self):
        return get_dataloaders(
            self.conf,
            self.train_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
        )[0]

    def val_dataloader(self):
        return get_dataloaders(
            self.conf,
            self.val_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
        )[0]

    def test_dataloader(self):
        return get_dataloaders(
            self.conf,
            self.test_data,
            data_frac=self.data_frac,
            batch_size=self.batch_size,
            shuffle=False,
        )[0]


def get_dataloaders(conf, *datasets, data_frac=1, batch_size=1, shuffle=True):

    if data_frac != 1.0:
        datasets = [
            torch.utils.data.Subset(
                dataset,
                np.random.choice(
                    len(dataset), int(len(dataset) * data_frac), replace=False
                ),
            )
            for dataset in datasets
        ]

    return [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
        )
        for dataset in datasets
    ]


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class SemanticSegmentationTaskPlus(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class_labels = ["background", "tree"]

        self.train_metrics = MetricCollection(
            metrics={
                "accuracy": ClasswiseWrapper(
                    Accuracy(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "precision": ClasswiseWrapper(
                    Precision(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "recall": ClasswiseWrapper(
                    Recall(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "f1": ClasswiseWrapper(
                    F1Score(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "jaccard_index": ClasswiseWrapper(
                    JaccardIndex(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "dice": ClasswiseWrapper(
                    Dice(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                "confusion_matrix": ConfusionMatrix(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
            },
            prefix="train_",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val_")

        self.test_metrics = self.train_metrics.clone(prefix="test_")
        # Note, since this is computed over all images, this can be *extremely*
        # compute intensive to calculate in full. Best done once at the end of training.
        # Setting thresholds in advance uses constant memory.
        self.test_metrics.add_metrics(
            {
                "pr_curve": MulticlassPrecisionRecallCurve(
                    num_classes=self.hyperparams["num_classes"],
                    thresholds=torch.from_numpy(np.linspace(0, 1, 20)),
                )
            }
        )

    def config_task(self) -> None:
        import segmentation_models_pytorch as smp
        from torch import nn

        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hyperparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hyperparams["encoder_name"],
                encoder_weights=self.hyperparams["encoder_weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hyperparams["encoder_name"],
                encoder_weights=self.hyperparams["encoder_weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["segmentation_model"] == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=self.hyperparams["encoder_name"],
                encoder_weights=self.hyperparams["encoder_weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['segmentation_model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hyperparams["loss"] == "ce":
            ignore_value = -1000 if self.ignore_index is None else self.ignore_index
            self.loss = nn.CrossEntropyLoss(ignore_index=ignore_value)
        elif self.hyperparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hyperparams["num_classes"]
            )
        elif self.hyperparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=self.ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def log_image(self, image, key, caption=""):
        logger.debug(f"Logging image ({caption})")
        images = wandb.Image(image, caption)
        wandb.log({key: images})

    def validation_step(self, *args, **kwargs) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]

        if batch_idx < 10:

            loss, y_hat, y_hat_hard = self._predict_batch(batch)
            y = batch["mask"]
            self.log(f"val_loss", loss, on_step=False, on_epoch=True)
            self.val_metrics(y_hat, y)
            batch["prediction"] = y_hat_hard

            sm = torch.nn.Softmax2d()
            batch["probability"] = sm(y_hat)

            self._log_prediction_images(batch, "val")

    def test_step(self, *args, **kwargs) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        loss, y_hat, y_hat_hard = self._predict_batch(batch)
        y = batch["mask"]
        self.log(f"test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat, y)
        batch["prediction"] = y_hat_hard

        sm = torch.nn.Softmax2d()
        batch["probability"] = sm(y_hat)

        self._log_prediction_images(batch, "test")

    def _predict_batch(self, batch):
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        return loss, y_hat, y_hat_hard

    def _log_prediction_images(self, batch, split):
        """Plot images during training

        Parameters
        ----------
        batch
            batch from dataloader
        split
            dataset split (e.g. 'test', 'train', 'validation')
        """

        try:
            for key in ["image", "mask", "prediction", "probability"]:
                batch[key] = batch[key].cpu()

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
            logger.debug(f"Logging {split} images")
            self.log_image(
                image_grid,
                key=f"{split}_examples (original/ground truth/prediction/confidence)",
                caption=f"Sample {split} images",
            )
        except AttributeError as e:
            logger.error(e)

    def _log_metrics(self, computed, split):
        """Logs metrics for a particular split

        Parameters
        ----------
        computed
            Dictionary of results from a MetricCollection
        split
            Dataset split
        """

        # Pop + log confusion matrix
        conf_mat = computed.pop(f"{split}_confusion_matrix").cpu().numpy()

        if split in ["val", "test"]:
            conf_mat = (conf_mat / np.sum(conf_mat)) * 100
            cm = px.imshow(conf_mat, text_auto=".2f")
            logger.debug(f"Logging {split} confusion matrix")
            wandb.log({f"{split}_confusion_matrix": cm})

        # Pop + log PR curve
        key = f"{split}_pr_curve"
        if key in computed:
            logger.info("Logging PR curve")

            precision, recall, _ = computed.pop(key)
            classes = ["background", "tree"]

            for i in range(len(precision)):

                recall_np = recall[i].cpu().numpy()
                precision_np = precision[i].cpu().numpy()

                data = [[x, y] for (x, y) in zip(recall_np, precision_np)]

                table = wandb.Table(data=data, columns=["Recall", "Precision"])
                wandb.log(
                    {
                        f"{split}_pr_curve_{classes[i]}": wandb.plot.line(
                            table,
                            "Recall",
                            "Precision",
                            title=f"Precision Recall for {classes[i]}",
                        )
                    }
                )

        # Log everything else
        logger.debug(f"Logging {split} metrics")
        self.log_dict(computed)

    def training_epoch_end(self, outputs):
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        self._log_metrics(computed, "train")
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs):
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        computed = self.val_metrics.compute()
        self._log_metrics(computed, "val")
        self.val_metrics.reset()

    def test_epoch_end(self, outputs):
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        computed = self.test_metrics.compute()
        self._log_metrics(computed, "test")
        self.test_metrics.reset()


class SemanticSegmentationModel(TiledModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_processor = SegmentationPostProcessor(config)

    def load_model(self):

        with open(self.config.model.config, "r") as fp:
            self._cfg = dotmap.DotMap(yaml.load(fp, yaml.SafeLoader))

        logging.info(f"Loading checkpoint: {self.config.model.weights}")
        self.model = SemanticSegmentationTaskPlus.load_from_checkpoint(
            self.config.model.weights, stric=True
        )

    def train(self):

        # init wandb
        wandb.init(
            # config=conf["model"],
            entity="dsl-ethz-restor",
            project=self._cfg.wandb.project_name,
        )

        self.model = SemanticSegmentationTaskPlus(
            segmentation_model=self._cfg.model.segmentation_model,
            encoder_name=self._cfg.model.backbone,
            encoder_weights="imagenet",
            in_channels=int(self._cfg.model.in_channels),
            num_classes=int(self._cfg.model.num_classes),
            loss=self._cfg.model.loss,
            ignore_index=None,
            learning_rate=float(self._cfg.model.learning_rate),
            learning_rate_schedule_patience=int(
                self._cfg.model.learning_rate_schedule_patience
            ),
        )

        # load data
        data_module = TreeDataModule(
            self.config,
            augment=conf["datamodule"]["augment"] == "on",
            batch_size=int(conf["datamodule"]["batch_size"]),
        )

        log_dir = os.path.join(LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)

        # checkpoints and loggers
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1score_tree",
            mode="max",
            dirpath=os.path.join(log_dir, "checkpoints"),
            auto_insert_metric_name=True,
            save_top_k=1,
            save_last=True,
            verbose=True,
        )

        # Setting a short patience of ~O(10) might trigger premature stopping due to a "lucky" batch.
        stopping_patience = int(conf["trainer"]["early_stopping_patience"])
        early_stopping_callback = EarlyStopping(
            monitor="val_f1score_tree",
            min_delta=0.00,
            patience=stopping_patience,
            check_finite=True,
            mode="max",
        )

        csv_logger = CSVLogger(save_dir=log_dir, name="logs")
        wandb_logger = WandbLogger(
            project=conf["wandb"]["project_name"], log_model=True
        )  # log_model='all' cache gets full quite fast

        lr_monitor = LearningRateMonitor(logging_interval="step")
        stats_monitor = DeviceStatsMonitor(cpu_stats=True)

        wandb.run.summary["log_dir"] = log_dir

        # trainer

        auto_scale_batch = conf["trainer"]["auto_scale_batch_size"] == "True"
        auto_lr = conf["trainer"]["auto_lr_find"] == "True"

        trainer = Trainer(
            callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
            logger=[csv_logger, wandb_logger],
            default_root_dir=log_dir,
            accelerator="gpu",
            max_epochs=int(conf["trainer"]["max_epochs"]),
            max_time=conf["trainer"]["max_time"],
            auto_lr_find=auto_lr,
            auto_scale_batch_size="binsearch" if auto_scale_batch else False,
            fast_dev_run=conf["trainer"]["debug_run"] == "True",
        )

        wandb_logger.watch(task.model, log="parameters", log_graph=True)

        if auto_scale_batch or auto_lr:
            try:
                logger.info("Tuning trainer")
                trainer.tune(model=task, train_dataloaders=data_module)
            except Exception as e:
                logger.error("Tuning stage failed")
                logger.error(e)
                logger.error(traceback.print_exc())

        try:
            logger.info("Starting trainer")
            trainer.fit(model=task, datamodule=data_module)
        except Exception as e:
            logger.error("Training failed")
            logger.error()
            logger.error(traceback.print_exc())

        try:
            logger.info("Train complete, starting test")
            trainer.test(model=task, datamodule=data_module)
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
            logger.error(traceback.print_exc())

    def evaluate(self, dataset, output_folder):
        pass

    def predict(self, image):
        """Run inference on an image file or Tensor

        Args:
            image (Union[str, torch.Tensor]): Path to image, or, float tensor in CHW order, un-normalised

        Returns:
            predictions: Detectron2 prediction dictionary
        """

        if isinstance(image, str):
            image = np.array(Image.open(image))
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        elif isinstance(image, torch.Tensor):
            image_tensor = image
        elif isinstance(image, rasterio.io.DatasetReader):
            image_tensor = torch.as_tensor(image.read().astype("float32"))
        else:
            logger.error(
                f"Provided image of type {type(image)} which is not supported."
            )
            raise NotImplementedError

        if self.model is None:
            self.load_model()

        self.model.eval()
        self.should_reload = False
        predictions = None

        t_start_s = time.time()

        with torch.no_grad():
            # removing alpha channel
            inputs = torch.unsqueeze(image_tensor[:3, :, :], dim=0)

            try:
                predictions = self.model(inputs)
            except RuntimeError as e:
                logger.error(f"Runtime error: {e}")
                self.should_reload = True
            except Exception as e:
                logger.error(
                    f"Failed to run inference: {e}. Attempting to reload model."
                )
                self.should_reload = True

        t_elapsed_s = time.time() - t_start_s
        logger.debug(f"Predicted tile in {t_elapsed_s:1.2f}s")

        return predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.load(fp, yaml.SafeLoader)

    # sweep: hyperparameter tuning
    project_name = config["wandb"]["project_name"]
    logger.info(f"Using project {project_name}")

    if config["experiment"]["sweep"] == "True":

        logger.info("Sweep enabled")

        sweep_file = "conf_sweep.yaml"
        with open(sweep_file, "r") as fp:
            conf_sweep = yaml.safe_load(fp)

        sweep_configuration = {
            "method": "grid",
            "name": "vanilla-model-sweep-runs",
            "program": "vanilla_model.py",
            "metric": {"goal": "minimize", "name": "loss"},
            "parameters": {
                "loss": {
                    "values": conf_sweep["parameters"]["loss"]["values"]
                },  # ['ce','focal']
                "segmentation_model": {
                    "values": conf_sweep["parameters"]["segmentation_model"]["values"]
                },  # ['unet','deeplabv3+']
                "encoder_name": {
                    "values": conf_sweep["parameters"]["encoder_name"]["values"]
                },  # ['resnet18','resnet34','resnet50']
                "augment": {
                    "values": conf_sweep["parameters"]["augment"]["values"]
                },  # ['off','on']
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(
            sweep_id, project=conf["wandb"]["project_name"], function=train
        )  # , count=5)
        logger.debug("Logging sweep config")
        wandb.log(sweep_configuration)

    torch.cuda.empty_cache()
    train()

    wandb.finish()
