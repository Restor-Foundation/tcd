import logging
import warnings

import segmentation_models_pytorch as smp
from torch import nn

from .segmentationmodule import SegmentationModule

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class SMPModule(SegmentationModule):
    def configure_models(self, init_pretrained=False) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""

        if self.hparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["backbone"],
                encoder_weights=self.hparams["weights"] if init_pretrained else None,
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["backbone"],
                encoder_weights=self.hparams["weights"] if init_pretrained else None,
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["model"] == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=self.hparams["backbone"],
                encoder_weights=self.hparams["weights"] if init_pretrained else None,
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
