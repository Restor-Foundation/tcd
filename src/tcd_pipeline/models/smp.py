import logging
import os
from typing import List, Union

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import SegmentationModel

from .semantic_segmentation import SemanticSegmentationModel

logger = logging.getLogger(__name__)


class SMPModel(SemanticSegmentationModel):
    """
    Wrapper around Segmentation Models Pytorch (SMP) family of semantic
    segmentation models. Currently supports unet, deeplabv3+ and unet++.

    In theory any model variant is supported, but it becomes very verbose
    specifiying all the classes. If you need to add more, it should be close
    to copy-paste in the load_model function.
    """

    model: SegmentationModel

    def setup(self):
        pass

    def load_model(self) -> None:
        """
        Load model weights. First, instantiates a model object from the
        segmentation models library. Then, if the model path exists on the
        system, those weights will be loaded. Otherwise, the model will be
        downloaded from HuggingFace Hub (assuming it exists and is accessible).

        The model weights should be a state dictionary for the specified
        architecture. Other parameters like the backbone type (e.g. resnet)
        and input/output channels should be specified via config.
        """

        logger.info("Loading SMP model")

        if self.config.model.name == "unet":
            model = smp.Unet(
                encoder_name=self.config.model.backbone,
                classes=self.config.model.num_classes,
                in_channels=self.config.model.in_channels,
            )
        elif self.config.model.name == "deeplabv3+":
            model = smp.DeepLabV3Plus(
                encoder_name=self.config.model.backbone,
                classes=self.config.model.num_classes,
                in_channels=self.config.model.in_channels,
            )
        elif self.config.model.name == "unet++":
            model = smp.UnetPlusPlus(
                encoder_name=self.config.model.backbone,
                classes=self.config.model.num_classes,
                in_channels=self.config.model.in_channels,
            )
        else:
            raise ValueError(
                f"Model type '{self.config.model.name}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'unet++'."
            )

        if not os.path.exists(self.config.model.weights):
            from huggingface_hub import HfApi

            api = HfApi()
            self.config.model.weights = api.hf_hub_download(
                repo_id=self.config.model.weights,
                filename="model.pt",
                revision=self.config.model.revision,
            )

        assert os.path.exists(self.config.model.weights)

        model.load_state_dict(
            torch.load(self.config.model.weights, map_location=self.device), strict=True
        )

        self.model = model.to(self.device)
        self.model.eval()

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: List[torch.Tensor]

        Returns:
            Interpolated semantic segmentation predictions with softmax applied
        """

        if isinstance(x, list):
            x = torch.stack(x)

        with torch.no_grad():
            logits = self.model(x)
            preds = logits.softmax(dim=1)

        return preds
