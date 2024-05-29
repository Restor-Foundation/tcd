import logging
import os
from typing import List, Union

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .semantic_segmentation import SemanticSegmentationModel

logger = logging.getLogger(__name__)


class Segformer(SemanticSegmentationModel):
    """
    Wrapper around HuggingFace's Segformer implementation.
    """

    model: SegformerForSemanticSegmentation

    def setup(self):
        """
        Performs any setup action - in the case of SegFormer,
        just checks whether to force HuggingFace to use local
        files only.
        """
        self.use_local = os.getenv("HF_FORCE_LOCAL") is not None

    def load_model(self):
        """
        Load model weights from HuggingFace Hub or local storage. The
        config key model.weights is used. If you want to force using
        local files only, you can set the environment variable:

        HF_FORCE_LOCAL

        this can be useful for testing in offline environments.

        It is assumed that the image processor has the same name as the
        model; if you're providing a local checkpoint then the 'weight'
        path should be a directory containing the state dictionary of the
        model (saved using save_pretrained) and a `preprocessor_config.json`
        file.
        """
        logger.info("Loading Segformer model")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.config.model.weights,
            local_files_only=self.use_local,
        ).to(self.device)

        self.processor = SegformerImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.config.model.weights, do_resize=False
        )

    def forward(self, x: Union[torch.tensor, List[torch.tensor]]) -> torch.Tensor:
        """Forward pass of the model. The batch is first run
        through the processor which constructs a dictionary of inputs
        for the model. This processor handles varying types of input, for
        example tensors, numpy arrays, PIL images. Within the pipeline
        this function is normally called with images pre-converted to tensors
        as they are tiles sampled from a source (geo) image.

        The output from Segformer is fixed at (256,256). We perform bilinear
        interpolation back to the source tile size and apply softmax to
        convert the results to predictions in [0,1].

        Args:
            x: List[torch.Tensor] or torch.Tensor

        Returns:
            Interpolated semantic segmentation predictions with softmax applied
        """
        encoded_inputs = self.processor(images=x, return_tensors="pt")

        with torch.no_grad():
            encoded_inputs.to(self.model.device)
            logits = self.model(pixel_values=encoded_inputs.pixel_values).logits

        pred = nn.functional.interpolate(
            logits,
            size=encoded_inputs.pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).softmax(dim=1)

        return pred
