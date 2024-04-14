import logging
import warnings

from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .segmentationmodule import SegmentationModule

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class SegformerModule(SegmentationModule):
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
