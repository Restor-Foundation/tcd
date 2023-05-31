import wandb

wandb.init(project="tcd-fastai-unets")

import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from fastai.callback.hook import summary
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import fit_flat_cos, lr_find
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.callback.tracker import ReduceLROnPlateau, SaveModelCallback
from fastai.callback.wandb import *
from fastai.data.block import DataBlock
from fastai.data.external import URLs, untar_data
from fastai.data.transforms import FuncSplitter, Normalize, get_image_files
from fastai.optimizer import ranger
from fastai.torch_core import tensor
from fastai.vision.augment import (
    Brightness,
    Contrast,
    Dihedral,
    Flip,
    Hue,
    RandomCrop,
    RandomResizedCrop,
    Rotate,
)
from fastai.vision.core import PILImage, PILMask
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.learner import unet_learner
from fastai.vision.models.xresnet import xresnet18, xresnet34, xresnet50, xresnet101
from fastcore.xtras import Path
from PIL import Image
from torch import Tensor, nn

torch.multiprocessing.set_sharing_strategy("file_system")


import argparse

from fastai.basics import patch, store_attr, use_kwargs_dict
from fastai.losses import BaseLoss, DiceLoss, FocalLoss, FocalLossFlat
from fastai.torch_core import Module as FastaiModule
from fastcore.foundation import *
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="xresnet34")
parser.add_argument("--size", type=int, choices=[256, 512, 1024, 2048], default=1024)
parser.add_argument("--bs", type=int, default=8, help="Batch size. Default: 8")

parser.add_argument("--epochs", type=int, default=50, help="Epochs. Default: 50")


args = parser.parse_args()


class CombinedLoss:
    "Dice and Focal combined"

    def __init__(self, axis=1, smooth=1.0, alpha=1.0):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis, reduction="mean")
        self.dice_loss = DiceLoss(axis, smooth, reduction="mean")

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)


class WeightedFocalLoss(FastaiModule):
    y_int = True  # y interpolation

    def __init__(
        self,
        gamma: float = 2.0,  # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        weight: Tensor = None,  # Manual rescaling weight given to each class
        reduction: str = "mean",  # PyTorch reduction to apply to the output
    ):
        "Applies Focal Loss: https://arxiv.org/pdf/1708.02002.pdf"
        store_attr()

    def unet_weight_map(self, y, wc=None, w0=10, sigma=5):
        """
        Generate weight maps as specified in the U-Net paper
        for boolean mask.

        "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        https://arxiv.org/pdf/1505.04597.pdf

        Parameters
        ----------
        mask: Numpy array
            2D array of shape (image_height, image_width) representing binary mask
            of objects.
        wc: dict
            Dictionary of weight classes.
        w0: int
            Border weight parameter.
        sigma: int
            Border width parameter.
        Returns
        -------
        Numpy array
            Training weights. A 2D array of shape (image_height, image_width).
        """
        y = y.cpu().numpy()
        labels = label(y)
        no_labels = labels == 0
        label_ids = sorted(np.unique(labels))[1:]

        if len(label_ids) > 1:
            distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

            for i, label_id in enumerate(label_ids):
                distances[:, :, i] = distance_transform_edt(labels != label_id)

            distances = np.sort(distances, axis=2)
            d1 = distances[:, :, 0]
            d2 = distances[:, :, 1]
            w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2) * no_labels

            if wc:
                class_weights = np.zeros_like(y)
                for k, v in wc.items():
                    class_weights[y == k] = v
                w = w + class_weights
        else:
            w = np.zeros_like(y)

        return w

    def forward(self, inp: Tensor, targ: Tensor) -> Tensor:
        "Applies focal loss based on https://arxiv.org/pdf/1708.02002.pdf"
        ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss

        weights = [self.unet_weight_map(t) for t in targ]
        for idx in range(len(loss)):
            loss[idx] *= torch.from_numpy(weights[idx]).to(loss.device)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class WeightedFocalLossFlat(BaseLoss):
    """
    Same as CrossEntropyLossFlat but with focal paramter, `gamma`. Focal loss is introduced by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf. Note the class weighting factor in the paper, alpha, can be
    implemented through pytorch `weight` argument passed through to F.cross_entropy.
    """

    y_int = True  # y interpolation

    @use_kwargs_dict(keep=True, weight=None, reduction="mean")
    def __init__(
        self,
        *args,
        gamma: float = 2.0,  # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        axis: int = -1,  # Class axis
        **kwargs,
    ):
        super().__init__(WeightedFocalLoss, *args, gamma=gamma, axis=axis, **kwargs)

    def decodes(self, x: Tensor) -> Tensor:
        "Converts model output to target format"
        return x.argmax(dim=self.axis)

    def activation(self, x: Tensor) -> Tensor:
        "`F.cross_entropy`'s fused activation function applied to model output"
        return F.softmax(x, dim=self.axis)


path = Path(f"../data/restor-tcd-oam/downsampled/{args.size}")
path_im = path / "images"
path_lbl = path / "masks"

fnames = get_image_files(path_im)
lbl_names = get_image_files(path_lbl)
get_msk = lambda o: path / "masks" / f"{o.stem}.png"

from pycocotools.coco import COCO

valid_anns = COCO("../data/restor-tcd-oam/val_20221010_noempty.json")
valid_images = [img["file_name"] for img in valid_anns.dataset["images"]]

train_anns = COCO("../data/restor-tcd-oam/train_20221010_noempty.json")
train_images = [img["file_name"] for img in train_anns.dataset["images"]]

codes = ["background", "tree"]


def train_val_splitter(fnames):
    return (
        [idx for idx, f in enumerate(fnames) if f.name in train_images],
        [idx for idx, f in enumerate(fnames) if f.name in valid_images],
    )


crop_size = 256
if args.model == "xresnet18":
    arch = xresnet18
    model_name = "xresnet18"
    crop_size = min(args.size, 1024)
elif args.model == "xresnet34":
    arch = xresnet34
    model_name = "xresnet34"
    crop_size = min(args.size, 1024)
elif args.model == "xresnet50":
    arch = xresnet50
    model_name = "xresnet50"
    crop_size = min(args.size, 512)
elif args.model == "xresnet101":
    arch = xresnet101
    model_name = "xresnet101"
    crop_size = min(args.size, 512)
else:
    raise ValueError("Unknown model")


import datetime
import os

date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"./results/{date_str}_{model_name}_{args.size}"
os.makedirs(log_dir, exist_ok=False)  # Don't overwrite existing logs

crop_scale = crop_size / 2048

print("Cropping to size", crop_size)
print("Crop scale", crop_scale)

tcd_block = DataBlock(
    blocks=(ImageBlock, MaskBlock(codes)),
    get_items=get_image_files,
    splitter=train_val_splitter,
    get_y=get_msk,
    # item_tfms=[
    #    RandomCrop(size=crop_size, p=1),
    # ],
    batch_tfms=[
        Brightness(),
        Contrast(),
        Hue(max_hue=0.2),
        Dihedral(),
        Rotate(max_deg=90),
        RandomCrop(size=crop_size),
        Normalize.from_stats(*imagenet_stats),
    ],
)

dls = tcd_block.dataloaders(path / "images", bs=args.bs, num_workers=2)
tcd_block.summary(path / "images", bs=args.bs, num_workers=2)

plt.figure()
dls.show_batch(max_n=4, vmin=0, vmax=2, figsize=(14, 10))
plt.savefig(os.path.join(log_dir, "sample_batch.png"))

dls.vocab = codes

# Sanity check dataloader
height, width = dls.one_batch()[0].shape[-2:]
# assert height == width == crop_size, print("Cropping failed, got", height, width)

from fastai.metrics import Dice, F1ScoreMulti, foreground_acc

print("Setting up learner")
opt = ranger
learn = unet_learner(
    dls,
    arch,
    metrics=[Dice(), foreground_acc],
    self_attention=True,
    act_cls=nn.ReLU,
    opt_func=opt,
    cbs=[
        # ReduceLROnPlateau(monitor="valid_loss", min_delta=0.01, patience=5, factor=0.5),
        SaveModelCallback(monitor="dice", fname=f"model_{model_name}_{args.size}")
    ],
)

from fastai.vision.all import *


class PlotValidationBatchCallback(Callback):
    def __init__(self):
        self.fig, self.axs = plt.subplots(3, 3, figsize=(9, 9))
        self.counter = 0

    def after_epoch(self):
        self.learn.show_results(
            ds_idx=1, nrows=3, ncols=3, figsize=(9, 9), ax=self.axs.flatten()
        )
        plt.tight_layout()
        self.writer.add_figure("Validation Batch Examples", self.fig, self.epoch)
        self.counter += 1
        self.fig.clear()


learn.summary()

# plt.figure()
# print("Running LR finder")
# learn.lr_find()
# plt.savefig(os.path.join(log_dir, "lr_finder.png"), bbox_inches="tight")

print("Training with frozen model first")
lr = 5e-5
learn.fit_flat_cos(
    10,
    slice(lr),
    cbs=[TensorBoardCallback(log_dir=log_dir, trace_model=True), WandbCallback()],
)

# learn.save(f"{date_str}_{model_name}_{args.size}_frozen")
torch.save(learn.model, os.path.join(log_dir, "unfrozen.pth"))
# learn.load(f"{date_str}_{model_name}_{args.size}_frozen")

plt.figure()
learn.show_results(max_n=4, figsize=(12, 6))
plt.savefig(os.path.join(log_dir, "results_frozen.png"), bbox_inches="tight")
plt.close()


print("Training with unfrozen model")
lrs = slice(lr / 400, lr / 4)
learn.unfreeze()

# learn.add_cb(PlotValidationBatchCallback())

learn.fit_flat_cos(
    args.epochs,
    lrs,
    cbs=[TensorBoardCallback(log_dir=log_dir, trace_model=True), WandbCallback()],
)

# learn.save(f"{date_str}_{model_name}_{args.size}_unfrozen")
# learn.load(f"{date_str}_{model_name}_{args.size}_unfrozen")

plt.figure()
learn.show_results(max_n=4, figsize=(12, 6))
plt.savefig(os.path.join(log_dir, "results_unfrozen.png"), bbox_inches="tight")
plt.close()

learn.model.eval()
torch.save(learn.model, os.path.join(log_dir, "unfrozen.pth"))
# os.remove(f"./models/{date_str}_{model_name}_{args.size}_frozen.pth")


x, _ = dls.one_batch()
learn.model.cuda()
learn.model.eval()

from fastai.torch_core import TensorBase


@patch
def requires_grad_(self: TensorBase, requires_grad=True):
    self.requires_grad = requires_grad
    return self


torch.jit.save(
    torch.jit.trace(learn.model, x), os.path.join(log_dir, "traced_model.pts")
)

artifact = wandb.Artifact("jit_model", type="model")
artifact.add_file(os.path.join(log_dir, "traced_model.pts"))
wandb.log_artifact(artifact)

wandb.finish()
