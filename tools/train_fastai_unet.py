import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from fastai.callback.hook import summary
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import fit_flat_cos, lr_find
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.data.block import DataBlock
from fastai.data.external import URLs, untar_data
from fastai.data.transforms import FuncSplitter, Normalize, get_image_files
from fastai.layers import Mish
from fastai.losses import BaseLoss
from fastai.optimizer import ranger
from fastai.torch_core import tensor
from fastai.vision.augment import Brightness, Flip, Rotate
from fastai.vision.core import PILImage, PILMask
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.learner import unet_learner
from fastcore.xtras import Path
from PIL import Image
from torch import nn
from torchvision.models.resnet import resnet34

torch.multiprocessing.set_sharing_strategy("file_system")


path = Path("../data/restor-tcd-oam/downsampled/256")
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


tcd_block = DataBlock(
    blocks=(ImageBlock, MaskBlock(codes)),
    get_items=get_image_files,
    splitter=train_val_splitter,
    get_y=get_msk,
    batch_tfms=[Flip(), Rotate(max_deg=90), Normalize.from_stats(*imagenet_stats)],
)

dls = tcd_block.dataloaders(path / "images", bs=8)
# dls.show_batch(max_n=4, vmin=0, vmax=2, figsize=(14,10))

dls.vocab = codes

from fastai.metrics import Dice

# opt = ranger // Use Adam for now
learn = unet_learner(
    dls, resnet34, metrics=[Dice()]
)  # , self_attention=True, act_cls=Mish, opt_func=opt
print(learn.summary())

plt.figure()
# print("Running LR finder")
learn.lr_find()
plt.savefig("lr_find.png")

lr = 1e-4
# learn.fit_one_cycle(10, slice(lr))

# learn.save('stage-1')
learn.load("stage-1")

# plt.figure()
# learn.show_results(max_n=4, figsize=(12,6))
# plt.savefig("results_1.png", bbox_inches='tight')
# plt.close()

lrs = slice(lr / 400, lr / 4)
learn.unfreeze()
learn.fit_flat_cos(
    50, lrs, cbs=TensorBoardCallback(log_dir="./logs/", trace_model=True)
)

learn.save("model_1")
learn.load("model_1")

plt.figure()
learn.show_results(max_n=4, figsize=(12, 6))
plt.savefig("results_2.png", bbox_inches="tight")
plt.close()
