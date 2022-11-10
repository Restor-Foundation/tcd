import itertools
import json
import os
import shutil

import numpy as np
from decouple import config
from PIL import Image
from skimage.io import imread
from skimage.transform import downscale_local_mean
from tqdm import tqdm

DATA_DIR = config("DATA_DIR")
IMAGE_DIR = f"{DATA_DIR}/images/"
D_IMAGE_DIR = f"{DATA_DIR}downsampled_images/"

MASK_DIR = f"{DATA_DIR}/masks/"
D_MASK_DIR = f"{DATA_DIR}downsampled_masks/"


def scale(img):
    if img.shape[0] % 32 != 0:
        pad_size = 32 - img.shape[0] % 32
        img = np.pad(img, ((pad_size, 0), (pad_size, 0), (0, 0)), mode="constant")
    return img


def sampler(factor):

    F_D_IMAGE_DIR = f"{D_IMAGE_DIR}sampling_factor_{factor}/"
    F_D_MASK_DIR = f"{D_MASK_DIR}sampling_factor_{factor}/"

    if not os.path.exists(D_IMAGE_DIR):
        print("Creating folder for downsampled images")
        os.makedirs(D_IMAGE_DIR)

    if not os.path.exists(D_MASK_DIR):
        print("Creating folder for downsampled masks")
        os.makedirs(D_MASK_DIR)

    if not os.path.exists(F_D_IMAGE_DIR):
        print(f"Creating image folder for factor {factor}")
        os.makedirs(F_D_IMAGE_DIR)

    if not os.path.exists(F_D_MASK_DIR):
        print(f"Creating mask folder for factor {factor}")
        os.makedirs(F_D_MASK_DIR)

    print(f"Compressing images with factor{factor}")

    if len(os.listdir(F_D_IMAGE_DIR)) == 0 or len(os.listdir(F_D_MASK_DIR)) == 0:
        for image_n, mask_n in tqdm(
            itertools.zip_longest(os.listdir(IMAGE_DIR), os.listdir(MASK_DIR)),
            total=len(os.listdir(IMAGE_DIR)),
        ):
            try:
                im = Image.open(IMAGE_DIR + image_n)
                im_downscaled = downscale_local_mean(np.array(im), (factor, factor, 1))
                im_downscaled = im_downscaled.astype(np.uint8)
                if im_downscaled.shape[0] % 32 != 0:
                    pad_size = 32 - im_downscaled.shape[0] % 32
                    im_downscaled = np.pad(
                        im_downscaled,
                        ((pad_size, 0), (pad_size, 0), (0, 0)),
                        mode="constant",
                    )
                im_downscaled = Image.fromarray(im_downscaled, "RGB")
                im_downscaled.save(F_D_IMAGE_DIR + image_n, compression="jpeg")
            except:
                pass
            try:
                mask = np.load(MASK_DIR + mask_n)["arr_0"].astype(int)
                mask_downscaled = downscale_local_mean(mask, (factor, factor)).astype(
                    int
                )
                if mask_downscaled.shape[0] % 32 != 0:
                    pad_size = 32 - mask_downscaled.shape[0] % 32
                    mask_downscaled = np.pad(
                        mask_downscaled, ((pad_size, 0), (pad_size, 0)), mode="constant"
                    )
                np.savez_compressed(F_D_MASK_DIR + mask_n, mask_downscaled)
            except:
                pass

    print("Images and masks compressed and saved.")
