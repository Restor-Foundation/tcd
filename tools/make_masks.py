import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from tqdm import tqdm

logger = logging.getLogger("mask_generator")
logging.basicConfig(level=logging.INFO)

class_map = {"tree": 1, "canopy": 2}


def unet_weight_map(y, wc=None, w0=10, sigma=5):

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


def plot_images(imgs, data_dir, coco_obj):
    """
    Plot images without (left) and with (right) segmentation for visualization
    """
    _, axs = plt.subplots(len(imgs), 2, figsize=(10, 5 * len(imgs)))
    for img, ax in zip(imgs, axs):
        I = Image.open(data_dir / img["file_name"])
        ann_ids = coco_obj.getAnnIds(imgIds=[img["id"]])
        anns = coco_obj.loadAnns(ann_ids)
        ax[0].imshow(I)
        ax[1].imshow(I)
        plt.sca(ax[1])
        coco_obj.showAnns(anns, draw_bbox=False)


def plot_mask(img_mask):
    """
    Plot mask as black-white (notree-tree) image for visualization
    """
    plt.imshow(img_mask, cmap="gray", interpolation="nearest")
    plt.show()


"""
The following is nabbed from the COCO API, but we decouple it
so we can parallelize things without having to unnecessarily
pass around the COCO object
"""


def annToRLE(img, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    h, w = img["height"], img["width"]
    segm = ann["segmentation"]
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann["segmentation"]
    return rle


def annToMask(img, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(img, ann)
    m = maskUtils.decode(rle)
    return m


def get_mask(img, anns, cats, size, binary) -> np.ndarray:
    """Takes a coco image dict and returns a mask"""
    mask = np.zeros((size, size), dtype=np.uint8)

    cat_names = {cats[idx]["id"]: cats[idx]["name"] for idx in cats}

    for ann in anns:
        new_mask = annToMask(img, ann)
        new_mask = cv2.resize(new_mask, (size, size), interpolation=cv2.INTER_NEAREST)

        # For trees only
        if cat_names[ann["category_id"]].lower() == "tree" and not binary:
            new_mask = cv2.erode(new_mask, np.ones((3, 3), np.uint8), iterations=1)

        mask[new_mask > 0] = ann["category_id"]

    if binary:
        mask[mask != 0] = 1

    return mask


def plot_discrete(arr):
    import matplotlib
    from matplotlib.colors import ListedColormap

    vals = np.unique(arr)

    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
    col_dict = {1: "#fe1493", 2: "#ccff00"}

    # We create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    labels = np.array(["tree", "canopy"])
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    # Plot our figure
    plt.imshow(arr, cmap=cm, norm=norm, interpolation="nearest", alpha=0.5)


def process_mask(
    img: dict,
    anns: list,
    cats: dict,
    image_root: str,
    output_root: str,
    output_size: int,
    prefix: str = "",
    binary: bool = False,
    visualise: bool = False,
    weights: bool = False,
    integrity_check: bool = False,
):

    logger.debug("Starting: " + img["file_name"])

    mask = get_mask(img, anns, cats, output_size, binary)
    img_path = image_root / img["file_name"]
    img_stem = img_path.stem
    ext = img_path.suffix

    invalid = mask != cv2.medianBlur(mask, 3)
    mask[invalid] = 0

    if output_size != img["width"]:

        with rasterio.open(img_path) as src:
            w = src.read()
            profile = src.profile
            profile["width"] = output_size
            profile["height"] = output_size
            with rasterio.open(
                output_root / "images" / f"{prefix}{img_stem}{ext}", "w", **profile
            ) as dst:
                dst.write(w)

    else:
        shutil.copy(
            image_root / img["file_name"],
            output_root / "images" / f"{prefix}{img_stem}{ext}",
        )

    if weights:
        weights = unet_weight_map(mask)
        np.savez(output_root / "weights" / f"{prefix}{img_stem}.npz", weights)

    mask_path = str(output_root / "masks" / f"{prefix}{img_stem}.png")
    cv2.imwrite(mask_path, mask)

    if integrity_check:
        assert np.allclose(mask, cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))

    if visualise:
        plt.imshow(plt.imread(output_root / "images" / f"{prefix}{img_stem}{ext}"))
        plot_discrete(np.ma.masked_equal(mask, 0))
        plt.axis("off")
        plt.savefig(
            output_root / "preview" / f"{prefix}{img_stem}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        if weights:
            # Also vis weights
            plt.imshow(weights, cmap="jet")
            plt.axis("off")
            plt.savefig(
                output_root / "preview" / f"{prefix}{img_stem}_weights.png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

    logger.debug("Done: " + img["file_name"])


def process_mask_star(args):

    try:
        process_mask(*args)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(e)
        logger.error("Error processing: " + args[0]["file_name"])


class MaskGenerator:
    def __init__(self, args):

        # Check annotation file
        assert os.path.exists(args.annotations)
        self.annotation_path = Path(args.annotations)
        self.coco = COCO(args.annotations)
        logger.info(f"Getting masks for {annotation_file}")

        # Check image root - we need this if the size differs
        self.image_root = Path(args.images)
        assert os.path.exists(self.image_root)
        logger.info(f"Using images in {self.image_root} for viz")

        # Make output if it doesn't exist
        self.output_root = Path(args.output)
        os.makedirs(self.output_root / "masks", exist_ok=True)
        os.makedirs(self.output_root / "preview", exist_ok=True)
        os.makedirs(self.output_root / "images", exist_ok=True)
        os.makedirs(self.output_root / "weights", exist_ok=True)

        logger.info(f"Storing masks to {self.output_root}")

        # Output options
        self.size = args.size
        self.prefix = args.prefix
        self.visualise = args.visualise
        self.binary = args.binary
        self.weights = args.weights

        if self.visualise:
            logger.info("Visualising masks")

        if self.binary:
            logger.info("Generating binary masks")

    def _parallel_process(self, imgs):

        n_cores = max(1, os.cpu_count() - 2)
        logger.info("Starting {} parallel processes.".format(n_cores))

        with mp.Pool(processes=n_cores) as pool:

            args = [
                (
                    self.coco.imgs[img_id],
                    self.coco.imgToAnns[img_id],
                    self.coco.cats,
                    self.image_root,
                    self.output_root,
                    self.size,
                    self.prefix,
                    self.binary,
                    self.visualise,
                    self.weights,
                )
                for img_id in imgs
            ]
            list(tqdm(pool.imap(process_mask_star, args), total=len(imgs)))

    def generate_masks(self) -> None:
        imgs = self.coco.getImgIds()
        logger.info(f"Creating {len(imgs)} masks")

        if args.parallel:
            self._parallel_process(imgs)
        else:
            for img_id in tqdm(imgs):
                process_mask(
                    self.coco,
                    self.coco.imgToAnns[img_id],
                    self.coco.cats,
                    self.image_root,
                    self.output_root,
                    self.size,
                    self.prefix,
                    self.binary,
                    self.visualise,
                    self.weights,
                )

    def resize_instances(self):

        new_coco = dict(self.coco.dataset)

        for ann in tqdm(new_coco["annotations"]):

            image_id = ann["image_id"]
            old_width = self.coco.imgs[image_id]["width"]
            old_height = self.coco.imgs[image_id]["height"]

            # Resize bounding box
            ann["bbox"] = [
                ann["bbox"][0] * self.size / old_width,
                ann["bbox"][1] * self.size / old_height,
                ann["bbox"][2] * self.size / old_width,
                ann["bbox"][3] * self.size / old_height,
            ]

            # Resize segmentation
            # If the annotation has a RLE mask:
            if "counts" in ann["segmentation"]:
                # Resize the RLE mask
                mask = annToMask(self.coco.imgs[image_id], ann)
                new_mask = cv2.resize(
                    mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST
                )
                ann["segmentation"] = maskUtils.encode(np.asfortranarray(new_mask))
                ann["segmentation"]["counts"] = ann["segmentation"]["counts"].decode(
                    "utf-8"
                )
                # If the annotation has a polygon:
            else:
                # Resize polygons
                for idx, poly in enumerate(ann["segmentation"]):
                    ann["segmentation"][idx][0::2] = [
                        x * self.size / int(old_width) for x in poly[0::2]
                    ]
                    ann["segmentation"][idx][1::2] = [
                        y * self.size / int(old_height) for y in poly[1::2]
                    ]

            ann["area"] = (
                ann["area"] * (self.size / old_width) * (self.size / old_height)
            )

        # Update the image sizes
        for img in tqdm(new_coco["images"]):
            img["width"] = self.size
            img["height"] = self.size

        # Save the new annotations
        with open(self.output_root / os.path.basename(self.annotation_path), "w") as f:
            json.dump(new_coco, f, indent=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--images",
        help="Path to folder containing images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--annotations", help="Annotation file, COCO format", type=str, required=True
    )
    parser.add_argument(
        "-s", "--size", help="Output size, pixels", type=int, default=2048
    )
    parser.add_argument(
        "--prefix",
        help="Prefix appended to generated masks, set to split name",
        type=str,
        default="",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to store masks, defaults to a folder called masks at the same level of the image folder",
        type=str,
    )
    parser.add_argument(
        "--parallel",
        help="Parallelize mask generation",
        action="store_true",
    )
    parser.add_argument(
        "--visualise",
        help="Visualise",
        action="store_true",
    )

    parser.add_argument(
        "--binary",
        help="Binarise masks",
        action="store_true",
    )

    parser.add_argument(
        "--resize_instances",
        help="Resize instances",
        action="store_true",
    )

    parser.add_argument(
        "--weights",
        help="Generate weights for each instance",
        action="store_true",
    )

    args = parser.parse_args()

    # Load args / default paths
    image_dir = Path(args.images)
    annotation_file = args.annotations

    if args.output is None:
        args.output_dir = image_dir.parent.absolute() / "masks"
    else:
        args.output_dir = Path(args.output)

    gen = MaskGenerator(args)
    gen.generate_masks()

    if args.resize_instances:
        gen.resize_instances()

    """
    coco = COCO(Path(args.output_dir) / os.path.basename(args.annotations))
    import random
    for img_id in coco.getImgIds():
        # plot random image + annotations
        image_path = Path(args.output_dir) / "images" / coco.imgs[img_id]['file_name']
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path)
        plt.imshow(image)

        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)
        coco.showAnns(coco.imgToAnns[img_id])
        #plt.imshow(np.ma.masked_equal(mask, 0), interpolation='nearest', cmap='jet')
        plt.axis('off')
        plt.show()
    """
