import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from tqdm import tqdm
from decouple import config
import argparse
import logging

logger = logging.getLogger("mask_genereator")
logging.basicConfig(level=logging.INFO)

# from rasterio.features import rasterize

# constants
IMG_WIDTH = 2048  # pixels
IMG_HEIGTH = 2048  # pixels

def extract_images(annotation_file):
    """
    json_dir: string, file directory of json file in pycoco format
    Returns corresponding pycocotools object, list of images, and their ids
    """
    coco_obj = COCO(annotation_file)
    img_ids = coco_obj.getImgIds()
    imgs = coco_obj.loadImgs(img_ids)
    return coco_obj, imgs, img_ids


def extract_annotations(img, img_dir, coco_obj):
    """
    img: image dict
    img_dir: directory of stored images
    coco_obj: pycoco.COCO object for corresponding json file
    Returns list of annotations for this img, and list of corresponding annotation ids
    """
    I = Image.open(img_dir / img["file_name"])
    ann_ids = coco_obj.getAnnIds(imgIds=[img["id"]])
    anns = coco_obj.loadAnns(ann_ids)
    return anns, ann_ids


def get_mask(img, img_dir, coco_obj):
    """
    img: image dict
    img_dir: directory of stored images
    coco_obj: pycoco.COCO object for corresponding json file
    Returns mask for this image as np array of np.bool_ entries (1: tree, 0: no tree)
    """
    # alternatively, using rasterio
    # mask_list = []
    # for ann in anns:
    #     if 'segmentation' in ann.keys() and type(ann['segmentation']) is list:
    #         #print(type(ann['segmentation']))
    #         poly_list = ann['segmentation'][0]
    #         mask_list.append(mask_from_poly_list(poly_list))
    #     elif 'segmentation' in ann.keys() and type(ann['segmentation']) is dict:
    #         mask_list.append(None) # insert benji's function here for the case where we have 'counts' and not 'segmentation' format
    #     else:
    #         mask_list.append(None) # any potential other cases

    anns, ann_ids = extract_annotations(img, img_dir, coco_obj)
    if len(anns) > 0:
        mask = coco_obj.annToMask(anns[0]) > 0  # simpler
        for i in range(len(anns)):
            mask += coco_obj.annToMask(anns[i]) > 0
    else:
        mask = np.zeros((IMG_HEIGTH, IMG_WIDTH), dtype=np.bool_)
    return mask, anns, ann_ids


def get_all_masks(imgs, img_dir, coco_obj):
    """
    imgs: list of images (each image is a dict)
    img_dir: directory of stored images
    coco_obj: pycoco.COCO object for corresponding json file
    Returns list of masks for all images
    """
    all_masks = []
    for img in tqdm(imgs):
        mask, _, _ = get_mask(img, img_dir, coco_obj)
        all_masks.append(mask)
    return all_masks


# alternatively, when using rasterio
# def mask_from_poly_list(poly_list):
#     # poly stored as in "segmentation"
#     l = [round(a) for a in poly_list]
#     l = [(l[2*i], l[2*i+1]) for i in range(int(len(l)/2))]
#     polygon = Polygon(l)
#     img_mask = rasterize([polygon], out_shape=[IMG_HEIGTH, IMG_WIDTH])
#     return img_mask


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
    # plt.imshow(img_mask, interpolation='nearest')
    plt.imshow(img_mask, cmap="gray", interpolation="nearest")
    plt.show()

def generate_masks(annotation_file, image_dir, output_dir, prefix):

    coco_obj, imgs, img_ids = extract_images(annotation_file)
    masks = get_all_masks(imgs, image_dir, coco_obj)
    
    logger.info(f"Creating {len(masks)} masks")
    
    # ugly but quickly needed this for fixing mask.npz formats
    for i, mask in tqdm(enumerate(masks)):
        idx = img_ids[i]
        np.savez_compressed(output_dir / f"{prefix}_mask_{idx}", mask)

if __name__ == "__main__":

    

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="Path to folder containing images", type=str, required=True)
    parser.add_argument("--annotations", help="Annotation file, COCO format", type=str, required=True)
    parser.add_argument("--prefix", help="Prefix appended to generated masks, set to split name", type=str, required=True)
    parser.add_argument("--output", help="Path to store masks, defaults to a folder called masks at the same level of the image folder", type=str)

    args = parser.parse_args()

    # Load args / default paths
    image_dir = Path(args.images)
    annotation_file = args.annotations

    if args.output is None:
        output_dir = image_dir.parent.absolute() / "masks"
    else:
        output_dir = Path(args.output)

    assert os.path.exists(image_dir)
    assert os.path.exists(annotation_file)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Getting masks for {annotation_file}")
    logger.info(f"Storing masks to {output_dir}")

    generate_masks(annotation_file, image_dir, output_dir, args.prefix)
