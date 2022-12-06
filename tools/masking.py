import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from tqdm import tqdm

# from rasterio.features import rasterize

# constants
IMG_WIDTH = 2048  # pixels
IMG_HEIGTH = 2048  # pixels
DATA_DIR = "../data/"


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


if __name__ == "__main__":

    img_dir = Path(DATA_DIR + "images")

    # TRAIN
    print("##### GETTING MASKS TRAINING #####")
    train_annotation = Path(DATA_DIR + "train_20221010.json")
    train_coco_obj, train_imgs, train_img_ids = extract_images(train_annotation)
    train_masks = get_all_masks(train_imgs, img_dir, train_coco_obj)

    # ugly but quickly needed this for fixing mask.npz formats
    for i in range(len(train_masks)):
        mask = train_masks[i]
        idx = train_img_ids[i]
        np.savez_compressed(DATA_DIR + "masks/train_mask_" + str(idx), mask)

    # np.savez_compressed(DATA_DIR + "train_masks", train_masks)

    # VALIDATION
    print("##### GETTING MASKS VALIDATION #####")
    val_annotation = Path(DATA_DIR + "val_20221010.json")
    val_coco_obj, val_imgs, val_img_ids = extract_images(val_annotation)
    val_masks = get_all_masks(val_imgs, img_dir, val_coco_obj)
    for i in range(len(val_masks)):
        mask = val_masks[i]
        idx = val_img_ids[i]
        np.savez_compressed(DATA_DIR + "masks/val_mask_" + str(idx), mask)

    # np.savez_compressed(DATA_DIR + "val_masks", val_masks)

    # TEST
    print("##### GETTING MASKS TEST #####")
    test_annotation = Path(DATA_DIR + "test_20221010.json")
    test_coco_obj, test_imgs, test_img_ids = extract_images(test_annotation)
    test_masks = get_all_masks(test_imgs, img_dir, test_coco_obj)
    for i in range(len(test_masks)):
        mask = test_masks[i]
        idx = test_img_ids[i]
        np.savez_compressed(DATA_DIR + "masks/test_mask_" + str(idx), mask)

    # np.savez_compressed(DATA_DIR + "test_masks", test_masks)

    print("Extracted all masks for train, val, test images")
