import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

splits = ["train", "val", "test"]

for split in splits:

    annotations = f"data/restor-tcd-oam/{split}_20221010.json"

    dataset = COCO(annotation_file=annotations)
    empty_images = []

    for idx in dataset.imgs:

        anns = dataset.getAnnIds([idx])
        if len(anns) == 0:
            # drop image
            empty_images.append(idx)

    with open(annotations, "r") as fp:
        dirty_data = json.load(fp)

        for empty_image in empty_images:
            images = [
                image
                for image in dirty_data["images"]
                if image["id"] not in empty_images
            ]

        dirty_data["images"] = images

    with open(f"data/restor-tcd-oam/{split}_20221010_noempty.json", "w") as fp:
        json.dump(dirty_data, fp, indent=1)
