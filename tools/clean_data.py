import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from decouple import config
from PIL import Image
from tqdm import tqdm

DATA_DIR = config("DATA_DIR")
C_DATA_DIR = f"{DATA_DIR}corrupted_images/"

with open(f"{DATA_DIR}val_20221010.json", "r") as file:
    val_data = json.load(file)

ctr = 0
ids = []

# check shapes of all images
print("checking for corrupted images")
for i, image in enumerate(tqdm(list(val_data.items())[2][1])):

    img_name = list(val_data.items())[2][1][i]["file_name"]
    img_path = os.path.join(DATA_DIR, "images", img_name)
    im = np.array(Image.open(img_path))

    if len(im.shape) != 3:
        ctr += 1
        ids.append(list(val_data.items())[2][1][i]["id"])
        if not os.path.exists(C_DATA_DIR):
            print("Creating folder for corrupted images")
            os.makedirs(C_DATA_DIR)
        os.rename(img_path, f"{C_DATA_DIR}{img_name}")

print(f"Found and moved {ctr} corrupted images")
print(f"Removed ids: {ids}")
